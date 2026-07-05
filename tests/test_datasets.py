import bz2
import hashlib
import io
import os
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import hawkes_tools.datasets as datasets  # noqa: E402
from hawkes_tools.datasets import (
    DATASET_MANIFEST,
    DATASETS_SOURCE,
    DATASETS_TREE_SHA,
    LOCAL_EXTERNAL_DATASETS_SOURCE,
    DATASET_RELEASE_MANIFEST_URL,
    KDD2010_TRAIN_DATASET_SHA256_URL,
    KDD2010_TRAIN_DATASET_URL,
    URL_DATASET_N_FEATURES,
    URL_DATASET_PATH,
    dataset_metadata,
    download_dataset,
    external_dataset_metadata,
    fetch_dataset,
    fetch_hawkes_bund_data,
    get_data_home,
    is_dataset_vendored,
    list_external_datasets,
    list_datasets,
    load_dataset,
    vendored_dataset_path,
    fetch_url_dataset,
    load_url_dataset_day,
)


class DatasetLoaderTest(unittest.TestCase):
    large_external_dataset_path = "binary/kdd2010/kdd2010.trn.bz2"

    def test_hawkes_bund_data_loads_from_vendored_payload(self):
        timestamps = fetch_hawkes_bund_data()

        self.assertIn("hawkes/bund/bund.npz", list_datasets())
        self.assertEqual(len(timestamps), 20)
        self.assertTrue(all(len(realization) == 4 for realization in timestamps))
        self.assertTrue(all(np.all(np.diff(stream) >= 0.0) for realization in timestamps for stream in realization))
        self.assertGreater(sum(len(stream) for realization in timestamps for stream in realization), 0)

    def test_vendored_svmlight_datasets_load_without_external_dependency(self):
        adult_x, adult_y = fetch_dataset("binary/adult/adult.trn.bz2")
        abalone_x, abalone_y = fetch_dataset("regression/abalone/abalone.trn.bz2")

        self.assertIn("binary/adult/adult.trn.bz2", list_datasets())
        self.assertEqual(adult_x.shape, (32561, 123))
        self.assertEqual(adult_y.shape, (32561,))
        self.assertEqual(abalone_x.shape, (4177, 8))
        self.assertEqual(abalone_y.shape, (4177,))

    def test_dataset_manifest_entries_are_available_or_external(self):
        manifest = list_datasets()

        self.assertEqual(DATASETS_TREE_SHA, "9d959b6e53e17145e93e9849ff1f9f6d2de8ae51")
        self.assertEqual(len(manifest), len(set(manifest)))
        self.assertEqual(len(manifest), 11)
        self.assertEqual(set(DATASET_MANIFEST), set(manifest))
        for dataset_path in manifest:
            with self.subTest(dataset_path=dataset_path):
                metadata = dataset_metadata(dataset_path)
                self.assertIn("format", metadata)
                self.assertTrue("shape" in metadata or "x_shape" in metadata)
                self.assertIn("size_bytes", metadata)
                self.assertIn("sha256", metadata)
                self.assertEqual(metadata["vendored"], is_dataset_vendored(dataset_path))
                if metadata["vendored"]:
                    self.assertEqual(metadata["source"], DATASETS_SOURCE)
                    payload_path = vendored_dataset_path(dataset_path)
                    self.assertEqual(payload_path.stat().st_size, metadata["size_bytes"])
                    self.assertEqual(_sha256_file(payload_path), metadata["sha256"])
                else:
                    self.assertEqual(dataset_path, self.large_external_dataset_path)
                    self.assertEqual(metadata["source"], LOCAL_EXTERNAL_DATASETS_SOURCE)
                    self.assertFalse(vendored_dataset_path(dataset_path).exists())

        adult_metadata = dataset_metadata("binary/adult/adult.trn.bz2")
        self.assertEqual(adult_metadata["x_shape"], adult_x_shape := (32_561, 123))
        adult_x, adult_y = fetch_dataset("binary/adult/adult.trn.bz2")
        self.assertEqual(adult_x.shape, adult_x_shape)
        self.assertEqual(adult_y.shape, adult_metadata["y_shape"])

    def test_url_dataset_is_managed_external_not_bundled_payload(self):
        self.assertNotIn(URL_DATASET_PATH, list_datasets())
        self.assertIn(URL_DATASET_PATH, list_external_datasets())
        self.assertIn(self.large_external_dataset_path, list_external_datasets())

        metadata = external_dataset_metadata(URL_DATASET_PATH)
        self.assertFalse(metadata["vendored"])
        self.assertEqual(metadata["n_features"], URL_DATASET_N_FEATURES)
        self.assertEqual(metadata["max_days"], 120)
        self.assertIn("3_231_961", metadata["shape"])
        self.assertIn("archive.ics.uci.edu", metadata["url"])

        large_metadata = external_dataset_metadata(self.large_external_dataset_path)
        self.assertFalse(large_metadata["vendored"])
        self.assertEqual(large_metadata["source"], LOCAL_EXTERNAL_DATASETS_SOURCE)
        self.assertEqual(large_metadata["url"], KDD2010_TRAIN_DATASET_URL)
        self.assertEqual(large_metadata["checksum_url"], KDD2010_TRAIN_DATASET_SHA256_URL)
        self.assertEqual(large_metadata["release_manifest_url"], DATASET_RELEASE_MANIFEST_URL)
        self.assertIn("data_home", large_metadata["note"])

    def test_large_external_dataset_loads_from_explicit_data_home(self):
        with tempfile.TemporaryDirectory() as data_home:
            payload = Path(data_home) / self.large_external_dataset_path
            payload.parent.mkdir(parents=True)
            with bz2.open(payload, "wt", encoding="utf-8") as stream:
                stream.write("1 1:0.5 3:1.0\n")
                stream.write("-1 2:2.0\n")

            x_data, y_data = load_dataset(
                self.large_external_dataset_path,
                data_home=data_home,
                n_features=3,
            )

            self.assertEqual(x_data.shape, (2, 3))
            np.testing.assert_array_equal(y_data, np.array([1.0, -1.0]))

    def test_large_external_dataset_downloads_from_release_and_validates(self):
        payload_bytes = bz2.compress(b"1 1:0.5 3:1.0\n-1 2:2.0\n")
        payload_sha256 = hashlib.sha256(payload_bytes).hexdigest()
        metadata = {
            **datasets.EXTERNAL_DATASETS[self.large_external_dataset_path],
            "url": "https://example.invalid/kdd2010.trn.bz2",
            "size_bytes": len(payload_bytes),
            "sha256": payload_sha256,
        }

        with tempfile.TemporaryDirectory() as data_home:
            with patch.dict(datasets.EXTERNAL_DATASETS, {self.large_external_dataset_path: metadata}):
                with patch("hawkes_tools.datasets.urlopen", return_value=_BytesResponse(payload_bytes)) as mocked:
                    downloaded_path = download_dataset(
                        self.large_external_dataset_path,
                        data_home=data_home,
                        verbose=False,
                    )
                    x_data, y_data = fetch_dataset(
                        self.large_external_dataset_path,
                        data_home=data_home,
                        n_features=3,
                        verbose=False,
                    )

                mocked.assert_called_once_with("https://example.invalid/kdd2010.trn.bz2", timeout=120)
                self.assertEqual(downloaded_path, Path(data_home) / self.large_external_dataset_path)
                self.assertEqual(_sha256_file(downloaded_path), payload_sha256)
                self.assertEqual(x_data.shape, (2, 3))
                np.testing.assert_array_equal(y_data, np.array([1.0, -1.0]))

                with patch("hawkes_tools.datasets.urlopen", side_effect=AssertionError("cache miss")):
                    x_cached, y_cached = fetch_dataset(
                        self.large_external_dataset_path,
                        data_home=data_home,
                        n_features=3,
                        verbose=False,
                    )
                self.assertEqual(x_cached.shape, (2, 3))
                np.testing.assert_array_equal(y_cached, y_data)

    def test_large_external_dataset_rejects_bad_release_checksum(self):
        payload_bytes = bz2.compress(b"1 1:0.5\n")
        metadata = {
            **datasets.EXTERNAL_DATASETS[self.large_external_dataset_path],
            "url": "https://example.invalid/kdd2010.trn.bz2",
            "size_bytes": len(payload_bytes),
            "sha256": "0" * 64,
        }

        with tempfile.TemporaryDirectory() as data_home:
            with patch.dict(datasets.EXTERNAL_DATASETS, {self.large_external_dataset_path: metadata}):
                with patch("hawkes_tools.datasets.urlopen", return_value=_BytesResponse(payload_bytes)):
                    with self.assertRaisesRegex(ValueError, "SHA-256"):
                        download_dataset(
                            self.large_external_dataset_path,
                            data_home=data_home,
                            verbose=False,
                        )
            self.assertFalse((Path(data_home) / self.large_external_dataset_path).exists())

    def test_url_dataset_rejects_invalid_day_requests_before_download(self):
        invalid_n_days = [0, -1, 121, True, 1.5, "2"]
        with tempfile.TemporaryDirectory() as data_home:
            for n_days in invalid_n_days:
                with self.subTest(n_days=n_days):
                    with self.assertRaisesRegex(ValueError, "n_days"):
                        fetch_url_dataset(n_days=n_days, data_home=data_home, verbose=False)

        for days in [[], [-1], [120], [True], [1.5], ["2"]]:
            with self.subTest(days=days):
                with self.assertRaisesRegex(ValueError, "URL dataset days|at least one"):
                    load_url_dataset_day("does-not-need-to-exist.tar.gz", days)

    def test_data_home_uses_hawkes_tools_environment_only(self):
        with tempfile.TemporaryDirectory() as root:
            hawkes_home = Path(root) / "hawkes"
            old_our_home = Path(root) / "old-our"

            with patch.dict(
                os.environ,
                {
                    "HAWKES_TOOLS_DATASETS": str(hawkes_home),
                    "OUR_HAWKES_DATASETS": str(old_our_home),
                },
                clear=False,
            ):
                self.assertEqual(get_data_home(), hawkes_home)

            with patch.dict(
                os.environ,
                {
                    "OUR_HAWKES_DATASETS": str(old_our_home),
                },
                clear=True,
            ):
                with patch("pathlib.Path.home", return_value=Path(root)):
                    with self.assertWarnsRegex(RuntimeWarning, "HAWKES_TOOLS_DATASETS"):
                        self.assertEqual(get_data_home(), Path(root) / "hawkes_tools_datasets")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class _BytesResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
        return False


if __name__ == "__main__":
    unittest.main()

