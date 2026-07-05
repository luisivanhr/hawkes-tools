import hashlib
import os
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hawkes_tools.datasets import (
    TICK_DATASET_MANIFEST,
    TICK_DATASETS_TREE_SHA,
    URL_DATASET_N_FEATURES,
    URL_DATASET_PATH,
    external_dataset_metadata,
    fetch_hawkes_bund_data,
    fetch_tick_dataset,
    get_data_home,
    is_tick_dataset_vendored,
    list_external_datasets,
    list_tick_datasets,
    tick_dataset_metadata,
    vendored_tick_dataset_path,
    fetch_url_dataset,
    load_url_dataset_day,
)


class DatasetLoaderTest(unittest.TestCase):
    def test_hawkes_bund_data_loads_from_vendored_payload(self):
        timestamps = fetch_hawkes_bund_data()

        self.assertIn("hawkes/bund/bund.npz", list_tick_datasets())
        self.assertEqual(len(timestamps), 20)
        self.assertTrue(all(len(realization) == 4 for realization in timestamps))
        self.assertTrue(all(np.all(np.diff(stream) >= 0.0) for realization in timestamps for stream in realization))
        self.assertGreater(sum(len(stream) for realization in timestamps for stream in realization), 0)

    def test_vendored_svmlight_datasets_load_without_tick(self):
        adult_x, adult_y = fetch_tick_dataset("binary/adult/adult.trn.bz2")
        abalone_x, abalone_y = fetch_tick_dataset("regression/abalone/abalone.trn.bz2")

        self.assertIn("binary/adult/adult.trn.bz2", list_tick_datasets())
        self.assertEqual(adult_x.shape, (32561, 123))
        self.assertEqual(adult_y.shape, (32561,))
        self.assertEqual(abalone_x.shape, (4177, 8))
        self.assertEqual(abalone_y.shape, (4177,))

    def test_tick_dataset_manifest_entries_are_vendored(self):
        manifest = list_tick_datasets()

        self.assertEqual(TICK_DATASETS_TREE_SHA, "9d959b6e53e17145e93e9849ff1f9f6d2de8ae51")
        self.assertEqual(len(manifest), len(set(manifest)))
        self.assertEqual(len(manifest), 11)
        self.assertEqual(set(TICK_DATASET_MANIFEST), set(manifest))
        for dataset_path in manifest:
            with self.subTest(dataset_path=dataset_path):
                self.assertTrue(is_tick_dataset_vendored(dataset_path))
                metadata = tick_dataset_metadata(dataset_path)
                self.assertEqual(metadata["source"], "X-DataInitiative/tick-datasets")
                self.assertIn("format", metadata)
                self.assertTrue("shape" in metadata or "x_shape" in metadata)
                self.assertIn("size_bytes", metadata)
                self.assertIn("sha256", metadata)
                payload_path = vendored_tick_dataset_path(dataset_path)
                self.assertEqual(payload_path.stat().st_size, metadata["size_bytes"])
                self.assertEqual(_sha256_file(payload_path), metadata["sha256"])

        adult_metadata = tick_dataset_metadata("binary/adult/adult.trn.bz2")
        self.assertEqual(adult_metadata["x_shape"], adult_x_shape := (32_561, 123))
        adult_x, adult_y = fetch_tick_dataset("binary/adult/adult.trn.bz2")
        self.assertEqual(adult_x.shape, adult_x_shape)
        self.assertEqual(adult_y.shape, adult_metadata["y_shape"])

    def test_url_dataset_is_managed_external_not_tick_datasets_payload(self):
        self.assertNotIn(URL_DATASET_PATH, list_tick_datasets())
        self.assertIn(URL_DATASET_PATH, list_external_datasets())

        metadata = external_dataset_metadata(URL_DATASET_PATH)
        self.assertFalse(metadata["vendored"])
        self.assertEqual(metadata["n_features"], URL_DATASET_N_FEATURES)
        self.assertEqual(metadata["max_days"], 120)
        self.assertIn("3_231_961", metadata["shape"])
        self.assertIn("archive.ics.uci.edu", metadata["url"])

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
            old_tick_home = Path(root) / "old-tick"

            with patch.dict(
                os.environ,
                {
                    "HAWKES_TOOLS_DATASETS": str(hawkes_home),
                    "OUR_HAWKES_DATASETS": str(old_our_home),
                    "TICK_DATASETS": str(old_tick_home),
                },
                clear=False,
            ):
                self.assertEqual(get_data_home(), hawkes_home)

            with patch.dict(
                os.environ,
                {
                    "OUR_HAWKES_DATASETS": str(old_our_home),
                    "TICK_DATASETS": str(old_tick_home),
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


if __name__ == "__main__":
    unittest.main()

