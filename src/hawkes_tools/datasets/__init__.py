"""Load vendored payloads and explicit external datasets."""

from __future__ import annotations

import bz2
import hashlib
import math
import os
import shutil
import tarfile
import warnings
from collections.abc import Iterable
from pathlib import Path, PurePosixPath
from urllib.request import urlopen

import numpy as np
import scipy.sparse

DEFAULT_URL_DATASET_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/url/"
    "url_svmlight.tar.gz"
)
URL_DATASET_PATH = "url/url_svmlight.tar.gz"
URL_DATASET_N_FEATURES = 3_231_961
URL_DATASET_MAX_DAYS = 120
DATASETS_SOURCE = "bundled public dataset archive"
RELEASE_DATASETS_SOURCE = "hawkes-tools GitHub release dataset asset"
LOCAL_EXTERNAL_DATASETS_SOURCE = RELEASE_DATASETS_SOURCE
DATASETS_TREE_SHA = "9d959b6e53e17145e93e9849ff1f9f6d2de8ae51"
DATASET_RELEASE_TAG = "dataset"
DATASET_RELEASE_BASE_URL = (
    f"https://github.com/luisivanhr/hawkes-tools/releases/download/{DATASET_RELEASE_TAG}"
)
DATASET_RELEASE_MANIFEST_URL = f"{DATASET_RELEASE_BASE_URL}/hawkes-tools-datasets.json"
KDD2010_TRAIN_DATASET_PATH = "binary/kdd2010/kdd2010.trn.bz2"
KDD2010_TRAIN_DATASET_URL = f"{DATASET_RELEASE_BASE_URL}/kdd2010.trn.bz2"
KDD2010_TRAIN_DATASET_SHA256_URL = f"{DATASET_RELEASE_BASE_URL}/kdd2010.trn.bz2.sha256"

_DATA_HOME_ENV = "HAWKES_TOOLS_DATASETS"
_DEFAULT_HOME_NAME = "hawkes_tools_datasets"
VENDORED_DATASETS_PATH = Path(__file__).resolve().parent / "vendor" / "datasets"
LOCAL_EXTERNAL_DATASET_FILES: tuple[str, ...] = (
    KDD2010_TRAIN_DATASET_PATH,
)

# Data payloads from the crawled upstream dataset tree. Support scripts and
# README files are intentionally excluded.
DATASET_FILES: tuple[str, ...] = (
    "binary/adult/adult.trn.bz2",
    "binary/adult/adult.tst.bz2",
    "binary/covtype/covtype.trn.bz2",
    "binary/ijcnn1/ijcnn1.trn.bz2",
    "binary/ijcnn1/ijcnn1.tst.bz2",
    KDD2010_TRAIN_DATASET_PATH,
    "binary/kdd2010/kdd2010.tst.bz2",
    "binary/reuters/reuters.trn.bz2",
    "binary/reuters/reuters.tst.bz2",
    "hawkes/bund/bund.npz",
    "regression/abalone/abalone.trn.bz2",
)
VENDORED_DATASET_FILES: tuple[str, ...] = tuple(
    dataset_path for dataset_path in DATASET_FILES if dataset_path not in LOCAL_EXTERNAL_DATASET_FILES
)

DATASET_MANIFEST: dict[str, dict[str, object]] = {
    "hawkes/bund/bund.npz": {
        "source": DATASETS_SOURCE,
        "format": "npz object array",
        "shape": "20 realizations keyed by trading date; each realization has 4 timestamp streams",
        "event_count_range": (28_361, 56_086),
        "stream_event_count_range": (3_371, 23_381),
        "size_bytes": 6_161_595,
        "sha256": "fa1d4679fff7f6e2ee7819cbc461df14056bf1786ad0c98cf9e9019feff4bd8c",
    },
    "binary/adult/adult.trn.bz2": {
        "source": DATASETS_SOURCE,
        "format": "binary SVMlight bz2",
        "x_shape": (32_561, 123),
        "y_shape": (32_561,),
        "nnz": 451_592,
        "size_bytes": 100_554,
        "sha256": "73ce74d43bbfc988b08e6ecf738cf5484f19dafa143c31f702f3a8b779d277e5",
    },
    "binary/adult/adult.tst.bz2": {
        "source": DATASETS_SOURCE,
        "format": "binary SVMlight bz2",
        "x_shape": (16_281, 123),
        "y_shape": (16_281,),
        "nnz": 225_732,
        "size_bytes": 50_588,
        "sha256": "bdb2ec9642e77947a67269471506b18bc4dcf03413f44814dc24833432ba818d",
    },
    "binary/covtype/covtype.trn.bz2": {
        "source": DATASETS_SOURCE,
        "format": "binary SVMlight bz2",
        "x_shape": (581_012, 54),
        "y_shape": (581_012,),
        "nnz": 6_940_438,
        "size_bytes": 8_205_541,
        "sha256": "31731bccafcff5595fe34cc6ee15705bec7d15c962020c3f4280c4ecea74244d",
    },
    "binary/ijcnn1/ijcnn1.trn.bz2": {
        "source": DATASETS_SOURCE,
        "format": "binary SVMlight bz2",
        "x_shape": (49_990, 22),
        "y_shape": (49_990,),
        "nnz": 649_870,
        "size_bytes": 1_395_175,
        "sha256": "b38a4198ce142d43e96fb86d17b87211feec1f909ac344fa057455c69de29de3",
    },
    "binary/ijcnn1/ijcnn1.tst.bz2": {
        "source": DATASETS_SOURCE,
        "format": "binary SVMlight bz2",
        "x_shape": (91_701, 22),
        "y_shape": (91_701,),
        "nnz": 1_192_113,
        "size_bytes": 2_597_597,
        "sha256": "acd1db430684fdd3f0c768ecc49aa8e749d0004fc117611220c2aa576a7b6a24",
    },
    KDD2010_TRAIN_DATASET_PATH: {
        "source": RELEASE_DATASETS_SOURCE,
        "format": "binary SVMlight bz2",
        "x_shape": (19_264_097, 1_129_522),
        "y_shape": (19_264_097,),
        "nnz": 173_376_873,
        "size_bytes": 100_571_441,
        "sha256": "41f633927ed2d5f6d634f446bb007eb957128d806ec03bbc1b471bc6ee330a28",
    },
    "binary/kdd2010/kdd2010.tst.bz2": {
        "source": DATASETS_SOURCE,
        "format": "binary SVMlight bz2",
        "x_shape": (748_401, 1_163_024),
        "y_shape": (748_401,),
        "nnz": 6_735_609,
        "size_bytes": 4_402_617,
        "sha256": "00148b70303ab8497b3b1682f9385e13b1bff2940393e1b140a773ea752b77ce",
    },
    "binary/reuters/reuters.trn.bz2": {
        "source": DATASETS_SOURCE,
        "format": "binary SVMlight bz2",
        "x_shape": (7_770, 8_315),
        "y_shape": (7_770,),
        "nnz": 339_837,
        "size_bytes": 3_107_267,
        "sha256": "e77a3368662c02caebe0d9cbcc97d856019bcfb33a8e9b7c48e0b51ee4cdfb60",
    },
    "binary/reuters/reuters.tst.bz2": {
        "source": DATASETS_SOURCE,
        "format": "binary SVMlight bz2",
        "x_shape": (3_299, 8_315),
        "y_shape": (3_299,),
        "nnz": 136_821,
        "size_bytes": 763_856,
        "sha256": "3d699b9644a38160bf3228059d18fbfc07474057d8e82d512576a2b82d72e211",
    },
    "regression/abalone/abalone.trn.bz2": {
        "source": DATASETS_SOURCE,
        "format": "regression SVMlight bz2",
        "x_shape": (4_177, 8),
        "y_shape": (4_177,),
        "nnz": 32_080,
        "size_bytes": 52_435,
        "sha256": "9d24998fecb496cca34cb56386f99d36edc2aa113f800b1f7c02316e05da80f1",
    },
}

EXTERNAL_DATASETS: dict[str, dict[str, object]] = {
    KDD2010_TRAIN_DATASET_PATH: {
        **DATASET_MANIFEST[KDD2010_TRAIN_DATASET_PATH],
        "url": KDD2010_TRAIN_DATASET_URL,
        "checksum_url": KDD2010_TRAIN_DATASET_SHA256_URL,
        "release_manifest_url": DATASET_RELEASE_MANIFEST_URL,
        "vendored": False,
        "note": "Large release-backed payload; downloaded into data_home on first fetch.",
    },
    URL_DATASET_PATH: {
        "url": DEFAULT_URL_DATASET_URL,
        "source": "UCI Machine Learning Repository",
        "format": "tar.gz of SVMlight day files",
        "shape": "X CSR shape (sum selected Day*.svm rows, 3_231_961); y shape (same row count,)",
        "n_features": URL_DATASET_N_FEATURES,
        "max_days": URL_DATASET_MAX_DAYS,
        "vendored": False,
        "note": "Managed external URL reputation dataset; not part of the bundled payloads.",
    },
}


def get_data_home(data_home: str | os.PathLike[str] | None = None) -> Path:
    """Return the cache directory for downloaded datasets.

    Resolution order is explicit ``data_home``, ``HAWKES_TOOLS_DATASETS``,
    then ``~/hawkes_tools_datasets``.
    """

    if data_home is None:
        data_home = os.environ.get(_DATA_HOME_ENV)
    if data_home is None:
        data_home = Path.home() / _DEFAULT_HOME_NAME
        warnings.warn(
            f"{_DATA_HOME_ENV} was not set. Saving datasets to {data_home}",
            RuntimeWarning,
            stacklevel=2,
        )
    path = Path(data_home).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    return path


def list_datasets() -> tuple[str, ...]:
    """Return all known dataset payload paths."""

    return DATASET_FILES


def dataset_metadata(dataset_path: str) -> dict[str, object]:
    """Return shape/source metadata for a known dataset payload."""

    normalized = _normalize_dataset_path(dataset_path)
    if normalized not in DATASET_MANIFEST:
        raise KeyError(f"unknown dataset: {normalized}")
    metadata = dict(DATASET_MANIFEST[normalized])
    metadata.setdefault("vendored", normalized in VENDORED_DATASET_FILES)
    return metadata


def list_external_datasets() -> tuple[str, ...]:
    """Return managed external dataset payload paths."""

    return tuple(EXTERNAL_DATASETS)


def external_dataset_metadata(dataset_path: str) -> dict[str, object]:
    """Return metadata for a managed external dataset."""

    normalized = _normalize_dataset_path(dataset_path)
    if normalized not in EXTERNAL_DATASETS:
        raise KeyError(f"unknown external dataset: {normalized}")
    return dict(EXTERNAL_DATASETS[normalized])


def vendored_dataset_path(dataset_path: str) -> Path:
    """Return the bundled path for a dataset payload."""

    return VENDORED_DATASETS_PATH / Path(_normalize_dataset_path(dataset_path))


def is_dataset_vendored(dataset_path: str) -> bool:
    """Return whether a dataset payload is bundled with the package."""

    return vendored_dataset_path(dataset_path).exists()


def _download_dataset(
    source_url: str,
    dataset_path: str,
    data_home: str | os.PathLike[str] | None = None,
    *,
    force: bool = False,
    verbose: bool = True,
    expected_sha256: str | None = None,
    expected_size: int | None = None,
) -> Path:
    """Download ``source_url`` into ``data_home / dataset_path``."""

    data_home_path = get_data_home(data_home)
    cache_path = _cache_path(dataset_path, data_home_path)
    if cache_path.exists() and not force:
        _validate_dataset_file(cache_path, expected_sha256=expected_sha256, expected_size=expected_size)
        return cache_path
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if verbose:
        print(f"Downloading dataset from {source_url}")
    temp_path = cache_path.with_name(cache_path.name + ".download")
    if temp_path.exists():
        temp_path.unlink()
    try:
        with urlopen(source_url, timeout=120) as response, temp_path.open("wb") as output:
            shutil.copyfileobj(response, output)
        _validate_dataset_file(temp_path, expected_sha256=expected_sha256, expected_size=expected_size)
        temp_path.replace(cache_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()
    return cache_path


def fetch_dataset(
    dataset_path: str,
    data_home: str | os.PathLike[str] | None = None,
    n_features: int | None = None,
    *,
    force: bool = False,
    verbose: bool = True,
):
    """Fetch and load a dataset payload from cache, package data, or release.

    ``.npz`` files load as either a single array or a tuple of ``(key, array)``
    pairs. SVMlight-style ``.bz2`` files load as ``(X, y)`` with ``X`` a SciPy
    CSR matrix.
    """

    normalized = _normalize_dataset_path(dataset_path)
    source_path = None if force else _resolve_existing_dataset_path(normalized, data_home)
    if source_path is None:
        source_path = _download_external_dataset_if_available(
            normalized,
            data_home=data_home,
            force=force,
            verbose=verbose,
        )
    if source_path is None:
        raise _missing_dataset_error(normalized)
    _validate_external_dataset_if_managed(normalized, source_path)
    return _load_dataset_path(source_path, n_features=n_features)


def load_dataset(
    dataset_path: str,
    data_home: str | os.PathLike[str] | None = None,
    n_features: int | None = None,
):
    """Load a dataset payload already present in the local cache."""

    source_path = _resolve_existing_dataset_path(dataset_path, data_home)
    if source_path is None:
        raise _missing_dataset_error(_normalize_dataset_path(dataset_path))
    return _load_dataset_path(source_path, n_features=n_features)


def _load_dataset_path(
    source_path: Path,
    n_features: int | None = None,
):
    """Load a dataset payload from a concrete filesystem path."""

    cache_path = Path(source_path)
    suffixes = "".join(cache_path.suffixes)
    if suffixes.endswith(".npz"):
        with np.load(cache_path, allow_pickle=True) as dataset:
            if len(dataset.files) == 1:
                return dataset[dataset.files[0]]
            return tuple((key, dataset[key]) for key in dataset.files)
    return load_svmlight_file(cache_path, n_features=n_features)


def fetch_hawkes_bund_data(
    data_home: str | os.PathLike[str] | None = None,
) -> list[list[np.ndarray]]:
    """Load the 20-day, 4-stream Bund Hawkes dataset."""

    dataset = fetch_dataset(
        "hawkes/bund/bund.npz",
        data_home=data_home,
    )
    return [list(timestamps) for _, timestamps in dataset]


def download_url_dataset(
    data_home: str | os.PathLike[str] | None = None,
    *,
    force: bool = False,
    verbose: bool = True,
    dataset_url: str = DEFAULT_URL_DATASET_URL,
) -> Path:
    """Download the URL reputation SVMlight tarball."""

    return _download_dataset(
        dataset_url,
        URL_DATASET_PATH,
        data_home=data_home,
        force=force,
        verbose=verbose,
    )


def download_dataset(
    dataset_path: str,
    data_home: str | os.PathLike[str] | None = None,
    *,
    force: bool = False,
    verbose: bool = True,
) -> Path:
    """Download a managed external dataset payload and return its cache path."""

    normalized = _normalize_dataset_path(dataset_path)
    cache_path = _download_external_dataset_if_available(
        normalized,
        data_home=data_home,
        force=force,
        verbose=verbose,
    )
    if cache_path is None:
        raise KeyError(f"unknown managed downloadable dataset: {normalized}")
    return cache_path


def load_url_dataset_day(
    cache_path: str | os.PathLike[str],
    days: Iterable[int],
) -> tuple[scipy.sparse.csr_matrix, np.ndarray]:
    """Load one or more day files from the URL reputation tarball."""

    validated_days = _validate_url_days(days)
    features = []
    labels = []
    with tarfile.open(cache_path, "r:gz") as tar_file:
        for day in validated_days:
            member = tar_file.extractfile(f"url_svmlight/Day{day}.svm")
            if member is None:
                raise FileNotFoundError(f"Day{day}.svm not found in {cache_path}")
            x_day, y_day = load_svmlight_file(member, n_features=URL_DATASET_N_FEATURES)
            features.append(x_day)
            labels.append(y_day)
    return scipy.sparse.vstack(features, format="csr"), np.hstack(labels)


def fetch_url_dataset(
    n_days: int = 120,
    data_home: str | os.PathLike[str] | None = None,
    *,
    force: bool = False,
    verbose: bool = True,
    dataset_url: str = DEFAULT_URL_DATASET_URL,
) -> tuple[scipy.sparse.csr_matrix, np.ndarray]:
    """Fetch and load the URL reputation dataset."""

    n_days = _validate_n_days(n_days)
    data_home_path = get_data_home(data_home)
    cache_path = _cache_path(URL_DATASET_PATH, data_home_path)
    if force or not cache_path.exists():
        download_url_dataset(
            data_home=data_home_path,
            force=force,
            verbose=verbose,
            dataset_url=dataset_url,
        )
    return load_url_dataset_day(cache_path, range(n_days))


def load_svmlight_file(
    path_or_file,
    n_features: int | None = None,
) -> tuple[scipy.sparse.csr_matrix, np.ndarray]:
    """Load a SVMlight/libsvm file into ``(X, y)`` without scikit-learn."""

    close_file = False
    if hasattr(path_or_file, "read"):
        stream = path_or_file
    else:
        path = Path(path_or_file)
        if path.suffix == ".bz2":
            stream = bz2.open(path, "rb")
        else:
            stream = path.open("rb")
        close_file = True
    try:
        rows: list[int] = []
        cols: list[int] = []
        values: list[float] = []
        labels: list[float] = []
        min_col = math.inf
        max_col = -1
        for row, raw_line in enumerate(stream):
            line = raw_line.decode("utf-8", errors="replace") if isinstance(raw_line, bytes) else raw_line
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            tokens = line.split()
            labels.append(float(tokens[0]))
            for token in tokens[1:]:
                if ":" not in token:
                    continue
                col_text, value_text = token.split(":", 1)
                col = int(col_text)
                min_col = min(min_col, col)
                max_col = max(max_col, col)
                rows.append(len(labels) - 1)
                cols.append(col)
                values.append(float(value_text))
        if labels:
            zero_based = min_col == 0
            if not zero_based:
                cols = [col - 1 for col in cols]
                max_col -= 1
            inferred_features = max_col + 1
        else:
            inferred_features = 0
        n_columns = int(n_features) if n_features is not None else max(inferred_features, 0)
        matrix = scipy.sparse.csr_matrix(
            (values, (rows, cols)),
            shape=(len(labels), n_columns),
            dtype=float,
        )
        return matrix, np.asarray(labels, dtype=float)
    finally:
        if close_file:
            stream.close()


def clear_dataset(
    dataset_path: str,
    data_home: str | os.PathLike[str] | None = None,
) -> None:
    """Remove one cached dataset payload if present."""

    path = _cache_path(dataset_path, get_data_home(data_home))
    if path.exists():
        path.unlink()


def clear_data_home(data_home: str | os.PathLike[str] | None = None) -> None:
    """Delete the whole dataset cache directory."""

    path = get_data_home(data_home)
    if path.exists():
        shutil.rmtree(path)


def _normalize_dataset_path(dataset_path: str) -> str:
    normalized = str(dataset_path).replace("\\", "/").lstrip("/")
    path = PurePosixPath(normalized)
    if (
        not normalized
        or path.is_absolute()
        or ".." in path.parts
        or (path.parts and ":" in path.parts[0])
    ):
        raise ValueError(f"dataset_path must stay inside the dataset cache: {dataset_path!r}")
    return path.as_posix()


def _validate_n_days(n_days) -> int:
    if isinstance(n_days, bool):
        raise ValueError("n_days must be an integer between 1 and 120")
    try:
        value = int(n_days)
    except (TypeError, ValueError) as exc:
        raise ValueError("n_days must be an integer between 1 and 120") from exc
    if value != n_days or not 1 <= value <= URL_DATASET_MAX_DAYS:
        raise ValueError("n_days must be an integer between 1 and 120")
    return value


def _validate_url_days(days: Iterable[int]) -> tuple[int, ...]:
    validated = []
    for day in days:
        if isinstance(day, bool):
            raise ValueError("URL dataset days must be integers in [0, 119]")
        try:
            value = int(day)
        except (TypeError, ValueError) as exc:
            raise ValueError("URL dataset days must be integers in [0, 119]") from exc
        if value != day or not 0 <= value < URL_DATASET_MAX_DAYS:
            raise ValueError("URL dataset days must be integers in [0, 119]")
        validated.append(value)
    if not validated:
        raise ValueError("at least one URL dataset day is required")
    return tuple(validated)


def _cache_path(dataset_path: str, data_home: Path) -> Path:
    return data_home / Path(_normalize_dataset_path(dataset_path))


def _candidate_cache_path(
    dataset_path: str,
    data_home: str | os.PathLike[str] | None,
) -> Path | None:
    if data_home is None:
        data_home = os.environ.get(_DATA_HOME_ENV)
        if data_home is None:
            return None
    return _cache_path(dataset_path, Path(data_home).expanduser())


def _resolve_existing_dataset_path(
    dataset_path: str,
    data_home: str | os.PathLike[str] | None,
) -> Path | None:
    cache_path = _candidate_cache_path(dataset_path, data_home)
    if cache_path is not None and cache_path.exists():
        return cache_path
    vendor_path = vendored_dataset_path(dataset_path)
    if vendor_path.exists():
        return vendor_path
    return None


def _download_external_dataset_if_available(
    normalized: str,
    data_home: str | os.PathLike[str] | None,
    *,
    force: bool,
    verbose: bool,
) -> Path | None:
    metadata = EXTERNAL_DATASETS.get(normalized)
    if metadata is None or normalized not in DATASET_MANIFEST:
        return None
    source_url = metadata.get("url")
    if not isinstance(source_url, str):
        return None
    return _download_dataset(
        source_url,
        normalized,
        data_home=data_home,
        force=force,
        verbose=verbose,
        expected_sha256=_metadata_sha256(metadata),
        expected_size=_metadata_size(metadata),
    )


def _validate_external_dataset_if_managed(normalized: str, source_path: Path) -> None:
    metadata = EXTERNAL_DATASETS.get(normalized)
    if metadata is None or normalized not in DATASET_MANIFEST:
        return
    _validate_dataset_file(
        source_path,
        expected_sha256=_metadata_sha256(metadata),
        expected_size=_metadata_size(metadata),
    )


def _validate_dataset_file(
    path: Path,
    *,
    expected_sha256: str | None = None,
    expected_size: int | None = None,
) -> None:
    if expected_size is not None and path.stat().st_size != expected_size:
        raise ValueError(
            f"dataset file has unexpected size for {path}: "
            f"{path.stat().st_size} != {expected_size}"
        )
    if expected_sha256 is not None:
        actual_sha256 = _sha256_file(path)
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"dataset file has unexpected SHA-256 for {path}: "
                f"{actual_sha256} != {expected_sha256}"
            )


def _metadata_sha256(metadata: dict[str, object]) -> str | None:
    value = metadata.get("sha256")
    return value if isinstance(value, str) else None


def _metadata_size(metadata: dict[str, object]) -> int | None:
    value = metadata.get("size_bytes")
    return value if isinstance(value, int) else None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _missing_dataset_error(normalized: str) -> FileNotFoundError:
    if normalized in LOCAL_EXTERNAL_DATASET_FILES:
        return FileNotFoundError(
            f"dataset is not vendored and could not be found or downloaded into data_home "
            f"or {_DATA_HOME_ENV}: {normalized}"
        )
    return FileNotFoundError(f"dataset is neither cached nor vendored: {normalized}")

