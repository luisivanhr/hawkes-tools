# Datasets

`hawkes_tools.datasets` manages three kinds of payloads:

- Package-data payloads bundled inside the wheel.
- Release-backed payloads downloaded into a local cache on first use.
- Managed external archives with explicit loader helpers.

The cache root is controlled by `data_home=` or by the
`HAWKES_TOOLS_DATASETS` environment variable. If neither is set, helpers that
need a writable cache use `~/hawkes_tools_datasets` and emit a warning.

## Common Helpers

```python
from hawkes_tools.datasets import (
    dataset_metadata,
    download_dataset,
    external_dataset_metadata,
    fetch_dataset,
    fetch_hawkes_bund_data,
    list_datasets,
    list_external_datasets,
)
```

Use `fetch_dataset(...)` when you want to load the payload. Use
`download_dataset(...)` when you want to prepopulate the cache for a managed
external payload without loading a large matrix into memory.

## Bundled Payloads

These payloads are shipped as package data:

| Dataset path | Format |
| --- | --- |
| `hawkes/bund/bund.npz` | Object-array Hawkes timestamp payload |
| `binary/adult/adult.trn.bz2` | Binary SVMlight |
| `binary/adult/adult.tst.bz2` | Binary SVMlight |
| `binary/covtype/covtype.trn.bz2` | Binary SVMlight |
| `binary/ijcnn1/ijcnn1.trn.bz2` | Binary SVMlight |
| `binary/ijcnn1/ijcnn1.tst.bz2` | Binary SVMlight |
| `binary/kdd2010/kdd2010.tst.bz2` | Binary SVMlight |
| `binary/reuters/reuters.trn.bz2` | Binary SVMlight |
| `binary/reuters/reuters.tst.bz2` | Binary SVMlight |
| `regression/abalone/abalone.trn.bz2` | Regression SVMlight |

Example:

```python
from hawkes_tools.datasets import fetch_hawkes_bund_data

timestamps = fetch_hawkes_bund_data()
```

## Release-Backed KDD2010 Training Payload

The large KDD2010 training file is not shipped in the repository or wheel. It
is hosted on the `dataset` GitHub release:

```text
https://github.com/luisivanhr/hawkes-tools/releases/download/dataset/kdd2010.trn.bz2
```

Metadata:

| Field | Value |
| --- | --- |
| Dataset path | `binary/kdd2010/kdd2010.trn.bz2` |
| Size | `100_571_441` bytes |
| SHA-256 | `41f633927ed2d5f6d634f446bb007eb957128d806ec03bbc1b471bc6ee330a28` |
| Format | Binary SVMlight, compressed with bzip2 |
| `X` shape | `(19_264_097, 1_129_522)` |
| `y` shape | `(19_264_097,)` |
| Nonzeros | `173_376_873` |

Release sidecar assets:

```text
https://github.com/luisivanhr/hawkes-tools/releases/download/dataset/kdd2010.trn.bz2.sha256
https://github.com/luisivanhr/hawkes-tools/releases/download/dataset/hawkes-tools-datasets.json
```

Pre-download the file without loading it:

```python
from hawkes_tools.datasets import download_dataset

path = download_dataset(
    "binary/kdd2010/kdd2010.trn.bz2",
    data_home="C:/data/hawkes-tools-datasets",
)
print(path)
```

Load the full CSR matrix only when you have enough memory:

```python
from hawkes_tools.datasets import fetch_dataset

X, y = fetch_dataset(
    "binary/kdd2010/kdd2010.trn.bz2",
    data_home="C:/data/hawkes-tools-datasets",
)
```

On cache miss, `fetch_dataset` downloads the release asset into the cache and
validates both byte size and SHA-256 before loading.

## URL Reputation Dataset

The URL reputation tarball is a separate managed external dataset from UCI.
Use its dedicated helper:

```python
from hawkes_tools.datasets import fetch_url_dataset

X, y = fetch_url_dataset(n_days=3, data_home="C:/data/hawkes-tools-datasets")
```

`n_days` must be an integer from `1` to `120`. The helper loads Day0 through
Day`n_days - 1`.

## Metadata Inspection

```python
from hawkes_tools.datasets import dataset_metadata, external_dataset_metadata

print(dataset_metadata("hawkes/bund/bund.npz"))
print(external_dataset_metadata("binary/kdd2010/kdd2010.trn.bz2"))
```

`dataset_metadata` returns known shape, size, checksum, and vendored status.
`external_dataset_metadata` additionally exposes managed external URLs when
available.
