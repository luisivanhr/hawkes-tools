# Vendored dataset payloads

These files are vendored from the public dataset archive used during the
standalone port.

Source archive SHA: `9d959b6e53e17145e93e9849ff1f9f6d2de8ae51`.

The bundled payloads mirror the smaller public payload blobs in that tree and
let `hawkes_tools.datasets` load those public datasets without importing the
original compiled package or requiring network access at runtime. The large
KDD2010 training payload is managed through a GitHub release-backed external
cache instead of package data. Repository support scripts and README files are
intentionally not package payloads.

Included payload groups:

- `binary/adult`
- `binary/covtype`
- `binary/ijcnn1`
- `binary/kdd2010`
- `binary/reuters`
- `hawkes/bund`
- `regression/abalone`

The UCI URL reputation tarball and the KDD2010 training payload are not part of
the bundled archive. The URL standalone fetcher remains represented as a
managed external dataset and available through `download_url_dataset` and
`fetch_url_dataset`; the KDD2010 training payload downloads from the
hawkes-tools `dataset` GitHub release into `data_home` or `HAWKES_TOOLS_DATASETS`.

