# Vendored dataset payloads

These files are vendored from the public dataset archive used during the
standalone port.

Source archive SHA: `9d959b6e53e17145e93e9849ff1f9f6d2de8ae51`.

The bundled payloads mirror every public payload blob in that tree and let
`hawkes_tools.datasets` load the same public datasets without importing the
original compiled package or requiring network access at runtime. Repository
support scripts and README files are intentionally not package payloads.

Included payload groups:

- `binary/adult`
- `binary/covtype`
- `binary/ijcnn1`
- `binary/kdd2010`
- `binary/reuters`
- `hawkes/bund`
- `regression/abalone`

The UCI URL reputation tarball is not part of the bundled archive. The
standalone fetcher for that tarball remains represented as a managed external
dataset and available through `download_url_dataset` and `fetch_url_dataset`.

