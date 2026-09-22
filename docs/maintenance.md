# Maintenance

## Repository Hygiene

Keep local prompt and agent files out of the repository. `.gitignore` excludes
local agent directories, common prompt file names, Python/Numba caches, and
disposable experiment outputs. Before committing:

```powershell
git status --short
```

If generated cache files appear, clean only generated cache artifacts. Do not
use broad destructive commands against the repository.

## Experiment Storage

Keep this checkout focused on the standalone `hawkes_tools` library, its tests,
examples, benchmarks and public documentation. Renewal-Hawkes estimation studies
now live in the separate sibling workspace `../renewal-hawkes-research/`.
Its README identifies the active estimator, current study, preserved findings,
latest fitted curves and the verified relocation inventory.

New research studies must be created in that research workspace. Put raw
observations, evaluation caches and compiled/runtime scratch in its ignored
`_artifacts/<study-id>/` directory. Keep scripts, seed registries, configuration,
receipts, selected models, reports, final figures and plotted arrays with the
study. Never put the sole seed record or finding in disposable output.

For library-specific large benchmarks, this checkout retains the read-only
`tools/check_experiment_storage.py` preflight. Pass the peak additional disk
requirement, including temporary and simultaneous worker output. Default limits
are 1 GiB of Git data, 5 GiB of project data including expected growth, and a
10 GiB free-space reserve. New library benchmark scratch belongs in its ignored
`_artifacts/` directory. This preflight is not a background quota.

The research workspace excludes its immutable historical archive from ordinary
Git staging and snapshots. The active numerical source closure is separately
vendored and checksum-pinned; new studies have their own source locks, peak
storage checks and automatic reclamation of replayed evaluation caches. This
keeps future runtime imports independent of obsolete worktrees and their raw
simulations. See that workspace's `STORAGE_POLICY.md` for retention rules.

The 2026-09-22 separation preserved all research source, findings, reports,
seed/input records and the latest study's observations. It deleted obsolete
simulation/trace archives and retired the two old worktrees after retaining
unique files and edits. Named Git history was retained. The exact old-to-new
path and SHA-256 mapping lives in the research maintenance records. Original
historical manifests remain unchanged and can refer to deliberately deleted
obsolete outputs; they are evidence, not current resume permissions.

Ignore rules do not untrack existing files or override `git add -f`. Stage
source and report paths explicitly. Do not use immediate Git object pruning,
history deletion or whole-tree backup archives as routine storage management.

## Documentation Policy

Keep this `docs/` tree independent from `README.md`. The README can stay as a
short public landing page, while this directory carries detailed installation,
dataset, API, validation, and maintenance material.

When changing behavior:

- Update the relevant file under `docs/`.
- Update `PARITY.md` when support status or validation baselines change.
- Avoid adding prompt or agent instructions as tracked repository files.

## Dataset Release Assets

The large KDD2010 training payload is intentionally not tracked in Git. It is
hosted as assets on the `dataset` GitHub release:

```text
kdd2010.trn.bz2
kdd2010.trn.bz2.sha256
hawkes-tools-datasets.json
```

Current payload metadata:

```text
size_bytes = 100571441
sha256 = 41f633927ed2d5f6d634f446bb007eb957128d806ec03bbc1b471bc6ee330a28
```

If the payload changes:

1. Upload the new `kdd2010.trn.bz2` release asset.
2. Regenerate and upload `kdd2010.trn.bz2.sha256`.
3. Regenerate and upload `hawkes-tools-datasets.json`.
4. Update constants and metadata in `src/hawkes_tools/datasets/__init__.py`.
5. Run `python -m unittest tests.test_datasets`.
6. Run the full test suite before committing.

## Dataset Package Data

Package-data entries live in `pyproject.toml`. The KDD2010 test payload remains
bundled:

```text
vendor/datasets/binary/kdd2010/kdd2010.tst.bz2
```

The KDD2010 training payload must not be reintroduced under package data unless
there is an explicit packaging decision to ship a roughly 96 MB wheel payload.

## Validation Baseline

Before publication or behavior-changing commits, run:

```powershell
python -m unittest discover -s tests
python tests\hawkes_equivalence\report_equivalence.py
```

For publication work, also build a wheel and inspect the resulting package
contents for:

- Expected package data.
- No generated cache files.
- No prompt or agent artifacts.
- No large external dataset payloads.

## Release-Backed Dataset Fetch Path

The external KDD2010 training dataset should be fetched with:

```python
from hawkes_tools.datasets import download_dataset

download_dataset("binary/kdd2010/kdd2010.trn.bz2", data_home="C:/data/hawkes-tools-datasets")
```

The fetch helper validates byte size and SHA-256 before moving a downloaded
temporary file into the cache path. Keep that atomic write-and-validate pattern
when extending release-backed datasets.
