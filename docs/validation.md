# Validation

Run validation from the repository root.

## Unit Tests

Using the maintainer workstation interpreter:

```powershell
& "C:\Users\luisi\Documents\Programming\Python\.misc314\Scripts\python.exe" -m unittest discover -s tests
```

Expected current baseline:

```text
Ran 260 tests
OK
```

For a normal development environment:

```powershell
python -m unittest discover -s tests
```

## Dataset Tests

Dataset tests cover bundled payload metadata, size and SHA-256 checks,
release-backed KDD2010 download behavior, cached reload behavior, and checksum
rejection.

```powershell
python -m unittest tests.test_datasets
```

Expected current baseline:

```text
Ran 9 tests
OK
```

## Hawkes Equivalence Report

The source-backed Hawkes behavior ledger is checked with:

```powershell
python tests\hawkes_equivalence\report_equivalence.py
```

Expected current output:

```text
Hawkes reference tests inventoried: 171
pass: 171
xfail_equivalence_gap: 0
skip_optional_backend: 0
out_of_scope_non_hawkes: 0
```

## Gallery Notebook

The restored gallery notebook is:

```text
examples\gallery_reproduction.ipynb
```

It should execute 19 code cells and produce 25 gallery records. Use an
in-memory notebook execution check so the notebook file is not rewritten during
routine validation.

## Import Checks

From a checkout, confirm the import resolves to local source:

```powershell
python -c "import hawkes_tools; print(hawkes_tools.__file__)"
```

Confirm the dataset release metadata:

```powershell
python -c "from hawkes_tools.datasets import external_dataset_metadata; print(external_dataset_metadata('binary/kdd2010/kdd2010.trn.bz2')['url'])"
```

Expected URL:

```text
https://github.com/luisivanhr/hawkes-tools/releases/download/dataset/kdd2010.trn.bz2
```

## Cache Cleanup

Validation runs may create Python bytecode and Numba cache files under
`__pycache__`. Do not commit generated cache files. After validation, check:

```powershell
git status --short
```

Generated cache files should be restored or removed before committing.
