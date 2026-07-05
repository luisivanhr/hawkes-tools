# Prompt: Finish tick Port, Vendor Datasets, and Rename to hawkes-tools

You are a coding agent working in:

`C:\Users\luisi\Documents\Programming\Python\Hawkes\tick`

The active standalone package currently lives inside this checkout at:

`C:\Users\luisi\Documents\Programming\Python\Hawkes\tick\our-hawkes`

The user wants this package finished as a production-ready standalone library named **hawkes-tools**. Treat the current `our-hawkes` name and `our_hawkes` import package as temporary migration names.

## Current State

The gallery reproduction work is complete: all examples from `https://x-datainitiative.github.io/tick/auto_examples/index.html` run in `our-hawkes/examples/tick_gallery_reproduction.ipynb`.

Recent validation evidence:

- Full gallery notebook execution: 25 recorded examples, OK, about 79 seconds.
- Full `our-hawkes` unittest suite: 160 tests OK, 1 skipped.
- The non-constant baseline Hawkes fit now runs.
- The asynchronous stochastic solver example now runs at original scale; the previous bottleneck was sparse matrix construction, not simulation, JIT, or estimation.

Important current files:

- `our-hawkes/PARITY.md`
- `our-hawkes/README.md`
- `our-hawkes/pyproject.toml`
- `our-hawkes/examples/tick_gallery_reproduction.ipynb`
- `our-hawkes/src/our_hawkes/`
- `our-hawkes/tests/`

Some docs may still contain stale baseline counts or wording from before the latest fixes. Recheck the current tree before editing.

Use this interpreter on this Windows host unless the environment has been deliberately changed:

```powershell
& "C:\Users\luisi\Documents\Programming\Python\.misc314\Scripts\python.exe" -m unittest discover -s our-hawkes\tests
```

## Hard Requirements

1. Do not break the currently working gallery notebook or existing tests.
2. Keep the implementation independent of tick's compiled C++ extensions.
3. Use tick's original Python/C++ source and tests as the mathematical/API reference before inventing alternatives.
4. Prefer faithful pure-Python, NumPy, SciPy, and Numba implementations. Independent approximations are a last resort and must be documented as gaps.
5. Vendor or otherwise support all public tick datasets, not only the gallery subset.
6. Rename the production package consistently to **hawkes-tools**.
7. Prepare the package so it can later live as a standalone folder/repository, not as a subfolder/submodule inside the original tick checkout.

## Rename Target

Python distribution names may contain hyphens, but import packages cannot. Use:

- Distribution/project name: `hawkes-tools`
- Source folder/import package: `hawkes_tools`
- Repository/folder name when split out: `hawkes-tools`

Migration expectation:

- Rename `our-hawkes/` to `hawkes-tools/` when safe.
- Rename `src/our_hawkes/` to `src/hawkes_tools/`.
- Update imports in code, examples, notebooks, tests, docs, package metadata, and any dataset package-data configuration.
- Prefer the new import:

```python
from hawkes_tools.hawkes import HawkesExpKern, SimuHawkesExpKernels
```

Temporary compatibility is acceptable while migrating:

```python
# src/our_hawkes/__init__.py
# Deprecated compatibility shim importing from hawkes_tools.
```

But do not leave `our_hawkes` as the primary public API in production docs, examples, or tests. If a shim remains, add explicit deprecation tests and document its planned removal.

## Initial Onboarding Steps

Start by reading these files:

1. `our-hawkes/PARITY.md`
2. `our-hawkes/README.md`
3. `our-hawkes/pyproject.toml`
4. `our-hawkes/tests/tick_equivalence/README.md`
5. `our-hawkes/examples/tick_gallery_reproduction.ipynb`
6. The local tick source directories that correspond to the next module you port.

Before changing code, run an inventory:

```powershell
rg --files tick
rg --files our-hawkes\src\our_hawkes
rg -n "our_hawkes|our-hawkes|our hawkes|our-hawkes|our_hawkes" our-hawkes
```

Then generate a gap ledger comparing local tick package areas to the current standalone package.

## Remaining Porting Scope

The current package has strong Hawkes-gallery coverage but not full tick package parity. Finish the remaining module families deliberately.

Port or explicitly close gaps for:

- `tick.array` and `tick.array_test`
- `tick.base`
- `tick.base_model`
- `tick.dataset`
- `tick.linear_model`
- `tick.metrics`
- `tick.plot`
- `tick.preprocessing`
- `tick.prox`
- `tick.random`
- `tick.robust`
- `tick.simulation`
- `tick.solver`
- `tick.survival`
- deep import-path compatibility for Hawkes modules where useful

For each area:

- Inventory public exports, constructor signatures, methods, properties, and tests.
- Decide whether the package should expose the full tick-compatible name, a renamed production API, or both.
- Implement the missing math/API surface.
- Add focused tests derived from tick's tests where possible.
- Update `PARITY.md` with concrete status and remaining gaps.

Do not claim full parity unless tests support it.

## Dataset Requirement

The current vendored dataset subset under `our_hawkes.datasets.vendor.tick-datasets` includes:

- `binary/adult`
- `binary/covtype`
- `binary/ijcnn1`
- `binary/kdd2010`
- `binary/reuters`
- `hawkes/bund`
- `regression/abalone`

The user wants all datasets because they may be useful later. Finish this by:

1. Inventorying the full public `X-DataInitiative/tick-datasets` repository.
2. Mirroring every public payload path into the standalone package or a clearly managed local dataset artifact.
3. Updating the dataset manifest constant currently named `TICK_DATASET_FILES`.
4. Updating package-data rules in `pyproject.toml`.
5. Adding tests that check every manifest entry is either vendored or intentionally external.
6. Keeping `fetch_tick_dataset`, `load_dataset`, `fetch_hawkes_bund_data`, `fetch_url_dataset`, and URL dataset helpers working without importing `tick`.

If a dataset is too large to commit/package safely, do not silently skip it. Implement a manifest entry with URL, expected path, size/hash if available, cache behavior, and a test that marks it as external rather than missing.

The URL reputation tarball is not part of `tick-datasets`; keep it as an explicit external helper unless the user decides to vendor it.

## Standalone Packaging Work

The package should not depend on being nested inside the tick checkout.

Required checks:

- Install from a clean directory with editable mode.
- Import `hawkes_tools` without adding the original tick repo to `PYTHONPATH`.
- Confirm no runtime imports from `tick`.
- Confirm vendored datasets are included in package data.
- Confirm examples and notebooks resolve local package imports in both editable and source-tree modes.

Suggested validation commands:

```powershell
cd C:\Users\luisi\Documents\Programming\Python\Hawkes\tick\hawkes-tools
& "C:\Users\luisi\Documents\Programming\Python\.misc314\Scripts\python.exe" -m pip install -e ".[dev]"
& "C:\Users\luisi\Documents\Programming\Python\.misc314\Scripts\python.exe" -m unittest discover -s tests
& "C:\Users\luisi\Documents\Programming\Python\.misc314\Scripts\python.exe" -m unittest discover -s tests\tick_equivalence
```

Also run a clean import from outside the repository:

```powershell
cd $env:TEMP
& "C:\Users\luisi\Documents\Programming\Python\.misc314\Scripts\python.exe" -c "import hawkes_tools; import hawkes_tools.hawkes; print(hawkes_tools.__file__)"
```

## Validation Expectations

Keep these checks green after each substantial module family:

```powershell
& "C:\Users\luisi\Documents\Programming\Python\.misc314\Scripts\python.exe" -m unittest discover -s our-hawkes\tests
& "C:\Users\luisi\Documents\Programming\Python\.misc314\Scripts\python.exe" -m unittest discover -s our-hawkes\tests\tick_equivalence
```

After the rename, update command paths from `our-hawkes` to `hawkes-tools`.

For notebook validation, execute `examples/tick_gallery_reproduction.ipynb` or use an equivalent programmatic runner that executes every code cell and confirms 25 records. Do not reduce the notebook to a smoke demo.

## Implementation Guidance

- Preserve current working behavior first; make small, testable changes.
- Use `rg` for all searches.
- Use `apply_patch` for manual file edits.
- Avoid broad refactors unless required by the rename or module extraction.
- Keep generated caches (`__pycache__`, `.nbc`, `.nbi`) out of the durable package state.
- Do not rely on the original `tick` package at runtime. It can be used only as a reference source and test oracle when safe.
- For high-cost modules, start with the exact public API and tests, then optimize hot paths only after correctness is established.
- Use Numba for hot loops when pure NumPy/SciPy is not fast enough.
- For stochastic solvers, keep the math faithful to tick. If true C++-style atomic async writes are not implemented, document that as a performance-backend gap rather than pretending it is solved.

## Completion Definition

This work is complete when:

1. The package is consistently named `hawkes-tools` / `hawkes_tools`.
2. It installs and imports from a clean directory without the original tick checkout.
3. All current gallery examples still run.
4. All relevant tests pass under the renamed package.
5. Every public tick module family has either been ported or has an explicit tested/documented gap.
6. All public tick-datasets payloads are vendored or explicitly represented as managed external datasets.
7. `README.md`, `PARITY.md`, `pyproject.toml`, examples, notebooks, and tests all reflect the new package name and current validation state.
