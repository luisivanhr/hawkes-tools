# hawkes-tools

Standalone Python Hawkes and point-process tools for theoretical and practical research.
Several tools provided here are excelent for data science applications.

The package lives under `hawkes-tools/`, while the import package is
`hawkes_tools`. The old migration namespace `our_hawkes` is not packaged.

```python
from hawkes_tools.hawkes import SimuHawkesExpKernels, HawkesExpKern
from hawkes_tools.linear_model import LogisticRegression, SimuLogReg
```

See [PARITY.md](PARITY.md) for the current API support matrix.

The implementation uses NumPy/SciPy for numerical work, mandatory Numba JIT
helpers for hot loops, and Python parallelism for repeated simulations.

## Datasets

`hawkes_tools.datasets` includes standalone loaders and metadata for the
Hawkes Bund, Adult, Covtype, IJCNN, KDD, Reuters, and Abalone payloads. It
loads smaller payloads from package data or an explicit local `data_home`
without depending on any external Hawkes package at runtime. The large KDD2010
training payload is downloaded from the `dataset` GitHub release on first use
to keep the repository and wheel small. The URL reputation tarball is
represented separately as a managed external dataset because it is not part of
the bundled package data. Set `HAWKES_TOOLS_DATASETS` or pass `data_home=` to
control the local cache for managed external datasets.

```python
from hawkes_tools.datasets import fetch_hawkes_bund_data

timestamps = fetch_hawkes_bund_data()
```

Use the dataset metadata helpers to inspect source, format, shape, file-size,
and checksum metadata programmatically.

Dataset payload shapes:

| Dataset path | Loader output shape |
| --- | --- |
| `hawkes/bund/bund.npz` | 20 realizations keyed by trading date; each realization is an object array of 4 timestamp streams. Total events per day range from 28,361 to 56,086; per-stream event counts range from 3,371 to 23,381. |
| `binary/adult/adult.trn.bz2` | Binary SVMlight: `X` CSR shape `(32_561, 123)`, `y` shape `(32_561,)`, 451,592 nonzeros. |
| `binary/adult/adult.tst.bz2` | Binary SVMlight: `X` CSR shape `(16_281, 123)`, `y` shape `(16_281,)`, 225,732 nonzeros. |
| `binary/covtype/covtype.trn.bz2` | Binary SVMlight: `X` CSR shape `(581_012, 54)`, `y` shape `(581_012,)`, 6,940,438 nonzeros. |
| `binary/ijcnn1/ijcnn1.trn.bz2` | Binary SVMlight: `X` CSR shape `(49_990, 22)`, `y` shape `(49_990,)`, 649,870 nonzeros. |
| `binary/ijcnn1/ijcnn1.tst.bz2` | Binary SVMlight: `X` CSR shape `(91_701, 22)`, `y` shape `(91_701,)`, 1,192,113 nonzeros. |
| `binary/kdd2010/kdd2010.trn.bz2` | Release-backed binary SVMlight payload: `X` CSR shape `(19_264_097, 1_129_522)` when loaded standalone, `y` shape `(19_264_097,)`, 173,376,873 nonzeros. `fetch_dataset` downloads and verifies it into `HAWKES_TOOLS_DATASETS` or `data_home=` on cache miss. |
| `binary/kdd2010/kdd2010.tst.bz2` | Binary SVMlight: `X` CSR shape `(748_401, 1_163_024)` when loaded standalone, `y` shape `(748_401,)`, 6,735,609 nonzeros. |
| `binary/reuters/reuters.trn.bz2` | Binary SVMlight: `X` CSR shape `(7_770, 8_315)`, `y` shape `(7_770,)`, 339,837 nonzeros. |
| `binary/reuters/reuters.tst.bz2` | Binary SVMlight: `X` CSR shape `(3_299, 8_315)`, `y` shape `(3_299,)`, 136,821 nonzeros. |
| `regression/abalone/abalone.trn.bz2` | Regression SVMlight: `X` CSR shape `(4_177, 8)`, `y` shape `(4_177,)`, 32,080 nonzeros. |

Managed external dataset:

| Dataset path | Loader output shape |
| --- | --- |
| `binary/kdd2010/kdd2010.trn.bz2` | Release-backed KDD2010 training payload. `fetch_dataset(..., data_home=...)` downloads it from `https://github.com/luisivanhr/hawkes-tools/releases/download/dataset/kdd2010.trn.bz2` on cache miss and verifies size plus SHA-256 before loading. |
| `url/url_svmlight.tar.gz` | URL reputation tarball from UCI. `fetch_url_dataset(n_days=k)` loads Day0 through Day`k-1` for `1 <= k <= 120` as `X` CSR shape `(sum selected day rows, 3_231_961)` and `y` shape `(same row count,)`. |

## NumPy/Numba Runtime

Numba is a required runtime dependency. The package fails fast if Numba is not
installed, because JIT dispatch is part of the standalone performance contract.

```powershell
& "C:\Users\luisi\Documents\Programming\Python\.misc314\Scripts\python.exe" -m pip install -e .
```

The first call to a JIT-backed helper includes compile latency. Benchmark cold,
warm, and reference timings separately:

```powershell
& "C:\Users\luisi\Documents\Programming\Python\.misc314\Scripts\python.exe" benchmarks\benchmark_numba_hot_paths.py
& "C:\Users\luisi\Documents\Programming\Python\.misc314\Scripts\python.exe" -m unittest discover -s tests -p test_benchmark_smoke.py
```

Numba cache files (`.nbc` / `.nbi`) may be written near `__pycache__`.

## Current API status

`hawkes_tools.hawkes` exports the Hawkes-focused public surface:

- Kernels: `HawkesKernel0`, `HawkesKernelExp`, `HawkesKernelSumExp`,
  `HawkesKernelPowerLaw`, and `HawkesKernelTimeFunc`.
- Simulation: `SimuPoissonProcess`, `SimuInhomogeneousPoisson`, `SimuHawkes`,
  `SimuHawkesExpKernels`, `SimuHawkesSumExpKernels`, and `SimuHawkesMulti`.
- Parametric models and learners: exponential and sum-exponential log-likelihood
  and least-squares models, `HawkesExpKern`, `HawkesSumExpKern`, and `HawkesADM4`.
- Non-parametric and cumulant learners: `HawkesEM`, `HawkesBasisKernels`,
  `HawkesSumGaussians`, `HawkesConditionalLaw`, `HawkesCumulantMatching`,
  `RKHSHawkes` with RKHS basis recovery and plotting, PyTorch-backed
  `HawkesCumulantMatchingPyT`, and the legacy `HawkesCumulantMatchingTf` class
  name backed by the same PyTorch optimizer.
- GLM utilities: `hawkes_tools.linear_model` exports JIT-backed linear,
  logistic, Poisson, and hinge-family model classes, plus learners and
  simulators for the linear/logistic/Poisson families. Model and learner
  methods validate fit state, feature shapes, coefficient lengths, finite
  numeric inputs, supported solver settings, and learner penalties
  `none`, `l1`, `l2`, `elasticnet`, `tv`, and `binarsity`. `binarsity`
  requires explicit `blocks_start` and `blocks_length` block metadata.
- Robust utilities: `hawkes_tools.robust` exports the robust linear-regression
  learner, sample-intercept linear model, robust scale estimators, and the
  standalone first-order loss models `ModelHuber`, `ModelModifiedHuber`,
  `ModelEpsilonInsensitive`, and `ModelAbsoluteRegression`.
- Base model protocols are available from `hawkes_tools.base_model` for model
  counters, fit guards, feature/label storage, and Lipschitz or second-order
  abstract interfaces.
- Base utilities are available from `hawkes_tools.base`, including
  `Base`, `BaseEstimator`, `TimeFunction`, `History`, `actual_kwargs`, and a
  small standalone `ThreadPool`.
- Generic simulation helpers are available from `hawkes_tools.simulation`:
  `features_normal_cov_uniform`, `features_normal_cov_toeplitz`,
  `weights_sparse_exp`, and `weights_sparse_gauss`.
- Array and random helpers are available from `hawkes_tools.array` and
  `hawkes_tools.random`; array serialization uses a standalone NumPy container,
  and random helpers provide hawkes-tools reproducibility without claiming
  compiled RNG bitstream parity.
- Optimizer compatibility: `hawkes_tools.prox` and `hawkes_tools.solver` expose
  pure-Python proximal operators and solver flows for compatible Hawkes or GLM
  models, with explicit validation for solver options, starting vectors, and
  malformed model/prox inputs.
- Preprocessing helpers are available from `hawkes_tools.preprocessing`,
  including `FeaturesBinarizer` and the longitudinal product, lagger, and
  sample-filter transforms.
- Survival helpers include `CoxRegression`, `ModelCoxRegPartialLik`,
  `ModelSCCS`, `SimuCoxReg`, `SimuCoxRegWithCutPoints`, `SimuSCCS`,
  `ConvSCCS`, `BatchConvSCCS`, `StreamConvSCCS`, `kaplan_meier`, and
  `nelson_aalen`. Cox and SCCS code validates survival times, censoring
  indicators, SCCS lag vectors, learner/simulator constructor settings,
  fitted-data scoring paths, and batch/thread wrapper settings. Batch/stream
  SCCS wrappers use the same standalone sequential backend as `ConvSCCS`.
  `ConvSCCS` includes TV and group-lasso penalties in its optimization
  objective when `C_tv` or `C_group_l1` are set.
- Hawkes plotting helpers are available from `hawkes_tools.plot` and
  `hawkes_tools.hawkes.plot`.
- Generic stem plots are available as `hawkes_tools.plot.stems`.
- Vendored and release-backed dataset loaders are available from
  `hawkes_tools.datasets`; use `download_dataset` to prepopulate a managed
  external payload cache without loading it.

This is a standalone production API. Intentional exclusions include exact
compiled attribute semantics and compiled backend internals. TensorFlow is not
a supported cumulant backend; use the PyTorch-backed cumulant classes. See
[PARITY.md](PARITY.md) for the more detailed test-backed status and documented
exclusions.

## Examples

The scripts in `examples/` prepend the local `src/` directory to `sys.path`, so
they run against this checkout without installing `hawkes-tools` and without
depending on another Hawkes package.

```powershell
& "C:\Users\luisi\Documents\Programming\Python\.misc314\Scripts\python.exe" examples\plot_hawkes_simulation.py
& "C:\Users\luisi\Documents\Programming\Python\.misc314\Scripts\python.exe" examples\plot_hawkes_em.py
```

All example scripts are intended to run headlessly with Matplotlib's `Agg`
backend.

The notebook `examples/hawkes_time_rescaling_gof.ipynb` demonstrates
time-rescaling goodness-of-fit diagnostics for univariate Hawkes processes with
exponential, sum-exponential, power-law, and time-function kernels.

The restored full gallery sanity notebook covers 25 example records with
standalone replacements, including vendored finance/GLM datasets and the
original-scale Hawkes EM example. The mixed exponential/time-function
simulation path uses exact recursive exponential updates plus JIT-backed
finite-support time-function windows, so old events outside finite kernel
support are skipped instead of scanned.

## Development

Use the requested environment:

```powershell
& "C:\Users\luisi\Documents\Programming\Python\.misc314\Scripts\python.exe" -m unittest discover -s tests
```

The current stabilization baseline is:

- `unittest discover -s tests`: 260 tests run, OK, with no skips.
- Equivalence ledger tests: 171 source-backed Hawkes cases from the frozen
  manifest; current classification is 171 pass, 0 unresolved equivalence gaps,
  and 0 optional backend cases.
- Equivalence status: 171 pass, 0 xfail, 0 optional skips.
- The ledger report script prints the counts by `pass`,
  `xfail_equivalence_gap`, and `skip_optional_backend`.
- Full gallery notebook execution: 25 recorded examples, OK.
- All scripts in `examples/`: smoke-tested successfully from this checkout.
- Clean-directory import: verified with `PYTHONPATH` set to the absolute
  `hawkes-tools/src` directory.
- Local isolated install: `pip install . --no-deps --no-build-isolation
  --target %TEMP%\hawkes-tools-install-check-20260706-standalone --upgrade`
  builds `hawkes_tools-0.1.0-py3-none-any.whl`; importing from that temp target
  loads `hawkes_tools`, excludes `our_hawkes`, and exposes 11 known dataset
  entries, with 10 payloads from installed package data and the KDD2010
  training payload managed through the release-backed external cache.
- Public export check: every name listed in `hawkes_tools.hawkes.__all__` imports
  from the local source tree.

Install editable dependencies when needed:

```powershell
& "C:\Users\luisi\Documents\Programming\Python\.misc314\Scripts\python.exe" -m pip install -e ".[dev]"
```
