# API Overview

This page summarizes the stable public areas. See `PARITY.md` for the detailed
test-backed support ledger.

## Hawkes Processes

Import from `hawkes_tools.hawkes`:

```python
from hawkes_tools.hawkes import (
    HawkesADM4,
    HawkesBasisKernels,
    HawkesConditionalLaw,
    HawkesCumulantMatching,
    HawkesCumulantMatchingPyT,
    HawkesEM,
    HawkesExpKern,
    HawkesKernelExp,
    HawkesKernelPowerLaw,
    HawkesKernelSumExp,
    HawkesKernelTimeFunc,
    HawkesSumExpKern,
    HawkesSumGaussians,
    SimuHawkes,
    SimuHawkesExpKernels,
    SimuHawkesMulti,
    SimuHawkesSumExpKernels,
)
```

Core coverage includes kernels, simulation, parametric learners,
non-parametric learners, cumulant matching, conditional-law estimation, model
losses, gradients, Hessian helpers, and plotting helpers.

## Linear Models and GLM Utilities

`hawkes_tools.linear_model` provides linear, logistic, Poisson, and
hinge-family models plus learners and simulators. Dense loss and gradient
paths use Numba-backed loops; sparse paths use SciPy CSR algebra where
applicable.

Common penalty names are:

- `none`
- `l1`
- `l2`
- `elasticnet`
- `tv`
- `binarsity`

`binarsity` requires explicit block metadata.

## Solvers and Prox Operators

Use `hawkes_tools.solver` for first-order and stochastic solvers:

- `GD`
- `AGD`
- `BFGS`
- `GFB`
- `SCPG`
- `SGD`
- `AdaGrad`
- `SVRG`
- `SAGA`
- `SDCA`

Use `hawkes_tools.prox` for proximal operators:

- `ProxZero`
- `ProxPositive`
- `ProxL1`
- `ProxL1w`
- `ProxL2`
- `ProxL2Sq`
- `ProxTV`
- `ProxNuclear`
- `ProxSlope`
- `ProxElasticNet`
- `ProxMulti`
- `ProxEquality`
- `ProxBinarsity`
- `ProxGroupL1`

The typical flow is:

```python
solver.set_model(model).set_prox(prox).solve(start)
```

Malformed solver settings, invalid starts, and unsupported model/prox
combinations fail with explicit validation errors.

## Survival, Robust, Preprocessing, Metrics

`hawkes_tools.survival` includes Cox and SCCS models, simulators, convolutional
SCCS wrappers, Kaplan-Meier, and Nelson-Aalen helpers.

`hawkes_tools.robust` includes robust linear regression, robust scale
estimators, and first-order robust loss models.

`hawkes_tools.preprocessing` includes feature binarization and longitudinal
feature transforms.

`hawkes_tools.metrics` includes support recovery metrics.

## Base Utilities

`hawkes_tools.base` provides shared utilities such as:

- `Base`
- `BaseEstimator`
- `TimeFunction`
- `History`
- `actual_kwargs`
- `ThreadPool`

`hawkes_tools.base_model` provides model protocol classes and fit-state guards.

## Plotting

`hawkes_tools.plot` and `hawkes_tools.hawkes.plot` expose Matplotlib-based
helpers for kernels, baselines, norms, point processes, QQ diagnostics, time
functions, optimization history, estimated intensity, and generic stem plots.

Optional Bokeh support for generic stems is imported only when available.

## Datasets

Use `hawkes_tools.datasets` for packaged datasets, release-backed KDD2010
training data, and URL reputation helpers. See [Datasets](datasets.md).
