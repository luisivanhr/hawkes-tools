# hawkes-tools Documentation

`hawkes-tools` is a standalone Python distribution for Hawkes processes,
point-process simulation, GLM utilities, first-order solvers, survival models,
robust models, preprocessing helpers, plotting, and dataset loading.

The distribution name is `hawkes-tools`; the import package is
`hawkes_tools`.

This project is a production library with its own public API. It does not
install a compatibility namespace for the previous compiled project, and it
does not depend on that project at runtime.

## Documentation Map

| Document | Purpose |
| --- | --- |
| [Installation](installation.md) | Install from GitHub, install from a local checkout, and prepare optional extras. |
| [Datasets](datasets.md) | Bundled datasets, release-backed KDD2010 training data, cache locations, checksums, and fetch helpers. |
| [API Overview](api-overview.md) | Main modules, exported model families, and common usage patterns. |
| [Validation](validation.md) | Test commands, equivalence report, gallery notebook check, and expected baselines. |
| [Maintenance](maintenance.md) | Release asset updates, documentation rules, cache cleanup, and repository hygiene. |

## Quick Example

```python
import numpy as np

from hawkes_tools.hawkes import HawkesExpKern, SimuHawkesExpKernels

baseline = np.array([0.1, 0.2])
adjacency = np.array([[0.2, 0.1], [0.0, 0.3]])

sim = SimuHawkesExpKernels(
    adjacency=adjacency,
    decays=1.5,
    baseline=baseline,
    end_time=100.0,
    seed=123,
    verbose=False,
)
sim.simulate()

learner = HawkesExpKern(decays=1.5, penalty="none", max_iter=50, verbose=False)
learner.fit(sim.timestamps)

print(learner.baseline)
print(learner.adjacency)
```

## Runtime Contract

The package requires NumPy, SciPy, Matplotlib, and Numba. Numba is not an
optional speedup here: several hot loops are dispatched through JIT-backed
helpers, so the first call to those paths includes compile latency. Benchmark
cold compile time, warm runtime, and reference formula timing separately.

## Dataset Contract

Small and medium public datasets are included as package data. The large
KDD2010 training payload is intentionally stored as a GitHub release asset and
downloaded into a local cache on demand. See [Datasets](datasets.md) for exact
paths, URLs, checksums, and examples.
