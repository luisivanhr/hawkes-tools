"""Public pure-Python optimization solvers.

The solvers expose the familiar tick-style ``set_model(...).set_prox(...)`` and
``solve`` flow for models with ``loss`` and ``grad`` methods.  They are compact
reference implementations intended for Hawkes sanity checks and small custom
models, not replacements for tick's compiled high-throughput solvers.
"""

from hawkes_tools.optim import (
    AGD,
    BFGS,
    GD,
    GFB,
    SAGA,
    SCPG,
    SDCA,
    SGD,
    SVRG,
    AdaGrad,
    History,
    Solver,
)

__all__ = [
    "Solver",
    "GD",
    "AGD",
    "BFGS",
    "GFB",
    "SCPG",
    "SGD",
    "AdaGrad",
    "SVRG",
    "SAGA",
    "SDCA",
    "History",
]

