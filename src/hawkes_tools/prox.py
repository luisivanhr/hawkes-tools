"""Public pure-Python proximal operators.

These classes mirror the tick prox names that are useful with Hawkes learners
and custom Hawkes models.  They are implemented in Python/NumPy and share the
same code used internally by :mod:`hawkes_tools.hawkes`.
"""

from hawkes_tools.optim import (
    Prox,
    ProxBinarsity,
    ProxElasticNet,
    ProxEquality,
    ProxGroupL1,
    ProxL1,
    ProxL1w,
    ProxL2,
    ProxL2Sq,
    ProxMulti,
    ProxNuclear,
    ProxPositive,
    ProxSlope,
    ProxTV,
    ProxZero,
)

__all__ = [
    "Prox",
    "ProxZero",
    "ProxPositive",
    "ProxL1",
    "ProxL1w",
    "ProxL2Sq",
    "ProxL2",
    "ProxTV",
    "ProxNuclear",
    "ProxSlope",
    "ProxElasticNet",
    "ProxMulti",
    "ProxEquality",
    "ProxBinarsity",
    "ProxGroupL1",
]

