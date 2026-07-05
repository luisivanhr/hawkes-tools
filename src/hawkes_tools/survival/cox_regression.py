"""Cox regression learner using the standalone optimization stack."""

from __future__ import annotations

from warnings import warn

import numpy as np

from hawkes_tools.optim import (
    AGD,
    GD,
    ProxBinarsity,
    ProxElasticNet,
    ProxL1,
    ProxL2Sq,
    ProxTV,
    ProxZero,
)
from hawkes_tools.preprocessing.utils import safe_array

from .model_coxreg_partial_lik import ModelCoxRegPartialLik

__all__ = ["CoxRegression"]


def _optional_positive_finite(name, value):
    if value is None:
        return None
    numeric = float(value)
    if not np.isfinite(numeric) or numeric <= 0.0:
        raise ValueError(f"``{name}`` must be positive")
    return numeric


def _nonnegative_finite(name, value):
    numeric = float(value)
    if not np.isfinite(numeric) or numeric < 0.0:
        raise ValueError(f"``{name}`` must be non-negative")
    return numeric


def _integer_at_least(name, value, minimum):
    if isinstance(value, bool):
        raise ValueError(f"``{name}`` must be an integer >= {minimum}")
    numeric = int(value)
    if numeric != value or numeric < minimum:
        raise ValueError(f"``{name}`` must be an integer >= {minimum}")
    return numeric


class CoxRegression:
    """Cox regression learner for proportional hazards."""

    _solver_classes = {"gd": GD, "agd": AGD}
    _penalties = {"none", "l1", "l2", "elasticnet", "tv", "binarsity"}

    def __init__(
        self,
        penalty: str = "l2",
        C: float = 1e3,
        solver: str = "agd",
        step: float | None = None,
        tol: float = 1e-5,
        max_iter: int = 100,
        verbose: bool = False,
        warm_start: bool = False,
        print_every: int = 10,
        record_every: int = 10,
        elastic_net_ratio: float = 0.95,
        random_state=None,
        blocks_start=None,
        blocks_length=None,
    ):
        self.penalty = str(penalty).lower()
        if self.penalty not in self._penalties:
            allowed = ", ".join(sorted(self._penalties))
            raise ValueError(f"``penalty`` must be one of {allowed}, got {penalty}")
        self.solver = str(solver).lower()
        if self.solver not in self._solver_classes:
            allowed = ", ".join(sorted(self._solver_classes))
            raise ValueError(f"``solver`` must be one of {allowed}, got {solver}")
        self.step = _optional_positive_finite("step", step)
        self.tol = _nonnegative_finite("tol", tol)
        self.max_iter = _integer_at_least("max_iter", max_iter, 0)
        self.verbose = bool(verbose)
        self.warm_start = bool(warm_start)
        self.print_every = _integer_at_least("print_every", print_every, 1)
        self.record_every = _integer_at_least("record_every", record_every, 1)
        self.random_state = random_state
        self.blocks_start = blocks_start
        self.blocks_length = blocks_length
        self.coeffs = None
        self._fitted = False
        self._model_obj = ModelCoxRegPartialLik()
        self.elastic_net_ratio = float(elastic_net_ratio)
        self.C = C
        self._solver_obj = self._construct_solver()
        self._prox_obj = self._construct_prox()

    def _construct_solver(self):
        solver_class = self._solver_classes[self.solver]
        return solver_class(
            step=self.step,
            tol=self.tol,
            max_iter=self.max_iter,
            verbose=self.verbose,
            print_every=self.print_every,
            record_every=self.record_every,
            random_state=self.random_state,
        )

    def _construct_prox(self):
        if self.penalty == "none":
            if self.C != 1e3:
                warn('You cannot set C for penalty "none"', RuntimeWarning, stacklevel=2)
            return ProxZero()
        strength = 1.0 / self.C
        if self.penalty == "l1":
            return ProxL1(strength)
        if self.penalty == "l2":
            return ProxL2Sq(strength)
        if self.penalty == "elasticnet":
            return ProxElasticNet(strength, self.elastic_net_ratio)
        if self.penalty == "tv":
            return ProxTV(strength)
        if self.penalty == "binarsity":
            if self.blocks_start is None or self.blocks_length is None:
                raise ValueError("binarsity penalty requires blocks_start and blocks_length")
            return ProxBinarsity(strength, self.blocks_start, self.blocks_length)
        raise AssertionError("unreachable")

    @property
    def C(self):
        return self._C

    @C.setter
    def C(self, value):
        value = float(value)
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"``C`` must be positive, got {value:g}")
        self._C = value
        if hasattr(self, "_prox_obj") and self.penalty != "none":
            self._prox_obj.strength = 1.0 / value

    @property
    def elastic_net_ratio(self):
        return self._elastic_net_ratio

    @elastic_net_ratio.setter
    def elastic_net_ratio(self, value):
        value = float(value)
        if not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("``elastic_net_ratio`` must be in [0, 1]")
        self._elastic_net_ratio = value
        if hasattr(self, "_prox_obj") and isinstance(self._prox_obj, ProxElasticNet):
            self._prox_obj.ratio = value

    def _all_safe(self, features, times, censoring):
        if not set(np.unique(censoring)).issubset({0, 1}):
            raise ValueError("``censoring`` must only have values in {0, 1}")
        if not np.all(np.asarray(times) >= 0):
            raise ValueError("``times`` array must contain only non-negative entries")
        features = safe_array(features)
        times = safe_array(np.asarray(times))
        censoring = safe_array(np.asarray(censoring), np.ushort)
        return features, times, censoring

    def fit(self, features, times, censoring):
        features, times, censoring = self._all_safe(features, times, censoring)
        model = self._model_obj
        model.fit(features, times, censoring)
        self._prox_obj.range = (0, model.n_coeffs)
        self._solver_obj.set_model(model).set_prox(self._prox_obj)

        coeffs_start = None
        if self.warm_start and self.coeffs is not None and self.coeffs.shape == (model.n_coeffs,):
            coeffs_start = self.coeffs
        self.coeffs = self._solver_obj.solve(coeffs_start)
        self._fitted = True
        return self

    def score(self, features=None, times=None, censoring=None):
        if not self._fitted:
            raise RuntimeError("You must fit the model first")
        if features is None and times is None and censoring is None:
            return self._model_obj.loss(self.coeffs)
        if features is None:
            raise ValueError("Passed ``features`` is None")
        if times is None:
            raise ValueError("Passed ``times`` is None")
        if censoring is None:
            raise ValueError("Passed ``censoring`` is None")
        features, times, censoring = self._all_safe(features, times, censoring)
        return ModelCoxRegPartialLik().fit(features, times, censoring).loss(self.coeffs)
