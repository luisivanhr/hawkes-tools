"""Robust linear regression with sample intercepts.

This module provides the robust linear-regression surface used by the gallery.
The model is a linear least-squares objective augmented with one intercept per
sample, with SLOPE regularization on those sample intercepts.
"""

from __future__ import annotations

from dataclasses import dataclass
from warnings import warn

import numpy as np
from scipy import sparse
from scipy.linalg import toeplitz
from scipy.special import erfinv
from scipy.stats import iqr, norm

from hawkes_tools.base_model import ModelFirstOrder, ModelGeneralizedLinear, ModelLipschitz
from hawkes_tools.linear_model import _as_1d_float, _as_2d_float, _design
from hawkes_tools.optim import AGD, GD, ProxElasticNet, ProxL1, ProxL2Sq, ProxMulti, ProxSlope, ProxZero

__all__ = [
    "ModelAbsoluteRegression",
    "ModelEpsilonInsensitive",
    "ModelHuber",
    "ModelLinRegWithIntercepts",
    "ModelModifiedHuber",
    "RobustLinearRegression",
    "features_normal_cov_toeplitz",
    "std_iqr",
    "std_mad",
]


def std_mad(x) -> float:
    """Corrected median absolute deviation."""

    x = np.asarray(x, dtype=float)
    correction = 1.0 / norm.ppf(3.0 / 4.0)
    return float(correction * np.median(np.abs(x - np.median(x))))


def std_iqr(x) -> float:
    """Corrected inter-quartile distance."""

    return float((2.0**0.5) * erfinv(0.5) * iqr(np.asarray(x, dtype=float)))


def features_normal_cov_toeplitz(
    n_samples: int = 200,
    n_features: int = 30,
    cov_corr: float = 0.5,
    dtype="float64",
) -> np.ndarray:
    """Generate Gaussian features with Toeplitz covariance."""

    covariance = toeplitz(float(cov_corr) ** np.arange(0, int(n_features)))
    features = np.random.multivariate_normal(np.zeros(int(n_features)), covariance, size=int(n_samples))
    if dtype != "float64":
        return features.astype(dtype)
    return features


class _RobustGLMMixin:
    """Shared pure-Python generalized-linear robust loss machinery."""

    def __init__(self, fit_intercept: bool = True, n_threads: int = 1):
        super().__init__(fit_intercept=fit_intercept)
        self.n_threads = int(n_threads)

    def _set_data(self, features, labels):
        features = _as_2d_float(features)
        labels = _as_1d_float(labels)
        if features.shape[0] != labels.size:
            raise ValueError("features and labels have inconsistent sample counts")
        if features.shape[0] == 0:
            raise ValueError("features must contain at least one sample")
        if features.shape[1] == 0:
            raise ValueError("features must contain at least one feature")
        if sparse.issparse(features):
            finite_features = np.all(np.isfinite(features.data))
        else:
            finite_features = np.all(np.isfinite(features))
        if not finite_features:
            raise ValueError("features must contain only finite values")
        labels = self._prepare_labels(labels)
        if not np.all(np.isfinite(labels)):
            raise ValueError("labels must contain only finite values")

        self._set("features", features)
        self._set("X", _design(features, self.fit_intercept))
        self._set("labels", np.ascontiguousarray(labels, dtype=np.float64))
        self._set("n_samples", int(features.shape[0]))
        self._set("n_features", int(features.shape[1]))
        self._set("dtype", np.dtype("float64"))

    def _prepare_labels(self, labels):
        return labels

    def _linear_scores(self, coeffs):
        return np.asarray(self.X @ coeffs, dtype=float).reshape(-1)

    def _sample_residual_and_loss(self, scores):
        raise NotImplementedError

    def _loss_and_grad(self, coeffs: np.ndarray, out: np.ndarray) -> float:
        residual, sample_losses = self._sample_residual_and_loss(self._linear_scores(coeffs))
        out[:] = np.asarray(self.X.T @ residual, dtype=float).reshape(-1) / self.n_samples
        return float(np.mean(sample_losses))

    def _loss(self, coeffs: np.ndarray) -> float:
        _, sample_losses = self._sample_residual_and_loss(self._linear_scores(coeffs))
        return float(np.mean(sample_losses))

    def _grad(self, coeffs: np.ndarray, out: np.ndarray) -> None:
        residual, _ = self._sample_residual_and_loss(self._linear_scores(coeffs))
        out[:] = np.asarray(self.X.T @ residual, dtype=float).reshape(-1) / self.n_samples


class _RobustLipschitzMixin:
    _lip_factor = 1.0

    def _feature_row_norm_sq(self) -> np.ndarray:
        if sparse.issparse(self.features):
            row_norms = np.asarray(self.features.multiply(self.features).sum(axis=1)).reshape(-1)
        else:
            row_norms = np.sum(np.asarray(self.features, dtype=float) ** 2, axis=1)
        if self.fit_intercept:
            row_norms = row_norms + 1.0
        return np.asarray(row_norms, dtype=float)

    def get_lip_mean(self) -> float:
        if not self._fitted:
            raise ValueError("call ``fit`` before using ``get_lip_mean``")
        return float(self._lip_factor * np.mean(self._feature_row_norm_sq()))

    def get_lip_max(self) -> float:
        if not self._fitted:
            raise ValueError("call ``fit`` before using ``get_lip_max``")
        return float(self._lip_factor * np.max(self._feature_row_norm_sq()))

    def _get_lip_best(self) -> float:
        features = self.features.toarray() if sparse.issparse(self.features) else np.asarray(self.features, dtype=float)
        spectral_sq = 0.0 if features.size == 0 else float(np.linalg.svd(features, full_matrices=False, compute_uv=False)[0] ** 2)
        if self.fit_intercept:
            spectral_sq += 1.0
        return float(self._lip_factor * spectral_sq / self.n_samples)


class _ThresholdMixin:
    def _set_threshold(self, value):
        value = float(value)
        if not np.isfinite(value) or value <= 0.0:
            raise RuntimeError("threshold must be > 0")
        self._threshold = value

    @property
    def threshold(self):
        return self._threshold

    @threshold.setter
    def threshold(self, value):
        self._set_threshold(value)

    def get_threshold(self):
        return self.threshold


class ModelHuber(
    _ThresholdMixin,
    _RobustGLMMixin,
    _RobustLipschitzMixin,
    ModelGeneralizedLinear,
    ModelFirstOrder,
    ModelLipschitz,
):
    """Huber loss for robust linear regression."""

    _lip_factor = 1.0

    def __init__(self, fit_intercept: bool = True, threshold: float = 1, n_threads: int = 1):
        super().__init__(fit_intercept=fit_intercept, n_threads=n_threads)
        self.threshold = threshold

    def _sample_residual_and_loss(self, scores):
        residual = scores - self.labels
        abs_residual = np.abs(residual)
        quadratic = abs_residual <= self.threshold
        losses = np.where(
            quadratic,
            0.5 * residual * residual,
            self.threshold * (abs_residual - 0.5 * self.threshold),
        )
        grad_residual = np.clip(residual, -self.threshold, self.threshold)
        return grad_residual, losses


class ModelAbsoluteRegression(_RobustGLMMixin, ModelGeneralizedLinear, ModelFirstOrder):
    """Absolute-value loss for robust linear regression."""

    def _sample_residual_and_loss(self, scores):
        residual = scores - self.labels
        return np.sign(residual), np.abs(residual)


class ModelEpsilonInsensitive(
    _ThresholdMixin,
    _RobustGLMMixin,
    ModelGeneralizedLinear,
    ModelFirstOrder,
):
    """Epsilon-insensitive loss for robust linear regression."""

    def __init__(self, fit_intercept: bool = True, threshold: float = 1, n_threads: int = 1):
        super().__init__(fit_intercept=fit_intercept, n_threads=n_threads)
        self.threshold = threshold

    def _sample_residual_and_loss(self, scores):
        residual = scores - self.labels
        abs_residual = np.abs(residual)
        active = abs_residual > self.threshold
        losses = np.where(active, abs_residual - self.threshold, 0.0)
        grad_residual = np.where(active, np.sign(residual), 0.0)
        return grad_residual, losses


class ModelModifiedHuber(
    _RobustGLMMixin,
    _RobustLipschitzMixin,
    ModelGeneralizedLinear,
    ModelFirstOrder,
    ModelLipschitz,
):
    """Modified Huber loss for binary classification."""

    _lip_factor = 2.0

    def _prepare_labels(self, labels):
        labels = np.asarray(labels, dtype=float).reshape(-1)
        classes = np.unique(labels)
        if classes.size != 2:
            raise ValueError("modified Huber labels must contain exactly two classes")
        self.classes_ = classes.copy()
        return np.where(labels == classes[-1], 1.0, -1.0)

    def _sample_residual_and_loss(self, scores):
        margin = self.labels * scores
        losses = np.zeros_like(margin, dtype=float)
        residual = np.zeros_like(margin, dtype=float)

        linear = margin <= -1.0
        quadratic = (margin > -1.0) & (margin < 1.0)

        losses[linear] = -4.0 * margin[linear]
        residual[linear] = -4.0 * self.labels[linear]

        violation = 1.0 - margin[quadratic]
        losses[quadratic] = violation * violation
        residual[quadratic] = -2.0 * self.labels[quadratic] * violation
        return residual, losses


class ModelLinRegWithIntercepts:
    """Linear regression model with one additional intercept per sample."""

    def __init__(self, fit_intercept: bool = True, n_threads: int = 1):
        self.fit_intercept = bool(fit_intercept)
        self.n_threads = int(n_threads)

    def fit(self, features, labels):
        self.features = _as_2d_float(features)
        self.labels = _as_1d_float(labels)
        if self.features.shape[0] != self.labels.size:
            raise ValueError("features and labels have inconsistent sample counts")
        self.n_samples = int(self.features.shape[0])
        self.n_features = int(self.features.shape[1])
        self.n_coeffs = self.n_features + self.n_samples + int(self.fit_intercept)
        return self

    def _split_coeffs(self, coeffs):
        coeffs = np.asarray(coeffs, dtype=float).reshape(-1)
        if coeffs.size != self.n_coeffs:
            raise ValueError(f"coeffs has size {coeffs.size}, expected {self.n_coeffs}")
        weights = coeffs[: self.n_features]
        if self.fit_intercept:
            intercept = float(coeffs[self.n_features])
            sample_intercepts = coeffs[self.n_features + 1 :]
        else:
            intercept = 0.0
            sample_intercepts = coeffs[self.n_features :]
        return weights, intercept, sample_intercepts

    def _linear_scores(self, coeffs):
        weights, intercept, sample_intercepts = self._split_coeffs(coeffs)
        scores = np.asarray(self.features @ weights, dtype=float).reshape(-1)
        return scores + intercept + sample_intercepts

    def loss(self, coeffs) -> float:
        residual = self._linear_scores(coeffs) - self.labels
        return float(0.5 * np.mean(residual * residual))

    def grad(self, coeffs) -> np.ndarray:
        residual = self._linear_scores(coeffs) - self.labels
        grad = np.empty(self.n_coeffs, dtype=float)
        grad[: self.n_features] = np.asarray(self.features.T @ residual, dtype=float).reshape(-1) / self.n_samples
        if self.fit_intercept:
            grad[self.n_features] = float(np.sum(residual) / self.n_samples)
            grad[self.n_features + 1 :] = residual / self.n_samples
        else:
            grad[self.n_features :] = residual / self.n_samples
        return grad

    def _feature_row_norm_sq(self) -> np.ndarray:
        if sparse.issparse(self.features):
            row_norms = np.asarray(self.features.multiply(self.features).sum(axis=1)).reshape(-1)
        else:
            row_norms = np.sum(np.asarray(self.features, dtype=float) ** 2, axis=1)
        row_norms = row_norms + 1.0
        if self.fit_intercept:
            row_norms = row_norms + 1.0
        return np.asarray(row_norms, dtype=float)

    def get_lip_mean(self) -> float:
        return float(np.mean(self._feature_row_norm_sq()))

    def get_lip_max(self) -> float:
        return float(np.max(self._feature_row_norm_sq()))

    def get_lip_best(self) -> float:
        if sparse.issparse(self.features):
            features = self.features.toarray()
        else:
            features = np.asarray(self.features, dtype=float)
        spectral_sq = 0.0 if features.size == 0 else float(np.linalg.svd(features, full_matrices=False, compute_uv=False)[0] ** 2)
        spectral_sq += 2.0 if self.fit_intercept else 1.0
        return float(spectral_sq / max(self.n_samples, 1))


@dataclass
class RobustLinearRegression:
    """Robust linear regression with sparse sample intercepts."""

    C_sample_intercepts: float
    C: float = 1e3
    fdr: float = 0.05
    penalty: str = "l2"
    fit_intercept: bool = True
    refit: bool = False
    solver: str = "agd"
    warm_start: bool = False
    step: float | None = None
    tol: float = 1e-7
    max_iter: int = 200
    verbose: bool = True
    print_every: int = 10
    record_every: int = 10
    elastic_net_ratio: float = 0.95
    slope_fdr: float = 0.05

    _solvers = {"gd": GD, "agd": AGD}
    _penalties = {
        "none": ProxZero,
        "l1": ProxL1,
        "l2": ProxL2Sq,
        "elasticnet": ProxElasticNet,
        "slope": ProxSlope,
    }

    def __post_init__(self):
        self.penalty = str(self.penalty).lower()
        self.solver = str(self.solver).lower()
        self._validate_positive_finite("C_sample_intercepts", self.C_sample_intercepts)
        self._validate_fdr("fdr", self.fdr)
        if self.refit:
            raise NotImplementedError("``refit`` can only be set to `False` for now")
        if self.solver not in self._solvers:
            raise ValueError(f"unknown solver {self.solver!r}")
        if self.penalty not in self._penalties:
            raise ValueError(f"unknown penalty {self.penalty!r}")
        if self.penalty != "none":
            self._validate_positive_finite("C", self.C)
        if self.penalty == "slope":
            self._validate_fdr("slope_fdr", self.slope_fdr)
        self.sample_intercepts = None
        self.coeffs = None
        self.weights = None
        self.intercept = None
        self.history = None
        self._fitted = False

    @staticmethod
    def _validate_positive_finite(name: str, value) -> None:
        if value is None:
            raise ValueError(f"``{name}`` cannot be `None`")
        if value == 0.0:
            raise ValueError(f"``{name}`` cannot be 0.")
        if value <= 0:
            raise ValueError(f"``{name}`` must be positive, got {value}")
        if np.isinf(value):
            raise ValueError(f"``{name}`` must be a finite number, got {value}")

    @staticmethod
    def _validate_fdr(name: str, value) -> None:
        if value is None:
            raise ValueError(f"``{name}`` cannot be `None`")
        if np.isinf(value):
            raise ValueError(f"``{name}`` must be a finite number, got {value}")
        if value <= 0 or value >= 1:
            raise ValueError(f"``{name}`` must be in (0, 1), got {value}")

    def _weights_prox(self, n_features: int):
        del n_features
        if self.penalty == "none":
            return ProxZero(range=(0, self.model.n_features))
        strength = 1.0 / float(self.C)
        prox_range = (0, self.model.n_features)
        if self.penalty == "l1":
            return ProxL1(strength, range=prox_range)
        if self.penalty == "l2":
            return ProxL2Sq(strength, range=prox_range)
        if self.penalty == "elasticnet":
            return ProxElasticNet(strength, ratio=self.elastic_net_ratio, range=prox_range)
        if self.penalty == "slope":
            return ProxSlope(strength, fdr=self.slope_fdr, range=prox_range)
        raise ValueError(f"unknown penalty {self.penalty!r}")

    def _intercepts_prox(self):
        start = self.model.n_features + int(self.fit_intercept)
        end = start + self.model.n_samples
        return ProxSlope(1.0 / float(self.C_sample_intercepts), fdr=self.fdr, range=(start, end))

    def fit(self, X, y):
        self.model = ModelLinRegWithIntercepts(fit_intercept=self.fit_intercept).fit(X, y)
        if self.step is None:
            self.step = 1.0 / self.model.get_lip_best()

        intercepts_prox = self._intercepts_prox()
        if self.penalty == "none":
            prox = intercepts_prox
        else:
            prox = ProxMulti([self._weights_prox(self.model.n_features), intercepts_prox])

        if self.warm_start and self.coeffs is not None:
            if self.coeffs.shape != (self.model.n_coeffs,):
                raise ValueError("Cannot warm start, coeffs don't have the right shape")
            coeffs_start = self.coeffs.copy()
        else:
            coeffs_start = np.zeros(self.model.n_coeffs, dtype=float)

        solver_cls = self._solvers[self.solver]
        solver = solver_cls(
            step=self.step,
            tol=self.tol,
            max_iter=self.max_iter,
            verbose=self.verbose,
            print_every=self.print_every,
            record_every=self.record_every,
        )
        coeffs = solver.set_model(self.model).set_prox(prox).solve(coeffs_start)

        self._solver_obj = solver
        self.history = solver.history
        self.coeffs = np.asarray(coeffs, dtype=float)
        self.weights = self.coeffs[: self.model.n_features].copy()
        if self.fit_intercept:
            self.intercept = float(self.coeffs[self.model.n_features])
            self.sample_intercepts = self.coeffs[self.model.n_features + 1 :].copy()
        else:
            self.intercept = None
            self.sample_intercepts = self.coeffs[self.model.n_features :].copy()
        self.coef_ = self.weights
        self.intercept_ = self.intercept
        self._fitted = True
        return self

    def predict(self, X):
        del X
        raise NotImplementedError("Not available for now.")

    def score(self, X):
        del X
        raise NotImplementedError("Not available for now.")

    def get_params(self):
        return {
            "C_sample_intercepts": self.C_sample_intercepts,
            "C": self.C,
            "fdr": self.fdr,
            "penalty": self.penalty,
            "fit_intercept": self.fit_intercept,
            "refit": self.refit,
            "solver": self.solver,
            "warm_start": self.warm_start,
            "step": self.step,
            "tol": self.tol,
            "max_iter": self.max_iter,
            "verbose": self.verbose,
            "print_every": self.print_every,
            "record_every": self.record_every,
            "elastic_net_ratio": self.elastic_net_ratio,
            "slope_fdr": self.slope_fdr,
        }

    def set_params(self, **params):
        for key, value in params.items():
            if not hasattr(self, key):
                raise ValueError(f"Unknown parameter {key!r}")
            setattr(self, key, value)
        self.__post_init__()
        return self

    @property
    def slope_fdr_value(self):
        if self.penalty == "slope":
            return self.slope_fdr
        warn(f'Penalty "{self.penalty}" has no ``slope_fdr`` attribute', RuntimeWarning, stacklevel=2)
        return None

