"""Standalone generalized linear models and simulators.

The module provides the GLM surface used by the example gallery while staying
pure Python/NumPy. Coefficient vectors follow the established convention:
feature weights first and the optional intercept last.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from warnings import warn

import numpy as np
from scipy import sparse
from scipy.linalg import toeplitz
from scipy.special import gammaln

from hawkes_tools.optim import (
    AGD,
    BFGS,
    GD,
    SAGA,
    SDCA,
    SGD,
    SVRG,
    ProxBinarsity,
    ProxElasticNet,
    ProxL1,
    ProxL2Sq,
    ProxTV,
    ProxZero,
)

if not hasattr(np, "trapz") and hasattr(np, "trapezoid"):
    np.trapz = np.trapezoid
if not hasattr(np, "in1d") and hasattr(np, "isin"):
    np.in1d = np.isin

try:
    from numba import njit
except Exception as exc:  # pragma: no cover
    raise ImportError("hawkes_tools.linear_model requires Numba") from exc

__all__ = [
    "weights_sparse_gauss",
    "ModelLinReg",
    "ModelLogReg",
    "ModelPoisReg",
    "ModelHinge",
    "ModelQuadraticHinge",
    "ModelSmoothedHinge",
    "LinearRegression",
    "LogisticRegression",
    "PoissonRegression",
    "LearnerLinReg",
    "LearnerLogReg",
    "LearnerPoisReg",
    "SimuLinReg",
    "SimuLogReg",
    "SimuPoisReg",
]


def _compile(func):
    return njit(cache=True)(func)


def weights_sparse_gauss(n_weights: int = 100, nnz: int = 10, std: float = 1.0, dtype="float64") -> np.ndarray:
    """Return a sparse Gaussian coefficient vector."""

    if nnz >= n_weights:
        warn("nnz must be smaller than n_weights using nnz=n_weights instead", RuntimeWarning, stacklevel=2)
        nnz = n_weights
    weights0 = np.zeros(int(n_weights), dtype=dtype)
    idx = np.arange(int(n_weights))
    np.random.shuffle(idx)
    weights0[idx[: int(nnz)]] = np.random.randn(int(nnz))
    weights0 *= float(std)
    return weights0


def _as_2d_float(X):
    if sparse.issparse(X):
        arr = X.tocsr().astype(np.float64)
        if arr.ndim != 2:
            raise ValueError("X must be a 2-dimensional array")
        if np.any(~np.isfinite(arr.data)):
            raise ValueError("X must contain only finite values")
        return arr
    arr = np.asarray(X, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("X must be a 2-dimensional array")
    if np.any(~np.isfinite(arr)):
        raise ValueError("X must contain only finite values")
    return np.ascontiguousarray(arr)


def _as_1d_float(y) -> np.ndarray:
    arr = np.ascontiguousarray(np.asarray(y, dtype=np.float64).reshape(-1))
    if np.any(~np.isfinite(arr)):
        raise ValueError("values must contain only finite numbers")
    return arr


def _design(X, fit_intercept: bool):
    X = _as_2d_float(X)
    if not fit_intercept:
        return X
    if sparse.issparse(X):
        ones = sparse.csr_matrix(np.ones((X.shape[0], 1), dtype=np.float64))
        return sparse.hstack([X, ones], format="csr")
    out = np.empty((X.shape[0], X.shape[1] + 1), dtype=np.float64)
    out[:, : X.shape[1]] = X
    out[:, -1] = 1.0
    return out


def _sigmoid(z):
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    mask = z >= 0.0
    out[mask] = 1.0 / (1.0 + np.exp(-z[mask]))
    ez = np.exp(z[~mask])
    out[~mask] = ez / (1.0 + ez)
    return out


def _linear_loss_grad_impl(X, y, beta, l2, l2_end):
    n, p = X.shape
    grad = np.zeros(p, dtype=np.float64)
    loss = 0.0
    for i in range(n):
        pred = 0.0
        for j in range(p):
            pred += X[i, j] * beta[j]
        residual = pred - y[i]
        loss += 0.5 * residual * residual
        for j in range(p):
            grad[j] += residual * X[i, j]
    loss /= n
    for j in range(p):
        grad[j] /= n
    for j in range(l2_end):
        loss += 0.5 * l2 * beta[j] * beta[j]
        grad[j] += l2 * beta[j]
    return loss, grad


def _logistic_loss_grad_impl(X, y, beta, l2, l2_end):
    n, p = X.shape
    grad = np.zeros(p, dtype=np.float64)
    loss = 0.0
    for i in range(n):
        z = 0.0
        for j in range(p):
            z += X[i, j] * beta[j]
        if z >= 0.0:
            prob = 1.0 / (1.0 + math.exp(-z))
            log_term = z + math.log1p(math.exp(-z))
        else:
            ez = math.exp(z)
            prob = ez / (1.0 + ez)
            log_term = math.log1p(ez)
        loss += log_term - y[i] * z
        residual = prob - y[i]
        for j in range(p):
            grad[j] += residual * X[i, j]
    loss /= n
    for j in range(p):
        grad[j] /= n
    for j in range(l2_end):
        loss += 0.5 * l2 * beta[j] * beta[j]
        grad[j] += l2 * beta[j]
    return loss, grad


def _poisson_loss_grad_impl(X, y, beta, l2, l2_end):
    n, p = X.shape
    grad = np.zeros(p, dtype=np.float64)
    loss = 0.0
    for i in range(n):
        eta = 0.0
        for j in range(p):
            eta += X[i, j] * beta[j]
        eta_eval = min(20.0, max(-20.0, eta))
        rate = math.exp(eta_eval)
        loss += rate - y[i] * eta_eval + math.lgamma(y[i] + 1.0)
        residual = rate - y[i]
        for j in range(p):
            grad[j] += residual * X[i, j]
    loss /= n
    for j in range(p):
        grad[j] /= n
    for j in range(l2_end):
        loss += 0.5 * l2 * beta[j] * beta[j]
        grad[j] += l2 * beta[j]
    return loss, grad


_linear_loss_grad = _compile(_linear_loss_grad_impl)
_logistic_loss_grad = _compile(_logistic_loss_grad_impl)
_poisson_loss_grad = _compile(_poisson_loss_grad_impl)


class _BaseModel:
    _lip_factor = 1.0

    def __init__(self, fit_intercept: bool = True, l2_strength: float = 0.0, dtype="float64", **kwargs):
        del kwargs
        self.fit_intercept = bool(fit_intercept)
        self.l2_strength = float(l2_strength)
        if not np.isfinite(self.l2_strength) or self.l2_strength < 0.0:
            raise ValueError("l2_strength must be a non-negative finite number")
        self.dtype = np.dtype(dtype)
        self._fitted = False

    def fit(self, X, y=None):
        if y is None:
            raise ValueError("fit requires features and labels")
        if not sparse.issparse(X) and np.asarray(X).ndim == 1 and not sparse.issparse(y) and np.asarray(y).ndim == 2:
            X, y = y, X
        self.features = _as_2d_float(X)
        if self.features.shape[0] == 0:
            raise ValueError("X must contain at least one sample")
        if self.features.shape[1] == 0:
            raise ValueError("X must contain at least one feature")
        self.X = _design(self.features, self.fit_intercept)
        self.y = self._prepare_y(y)
        if self.X.shape[0] != self.y.size:
            raise ValueError("X and y have inconsistent sample counts")
        self.n_samples = int(self.X.shape[0])
        self.n_features = int(self.features.shape[1])
        self.n_coeffs = int(self.X.shape[1])
        self._fitted = True
        return self

    def astype(self, dtype_or_object_with_dtype):
        dtype = getattr(dtype_or_object_with_dtype, "dtype", dtype_or_object_with_dtype)
        self.dtype = np.dtype(dtype)
        return self

    def _prepare_y(self, y):
        return _as_1d_float(y)

    @property
    def _l2_end(self):
        return self.n_features if self.fit_intercept else self.n_coeffs

    def _loss_grad(self, beta):
        raise NotImplementedError

    def loss(self, beta):
        loss, _ = self._loss_grad(self._validate_coeffs(beta))
        return float(loss)

    def grad(self, beta):
        _, grad = self._loss_grad(self._validate_coeffs(beta))
        return np.asarray(grad, dtype=float)

    def batch_grad(self, beta, indices) -> np.ndarray:
        beta = self._validate_coeffs(beta)
        indices = self._validate_indices(indices)
        if indices.size == 0:
            return np.zeros(self.n_coeffs, dtype=float)
        X_batch = self.X[indices]
        y_batch = self.y[indices]
        residual = self._batch_residual(X_batch, y_batch, beta)
        grad = np.asarray(X_batch.T @ residual, dtype=float).reshape(-1) / indices.size
        return self._add_smooth_l2_grad(beta, grad)

    def sample_residuals(self, beta, indices=None) -> np.ndarray:
        beta = self._validate_coeffs(beta)
        if indices is None:
            X_values = self.X
            y_values = self.y
        else:
            indices = self._validate_indices(indices)
            X_values = self.X[indices]
            y_values = self.y[indices]
        return np.asarray(self._batch_residual(X_values, y_values, beta), dtype=float).reshape(-1)

    def grad_from_residuals(self, indices, residuals, denominator: int | float) -> np.ndarray:
        self._check_fitted()
        indices = self._validate_indices(indices)
        residuals = np.asarray(residuals, dtype=float).reshape(-1)
        denominator = float(denominator)
        if not np.isfinite(denominator) or denominator <= 0.0:
            raise ValueError("denominator must be a positive finite number")
        if np.any(~np.isfinite(residuals)):
            raise ValueError("residuals must contain only finite values")
        if indices.size == 0:
            return np.zeros(self.n_coeffs, dtype=float)
        if residuals.size != indices.size:
            raise ValueError("residuals must have the same size as indices")
        grad = np.asarray(self.X[indices].T @ residuals, dtype=float).reshape(-1) / denominator
        return grad

    def smooth_l2_grad(self, beta) -> np.ndarray:
        beta = self._validate_coeffs(beta)
        grad = np.zeros(self.n_coeffs, dtype=float)
        return self._add_smooth_l2_grad(beta, grad)

    def _check_fitted(self):
        if not getattr(self, "_fitted", False):
            raise ValueError("You must call ``fit`` before")

    def _validate_coeffs(self, beta) -> np.ndarray:
        self._check_fitted()
        beta = np.ascontiguousarray(np.asarray(beta, dtype=np.float64).reshape(-1))
        if beta.size != self.n_coeffs:
            raise ValueError("coeffs length must match model.n_coeffs")
        if np.any(~np.isfinite(beta)):
            raise ValueError("coeffs must contain only finite values")
        return beta

    def _validate_indices(self, indices) -> np.ndarray:
        self._check_fitted()
        indices = np.asarray(indices, dtype=np.int64).reshape(-1)
        if indices.size and (np.any(indices < 0) or np.any(indices >= self.n_samples)):
            raise ValueError("indices are out of bounds")
        return indices

    def _batch_residual(self, X_batch, y_batch, beta):
        raise NotImplementedError

    def _add_l2(self, beta, loss, grad):
        if self.l2_strength:
            selected = beta[: self._l2_end]
            loss += 0.5 * self.l2_strength * float(np.dot(selected, selected))
            grad[: self._l2_end] += self.l2_strength * selected
        return float(loss), np.asarray(grad, dtype=float)

    def _add_smooth_l2_grad(self, beta, grad):
        if self.l2_strength:
            grad[: self._l2_end] += self.l2_strength * beta[: self._l2_end]
        return np.asarray(grad, dtype=float)

    def _linear_scores(self, beta):
        values = self.X @ beta
        return np.asarray(values, dtype=float).reshape(-1)

    def _feature_row_norm_sq(self):
        if sparse.issparse(self.features):
            norms = np.asarray(self.features.multiply(self.features).sum(axis=1)).reshape(-1)
        else:
            norms = np.sum(np.asarray(self.features, dtype=float) ** 2, axis=1)
        if self.fit_intercept:
            norms = norms + 1.0
        return np.asarray(norms, dtype=float)

    def get_lip_max(self) -> float:
        row_norms = self._feature_row_norm_sq()
        return float(self._lip_factor * np.max(row_norms) + self.l2_strength)

    def get_lip_mean(self) -> float:
        row_norms = self._feature_row_norm_sq()
        return float(self._lip_factor * np.mean(row_norms) + self.l2_strength)

    def get_lip_best(self) -> float:
        if sparse.issparse(self.features) and self.features.shape[0] * self.features.shape[1] > 2_000_000:
            return self.get_lip_max()
        X = self.features.toarray() if sparse.issparse(self.features) else np.asarray(self.features, dtype=float)
        if X.size == 0:
            spectral_sq = 0.0
        else:
            spectral_sq = float(np.linalg.svd(X, full_matrices=False, compute_uv=False)[0] ** 2)
        if self.fit_intercept:
            spectral_sq += 1.0
        return float(self._lip_factor * spectral_sq / max(self.n_samples, 1) + self.l2_strength)


class ModelLinReg(_BaseModel):
    _lip_factor = 1.0

    def _batch_residual(self, X_batch, y_batch, beta):
        return np.asarray(X_batch @ beta, dtype=float).reshape(-1) - y_batch

    def _loss_grad(self, beta):
        if not sparse.issparse(self.X):
            return _linear_loss_grad(self.X, self.y, beta, self.l2_strength, self._l2_end)
        residual = self._linear_scores(beta) - self.y
        loss = 0.5 * float(np.mean(residual * residual))
        grad = np.asarray(self.X.T @ residual, dtype=float).reshape(-1) / self.n_samples
        return self._add_l2(beta, loss, grad)


class ModelLogReg(_BaseModel):
    _lip_factor = 0.25

    def _prepare_y(self, y):
        y = np.asarray(y).reshape(-1)
        classes = np.unique(y)
        if classes.size != 2:
            raise ValueError("logistic regression requires exactly two classes")
        self.classes_ = classes.astype(y.dtype, copy=True)
        return np.ascontiguousarray((y == classes[-1]).astype(np.float64))

    def _loss_grad(self, beta):
        if not sparse.issparse(self.X):
            return _logistic_loss_grad(self.X, self.y, beta, self.l2_strength, self._l2_end)
        z = self._linear_scores(beta)
        prob = _sigmoid(z)
        loss = float(np.mean(np.logaddexp(0.0, z) - self.y * z))
        grad = np.asarray(self.X.T @ (prob - self.y), dtype=float).reshape(-1) / self.n_samples
        return self._add_l2(beta, loss, grad)

    def _batch_residual(self, X_batch, y_batch, beta):
        return _sigmoid(np.asarray(X_batch @ beta, dtype=float).reshape(-1)) - y_batch


class _BaseHingeModel(_BaseModel):
    _lip_factor = 0.0

    def _prepare_y(self, y):
        y = np.asarray(y).reshape(-1)
        classes = np.unique(y)
        if classes.size != 2:
            raise ValueError("hinge models require exactly two classes")
        self.classes_ = classes.astype(y.dtype, copy=True)
        return np.ascontiguousarray(np.where(y == classes[-1], 1.0, -1.0))

    def _loss_grad(self, beta):
        scores = self._linear_scores(beta)
        residual, sample_losses = self._margin_residual_and_loss(self.y, scores)
        loss = float(np.mean(sample_losses))
        grad = np.asarray(self.X.T @ residual, dtype=float).reshape(-1) / self.n_samples
        return self._add_l2(beta, loss, grad)

    def _batch_residual(self, X_batch, y_batch, beta):
        scores = np.asarray(X_batch @ beta, dtype=float).reshape(-1)
        residual, _ = self._margin_residual_and_loss(y_batch, scores)
        return residual

    def _margin_residual_and_loss(self, y, scores):
        raise NotImplementedError


class ModelHinge(_BaseHingeModel):
    """Linear hinge loss model."""

    def _margin_residual_and_loss(self, y, scores):
        margin = y * scores
        active = margin < 1.0
        losses = np.maximum(0.0, 1.0 - margin)
        residual = np.where(active, -y, 0.0)
        return residual, losses


class ModelQuadraticHinge(_BaseHingeModel):
    """Squared hinge loss model using ``0.5 * max(0, 1 - yz)^2``."""

    _lip_factor = 1.0

    def _margin_residual_and_loss(self, y, scores):
        violation = np.maximum(0.0, 1.0 - y * scores)
        losses = 0.5 * violation * violation
        residual = -y * violation
        return residual, losses


class ModelSmoothedHinge(_BaseHingeModel):
    """Huber-smoothed hinge loss model."""

    def __init__(
        self,
        fit_intercept: bool = True,
        l2_strength: float = 0.0,
        smoothness: float = 1.0,
        **kwargs,
    ):
        super().__init__(
            fit_intercept=fit_intercept, l2_strength=l2_strength, **kwargs
        )
        self._smoothness = None
        self.smoothness = smoothness
        self._model = self

    @property
    def smoothness(self):
        return self._smoothness

    @smoothness.setter
    def smoothness(self, value):
        value = float(value)
        if not 0.01 <= value <= 1.0:
            raise RuntimeError("smoothness should be between 0.01 and 1")
        self._smoothness = value
        self._lip_factor = 1.0 / value

    def get_smoothness(self):
        return self.smoothness

    def _margin_residual_and_loss(self, y, scores):
        violation = 1.0 - y * scores
        losses = np.zeros_like(violation, dtype=float)
        residual = np.zeros_like(violation, dtype=float)

        quadratic = (violation > 0.0) & (violation <= self.smoothness)
        linear = violation > self.smoothness

        losses[quadratic] = (
            violation[quadratic] * violation[quadratic] / (2.0 * self.smoothness)
        )
        residual[quadratic] = -y[quadratic] * violation[quadratic] / self.smoothness

        losses[linear] = violation[linear] - self.smoothness / 2.0
        residual[linear] = -y[linear]
        return residual, losses


class ModelPoisReg(_BaseModel):
    _lip_factor = 1.0

    def __init__(self, fit_intercept: bool = True, l2_strength: float = 0.0, link: str = "exponential", **kwargs):
        super().__init__(fit_intercept=fit_intercept, l2_strength=l2_strength, **kwargs)
        self.link = str(link).lower()
        if self.link not in {"exponential", "identity"}:
            raise ValueError("link must be 'exponential' or 'identity'")

    def _prepare_y(self, y):
        y = _as_1d_float(y)
        if np.any(y < 0.0):
            raise ValueError("Poisson labels must be non-negative")
        return y

    def _loss_grad(self, beta):
        if self.link == "identity":
            return self._identity_loss_grad(beta)
        if not sparse.issparse(self.X):
            return _poisson_loss_grad(self.X, self.y, beta, self.l2_strength, self._l2_end)
        eta = np.clip(self._linear_scores(beta), -20.0, 20.0)
        rate = np.exp(eta)
        loss = float(np.mean(rate - self.y * eta + gammaln(self.y + 1.0)))
        grad = np.asarray(self.X.T @ (rate - self.y), dtype=float).reshape(-1) / self.n_samples
        return self._add_l2(beta, loss, grad)

    def _identity_loss_grad(self, beta):
        eta = np.maximum(self._linear_scores(beta), 1e-12)
        loss = float(np.mean(eta - self.y * np.log(eta) + gammaln(self.y + 1.0)))
        residual = 1.0 - self.y / eta
        grad = np.asarray(self.X.T @ residual, dtype=float).reshape(-1) / self.n_samples
        return self._add_l2(beta, loss, grad)

    def _batch_residual(self, X_batch, y_batch, beta):
        if self.link == "identity":
            eta = np.maximum(np.asarray(X_batch @ beta, dtype=float).reshape(-1), 1e-12)
            return 1.0 - y_batch / eta
        eta = np.clip(np.asarray(X_batch @ beta, dtype=float).reshape(-1), -20.0, 20.0)
        return np.exp(eta) - y_batch


class _BaseLearner:
    model_class = _BaseModel

    _solver_classes = {
        "gd": GD,
        "agd": AGD,
        "bfgs": BFGS,
        "svrg": SVRG,
        "saga": SAGA,
        "sdca": SDCA,
        "sgd": SGD,
    }

    def __init__(
        self,
        penalty: str = "none",
        C: float = 1e3,
        fit_intercept: bool = True,
        solver: str = "bfgs",
        step: float | None = None,
        max_iter: int = 100,
        tol: float = 1e-5,
        verbose: bool = False,
        warm_start: bool = False,
        print_every: int = 10,
        record_every: int = 1,
        elastic_net_ratio: float = 0.95,
        blocks_start=None,
        blocks_length=None,
        random_state: int | None = None,
        n_threads: int = 1,
        **kwargs,
    ):
        self.penalty = penalty.lower()
        self.C = float(C)
        self.fit_intercept = bool(fit_intercept)
        self.solver = solver.lower()
        self.step = step
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.verbose = bool(verbose)
        self.warm_start = bool(warm_start)
        self.print_every = int(print_every)
        self.record_every = int(record_every)
        self.elastic_net_ratio = float(elastic_net_ratio)
        self.blocks_start = blocks_start
        self.blocks_length = blocks_length
        self.random_state = random_state
        self.n_threads = int(n_threads)
        self.extra_kwargs = dict(kwargs)
        self._fitted = False
        if self.solver not in self._solver_classes:
            raise ValueError(f"unknown solver {self.solver!r}")
        supported_penalties = {
            "none",
            "zero",
            "",
            "l2",
            "ridge",
            "l1",
            "lasso",
            "elasticnet",
            "elastic_net",
            "tv",
            "binarsity",
        }
        if self.penalty not in supported_penalties:
            raise ValueError("supported penalties are 'none', 'l1', 'l2', 'elasticnet', 'tv', and 'binarsity'")
        if self.penalty not in {"none", "zero", ""} and (not np.isfinite(self.C) or self.C <= 0.0):
            raise ValueError("C must be positive")
        if self.penalty in {"elasticnet", "elastic_net"} and not 0.0 <= self.elastic_net_ratio <= 1.0:
            raise ValueError("elastic_net_ratio must be between 0 and 1")
        if self.penalty == "binarsity":
            if blocks_start is None:
                raise ValueError("Penalty 'binarsity' requires ``blocks_start``, got None")
            if blocks_length is None:
                raise ValueError("Penalty 'binarsity' requires ``blocks_length``, got None")
            validator = ProxBinarsity(0.0, blocks_start=blocks_start, blocks_length=blocks_length)
            self.blocks_start = validator.blocks_start
            self.blocks_length = validator.blocks_length
        self._prox_obj = None

    def _penalty_strength(self):
        if self.penalty in {"none", "zero", ""}:
            return 0.0
        if self.C <= 0:
            raise ValueError("C must be positive")
        return 1.0 / self.C

    def _prox_range(self):
        return (0, self.model.n_features) if self.fit_intercept else None

    def _prox(self):
        strength = self._penalty_strength()
        if self.penalty in {"none", "zero", ""}:
            return ProxZero()
        if self.penalty in {"l2", "ridge"}:
            return ProxL2Sq(strength, range=self._prox_range())
        if self.penalty in {"l1", "lasso"}:
            return ProxL1(strength, range=self._prox_range())
        if self.penalty in {"elasticnet", "elastic_net"}:
            return ProxElasticNet(strength, ratio=self.elastic_net_ratio, range=self._prox_range())
        if self.penalty == "tv":
            return ProxTV(strength, range=self._prox_range())
        if self.penalty == "binarsity":
            return ProxBinarsity(
                strength,
                blocks_start=self.blocks_start,
                blocks_length=self.blocks_length,
                range=self._prox_range(),
            )
        raise ValueError("supported penalties are 'none', 'l1', 'l2', 'elasticnet', 'tv', and 'binarsity'")

    def _construct_model_obj(self, fit_intercept=True):
        return self.model_class(fit_intercept=fit_intercept)

    def fit(self, X, y):
        self.model = self._construct_model_obj(self.fit_intercept).fit(X, y)
        if self.warm_start and self._fitted and getattr(self, "coeffs", None) is not None and self.coeffs.size == self.model.n_coeffs:
            beta0 = self.coeffs.copy()
        else:
            beta0 = np.zeros(self.model.n_coeffs, dtype=float)
        solver_cls = self._solver_classes.get(self.solver)
        if solver_cls is None:
            raise ValueError(f"unknown solver {self.solver!r}")
        if self.solver == "bfgs":
            solver = solver_cls(
                tol=self.tol,
                max_iter=self.max_iter,
                verbose=self.verbose,
                print_every=self.print_every,
                record_every=self.record_every,
                random_state=self.random_state,
                n_threads=self.n_threads,
            )
        else:
            solver = solver_cls(
                step=self.step,
                tol=self.tol,
                max_iter=self.max_iter,
                verbose=self.verbose,
                print_every=self.print_every,
                record_every=self.record_every,
                random_state=self.random_state,
                n_threads=self.n_threads,
            )
        prox = self._prox()
        beta = solver.set_model(self.model).set_prox(prox).solve(beta0)
        self._solver_obj = solver
        self._prox_obj = prox
        self.history = solver.history
        self.coeffs = np.asarray(beta, dtype=float)
        if self.fit_intercept:
            self.coef_ = self.coeffs[: self.model.n_features].copy()
            self.intercept_ = float(self.coeffs[-1])
        else:
            self.coef_ = self.coeffs.copy()
            self.intercept_ = None
        self._fitted = True
        return self

    @property
    def weights(self):
        return self.coef_

    @property
    def intercept(self):
        return self.intercept_

    def _check_fitted(self):
        if not self._fitted:
            raise ValueError("You must call ``fit`` before")

    def decision_function(self, X):
        self._check_fitted()
        X = _as_2d_float(X)
        if X.shape[1] != self.coef_.size:
            raise ValueError("X has the wrong number of features")
        values = X @ self.coef_
        values = np.asarray(values, dtype=float).reshape(-1)
        if self.fit_intercept and self.intercept_ is not None:
            values = values + self.intercept_
        return values


class LinearRegression(_BaseLearner):
    model_class = ModelLinReg

    def predict(self, X):
        return self.decision_function(X)

    def score(self, X, y):
        y = _as_1d_float(y)
        pred = self.predict(X)
        return 1.0 - float(np.sum((y - pred) ** 2)) / max(float(np.sum((y - y.mean()) ** 2)), 1e-15)


class LogisticRegression(_BaseLearner):
    model_class = ModelLogReg

    def __init__(self, penalty: str = "l2", **kwargs):
        super().__init__(penalty=penalty, **kwargs)
        self.classes = None

    def fit(self, X, y):
        super().fit(X, y)
        self.classes_ = self.model.classes_
        self.classes = self.classes_
        return self

    def predict_proba(self, X):
        self._check_fitted()
        p = _sigmoid(self.decision_function(X))
        return np.column_stack([1.0 - p, p])

    def predict(self, X):
        return np.where(self.predict_proba(X)[:, 1] >= 0.5, self.classes_[-1], self.classes_[0])

    def score(self, X, y):
        return float(np.mean(self.predict(X) == np.asarray(y)))


class PoissonRegression(_BaseLearner):
    model_class = ModelPoisReg

    def __init__(self, step: float | None = 1e-3, penalty: str = "l2", **kwargs):
        super().__init__(step=step, penalty=penalty, **kwargs)

    def _construct_model_obj(self, fit_intercept=True):
        return ModelPoisReg(fit_intercept=fit_intercept, link="exponential")

    def decision_function(self, X):
        linear = super().decision_function(X)
        return np.exp(np.clip(linear, -20.0, 20.0))

    def predict(self, X):
        return np.rint(self.decision_function(X))

    def loglik(self, X, y):
        self._check_fitted()
        coeffs = self.coeffs if self.fit_intercept else self.coef_
        model = self._construct_model_obj(self.fit_intercept).fit(X, y)
        return model.loss(coeffs)


LearnerLinReg = LinearRegression
LearnerLogReg = LogisticRegression
LearnerPoisReg = PoissonRegression


@dataclass
class _BaseSimuReg:
    coeffs: np.ndarray | None = None
    weights: np.ndarray | None = None
    intercept: float | None = 0.0
    n_samples: int = 100
    n_features: int | None = None
    features: object | None = None
    features_type: str = "none"
    cov_corr: float = 0.5
    features_scaling: str = "none"
    seed: int | None = None
    verbose: bool = False
    dtype: str = "float64"

    def _coeffs(self, rng, n_features):
        if self.coeffs is not None and self.weights is not None:
            coeffs = _as_1d_float(self.coeffs)
            weights = _as_1d_float(self.weights)
            if coeffs.size != weights.size or not np.allclose(coeffs, weights):
                raise ValueError("coeffs and weights both provided with different values")
            return coeffs
        raw = self.coeffs if self.coeffs is not None else self.weights
        coeffs = rng.normal(scale=0.5, size=n_features) if raw is None else _as_1d_float(raw)
        if coeffs.size != n_features:
            raise ValueError("coeffs size must match number of features")
        return coeffs.astype(self.dtype, copy=False)

    def _simulate_features(self, rng, n_features):
        features_type = str(self.features_type).lower()
        if features_type == "cov_toeplitz":
            covariance = toeplitz(float(self.cov_corr) ** np.arange(n_features))
            X = rng.multivariate_normal(np.zeros(n_features), covariance, size=int(self.n_samples))
        elif features_type == "cov_uniform":
            X = rng.uniform(-1.0, 1.0, size=(int(self.n_samples), n_features))
        elif features_type in {"none", "independent", "gaussian"}:
            X = rng.normal(size=(int(self.n_samples), n_features))
        else:
            raise ValueError("features_type must be 'cov_toeplitz', 'cov_uniform', or 'none'")
        if str(self.features_scaling).lower() == "standard":
            X = (X - X.mean(axis=0)) / np.maximum(X.std(axis=0), 1e-12)
        return np.asarray(X, dtype=self.dtype)

    def _features_and_coeffs(self):
        rng = np.random.default_rng(self.seed)
        if self.features is None:
            n_features = self.n_features or (2 if self.coeffs is None and self.weights is None else _as_1d_float(self.coeffs if self.coeffs is not None else self.weights).size)
            X = self._simulate_features(rng, int(n_features))
        else:
            X = _as_2d_float(self.features)
            n_features = X.shape[1]
        coeffs = self._coeffs(rng, int(n_features))
        return X, coeffs, rng

    def _intercept_value(self):
        return 0.0 if self.intercept is None else float(self.intercept)


@dataclass
class SimuLinReg(_BaseSimuReg):
    noise_std: float = 1.0

    def simulate(self):
        X, coeffs, rng = self._features_and_coeffs()
        y = self._intercept_value() + np.asarray(X @ coeffs).reshape(-1) + rng.normal(scale=self.noise_std, size=X.shape[0])
        self.features, self.coeffs, self.weights, self.labels = X, coeffs, coeffs, y
        return X, y


@dataclass
class SimuLogReg(_BaseSimuReg):
    label_values: tuple[float, float] = (-1.0, 1.0)

    def simulate(self):
        X, coeffs, rng = self._features_and_coeffs()
        probs = _sigmoid(self._intercept_value() + np.asarray(X @ coeffs).reshape(-1))
        draws = rng.binomial(1, probs)
        self.features, self.coeffs, self.weights, self.probabilities = X, coeffs, coeffs, probs
        self.labels = np.where(draws > 0, self.label_values[1], self.label_values[0])
        return X, self.labels


@dataclass
class SimuPoisReg(_BaseSimuReg):
    clip: float = 8.0
    link: str = "exponential"

    def simulate(self):
        X, coeffs, rng = self._features_and_coeffs()
        linear = self._intercept_value() + np.asarray(X @ coeffs).reshape(-1)
        if str(self.link).lower() == "identity":
            rates = np.maximum(linear, 1e-12)
        elif str(self.link).lower() == "exponential":
            rates = np.exp(np.clip(linear, -float(self.clip), float(self.clip)))
        else:
            raise ValueError("link must be 'identity' or 'exponential'")
        self.features, self.coeffs, self.weights, self.rates = X, coeffs, coeffs, rates
        self.labels = rng.poisson(rates).astype(float)
        return X, self.labels

