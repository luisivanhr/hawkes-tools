"""Convolutional SCCS learner for gallery-scale pure-Python workflows."""

from __future__ import annotations

import time

import numpy as np
from scipy import sparse
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.stats import norm

from .simu_sccs import _lag_feature_matrix, _validate_n_lags

__all__ = ["ConvSCCS", "BatchConvSCCS", "StreamConvSCCS"]


class ConvSCCS:
    """Estimate SCCS lag coefficients from longitudinal exposure matrices."""

    def __init__(
        self,
        n_lags,
        penalized_features=None,
        C_tv=None,
        C_group_l1=None,
        step: float | None = None,
        tol: float = 1e-5,
        max_iter: int = 100,
        verbose: bool = False,
        print_every: int = 10,
        record_every: int = 10,
        random_state: int | None = None,
    ):
        self.n_lags = _validate_n_lags(n_lags)
        self.penalized_features = None if penalized_features is None else np.asarray(penalized_features)
        self.C_tv = self._optional_positive_finite("C_tv", C_tv)
        self.C_group_l1 = self._optional_positive_finite("C_group_l1", C_group_l1)
        self.step = self._optional_positive_finite("step", step)
        self.tol = float(tol)
        self.max_iter = int(max_iter)
        if not np.isfinite(self.tol) or self.tol < 0.0:
            raise ValueError("tol should be non-negative")
        if self.max_iter < 0:
            raise ValueError("max_iter should be non-negative")
        self.verbose = bool(verbose)
        self.print_every = int(print_every)
        self.record_every = int(record_every)
        if self.print_every <= 0:
            raise ValueError("print_every should be greater than 0")
        if self.record_every <= 0:
            raise ValueError("record_every should be greater than 0")
        self.random_state = random_state
        self.n_cases = None
        self.n_intervals = None
        self.n_features = None
        self.n_coeffs = int(self.n_lags.sum() + self.n_lags.size)
        self._features_offset = self._offsets(self.n_lags)
        self._coeffs = np.zeros(self.n_coeffs, dtype=float)
        self.confidence_intervals = {"refit_coeffs": [], "lower_bound": [], "upper_bound": [], "confidence_level": None}
        self._fitted = False
        self._fitted_design = None
        self._fitted_y = None
        self._fitted_lengths = None
        self.time_elapsed = None

    @staticmethod
    def _optional_positive_finite(name, value):
        if value is None:
            return None
        numeric = float(value)
        if not np.isfinite(numeric) or numeric <= 0.0:
            raise ValueError(f"{name} should be a float greater than zero.")
        return numeric

    @staticmethod
    def _offsets(n_lags):
        offsets = [0]
        for lag in n_lags:
            offsets.append(offsets[-1] + int(lag) + 1)
        return offsets

    @property
    def coeffs(self):
        return self._format_coeffs(self._coeffs)

    @property
    def intensities(self):
        return [np.exp(coeff) for coeff in self.coeffs]

    def fit(
        self,
        features,
        labels,
        censoring,
        confidence_intervals: bool = False,
        n_samples_bootstrap: int = 200,
        confidence_level: float = 0.95,
    ):
        del n_samples_bootstrap
        start = time.perf_counter()
        design, y, lengths = self._stack_lagged_data(features, labels, censoring)
        self._fitted_design = design
        self._fitted_y = y
        self._fitted_lengths = lengths
        beta0, event_counts, expected_counts = self._score_ratio_initialization(design, y, lengths)

        objective = self._loss_grad_factory(design, y, lengths, include_penalty=True)
        result = minimize(
            objective,
            beta0,
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": self.max_iter, "ftol": self.tol, "gtol": self.tol, "maxls": 20},
        )
        self._coeffs = np.asarray(result.x if result.success or result.x is not None else beta0, dtype=float)
        self._center_singleton_identifiability()
        self._fitted = True
        if confidence_intervals:
            self.confidence_intervals = self._confidence_intervals(event_counts, expected_counts, confidence_level)
        else:
            self.confidence_intervals = {
                "refit_coeffs": self.coeffs,
                "lower_bound": [None for _ in self.coeffs],
                "upper_bound": [None for _ in self.coeffs],
                "confidence_level": None,
            }
        self.time_elapsed = time.perf_counter() - start
        return self.coeffs, self.confidence_intervals

    def _stack_lagged_data(self, features, labels, censoring):
        if features is None:
            raise ValueError("Passed ``features`` is None")
        if labels is None:
            raise ValueError("Passed ``labels`` is None")
        if censoring is None:
            raise ValueError("Passed ``censoring`` is None")
        if len(features) != len(labels):
            raise ValueError("features and labels must have the same length")
        if len(features) == 0:
            raise ValueError("features must contain at least one case")
        censoring = np.asarray(censoring, dtype=np.uint64).reshape(-1)
        if censoring.size != len(features):
            raise ValueError("censoring must have one entry per case")
        self.n_cases = len(features)
        self.n_intervals, self.n_features = features[0].shape
        if self.n_lags.size != self.n_features:
            raise ValueError("n_lags must have one entry per feature")
        self._validate_penalized_features()
        self.n_coeffs = int(self.n_lags.sum() + self.n_features)
        self._features_offset = self._offsets(self.n_lags)

        blocks = []
        y_parts = []
        lengths = []
        for feature, label, stop in zip(features, labels, censoring):
            lagged = _lag_feature_matrix(feature, self.n_lags)
            stop_i = int(min(max(stop, 1), lagged.shape[0]))
            label_arr = np.asarray(label, dtype=np.int32).reshape(-1)
            if label_arr.size != lagged.shape[0]:
                raise ValueError("labels must match the number of intervals")
            blocks.append(lagged[:stop_i])
            y_parts.append(label_arr[:stop_i])
            lengths.append(stop_i)
        return sparse.vstack(blocks, format="csr"), np.concatenate(y_parts), np.asarray(lengths, dtype=np.int64)

    def _score_ratio_initialization(self, design, y, lengths):
        event_counts = np.asarray(design.T @ y, dtype=float).reshape(-1)
        expected_counts = np.zeros(design.shape[1], dtype=float)
        start = 0
        for length in lengths:
            stop = start + int(length)
            event_count = float(np.sum(y[start:stop]))
            if event_count > 0:
                expected_counts += event_count * np.asarray(design[start:stop].mean(axis=0)).reshape(-1)
            start = stop
        smooth = 0.5
        beta0 = np.log((event_counts + smooth) / (expected_counts + smooth))
        beta0[~np.isfinite(beta0)] = 0.0
        return np.clip(beta0, -4.0, 4.0), event_counts, expected_counts

    def _loss_grad_factory(self, design, y, lengths, *, include_penalty: bool = False):
        n_effective = max(int(np.sum([np.sum(y[start : start + length]) > 0 for start, length in self._starts(lengths)])), 1)

        def loss_grad(coeffs):
            coeffs = np.asarray(coeffs, dtype=float)
            eta = np.asarray(design @ coeffs, dtype=float).reshape(-1)
            residual = np.zeros_like(eta)
            loss = 0.0
            for start, length in self._starts(lengths):
                stop = start + length
                y_seg = y[start:stop]
                event_count = float(np.sum(y_seg))
                if event_count <= 0:
                    continue
                eta_seg = eta[start:stop]
                log_norm = logsumexp(eta_seg)
                probabilities = np.exp(eta_seg - log_norm)
                residual[start:stop] = event_count * probabilities - y_seg
                loss += event_count * log_norm - float(np.dot(y_seg, eta_seg))
            loss /= n_effective
            grad = np.asarray(design.T @ residual, dtype=float).reshape(-1) / n_effective
            ridge = 1e-8
            value = float(loss + 0.5 * ridge * np.dot(coeffs, coeffs))
            grad = grad + ridge * coeffs
            if include_penalty:
                penalty_value, penalty_grad = self._penalty_value_grad(coeffs)
                value += penalty_value
                grad = grad + penalty_grad
            return value, grad

        return loss_grad

    def _validate_penalized_features(self):
        if self.penalized_features is None:
            return
        arr = np.asarray(self.penalized_features)
        if arr.ndim != 1:
            raise ValueError("penalized_features must be a one-dimensional array")
        if arr.dtype == bool:
            if arr.size != self.n_features:
                raise ValueError("boolean penalized_features must have one entry per feature")
            return
        numeric = arr.astype(int)
        if not np.all(numeric == arr):
            raise ValueError("penalized_features must contain integer feature indices")
        if np.any(numeric < 0) or np.any(numeric >= self.n_features):
            raise ValueError("penalized_features indices are out of bounds")

    def _penalized_feature_indices(self) -> np.ndarray:
        if self.C_tv is None and self.C_group_l1 is None:
            return np.asarray([], dtype=int)
        if self.penalized_features is None:
            return np.arange(self.n_features, dtype=int)
        arr = np.asarray(self.penalized_features)
        if arr.dtype == bool:
            return np.flatnonzero(arr)
        return np.unique(arr.astype(int))

    def _penalty_value_grad(self, coeffs):
        value = 0.0
        grad = np.zeros_like(coeffs, dtype=float)
        for feature_idx in self._penalized_feature_indices():
            start = int(self._features_offset[feature_idx])
            stop = int(self._features_offset[feature_idx + 1])
            block = coeffs[start:stop]
            if self.C_tv is not None and block.size > 1:
                strength = 1.0 / float(self.C_tv)
                diffs = np.diff(block)
                signs = np.sign(diffs)
                value += strength * float(np.sum(np.abs(diffs)))
                grad[start : stop - 1] -= strength * signs
                grad[start + 1 : stop] += strength * signs
            if self.C_group_l1 is not None and block.size:
                strength = 1.0 / float(self.C_group_l1)
                norm = float(np.linalg.norm(block))
                scale = strength * np.sqrt(block.size)
                value += scale * norm
                if norm > 0.0:
                    grad[start:stop] += scale * block / norm
        return float(value), grad

    @staticmethod
    def _starts(lengths):
        start = 0
        for length in lengths:
            yield start, int(length)
            start += int(length)

    def _center_singleton_identifiability(self):
        singleton = np.asarray([int(lag) == 0 for lag in self.n_lags])
        if not np.any(singleton):
            return
        indices = [self._features_offset[i] for i, is_single in enumerate(singleton) if is_single]
        if len(indices) > 1:
            self._coeffs[indices] -= float(np.mean(self._coeffs[indices]))

    def _confidence_intervals(self, event_counts, expected_counts, confidence_level):
        if confidence_level <= 0 or confidence_level >= 1:
            raise ValueError("`confidence_level` should be in (0, 1)")
        smooth = 0.5
        z_value = norm.ppf(1.0 - (1.0 - confidence_level) / 2.0)
        se = np.sqrt(1.0 / (event_counts + smooth) + 1.0 / (expected_counts + smooth))
        se[~np.isfinite(se)] = 0.0
        lower = self._coeffs - z_value * se
        upper = self._coeffs + z_value * se
        return {
            "refit_coeffs": self.coeffs,
            "lower_bound": self._format_coeffs(lower),
            "upper_bound": self._format_coeffs(upper),
            "confidence_level": float(confidence_level),
        }

    def _format_coeffs(self, coeffs):
        coeffs = np.asarray(coeffs, dtype=float).reshape(-1)
        return [
            coeffs[int(self._features_offset[i]) : int(self._features_offset[i + 1])].copy()
            for i in range(self.n_lags.size)
        ]

    def score(self, features=None, labels=None, censoring=None):
        if not self._fitted:
            raise RuntimeError("You must fit the model first")
        if features is None and labels is None and censoring is None:
            return self._loss_grad_factory(self._fitted_design, self._fitted_y, self._fitted_lengths)(self._coeffs)[0]
        design, y, lengths = self._stack_lagged_data(features, labels, censoring)
        return self._loss_grad_factory(design, y, lengths)(self._coeffs)[0]


def _validate_positive_int(name, value) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer greater than 0")
    try:
        numeric = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer greater than 0") from exc
    if numeric != value or numeric <= 0:
        raise ValueError(f"{name} must be an integer greater than 0")
    return numeric


class BatchConvSCCS(ConvSCCS):
    """Batch-oriented ConvSCCS API using the standalone sequential backend.

    The original tick class used compiled multi-solve support to parallelize
    cross-validation and bootstrap work. In hawkes-tools, performance backend
    parity is intentionally out of scope, so this class keeps the public
    construction surface and validation while reusing ``ConvSCCS.fit`` and
    ``ConvSCCS.score``.
    """

    def __init__(self, *args, batch_size: int = 1, **kwargs):
        self.batch_size = _validate_positive_int("batch_size", batch_size)
        super().__init__(*args, **kwargs)


class StreamConvSCCS(ConvSCCS):
    """Thread-oriented ConvSCCS API using the standalone sequential backend.

    The ``threads`` attribute is validated and stored for API continuity, but
    the pure-Python implementation does not launch tick's compiled threadpool.
    """

    def __init__(self, *args, threads: int = 1, **kwargs):
        self.threads = _validate_positive_int("threads", threads)
        super().__init__(*args, **kwargs)
