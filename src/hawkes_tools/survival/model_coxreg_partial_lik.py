"""Cox proportional-hazards partial likelihood model."""

from __future__ import annotations

import numpy as np
from numba import njit
from scipy import sparse

from hawkes_tools.base_model import ModelFirstOrder
from hawkes_tools.preprocessing.utils import safe_array

__all__ = ["ModelCoxRegPartialLik"]


@njit(cache=True)
def _cox_risk_prefixes(scores, features, times):
    """Accumulate scaled risk masses and normalized feature means in O(n p)."""
    n_samples, n_features = features.shape
    shifts = np.empty(n_samples)
    masses = np.empty(n_samples)
    means = np.empty((n_samples, n_features))
    group_end = np.empty(n_samples, dtype=np.int64)
    shift = scores[0]
    mass = 1.0
    shifts[0] = shift
    masses[0] = mass
    means[0] = features[0]
    tiny = np.finfo(np.float64).tiny

    for index in range(1, n_samples):
        new_shift = max(shift, scores[index])
        rescale = np.exp(shift - new_shift)
        old_mass = mass * rescale
        added_mass = np.exp(scores[index] - new_shift)
        new_mass = old_mass + added_mass
        old_weight = old_mass / new_mass
        added_weight = added_mass / new_mass

        # A tiny probability can still contribute a finite weighted feature.
        # Split its exponential before multiplying, avoiding early underflow.
        old_factor, old_factor_tail = old_weight, 1.0
        if old_weight < tiny or rescale < tiny:
            log_weight = np.log(mass) + (shift - new_shift) - np.log(new_mass)
            old_factor = np.exp(0.5 * log_weight)
            old_factor_tail = old_factor
        added_factor, added_factor_tail = added_weight, 1.0
        if added_weight < tiny:
            log_weight = (scores[index] - new_shift) - np.log(new_mass)
            added_factor = np.exp(0.5 * log_weight)
            added_factor_tail = added_factor

        for feature in range(n_features):
            previous = means[index - 1, feature]
            value = features[index, feature]
            if (previous >= 0.0) == (value >= 0.0):
                # Same-sign differences cannot overflow. Interpolate from the
                # dominant endpoint to retain small residual contributions.
                if old_weight <= added_weight:
                    mean = value + ((previous - value) * old_factor) * old_factor_tail
                else:
                    mean = previous + ((value - previous) * added_factor) * added_factor_tail
            else:
                # Opposite-sign differences can overflow; weighted terms cannot.
                mean = (previous * old_factor) * old_factor_tail
                mean += (value * added_factor) * added_factor_tail
            means[index, feature] = mean
        shift, mass = new_shift, new_mass
        shifts[index] = shift
        masses[index] = mass

    start = 0
    while start < n_samples:
        end = start
        while end + 1 < n_samples and times[end + 1] == times[start]:
            end += 1
        for index in range(start, end + 1):
            group_end[index] = end
        start = end + 1
    return masses, means, group_end, shifts


class ModelCoxRegPartialLik(ModelFirstOrder):
    """Negative partial log-likelihood for Cox regression."""

    def __init__(self):
        super().__init__()
        self.features = None
        self.times = None
        self.censoring = None
        self.n_samples = None
        self.n_features = None
        self.n_failures = None
        self.censoring_rate = None
        self._features_dense = None

    def fit(self, features, times, censoring):
        return super().fit(features, times, censoring)

    def _set_data(self, features, times, censoring):
        if sparse.issparse(features):
            dtype = np.dtype(features.dtype)
            if np.any(~np.isfinite(features.data)):
                raise ValueError("features must contain only finite values")
        else:
            features_arr = np.asarray(features)
            dtype = features_arr.dtype
            if np.any(~np.isfinite(features_arr)):
                raise ValueError("features must contain only finite values")
        times = np.asarray(times)
        if dtype != times.dtype:
            raise ValueError("Features and labels differ in data types")
        if np.any(~np.isfinite(times)) or np.any(times < 0):
            raise ValueError("times must contain only finite non-negative entries")
        censoring_arr = np.asarray(censoring)
        if not set(np.unique(censoring_arr)).issubset({0, 1}):
            raise ValueError("censoring must only have values in {0, 1}")

        n_samples, n_features = features.shape
        if n_samples != times.shape[0]:
            raise ValueError(
                "Features has %i samples while times have %i"
                % (n_samples, times.shape[0])
            )
        if n_samples != censoring_arr.shape[0]:
            raise ValueError(
                "Features has %i samples while censoring have %i"
                % (n_samples, censoring_arr.shape[0])
            )

        features = safe_array(features, dtype=dtype)
        times = safe_array(times, dtype=dtype)
        censoring = safe_array(censoring_arr, np.ushort)
        n_failures = int(np.sum(censoring != 0))
        if n_failures <= 0:
            raise ValueError("censoring must contain at least one failure")

        self.dtype = np.dtype(dtype)
        self._set("features", features)
        self._set("times", times)
        self._set("censoring", censoring)
        self._set("n_samples", int(n_samples))
        self._set("n_features", int(n_features))
        self._set("n_failures", n_failures)
        self._set("censoring_rate", 1.0 - n_failures / float(n_samples))
        if sparse.issparse(features):
            self._features_dense = np.asarray(features.toarray(), dtype=self.dtype)
        else:
            self._features_dense = np.asarray(features, dtype=self.dtype)

    def _get_n_coeffs(self):
        return self.n_features

    @property
    def _epoch_size(self):
        return self.n_failures

    @property
    def _rand_max(self):
        return self.n_failures

    def _risk_cache(self, coeffs):
        X = self._features_dense
        scores = np.asarray(X @ coeffs, dtype=float).reshape(-1)
        order = np.argsort(-self.times, kind="mergesort")
        times_sorted = self.times[order]
        scores_sorted = scores[order]
        X_sorted = X[order]
        censoring_sorted = self.censoring[order]

        # Each prefix has its own scale. Normalized feature means avoid an
        # overflowing numerator even when the eventual weighted mean is finite.
        cum_exp, risk_means, group_end, shifts = _cox_risk_prefixes(
            scores_sorted, X_sorted, times_sorted
        )
        return scores_sorted, X_sorted, censoring_sorted, cum_exp, risk_means, group_end, shifts

    def _loss(self, coeffs) -> float:
        return self._loss_from_risk(self._risk_cache(coeffs))

    def _grad(self, coeffs, out) -> None:
        self._grad_from_risk(self._risk_cache(coeffs), out)

    def _loss_and_grad(self, coeffs, out) -> float:
        risk = self._risk_cache(coeffs)
        self._grad_from_risk(risk, out)
        return self._loss_from_risk(risk)

    def _loss_from_risk(self, risk) -> float:
        (
            scores_sorted,
            _,
            censoring_sorted,
            cum_exp,
            _,
            group_end,
            shifts,
        ) = risk
        failure_positions = np.flatnonzero(censoring_sorted != 0)
        risk_indices = group_end[failure_positions]
        # Subtract scores before adding the log sum, preserving small losses
        # even when all scores have a large common offset.
        loss = np.sum(
            (shifts[risk_indices] - scores_sorted[failure_positions])
            + np.log(cum_exp[risk_indices])
        )
        return float(loss / self.n_failures)

    def _grad_from_risk(self, risk, out) -> None:
        (
            _,
            X_sorted,
            censoring_sorted,
            _,
            risk_means,
            group_end,
            _,
        ) = risk
        out.fill(0.0)
        failure_positions = np.flatnonzero(censoring_sorted != 0)
        risk_indices = group_end[failure_positions]
        weighted_mean = risk_means[risk_indices]
        out[:] = np.sum(weighted_mean - X_sorted[failure_positions], axis=0)
        out[:] /= self.n_failures

    def get_lip_max(self) -> float:
        row_norms = np.sum(self._features_dense * self._features_dense, axis=1)
        return float(max(np.max(row_norms), 1e-12))
