"""Cox proportional-hazards partial likelihood model."""

from __future__ import annotations

import numpy as np
from scipy import sparse

from hawkes_tools.base_model import ModelFirstOrder
from hawkes_tools.preprocessing.utils import safe_array

__all__ = ["ModelCoxRegPartialLik"]


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

        shift = float(np.max(scores_sorted))
        exp_scores = np.exp(scores_sorted - shift)
        cum_exp = np.cumsum(exp_scores)
        cum_weighted = np.cumsum(exp_scores[:, None] * X_sorted, axis=0)

        group_end = np.empty(times_sorted.shape[0], dtype=np.int64)
        start = 0
        while start < times_sorted.shape[0]:
            end = start
            while end + 1 < times_sorted.shape[0] and times_sorted[end + 1] == times_sorted[start]:
                end += 1
            group_end[start : end + 1] = end
            start = end + 1

        return scores_sorted, X_sorted, censoring_sorted, cum_exp, cum_weighted, group_end, shift

    def _loss(self, coeffs) -> float:
        (
            scores_sorted,
            _,
            censoring_sorted,
            cum_exp,
            _,
            group_end,
            shift,
        ) = self._risk_cache(coeffs)
        failure_positions = np.flatnonzero(censoring_sorted != 0)
        risk_indices = group_end[failure_positions]
        log_risk = shift + np.log(cum_exp[risk_indices])
        loss = np.sum(log_risk - scores_sorted[failure_positions])
        return float(loss / self.n_failures)

    def _grad(self, coeffs, out) -> None:
        (
            _,
            X_sorted,
            censoring_sorted,
            cum_exp,
            cum_weighted,
            group_end,
            _,
        ) = self._risk_cache(coeffs)
        out.fill(0.0)
        failure_positions = np.flatnonzero(censoring_sorted != 0)
        risk_indices = group_end[failure_positions]
        weighted_mean = cum_weighted[risk_indices] / cum_exp[risk_indices, None]
        out[:] = np.sum(weighted_mean - X_sorted[failure_positions], axis=0)
        out[:] /= self.n_failures

    def get_lip_max(self) -> float:
        row_norms = np.sum(self._features_dense * self._features_dense, axis=1)
        return float(max(np.max(row_norms), 1e-12))
