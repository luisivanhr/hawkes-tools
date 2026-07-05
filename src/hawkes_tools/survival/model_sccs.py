"""Self-controlled case series likelihood model."""

from __future__ import annotations

import numpy as np
from scipy import sparse

from hawkes_tools.base_model import ModelFirstOrder
from hawkes_tools.preprocessing.utils import (
    check_censoring_consistency,
    check_longitudinal_features_consistency,
)

from .simu_sccs import _validate_n_lags

__all__ = ["ModelSCCS"]


class ModelSCCS(ModelFirstOrder):
    """Discrete-time SCCS conditional multinomial likelihood."""

    def __init__(self, n_intervals: int, n_lags):
        super().__init__()
        self.n_intervals = int(n_intervals)
        if self.n_intervals <= 0:
            raise ValueError("n_intervals should be greater than 0")
        self.n_lags = _validate_n_lags(n_lags, n_intervals=self.n_intervals)
        self.n_features = int(len(self.n_lags))
        self.features = None
        self.labels = None
        self.censoring = None
        self.n_cases = None
        self._n_coeffs = None
        self._features_dense = None

    def fit(self, features, labels, censoring=None):
        return super().fit(features, labels, censoring)

    def _set_data(self, features, labels, censoring):
        n_intervals, n_coeffs = features[0].shape
        self._set("n_intervals", int(n_intervals))
        self._set("_n_coeffs", int(n_coeffs))
        self._set("n_cases", len(features))
        if len(labels) != self.n_cases:
            raise ValueError("Features and labels lists should have the same length.")
        if censoring is None:
            censoring = np.full(self.n_cases, self.n_intervals, dtype="uint64")
        censoring = check_censoring_consistency(censoring, self.n_cases)
        features = check_longitudinal_features_consistency(
            features, (n_intervals, n_coeffs), "float64"
        )
        labels = check_longitudinal_features_consistency(
            labels, (self.n_intervals,), "int32"
        )

        self.dtype = np.dtype("float64")
        self._set("features", features)
        self._set("labels", labels)
        self._set("censoring", censoring)
        self._features_dense = [
            np.asarray(x.toarray() if sparse.issparse(x) else x, dtype=float)
            for x in features
        ]

    def _get_n_coeffs(self):
        return self._n_coeffs

    @property
    def _epoch_size(self):
        return int(sum(np.sum(y[: int(c)] > 0) for y, c in zip(self.labels, self.censoring)))

    @property
    def _rand_max(self):
        return self._epoch_size

    def _loss(self, coeffs) -> float:
        coeffs = np.asarray(coeffs, dtype=float)
        total = 0.0
        for X, y, censoring_i in zip(self._features_dense, self.labels, self.censoring):
            c = int(censoring_i)
            Xc = X[:c]
            yc = np.asarray(y[:c], dtype=float)
            n_events = float(np.sum(yc))
            if n_events <= 0:
                continue
            scores = np.asarray(Xc @ coeffs, dtype=float).reshape(-1)
            shift = float(np.max(scores))
            log_sum = shift + np.log(np.sum(np.exp(scores - shift)))
            total += -(float(yc @ scores) - n_events * log_sum)
        return float(total / self.n_cases)

    def _grad(self, coeffs, out) -> None:
        coeffs = np.asarray(coeffs, dtype=float)
        out.fill(0.0)
        for X, y, censoring_i in zip(self._features_dense, self.labels, self.censoring):
            c = int(censoring_i)
            Xc = X[:c]
            yc = np.asarray(y[:c], dtype=float)
            n_events = float(np.sum(yc))
            if n_events <= 0:
                continue
            scores = np.asarray(Xc @ coeffs, dtype=float).reshape(-1)
            shift = float(np.max(scores))
            weights = np.exp(scores - shift)
            probs = weights / np.sum(weights)
            out[:] += n_events * (probs @ Xc) - (yc @ Xc)
        out[:] /= self.n_cases

    def get_lip_max(self) -> float:
        best = 0.0
        for X, y, censoring_i in zip(self._features_dense, self.labels, self.censoring):
            c = int(censoring_i)
            Xc = X[:c]
            n_events = float(np.sum(y[:c]))
            if n_events <= 0 or Xc.shape[0] <= 1:
                continue
            dmax = 0.0
            for i in range(Xc.shape[0]):
                diffs = Xc[i + 1 :] - Xc[i]
                if diffs.size:
                    dmax = max(dmax, float(np.max(np.sum(diffs * diffs, axis=1))))
            best = max(best, n_events * dmax / (4.0 * self.n_cases))
        return float(best)

    def get_lip_mean(self) -> float:
        return self.get_lip_max()

    def get_lip_best(self) -> float:
        raise NotImplementedError(
            "ModelSCCS is meant to be used with SVRG. Please use get_lip_max instead."
        )
