"""Longitudinal lag feature construction."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sps

from .base import LongitudinalPreprocessor
from .utils import check_censoring_consistency, check_longitudinal_features_consistency


class LongitudinalFeaturesLagger(LongitudinalPreprocessor):
    """Add current and lagged exposure columns for each longitudinal feature."""

    def __init__(self, n_lags, n_jobs: int = -1):
        super().__init__(n_jobs=n_jobs)
        if not isinstance(n_lags, np.ndarray) or n_lags.dtype != "uint64":
            raise ValueError("`n_lags` should be a numpy array of dtype uint64")
        self.n_lags = n_lags
        self._reset()

    def _reset(self):
        self._n_init_features = None
        self._n_output_features = None
        self._n_intervals = None
        self._fitted = False

    def fit(self, features, labels=None, censoring=None):
        del labels, censoring
        self._reset()
        base_shape = features[0].shape
        if base_shape[1] != len(self.n_lags):
            raise ValueError("Number of columns from feature matrices differs from self.n_lags length.")
        features = check_longitudinal_features_consistency(features, base_shape, "float64")
        del features
        n_intervals, n_init_features = base_shape
        self._n_init_features = n_init_features
        self._n_intervals = n_intervals
        self._n_output_features = int((self.n_lags + 1).sum())
        self._fitted = True
        return self

    def transform(self, features, labels=None, censoring=None):
        if not self._fitted:
            raise ValueError("cannot transform before fit")
        n_samples = len(features)
        if censoring is None:
            censoring = np.full((n_samples,), self._n_intervals, dtype="uint64")
        censoring = check_censoring_consistency(censoring, n_samples)
        base_shape = (self._n_intervals, self._n_init_features)
        features = check_longitudinal_features_consistency(features, base_shape, "float64")

        if sps.issparse(features[0]):
            X_with_lags = [
                self._sparse_lagger(x, int(censoring[i])) for i, x in enumerate(features)
            ]
        else:
            X_with_lags = [
                self._dense_lagger(x, int(censoring[i])) for i, x in enumerate(features)
            ]
        return X_with_lags, labels, censoring

    def _feature_offsets(self):
        offsets = np.zeros(len(self.n_lags), dtype=int)
        cursor = 0
        for i, n_lag in enumerate(self.n_lags):
            offsets[i] = cursor
            cursor += int(n_lag) + 1
        return offsets

    def _dense_lagger(self, feature_matrix, censoring_i):
        output = np.zeros((self._n_intervals, self._n_output_features), dtype="float64")
        censoring_i = min(max(int(censoring_i), 0), self._n_intervals)
        col_start = 0
        for feature_idx, n_lag in enumerate(self.n_lags):
            for lag in range(int(n_lag) + 1):
                if lag < censoring_i:
                    output[lag:censoring_i, col_start + lag] = feature_matrix[
                        : censoring_i - lag, feature_idx
                    ]
            col_start += int(n_lag) + 1
        return output

    def _sparse_lagger(self, feature_matrix, censoring_i):
        censoring_i = min(max(int(censoring_i), 0), self._n_intervals)
        offsets = self._feature_offsets()
        coo = feature_matrix.tocoo()
        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []
        for row, col, value in zip(coo.row, coo.col, coo.data):
            max_lag = int(self.n_lags[col])
            for lag in range(max_lag + 1):
                out_row = int(row) + lag
                if out_row < censoring_i:
                    rows.append(out_row)
                    cols.append(int(offsets[col]) + lag)
                    data.append(float(value))
        return sps.csr_matrix(
            (data, (rows, cols)),
            shape=(self._n_intervals, self._n_output_features),
            dtype="float64",
        )
