"""Longitudinal sample filtering."""

from __future__ import annotations

from operator import itemgetter

import numpy as np

from .base import LongitudinalPreprocessor


class LongitudinalSamplesFilter(LongitudinalPreprocessor):
    """Filter samples with empty features or all-zero labels."""

    def __init__(self, n_jobs: int = -1):
        super().__init__(n_jobs=n_jobs)
        self._mask = None
        self._n_active_patients = None
        self._n_patients = None

    def fit(self, features, labels, censoring):
        del censoring
        nnz = [len(np.nonzero(arr)[0]) > 0 for arr in labels]
        self._mask = [
            idx for idx, feat in enumerate(features) if feat.sum() > 0 and nnz[idx]
        ]
        self._n_active_patients = len(self._mask)
        self._n_patients = len(features)
        return self

    def transform(self, features, labels, censoring):
        if self._n_active_patients <= 1:
            raise ValueError(
                "There should be more than one positive sample per batch with "
                "nonzero_features. Please check the input data."
            )
        if self._n_active_patients < self._n_patients:
            features_filter = itemgetter(*self._mask)
            features = features_filter(features)
            labels = features_filter(labels)
            censoring = censoring[self._mask]
        return features, labels, censoring
