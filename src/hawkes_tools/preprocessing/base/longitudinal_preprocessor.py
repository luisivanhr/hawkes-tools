"""Shared base class for longitudinal preprocessing transforms."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ...base import BaseEstimator


class LongitudinalPreprocessor(ABC, BaseEstimator):
    """Abstract base class for longitudinal data preprocessors."""

    def __init__(self, n_jobs: int = -1):
        self.n_jobs = n_jobs

    @abstractmethod
    def fit(self, features: list[Any], labels: Any = None, censoring: Any = None):
        """Fit the preprocessor."""

    @abstractmethod
    def transform(self, features: list[Any], labels: Any = None, censoring: Any = None):
        """Transform longitudinal arrays."""

    def fit_transform(self, features: list[Any], labels: Any = None, censoring: Any = None):
        self.fit(features, labels, censoring)
        return self.transform(features, labels, censoring)
