"""Feature binarization utilities."""

from __future__ import annotations

from typing import Any

import numpy as np
import scipy.sparse as sps

from ..base import BaseEstimator


def _is_dataframe_like(value) -> bool:
    return hasattr(value, "values") and hasattr(value, "columns")


def _sorted_unique(values: np.ndarray) -> np.ndarray:
    try:
        return np.unique(values)
    except TypeError:
        uniques: list[Any] = []
        for value in values:
            if not any(value == existing for existing in uniques):
                uniques.append(value)

        def sort_key(value):
            try:
                return (0, float(value), str(type(value)), str(value))
            except (TypeError, ValueError):
                return (1, str(value))

        return np.asarray(sorted(uniques, key=sort_key), dtype=object)


class _SimpleOneHotEncoder:
    """Small one-hot encoder with the attributes used internally."""

    def fit(self, X):
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("X must be two-dimensional")
        self.categories_ = [_sorted_unique(X[:, j]) for j in range(X.shape[1])]
        return self

    def transform(self, X):
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("X must be two-dimensional")
        if X.shape[1] != len(self.categories_):
            raise ValueError("X has a different number of features than fit data")

        offsets = [0]
        for categories in self.categories_:
            offsets.append(offsets[-1] + len(categories))

        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []
        for feature_idx, categories in enumerate(self.categories_):
            mapping = {category: idx for idx, category in enumerate(categories)}
            for row_idx, value in enumerate(X[:, feature_idx]):
                if value not in mapping:
                    raise ValueError(
                        "Found unknown category %r in column %d during transform"
                        % (value, feature_idx)
                    )
                rows.append(row_idx)
                cols.append(offsets[feature_idx] + mapping[value])
                data.append(1.0)

        return sps.csr_matrix(
            (data, (rows, cols)), shape=(X.shape[0], offsets[-1]), dtype=float
        )


class FeaturesBinarizer(BaseEstimator):
    """Transform continuous and discrete features into one-hot binary columns."""

    def __init__(
        self,
        method: str = "quantile",
        n_cuts: int = 10,
        detect_column_type: str = "auto",
        remove_first: bool = False,
        bins_boundaries: dict[str, np.ndarray] | None = None,
    ):
        self.method = method
        self.n_cuts = n_cuts
        self.detect_column_type = detect_column_type
        self.remove_first = remove_first
        self.bins_boundaries = bins_boundaries
        self.reset()

    def reset(self):
        self.one_hot_encoder = _SimpleOneHotEncoder()
        self.mapper: dict[str, dict[Any, int]] = {}
        self.feature_type: dict[str, str] = {}
        self._fitted = False
        if self.method != "given":
            self.bins_boundaries = {}

    @property
    def boundaries(self):
        if not self._fitted:
            raise ValueError("cannot get bins_boundaries if object has not been fitted")
        return self.bins_boundaries

    @property
    def blocks_start(self):
        if not self._fitted:
            raise ValueError("cannot get blocks_start if object has not been fitted")
        return self._get_feature_indices()[:-1]

    @property
    def blocks_length(self):
        if not self._fitted:
            raise ValueError("cannot get blocks_length if object has not been fitted")
        return self._get_n_values()

    @staticmethod
    def cast_to_array(X):
        if _is_dataframe_like(X):
            columns = list(X.columns)
            X = np.asarray(X.values)
        else:
            X = np.asarray(X)
            columns = [str(i) for i in range(X.shape[1])]
        return X, columns

    def fit(self, X):
        self.reset()
        X, columns = FeaturesBinarizer.cast_to_array(X)
        categorical_X = np.empty(X.shape, dtype=object)
        for i, column in enumerate(columns):
            categorical_X[:, i] = self._assign_interval(column, X[:, i], fit=True)

        self.one_hot_encoder.fit(categorical_X)
        self._fitted = True
        return self

    def transform(self, X):
        X, columns = FeaturesBinarizer.cast_to_array(X)
        categorical_X = np.empty(X.shape, dtype=object)
        for i, column in enumerate(columns):
            categorical_X[:, i] = self._assign_interval(column, X[:, i], fit=False)

        binarized_X = self.one_hot_encoder.transform(categorical_X)
        if self.remove_first:
            feature_indices = self._get_feature_indices()
            mask = np.ones(binarized_X.shape[1], dtype=bool)
            mask[feature_indices[:-1]] = False
            binarized_X = binarized_X[:, mask]
        return binarized_X

    def fit_transform(self, X, y=None, **kwargs):
        del y, kwargs
        self.fit(X)
        return self.transform(X)

    @staticmethod
    def _detect_feature_type(
        feature,
        detect_column_type: str = "auto",
        feature_name: str | None = None,
        continuous_threshold: int | float | str = "auto",
    ) -> str:
        if detect_column_type == "column_names":
            if feature_name is None:
                raise ValueError(
                    "feature_name must be set in order to use 'column_names' detection type"
                )
            if feature_name.endswith(":continuous"):
                return "continuous"
            if feature_name.endswith(":discrete"):
                return "discrete"
            raise ValueError(
                "feature name '%s' should end with ':continuous' or ':discrete'"
                % feature_name
            )

        if detect_column_type != "auto":
            raise ValueError("detect_type should be one of 'column_names' or 'auto'")

        if continuous_threshold == "auto":
            threshold = 15 if len(feature) > 30 else len(feature) / 2
        else:
            threshold = continuous_threshold

        uniques = _sorted_unique(np.asarray(feature))
        if len(uniques) > threshold:
            try:
                np.asarray(uniques).astype(float)
                return "continuous"
            except ValueError:
                return "discrete"
        return "discrete"

    def _get_feature_type(self, feature_name, feature, fit=False):
        if fit:
            feature_type = FeaturesBinarizer._detect_feature_type(
                feature,
                feature_name=feature_name,
                detect_column_type=self.detect_column_type,
            )
            self.feature_type[feature_name] = feature_type
        elif self._fitted:
            feature_type = self.feature_type[feature_name]
        else:
            raise ValueError("cannot call method with fit=False if object has not been fitted")
        return feature_type

    @staticmethod
    def _detect_boundaries(feature, n_cuts, method):
        feature = np.asarray(feature, dtype=float)
        if method == "quantile":
            quantile_cuts = np.linspace(0, 100, n_cuts + 2)
            try:
                boundaries = np.percentile(feature, quantile_cuts, method="nearest")
            except TypeError:
                boundaries = np.percentile(
                    feature, quantile_cuts, interpolation="nearest"
                )
            boundaries = np.unique(boundaries)
        elif method == "linspace":
            boundaries = np.linspace(np.min(feature), np.max(feature), n_cuts + 2)
        else:
            raise ValueError("Method '%s' should be 'quantile' or 'linspace'" % method)

        boundaries[0] = -np.inf
        boundaries[-1] = np.inf
        return boundaries

    def _get_boundaries(self, feature_name, feature, fit=False):
        if fit:
            if self.method == "given":
                if self.bins_boundaries is None:
                    raise ValueError("bins_boundaries required when `method` equals 'given'")
                if not isinstance(self.bins_boundaries.get(feature_name), np.ndarray):
                    raise ValueError("feature %s not found in bins_boundaries" % feature_name)
                boundaries = self.bins_boundaries[feature_name]
            else:
                boundaries = FeaturesBinarizer._detect_boundaries(
                    feature, self.n_cuts, self.method
                )
                self.bins_boundaries[feature_name] = boundaries
        elif self._fitted:
            boundaries = self.bins_boundaries[feature_name]
        else:
            raise ValueError("cannot call method with fit=False if object has not been fitted")
        return boundaries

    def _categorical_to_interval(self, feature, feature_name, fit=False):
        if fit:
            uniques = _sorted_unique(np.asarray(feature))
            mapper = {category: interval for interval, category in enumerate(uniques)}
            self.mapper[feature_name] = mapper
        else:
            mapper = self.mapper[feature_name]

        def category_to_interval(category):
            if category in mapper:
                return mapper[category]
            return len(mapper) + 1

        return np.vectorize(category_to_interval)(feature)

    def _assign_interval(self, feature_name, feature, fit=False):
        feature_type = self._get_feature_type(feature_name, feature, fit)
        if feature_type == "continuous":
            feature = np.asarray(feature, dtype=float)
            boundaries = self._get_boundaries(feature_name, feature, fit)
            return np.searchsorted(boundaries, feature, side="left") - 1
        return self._categorical_to_interval(feature, feature_name, fit=fit)

    def _get_n_values(self):
        return [len(categories) for categories in self.one_hot_encoder.categories_]

    def _get_feature_indices(self):
        feature_indices = [0]
        for categories in self.one_hot_encoder.categories_:
            feature_indices.append(feature_indices[-1] + len(categories))
        return np.asarray(feature_indices)
