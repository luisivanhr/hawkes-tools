"""Longitudinal product feature construction."""

from __future__ import annotations

from copy import deepcopy
from itertools import combinations

import numpy as np
import scipy.sparse as sps
from scipy.special import comb

from .base import LongitudinalPreprocessor
from .utils import check_longitudinal_features_consistency


class LongitudinalFeaturesProduct(LongitudinalPreprocessor):
    """Add pairwise product columns to longitudinal feature matrices."""

    def __init__(self, exposure_type: str = "infinite", n_jobs: int = -1):
        super().__init__(n_jobs=n_jobs)
        if exposure_type not in ["infinite", "finite"]:
            raise ValueError(
                "exposure_type should be either 'infinite' or 'finite', not %s"
                % exposure_type
            )
        self.exposure_type = exposure_type
        self._reset()

    def _reset(self):
        self._mapper = {}
        self._n_init_features = None
        self._n_output_features = None
        self._n_intervals = None
        self._fitted = False

    @property
    def mapper(self):
        if not self._fitted:
            raise ValueError("cannot get mapper if object has not been fitted.")
        return deepcopy(self._mapper)

    def fit(self, features, labels=None, censoring=None):
        del labels, censoring
        self._reset()
        base_shape = features[0].shape
        features = check_longitudinal_features_consistency(features, base_shape, "float64")
        del features
        n_intervals, n_init_features = base_shape
        if n_init_features < 2:
            raise ValueError("There should be at least two features to compute product features.")

        self._n_init_features = n_init_features
        self._n_intervals = n_intervals
        self._mapper = {
            i + n_init_features: pair
            for i, pair in enumerate(combinations(range(n_init_features), 2))
        }
        self._n_output_features = int(n_init_features + comb(n_init_features, 2))
        self._fitted = True
        return self

    def transform(self, features, labels=None, censoring=None):
        if not self._fitted:
            raise ValueError("cannot transform before fit")
        base_shape = (self._n_intervals, self._n_init_features)
        features = check_longitudinal_features_consistency(features, base_shape, "float64")
        if self.exposure_type == "finite":
            X_with_products = self._finite_exposure_products(features)
        elif self.exposure_type == "infinite":
            X_with_products = self._infinite_exposure_products(features)
        else:
            raise ValueError(
                "exposure_type should be either 'infinite' or 'finite', not %s"
                % self.exposure_type
            )
        return X_with_products, labels, censoring

    def _finite_exposure_products(self, features):
        if sps.issparse(features[0]):
            return [self._sparse_finite_product(arr) for arr in features]
        return [self._dense_finite_product(arr) for arr in features]

    def _infinite_exposure_products(self, features):
        if not sps.issparse(features[0]):
            raise ValueError(
                "Infinite exposures should be stored in sparse matrices as this "
                "hypothesis induces sparsity in the feature matrix."
            )
        return [self._sparse_infinite_product(arr) for arr in features]

    def _dense_finite_product(self, feat_mat):
        feat = [feat_mat]
        feat.extend(
            (feat_mat[:, i] * feat_mat[:, j]).reshape((-1, 1))
            for i, j in self._mapper.values()
        )
        return np.hstack(feat)

    def _sparse_finite_product(self, feat_mat):
        feat_mat = feat_mat.tocsr()
        feat = [feat_mat]
        feat.extend(feat_mat[:, i].multiply(feat_mat[:, j]) for i, j in self.mapper.values())
        return sps.hstack(feat, format="csr")

    def _sparse_infinite_product(self, feat_mat):
        feat_mat = feat_mat.tocsr()
        coo = feat_mat.tocoo()
        first_by_col: dict[int, tuple[int, float]] = {}
        for row, col, value in zip(coo.row, coo.col, coo.data):
            if value == 0:
                continue
            col = int(col)
            row = int(row)
            if col not in first_by_col or row < first_by_col[col][0]:
                first_by_col[col] = (row, float(value))

        columns = [feat_mat]
        for i, j in self._mapper.values():
            if i in first_by_col and j in first_by_col:
                row = max(first_by_col[i][0], first_by_col[j][0])
                value = first_by_col[i][1] * first_by_col[j][1]
                product_col = sps.csr_matrix(
                    ([value], ([row], [0])), shape=(self._n_intervals, 1)
                )
            else:
                product_col = sps.csr_matrix((self._n_intervals, 1), dtype="float64")
            columns.append(product_col)
        return sps.hstack(columns, format="csr")
