import unittest

import numpy as np
from scipy.sparse import csr_matrix

from hawkes_tools.preprocessing import (
    FeaturesBinarizer,
    LongitudinalFeaturesLagger,
    LongitudinalFeaturesProduct,
    LongitudinalSamplesFilter,
)


class _FrameLike:
    def __init__(self, values, columns):
        self.values = values
        self.columns = columns


class TestFeaturesBinarizer(unittest.TestCase):
    def setUp(self):
        self.features = np.array(
            [
                [0.00902084, 0.54159776, 0.0, "z"],
                [0.46599565, -0.71875887, 0.0, 2.0],
                [0.52091721, -0.83803094, 1.0, 2.0],
                [0.47315496, 0.0730993, 1.0, 1.0],
                [0.08180209, -1.11447889, 0.0, 0.0],
                [0.45011727, -0.57931684, 0.0, 0.0],
                [2.04347947, -0.10127498, 1.0, 20.0],
                [-0.98909384, 1.36281079, 0.0, 0.0],
                [-0.30637613, -0.19147753, 1.0, 1.0],
                [0.27110903, 0.44583304, 0.0, 0.0],
            ],
            dtype=object,
        )
        self.columns = [
            "c:continuous",
            "a:continuous",
            "d:discrete",
            "b:discrete",
        ]
        self.df_features = _FrameLike(self.features, self.columns)
        self.default_expected_intervals = np.array(
            [
                [0, 3, 0, 4],
                [2, 0, 0, 2],
                [3, 0, 1, 2],
                [2, 2, 1, 1],
                [1, 0, 0, 0],
                [2, 1, 0, 0],
                [3, 2, 1, 3],
                [0, 3, 0, 0],
                [0, 1, 1, 1],
                [1, 2, 0, 0],
            ]
        )

    def _expected_one_hot(self):
        lengths = [4, 4, 2, 5]
        offsets = np.cumsum([0] + lengths[:-1])
        out = np.zeros((self.default_expected_intervals.shape[0], sum(lengths)))
        for row, values in enumerate(self.default_expected_intervals):
            for feature, value in enumerate(values):
                out[row, offsets[feature] + value] = 1.0
        return out

    def test_column_type_detection(self):
        expected_column_types = ["continuous", "continuous", "discrete", "discrete"]
        for i, expected_type in enumerate(expected_column_types):
            features_i = self.features[:, i]
            self.assertEqual(
                FeaturesBinarizer._detect_feature_type(features_i),
                expected_type,
            )
            self.assertEqual(
                FeaturesBinarizer._detect_feature_type(
                    features_i, continuous_threshold=7
                ),
                expected_type,
            )
            self.assertEqual(
                FeaturesBinarizer._detect_feature_type(
                    features_i,
                    detect_column_type="column_names",
                    feature_name=self.columns[i],
                ),
                expected_type,
            )

    def test_boundaries_detection(self):
        quantile = FeaturesBinarizer(
            method="quantile", n_cuts=3, detect_column_type="column_names"
        )
        np.testing.assert_array_almost_equal(
            quantile._get_boundaries(self.columns[0], self.features[:, 0], fit=True),
            np.array([-np.inf, 0.009021, 0.271109, 0.473155, np.inf]),
        )

        linspace = FeaturesBinarizer(
            method="linspace", n_cuts=3, detect_column_type="column_names"
        )
        np.testing.assert_array_almost_equal(
            linspace._get_boundaries(self.columns[0], self.features[:, 0], fit=True),
            np.array([-np.inf, -0.230951, 0.527193, 1.285336, np.inf]),
        )

    def test_assign_interval(self):
        binarizer = FeaturesBinarizer(
            method="quantile", n_cuts=3, detect_column_type="column_names"
        )
        for i, expected_interval in enumerate(self.default_expected_intervals.T):
            interval = binarizer._assign_interval(
                self.columns[i], self.features[:, i], fit=True
            )
            np.testing.assert_array_equal(expected_interval, interval)

    def test_fit_transform_and_blocks(self):
        binarizer = FeaturesBinarizer(method="quantile", n_cuts=3)
        binarized_df = binarizer.fit_transform(self.df_features)
        self.assertEqual(binarized_df.__class__, csr_matrix)
        np.testing.assert_array_equal(self._expected_one_hot(), binarized_df.toarray())
        np.testing.assert_array_equal(binarizer.blocks_start, np.array([0, 4, 8, 10]))
        np.testing.assert_array_equal(binarizer.blocks_length, np.array([4, 4, 2, 5]))

        binarized_array = binarizer.fit_transform(self.features)
        np.testing.assert_array_equal(self._expected_one_hot(), binarized_array.toarray())

    def test_binarizer_remove_first(self):
        binarizer = FeaturesBinarizer(method="quantile", n_cuts=3, remove_first=True)
        binarized_array = binarizer.fit_transform(self.features)
        expected = np.delete(self._expected_one_hot(), [0, 4, 8, 10], 1)
        np.testing.assert_array_equal(expected, binarized_array.toarray())


class TestLongitudinalFeaturesLagger(unittest.TestCase):
    @staticmethod
    def _with_int64_sparse_indices(matrix):
        matrix = matrix.copy()
        matrix.indices = matrix.indices.astype(np.int64)
        matrix.indptr = matrix.indptr.astype(np.int64)
        return matrix

    def setUp(self):
        self.features = [
            np.array([[0, 1, 0], [0, 0, 0], [0, 1, 1]], dtype="float64"),
            np.array([[1, 1, 1], [0, 0, 1], [1, 1, 0]], dtype="float64"),
        ]
        self.sparse_features = [csr_matrix(f) for f in self.features]
        self.censoring = np.array([2, 3], dtype="uint64")
        self.expected_output = [
            np.array(
                [
                    [0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0.0],
                ]
            ),
            np.array(
                [
                    [1, 0, 1, 0, 0, 1, 0],
                    [0, 1, 0, 1, 0, 1, 1],
                    [1, 0, 1, 0, 1, 0, 1.0],
                ]
            ),
        ]
        self.n_lags = np.array([1, 2, 1], dtype="uint64")

    def test_dense_pre_convolution(self):
        feat_prod, _, _ = LongitudinalFeaturesLagger(self.n_lags).fit_transform(
            self.features, censoring=self.censoring
        )
        np.testing.assert_equal(feat_prod, self.expected_output)

    def test_sparse_pre_convolution(self):
        feat_prod, _, _ = LongitudinalFeaturesLagger(self.n_lags).fit_transform(
            self.sparse_features, censoring=self.censoring
        )
        np.testing.assert_equal([f.toarray() for f in feat_prod], self.expected_output)

    def test_sparse_fit_accepts_int64_indices(self):
        int64_sparse_features = [
            self._with_int64_sparse_indices(feature) for feature in self.sparse_features
        ]
        feat_prod, _, _ = LongitudinalFeaturesLagger(self.n_lags).fit_transform(
            int64_sparse_features, censoring=self.censoring
        )
        np.testing.assert_equal([f.toarray() for f in feat_prod], self.expected_output)


class TestLongitudinalFeaturesProduct(unittest.TestCase):
    def setUp(self):
        self.finite_exposures = [
            np.array([[0, 1, 0], [0, 0, 0], [0, 1, 1]], dtype="float64"),
            np.array([[1, 1, 1], [0, 0, 1], [1, 1, 0]], dtype="float64"),
        ]
        self.sparse_finite_exposures = [csr_matrix(f) for f in self.finite_exposures]
        self.infinite_exposures = [
            np.array([[0, 1, 0], [0, 0, 0], [0, 0, 1]], dtype="float64"),
            np.array([[1, 1, 0], [0, 0, 1], [0, 0, 0]], dtype="float64"),
        ]

    def test_finite_features_product(self):
        expected_output = [
            np.array(
                [
                    [0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 1, 1, 0, 0, 1],
                ],
                dtype="float64",
            ),
            np.array(
                [
                    [1, 1, 1, 1, 1, 1],
                    [0, 0, 1, 0, 0, 0],
                    [1, 1, 0, 1, 0, 0],
                ],
                dtype="float64",
            ),
        ]
        for features in (self.finite_exposures, self.sparse_finite_exposures):
            feat_prod, _, _ = LongitudinalFeaturesProduct("finite").fit_transform(
                features
            )
            if hasattr(feat_prod[0], "toarray"):
                feat_prod = [f.toarray() for f in feat_prod]
            np.testing.assert_equal(feat_prod, expected_output)

    def test_sparse_infinite_features_product(self):
        expected_output = [
            np.array(
                [
                    [0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 1],
                ],
                dtype="float64",
            ),
            np.array(
                [
                    [1, 1, 0, 1, 0, 0],
                    [0, 0, 1, 0, 1, 1],
                    [0, 0, 0, 0, 0, 0],
                ],
                dtype="float64",
            ),
        ]
        sparse_feat = [csr_matrix(f) for f in self.infinite_exposures]
        feat_prod, _, _ = LongitudinalFeaturesProduct("infinite").fit_transform(
            sparse_feat
        )
        np.testing.assert_equal([f.toarray() for f in feat_prod], expected_output)


class TestLongitudinalSamplesFilter(unittest.TestCase):
    def setUp(self):
        self.features = [
            np.array([[0, 1, 0], [0, 0, 0], [0, 1, 1]], dtype="float64"),
            np.array([[1, 1, 1], [0, 0, 1], [1, 1, 0]], dtype="float64"),
            np.zeros((3, 3), dtype="float64"),
            np.array([[1, 1, 1], [0, 0, 1], [1, 1, 0]], dtype="float64"),
        ]
        self.sparse_features = [csr_matrix(f) for f in self.features]
        self.labels = [
            np.array([0, 0, 1], dtype="float64"),
            np.array([1, 0, 0], dtype="float64"),
            np.array([0, 1, 0], dtype="float64"),
            np.zeros((3,), dtype="float64"),
        ]
        self.censoring = np.array([2, 3, 3, 1], dtype="uint64")
        self.expected_output = (self.features[0:2], self.labels[0:2], self.censoring[0:2])

    def test_dense_filtering(self):
        output = LongitudinalSamplesFilter().fit_transform(
            self.features, self.labels, self.censoring
        )
        np.testing.assert_equal(output[0], self.expected_output[0])
        np.testing.assert_equal(output[1], self.expected_output[1])
        np.testing.assert_equal(output[2], self.expected_output[2])

    def test_sparse_filtering(self):
        output = LongitudinalSamplesFilter().fit_transform(
            self.sparse_features, self.labels, self.censoring
        )
        np.testing.assert_equal([out.toarray() for out in output[0]], self.expected_output[0])
        np.testing.assert_equal(output[1], self.expected_output[1])
        np.testing.assert_equal(output[2], self.expected_output[2])


if __name__ == "__main__":
    unittest.main()
