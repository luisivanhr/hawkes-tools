"""Cox risk-set regressions for extreme scores and Breslow ties."""

import sys
import unittest
from pathlib import Path

import numpy as np
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hawkes_tools.survival import ModelCoxRegPartialLik


def direct_breslow_loss_grad(features, times, censoring, coeffs):
    """Independent oracle that explicitly constructs each failure's risk set."""
    scores = features @ coeffs
    failures = np.flatnonzero(censoring)
    loss = 0.0
    gradient = np.zeros(features.shape[1])
    for failure in failures:
        at_risk = times >= times[failure]
        risk_scores = scores[at_risk]
        shift = np.max(risk_scores)
        weights = np.exp(risk_scores - shift)
        denominator = np.sum(weights)
        loss += (shift - scores[failure]) + np.log(denominator)
        gradient += (weights / denominator) @ features[at_risk] - features[failure]
    return loss / failures.size, gradient / failures.size


class CoxRiskStabilityTest(unittest.TestCase):
    def test_finite_extreme_features_do_not_overflow_before_later_rescaling(self):
        coeffs = np.array([0.0, 1.0])
        for values in (
            [1e308, 1e308],
            [-1e308, -1e308],
            [1e308, 1e308, -1e308, -1e308],
        ):
            features = np.column_stack((values + [0.0], [0.0] * len(values) + [700.0]))
            times = np.arange(features.shape[0], 0, -1, dtype=float)
            censoring = np.zeros(features.shape[0], dtype=np.ushort)
            censoring[-1] = 1
            with np.errstate(divide="raise", invalid="raise", over="raise"):
                expected_loss, expected_gradient = direct_breslow_loss_grad(
                    features, times, censoring, coeffs
                )
            if values == [1e308, 1e308]:
                np.testing.assert_allclose(
                    expected_gradient, [19719.353087519543, 0.0], rtol=2e-15, atol=0
                )
            for data in (features, sparse.csr_matrix(features)):
                with self.subTest(values=values, sparse=sparse.issparse(data)):
                    model = ModelCoxRegPartialLik().fit(data, times, censoring)
                    with np.errstate(divide="raise", invalid="raise", over="raise"):
                        loss, gradient = model.loss_and_grad(coeffs)
                        separate_loss = model.loss(coeffs)
                        separate_gradient = model.grad(coeffs)
                    self.assertTrue(np.isfinite(loss))
                    self.assertTrue(np.all(np.isfinite(gradient)))
                    self.assertAlmostEqual(loss, expected_loss, places=14)
                    np.testing.assert_allclose(
                        gradient, expected_gradient, rtol=2e-14, atol=1e-10
                    )
                    self.assertEqual(separate_loss, loss)
                    np.testing.assert_allclose(
                        separate_gradient, gradient, rtol=2e-14, atol=1e-10
                    )

    def test_joint_evaluation_matches_independent_risk_sets_and_separate_calls(self):
        features = np.array([[0.4, 2.0], [-0.2, -3.0], [0.8, 1.0], [1.0, -1.0]])
        times = np.array([3.0, 2.0, 2.0, 1.0])
        censoring = np.array([1, 1, 0, 1], dtype=np.ushort)
        coeffs = np.array([0.2, -0.1])
        for dtype, tolerance in ((np.float64, 2e-15), (np.float32, 1e-6)):
            typed_coeffs = coeffs.astype(dtype)
            for size in (4, 1):
                typed_features = features[:size].astype(dtype)
                typed_times = times[:size].astype(dtype)
                typed_censoring = censoring[:size]
                expected_loss, expected_gradient = direct_breslow_loss_grad(
                    typed_features, typed_times, typed_censoring, typed_coeffs
                )
                for data in (typed_features, sparse.csr_matrix(typed_features)):
                    with self.subTest(dtype=dtype, size=size, sparse=sparse.issparse(data)):
                        model = ModelCoxRegPartialLik().fit(data, typed_times, typed_censoring)
                        output = np.full(coeffs.size, np.nan, dtype=dtype)
                        loss, gradient = model.loss_and_grad(typed_coeffs, out=output)
                        self.assertIs(gradient, output)
                        self.assertEqual(gradient.dtype, dtype)
                        self.assertAlmostEqual(loss, expected_loss, delta=tolerance)
                        np.testing.assert_allclose(
                            gradient, expected_gradient, atol=tolerance, rtol=0
                        )
                        self.assertAlmostEqual(loss, model.loss(typed_coeffs), delta=tolerance)
                        np.testing.assert_allclose(
                            gradient, model.grad(typed_coeffs), atol=tolerance, rtol=0
                        )

    def test_constant_maximum_finite_feature_cancels_from_tied_risk_sets(self):
        maximum = np.finfo(float).max
        for value in (maximum, -maximum):
            features = np.full((3, 1), value)
            times = np.ones(3)
            censoring = np.ones(3, dtype=np.ushort)
            for data in (features, sparse.csr_matrix(features)):
                with self.subTest(value=value, sparse=sparse.issparse(data)):
                    model = ModelCoxRegPartialLik().fit(data, times, censoring)
                    with np.errstate(divide="raise", invalid="raise", over="raise"):
                        loss, gradient = model.loss_and_grad(np.zeros(1))
                    self.assertAlmostEqual(loss, np.log(3.0), places=14)
                    np.testing.assert_array_equal(gradient, np.zeros(1))

    def test_tiny_risk_probability_retains_representable_weighted_feature(self):
        # exp(-750) underflows, but its product with a finite large feature does not.
        # Splitting the exponential gives an independent finite analytical value.
        expected_component = 2.0 * ((1e308 * np.exp(-375.0)) * np.exp(-375.0))
        for sign in (1.0, -1.0):
            features = np.array([[sign * 1e308, 0.0], [sign * 1e308, 0.0], [0.0, 750.0]])
            censoring = np.array([0, 0, 1], dtype=np.ushort)
            coeffs = np.array([0.0, 1.0])
            expected_gradient = np.array([sign * expected_component, 0.0])
            for times, order in (
                (np.array([3.0, 2.0, 1.0]), np.arange(3)),
                (np.ones(3), np.arange(3)),
                (np.ones(3), np.array([2, 1, 0])),
                (np.ones(3), np.array([1, 2, 0])),
            ):
                for data in (features[order], sparse.csr_matrix(features[order])):
                    with self.subTest(
                        sign=sign, times=times.tolist(), order=order.tolist(),
                        sparse=sparse.issparse(data),
                    ):
                        model = ModelCoxRegPartialLik().fit(data, times[order], censoring[order])
                        with np.errstate(divide="raise", invalid="raise", over="raise"):
                            loss, gradient = model.loss_and_grad(coeffs)
                        self.assertEqual(loss, 0.0)
                        np.testing.assert_allclose(gradient, expected_gradient, rtol=2e-13, atol=0)

        # A subnormal rescale can lose precision before its accumulated mass
        # brings the final probability back into the normal floating-point range.
        n_censored = 100000
        features = np.zeros((n_censored + 1, 2))
        features[:-1, 0] = 1e308
        features[-1, 1] = 718.0
        times = np.arange(n_censored + 1, 0, -1, dtype=float)
        censoring = np.zeros(n_censored + 1, dtype=np.ushort)
        censoring[-1] = 1
        expected_component = n_censored * ((1e308 * np.exp(-359.0)) * np.exp(-359.0))
        model = ModelCoxRegPartialLik().fit(features, times, censoring)
        with np.errstate(divide="raise", invalid="raise", over="raise"):
            loss, gradient = model.loss_and_grad(np.array([0.0, 1.0]))
        self.assertEqual(loss, 0.0)
        np.testing.assert_allclose(gradient, [expected_component, 0.0], rtol=2e-13, atol=0)

    def test_later_global_maximum_does_not_underflow_earlier_risk_set(self):
        features = np.array([[0.0], [1000.0]])
        times = np.array([2.0, 1.0])
        censoring = np.array([1, 1], dtype=np.ushort)
        for data in (features, sparse.csr_matrix(features)):
            with self.subTest(sparse=sparse.issparse(data)):
                model = ModelCoxRegPartialLik().fit(data, times, censoring)
                with np.errstate(divide="raise", invalid="raise", over="raise"):
                    loss = model.loss(np.ones(1))
                    gradient = model.grad(np.ones(1))
                self.assertEqual(loss, 0.0)
                np.testing.assert_array_equal(gradient, np.zeros(1))

    def test_tied_failures_include_censored_observations_in_breslow_denominator(self):
        features = np.array([[1000.0], [999.0], [1000.0]])
        times = np.ones(3)
        censoring = np.array([1, 1, 0], dtype=np.ushort)
        expected_loss = np.log(2.0 + np.exp(-1.0)) + 0.5
        expected_gradient = 0.5 - np.exp(-1.0) / (2.0 + np.exp(-1.0))
        for order in (np.array([0, 1, 2]), np.array([2, 1, 0])):
            for data in (features[order], sparse.csr_matrix(features[order])):
                with self.subTest(order=order.tolist(), sparse=sparse.issparse(data)):
                    model = ModelCoxRegPartialLik().fit(data, times, censoring[order])
                    self.assertAlmostEqual(model.loss(np.ones(1)), expected_loss, places=12)
                    np.testing.assert_allclose(
                        model.grad(np.ones(1)), [expected_gradient], atol=2e-13, rtol=0
                    )

    def test_extreme_prefixes_match_independent_risk_sets_and_finite_differences(self):
        features = np.array([
            [-1000.0, -2.0], [-999.8, 1.0], [0.0, 4.0], [0.2, -1.0],
            [1000.0, 3.0], [999.7, -4.0], [1000.1, 2.0],
        ])
        times = np.array([4.0, 4.0, 3.0, 3.0, 2.0, 2.0, 1.0])
        censoring = np.array([1, 0, 1, 1, 0, 1, 1], dtype=np.ushort)
        coeffs = np.array([0.9, 0.3])
        expected_loss, expected_gradient = direct_breslow_loss_grad(
            features, times, censoring, coeffs
        )
        for order in (np.arange(7), np.array([4, 1, 6, 2, 0, 5, 3])):
            for data in (features[order], sparse.csr_matrix(features[order])):
                with self.subTest(order=order.tolist(), sparse=sparse.issparse(data)):
                    model = ModelCoxRegPartialLik().fit(data, times[order], censoring[order])
                    with np.errstate(divide="raise", invalid="raise", over="raise"):
                        self.assertAlmostEqual(model.loss(coeffs), expected_loss, places=12)
                        gradient = model.grad(coeffs)
                    np.testing.assert_allclose(gradient, expected_gradient, atol=2e-13, rtol=0)
                    numerical_gradient = np.zeros_like(coeffs)
                    for index in range(coeffs.size):
                        step = np.zeros_like(coeffs)
                        step[index] = 1e-5
                        numerical_gradient[index] = (
                            model.loss(coeffs + step) - model.loss(coeffs - step)
                        ) / (2e-5)
                    np.testing.assert_allclose(gradient, numerical_gradient, atol=2e-8, rtol=1e-7)

    def test_common_large_score_offset_preserves_loss_and_gradient(self):
        features = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, -1.0]])
        model = ModelCoxRegPartialLik().fit(features, np.ones(3), np.ones(3, dtype=np.ushort))
        reference_coeffs = np.zeros(2)
        shifted_coeffs = np.array([1e16, 0.0])
        self.assertAlmostEqual(model.loss(shifted_coeffs), np.log(3.0), places=14)
        self.assertEqual(model.loss(shifted_coeffs), model.loss(reference_coeffs))
        np.testing.assert_allclose(model.grad(shifted_coeffs), model.grad(reference_coeffs))


if __name__ == "__main__":
    unittest.main()
