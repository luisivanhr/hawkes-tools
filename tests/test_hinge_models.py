import unittest

import numpy as np
from scipy import sparse

from hawkes_tools.linear_model import (
    ModelHinge,
    ModelQuadraticHinge,
    ModelSmoothedHinge,
)


def finite_difference_grad(model, coeffs, eps=1e-6):
    grad = np.zeros_like(coeffs, dtype=float)
    for i in range(coeffs.size):
        step = np.zeros_like(coeffs, dtype=float)
        step[i] = eps
        grad[i] = (model.loss(coeffs + step) - model.loss(coeffs - step)) / (2 * eps)
    return grad


class HingeModelsTest(unittest.TestCase):
    def setUp(self):
        self.X = np.array(
            [
                [1.0, -0.5, 0.25],
                [0.3, 1.2, -0.7],
                [-0.8, 0.4, 1.5],
                [1.5, 0.2, -1.0],
                [-0.2, -1.3, 0.6],
            ]
        )
        self.y = np.array([-1.0, 1.0, -1.0, 1.0, -1.0])
        self.beta = np.array([0.2, -0.1, 0.3, 0.4])

    def test_hinge_dense_and_sparse_match_source_formula(self):
        model = ModelHinge(fit_intercept=True).fit(self.X, self.y)
        sparse_model = ModelHinge(fit_intercept=True).fit(sparse.csr_matrix(self.X), self.y)

        scores = np.hstack([self.X, np.ones((self.X.shape[0], 1))]) @ self.beta
        margins = self.y * scores
        expected_loss = np.mean(np.maximum(0.0, 1.0 - margins))
        expected_grad = (
            np.hstack([self.X, np.ones((self.X.shape[0], 1))]).T
            @ np.where(margins < 1.0, -self.y, 0.0)
            / self.X.shape[0]
        )

        self.assertAlmostEqual(model.loss(self.beta), expected_loss)
        np.testing.assert_allclose(model.grad(self.beta), expected_grad)
        self.assertAlmostEqual(sparse_model.loss(self.beta), model.loss(self.beta))
        np.testing.assert_allclose(sparse_model.grad(self.beta), model.grad(self.beta))

    def test_quadratic_hinge_gradient_lipschitz_and_intercept_equivalence(self):
        model = ModelQuadraticHinge(fit_intercept=True).fit(self.X, self.y)
        np.testing.assert_allclose(
            model.grad(self.beta),
            finite_difference_grad(model, self.beta),
            atol=1e-6,
        )

        hardcoded_intercept = np.hstack([self.X, np.ones((self.X.shape[0], 1))])
        no_intercept = ModelQuadraticHinge(fit_intercept=False).fit(
            hardcoded_intercept, self.y
        )
        np.testing.assert_allclose(model.grad(self.beta), no_intercept.grad(self.beta))

        row_norms = np.sum(self.X * self.X, axis=1) + 1.0
        self.assertAlmostEqual(model.get_lip_mean(), float(np.mean(row_norms)))
        self.assertAlmostEqual(model.get_lip_max(), float(np.max(row_norms)))
        expected_best = (
            np.linalg.svd(self.X, full_matrices=False, compute_uv=False)[0] ** 2
            + 1.0
        ) / self.X.shape[0]
        self.assertAlmostEqual(model.get_lip_best(), expected_best)

    def test_smoothed_hinge_gradient_lipschitz_and_smoothness_validation(self):
        model = ModelSmoothedHinge(fit_intercept=True, smoothness=0.2).fit(
            self.X, self.y
        )
        np.testing.assert_allclose(
            model.grad(self.beta),
            finite_difference_grad(model, self.beta),
            atol=1e-6,
        )
        self.assertAlmostEqual(model.get_smoothness(), 0.2)

        quad = ModelQuadraticHinge(fit_intercept=True).fit(self.X, self.y)
        self.assertAlmostEqual(model.get_lip_mean(), 5.0 * quad.get_lip_mean())
        self.assertAlmostEqual(model.get_lip_max(), 5.0 * quad.get_lip_max())
        self.assertAlmostEqual(model.get_lip_best(), 5.0 * quad.get_lip_best())

        model.smoothness = 0.75
        self.assertAlmostEqual(model.get_smoothness(), 0.75)
        for bad in (0.0, -1.0, 1.2):
            with self.assertRaisesRegex(RuntimeError, "smoothness should be between"):
                ModelSmoothedHinge(smoothness=bad)
            with self.assertRaisesRegex(RuntimeError, "smoothness should be between"):
                model.smoothness = bad


if __name__ == "__main__":
    unittest.main()
