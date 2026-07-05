import sys
import unittest
from pathlib import Path

import numpy as np
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hawkes_tools.robust import (
    ModelAbsoluteRegression,
    ModelEpsilonInsensitive,
    ModelHuber,
    ModelModifiedHuber,
)


def finite_difference_grad(model, coeffs, eps=1e-6):
    grad = np.zeros_like(coeffs, dtype=float)
    for i in range(coeffs.size):
        step = np.zeros_like(coeffs, dtype=float)
        step[i] = eps
        grad[i] = (model.loss(coeffs + step) - model.loss(coeffs - step)) / (2.0 * eps)
    return grad


class RobustLossModelsTest(unittest.TestCase):
    def setUp(self):
        self.X = np.array(
            [
                [1.0, -0.5, 0.25],
                [0.3, 1.2, -0.7],
                [-0.8, 0.4, 1.5],
                [1.5, 0.2, -1.0],
                [-0.2, -1.3, 0.6],
            ],
            dtype=float,
        )
        self.y_reg = np.array([0.1, -0.8, 0.6, 1.7, -1.1], dtype=float)
        self.y_cls = np.array([-1.0, 1.0, -1.0, 1.0, -1.0], dtype=float)
        self.beta = np.array([0.4, -0.35, 0.2, 0.15], dtype=float)

    def test_public_exports_cover_reference_robust_loss_names(self):
        import hawkes_tools.robust as robust

        expected = {
            "ModelAbsoluteRegression",
            "ModelEpsilonInsensitive",
            "ModelHuber",
            "ModelLinRegWithIntercepts",
            "ModelModifiedHuber",
            "RobustLinearRegression",
            "features_normal_cov_toeplitz",
            "std_iqr",
            "std_mad",
        }
        self.assertEqual(set(), expected - set(robust.__all__))

    def test_huber_formula_gradient_sparse_parity_and_lipschitz(self):
        threshold = 0.7
        model = ModelHuber(fit_intercept=True, threshold=threshold).fit(self.X, self.y_reg)
        sparse_model = ModelHuber(fit_intercept=True, threshold=threshold).fit(sparse.csr_matrix(self.X), self.y_reg)

        scores = np.hstack([self.X, np.ones((self.X.shape[0], 1))]) @ self.beta
        residual = scores - self.y_reg
        expected_loss = np.mean(
            np.where(
                np.abs(residual) <= threshold,
                0.5 * residual * residual,
                threshold * (np.abs(residual) - 0.5 * threshold),
            )
        )
        expected_grad = (
            np.hstack([self.X, np.ones((self.X.shape[0], 1))]).T
            @ np.clip(residual, -threshold, threshold)
            / self.X.shape[0]
        )

        self.assertAlmostEqual(model.loss(self.beta), expected_loss)
        np.testing.assert_allclose(model.grad(self.beta), expected_grad)
        np.testing.assert_allclose(model.grad(self.beta), finite_difference_grad(model, self.beta), atol=1e-6)
        self.assertAlmostEqual(sparse_model.loss(self.beta), model.loss(self.beta))
        np.testing.assert_allclose(sparse_model.grad(self.beta), model.grad(self.beta))
        self.assertAlmostEqual(model.get_threshold(), threshold)

        row_norms = np.sum(self.X * self.X, axis=1) + 1.0
        self.assertAlmostEqual(model.get_lip_mean(), float(np.mean(row_norms)))
        self.assertAlmostEqual(model.get_lip_max(), float(np.max(row_norms)))
        expected_best = (np.linalg.svd(self.X, full_matrices=False, compute_uv=False)[0] ** 2 + 1.0) / self.X.shape[0]
        self.assertAlmostEqual(model.get_lip_best(), expected_best)

    def test_absolute_regression_formula_gradient_and_intercept_equivalence(self):
        model = ModelAbsoluteRegression(fit_intercept=True).fit(self.X, self.y_reg)
        sparse_model = ModelAbsoluteRegression(fit_intercept=True).fit(sparse.csr_matrix(self.X), self.y_reg)

        scores = np.hstack([self.X, np.ones((self.X.shape[0], 1))]) @ self.beta
        residual = scores - self.y_reg
        expected_loss = np.mean(np.abs(residual))
        expected_grad = (
            np.hstack([self.X, np.ones((self.X.shape[0], 1))]).T
            @ np.sign(residual)
            / self.X.shape[0]
        )

        self.assertAlmostEqual(model.loss(self.beta), expected_loss)
        np.testing.assert_allclose(model.grad(self.beta), expected_grad)
        np.testing.assert_allclose(model.grad(self.beta), finite_difference_grad(model, self.beta), atol=1e-6)
        self.assertAlmostEqual(sparse_model.loss(self.beta), model.loss(self.beta))
        np.testing.assert_allclose(sparse_model.grad(self.beta), model.grad(self.beta))

        hardcoded_intercept = np.hstack([self.X, np.ones((self.X.shape[0], 1))])
        no_intercept = ModelAbsoluteRegression(fit_intercept=False).fit(hardcoded_intercept, self.y_reg)
        np.testing.assert_allclose(model.grad(self.beta), no_intercept.grad(self.beta))

    def test_epsilon_insensitive_formula_gradient_and_threshold_validation(self):
        threshold = 0.25
        model = ModelEpsilonInsensitive(fit_intercept=True, threshold=threshold).fit(self.X, self.y_reg)
        sparse_model = ModelEpsilonInsensitive(fit_intercept=True, threshold=threshold).fit(sparse.csr_matrix(self.X), self.y_reg)

        scores = np.hstack([self.X, np.ones((self.X.shape[0], 1))]) @ self.beta
        residual = scores - self.y_reg
        active = np.abs(residual) > threshold
        expected_loss = np.mean(np.where(active, np.abs(residual) - threshold, 0.0))
        expected_grad = (
            np.hstack([self.X, np.ones((self.X.shape[0], 1))]).T
            @ np.where(active, np.sign(residual), 0.0)
            / self.X.shape[0]
        )

        self.assertAlmostEqual(model.loss(self.beta), expected_loss)
        np.testing.assert_allclose(model.grad(self.beta), expected_grad)
        np.testing.assert_allclose(model.grad(self.beta), finite_difference_grad(model, self.beta), atol=1e-6)
        self.assertAlmostEqual(sparse_model.loss(self.beta), model.loss(self.beta))
        np.testing.assert_allclose(sparse_model.grad(self.beta), model.grad(self.beta))

        model.threshold = 1.3
        self.assertAlmostEqual(model.get_threshold(), 1.3)
        for bad in (0.0, -1.0, np.inf):
            with self.assertRaisesRegex(RuntimeError, "threshold must be > 0"):
                ModelEpsilonInsensitive(threshold=bad)
            with self.assertRaisesRegex(RuntimeError, "threshold must be > 0"):
                model.threshold = bad
        with self.assertRaisesRegex(RuntimeError, "threshold must be > 0"):
            ModelHuber(threshold=0.0)

    def test_modified_huber_formula_gradient_sparse_parity_and_validation(self):
        model = ModelModifiedHuber(fit_intercept=True).fit(self.X, self.y_cls)
        sparse_model = ModelModifiedHuber(fit_intercept=True).fit(sparse.csr_matrix(self.X), self.y_cls)

        scores = np.hstack([self.X, np.ones((self.X.shape[0], 1))]) @ self.beta
        margin = self.y_cls * scores
        expected_loss = np.zeros_like(margin)
        expected_residual = np.zeros_like(margin)
        linear = margin <= -1.0
        quadratic = (margin > -1.0) & (margin < 1.0)
        expected_loss[linear] = -4.0 * margin[linear]
        expected_residual[linear] = -4.0 * self.y_cls[linear]
        expected_loss[quadratic] = (1.0 - margin[quadratic]) ** 2
        expected_residual[quadratic] = -2.0 * self.y_cls[quadratic] * (1.0 - margin[quadratic])
        expected_grad = (
            np.hstack([self.X, np.ones((self.X.shape[0], 1))]).T
            @ expected_residual
            / self.X.shape[0]
        )

        self.assertAlmostEqual(model.loss(self.beta), float(np.mean(expected_loss)))
        np.testing.assert_allclose(model.grad(self.beta), expected_grad)
        np.testing.assert_allclose(model.grad(self.beta), finite_difference_grad(model, self.beta), atol=1e-6)
        self.assertAlmostEqual(sparse_model.loss(self.beta), model.loss(self.beta))
        np.testing.assert_allclose(sparse_model.grad(self.beta), model.grad(self.beta))

        row_norms = np.sum(self.X * self.X, axis=1) + 1.0
        self.assertAlmostEqual(model.get_lip_mean(), 2.0 * float(np.mean(row_norms)))
        self.assertAlmostEqual(model.get_lip_max(), 2.0 * float(np.max(row_norms)))
        with self.assertRaisesRegex(ValueError, "exactly two classes"):
            ModelModifiedHuber().fit(self.X, np.ones(self.X.shape[0]))

    def test_validation_errors_are_explicit(self):
        model = ModelHuber()
        with self.assertRaisesRegex(ValueError, "fit"):
            model.loss(self.beta)
        with self.assertRaisesRegex(ValueError, "sample counts"):
            ModelHuber().fit(self.X, self.y_reg[:-1])
        with self.assertRaisesRegex(ValueError, "finite"):
            bad_X = self.X.copy()
            bad_X[0, 0] = np.nan
            ModelHuber().fit(bad_X, self.y_reg)
        with self.assertRaisesRegex(ValueError, "coefficients"):
            ModelHuber().fit(self.X, self.y_reg).grad(np.zeros(self.X.shape[1]))


if __name__ == "__main__":
    unittest.main()
