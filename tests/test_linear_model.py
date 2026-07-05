import sys
import unittest
from pathlib import Path

import numpy as np
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hawkes_tools.linear_model import (
    LinearRegression,
    LogisticRegression,
    ModelLogReg,
    ModelLinReg,
    ModelPoisReg,
    PoissonRegression,
    SimuLinReg,
    SimuLogReg,
    SimuPoisReg,
    weights_sparse_gauss,
)
from hawkes_tools.prox import ProxBinarsity, ProxElasticNet, ProxL1, ProxL2Sq, ProxTV, ProxZero


class LinearModelPublicSurfaceTest(unittest.TestCase):
    def test_public_exports_cover_migrated_glm_names(self):
        import hawkes_tools.linear_model as linear_model

        expected = {
            "ModelLinReg",
            "ModelLogReg",
            "ModelPoisReg",
            "LinearRegression",
            "LogisticRegression",
            "PoissonRegression",
            "LearnerLinReg",
            "LearnerLogReg",
            "LearnerPoisReg",
            "SimuLinReg",
            "SimuLogReg",
            "SimuPoisReg",
            "weights_sparse_gauss",
        }
        self.assertEqual(set(), expected - set(linear_model.__all__))

    def test_fit_intercept_uses_tick_weights_first_layout(self):
        X = np.array([[1.0, 2.0], [0.5, -1.0], [-0.25, 0.75]])
        y = X @ np.array([0.4, -0.2]) + 0.7
        model = ModelLinReg(fit_intercept=True).fit(X, y)

        self.assertEqual(model.n_features, 2)
        self.assertEqual(model.n_coeffs, 3)
        np.testing.assert_allclose(model.X[:, -1], np.ones(X.shape[0]))
        self.assertEqual(model._l2_end, 2)

    def test_sparse_logistic_model_supports_gallery_scale_contract(self):
        rng = np.random.default_rng(123)
        X = sparse.random(200, 100, density=0.02, format="csr", random_state=123)
        weights = weights_sparse_gauss(100, nnz=8)
        linear = np.asarray(X @ weights).reshape(-1)
        y = np.where(rng.random(X.shape[0]) < 1.0 / (1.0 + np.exp(-linear)), 1.0, -1.0)
        model = ModelLogReg(fit_intercept=True).fit(X, y)
        beta = np.zeros(model.n_coeffs)

        self.assertTrue(np.isfinite(model.loss(beta)))
        self.assertEqual(model.grad(beta).shape, (101,))
        self.assertGreater(model.get_lip_max(), 0.0)

    def test_logistic_and_poisson_gradients_match_finite_differences(self):
        rng = np.random.default_rng(12)
        X = rng.normal(size=(30, 3))
        beta = np.array([0.2, -0.4, 0.3, 0.1])

        log_y = rng.binomial(1, 0.45, size=X.shape[0])
        log_model = ModelLogReg(fit_intercept=True, l2_strength=1e-3).fit(X, log_y)
        self._assert_finite_difference_gradient(log_model, beta, log_model.grad(beta))

        pois_y = rng.poisson(np.exp(0.1 + X @ np.array([0.2, -0.1, 0.05])))
        pois_model = ModelPoisReg(fit_intercept=True, l2_strength=1e-3).fit(X, pois_y)
        self._assert_finite_difference_gradient(pois_model, beta, pois_model.grad(beta))

    def _assert_finite_difference_gradient(self, model, beta, grad):
        eps = 1e-6
        for j in range(beta.size):
            step = np.zeros_like(beta)
            step[j] = eps
            numeric = (model.loss(beta + step) - model.loss(beta - step)) / (2.0 * eps)
            self.assertAlmostEqual(numeric, grad[j], places=4)

    def test_model_methods_validate_fit_state_coefficients_and_indices(self):
        X = np.array([[1.0, 0.5], [-0.5, 2.0], [0.25, -1.0]])
        y = np.array([0.3, -1.2, 0.7])

        with self.assertRaisesRegex(ValueError, "fit"):
            ModelLinReg().loss(np.zeros(2))

        model = ModelLinReg(fit_intercept=True).fit(X, y)
        with self.assertRaisesRegex(ValueError, "model.n_coeffs"):
            model.loss(np.zeros(2))
        with self.assertRaisesRegex(ValueError, "finite"):
            model.grad(np.array([0.0, np.nan, 0.0]))
        with self.assertRaisesRegex(ValueError, "out of bounds"):
            model.batch_grad(np.zeros(3), np.array([0, 3]))
        with self.assertRaisesRegex(ValueError, "denominator"):
            model.grad_from_residuals(np.array([0]), np.array([1.0]), denominator=0)

    def test_learner_constructor_validation_is_explicit(self):
        with self.assertRaisesRegex(ValueError, "unknown solver"):
            LinearRegression(solver="wrong_name")
        with self.assertRaisesRegex(ValueError, "supported penalties"):
            LogisticRegression(penalty="wrong_name")
        with self.assertRaisesRegex(ValueError, "C must be positive"):
            PoissonRegression(penalty="l2", C=0)
        with self.assertRaisesRegex(ValueError, "elastic_net_ratio"):
            LogisticRegression(penalty="elasticnet", elastic_net_ratio=1.5)
        with self.assertRaisesRegex(ValueError, "blocks_start"):
            LogisticRegression(penalty="binarsity")
        with self.assertRaisesRegex(ValueError, "blocks_length"):
            LinearRegression(penalty="binarsity", blocks_start=[0])
        with self.assertRaisesRegex(ValueError, "overlap"):
            PoissonRegression(penalty="binarsity", blocks_start=[0, 1], blocks_length=[2, 2])

    def test_learner_penalty_surface_matches_tick_prox_mapping(self):
        X = np.array(
            [
                [1.0, 0.0, -0.5],
                [0.5, 1.0, 0.25],
                [-0.75, 0.5, 1.0],
                [1.5, -0.25, 0.0],
                [0.0, -1.0, 0.75],
                [-0.5, 0.25, -1.0],
            ]
        )
        cases = [
            (LinearRegression, np.array([0.2, 0.8, -0.5, 1.0, -0.3, 0.1])),
            (LogisticRegression, np.array([0, 1, 0, 1, 1, 0])),
            (PoissonRegression, np.array([1, 0, 2, 1, 3, 0])),
        ]
        prox_classes = {
            "none": ProxZero,
            "l1": ProxL1,
            "l2": ProxL2Sq,
            "elasticnet": ProxElasticNet,
            "tv": ProxTV,
            "binarsity": ProxBinarsity,
        }

        for learner_cls, y in cases:
            for penalty, prox_class in prox_classes.items():
                kwargs = {
                    "penalty": penalty,
                    "solver": "gd",
                    "step": 0.05,
                    "max_iter": 1,
                    "fit_intercept": True,
                }
                if penalty == "binarsity":
                    kwargs.update(blocks_start=[0], blocks_length=[X.shape[1]])
                learner = learner_cls(**kwargs).fit(X, y)

                self.assertIsInstance(learner._prox_obj, prox_class)
                if penalty != "none":
                    self.assertEqual(learner._prox_obj.range, (0, X.shape[1]))

    def test_learners_require_fit_before_prediction_helpers(self):
        X = np.ones((3, 2))
        y = np.array([0.0, 1.0, 0.0])

        with self.assertRaisesRegex(ValueError, "fit"):
            LinearRegression().predict(X)
        with self.assertRaisesRegex(ValueError, "fit"):
            LogisticRegression().predict_proba(X)
        with self.assertRaisesRegex(ValueError, "fit"):
            PoissonRegression().loglik(X, y)


class LinearModelPipelineTest(unittest.TestCase):
    def test_linear_simulation_then_fit_recovers_coefficients(self):
        coeffs = np.array([0.3, -1.2, 0.8])
        X, y = SimuLinReg(coeffs=coeffs, intercept=0.4, n_samples=600, noise_std=0.05, seed=101).simulate()
        learner = LinearRegression(penalty="none", solver="bfgs", max_iter=80, tol=1e-10).fit(X, y)

        np.testing.assert_allclose(learner.intercept, 0.4, atol=0.02)
        np.testing.assert_allclose(learner.weights, coeffs, atol=0.02)
        self.assertGreater(learner.score(X, y), 0.99)

    def test_linear_regression_warm_start_reuses_previous_coefficients(self):
        coeffs = np.array([0.4, -0.7])
        X, y = SimuLinReg(coeffs=coeffs, intercept=0.2, n_samples=80, noise_std=0.0, seed=505).simulate()
        learner = LinearRegression(
            penalty="none",
            solver="gd",
            step=0.05,
            max_iter=2,
            tol=0.0,
            warm_start=True,
        ).fit(X, y)
        first_coeffs = learner.coeffs.copy()

        learner.fit(X, y)
        second_coeffs = learner.coeffs.copy()

        self.assertLess(
            learner._solver_obj.objective(second_coeffs),
            learner._solver_obj.objective(first_coeffs),
        )

    def test_logistic_simulation_then_fit_recovers_direction(self):
        coeffs = np.array([1.0, -1.4, 0.6])
        X, y = SimuLogReg(coeffs=coeffs, intercept=-0.2, n_samples=1_500, seed=202).simulate()
        learner = LogisticRegression(penalty="l2", C=1e3, solver="bfgs", max_iter=120, tol=1e-7).fit(X, y)

        direction = learner.weights / np.linalg.norm(learner.weights)
        expected = coeffs / np.linalg.norm(coeffs)
        self.assertGreater(float(direction @ expected), 0.95)
        self.assertGreater(learner.score(X, y), 0.72)

    def test_poisson_simulation_then_fit_recovers_coefficients(self):
        coeffs = np.array([0.25, -0.35, 0.15])
        X, y = SimuPoisReg(coeffs=coeffs, intercept=0.1, n_samples=2_000, seed=303).simulate()
        learner = PoissonRegression(penalty="l2", C=1e4, solver="bfgs", max_iter=120, tol=1e-7).fit(X, y)

        np.testing.assert_allclose(learner.intercept, 0.1, atol=0.08)
        np.testing.assert_allclose(learner.weights, coeffs, atol=0.08)
        self.assertGreater(np.corrcoef(learner.predict(X), y)[0, 1], 0.25)

    def test_poisson_loglik_and_prediction_follow_source_semantics(self):
        coeffs = np.array([0.15, -0.2])
        X, y = SimuPoisReg(coeffs=coeffs, intercept=0.25, n_samples=120, seed=404).simulate()
        learner = PoissonRegression(penalty="l2", C=1e4, solver="bfgs", max_iter=80, tol=1e-7).fit(X, y)

        reference_coeffs = np.r_[learner.weights, learner.intercept]
        reference_loss = ModelPoisReg(fit_intercept=True).fit(X, y).loss(reference_coeffs)
        self.assertAlmostEqual(learner.loglik(X, y), reference_loss)
        np.testing.assert_allclose(learner.predict(X[:10]), np.rint(learner.decision_function(X[:10])))


if __name__ == "__main__":
    unittest.main()

