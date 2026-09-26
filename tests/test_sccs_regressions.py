"""Regression checks for SCCS coefficient and step-size correctness."""

import sys
import unittest
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.special import softmax

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hawkes_tools.solver import GD
from hawkes_tools.survival import ConvSCCS, ModelSCCS


class ConvSCCSCoefficientTest(unittest.TestCase):
    def test_overlapping_zero_lag_exposures_preserve_optimum(self):
        features = np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.]])
        counts = np.array([1, 3, 5, 15], dtype=np.int32)
        optimum = np.log([3., 5.])
        censoring = np.array([4], dtype=np.uint64)
        for design in (features, sparse.csr_matrix(features)):
            with self.subTest(sparse=sparse.issparse(design)):
                learner = ConvSCCS(n_lags=[0, 0], max_iter=1000, tol=1e-12)
                fitted, intervals = learner.fit([design], [counts], censoring)
                beta = np.concatenate(fitted)
                np.testing.assert_allclose(beta, optimum, atol=1e-6)
                probabilities = softmax(features @ beta)
                np.testing.assert_allclose(probabilities, counts / counts.sum(), atol=1e-7)
                # score includes the learner's existing numerical ridge.
                expected_loss = -counts @ np.log(counts / counts.sum())
                expected_loss += 0.5e-8 * (optimum @ optimum)
                self.assertAlmostEqual(learner.score(), expected_loss, places=9)
                np.testing.assert_allclose(np.concatenate(intervals["refit_coeffs"]), beta)

    def test_overlapping_zero_lag_exposures_preserve_penalized_optimum(self):
        features = np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.]])
        counts = np.array([1, 3, 5, 15], dtype=np.int32)
        learner = ConvSCCS(
            n_lags=[0, 0], penalized_features=[1], C_group_l1=2.,
            max_iter=1000, tol=1e-12,
        )
        learner.fit([features], [counts], np.array([4], dtype=np.uint64))
        beta = np.concatenate(learner.coeffs)
        # Independent binary exposures factorize the conditional likelihood.
        # The positive second coefficient pays an L1 derivative of 1 / C.
        optimum = np.log([18. / 6., (20. - 0.5) / (4. + 0.5)])
        np.testing.assert_allclose(beta, optimum, atol=1e-6)
        probabilities = softmax(features @ beta)
        gradient = features.T @ (counts.sum() * probabilities - counts)
        gradient += 1e-8 * beta + np.array([0., 0.5])
        np.testing.assert_allclose(gradient, 0., atol=2e-6)


class SCCSLipschitzTest(unittest.TestCase):
    def test_identical_case_replication_preserves_bound_and_gd_convergence(self):
        features = np.array([[-1.], [1.]])
        counts = np.array([1, 2], dtype=np.int32)
        optimum = np.array([np.log(2.) / 2.])
        models = [
            ModelSCCS(n_intervals=2, n_lags=[0]).fit(
                [features.copy() for _ in range(n)],
                [counts.copy() for _ in range(n)],
            )
            for n in (1, 100)
        ]
        for beta in (np.zeros(1), optimum, np.array([-2.])):
            self.assertAlmostEqual(models[0].loss(beta), models[1].loss(beta), places=12)
            np.testing.assert_allclose(models[0].grad(beta), models[1].grad(beta), atol=1e-12)
        fits = []
        for model in models:
            self.assertEqual(model.get_lip_max(), 3.)
            self.assertEqual(model.get_lip_mean(), 3.)
            fitted = GD(max_iter=40, tol=0).set_model(model).solve()
            fits.append(fitted)
            np.testing.assert_allclose(fitted, optimum, atol=1e-10)
            self.assertAlmostEqual(model.loss(fitted), 1.9095425048844383, places=12)
        np.testing.assert_allclose(fits[0], fits[1], atol=1e-12)

    def test_bound_respects_case_maximum_and_censoring(self):
        features = [
            np.array([[-1.], [1.], [1e6]]),
            np.array([[-2.], [2.], [-1e6]]),
            np.array([[-1e6], [1e6], [0.]]),
        ]
        counts = [
            np.array([1, 2, 100], dtype=np.int32),
            np.array([1, 0, 100], dtype=np.int32),
            np.zeros(3, dtype=np.int32),
        ]
        censoring = np.array([2, 2, 3], dtype=np.uint64)
        model = ModelSCCS(n_intervals=3, n_lags=[0]).fit(features, counts, censoring)
        # Case bounds are 3, 4, and 0; censored rows must not affect them.
        self.assertEqual(model.get_lip_max(), 4.)
        for coefficient in (-2., 0., 0.5):
            beta = np.array([coefficient])
            hessian = 0.
            for X, y, c in zip(features, counts, censoring):
                x = X[:int(c), 0]
                p = softmax(x * coefficient)
                hessian += y[:int(c)].sum() * (p @ (x * x) - (p @ x) ** 2)
            hessian /= len(features)
            self.assertLessEqual(hessian, model.get_lip_max() + 1e-12)
            finite_difference = (model.grad(beta + 1e-5) - model.grad(beta - 1e-5)) / 2e-5
            np.testing.assert_allclose(finite_difference, [hessian], atol=1e-8)


if __name__ == "__main__":
    unittest.main()
