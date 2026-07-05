import sys
import unittest
import warnings
from pathlib import Path

import numpy as np
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hawkes_tools.linear_model import ModelLogReg
from hawkes_tools.prox import ProxElasticNet, ProxL1, ProxPositive
from hawkes_tools.solver import AGD, BFGS, GD, SAGA, SDCA, SVRG


class QuadraticModel:
    n_coeffs = 3

    def __init__(self):
        self.target = np.array([1.0, -2.0, 0.5])

    def loss(self, coeffs):
        delta = np.asarray(coeffs, dtype=float) - self.target
        return 0.5 * float(delta @ delta)

    def grad(self, coeffs):
        return np.asarray(coeffs, dtype=float) - self.target


class OptimCompatibilityTest(unittest.TestCase):
    def test_public_prox_modules_apply_expected_transforms(self):
        np.testing.assert_allclose(ProxL1(0.25).call(np.array([1.0, -0.1]), step=1.0), [0.75, 0.0])
        np.testing.assert_allclose(ProxPositive().call(np.array([1.0, -0.1])), [1.0, 0.0])

    def test_tick_style_solver_flow_runs_for_all_public_solvers(self):
        for solver_class in [GD, AGD, BFGS, SVRG, SAGA, SDCA]:
            with self.subTest(solver=solver_class.__name__):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    solution = (
                        solver_class(max_iter=25, step=0.3, verbose=False)
                        .set_model(QuadraticModel())
                        .set_prox(ProxElasticNet(0.0, ratio=0.5))
                        .solve(np.zeros(3))
                    )

                self.assertEqual(solution.shape, (3,))
                self.assertLess(np.linalg.norm(solution - QuadraticModel().target), 0.25)

    def test_svrg_and_saga_use_sparse_stochastic_updates(self):
        rng = np.random.default_rng(123)
        features = sparse.random(120, 80, density=0.04, format="csr", random_state=123)
        weights = np.zeros(80)
        weights[:8] = np.linspace(0.8, -0.5, 8)
        logits = np.asarray(features @ weights).reshape(-1) - 0.1
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        labels = (rng.random(features.shape[0]) < probabilities).astype(float)
        model = ModelLogReg(fit_intercept=True).fit(features, labels)
        start = np.zeros(model.n_coeffs)

        np.testing.assert_allclose(
            model.batch_grad(start, np.arange(model.n_samples)),
            model.grad(start),
            atol=1e-12,
        )

        prox = ProxElasticNet(1e-4, ratio=0.5, range=(0, features.shape[1]))
        start_obj = model.loss(start) + prox.value(start)
        for solver_class in [SVRG, SAGA]:
            with self.subTest(solver=solver_class.__name__):
                solver = solver_class(
                    step=0.25 / model.get_lip_max(),
                    max_iter=8,
                    tol=0.0,
                    record_every=1,
                    seed=42,
                    n_threads=2,
                    batch_size=20,
                    verbose=False,
                ).set_model(model).set_prox(prox)
                solution = solver.solve(start)

                self.assertEqual(solution.shape, start.shape)
                self.assertLess(solver.objective(solution), start_obj)
                self.assertEqual(solver.history.records[-1]["batch_size"], 20)
                self.assertEqual(solver.history.records[-1]["epoch_size"], model.n_samples)


if __name__ == "__main__":
    unittest.main()

