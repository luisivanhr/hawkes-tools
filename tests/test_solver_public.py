import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hawkes_tools import solver as solver_module
from hawkes_tools.prox import ProxZero
from hawkes_tools.solver import AGD, BFGS, GD, GFB, SAGA, SCPG, SDCA, SGD, SVRG, AdaGrad


class QuadraticModel:
    n_coeffs = 3

    def __init__(self):
        self.target = np.array([1.0, -2.0, 0.5])

    def loss(self, coeffs):
        delta = np.asarray(coeffs, dtype=float) - self.target
        return 0.5 * float(delta @ delta)

    def grad(self, coeffs):
        return np.asarray(coeffs, dtype=float) - self.target


class SolverPublicTest(unittest.TestCase):
    def test_public_solver_exports_cover_reference_top_level_names(self):
        expected = {
            "GD",
            "AGD",
            "BFGS",
            "SCPG",
            "SGD",
            "SVRG",
            "SAGA",
            "SDCA",
            "GFB",
            "AdaGrad",
            "History",
        }

        self.assertTrue(expected.issubset(set(solver_module.__all__)))

    def test_constructor_rejects_invalid_numeric_parameters(self):
        cases = [
            (GD, {"step": 0.0}, "step"),
            (GD, {"step": np.inf}, "step"),
            (GD, {"tol": -1.0}, "tol"),
            (GD, {"max_iter": -1}, "max_iter"),
            (GD, {"print_every": 0}, "print_every"),
            (GD, {"record_every": 0}, "record_every"),
            (SVRG, {"n_threads": 0}, "n_threads"),
            (SVRG, {"epoch_size": 0}, "epoch_size"),
            (SVRG, {"batch_size": 0}, "batch_size"),
        ]
        for solver_class, kwargs, message in cases:
            with self.subTest(solver=solver_class.__name__, kwargs=kwargs):
                with self.assertRaisesRegex(ValueError, message):
                    solver_class(**kwargs)

    def test_source_derived_stochastic_option_validation(self):
        svrg = SVRG(step_type="barzilai-borwein", variance_reduction="rand", rand_type="perm")
        self.assertEqual(svrg.step_type, "bb")
        self.assertEqual(svrg.variance_reduction, "rand")
        self.assertEqual(svrg.rand_type, "perm")

        with self.assertRaisesRegex(ValueError, "variance_reduction"):
            SVRG(variance_reduction="stuff")
        with self.assertRaisesRegex(ValueError, "step_type"):
            svrg.step_type = "stuff"
        with self.assertRaisesRegex(ValueError, "rand_type"):
            SGD(rand_type="stuff")

    def test_negative_reference_style_seed_means_unseeded(self):
        solver = SVRG(seed=-123, random_state=-1)

        self.assertIsNone(solver.seed)
        self.assertIsNone(solver.random_state)

    def test_set_model_and_solve_validate_inputs(self):
        class MissingGradient:
            n_coeffs = 3

            def loss(self, coeffs):
                return float(np.sum(coeffs))

        class InvalidNCoeffs(QuadraticModel):
            n_coeffs = 0

        with self.assertRaisesRegex(ValueError, "loss.*grad"):
            GD().set_model(MissingGradient())
        with self.assertRaisesRegex(ValueError, "model.n_coeffs"):
            GD().set_model(InvalidNCoeffs())

        solver = GD(max_iter=1, step=0.5, verbose=False).set_model(QuadraticModel()).set_prox(ProxZero())
        with self.assertRaisesRegex(ValueError, "one-dimensional"):
            solver.solve(np.zeros((3, 1)))
        with self.assertRaisesRegex(ValueError, "model.n_coeffs"):
            solver.solve(np.zeros(2))
        with self.assertRaisesRegex(ValueError, "finite"):
            solver.solve(np.array([0.0, np.nan, 0.0]))
        with self.assertRaisesRegex(ValueError, "step"):
            solver.solve(np.zeros(3), step=-0.1)

    def test_public_solvers_still_run_reference_quadratic_problem(self):
        for solver_class in [GD, AGD, BFGS, GFB, SCPG, SGD, AdaGrad, SVRG, SAGA, SDCA]:
            with self.subTest(solver=solver_class.__name__):
                solution = (
                    solver_class(max_iter=40, step=0.3, verbose=False)
                    .set_model(QuadraticModel())
                    .set_prox(ProxZero())
                    .solve(np.zeros(3))
                )

                self.assertEqual(solution.shape, (3,))
                self.assertLess(np.linalg.norm(solution - QuadraticModel().target), 0.35)


if __name__ == "__main__":
    unittest.main()
