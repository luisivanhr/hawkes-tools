import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import hawkes_tools.simulation as simulation
from hawkes_tools.robust import features_normal_cov_toeplitz as robust_toeplitz


class SimulationHelperCompatibilityTest(unittest.TestCase):
    def test_public_exports_match_tick_simulation_helpers(self):
        self.assertEqual(
            set(simulation.__all__),
            {
                "features_normal_cov_uniform",
                "features_normal_cov_toeplitz",
                "weights_sparse_exp",
                "weights_sparse_gauss",
            },
        )

    def test_weights_sparse_exp_matches_tick_formula(self):
        weights = simulation.weights_sparse_exp(n_weigths=6, nnz=4, scale=2.0)

        np.testing.assert_allclose(
            weights,
            np.array([-1.0, np.exp(-0.5), -np.exp(-1.0), np.exp(-1.5), 0.0, 0.0]),
        )

        weights32 = simulation.weights_sparse_exp(n_weigths=4, nnz=2, dtype="float32")
        self.assertEqual(weights32.dtype, np.dtype("float32"))

    def test_weights_sparse_exp_clips_nnz_like_tick(self):
        with self.assertWarnsRegex(RuntimeWarning, "nnz must be smaller"):
            weights = simulation.weights_sparse_exp(n_weigths=3, nnz=5)

        self.assertEqual(weights.shape, (3,))
        self.assertEqual(np.count_nonzero(weights), 3)

    def test_feature_generators_return_expected_shape_and_dtype(self):
        np.random.seed(123)
        uniform = simulation.features_normal_cov_uniform(8, 4, dtype="float32")
        self.assertEqual(uniform.shape, (8, 4))
        self.assertEqual(uniform.dtype, np.dtype("float32"))
        self.assertTrue(np.all(np.isfinite(uniform)))

        np.random.seed(456)
        toeplitz = simulation.features_normal_cov_toeplitz(6, 3, cov_corr=0.25)
        self.assertEqual(toeplitz.shape, (6, 3))
        self.assertTrue(np.all(np.isfinite(toeplitz)))

    def test_toeplitz_helper_matches_existing_robust_gallery_helper(self):
        np.random.seed(789)
        from_simulation = simulation.features_normal_cov_toeplitz(5, 4, cov_corr=0.4)
        np.random.seed(789)
        from_robust = robust_toeplitz(5, 4, cov_corr=0.4)

        np.testing.assert_allclose(from_simulation, from_robust)


if __name__ == "__main__":
    unittest.main()
