import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hawkes_tools.base import TimeFunction
import hawkes_tools.hawkes as hawkes
from hawkes_tools.hawkes import HawkesKernelTimeFunc, RKHSHawkes


class RKHSHawkesEstimatorTest(unittest.TestCase):
    def test_public_export_and_univariate_fit_accessors(self):
        self.assertIn("RKHSHawkes", hawkes.__all__)
        self.assertIs(hawkes.RKHSHawkes, RKHSHawkes)

        events = [np.linspace(0.5, 19.5, 20)]
        learner = RKHSHawkes(
            kernel_support=3.0,
            bandwidth=0.5,
            covariance_spline_order=1,
            rkhs_grid_size=101,
            rkhs_basis_size=6,
            quadrature_size=24,
        ).fit(events, end_times=20.0)

        self.assertEqual(learner.n_nodes, 1)
        self.assertEqual(learner.n_realizations, 1)
        np.testing.assert_allclose(learner.mean_intensity, np.array([1.0]))
        self.assertEqual(learner.baseline.shape, (1,))
        self.assertEqual(learner.adjacency.shape, (1, 1))
        self.assertEqual(learner.kernel.shape, (1, 1, learner.kernel_lags.size))
        self.assertIsInstance(learner.kernel_time_function, TimeFunction)
        self.assertEqual(learner.kernels.shape, (1, 1))
        self.assertIsInstance(learner.kernels[0, 0], HawkesKernelTimeFunc)
        self.assertIs(learner.kernels[0, 0].time_function, learner.kernel_time_function)
        self.assertTrue(np.all(np.diff(learner.autocorrelation_lags) > 0.0))
        self.assertTrue(np.all(np.diff(learner.kernel_lags) > 0.0))
        self.assertGreaterEqual(float(learner.kernel_lags[0]), 0.0)
        self.assertEqual(float(learner.kernel_lags[-1]), learner.kernel_support)
        self.assertTrue(np.all(np.isfinite(learner.autocorrelation)))
        self.assertTrue(np.all(np.isfinite(learner.kernel_values)))
        self.assertTrue(np.all(learner.kernel_values >= 0.0))
        np.testing.assert_allclose(learner.get_kernel_supports(), np.array([[3.0]]))
        np.testing.assert_allclose(learner.get_kernel_norms(), learner.adjacency)

        x = np.array([0.0, 0.5, 1.5, 3.0, 3.5])
        values = learner.get_kernel_values(0, 0, x)
        np.testing.assert_allclose(values, learner.kernels[0, 0].get_values(x))
        self.assertEqual(values.shape, x.shape)
        self.assertEqual(values[-1], 0.0)
        primitive = learner._compute_primitive_kernel_values(0, 0, x)
        np.testing.assert_allclose(primitive, learner.kernels[0, 0].get_primitive_values(x))
        self.assertTrue(np.all(np.diff(primitive[:-1]) >= -1e-12))

        lag_basis_x, lag_basis_y = learner.recover_basis_functions(2)
        self.assertEqual(lag_basis_y.shape, (3, lag_basis_x.size))
        self.assertTrue(np.all(np.diff(lag_basis_x) > 0.0))
        self.assertGreaterEqual(float(lag_basis_x[0]), 0.0)
        self.assertLessEqual(float(lag_basis_x[-1]), learner.kernel_support)
        rkhs_basis_x, rkhs_basis_y = learner.recover_basis_functions(2, domain="rkhs")
        np.testing.assert_allclose(rkhs_basis_x, learner._rkhs_domain)
        np.testing.assert_allclose(rkhs_basis_y, learner._rkhs_basis_functions[:3])

        fig = learner.plot_basis_functions(2, show=False)
        self.assertEqual(len(fig.axes), 1)
        self.assertEqual(len(fig.axes[0].lines), 3)
        import matplotlib.pyplot as plt

        plt.close(fig)
        fig_grid = learner.plot_basis_functions(3, show=False, layout="grid")
        self.assertEqual(len(fig_grid.axes), 4)
        self.assertTrue(all(len(axis.lines) == 1 for axis in fig_grid.axes))
        plt.close(fig_grid)
        self.assertTrue(math.isfinite(learner.score()))

    def test_parameter_validation_and_multivariate_rejection(self):
        with self.assertRaisesRegex(ValueError, "bandwidth"):
            RKHSHawkes(kernel_support=1.0, bandwidth=1.0)
        with self.assertRaisesRegex(ValueError, "rkhs_grid_size"):
            RKHSHawkes(kernel_support=1.0, bandwidth=0.2, rkhs_grid_size=4)
        with self.assertRaisesRegex(ValueError, "covariance_spline_order"):
            RKHSHawkes(kernel_support=1.0, bandwidth=0.2, covariance_spline_order=6)

        learner = RKHSHawkes(
            kernel_support=2.0,
            bandwidth=0.5,
            rkhs_grid_size=51,
            rkhs_basis_size=4,
            quadrature_size=16,
        )
        with self.assertRaisesRegex(ValueError, "univariate"):
            learner.fit([np.array([0.1, 0.4]), np.array([0.2, 0.5])], end_times=1.0)

    def test_basis_function_recovery_validation(self):
        learner = RKHSHawkes(
            kernel_support=2.0,
            bandwidth=0.5,
            rkhs_grid_size=51,
            rkhs_basis_size=4,
            quadrature_size=16,
        )
        with self.assertRaisesRegex(ValueError, "fit"):
            learner.recover_basis_functions(0)

        learner.fit([np.linspace(0.5, 9.5, 10)], end_times=10.0)
        with self.assertRaisesRegex(ValueError, "non-negative"):
            learner.recover_basis_functions(-1)
        with self.assertRaisesRegex(ValueError, "rkhs_basis_size"):
            learner.recover_basis_functions(4)
        with self.assertRaisesRegex(ValueError, "domain"):
            learner.recover_basis_functions(0, domain="time")
        with self.assertRaisesRegex(ValueError, "layout"):
            learner.plot_basis_functions(0, layout="stack")


if __name__ == "__main__":
    unittest.main()
