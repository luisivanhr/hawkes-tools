import sys
import unittest
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hawkes_tools.base import History
from hawkes_tools.plot import _extract_process_interval, plot_history


class _Solver:
    def __init__(self, name, n_iter, obj):
        self.name = name
        self.history = History()
        for n, value in zip(n_iter, obj):
            self.history.append(n_iter=n, obj=value)


class _Learner:
    def __init__(self, solver):
        self._solver_obj = solver


class PlotPublicBehaviorTest(unittest.TestCase):
    def assert_array_list_equal(self, list1, list2):
        self.assertEqual(len(list1), len(list2))
        for array1, array2 in zip(list1, list2):
            np.testing.assert_array_almost_equal(array1, array2, decimal=10)

    def test_public_exports_cover_standalone_plot_surface(self):
        import hawkes_tools.plot as plot

        expected = {
            "stems",
            "plot_history",
            "plot_hawkes_kernels",
            "plot_hawkes_baseline_and_kernels",
            "plot_hawkes_kernel_norms",
            "plot_basis_kernels",
            "plot_estimated_intensity",
            "plot_timefunction",
            "plot_point_process",
            "qq_plots",
        }
        self.assertEqual(set(), expected - set(plot.__all__))

    def test_plot_history_dist_min_uses_global_minimum(self):
        solver1 = _Solver("GD", [0, 3, 6], [4.0, 3.0, 5.0])
        solver2 = _Solver("AGD", [1, 2, 3], [2.0, 6.0, 7.0])

        fig = plot_history([solver1, solver2], show=False, dist_min=True)
        ax = fig.axes[0]

        np.testing.assert_array_equal(ax.lines[0].get_xydata()[:, 1], np.array([2.0, 1.0, 3.0]))
        np.testing.assert_array_equal(ax.lines[1].get_xydata()[:, 1], np.array([0.0, 4.0, 5.0]))
        self.assertEqual(ax.lines[0].get_label(), "GD")
        self.assertEqual(ax.lines[1].get_label(), "AGD")

    def test_plot_history_reads_solver_history_from_learner(self):
        solver = _Solver("SVRG", [0, 1], [10.0, 8.0])

        fig = plot_history(_Learner(solver), show=False)
        ax = fig.axes[0]

        np.testing.assert_array_equal(ax.lines[0].get_xydata()[:, 0], np.array([0.0, 1.0]))
        np.testing.assert_array_equal(ax.lines[0].get_xydata()[:, 1], np.array([10.0, 8.0]))
        self.assertEqual(ax.lines[0].get_label(), "SVRG")

    def test_plot_history_validation_is_explicit(self):
        class Empty:
            pass

        with self.assertRaisesRegex(ValueError, "no history"):
            plot_history(Empty(), show=False)

        solver = _Solver("GD", [0], [1.0])
        with self.assertRaisesRegex(ValueError, "no history for time"):
            plot_history(solver, x="time", show=False)

        class BadHistory:
            values = {"n_iter": [0, 1], "obj": [1.0]}

            def __len__(self):
                return 2

        bad = Empty()
        bad.name = "BAD"
        bad.history = BadHistory()
        with self.assertRaisesRegex(ValueError, "inconsistent lengths"):
            plot_history(bad, show=False)

    def test_extract_process_interval_matches_source_cases(self):
        plot_nodes = range(2)
        end_time = 12.0
        original_timestamps = [
            np.linspace(1.0, 6.0, 11),
            np.linspace(4.0, 10.0, 5),
        ]
        original_intensity_times = np.linspace(0, end_time, 13)
        original_intensities = [
            np.linspace(1.0, 2.2, 13),
            np.linspace(4.4, 2.0, 13),
        ]

        timestamps, intensity_times, intensities = _extract_process_interval(
            plot_nodes,
            end_time,
            original_timestamps,
            t_min=5,
            intensity_times=original_intensity_times,
            intensities=original_intensities,
        )
        self.assert_array_list_equal(
            timestamps,
            [np.array([5.0, 5.5, 6.0]), np.array([5.5, 7.0, 8.5, 10.0])],
        )
        np.testing.assert_array_equal(intensity_times, [5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
        self.assert_array_list_equal(
            intensities,
            [
                np.array([1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2]),
                np.array([3.4, 3.2, 3.0, 2.8, 2.6, 2.4, 2.2, 2.0]),
            ],
        )

        timestamps, _, _ = _extract_process_interval(plot_nodes, end_time, original_timestamps, max_jumps=2)
        self.assert_array_list_equal(timestamps, [np.array([1.0, 1.5]), np.array([])])

        timestamps, _, _ = _extract_process_interval(
            plot_nodes,
            end_time,
            original_timestamps,
            max_jumps=4,
            t_max=8,
        )
        self.assert_array_list_equal(timestamps, [np.array([4.5, 5.0, 5.5, 6.0]), np.array([5.5, 7.0])])

        with self.assertRaisesRegex(ValueError, "t_min"):
            _extract_process_interval(plot_nodes, end_time, original_timestamps, t_min=12.0)
        with self.assertRaisesRegex(ValueError, "t_max"):
            _extract_process_interval(plot_nodes, end_time, original_timestamps, t_max=0.0)


if __name__ == "__main__":
    unittest.main()
