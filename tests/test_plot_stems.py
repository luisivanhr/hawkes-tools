import sys
import unittest
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hawkes_tools.plot import stems


class PlotStemsCompatibilityTest(unittest.TestCase):
    def test_stems_matplotlib_returns_one_axis_per_vector(self):
        fig = stems(
            [np.array([1.0, -2.0, 3.0]), np.array([0.5, 2.0])],
            titles=["first", "second"],
            sync_axes=True,
            rendering="matplotlib",
            fig_size=(4, 2),
        )

        self.assertEqual(len(fig.axes), 2)
        self.assertEqual(fig.axes[0].get_title(), "first")
        self.assertEqual(fig.axes[1].get_title(), "second")
        self.assertEqual(tuple(fig.axes[0].get_xlim()), tuple(fig.axes[1].get_xlim()))
        self.assertEqual(tuple(fig.axes[0].get_ylim()), tuple(fig.axes[1].get_ylim()))

    def test_stems_validates_titles_length(self):
        with self.assertRaisesRegex(ValueError, "titles"):
            stems([np.array([1.0]), np.array([2.0])], titles=["only one"])

    def test_stems_rejects_unknown_rendering(self):
        with self.assertRaisesRegex(ValueError, "Unknown rendering"):
            stems([np.array([1.0])], rendering="unknown")


if __name__ == "__main__":
    unittest.main()
