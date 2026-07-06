import json
import re
import unittest
from pathlib import Path


NOTEBOOK = Path(__file__).resolve().parents[1] / "examples" / "gallery_reproduction.ipynb"


def _cell_source(cell):
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else str(source)


class GalleryNotebookTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8-sig"))
        cls.sources = [_cell_source(cell) for cell in cls.notebook["cells"]]
        cls.code = "\n".join(
            source
            for cell, source in zip(cls.notebook["cells"], cls.sources)
            if cell.get("cell_type") == "code"
        )
        cls.markdown = "\n".join(
            source
            for cell, source in zip(cls.notebook["cells"], cls.sources)
            if cell.get("cell_type") == "markdown"
        )

    def test_restored_full_gallery_notebook_shape_is_preserved(self):
        self.assertGreaterEqual(len(self.notebook["cells"]), 30)
        self.assertGreaterEqual(len(re.findall(r"^## ", self.markdown, flags=re.MULTILINE)), 15)
        self.assertGreaterEqual(self.code.count("record("), 25)
        self.assertIn("## Gallery", self.markdown)
        self.assertIn("| # | Example | API area | Description |", self.markdown)
        self.assertIn("RKHS Hawkes estimator with alpha/Erlang kernel", self.markdown)
        self.assertIn("overlays true and RKHS-estimated kernels", self.markdown)
        self.assertNotIn("## Crawled Gallery Manifest", self.markdown)
        self.assertNotIn("Notebook handling", self.markdown)
        self.assertNotIn("Runnable with", self.markdown)

    def test_notebook_does_not_import_original_package_or_scikit(self):
        original_package = "".join(chr(code) for code in (116, 105, 99, 107))
        pattern = rf"^\s*(from|import)\s+{re.escape(original_package)}\b"
        self.assertIsNone(re.search(pattern, self.code, flags=re.MULTILINE))
        self.assertNotIn("sklearn", self.code)

    def test_requested_gallery_fixes_are_present(self):
        self.assertIn('pcolor_kwargs={"cmap": "RdBu"}', self.code)
        self.assertIn("ragged_shape", self.code)
        self.assertIn("Positive signed kernel norms are plotted in blue", self.markdown)
        self.assertIn("nonpositive_grid_values", self.code)
        self.assertIn("noisy conditional-law values rather than a Matplotlib rendering issue", self.markdown)
        self.assertIn("sim_elapsed", self.code)
        self.assertIn("fit_elapsed", self.code)
        self.assertIn("## 15. Automatic Step Choice", self.markdown)
        self.assertIn("## 17. Asynchronous Stochastic Solver", self.markdown)
        self.assertIn("## 24. Generalized Linear Models Solver Convergence", self.markdown)
        self.assertIn("weights_sparse_gauss", self.code)
        self.assertIn("plot_history", self.code)
        self.assertIn("range=(0, n_features)", self.code)
        self.assertIn("## Standalone RKHS Hawkes Estimator With Alpha Kernel", self.markdown)
        self.assertIn("RKHSHawkes", self.code)
        self.assertIn("alpha_erlang_kernel_values", self.code)
        self.assertIn('label="true alpha/Erlang kernel"', self.code)
        self.assertIn('label="RKHS estimate"', self.code)
        self.assertIn('plot_basis_functions(3, layout="grid"', self.code)
        self.assertIn("rkhs_alpha_erlang_kernel", self.code)
        self.assertNotIn("QuadraticModel", self.code)
        self.assertNotIn("Solver/Prox And GLM Compatibility", self.markdown)


if __name__ == "__main__":
    unittest.main()
