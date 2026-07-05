import tempfile
import unittest
from pathlib import Path

import numpy as np
import scipy.sparse

from hawkes_tools.array import load_array, serialize_array


class ArraySerializationTest(unittest.TestCase):
    def test_dense_arrays_round_trip_for_supported_dtypes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            for dtype in ["float32", "float64"]:
                with self.subTest(dtype=dtype):
                    path = Path(tmpdir) / f"dense_{dtype}.cereal"
                    array_1d = np.linspace(0.0, 1.0, 7).astype(dtype)
                    self.assertEqual(serialize_array(array_1d, path), str(path.resolve()))
                    np.testing.assert_array_equal(load_array(path, dtype=dtype), array_1d)

                    array_2d = np.arange(12, dtype=dtype).reshape(3, 4)
                    serialize_array(array_2d, path)
                    np.testing.assert_array_equal(
                        load_array(path, array_dim=2, dtype=dtype),
                        array_2d,
                    )

    def test_column_major_dense_round_trip_preserves_values_and_order(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "col_dense.cereal"
            row_array = np.arange(24, dtype="float64").reshape(4, 6)
            col_array = np.asfortranarray(row_array)
            serialize_array(col_array, path)
            loaded = load_array(path, array_dim=2, dtype="double", major="col")
            np.testing.assert_array_equal(loaded, row_array)
            self.assertTrue(loaded.flags["F_CONTIGUOUS"])

    def test_sparse_arrays_round_trip_for_row_and_column_major(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            row_path = Path(tmpdir) / "row_sparse.cereal"
            col_path = Path(tmpdir) / "col_sparse.cereal"
            dense = np.array(
                [
                    [0.0, 1.5, 0.0],
                    [2.5, 0.0, 3.5],
                    [0.0, 0.0, 4.5],
                ],
                dtype="float32",
            )
            csr = scipy.sparse.csr_matrix(dense)
            csc = scipy.sparse.csc_matrix(dense)

            serialize_array(csr, row_path)
            loaded_csr = load_array(row_path, array_type="sparse", array_dim=2, dtype="float32")
            self.assertTrue(scipy.sparse.isspmatrix_csr(loaded_csr))
            np.testing.assert_array_equal(loaded_csr.toarray(), dense)

            serialize_array(csc, col_path)
            loaded_csc = load_array(
                col_path,
                array_type="sparse",
                array_dim=2,
                dtype="float32",
                major="col",
            )
            self.assertTrue(scipy.sparse.isspmatrix_csc(loaded_csc))
            np.testing.assert_array_equal(loaded_csc.toarray(), dense)

    def test_validation_errors_are_explicit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "array.cereal"
            with self.assertRaisesRegex(ValueError, "Only float32/64 arrays can be serialized"):
                serialize_array(np.arange(3, dtype="int64"), path)
            with self.assertRaisesRegex(ValueError, "Only 1d and 2d arrays can be serialized"):
                serialize_array(np.zeros((2, 2, 2), dtype="float64"), path)
            with self.assertRaisesRegex(FileNotFoundError, "does not exists"):
                load_array(Path(tmpdir) / "missing.cereal")

            serialize_array(np.arange(3, dtype="float64"), path)
            with self.assertRaisesRegex(ValueError, "Stored dtype"):
                load_array(path, dtype="float32")
            with self.assertRaisesRegex(ValueError, "Stored array type"):
                load_array(path, array_type="sparse")
            with self.assertRaisesRegex(ValueError, "Stored array has dimension"):
                load_array(path, array_dim=2)


if __name__ == "__main__":
    unittest.main()
