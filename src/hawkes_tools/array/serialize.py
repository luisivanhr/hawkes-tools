"""Serialize dense and sparse floating-point arrays without tick's C++ layer."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import scipy.sparse

_SUPPORTED_DTYPES = (np.dtype("float32"), np.dtype("float64"))


def serialize_array(array, filepath):
    """Save a 1d/2d dense or 2d sparse float array to ``filepath``.

    The original tick helper writes a C++ cereal format. hawkes-tools has no C++
    runtime, so this standalone port writes an ``np.savez`` container to the
    exact requested path while preserving dtype, shape, sparse structure, and
    row/column-major sparse orientation.
    """

    dtype = np.dtype(array.dtype)
    if dtype not in _SUPPORTED_DTYPES:
        raise ValueError("Only float32/64 arrays can be serialized")

    path = Path(filepath)
    if scipy.sparse.issparse(array):
        if len(array.shape) != 2:
            raise ValueError("Only 2d sparse arrays can be serialized")
        sparse_format = "csc" if scipy.sparse.isspmatrix_csc(array) else "csr"
        stored = array.asformat(sparse_format)
        payload = {
            "kind": np.array("sparse"),
            "dtype": np.array(dtype.name),
            "format": np.array(sparse_format),
            "shape": np.asarray(stored.shape, dtype=np.int64),
            "data": np.asarray(stored.data, dtype=dtype),
            "indices": np.asarray(stored.indices, dtype=np.int64),
            "indptr": np.asarray(stored.indptr, dtype=np.int64),
        }
    else:
        dense = np.asarray(array)
        if dense.ndim not in (1, 2):
            raise ValueError("Only 1d and 2d arrays can be serialized")
        order = "F" if dense.ndim == 2 and dense.flags["F_CONTIGUOUS"] else "C"
        payload = {
            "kind": np.array("dense"),
            "dtype": np.array(dtype.name),
            "order": np.array(order),
            "array": np.asarray(dense, dtype=dtype, order=order),
        }

    with path.open("wb") as output:
        np.savez(output, **payload)
    return os.path.abspath(path)


def load_array(filepath, array_type="dense", array_dim=1, dtype="float64", major="row"):
    """Load an array written by :func:`serialize_array`.

    Parameters mirror tick's Python helper. ``array_type`` must be ``"dense"``
    or ``"sparse"``, ``array_dim`` must match the stored dimensionality, dtype
    must be ``"float32"``, ``"float64"``, or ``"double"``, and sparse
    ``major="col"`` returns CSC while row-major sparse data returns CSR.
    """

    path = Path(filepath)
    abspath = os.path.abspath(path)
    if not path.exists():
        raise FileNotFoundError(f"File {abspath} does not exists")

    expected_dtype = _normalize_dtype(dtype)
    if array_type not in {"dense", "sparse"}:
        raise ValueError("Cannot load this class of array")
    if int(array_dim) not in (1, 2):
        raise ValueError("Only 1d and 2d arrays can be loaded")
    if major not in {"row", "col"}:
        raise ValueError("major must be 'row' or 'col'")

    with np.load(path, allow_pickle=False) as payload:
        kind = str(payload["kind"])
        stored_dtype = _normalize_dtype(str(payload["dtype"]))
        if stored_dtype != expected_dtype:
            raise ValueError(
                f"Stored dtype {stored_dtype.name} does not match requested {expected_dtype.name}"
            )
        if kind != array_type:
            raise ValueError(f"Stored array type {kind} does not match requested {array_type}")

        if kind == "dense":
            array = np.asarray(payload["array"], dtype=expected_dtype)
            if array.ndim != int(array_dim):
                raise ValueError(f"Stored array has dimension {array.ndim}, expected {array_dim}")
            if int(array_dim) == 2 and major == "col":
                return np.asfortranarray(array)
            return np.ascontiguousarray(array) if int(array_dim) == 2 else array.copy()

        if int(array_dim) != 2:
            raise ValueError("Only 2d sparse arrays can be loaded")
        shape = tuple(int(v) for v in payload["shape"])
        data = np.asarray(payload["data"], dtype=expected_dtype)
        indices = np.asarray(payload["indices"], dtype=np.int64)
        indptr = np.asarray(payload["indptr"], dtype=np.int64)
        stored_format = str(payload["format"])
        if stored_format == "csc":
            matrix = scipy.sparse.csc_matrix((data, indices, indptr), shape=shape)
        elif stored_format == "csr":
            matrix = scipy.sparse.csr_matrix((data, indices, indptr), shape=shape)
        else:
            raise ValueError(f"Unsupported stored sparse format: {stored_format}")
        return matrix.tocsc() if major == "col" else matrix.tocsr()


def _normalize_dtype(dtype):
    normalized = np.dtype("float64" if dtype == "double" else dtype)
    if normalized not in _SUPPORTED_DTYPES:
        raise ValueError("Unhandled serialization type")
    return normalized
