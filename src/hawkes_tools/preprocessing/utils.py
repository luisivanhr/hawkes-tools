"""Validation helpers for preprocessing transforms."""

from __future__ import annotations

from warnings import warn

import numpy as np


def _is_dataframe_like(value) -> bool:
    return hasattr(value, "values") and hasattr(value, "columns")


def safe_array(X, dtype=np.float64):
    """Return ``X`` with the requested dtype and C-contiguous dense layout."""

    if _is_dataframe_like(X):
        X = X.values

    if isinstance(X, np.ndarray) and not X.flags["C_CONTIGUOUS"]:
        warn(
            "Copying array of size %s to create a C-contiguous version of it"
            % (str(X.shape),),
            RuntimeWarning,
        )
        X = np.ascontiguousarray(X)

    if X.dtype != dtype:
        warn(
            "Copying array of size %s to convert it in the right format"
            % (str(X.shape),),
            RuntimeWarning,
        )
        X = X.astype(dtype)

    return X


def check_longitudinal_features_consistency(X, shape, dtype):
    """Validate that all longitudinal feature matrices share shape and dtype."""

    if not all(x.shape == shape for x in X):
        raise ValueError("All the elements of X should have the same shape.")
    return [safe_array(x, dtype) for x in X]


def check_censoring_consistency(censoring, n_samples):
    """Validate a one-dimensional uint64 censoring vector."""

    if censoring.shape != (n_samples,):
        raise ValueError(
            "`censoring` should be a 1-D numpy ndarray of shape (%i,)" % n_samples
        )
    return safe_array(censoring, "uint64")
