"""Simulation helper functions compatible with :mod:`tick.simulation`."""

from __future__ import annotations

from warnings import warn

import numpy as np
from scipy.linalg import toeplitz

from hawkes_tools.linear_model import weights_sparse_gauss

__all__ = [
    "features_normal_cov_uniform",
    "features_normal_cov_toeplitz",
    "weights_sparse_exp",
    "weights_sparse_gauss",
]


def weights_sparse_exp(
    n_weigths: int = 100,
    nnz: int = 10,
    scale: float = 10.0,
    dtype="float64",
) -> np.ndarray:
    """Return tick's sparse exponentially decaying coefficient vector."""

    n_weigths = int(n_weigths)
    nnz = int(nnz)
    if nnz >= n_weigths:
        warn(
            "nnz must be smaller than n_weights using nnz=n_weigths instead",
            RuntimeWarning,
            stacklevel=2,
        )
        nnz = n_weigths
    idx = np.arange(nnz)
    out = np.zeros(n_weigths, dtype=dtype)
    out[:nnz] = np.exp(-idx / float(scale))
    out[:nnz:2] *= -1
    return out


def features_normal_cov_uniform(
    n_samples: int = 200,
    n_features: int = 30,
    dtype="float64",
) -> np.ndarray:
    """Generate Gaussian features with tick's random uniform covariance."""

    n_samples = int(n_samples)
    n_features = int(n_features)
    covariance_seed = np.random.uniform(size=(n_features, n_features))
    if dtype != "float64":
        covariance_seed = covariance_seed.astype(dtype)
    np.fill_diagonal(covariance_seed, 1.0)
    covariance = 0.5 * (covariance_seed + covariance_seed.T)
    features = np.random.multivariate_normal(
        np.zeros(n_features),
        covariance,
        size=n_samples,
    )
    if dtype != "float64":
        return features.astype(dtype)
    return features


def features_normal_cov_toeplitz(
    n_samples: int = 200,
    n_features: int = 30,
    cov_corr: float = 0.5,
    dtype="float64",
) -> np.ndarray:
    """Generate Gaussian features with tick's Toeplitz covariance."""

    n_samples = int(n_samples)
    n_features = int(n_features)
    covariance = toeplitz(float(cov_corr) ** np.arange(0, n_features))
    features = np.random.multivariate_normal(
        np.zeros(n_features),
        covariance,
        size=n_samples,
    )
    if dtype != "float64":
        return features.astype(dtype)
    return features
