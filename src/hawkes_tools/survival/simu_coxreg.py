"""Cox proportional-hazards simulation matching tick's gallery API."""

from __future__ import annotations

import time

import numpy as np
import scipy.sparse as sps
from scipy.linalg import toeplitz

from hawkes_tools.linear_model import _as_1d_float, _as_2d_float

__all__ = ["SimuCoxReg", "SimuCoxRegWithCutPoints"]


def _positive_int(name, value) -> int:
    if isinstance(value, bool):
        raise ValueError(f"``{name}`` must be a positive integer")
    numeric = int(value)
    if numeric != value or numeric <= 0:
        raise ValueError(f"``{name}`` must be a positive integer")
    return numeric


def _nonnegative_int(name, value) -> int:
    if isinstance(value, bool):
        raise ValueError(f"``{name}`` must be a non-negative integer")
    numeric = int(value)
    if numeric != value or numeric < 0:
        raise ValueError(f"``{name}`` must be a non-negative integer")
    return numeric


def _positive_finite(name, value) -> float:
    numeric = float(value)
    if not np.isfinite(numeric) or numeric <= 0.0:
        raise ValueError(f"``{name}`` must be strictly positive")
    return numeric


def _finite(name, value) -> float:
    numeric = float(value)
    if not np.isfinite(numeric):
        raise ValueError(f"``{name}`` must be finite")
    return numeric


def _ratio(name, value, *, include_zero: bool = True) -> float:
    numeric = float(value)
    lower_ok = numeric >= 0.0 if include_zero else numeric > 0.0
    if not np.isfinite(numeric) or not lower_ok or numeric > 1.0:
        interval = "[0, 1]" if include_zero else "(0, 1]"
        raise ValueError(f"``{name}`` must be in {interval}")
    return numeric


def _simulate_feature_matrix(
    rng,
    n_samples: int,
    n_features: int,
    features_type: str,
    cov_corr: float,
    features_scaling: str,
    dtype,
) -> np.ndarray:
    features_type = features_type.lower()
    if features_type == "cov_toeplitz":
        covariance = toeplitz(cov_corr ** np.arange(n_features))
    elif features_type == "cov_uniform":
        covariance = rng.uniform(size=(n_features, n_features))
        np.fill_diagonal(covariance, 1.0)
        covariance = 0.5 * (covariance + covariance.T)
    elif features_type in {"none", "gaussian", "independent"}:
        return rng.normal(size=(n_samples, n_features)).astype(dtype)
    else:
        raise ValueError("features_type must be 'cov_toeplitz', 'cov_uniform', or 'none'")

    features = rng.multivariate_normal(np.zeros(n_features), covariance, size=n_samples)
    scaling = features_scaling.lower()
    if scaling == "standard":
        features = (features - features.mean(axis=0)) / np.maximum(features.std(axis=0), 1e-12)
    elif scaling == "min-max":
        minimum = features.min(axis=0)
        features = (features - minimum) / np.maximum(features.max(axis=0) - minimum, 1e-12)
    elif scaling == "norm":
        features = features / np.maximum(np.linalg.norm(features, axis=0), 1e-12)
    elif scaling != "none":
        raise ValueError("features_scaling must be 'standard', 'min-max', 'norm', or 'none'")
    return features.astype(dtype)


def _simulate_weibull_times(
    rng,
    features,
    coeffs,
    scale: float,
    shape: float,
    censoring_factor: float,
    dtype,
):
    linear_risk = np.asarray(features @ coeffs, dtype=float).reshape(-1)
    exponential = rng.exponential(scale=1.0, size=features.shape[0])
    exponential *= np.exp(-linear_risk)
    true_times = (1.0 / scale) * exponential ** (1.0 / shape)
    censoring_times = rng.exponential(
        scale=censoring_factor * true_times.mean(), size=true_times.size
    )
    times = np.minimum(true_times, censoring_times).astype(dtype)
    censoring = (true_times <= censoring_times).astype(np.ushort)
    return times, censoring


class SimuCoxReg:
    """Simulation of Cox regression triplets ``(features, times, censoring)``."""

    def __init__(
        self,
        coeffs,
        features=None,
        n_samples: int = 200,
        times_distribution: str = "weibull",
        shape: float = 1.0,
        scale: float = 1.0,
        censoring_factor: float = 2.0,
        features_type: str = "cov_toeplitz",
        cov_corr: float = 0.5,
        features_scaling: str = "none",
        seed: int | None = None,
        verbose: bool = True,
        dtype="float64",
    ):
        self.coeffs = _as_1d_float(coeffs)
        self.features = None if features is None else _as_2d_float(features).astype(dtype, copy=False)
        self.n_samples = _positive_int("n_samples", n_samples)
        self.times_distribution = times_distribution
        self.shape = _positive_finite("shape", shape)
        self.scale = _positive_finite("scale", scale)
        self.censoring_factor = _positive_finite("censoring_factor", censoring_factor)
        self.features_type = str(features_type)
        self.cov_corr = _finite("cov_corr", cov_corr)
        self.features_scaling = str(features_scaling)
        self.seed = seed
        self.verbose = bool(verbose)
        self.dtype = np.dtype(dtype)
        self.times = None
        self.censoring = None
        self.time_elapsed = None

        if self.times_distribution != "weibull":
            raise ValueError("``times_distribution`` was not understood, try using 'weibull' instead")

    def simulate(self):
        start = time.perf_counter()
        rng = np.random.default_rng(self.seed)
        if self.features is None:
            self.features = self._simulate_features(rng)
        if self.features.shape[1] != self.coeffs.size:
            raise ValueError("features and coeffs have incompatible dimensions")

        self.times, self.censoring = _simulate_weibull_times(
            rng,
            self.features,
            self.coeffs,
            self.scale,
            self.shape,
            self.censoring_factor,
            self.dtype,
        )
        self.time_elapsed = time.perf_counter() - start
        return self.features, self.times, self.censoring

    def _simulate_features(self, rng) -> np.ndarray:
        return _simulate_feature_matrix(
            rng,
            self.n_samples,
            self.coeffs.size,
            self.features_type,
            self.cov_corr,
            self.features_scaling,
            self.dtype,
        )


class SimuCoxRegWithCutPoints:
    """Simulation of Cox regression with piecewise-constant feature effects."""

    def __init__(
        self,
        features=None,
        n_samples: int = 200,
        n_features: int = 5,
        n_cut_points: int | None = None,
        n_cut_points_factor: float = 0.7,
        times_distribution: str = "weibull",
        shape: float = 1.0,
        scale: float = 1.0,
        censoring_factor: float = 2.0,
        features_type: str = "cov_toeplitz",
        cov_corr: float = 0.5,
        features_scaling: str = "none",
        seed: int | None = None,
        verbose: bool = True,
        sparsity: float = 0.0,
        dtype="float64",
    ):
        self.features = None if features is None else _as_2d_float(features).astype(dtype, copy=False)
        self.n_samples = _positive_int("n_samples", n_samples)
        self.n_features = _positive_int("n_features", n_features)
        self.n_cut_points = n_cut_points
        self.n_cut_points_factor = _ratio("n_cut_points_factor", n_cut_points_factor, include_zero=False)
        self.times_distribution = times_distribution
        self.shape = _positive_finite("shape", shape)
        self.scale = _positive_finite("scale", scale)
        self.censoring_factor = _positive_finite("censoring_factor", censoring_factor)
        self.features_type = str(features_type)
        self.cov_corr = _finite("cov_corr", cov_corr)
        self.features_scaling = str(features_scaling)
        self.seed = seed
        self.verbose = bool(verbose)
        self.sparsity = _ratio("sparsity", sparsity)
        self.dtype = np.dtype(dtype)
        self.times = None
        self.censoring = None
        self.time_elapsed = None

        if self.times_distribution != "weibull":
            raise ValueError("``times_distribution`` was not understood, try using 'weibull' instead")
        if self.n_cut_points is not None:
            self.n_cut_points = _nonnegative_int("n_cut_points", self.n_cut_points)

    def simulate(self):
        start = time.perf_counter()
        rng = np.random.default_rng(self.seed)
        if self.features is None:
            self.features = _simulate_feature_matrix(
                rng,
                self.n_samples,
                self.n_features,
                self.features_type,
                self.cov_corr,
                self.features_scaling,
                self.dtype,
            )
        else:
            self.n_samples, self.n_features = self.features.shape

        cut_points, coeffs_binarized, sparse_blocks = self._simulate_cut_points_and_coeffs(rng)
        binarized_features = self._binarize_features(cut_points)
        self.times, self.censoring = _simulate_weibull_times(
            rng,
            binarized_features,
            coeffs_binarized,
            self.scale,
            self.shape,
            self.censoring_factor,
            self.dtype,
        )
        self.time_elapsed = time.perf_counter() - start
        return (
            self.features,
            self.times,
            self.censoring,
            cut_points,
            coeffs_binarized,
            sparse_blocks,
        )

    def _binarize_features(self, cut_points):
        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []
        offset = 0
        for feature_idx in range(self.n_features):
            boundaries = cut_points[str(feature_idx)]
            n_intervals = len(boundaries) - 1
            intervals = np.searchsorted(
                boundaries, self.features[:, feature_idx], side="left"
            ) - 1
            intervals = np.clip(intervals, 0, n_intervals - 1)
            for row, interval in enumerate(intervals):
                rows.append(row)
                cols.append(offset + int(interval))
                data.append(1.0)
            offset += n_intervals
        return sps.csr_matrix(
            (data, (rows, cols)), shape=(self.n_samples, offset), dtype=float
        )

    def _simulate_cut_points_and_coeffs(self, rng):
        sparse_count = round(self.n_features * self.sparsity)
        if sparse_count:
            sparse_blocks = np.sort(rng.choice(self.n_features, sparse_count, replace=False))
        else:
            sparse_blocks = np.asarray([], dtype=int)

        if self.n_cut_points is None:
            n_cut_points = rng.geometric(self.n_cut_points_factor, self.n_features)
        else:
            n_cut_points = np.repeat(int(self.n_cut_points), self.n_features)

        cut_points: dict[str, np.ndarray] = {}
        coeffs_binarized: list[np.ndarray] = []
        quantile_cuts = np.linspace(10, 90, 10)
        sparse_set = set(sparse_blocks.tolist())

        for feature_idx in range(self.n_features):
            feature = self.features[:, feature_idx]
            try:
                candidates = np.percentile(feature, quantile_cuts, method="nearest")
            except TypeError:
                candidates = np.percentile(feature, quantile_cuts, interpolation="nearest")

            count = int(n_cut_points[feature_idx])
            cut_points_j = rng.choice(candidates, count, replace=False)
            cut_points_j = np.sort(cut_points_j)
            cut_points_j = np.insert(cut_points_j, 0, -np.inf)
            cut_points_j = np.append(cut_points_j, np.inf)
            cut_points[str(feature_idx)] = cut_points_j

            if feature_idx in sparse_set:
                coeffs_block = np.zeros(count + 1)
            else:
                coeffs_block = np.abs(rng.normal(1.0, 0.5, count + 1))
                coeffs_block[::2] *= -1
            coeffs_block = coeffs_block - coeffs_block.mean()
            coeffs_binarized.append(coeffs_block)

        return cut_points, np.concatenate(coeffs_binarized), sparse_blocks

