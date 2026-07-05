"""Self-controlled case series simulation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
import time

import numpy as np
from scipy import sparse
from scipy.stats import beta, norm

__all__ = ["CustomEffects", "SimuSCCS", "_lag_feature_matrix"]


def _validate_n_lags(n_lags, n_features: int | None = None, n_intervals: int | None = None) -> np.ndarray:
    values = np.asarray(n_lags)
    if values.ndim != 1:
        raise ValueError("n_lags must be a one-dimensional array")
    if values.size == 0:
        raise ValueError("n_lags must not be empty")
    if n_features is not None and values.size != int(n_features):
        raise ValueError("n_lags must have one entry per feature")
    numeric = np.asarray(values, dtype=float)
    if np.any(~np.isfinite(numeric)) or np.any(numeric != np.floor(numeric)):
        raise ValueError("n_lags must contain integer values")
    if np.any(numeric < 0):
        raise ValueError("n_lags elements should be greater than or equal to 0")
    if n_intervals is not None and np.any(numeric >= int(n_intervals)):
        raise ValueError("n_lags elements should be smaller than n_intervals")
    return numeric.astype(np.uint64)


@dataclass
class _ExpKernelSpec:
    adjacency: np.ndarray
    decays: np.ndarray
    baseline: np.ndarray

    def adjust_spectral_radius(self, spectral_radius: float) -> None:
        eigvals = np.linalg.eigvals(self.adjacency)
        radius = float(np.max(np.abs(eigvals))) if eigvals.size else 0.0
        if radius > 0.0:
            self.adjacency = self.adjacency * (float(spectral_radius) / radius)

    def track_intensity(self, intensity_track_step: float = -1.0) -> None:
        self.intensity_track_step = float(intensity_track_step)


class SimuSCCS:
    """Simulation of exposure, outcome, and censoring data for SCCS models."""

    def __init__(
        self,
        n_cases,
        n_intervals,
        n_features,
        n_lags,
        time_drift=None,
        exposure_type="single_exposure",
        distribution="multinomial",
        sparse=True,
        censoring_prob=0,
        censoring_scale=None,
        coeffs=None,
        hawkes_exp_kernels=None,
        n_correlations=0,
        batch_size=None,
        seed=None,
        verbose=True,
    ):
        self.n_cases = int(n_cases)
        self.n_intervals = int(n_intervals)
        self.n_features = int(n_features)
        if self.n_cases <= 0:
            raise ValueError("n_cases should be greater than 0")
        if self.n_intervals <= 0:
            raise ValueError("n_intervals should be greater than 0")
        if self.n_features <= 0:
            raise ValueError("n_features should be greater than 0")
        self.sparse = bool(sparse)
        self.seed = seed
        self.verbose = bool(verbose)
        self.time_drift = time_drift
        self.n_correlations = int(n_correlations)
        if self.n_correlations < 0:
            raise ValueError("n_correlations should be non-negative")
        self.hawkes_exp_kernels = hawkes_exp_kernels
        self.hawkes_obj = None
        self.time_elapsed = None

        self._n_lags = None
        self._features_offset = None
        self.n_lags = n_lags
        self._coeffs = None
        self.coeffs = coeffs
        self.exposure_type = exposure_type
        self.distribution = distribution
        self.censoring_prob = float(censoring_prob)
        if not 0.0 <= self.censoring_prob <= 1.0:
            raise ValueError("censoring_prob value should be in [0, 1].")
        self.censoring_scale = float(censoring_scale) if censoring_scale is not None else self.n_intervals / 4.0
        if self.censoring_scale <= 0:
            raise ValueError("censoring_scale should be greater than 0.")
        self.batch_size = int(batch_size) if batch_size is not None else self.n_cases
        if self.batch_size <= 0:
            raise ValueError("batch_size should be greater than 0")

    @property
    def n_lags(self):
        return self._n_lags

    @n_lags.setter
    def n_lags(self, value):
        n_lags = _validate_n_lags(value, n_features=self.n_features, n_intervals=self.n_intervals)
        offsets = [0]
        for lag in n_lags:
            offsets.append(offsets[-1] + int(lag) + 1)
        self._n_lags = n_lags
        self._features_offset = offsets

    @property
    def exposure_type(self):
        return self._exposure_type

    @exposure_type.setter
    def exposure_type(self, value):
        if value not in {"single_exposure", "multiple_exposures"}:
            raise ValueError("exposure_type can be only 'single_exposure' or 'multiple_exposures'.")
        self._exposure_type = value

    @property
    def distribution(self):
        return self._distribution

    @distribution.setter
    def distribution(self, value):
        if value not in {"multinomial", "poisson"}:
            raise ValueError("distribution can be only 'multinomial' or 'poisson'.")
        self._distribution = value

    @property
    def coeffs(self):
        if self._coeffs is None:
            return None
        value = []
        for i, lag in enumerate(self.n_lags):
            start = int(self._features_offset[i])
            end = int(start + lag + 1)
            value.append(self._coeffs[start:end].copy())
        return value

    @coeffs.setter
    def coeffs(self, value):
        if value is None:
            self._coeffs = None
            return
        pieces = []
        for i, coeff in enumerate(value):
            arr = np.asarray(coeff, dtype=float).reshape(-1)
            expected = int(self.n_lags[i]) + 1
            if arr.size != expected:
                raise ValueError(f"Coeffs {i} th element should be of shape (n_lags[{i}] + 1,)")
            pieces.append(arr)
        self._coeffs = np.hstack(pieces)

    def simulate(self):
        start = time.perf_counter()
        rng = np.random.default_rng(self.seed)
        if self._coeffs is None:
            n_lagged_features = int(self.n_lags.sum() + self.n_features)
            self._coeffs = rng.normal(1e-3, 1.1, n_lagged_features)

        features = self.simulate_features(self.n_cases, rng)
        censored_features = [feat.copy() for feat in features]
        labels = self.simulate_outcomes(features, rng)
        censoring = np.full(self.n_cases, self.n_intervals, dtype=np.uint64)

        if self.censoring_prob:
            censored = rng.binomial(1, self.censoring_prob, size=self.n_cases).astype(bool)
            censoring[censored] = np.maximum(
                1,
                censoring[censored] - rng.poisson(self.censoring_scale, size=int(censored.sum())).astype(np.uint64),
            )
            censored_features = self._censor_array_list(censored_features, censoring)
            labels = self._censor_array_list(labels, censoring)

        self.time_elapsed = time.perf_counter() - start
        return features, censored_features, labels, censoring, self.coeffs

    def simulate_features(self, n_samples, rng=None):
        rng = np.random.default_rng(self.seed) if rng is None else rng
        if self.exposure_type == "multiple_exposures":
            return [self._sim_multiple_exposures(rng) for _ in range(int(n_samples))]
        if self.exposure_type == "single_exposure":
            return self._sim_single_exposures(int(n_samples), rng)
        raise ValueError("unknown exposure_type")

    def _sim_multiple_exposures(self, rng):
        features = np.zeros((self.n_intervals, self.n_features), dtype=float)
        while features.sum() == 0:
            features = rng.integers(0, 2, size=(self.n_intervals, self.n_features)).astype(float)
        return sparse.csr_matrix(features) if self.sparse else features

    def _sim_single_exposures(self, n_samples: int, rng):
        if not self.sparse:
            raise ValueError("'single_exposure' exposures can only be simulated as sparse feature matrices")
        kernel_spec = self._ensure_kernel_spec(rng)
        if hasattr(kernel_spec, "adjust_spectral_radius"):
            kernel_spec.adjust_spectral_radius(0.1)
        adjacency = np.asarray(kernel_spec.adjacency, dtype=float)
        decays = np.asarray(getattr(kernel_spec, "decays", 0.002 * np.ones_like(adjacency)), dtype=float)
        baseline = np.asarray(kernel_spec.baseline, dtype=float).reshape(-1)

        starts = np.full((n_samples, self.n_features), -1, dtype=np.int64)
        state = np.zeros((n_samples, self.n_features), dtype=float)
        decay = np.exp(-np.mean(decays, axis=1)).reshape(1, -1)
        jump = adjacency * decays
        for t in range(self.n_intervals):
            active = starts < 0
            intensities = np.maximum(baseline.reshape(1, -1) + state, 0.0)
            probabilities = np.clip(1.0 - np.exp(-intensities), 0.0, 0.95)
            draws = (rng.random((n_samples, self.n_features)) < probabilities) & active
            starts[draws] = t
            state = state * decay + draws.astype(float) @ jump.T
            if not np.any(starts < 0):
                break

        missing = ~np.any(starts >= 0, axis=1)
        if np.any(missing):
            rows = np.flatnonzero(missing)
            starts[rows, rng.integers(0, self.n_features, size=rows.size)] = rng.integers(
                0, self.n_intervals, size=rows.size
            )

        return [self.to_coo(row, (self.n_intervals, self.n_features)) for row in starts]

    def _ensure_kernel_spec(self, rng):
        if self.hawkes_exp_kernels is not None:
            return self.hawkes_exp_kernels
        decays = 0.002 * np.ones((self.n_features, self.n_features))
        baseline = 4.0 * rng.random(self.n_features) / self.n_intervals
        adjacency = rng.random(self.n_features) * np.eye(self.n_features)
        if self.n_correlations:
            pairs = list(permutations(range(self.n_features), 2))
            chosen = rng.choice(len(pairs), size=min(self.n_correlations, len(pairs)), replace=False)
            for idx in np.atleast_1d(chosen):
                i, j = pairs[int(idx)]
                adjacency[i, j] = rng.random()
        self.hawkes_exp_kernels = _ExpKernelSpec(adjacency=adjacency, decays=decays, baseline=baseline)
        return self.hawkes_exp_kernels

    def simulate_outcomes(self, features, rng=None):
        rng = np.random.default_rng(self.seed) if rng is None else rng
        lagged_features = [_lag_feature_matrix(feat, self.n_lags) for feat in features]
        if self.distribution == "poisson":
            return self._simulate_poisson_outcomes(lagged_features, self._coeffs, rng)
        return self._simulate_multinomial_outcomes(lagged_features, self._coeffs, rng)

    def _simulate_multinomial_outcomes(self, lagged_features, coeffs, rng):
        baseline = np.zeros(self.n_intervals)
        if self.time_drift is not None:
            baseline = np.asarray(self.time_drift(np.arange(self.n_intervals)), dtype=float)
        outcomes = []
        for feat in lagged_features:
            dot_product = baseline + np.asarray(feat @ coeffs, dtype=float).reshape(-1)
            dot_product = dot_product - np.max(dot_product)
            probabilities = np.exp(dot_product)
            probabilities = probabilities / np.sum(probabilities)
            outcomes.append(rng.multinomial(1, probabilities).astype(np.int32))
        return outcomes

    def _simulate_poisson_outcomes(self, lagged_features, coeffs, rng, first_event_only=True):
        baseline = np.zeros(self.n_intervals)
        if self.time_drift is not None:
            baseline = np.asarray(self.time_drift(np.arange(self.n_intervals)), dtype=float)
        outcomes = []
        for feat in lagged_features:
            dot_product = baseline + np.asarray(feat @ coeffs, dtype=float).reshape(-1)
            dot_product = dot_product - np.max(dot_product)
            event_counts = rng.poisson(np.exp(dot_product))
            if first_event_only:
                y = np.zeros_like(event_counts, dtype=np.int32)
                positive = np.flatnonzero(event_counts > 0)
                if positive.size:
                    y[int(positive[0])] = 1
            else:
                y = event_counts.astype(np.int32)
            outcomes.append(y)
        return outcomes

    @staticmethod
    def _censor_array_list(array_list, censoring):
        def censor(array, censoring_idx):
            if sparse.issparse(array):
                out = array.tolil(copy=True)
                out[int(censoring_idx) :] = 0
                return out.tocsr()
            out = np.asarray(array).copy()
            out[int(censoring_idx) :] = 0
            return out

        return [censor(arr, censoring[i]) for i, arr in enumerate(array_list)]

    @staticmethod
    def _filter_non_positive_samples(features, features_censored, labels, censoring):
        positive = [i for i, arr in enumerate(labels) if np.asarray(arr).sum() > 0]
        if not positive:
            raise ValueError("There should be at least one positive sample per batch. Try to increase batch_size.")
        return (
            [features[i] for i in positive],
            [features_censored[i] for i in positive],
            [labels[i] for i in positive],
            np.asarray(censoring)[positive],
            np.asarray(positive, dtype=np.uint64),
        )

    @staticmethod
    def to_coo(feat, shape):
        feat = np.asarray(feat)
        cols = np.where(feat >= 0)[0]
        rows = feat[feat >= 0].astype(int)
        if cols.size == 0:
            cols = np.asarray([0])
            rows = np.asarray([0])
        data = np.ones(cols.size, dtype=float)
        return sparse.csr_matrix((data, (rows, cols)), shape=shape, dtype="float64")


def _lag_feature_matrix(feature_matrix, n_lags) -> sparse.csr_matrix:
    matrix = feature_matrix.tocsc() if sparse.issparse(feature_matrix) else sparse.csc_matrix(feature_matrix)
    n_intervals, n_features = matrix.shape
    n_lags = np.asarray(n_lags, dtype=np.uint64)
    if n_lags.size != n_features:
        raise ValueError("n_lags must have one entry per feature")
    total_columns = int(n_lags.sum() + n_features)
    rows = []
    cols = []
    data = []
    offset = 0
    for feature in range(n_features):
        col = matrix.getcol(feature).tocoo()
        for lag in range(int(n_lags[feature]) + 1):
            shifted_rows = col.row + lag
            mask = shifted_rows < n_intervals
            if np.any(mask):
                rows.append(shifted_rows[mask])
                cols.append(np.full(mask.sum(), offset + lag, dtype=np.int64))
                data.append(col.data[mask])
        offset += int(n_lags[feature]) + 1
    if not rows:
        return sparse.csr_matrix((n_intervals, total_columns), dtype=float)
    return sparse.csr_matrix(
        (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n_intervals, total_columns),
        dtype=float,
    )


class CustomEffects:
    def __init__(self, n_intervals):
        self.n_intervals = int(n_intervals)
        self._curves_type_dict = {
            1: (5, 1),
            2: (2, 2),
            3: (0.5, 0.5),
            4: (2, 5),
            5: (1, 3),
        }

    def constant_effect(self, amplitude, cut=0):
        risk_curve = np.ones(self.n_intervals) * float(amplitude)
        if cut > 0:
            risk_curve[int(cut) :] = 1.0
        return risk_curve

    def bell_shaped_effect(self, amplitude, width, lag=0, cut=0):
        self._check_params(lag, width, amplitude, cut)
        width = int(width)
        if width % 2 == 0:
            width += 1
        effect = norm(0, width / 5).pdf(np.arange(width) - int(width / 2))
        return self._create_risk_curve(effect, amplitude, cut, width, int(lag))

    def increasing_effect(self, amplitude, lag=0, cut=0, curvature_type=1):
        width = self.n_intervals
        self._check_params(lag, width, amplitude, cut)
        if curvature_type not in np.arange(5) + 1:
            raise ValueError("curvature type should be in {1, 2, 3, 4, 5}")
        a, b = self._curves_type_dict[int(curvature_type)]
        effect = beta(a, b).cdf(np.arange(width) / width)
        return self._create_risk_curve(effect, amplitude, cut, width, int(lag))

    def _check_params(self, lag, width, amplitude, cut):
        if cut is not None and cut >= width:
            raise ValueError("cut should be < width")
        if lag > self.n_intervals:
            raise ValueError("n_intervals should be > lag")
        if amplitude <= 0:
            raise ValueError("amplitude should be > 0")

    def _create_risk_curve(self, effect, amplitude, cut, width, lag):
        cut = int(cut or 0)
        if cut:
            effect = effect[: int(width - cut)]
        end_effect = int(lag + width - cut)
        if end_effect > self.n_intervals:
            end_effect = self.n_intervals
        effect = effect[: end_effect - lag]
        maximum = effect.max()
        minimum = effect.min()
        effect = (effect - minimum) / (maximum - minimum)
        effect *= float(amplitude) - 1.0
        risk_curve = np.ones(self.n_intervals)
        risk_curve[lag:end_effect] += effect
        return risk_curve

    @staticmethod
    def negative_effect(positive_effect):
        return np.exp(-np.log(positive_effect))
