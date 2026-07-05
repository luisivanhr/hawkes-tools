"""Pure-Python solver and proximal-operator utilities.

The Hawkes learners only need a small subset of tick's optimization toolbox,
but exposing the same regularization concepts is useful for custom Hawkes
models.  These classes favor correctness and a stable Python API over exact
C++ implementation parity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Callable

import numpy as np
from scipy.optimize import OptimizeResult, minimize
from scipy.stats import norm as normal_dist

from hawkes_tools.base import History, relative_distance

__all__ = [
    "History",
    "Solver",
    "GD",
    "AGD",
    "BFGS",
    "GFB",
    "SCPG",
    "SGD",
    "AdaGrad",
    "SVRG",
    "SAGA",
    "SDCA",
    "Prox",
    "ProxZero",
    "ProxPositive",
    "ProxL1",
    "ProxL1w",
    "ProxL2",
    "ProxL2Sq",
    "ProxElasticNet",
    "ProxNuclear",
    "ProxEquality",
    "ProxSlope",
    "ProxTV",
    "ProxGroupL1",
    "ProxBinarsity",
    "ProxMulti",
    "soft_threshold",
    "group_l1_shrink",
    "optimize_positive_coeffs",
]


def soft_threshold(x: np.ndarray, threshold: float | np.ndarray) -> np.ndarray:
    """Apply element-wise soft-thresholding."""

    x = np.asarray(x, dtype=float)
    threshold = np.asarray(threshold, dtype=float)
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0.0)


def _validate_range(value: tuple[int, int] | None) -> tuple[int, int] | None:
    if value is None:
        return None
    if len(value) != 2:
        raise ValueError("``range`` must be a tuple with 2 elements")
    start, end = int(value[0]), int(value[1])
    if start < 0 or end < 0:
        raise ValueError("``range`` entries must be non-negative")
    if start >= end:
        raise ValueError("first element must be smaller than second element in ``range``")
    return start, end


def _validate_nonnegative_finite(name: str, value) -> float:
    value = float(value)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be a non-negative finite number")
    return value


def _validate_ratio(value) -> float:
    value = float(value)
    if not np.isfinite(value) or value < 0.0 or value > 1.0:
        raise ValueError("ratio must be between 0 and 1")
    return value


def _validate_positive_finite(name: str, value) -> float | None:
    if value is None:
        return None
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be a positive finite number")
    return value


def _validate_solver_tol(value) -> float:
    value = float(value)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError("tol must be a non-negative finite number")
    return value


def _validate_int(name: str, value, *, minimum: int, optional: bool = False) -> int | None:
    if optional and value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    try:
        int_value = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if int_value != value and not isinstance(value, np.integer):
        raise ValueError(f"{name} must be an integer")
    if int_value < minimum:
        if minimum == 0:
            raise ValueError(f"{name} must be non-negative")
        raise ValueError(f"{name} must be at least {minimum}")
    return int_value


def _validate_seed(value) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("seed must be an integer")
    try:
        seed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("seed must be an integer") from exc
    if seed != value and not isinstance(value, np.integer):
        raise ValueError("seed must be an integer")
    return None if seed < 0 else seed


def _validate_choice(name: str, value, allowed: set[str], aliases: dict[str, str] | None = None) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    normalized = value.lower()
    if aliases is not None:
        normalized = aliases.get(normalized, normalized)
    if normalized not in allowed:
        choices = ", ".join(sorted(allowed))
        raise ValueError(f"{name} must be one of {choices}")
    return normalized


def _step_for_segment(step: float | np.ndarray, size: int) -> float | np.ndarray:
    step_arr = np.asarray(step, dtype=float)
    if step_arr.ndim == 0:
        step_value = float(step_arr)
        if not np.isfinite(step_value) or step_value < 0.0:
            raise ValueError("step must be non-negative and finite")
        return step_value
    if step_arr.size != size:
        raise ValueError("step must be scalar or have the selected range length")
    if np.any(~np.isfinite(step_arr)) or np.any(step_arr < 0.0):
        raise ValueError("step must be non-negative and finite")
    return step_arr.reshape((size,))


class Prox:
    """Base class for proximal operators.

    Parameters
    ----------
    range : tuple of int, optional
        Half-open slice where the operator is applied. Entries outside this
        range are copied through unchanged.
    """

    positive: bool = False

    def __init__(self, range: tuple[int, int] | None = None):
        self._range = None
        self.range = range
        self.dtype = np.dtype("float64")

    @property
    def range(self) -> tuple[int, int] | None:
        return self._range

    @range.setter
    def range(self, value: tuple[int, int] | None) -> None:
        self._range = _validate_range(value)

    @property
    def name(self) -> str:
        return self.__class__.__name__

    def call(self, coeffs: np.ndarray, step: float | np.ndarray = 1.0, out: np.ndarray | None = None) -> np.ndarray:
        coeffs_arr = np.asarray(coeffs, dtype=float)
        if out is None:
            result = coeffs_arr.copy()
        else:
            result = out
            if result.shape != coeffs_arr.shape:
                raise ValueError("out must have the same shape as coeffs")
            result[...] = coeffs_arr

        selected = self._selected_view(result)
        selected_shape = selected.shape
        selected_flat = np.asarray(selected, dtype=float).reshape(-1)
        selected_step = _step_for_segment(step, selected_flat.size)
        updated = np.asarray(self._call_segment(selected_flat.copy(), selected_step), dtype=float)
        updated = self._apply_positive(updated).reshape(selected_shape)
        self._assign_selected(result, updated)
        return result

    def value(self, coeffs: np.ndarray) -> float:
        selected = self._selected_values(coeffs).reshape(-1)
        if getattr(self, "positive", False) and np.any(selected < 0.0):
            return float(np.inf)
        return float(self._value_segment(selected))

    def astype(self, dtype_or_object_with_dtype):
        dtype = getattr(dtype_or_object_with_dtype, "dtype", dtype_or_object_with_dtype)
        self.dtype = np.dtype(dtype)
        return self

    def _selected_view(self, coeffs: np.ndarray) -> np.ndarray:
        if self.range is None:
            return coeffs
        start, end = self.range
        return coeffs.reshape(-1)[start:end]

    def _selected_values(self, coeffs: np.ndarray) -> np.ndarray:
        arr = np.asarray(coeffs, dtype=float)
        if self.range is None:
            return arr
        start, end = self.range
        return arr.reshape(-1)[start:end]

    def _assign_selected(self, coeffs: np.ndarray, values: np.ndarray) -> None:
        if self.range is None:
            coeffs[...] = values.reshape(coeffs.shape)
            return
        start, end = self.range
        coeffs.reshape(-1)[start:end] = values.reshape(-1)

    def _apply_positive(self, values: np.ndarray) -> np.ndarray:
        if getattr(self, "positive", False):
            return np.maximum(values, 0.0)
        return values

    def _call_segment(self, values: np.ndarray, step: float | np.ndarray) -> np.ndarray:
        return values

    def _value_segment(self, values: np.ndarray) -> float:
        return 0.0


class ProxZero(Prox):
    """Identity prox for the null penalty."""


class ProxPositive(Prox):
    """Projection onto the non-negative orthant."""

    def __init__(self, range: tuple[int, int] | None = None, positive: bool = False):
        del positive
        super().__init__(range=range)
        self.positive = True

    def _call_segment(self, values: np.ndarray, step: float | np.ndarray) -> np.ndarray:
        del step
        return np.maximum(values, 0.0)


class ProxL1(Prox):
    """L1 soft-thresholding prox."""

    def __init__(self, strength: float = 1.0, range: tuple[int, int] | None = None, positive: bool = False):
        super().__init__(range=range)
        self.strength = _validate_nonnegative_finite("strength", strength)
        self.positive = bool(positive)

    def _call_segment(self, values: np.ndarray, step: float | np.ndarray) -> np.ndarray:
        return soft_threshold(values, self.strength * step)

    def _value_segment(self, values: np.ndarray) -> float:
        return float(self.strength * np.sum(np.abs(values)))


class ProxL1w(ProxL1):
    """Weighted L1 soft-thresholding prox."""

    def __init__(
        self,
        strength: float,
        weights: np.ndarray,
        range: tuple[int, int] | None = None,
        positive: bool = False,
    ):
        super().__init__(strength=strength, range=range, positive=positive)
        self.weights = np.asarray(weights, dtype=float)
        if self.weights.ndim != 1:
            raise ValueError("weights must be one-dimensional")
        if np.any(~np.isfinite(self.weights)) or np.any(self.weights < 0.0):
            raise ValueError("weights must contain non-negative finite values")

    def _weights_for(self, values: np.ndarray) -> np.ndarray:
        if self.weights.size != values.size:
            raise ValueError("Size of ``weights`` does not match the selected range")
        return self.weights

    def _call_segment(self, values: np.ndarray, step: float | np.ndarray) -> np.ndarray:
        return soft_threshold(values, self.strength * self._weights_for(values) * step)

    def _value_segment(self, values: np.ndarray) -> float:
        return float(self.strength * np.sum(self._weights_for(values) * np.abs(values)))


class ProxL2Sq(Prox):
    """Squared L2, or ridge, prox."""

    def __init__(self, strength: float = 1.0, range: tuple[int, int] | None = None, positive: bool = False):
        super().__init__(range=range)
        self.strength = _validate_nonnegative_finite("strength", strength)
        self.positive = bool(positive)

    def _call_segment(self, values: np.ndarray, step: float | np.ndarray) -> np.ndarray:
        return values / (1.0 + self.strength * step)

    def _value_segment(self, values: np.ndarray) -> float:
        return float(0.5 * self.strength * np.dot(values, values))


class ProxL2(ProxL2Sq):
    """L2 norm prox, useful for group-lasso blocks."""

    def _call_segment(self, values: np.ndarray, step: float | np.ndarray) -> np.ndarray:
        step_scalar = float(np.mean(step)) if np.asarray(step).ndim else float(step)
        return group_l1_shrink(values, self.strength * np.sqrt(values.size) * step_scalar, axis=0)

    def _value_segment(self, values: np.ndarray) -> float:
        return float(self.strength * np.sqrt(values.size) * np.linalg.norm(values))


class ProxElasticNet(Prox):
    """Elastic-net prox combining L1 and squared L2 penalties."""

    def __init__(
        self,
        strength: float = 1.0,
        ratio: float = 0.95,
        range: tuple[int, int] | None = None,
        positive: bool = False,
    ):
        super().__init__(range=range)
        self.strength = _validate_nonnegative_finite("strength", strength)
        self.ratio = _validate_ratio(ratio)
        self.positive = bool(positive)

    def _call_segment(self, values: np.ndarray, step: float | np.ndarray) -> np.ndarray:
        l1 = soft_threshold(values, self.strength * self.ratio * step)
        return l1 / (1.0 + self.strength * (1.0 - self.ratio) * step)

    def _value_segment(self, values: np.ndarray) -> float:
        return float(
            self.strength * self.ratio * np.sum(np.abs(values))
            + 0.5 * self.strength * (1.0 - self.ratio) * np.dot(values, values)
        )


class ProxNuclear(Prox):
    """Nuclear norm prox for flattened or two-dimensional matrices."""

    def __init__(
        self,
        strength: float = 1.0,
        n_rows: int | None = None,
        range: tuple[int, int] | None = None,
        positive: bool = False,
    ):
        super().__init__(range=range)
        self.strength = _validate_nonnegative_finite("strength", strength)
        if n_rows is not None and int(n_rows) <= 0:
            raise ValueError("n_rows must be positive")
        self.n_rows = None if n_rows is None else int(n_rows)
        self.positive = bool(positive)
        self.rank_max = None

    def call(self, coeffs: np.ndarray, step: float | np.ndarray = 1.0, out: np.ndarray | None = None) -> np.ndarray:
        coeffs_arr = np.asarray(coeffs, dtype=float)
        _step_for_segment(step, 1)
        result = coeffs_arr.copy() if out is None else out
        if out is not None:
            if result.shape != coeffs_arr.shape:
                raise ValueError("out must have the same shape as coeffs")
            result[...] = coeffs_arr
        selected = self._selected_values(result)
        matrix, output_shape = self._matrix(selected)
        u, s, vh = np.linalg.svd(matrix, full_matrices=False)
        step_scalar = float(np.mean(step)) if np.asarray(step).ndim else float(step)
        s = np.maximum(s - self.strength * step_scalar, 0.0)
        updated = (u * s) @ vh
        updated = self._apply_positive(updated).reshape(output_shape)
        self._assign_selected(result, updated)
        return result

    def value(self, coeffs: np.ndarray) -> float:
        selected = self._selected_values(coeffs)
        if self.positive and np.any(selected < 0.0):
            return float(np.inf)
        matrix, _ = self._matrix(selected)
        return float(self.strength * np.sum(np.linalg.svd(matrix, compute_uv=False)))

    def _matrix(self, coeffs: np.ndarray) -> tuple[np.ndarray, tuple[int, ...]]:
        coeffs = np.asarray(coeffs, dtype=float)
        if coeffs.ndim == 2:
            return coeffs, coeffs.shape
        flat = coeffs.reshape(-1)
        n_rows = self.n_rows
        if n_rows is None:
            n_rows = int(round(np.sqrt(flat.size)))
            if n_rows <= 0 or flat.size % n_rows != 0:
                raise ValueError("'n_rows' parameter must be set for non-square flattened inputs")
        if flat.size % int(n_rows) != 0:
            raise ValueError("selected coefficient length must be a multiple of ``n_rows``")
        matrix = flat.reshape((int(n_rows), flat.size // int(n_rows)))
        return matrix, flat.shape


class ProxEquality(Prox):
    """Projection onto vectors whose selected entries are all equal."""

    def __init__(self, strength: float = 0.0, range: tuple[int, int] | None = None, positive: bool = False):
        del strength
        super().__init__(range=range)
        self.positive = bool(positive)

    @property
    def strength(self):
        return None

    @strength.setter
    def strength(self, value):
        del value

    def _call_segment(self, values: np.ndarray, step: float | np.ndarray) -> np.ndarray:
        del step
        mean = float(np.mean(values)) if values.size else 0.0
        if self.positive:
            mean = max(mean, 0.0)
        return np.full_like(values, mean)

    def _value_segment(self, values: np.ndarray) -> float:
        if values.size <= 1 or np.allclose(values, values[0]):
            return 0.0
        return float(np.inf)


class ProxTV(ProxL1):
    """Total-variation prox on a one-dimensional selected segment."""

    def _call_segment(self, values: np.ndarray, step: float | np.ndarray) -> np.ndarray:
        if values.size <= 1 or self.strength == 0.0:
            return values
        step_scalar = float(np.mean(step)) if np.asarray(step).ndim else float(step)

        def objective(x):
            return 0.5 * float(np.dot(x - values, x - values)) + step_scalar * self._value_segment(x)

        bounds = [(0.0, None)] * values.size if self.positive else None
        result = minimize(objective, values, method="L-BFGS-B", bounds=bounds)
        return result.x if result.success else values

    def _value_segment(self, values: np.ndarray) -> float:
        return float(self.strength * np.sum(np.abs(np.diff(values))))


class ProxSlope(ProxL1):
    """Sorted L1 prox with Benjamini-Hochberg SLOPE weights."""

    def __init__(
        self,
        strength: float,
        fdr: float = 0.6,
        range: tuple[int, int] | None = None,
        positive: bool = False,
    ):
        super().__init__(strength=strength, range=range, positive=positive)
        if not 0.0 < fdr < 1.0:
            raise ValueError("fdr must be between 0 and 1")
        self.fdr = float(fdr)
        self.weights = None

    def _weights_for(self, size: int) -> np.ndarray:
        ranks = np.arange(1, size + 1, dtype=float)
        weights = normal_dist.isf(self.fdr * ranks / (2.0 * size))
        self.weights = weights
        return weights

    def _call_segment(self, values: np.ndarray, step: float | np.ndarray) -> np.ndarray:
        if values.size == 0:
            return values
        step_scalar = float(np.mean(step)) if np.asarray(step).ndim else float(step)
        threshold = step_scalar * self.strength * self._weights_for(values.size)
        return _sorted_l1_prox(values, threshold, positive=self.positive)

    def _value_segment(self, values: np.ndarray) -> float:
        sorted_abs = np.sort(np.abs(values))[::-1]
        return float(self.strength * np.dot(self._weights_for(values.size), sorted_abs))


def _sorted_l1_prox(values: np.ndarray, threshold: np.ndarray, positive: bool = False) -> np.ndarray:
    """Proximal operator of the sorted L1 norm.

    The solution is obtained by projecting ``sort(abs(values)) - threshold`` on
    the non-increasing non-negative cone with the pool-adjacent-violators
    algorithm, then restoring signs and the original order.
    """

    values = np.asarray(values, dtype=float)
    threshold = np.asarray(threshold, dtype=float)
    if values.size != threshold.size:
        raise ValueError("threshold must have the same size as values")

    if positive:
        signs = np.ones_like(values)
        magnitudes = np.maximum(values, 0.0)
    else:
        signs = np.sign(values)
        magnitudes = np.abs(values)

    order = np.argsort(magnitudes)[::-1]
    sorted_magnitudes = magnitudes[order]
    projected = _isotonic_nonincreasing(sorted_magnitudes - threshold)
    projected = np.maximum(projected, 0.0)

    out = np.zeros_like(values)
    out[order] = projected
    return signs * out


def _isotonic_nonincreasing(values: np.ndarray) -> np.ndarray:
    """Project a vector onto the non-increasing cone."""

    levels: list[float] = []
    weights: list[int] = []
    starts: list[int] = []
    ends: list[int] = []

    for index, value in enumerate(np.asarray(values, dtype=float)):
        levels.append(float(value))
        weights.append(1)
        starts.append(index)
        ends.append(index + 1)
        while len(levels) >= 2 and levels[-2] < levels[-1]:
            total_weight = weights[-2] + weights[-1]
            merged = (levels[-2] * weights[-2] + levels[-1] * weights[-1]) / total_weight
            levels[-2] = float(merged)
            weights[-2] = total_weight
            ends[-2] = ends[-1]
            levels.pop()
            weights.pop()
            starts.pop()
            ends.pop()

    projected = np.empty_like(values, dtype=float)
    for level, start, end in zip(levels, starts, ends):
        projected[start:end] = level
    return projected


class ProxGroupL1(Prox):
    """Group-lasso prox over non-overlapping blocks."""

    def __init__(
        self,
        strength: float,
        blocks_start,
        blocks_length,
        range: tuple[int, int] | None = None,
        positive: bool = False,
    ):
        super().__init__(range=range)
        self.strength = _validate_nonnegative_finite("strength", strength)
        self.positive = bool(positive)
        self.blocks_start = np.asarray(blocks_start, dtype=np.int64)
        self.blocks_length = np.asarray(blocks_length, dtype=np.int64)
        self._validate_blocks()

    @property
    def n_blocks(self) -> int:
        return int(self.blocks_start.size)

    def _validate_blocks(self) -> None:
        if self.blocks_start.shape != self.blocks_length.shape:
            raise ValueError("``blocks_start`` and ``blocks_length`` must have the same size")
        if np.any(self.blocks_start < 0):
            raise ValueError("all blocks must have non-negative starting indices")
        if np.any(self.blocks_length <= 0):
            raise ValueError("all blocks must be of positive size")
        if np.any(self.blocks_start[1:] < self.blocks_start[:-1]):
            raise ValueError("``block_start`` must be sorted")
        if np.any(self.blocks_start[1:] < self.blocks_start[:-1] + self.blocks_length[:-1]):
            raise ValueError("blocks must not overlap")
        if self.range is not None and self.blocks_start.size:
            selected_size = self.range[1] - self.range[0]
            if np.max(self.blocks_start + self.blocks_length) > selected_size:
                raise ValueError("last block is not within the selected range")

    def _check_blocks_fit_values(self, values: np.ndarray) -> None:
        if self.blocks_start.size and np.max(self.blocks_start + self.blocks_length) > values.size:
            raise ValueError("last block is not within the selected values")

    def _call_segment(self, values: np.ndarray, step: float | np.ndarray) -> np.ndarray:
        self._check_blocks_fit_values(values)
        step_scalar = float(np.mean(step)) if np.asarray(step).ndim else float(step)
        out = values.copy()
        for start, length in zip(self.blocks_start, self.blocks_length):
            start_i, end_i = int(start), int(start + length)
            block = out[start_i:end_i]
            out[start_i:end_i] = group_l1_shrink(block, self.strength * np.sqrt(block.size) * step_scalar, axis=0)
        return out

    def _value_segment(self, values: np.ndarray) -> float:
        self._check_blocks_fit_values(values)
        total = 0.0
        for start, length in zip(self.blocks_start, self.blocks_length):
            block = values[int(start) : int(start + length)]
            total += np.sqrt(block.size) * np.linalg.norm(block)
        return float(self.strength * total)


class ProxBinarsity(ProxGroupL1):
    """Binarsity-style block TV prox with block centering."""

    def _call_segment(self, values: np.ndarray, step: float | np.ndarray) -> np.ndarray:
        self._check_blocks_fit_values(values)
        out = values.copy()
        tv = ProxTV(self.strength, positive=self.positive)
        for start, length in zip(self.blocks_start, self.blocks_length):
            start_i, end_i = int(start), int(start + length)
            block = tv.call(out[start_i:end_i], step=step)
            out[start_i:end_i] = block - np.mean(block)
        return out

    def _value_segment(self, values: np.ndarray) -> float:
        self._check_blocks_fit_values(values)
        total = 0.0
        for start, length in zip(self.blocks_start, self.blocks_length):
            block = values[int(start) : int(start + length)]
            total += np.sum(np.abs(np.diff(block)))
        return float(self.strength * total)


class ProxMulti(Prox):
    """Sequential composition of proximal operators."""

    def __init__(self, proxs: tuple[Prox, ...] | list[Prox]):
        super().__init__(range=None)
        self.proxs = list(proxs) if proxs else [ProxZero()]
        for prox in self.proxs:
            if not isinstance(prox, Prox):
                raise ValueError(f"{prox.__class__.__name__} is not a Prox")

    def call(self, coeffs: np.ndarray, step: float | np.ndarray = 1.0, out: np.ndarray | None = None) -> np.ndarray:
        result = np.asarray(coeffs, dtype=float).copy()
        for prox in self.proxs:
            result = prox.call(result, step=step)
        if out is not None:
            out[...] = result
            return out
        return result

    def value(self, coeffs: np.ndarray) -> float:
        return float(sum(prox.value(coeffs) for prox in self.proxs))


def group_l1_shrink(coeffs: np.ndarray, strength: float, axis: int = -1) -> np.ndarray:
    coeffs = np.asarray(coeffs, dtype=float)
    norms = np.linalg.norm(coeffs, axis=axis, keepdims=True)
    scale = np.maximum(1.0 - strength / np.maximum(norms, 1e-15), 0.0)
    return coeffs * scale


@dataclass
class Solver:
    """Small first-order solver API compatible with Hawkes model objects."""

    step: float | None = None
    tol: float = 1e-5
    max_iter: int = 100
    verbose: bool = False
    print_every: int = 10
    record_every: int = 10
    seed: int | None = None
    random_state: int | None = None
    n_threads: int = 1
    epoch_size: int | None = None
    rand_type: str = "unif"
    batch_size: int | None = None
    variance_reduction: str = "last"
    step_type: str = "fixed"
    linesearch: bool = True
    method: str = "L-BFGS-B"
    model: object | None = field(default=None, init=False)
    prox: Prox | None = field(default=None, init=False)
    history: History = field(default_factory=History, init=False)
    solution: np.ndarray | None = field(default=None, init=False)
    time_elapsed: float | None = field(default=None, init=False)
    _solve_started_at: float = field(default=0.0, init=False)
    _warned_deterministic_stochastic: bool = field(default=False, init=False)

    def __setattr__(self, name: str, value) -> None:
        if name == "step":
            value = _validate_positive_finite("step", value)
        elif name == "tol":
            value = _validate_solver_tol(value)
        elif name == "max_iter":
            value = _validate_int("max_iter", value, minimum=0)
        elif name in {"print_every", "record_every", "n_threads"}:
            value = _validate_int(name, value, minimum=1)
        elif name in {"epoch_size", "batch_size"}:
            value = _validate_int(name, value, minimum=1, optional=True)
        elif name in {"seed", "random_state"}:
            value = _validate_seed(value)
        elif name == "rand_type":
            value = _validate_choice("rand_type", value, {"perm", "unif"})
        elif name == "variance_reduction":
            value = _validate_choice("variance_reduction", value, {"avg", "last", "rand"})
        elif name == "step_type":
            value = _validate_choice(
                "step_type",
                value,
                {"bb", "fixed"},
                aliases={"barzilai-borwein": "bb", "barzilai_borwein": "bb"},
            )
        super().__setattr__(name, value)

    @property
    def name(self) -> str:
        return self.__class__.__name__

    def set_model(self, model):
        if not callable(getattr(model, "loss", None)) or not callable(getattr(model, "grad", None)):
            raise ValueError("model must expose loss(coeffs) and grad(coeffs)")
        n_coeffs = getattr(model, "n_coeffs", None)
        if n_coeffs is not None:
            _validate_int("model.n_coeffs", n_coeffs, minimum=1)
        self.model = model
        return self

    def set_prox(self, prox: Prox):
        if not isinstance(prox, Prox):
            raise ValueError(f"Passed object of class {prox.__class__.__name__} is not a Prox class")
        self.prox = prox
        return self

    def objective(self, coeffs: np.ndarray, loss: float | None = None) -> float:
        if self.model is None:
            raise ValueError("You must first set the model using ``set_model``.")
        model_loss = self.model.loss(coeffs) if loss is None else loss
        prox_value = 0.0 if self.prox is None else self.prox.value(coeffs)
        return float(model_loss + prox_value)

    def solve(self, x0: np.ndarray | None = None, step: float | None = None) -> np.ndarray:
        if self.model is None:
            raise ValueError("You must first set the model using ``set_model``.")
        if self.prox is None:
            self.prox = ProxZero()
        if self.seed is None and self.random_state is not None:
            self.seed = self.random_state
        if x0 is None:
            n_coeffs = getattr(self.model, "n_coeffs", None)
            if n_coeffs is None:
                raise ValueError("x0 is required when model.n_coeffs is unavailable")
            n_coeffs = _validate_int("model.n_coeffs", n_coeffs, minimum=1)
            x = np.zeros(int(n_coeffs), dtype=float)
        else:
            x = self._validate_x0(x0)

        self.history.clear()
        start_time = perf_counter()
        self._solve_started_at = start_time
        self._warned_deterministic_stochastic = False
        if isinstance(self, BFGS):
            x = self._solve_bfgs(x)
        elif isinstance(self, (SGD, SVRG, SAGA)) and hasattr(self.model, "batch_grad"):
            x = self._solve_stochastic(x, step=step)
        else:
            x = self._solve_first_order(x, step=step)
        self.time_elapsed = perf_counter() - start_time
        self.solution = np.asarray(x, dtype=float)
        return self.solution

    def get_history(self, key=None):
        if key is None:
            return self.history.records
        return [record[key] for record in self.history.records if key in record]

    def _solve_bfgs(self, x: np.ndarray) -> np.ndarray:
        def objective(z):
            return self.objective(z)

        use_jac = isinstance(self.prox, (ProxZero, ProxL2Sq))

        def jac(z):
            grad = np.asarray(self.model.grad(z), dtype=float)
            if isinstance(self.prox, ProxL2Sq):
                selected = np.zeros_like(z)
                selected_values = self.prox._selected_values(z)
                selected_update = self.prox.strength * selected_values
                if self.prox.range is None:
                    selected[...] = selected_update.reshape(selected.shape)
                else:
                    start, end = self.prox.range
                    selected.reshape(-1)[start:end] = selected_update.reshape(-1)
                grad += selected
            return grad

        result = minimize(
            objective,
            x,
            jac=jac if use_jac else None,
            method=self.method,
            options={"maxiter": int(self.max_iter), "ftol": float(self.tol), "gtol": float(self.tol)},
            callback=lambda z: self._record(len(self.history) + 1, z),
        )
        self._record(max(len(self.history), 0), result.x, force=True)
        return result.x

    def _solve_first_order(self, x: np.ndarray, step: float | None = None) -> np.ndarray:
        if int(self.max_iter) <= 0:
            self._record(0, x, force=True)
            return x
        step_value = self._initial_step(step)
        prev_obj = self.objective(x)
        accumulator = np.zeros_like(x)
        velocity = np.zeros_like(x)
        y = x.copy()
        t = 1.0
        prev_x_for_bb = None
        prev_grad_for_bb = None

        for n_iter in range(1, int(self.max_iter) + 1):
            grad_point = y if isinstance(self, AGD) else x
            grad = np.asarray(self.model.grad(grad_point), dtype=float)
            if str(self.step_type).lower() in {"bb", "barzilai-borwein", "barzilai_borwein"}:
                if prev_x_for_bb is not None and prev_grad_for_bb is not None:
                    s = grad_point - prev_x_for_bb
                    y_diff = grad - prev_grad_for_bb
                    denom = float(np.dot(s, y_diff))
                    if denom > 1e-30:
                        candidate = float(np.dot(s, s) / denom)
                        if np.isfinite(candidate) and candidate > 0.0:
                            step_value = candidate
                prev_x_for_bb = grad_point.copy()
                prev_grad_for_bb = grad.copy()
            if isinstance(self, AdaGrad):
                accumulator += grad * grad
                scaled_step = step_value / (np.sqrt(accumulator) + 1e-12)
                next_x = self.prox.call(x - scaled_step * grad, step=scaled_step)
            else:
                next_x = self.prox.call(grad_point - step_value * grad, step=step_value)

            if isinstance(self, AGD):
                next_t = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
                velocity = next_x + ((t - 1.0) / next_t) * (next_x - x)
                y = velocity
                t = next_t

            rel_delta = relative_distance(next_x, x)
            x = next_x
            obj = self.objective(x)
            rel_obj = abs(obj - prev_obj) / max(abs(prev_obj), 1.0)
            if self._should_record_iter(n_iter) or rel_obj < self.tol:
                self.history.append(
                    n_iter=n_iter,
                    obj=obj,
                    rel_obj=rel_obj,
                    rel_delta=rel_delta,
                    step=step_value,
                    time=perf_counter() - self._solve_started_at,
                )
            if rel_obj < self.tol:
                break
            prev_obj = obj
        return x

    def _solve_stochastic(self, x: np.ndarray, step: float | None = None) -> np.ndarray:
        if int(self.max_iter) <= 0:
            self._record(0, x, force=True)
            return x
        step_value = self._initial_step(step)
        n_samples = int(getattr(self.model, "n_samples", 0))
        if n_samples <= 0:
            return self._solve_first_order(x, step=step_value)
        epoch_size = int(self.epoch_size) if self.epoch_size is not None else n_samples
        epoch_size = max(epoch_size, 1)
        batch_size = self._stochastic_batch_size(n_samples)
        variance_reduction = str(self.variance_reduction).lower()
        if variance_reduction not in {"last", "avg", "rand"}:
            raise ValueError("variance_reduction must be 'last', 'avg', or 'rand'")
        rng = np.random.default_rng(self.seed)
        prev_obj = self.objective(x)
        prev_x_for_bb = None
        prev_grad_for_bb = None
        stored_residuals = None
        average_grad = None
        if isinstance(self, SAGA):
            stored_residuals = self.model.sample_residuals(x)
            all_indices = np.arange(n_samples, dtype=np.int64)
            average_grad = self.model.grad_from_residuals(all_indices, stored_residuals, n_samples)

        for n_iter in range(1, int(self.max_iter) + 1):
            epoch_start = x.copy()
            snapshot = None
            full_grad = None
            if isinstance(self, SVRG):
                snapshot = x.copy()
                full_grad = np.asarray(self.model.grad(snapshot), dtype=float)
                if str(self.step_type).lower() in {"bb", "barzilai-borwein", "barzilai_borwein"}:
                    if prev_x_for_bb is not None and prev_grad_for_bb is not None:
                        s = snapshot - prev_x_for_bb
                        y_diff = full_grad - prev_grad_for_bb
                        denom = float(np.dot(s, y_diff))
                        if denom > 1e-30:
                            candidate = float(np.dot(s, s) / denom)
                            if np.isfinite(candidate) and candidate > 0.0:
                                step_value = candidate
                    prev_x_for_bb = snapshot.copy()
                    prev_grad_for_bb = full_grad.copy()

            avg_epoch_x = np.zeros_like(x) if isinstance(self, SVRG) and variance_reduction == "avg" else None
            rand_epoch_x = None
            n_updates = 0
            for indices in self._stochastic_batches(rng, n_samples, epoch_size, batch_size):
                if isinstance(self, SVRG):
                    grad = (
                        self.model.batch_grad(x, indices)
                        - self.model.batch_grad(snapshot, indices)
                        + full_grad
                    )
                elif isinstance(self, SAGA):
                    assert stored_residuals is not None and average_grad is not None
                    indices = np.unique(indices)
                    current_residuals = self.model.sample_residuals(x, indices)
                    previous_residuals = stored_residuals[indices]
                    grad = (
                        self.model.grad_from_residuals(indices, current_residuals, indices.size)
                        - self.model.grad_from_residuals(indices, previous_residuals, indices.size)
                        + average_grad
                        + self.model.smooth_l2_grad(x)
                    )
                    average_grad = average_grad + self.model.grad_from_residuals(
                        indices, current_residuals - previous_residuals, n_samples
                    )
                    stored_residuals[indices] = current_residuals
                else:
                    grad = self.model.batch_grad(x, indices)

                x = self.prox.call(x - step_value * np.asarray(grad, dtype=float), step=step_value)
                n_updates += 1
                if avg_epoch_x is not None:
                    avg_epoch_x += x
                if isinstance(self, SVRG) and variance_reduction == "rand":
                    if rand_epoch_x is None or int(rng.integers(n_updates)) == 0:
                        rand_epoch_x = x.copy()

            if isinstance(self, SVRG):
                if variance_reduction == "avg" and n_updates:
                    x = avg_epoch_x / float(n_updates)
                elif variance_reduction == "rand" and rand_epoch_x is not None:
                    x = rand_epoch_x

            obj = self.objective(x)
            rel_obj = abs(obj - prev_obj) / max(abs(prev_obj), 1.0)
            rel_delta = relative_distance(x, epoch_start)
            if self._should_record_iter(n_iter) or rel_obj < self.tol:
                self.history.append(
                    n_iter=n_iter,
                    obj=obj,
                    rel_obj=rel_obj,
                    rel_delta=rel_delta,
                    step=step_value,
                    batch_size=batch_size,
                    epoch_size=epoch_size,
                    time=perf_counter() - self._solve_started_at,
                )
            if rel_obj < self.tol:
                break
            prev_obj = obj
        return x

    def _stochastic_batch_size(self, n_samples: int) -> int:
        if self.batch_size is not None:
            return min(int(self.batch_size), n_samples)
        workers = int(self.n_threads)
        return max(1, min(2048 * workers, n_samples))

    def _stochastic_batches(self, rng, n_samples: int, epoch_size: int, batch_size: int):
        rand_type = str(self.rand_type).lower()
        if rand_type not in {"unif", "perm"}:
            raise ValueError("rand_type must be 'unif' or 'perm'")
        if rand_type == "perm":
            remaining = int(epoch_size)
            while remaining > 0:
                order = rng.permutation(n_samples)
                take = min(remaining, n_samples)
                for start in range(0, take, batch_size):
                    yield order[start : min(start + batch_size, take)]
                remaining -= take
            return
        remaining = int(epoch_size)
        while remaining > 0:
            size = min(batch_size, remaining)
            yield rng.integers(0, n_samples, size=size, dtype=np.int64)
            remaining -= size

    def _initial_step(self, step: float | None) -> float:
        if step is not None:
            self.step = step
            return float(self.step)
        if self.step is not None:
            return float(self.step)
        if hasattr(self.model, "get_lip_max"):
            lip = float(self.model.get_lip_max())
            if lip > 0:
                return 1.0 / lip
        return 1e-2

    def _should_record_iter(self, n_iter: int) -> bool:
        print_every = int(self.print_every)
        record_every = int(self.record_every)
        return n_iter % print_every == 0 or n_iter % record_every == 0 or n_iter == int(self.max_iter)

    def _validate_x0(self, x0: np.ndarray) -> np.ndarray:
        x = np.asarray(x0, dtype=float)
        if x.ndim != 1:
            raise ValueError("x0 must be a one-dimensional array")
        if np.any(~np.isfinite(x)):
            raise ValueError("x0 must contain only finite values")
        n_coeffs = getattr(self.model, "n_coeffs", None)
        if n_coeffs is not None and x.size != int(n_coeffs):
            raise ValueError("x0 length must match model.n_coeffs")
        return x.copy()

    def _record(self, n_iter: int, x: np.ndarray, force: bool = False) -> None:
        if force or self._should_record_iter(max(n_iter, 1)):
            self.history.append(
                n_iter=n_iter,
                obj=self.objective(x),
                x=x.copy(),
                time=perf_counter() - self._solve_started_at,
            )


class GD(Solver):
    pass


class AGD(Solver):
    pass


class BFGS(Solver):
    pass


class GFB(Solver):
    pass


class SCPG(Solver):
    pass


class SGD(Solver):
    pass


class AdaGrad(Solver):
    pass


class SVRG(Solver):
    pass


class SAGA(Solver):
    pass


class SDCA(Solver):
    pass


def optimize_positive_coeffs(
    model,
    start: np.ndarray,
    penalty: str = "none",
    C: float = 1e3,
    elastic_net_ratio: float = 0.95,
    max_iter: int = 100,
    tol: float = 1e-5,
    jac: bool = True,
    callback: Callable[[np.ndarray], None] | None = None,
    extra_penalty: Callable[[np.ndarray], float] | None = None,
    extra_grad: Callable[[np.ndarray], np.ndarray] | None = None,
    result_callback: Callable[[OptimizeResult], None] | None = None,
) -> np.ndarray:
    """Optimize a Hawkes model with non-negative coefficients."""

    start = np.maximum(np.asarray(start, dtype=float), 0.0)
    penalty = penalty.lower()
    if C is None or C <= 0:
        raise ValueError("C must be positive")

    strength = 0.0 if penalty == "none" else 1.0 / C

    def objective(x: np.ndarray) -> float:
        if not np.all(np.isfinite(x)):
            return 1e300
        value = model.loss(x)
        if not np.isfinite(value):
            return 1e300
        try:
            penalty_value = _penalty_value(x, penalty, strength, elastic_net_ratio, model)
            if extra_penalty is not None:
                penalty_value += float(extra_penalty(x))
        except (FloatingPointError, np.linalg.LinAlgError, ValueError):
            return 1e300
        total = float(value + penalty_value)
        return total if np.isfinite(total) else 1e300

    def gradient(x: np.ndarray) -> np.ndarray:
        grad = model.grad(x)
        grad = grad + _penalty_grad(x, penalty, strength, elastic_net_ratio, model)
        if extra_grad is not None:
            grad = grad + extra_grad(x)
        return np.nan_to_num(grad, nan=0.0, posinf=1e100, neginf=-1e100)

    if int(max_iter) <= 0:
        if result_callback is not None:
            result_callback(
                OptimizeResult(
                    x=start.copy(),
                    fun=objective(start),
                    jac=gradient(start) if jac and penalty != "nuclear" else None,
                    nit=0,
                    nfev=1,
                    success=True,
                    message="max_iter <= 0; returned starting point",
                )
            )
        return start.copy()

    use_jac = bool(jac and penalty != "nuclear" and (extra_penalty is None or extra_grad is not None))
    result = minimize(
        objective,
        start,
        jac=gradient if use_jac else None,
        method="L-BFGS-B",
        bounds=[(0.0, None)] * start.size,
        callback=callback,
        options={"maxiter": int(max_iter), "ftol": float(tol), "gtol": float(tol)},
    )
    if result_callback is not None:
        result_callback(result)
    if not result.success and not np.isfinite(result.fun):
        raise RuntimeError(f"optimization failed: {result.message}")
    return np.maximum(np.asarray(result.x, dtype=float), 0.0)


def _penalty_value(x, penalty, strength, ratio, model) -> float:
    if penalty == "none":
        return 0.0
    if penalty == "l1":
        return ProxL1(strength).value(x)
    if penalty == "l2":
        return ProxL2Sq(strength).value(x)
    if penalty == "elasticnet":
        return ProxElasticNet(strength, ratio).value(x)
    if penalty == "nuclear":
        n = model.n_nodes
        matrix = x[n:].reshape((n, -1))
        return ProxNuclear(strength).value(matrix)
    raise ValueError(f"unknown penalty {penalty!r}")


def _penalty_grad(x, penalty, strength, ratio, model) -> np.ndarray:
    del model
    if penalty == "none":
        return np.zeros_like(x)
    if penalty == "l1":
        return strength * np.sign(x)
    if penalty == "l2":
        return strength * x
    if penalty == "elasticnet":
        return strength * ratio * np.sign(x) + strength * (1.0 - ratio) * x
    if penalty == "nuclear":
        return np.zeros_like(x)
    raise ValueError(f"unknown penalty {penalty!r}")

