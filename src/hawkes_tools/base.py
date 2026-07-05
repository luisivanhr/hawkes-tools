"""Shared utilities for the pure-Python Hawkes implementation."""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass, field
from datetime import datetime
from functools import wraps
from inspect import signature
from typing import Any, Iterable

import numpy as np

__all__ = [
    "Base",
    "BaseEstimator",
    "History",
    "ThreadPool",
    "TimeFunction",
    "actual_kwargs",
    "normalize_end_times",
    "normalize_events",
    "now_string",
    "relative_distance",
]


def _as_1d_float_array(values: Any, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return np.ascontiguousarray(arr)


class BaseEstimator:
    """Small sklearn-like parameter helper used by learners and models."""

    @property
    def name(self) -> str:
        """Return the class name."""

        return getattr(self, "_name", self.__class__.__name__)

    @staticmethod
    def _get_now() -> str:
        return datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")

    def _set(self, key: str, val: Any) -> None:
        if not isinstance(key, str):
            raise ValueError("In _set function you must pass key as string")
        if key == "name":
            object.__setattr__(self, "_name", str(val))
            return
        object.__setattr__(self, key, val)

    def _as_dict(self) -> dict[str, Any]:
        attrinfos = getattr(self, "_attrinfos", {})
        if attrinfos:
            return {
                key: getattr(self, key)
                for key in attrinfos
                if not key.startswith("_") and hasattr(self, key)
            }
        params = self.get_params()
        params.setdefault("name", self.name)
        return params

    def _inc_attr(self, key: str, step: int = 1) -> None:
        self._set(key, getattr(self, key) + step)

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        del deep
        params: dict[str, Any] = {}
        for key, value in self.__dict__.items():
            if key.startswith("_"):
                continue
            if callable(value):
                continue
            params[key] = value
        return params

    def set_params(self, **params: Any) -> "BaseEstimator":
        for key, value in params.items():
            if not hasattr(self, key):
                raise ValueError(f"Unknown parameter {key!r}")
            setattr(self, key, value)
        return self


class Base(BaseEstimator):
    """Standalone public ``Base`` helper.

    The original class uses a metaclass to enforce read-only descriptors and to
    relay setters into C++ objects. hawkes-tools keeps the useful Python helper
    surface without compiled backend coupling.
    """


def actual_kwargs(function):
    """Decorate ``function`` with the keyword arguments passed at call time."""

    original_signature = signature(function)

    @wraps(function)
    def inner(*args, **kwargs):
        inner.actual_kwargs = dict(kwargs)
        return function(*args, **kwargs)

    inner.actual_kwargs = {}
    inner.__signature__ = original_signature
    return inner


class ThreadPool:
    """Small standalone thread pool utility."""

    def __init__(self, with_lock: bool = False, max_threads: int = 8):
        self._max_threads = int(max_threads)
        if self._max_threads <= 0:
            raise ValueError("max_threads must be positive")
        self._works: list[tuple[Any, tuple[Any, ...], dict[str, Any]]] = []
        self.lock = threading.Lock() if with_lock else None

    def add_work(self, func, *args, **kwargs) -> None:
        if not callable(func):
            raise TypeError("func must be callable")
        self._works.append((func, args, kwargs))

    def start(self) -> None:
        work_queue: queue.Queue[tuple[Any, tuple[Any, ...], dict[str, Any]]] = queue.Queue()
        errors: queue.Queue[BaseException] = queue.Queue()
        for work in self._works:
            work_queue.put(work)

        def worker():
            while True:
                try:
                    func, args, kwargs = work_queue.get_nowait()
                except queue.Empty:
                    return
                try:
                    func(*args, **kwargs)
                except BaseException as exc:
                    errors.put(exc)
                finally:
                    work_queue.task_done()

        threads = [
            threading.Thread(target=worker)
            for _ in range(min(len(self._works), self._max_threads))
        ]
        for thread in threads:
            thread.start()
        work_queue.join()
        for thread in threads:
            thread.join()
        if not errors.empty():
            raise errors.get()


@dataclass
class History:
    """Minimal optimization history container."""

    records: list[dict[str, Any]] = field(default_factory=list)
    print_order: list[str] = field(default_factory=lambda: ["n_iter", "obj"])
    minimum: float | None = None
    minimizer: np.ndarray | None = None

    def append(self, **kwargs: Any) -> None:
        self.records.append(dict(kwargs))

    def clear(self) -> None:
        self.records.clear()

    @property
    def values(self) -> dict[str, list[Any]]:
        keys: list[str] = []
        for record in self.records:
            for key in record:
                if key not in keys:
                    keys.append(key)
        return {key: [record.get(key) for record in self.records] for key in keys}

    def set_minimum(self, minimum: float) -> "History":
        self.minimum = float(minimum)
        return self

    def set_minimizer(self, minimizer: Any) -> "History":
        self.minimizer = np.asarray(minimizer, dtype=float).copy()
        return self

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self):
        return iter(self.records)

    def __getitem__(self, item):
        return self.records[item]


def relative_distance(new: np.ndarray, old: np.ndarray) -> float:
    """Return the relative Euclidean distance between two arrays."""

    new = np.asarray(new, dtype=float)
    old = np.asarray(old, dtype=float)
    denom = max(np.linalg.norm(old), 1.0)
    return float(np.linalg.norm(new - old) / denom)


def now_string() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def normalize_events(
    events: list[Any] | tuple[Any, ...],
    end_times: float | Iterable[float] | np.ndarray | None = None,
) -> tuple[list[list[np.ndarray]], np.ndarray, int]:
    """Normalize event input.

    Accepted event layouts are either a single realization
    ``list[np.ndarray]`` or multiple realizations
    ``list[list[np.ndarray]]``. Returned timestamps are sorted float64 arrays.
    """

    if events is None or len(events) == 0:
        raise ValueError("events must contain at least one realization")

    first = events[0]
    if isinstance(first, np.ndarray) or _looks_like_timestamp_sequence(first):
        raw_realizations = [events]
    else:
        raw_realizations = list(events)

    realizations: list[list[np.ndarray]] = []
    n_nodes: int | None = None
    for r, realization in enumerate(raw_realizations):
        if realization is None or len(realization) == 0:
            raise ValueError(f"realization {r} must contain at least one node")
        clean_realization: list[np.ndarray] = []
        for node, timestamps in enumerate(realization):
            arr = _as_1d_float_array(timestamps, f"events[{r}][{node}]")
            if arr.size and np.any(np.diff(arr) < 0):
                raise ValueError(
                    f"timestamps for realization {r}, node {node} must be sorted"
                )
            if arr.size and arr[0] < 0:
                raise ValueError("timestamps must be non-negative")
            clean_realization.append(arr)
        if n_nodes is None:
            n_nodes = len(clean_realization)
        elif len(clean_realization) != n_nodes:
            raise ValueError("all realizations must have the same number of nodes")
        realizations.append(clean_realization)

    assert n_nodes is not None
    clean_end_times = normalize_end_times(realizations, end_times)
    return realizations, clean_end_times, n_nodes


def normalize_end_times(
    realizations: list[list[np.ndarray]],
    end_times: float | Iterable[float] | np.ndarray | None,
) -> np.ndarray:
    if end_times is None:
        out = []
        for realization in realizations:
            max_time = 0.0
            for timestamps in realization:
                if timestamps.size:
                    max_time = max(max_time, float(timestamps[-1]))
            out.append(max_time)
        return np.asarray(out, dtype=float)

    if isinstance(end_times, (int, float, np.floating)):
        arr = np.asarray([float(end_times)], dtype=float)
    else:
        arr = np.asarray(list(end_times), dtype=float)

    if arr.ndim != 1:
        raise ValueError("end_times must be a scalar or one-dimensional array")
    if arr.size == 1 and len(realizations) > 1:
        arr = np.repeat(arr, len(realizations))
    if arr.size != len(realizations):
        raise ValueError(
            f"end_times must have length {len(realizations)}, got {arr.size}"
        )
    if np.any(arr < 0):
        raise ValueError("end_times must be non-negative")
    for r, realization in enumerate(realizations):
        latest = max((float(ts[-1]) for ts in realization if ts.size), default=0.0)
        if arr[r] < latest:
            raise ValueError(
                f"end_times[{r}]={arr[r]} is before latest timestamp {latest}"
            )
    return arr


def _looks_like_timestamp_sequence(value: Any) -> bool:
    if isinstance(value, (list, tuple)):
        return len(value) == 0 or isinstance(value[0], (int, float, np.floating))
    return False


class TimeFunction(BaseEstimator):
    """Piecewise time function for Hawkes APIs."""

    InterLinear = 0
    InterConstLeft = 1
    InterConstRight = 2

    Border0 = 0
    BorderConstant = 1
    BorderContinue = 2
    Cyclic = 3

    def __init__(
        self,
        values: float | tuple[Any, Any] | list[Any],
        border_type: int = Border0,
        inter_mode: int = InterLinear,
        dt: float = 0.0,
        border_value: float = 0.0,
    ):
        self.border_type = border_type
        self.inter_mode = inter_mode
        self.border_value = float(border_value)

        if isinstance(values, (int, float, np.floating)):
            self.is_constant = True
            self.constant = float(values)
            self.original_t = np.asarray([0.0], dtype=float)
            self.original_y = np.asarray([self.constant], dtype=float)
            self._dt = float(dt)
        else:
            if len(values) != 2:
                raise ValueError("values must be a scalar or (t_values, y_values)")
            t_values = _as_1d_float_array(values[0], "t_values")
            y_values = _as_1d_float_array(values[1], "y_values")
            if t_values.size != y_values.size:
                raise ValueError("t_values and y_values must have the same length")
            if t_values.size == 0:
                raise ValueError("time functions require at least one point")
            if np.any(np.diff(t_values) <= 0):
                raise ValueError("t_values must be strictly increasing")
            self.is_constant = False
            self.constant = 0.0
            self.original_t = t_values
            self.original_y = y_values
            self._dt = float(dt) if dt else self._infer_dt(t_values)
        self.sampled_y = self.original_y.copy()

    @staticmethod
    def _infer_dt(t_values: np.ndarray) -> float:
        if t_values.size < 2:
            return 0.0
        return float(np.min(np.diff(t_values)) / 5.0)

    @property
    def dt(self) -> float:
        return self._dt

    def value(self, t: float | np.ndarray) -> float | np.ndarray:
        arr = np.asarray(t, dtype=float)
        flat = arr.ravel()
        out = np.array([self._value_scalar(float(x)) for x in flat], dtype=float)
        out = out.reshape(arr.shape)
        if np.isscalar(t):
            return float(out)
        return out

    def _value_scalar(self, t: float) -> float:
        if self.is_constant:
            return self.constant if t >= 0 else 0.0
        if t < 0:
            return 0.0

        x = float(t)
        t0 = float(self.original_t[0])
        tn = float(self.original_t[-1])

        if self.border_type == self.Cyclic and tn > t0:
            period = tn - t0
            x = ((x - t0) % period) + t0

        if x < t0:
            return self._left_border()
        if x > tn:
            return self._right_border()
        if x == tn:
            return float(self.original_y[-1])

        idx = int(np.searchsorted(self.original_t, x, side="right") - 1)
        idx = min(max(idx, 0), self.original_t.size - 2)
        t_left = self.original_t[idx]
        t_right = self.original_t[idx + 1]
        y_left = self.original_y[idx]
        y_right = self.original_y[idx + 1]

        if self.inter_mode == self.InterConstLeft:
            return float(y_right)
        if self.inter_mode == self.InterConstRight:
            return float(y_left)
        weight = (x - t_left) / (t_right - t_left)
        return float((1.0 - weight) * y_left + weight * y_right)

    def _left_border(self) -> float:
        if self.border_type == self.BorderConstant:
            return self.border_value
        if self.border_type == self.BorderContinue:
            return float(self.original_y[0])
        return 0.0

    def _right_border(self) -> float:
        if self.border_type == self.BorderConstant:
            return self.border_value
        if self.border_type == self.BorderContinue:
            return float(self.original_y[-1])
        return 0.0

    def future_bound(self, t: float) -> float:
        """Conservative upper bound for values after ``t``."""

        if self.is_constant:
            return max(self.constant, 0.0)
        if self.border_type == self.Cyclic:
            return float(max(np.max(self.original_y), 0.0))
        mask = self.original_t >= t
        candidates = [0.0]
        if mask.any():
            candidates.append(float(np.max(self.original_y[mask])))
        candidates.append(self._right_border())
        candidates.append(float(self.value(t)))
        return float(max(candidates))

    def primitive(self, t: float, n_steps: int | None = None) -> float:
        """Numerically integrate the function on ``[0, t]``."""

        if t <= 0:
            return 0.0
        if self.is_constant:
            return self.constant * t
        if n_steps is None:
            step = self.dt if self.dt > 0 else max(t / 256.0, 1e-8)
            n_steps = max(2, int(np.ceil(t / step)) + 1)
        xs = np.linspace(0.0, t, n_steps)
        ys = np.asarray(self.value(xs), dtype=float)
        return float(np.trapezoid(ys, xs))

    def get_norm(self) -> float:
        if self.is_constant:
            return np.inf if self.constant > 0 else 0.0
        support = float(self.original_t[-1])
        return self.primitive(support)
