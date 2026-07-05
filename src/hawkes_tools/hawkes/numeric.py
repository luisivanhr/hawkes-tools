"""Numerical hot loops used by Hawkes models.

Reference implementations in this module are plain Python/NumPy and remain the
source of truth for tests. Public wrappers dispatch through mandatory Numba JIT
helpers for the standalone runtime.
"""

from __future__ import annotations

import math

import numpy as np

# Numba 0.60 still imports a few NumPy names removed in newer NumPy releases.
if not hasattr(np, "trapz") and hasattr(np, "trapezoid"):
    np.trapz = np.trapezoid
if not hasattr(np, "in1d") and hasattr(np, "isin"):
    np.in1d = np.isin

try:
    from numba import njit
except Exception as exc:  # pragma: no cover - broken dependency environment
    raise ImportError(
        "hawkes_tools requires Numba for the standalone JIT runtime. "
        "Install the package dependencies before importing hawkes_tools."
    ) from exc

NUMBA_AVAILABLE = True


def _compile(func):
    return njit(cache=True)(func)


def is_numba_enabled() -> bool:
    """Return whether public wrappers dispatch through Numba in this process."""

    return True


def _float_array(values) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(values, dtype=np.float64))


def _int_array(values) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(values, dtype=np.int64))


def pack_realization(realization) -> tuple[np.ndarray, np.ndarray]:
    """Pack a list of node timestamp arrays into dense arrays for Numba loops."""

    arrays = [_float_array(timestamps) for timestamps in realization]
    n_nodes = len(arrays)
    sizes = np.asarray([timestamps.size for timestamps in arrays], dtype=np.int64)
    max_size = int(np.max(sizes)) if sizes.size else 0
    events = np.zeros((n_nodes, max_size), dtype=np.float64)
    for node, timestamps in enumerate(arrays):
        if timestamps.size:
            events[node, : timestamps.size] = timestamps
    return events, sizes


def pack_realizations(realizations, end_times=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pack multiple realizations into dense arrays for Numba loops.

    Parameters
    ----------
    realizations
        Sequence of realizations, each a sequence of per-node sorted timestamps.
    end_times
        Optional scalar or one-dimensional sequence. If omitted, each end time
        is the latest timestamp in that realization, matching the local model
        normalization convention.
    """

    realizations = list(realizations)
    if len(realizations) == 0:
        raise ValueError("realizations must not be empty")
    n_realizations = len(realizations)
    n_nodes = len(realizations[0])
    arrays: list[list[np.ndarray]] = []
    max_size = 0
    for r, realization in enumerate(realizations):
        if len(realization) != n_nodes:
            raise ValueError(f"realization {r} has {len(realization)} nodes, expected {n_nodes}")
        packed_realization = [_float_array(node_events) for node_events in realization]
        arrays.append(packed_realization)
        for node_events in packed_realization:
            max_size = max(max_size, int(node_events.size))

    events = np.zeros((n_realizations, n_nodes, max_size), dtype=np.float64)
    sizes = np.zeros((n_realizations, n_nodes), dtype=np.int64)
    inferred_end_times = np.zeros(n_realizations, dtype=np.float64)
    for r, realization in enumerate(arrays):
        latest = 0.0
        for node, timestamps in enumerate(realization):
            size = int(timestamps.size)
            sizes[r, node] = size
            if size:
                events[r, node, :size] = timestamps
                latest = max(latest, float(timestamps[-1]))
        inferred_end_times[r] = latest

    if end_times is None:
        end_times_arr = inferred_end_times
    elif np.isscalar(end_times):
        end_times_arr = np.full(n_realizations, float(end_times), dtype=np.float64)
    else:
        end_times_arr = _float_array(end_times)
        if end_times_arr.shape != (n_realizations,):
            raise ValueError(f"end_times has shape {end_times_arr.shape}, expected {(n_realizations,)}")
    return events, sizes, end_times_arr


def pack_event_table(realization) -> tuple[np.ndarray, np.ndarray]:
    """Return event times and node ids sorted by time for one realization."""

    rows = []
    for node, timestamps in enumerate(realization):
        for timestamp in np.asarray(timestamps, dtype=float):
            rows.append((float(timestamp), int(node)))
    if not rows:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.int64)
    rows.sort(key=lambda item: (item[0], item[1]))
    times = np.asarray([item[0] for item in rows], dtype=np.float64)
    nodes = np.asarray([item[1] for item in rows], dtype=np.int64)
    return times, nodes


def homogeneous_poisson_events_reference(
    start_time: float,
    end_time: float,
    intensities: np.ndarray,
    uniforms: np.ndarray,
    out_times: np.ndarray,
    out_nodes: np.ndarray,
) -> tuple[int, float, bool]:
    """Fill homogeneous Poisson events from pre-generated uniforms.

    The caller owns RNG generation so simulation seeding remains controlled by
    the Python `Generator`; this helper only performs deterministic timestamp
    and node assignment arithmetic.
    """

    intensities = np.asarray(intensities, dtype=float)
    total_intensity = float(np.sum(intensities))
    if total_intensity <= 0.0:
        return 0, float(start_time), False
    current_time = float(start_time)
    count = 0
    for row in np.asarray(uniforms, dtype=float):
        candidate_time = current_time - math.log1p(-float(row[0])) / total_intensity
        if candidate_time >= end_time:
            return count, float(end_time), True
        threshold = float(row[1]) * total_intensity
        cumulative = 0.0
        node = intensities.size - 1
        for i, intensity in enumerate(intensities):
            cumulative += float(intensity)
            if threshold < cumulative:
                node = i
                break
        out_times[count] = candidate_time
        out_nodes[count] = node
        count += 1
        current_time = candidate_time
        if count >= out_times.size:
            return count, current_time, False
    return count, current_time, False


def exp_feature_at_time_reference(
    t: float,
    timestamps: np.ndarray,
    decay: float,
    include_current: bool = False,
) -> float:
    value = 0.0
    t = float(t)
    decay = float(decay)
    for tk in np.asarray(timestamps, dtype=float):
        tk = float(tk)
        if tk > t or (tk == t and not include_current):
            break
        value += decay * math.exp(-decay * (t - tk))
    return float(value)


def exp_primitive_sum_reference(end_time: float, timestamps: np.ndarray, decay: float) -> float:
    value = 0.0
    end_time = float(end_time)
    decay = float(decay)
    if decay == 0.0:
        return 0.0
    for tk in np.asarray(timestamps, dtype=float):
        tk = float(tk)
        if tk >= end_time:
            break
        value += 1.0 - math.exp(-decay * (end_time - tk))
    return float(value)


def sumexp_feature_at_time_reference(
    t: float,
    timestamps: np.ndarray,
    decays: np.ndarray,
    out: np.ndarray,
    include_current: bool = False,
) -> None:
    out[:] = 0.0
    t = float(t)
    for tk in np.asarray(timestamps, dtype=float):
        tk = float(tk)
        if tk > t or (tk == t and not include_current):
            break
        delay = t - tk
        for u, decay in enumerate(np.asarray(decays, dtype=float)):
            out[u] += float(decay) * math.exp(-float(decay) * delay)


def sumexp_primitive_sum_reference(
    end_time: float,
    timestamps: np.ndarray,
    decays: np.ndarray,
    out: np.ndarray,
) -> None:
    out[:] = 0.0
    end_time = float(end_time)
    for tk in np.asarray(timestamps, dtype=float):
        tk = float(tk)
        if tk >= end_time:
            break
        delay = end_time - tk
        for u, decay in enumerate(np.asarray(decays, dtype=float)):
            out[u] += 1.0 - math.exp(-float(decay) * delay)


def exp_kernel_convolution_reference(
    time: float,
    timestamps: np.ndarray,
    intensity: float,
    decay: float,
    include_current: bool = False,
) -> float:
    return float(
        float(intensity)
        * exp_feature_at_time_reference(time, timestamps, decay, include_current)
    )


def exp_kernel_primitive_convolution_reference(
    time: float,
    timestamps: np.ndarray,
    intensity: float,
    decay: float,
) -> float:
    return float(float(intensity) * exp_primitive_sum_reference(time, timestamps, decay))


def sumexp_kernel_convolution_reference(
    time: float,
    timestamps: np.ndarray,
    intensities: np.ndarray,
    decays: np.ndarray,
    include_current: bool = False,
) -> float:
    features = np.empty(np.asarray(decays, dtype=float).size, dtype=float)
    sumexp_feature_at_time_reference(time, timestamps, decays, features, include_current)
    return float(np.dot(np.asarray(intensities, dtype=float), features))


def sumexp_kernel_primitive_convolution_reference(
    time: float,
    timestamps: np.ndarray,
    intensities: np.ndarray,
    decays: np.ndarray,
) -> float:
    primitives = np.empty(np.asarray(decays, dtype=float).size, dtype=float)
    sumexp_primitive_sum_reference(time, timestamps, decays, primitives)
    return float(np.dot(np.asarray(intensities, dtype=float), primitives))


def exp_intensity_vector_reference(
    t: float,
    events: np.ndarray,
    sizes: np.ndarray,
    baseline: np.ndarray,
    adjacency: np.ndarray,
    decays: np.ndarray,
    include_current: bool = False,
) -> np.ndarray:
    values = np.asarray(baseline, dtype=float).copy()
    n_nodes = values.size
    for i in range(n_nodes):
        for j in range(n_nodes):
            values[i] += float(adjacency[i, j]) * exp_feature_at_time_reference(
                t,
                events[j, : int(sizes[j])],
                float(decays[i, j]),
                include_current,
            )
    return values


def sumexp_intensity_vector_reference(
    t: float,
    events: np.ndarray,
    sizes: np.ndarray,
    baseline: np.ndarray,
    adjacency: np.ndarray,
    decays: np.ndarray,
    include_current: bool = False,
) -> np.ndarray:
    values = np.asarray(baseline, dtype=float).copy()
    n_nodes = values.size
    tmp = np.empty(np.asarray(decays, dtype=float).size, dtype=float)
    for i in range(n_nodes):
        for j in range(n_nodes):
            sumexp_feature_at_time_reference(
                t,
                events[j, : int(sizes[j])],
                decays,
                tmp,
                include_current,
            )
            values[i] += float(np.dot(adjacency[i, j, :], tmp))
    return values


def exp_compensator_value_reference(
    node: int,
    t: float,
    events: np.ndarray,
    sizes: np.ndarray,
    baseline: np.ndarray,
    adjacency: np.ndarray,
    decays: np.ndarray,
) -> float:
    value = float(baseline[node]) * float(t)
    for j in range(np.asarray(baseline).size):
        value += float(adjacency[node, j]) * exp_primitive_sum_reference(
            t, events[j, : int(sizes[j])], float(decays[node, j])
        )
    return float(value)


def sumexp_compensator_value_reference(
    node: int,
    t: float,
    events: np.ndarray,
    sizes: np.ndarray,
    baseline: np.ndarray,
    adjacency: np.ndarray,
    decays: np.ndarray,
) -> float:
    value = float(baseline[node]) * float(t)
    tmp = np.empty(np.asarray(decays, dtype=float).size, dtype=float)
    for j in range(np.asarray(baseline).size):
        sumexp_primitive_sum_reference(t, events[j, : int(sizes[j])], decays, tmp)
        value += float(np.dot(adjacency[node, j, :], tmp))
    return float(value)


def exp_intensity_bound_reference(
    t: float,
    events: np.ndarray,
    sizes: np.ndarray,
    baseline: np.ndarray,
    adjacency: np.ndarray,
    decays: np.ndarray,
    include_current: bool = True,
) -> float:
    total = 0.0
    n_nodes = np.asarray(baseline).size
    for i in range(n_nodes):
        total += max(float(baseline[i]), 0.0)
        for j in range(n_nodes):
            convolution = float(adjacency[i, j]) * exp_feature_at_time_reference(
                t,
                events[j, : int(sizes[j])],
                float(decays[i, j]),
                include_current,
            )
            total += max(convolution, 0.0)
    return float(max(total, 0.0))


def sumexp_intensity_bound_reference(
    t: float,
    events: np.ndarray,
    sizes: np.ndarray,
    baseline: np.ndarray,
    adjacency: np.ndarray,
    decays: np.ndarray,
    include_current: bool = True,
) -> float:
    total = 0.0
    n_nodes = np.asarray(baseline).size
    tmp = np.empty(np.asarray(decays, dtype=float).size, dtype=float)
    for i in range(n_nodes):
        total += max(float(baseline[i]), 0.0)
        for j in range(n_nodes):
            sumexp_feature_at_time_reference(
                t,
                events[j, : int(sizes[j])],
                decays,
                tmp,
                include_current,
            )
            total += max(float(np.dot(adjacency[i, j, :], tmp)), 0.0)
    return float(max(total, 0.0))


def exp_loglik_loss_scan_reference(
    events: np.ndarray,
    sizes: np.ndarray,
    end_time: float,
    baseline: np.ndarray,
    adjacency: np.ndarray,
    decays: np.ndarray,
) -> float:
    n_nodes = np.asarray(baseline).size
    value = float((np.sum(baseline) - n_nodes) * float(end_time))
    for i in range(n_nodes):
        for j in range(n_nodes):
            value += float(adjacency[i, j]) * exp_primitive_sum_reference(
                end_time, events[j, : int(sizes[j])], float(decays[i, j])
            )
    for i in range(n_nodes):
        for k in range(int(sizes[i])):
            t = float(events[i, k])
            intensity = float(baseline[i])
            for j in range(n_nodes):
                intensity += float(adjacency[i, j]) * exp_feature_at_time_reference(
                    t, events[j, : int(sizes[j])], float(decays[i, j])
                )
            if intensity <= 0.0:
                return np.inf
            value -= math.log(intensity)
    return float(value)


def sumexp_loglik_loss_scan_reference(
    events: np.ndarray,
    sizes: np.ndarray,
    end_time: float,
    baseline: np.ndarray,
    adjacency: np.ndarray,
    decays: np.ndarray,
) -> float:
    n_nodes = np.asarray(baseline).size
    value = float((np.sum(baseline) - n_nodes) * float(end_time))
    tmp = np.empty(np.asarray(decays, dtype=float).size, dtype=float)
    for i in range(n_nodes):
        for j in range(n_nodes):
            sumexp_primitive_sum_reference(end_time, events[j, : int(sizes[j])], decays, tmp)
            value += float(np.dot(adjacency[i, j, :], tmp))
    for i in range(n_nodes):
        for k in range(int(sizes[i])):
            t = float(events[i, k])
            intensity = float(baseline[i])
            for j in range(n_nodes):
                sumexp_feature_at_time_reference(t, events[j, : int(sizes[j])], decays, tmp)
                intensity += float(np.dot(adjacency[i, j, :], tmp))
            if intensity <= 0.0:
                return np.inf
            value -= math.log(intensity)
    return float(value)


def exp_loglik_grad_scan_reference(
    events: np.ndarray,
    sizes: np.ndarray,
    end_time: float,
    baseline: np.ndarray,
    adjacency: np.ndarray,
    decays: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    n_nodes = np.asarray(baseline).size
    grad_baseline = np.full(n_nodes, float(end_time), dtype=np.float64)
    grad_adjacency = np.empty((n_nodes, n_nodes), dtype=np.float64)
    for i in range(n_nodes):
        for j in range(n_nodes):
            grad_adjacency[i, j] = exp_primitive_sum_reference(
                end_time,
                events[j, : int(sizes[j])],
                float(decays[i, j]),
            )
    for i in range(n_nodes):
        for k in range(int(sizes[i])):
            t = float(events[i, k])
            intensity = float(baseline[i])
            features = np.empty(n_nodes, dtype=np.float64)
            for j in range(n_nodes):
                features[j] = exp_feature_at_time_reference(
                    t,
                    events[j, : int(sizes[j])],
                    float(decays[i, j]),
                )
                intensity += float(adjacency[i, j]) * features[j]
            if intensity <= 0.0:
                continue
            grad_baseline[i] -= 1.0 / intensity
            grad_adjacency[i, :] -= features / intensity
    return grad_baseline, grad_adjacency


def sumexp_loglik_grad_scan_reference(
    events: np.ndarray,
    sizes: np.ndarray,
    end_time: float,
    baseline: np.ndarray,
    adjacency: np.ndarray,
    decays: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    n_nodes = np.asarray(baseline).size
    n_decays = np.asarray(decays).size
    grad_baseline = np.full(n_nodes, float(end_time), dtype=np.float64)
    grad_adjacency = np.empty((n_nodes, n_nodes, n_decays), dtype=np.float64)
    tmp = np.empty(n_decays, dtype=np.float64)
    for i in range(n_nodes):
        for j in range(n_nodes):
            sumexp_primitive_sum_reference(end_time, events[j, : int(sizes[j])], decays, tmp)
            grad_adjacency[i, j, :] = tmp
    features = np.empty(n_decays, dtype=np.float64)
    event_features = np.empty((n_nodes, n_decays), dtype=np.float64)
    for i in range(n_nodes):
        for k in range(int(sizes[i])):
            t = float(events[i, k])
            intensity = float(baseline[i])
            for j in range(n_nodes):
                sumexp_feature_at_time_reference(t, events[j, : int(sizes[j])], decays, features)
                event_features[j, :] = features
                intensity += float(np.dot(adjacency[i, j, :], features))
            if intensity <= 0.0:
                continue
            grad_baseline[i] -= 1.0 / intensity
            grad_adjacency[i, :, :] -= event_features / intensity
    return grad_baseline, grad_adjacency


def feature_product_integral_reference(
    end_time: float,
    timestamps_a: np.ndarray,
    decay_a: float,
    timestamps_b: np.ndarray,
    decay_b: float,
) -> float:
    if decay_a <= 0.0 or decay_b <= 0.0:
        return 0.0
    rate = float(decay_a + decay_b)
    value = 0.0
    for ta in np.asarray(timestamps_a, dtype=float):
        ta = float(ta)
        if ta >= end_time:
            break
        for tb in np.asarray(timestamps_b, dtype=float):
            tb = float(tb)
            if tb >= end_time:
                break
            start = max(ta, tb)
            remaining = float(end_time) - start
            if remaining <= 0.0:
                continue
            value += (
                decay_a
                * decay_b
                * math.exp(-decay_a * (start - ta) - decay_b * (start - tb))
                * (-math.expm1(-rate * remaining))
                / rate
            )
    return float(value)


def exp_ls_statistics_reference(
    events: np.ndarray,
    sizes: np.ndarray,
    end_time: float,
    target_decays: np.ndarray,
    target_node: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_nodes = np.asarray(sizes).size
    feature_integrals = np.empty(n_nodes, dtype=np.float64)
    event_feature_sums = np.zeros(n_nodes, dtype=np.float64)
    feature_products = np.empty((n_nodes, n_nodes), dtype=np.float64)
    for j in range(n_nodes):
        feature_integrals[j] = exp_primitive_sum_reference(
            end_time, events[j, : int(sizes[j])], float(target_decays[j])
        )
        for k in range(int(sizes[target_node])):
            event_feature_sums[j] += exp_feature_at_time_reference(
                float(events[target_node, k]),
                events[j, : int(sizes[j])],
                float(target_decays[j]),
            )
    for j in range(n_nodes):
        for k in range(n_nodes):
            feature_products[j, k] = feature_product_integral_reference(
                end_time,
                events[j, : int(sizes[j])],
                float(target_decays[j]),
                events[k, : int(sizes[k])],
                float(target_decays[k]),
            )
    return feature_integrals, feature_products, event_feature_sums


def sumexp_ls_integral_statistics_reference(
    events: np.ndarray,
    sizes: np.ndarray,
    end_time: float,
    decays: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    n_nodes = np.asarray(sizes).size
    n_decays = np.asarray(decays).size
    feature_integrals = np.empty((n_nodes, n_decays), dtype=np.float64)
    tmp = np.empty(n_decays, dtype=np.float64)
    for j in range(n_nodes):
        sumexp_primitive_sum_reference(end_time, events[j, : int(sizes[j])], decays, tmp)
        feature_integrals[j, :] = tmp

    n_features = n_nodes * n_decays
    feature_products = np.empty((n_features, n_features), dtype=np.float64)
    for j in range(n_nodes):
        for u in range(n_decays):
            left = j * n_decays + u
            for k in range(n_nodes):
                for v in range(n_decays):
                    right = k * n_decays + v
                    feature_products[left, right] = feature_product_integral_reference(
                        end_time,
                        events[j, : int(sizes[j])],
                        float(decays[u]),
                        events[k, : int(sizes[k])],
                        float(decays[v]),
                    )
    return feature_integrals, feature_products


def sumexp_ls_event_feature_sums_reference(
    events: np.ndarray,
    sizes: np.ndarray,
    decays: np.ndarray,
    target_node: int,
) -> np.ndarray:
    n_nodes = np.asarray(sizes).size
    n_decays = np.asarray(decays).size
    event_feature_sums = np.zeros((n_nodes, n_decays), dtype=np.float64)
    features = np.empty(n_decays, dtype=np.float64)
    for k in range(int(sizes[target_node])):
        event_time = float(events[target_node, k])
        for j in range(n_nodes):
            sumexp_feature_at_time_reference(event_time, events[j, : int(sizes[j])], decays, features)
            event_feature_sums[j, :] += features
    return event_feature_sums


def _exp_feature_at_time_numba_impl(t, timestamps, decay, include_current):
    value = 0.0
    for k in range(timestamps.size):
        tk = timestamps[k]
        if tk > t or (tk == t and not include_current):
            break
        value += decay * math.exp(-decay * (t - tk))
    return value


def _exp_primitive_sum_numba_impl(end_time, timestamps, decay):
    value = 0.0
    if decay == 0.0:
        return 0.0
    for k in range(timestamps.size):
        tk = timestamps[k]
        if tk >= end_time:
            break
        value += 1.0 - math.exp(-decay * (end_time - tk))
    return value


def _sumexp_feature_at_time_numba_impl(t, timestamps, decays, out, include_current):
    for u in range(decays.size):
        out[u] = 0.0
    for k in range(timestamps.size):
        tk = timestamps[k]
        if tk > t or (tk == t and not include_current):
            break
        delay = t - tk
        for u in range(decays.size):
            out[u] += decays[u] * math.exp(-decays[u] * delay)


def _sumexp_primitive_sum_numba_impl(end_time, timestamps, decays, out):
    for u in range(decays.size):
        out[u] = 0.0
    for k in range(timestamps.size):
        tk = timestamps[k]
        if tk >= end_time:
            break
        delay = end_time - tk
        for u in range(decays.size):
            out[u] += 1.0 - math.exp(-decays[u] * delay)


def _exp_intensity_vector_numba_impl(t, events, sizes, baseline, adjacency, decays, include_current, out):
    n_nodes = baseline.size
    for i in range(n_nodes):
        value = baseline[i]
        for j in range(n_nodes):
            feature = 0.0
            for k in range(sizes[j]):
                tk = events[j, k]
                if tk > t or (tk == t and not include_current):
                    break
                feature += decays[i, j] * math.exp(-decays[i, j] * (t - tk))
            value += adjacency[i, j] * feature
        out[i] = value


def _sumexp_intensity_vector_numba_impl(t, events, sizes, baseline, adjacency, decays, include_current, out):
    n_nodes = baseline.size
    n_decays = decays.size
    for i in range(n_nodes):
        value = baseline[i]
        for j in range(n_nodes):
            for u in range(n_decays):
                feature = 0.0
                for k in range(sizes[j]):
                    tk = events[j, k]
                    if tk > t or (tk == t and not include_current):
                        break
                    feature += decays[u] * math.exp(-decays[u] * (t - tk))
                value += adjacency[i, j, u] * feature
        out[i] = value


def _exp_compensator_value_numba_impl(node, t, events, sizes, baseline, adjacency, decays):
    value = baseline[node] * t
    n_nodes = baseline.size
    for j in range(n_nodes):
        primitive = 0.0
        decay = decays[node, j]
        if decay != 0.0:
            for k in range(sizes[j]):
                tk = events[j, k]
                if tk >= t:
                    break
                primitive += 1.0 - math.exp(-decay * (t - tk))
        value += adjacency[node, j] * primitive
    return value


def _sumexp_compensator_value_numba_impl(node, t, events, sizes, baseline, adjacency, decays):
    value = baseline[node] * t
    n_nodes = baseline.size
    n_decays = decays.size
    for j in range(n_nodes):
        for u in range(n_decays):
            primitive = 0.0
            decay = decays[u]
            for k in range(sizes[j]):
                tk = events[j, k]
                if tk >= t:
                    break
                primitive += 1.0 - math.exp(-decay * (t - tk))
            value += adjacency[node, j, u] * primitive
    return value


def _homogeneous_poisson_events_numba_impl(start_time, end_time, intensities, uniforms, out_times, out_nodes):
    total_intensity = 0.0
    for i in range(intensities.size):
        total_intensity += intensities[i]
    if total_intensity <= 0.0:
        return 0, start_time, False
    current_time = start_time
    count = 0
    for row in range(uniforms.shape[0]):
        candidate_time = current_time - math.log1p(-uniforms[row, 0]) / total_intensity
        if candidate_time >= end_time:
            return count, end_time, True
        threshold = uniforms[row, 1] * total_intensity
        cumulative = 0.0
        node = intensities.size - 1
        for i in range(intensities.size):
            cumulative += intensities[i]
            if threshold < cumulative:
                node = i
                break
        out_times[count] = candidate_time
        out_nodes[count] = node
        count += 1
        current_time = candidate_time
        if count >= out_times.size:
            return count, current_time, False
    return count, current_time, False


def _exp_intensity_bound_numba_impl(t, events, sizes, baseline, adjacency, decays, include_current):
    total = 0.0
    n_nodes = baseline.size
    for i in range(n_nodes):
        baseline_i = baseline[i]
        if baseline_i > 0.0:
            total += baseline_i
        for j in range(n_nodes):
            feature = 0.0
            decay = decays[i, j]
            for k in range(sizes[j]):
                tk = events[j, k]
                if tk > t or (tk == t and not include_current):
                    break
                feature += decay * math.exp(-decay * (t - tk))
            convolution = adjacency[i, j] * feature
            if convolution > 0.0:
                total += convolution
    return total if total > 0.0 else 0.0


def _simulate_exp_hawkes_numba_impl(
    baseline,
    adjacency,
    decays,
    end_time,
    max_jumps,
    seed,
    threshold_negative_intensity,
    capacity,
):
    if seed >= 0:
        np.random.seed(seed)
    n_nodes = baseline.size
    events = np.empty((n_nodes, capacity), dtype=np.float64)
    sizes = np.zeros(n_nodes, dtype=np.int64)
    effects = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    current_time = 0.0
    n_total_jumps = 0

    while current_time < end_time and n_total_jumps < max_jumps:
        bound = 0.0
        for i in range(n_nodes):
            if baseline[i] > 0.0:
                bound += baseline[i]
            for j in range(n_nodes):
                if effects[i, j] > 0.0:
                    bound += effects[i, j]
        if bound <= 0.0 or not math.isfinite(bound):
            current_time = end_time
            break

        candidate_time = current_time + np.random.exponential(1.0 / bound)
        if candidate_time >= end_time:
            current_time = end_time
            break

        elapsed = candidate_time - current_time
        for i in range(n_nodes):
            for j in range(n_nodes):
                effects[i, j] *= math.exp(-decays[i, j] * elapsed)
        current_time = candidate_time

        total_intensity = 0.0
        node_intensity = np.empty(n_nodes, dtype=np.float64)
        for i in range(n_nodes):
            intensity = baseline[i]
            for j in range(n_nodes):
                intensity += effects[i, j]
            if intensity < 0.0:
                if threshold_negative_intensity:
                    intensity = 0.0
                else:
                    return events, sizes, current_time, n_total_jumps, -1
            node_intensity[i] = intensity
            total_intensity += intensity

        if total_intensity <= 0.0 or np.random.random() * bound > total_intensity:
            continue
        if n_total_jumps >= capacity:
            return events, sizes, current_time, n_total_jumps, 1

        threshold = np.random.random() * total_intensity
        cumulative = 0.0
        node = n_nodes - 1
        for i in range(n_nodes):
            cumulative += node_intensity[i]
            if threshold < cumulative:
                node = i
                break
        events[node, sizes[node]] = current_time
        sizes[node] += 1
        n_total_jumps += 1

        for i in range(n_nodes):
            effects[i, node] += adjacency[i, node] * decays[i, node]

    return events, sizes, current_time, n_total_jumps, 0


def _mixed_timefunc_value(lag, t_values, y_values, size, inter_mode, border_type, border_value):
    if size <= 0:
        return 0.0
    if lag < 0.0:
        return 0.0
    t0 = t_values[0]
    tn = t_values[size - 1]
    x = lag
    if border_type == 3 and tn > t0:
        period = tn - t0
        x = ((x - t0) % period) + t0
    if x < t0:
        if border_type == 1:
            return border_value
        if border_type == 2:
            return y_values[0]
        return 0.0
    if x > tn:
        if border_type == 1:
            return border_value
        if border_type == 2:
            return y_values[size - 1]
        return 0.0
    if x == tn:
        return y_values[size - 1]
    idx = 0
    for k in range(size - 1):
        if t_values[k] <= x < t_values[k + 1]:
            idx = k
            break
    if inter_mode == 1:
        return y_values[idx + 1]
    if inter_mode == 2:
        return y_values[idx]
    t_left = t_values[idx]
    t_right = t_values[idx + 1]
    if t_right <= t_left:
        return y_values[idx]
    weight = (x - t_left) / (t_right - t_left)
    return (1.0 - weight) * y_values[idx] + weight * y_values[idx + 1]


def _mixed_timefunc_future_bound(lag, t_values, y_values, size, inter_mode, border_type, border_value):
    if size <= 0:
        return 0.0
    if border_type == 3:
        max_value = 0.0
        for k in range(size):
            if y_values[k] > max_value:
                max_value = y_values[k]
        return max_value
    value = _mixed_timefunc_value(lag, t_values, y_values, size, inter_mode, border_type, border_value)
    if value < 0.0:
        value = 0.0
    for k in range(size):
        if t_values[k] >= lag and y_values[k] > value:
            value = y_values[k]
    right = 0.0
    if border_type == 1:
        right = border_value
    elif border_type == 2:
        right = y_values[size - 1]
    if right > value:
        value = right
    return value if value > 0.0 else 0.0


def _mixed_kernel_value(
    kernel_type,
    lag,
    p1,
    p2,
    p3,
    support,
    t_values,
    y_values,
    size,
    inter_mode,
    border_type,
    border_value,
):
    if kernel_type == 0 or lag < 0.0:
        return 0.0
    if kernel_type != 1 and support >= 0.0 and lag >= support:
        return 0.0
    if kernel_type == 1:
        intensity = p1
        decay = p2
        if intensity == 0.0 or decay == 0.0:
            return 0.0
        return intensity * decay * math.exp(-decay * lag)
    if kernel_type == 2:
        multiplier = p1
        cutoff = p2
        exponent = p3
        if multiplier == 0.0:
            return 0.0
        return multiplier * (cutoff + lag) ** (-exponent)
    if kernel_type == 3:
        return _mixed_timefunc_value(lag, t_values, y_values, size, inter_mode, border_type, border_value)
    return 0.0


def _mixed_kernel_future_bound(
    kernel_type,
    lag,
    value_at_lag,
    p1,
    p2,
    p3,
    support,
    t_values,
    y_values,
    size,
    inter_mode,
    border_type,
    border_value,
):
    if kernel_type == 0 or lag < 0.0:
        return 0.0
    if kernel_type != 1 and support >= 0.0 and lag >= support:
        return 0.0
    if kernel_type == 1:
        return value_at_lag if value_at_lag > 0.0 else 0.0
    if kernel_type == 2:
        return value_at_lag if value_at_lag > 0.0 else 0.0
    if kernel_type == 3:
        return _mixed_timefunc_future_bound(lag, t_values, y_values, size, inter_mode, border_type, border_value)
    return 0.0


def _simulate_mixed_hawkes_numba_impl(
    baseline,
    kernel_types,
    p1,
    p2,
    p3,
    supports,
    tf_t_values,
    tf_y_values,
    tf_sizes,
    tf_inter_modes,
    tf_border_types,
    tf_border_values,
    end_time,
    max_jumps,
    seed,
    threshold_negative_intensity,
    capacity,
):
    if seed >= 0:
        np.random.seed(seed)
    n_nodes = baseline.size
    events = np.empty((n_nodes, capacity), dtype=np.float64)
    sizes = np.zeros(n_nodes, dtype=np.int64)
    exp_effects = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    current_time = 0.0
    n_total_jumps = 0

    while current_time < end_time and n_total_jumps < max_jumps:
        bound = 0.0
        for i in range(n_nodes):
            if baseline[i] > 0.0:
                bound += baseline[i]
            for j in range(n_nodes):
                ktype = kernel_types[i, j]
                if ktype == 1:
                    if exp_effects[i, j] > 0.0:
                        bound += exp_effects[i, j]
                elif ktype != 0:
                    first_time = current_time - supports[i, j]
                    for event_idx in range(sizes[j] - 1, -1, -1):
                        event_time = events[j, event_idx]
                        if event_time < first_time:
                            break
                        lag = current_time - event_time
                        value = _mixed_kernel_value(
                            ktype,
                            lag,
                            p1[i, j],
                            p2[i, j],
                            p3[i, j],
                            supports[i, j],
                            tf_t_values[i, j],
                            tf_y_values[i, j],
                            tf_sizes[i, j],
                            tf_inter_modes[i, j],
                            tf_border_types[i, j],
                            tf_border_values[i, j],
                        )
                        bound += _mixed_kernel_future_bound(
                            ktype,
                            lag,
                            value,
                            p1[i, j],
                            p2[i, j],
                            p3[i, j],
                            supports[i, j],
                            tf_t_values[i, j],
                            tf_y_values[i, j],
                            tf_sizes[i, j],
                            tf_inter_modes[i, j],
                            tf_border_types[i, j],
                            tf_border_values[i, j],
                        )
        if bound <= 0.0 or not math.isfinite(bound):
            current_time = end_time
            break

        candidate_time = current_time + np.random.exponential(1.0 / bound)
        if candidate_time >= end_time:
            current_time = end_time
            break

        elapsed = candidate_time - current_time
        for i in range(n_nodes):
            for j in range(n_nodes):
                if kernel_types[i, j] == 1:
                    exp_effects[i, j] *= math.exp(-p2[i, j] * elapsed)
        current_time = candidate_time

        total_intensity = 0.0
        node_intensity = np.empty(n_nodes, dtype=np.float64)
        for i in range(n_nodes):
            intensity = baseline[i]
            for j in range(n_nodes):
                ktype = kernel_types[i, j]
                if ktype == 1:
                    intensity += exp_effects[i, j]
                elif ktype != 0:
                    first_time = current_time - supports[i, j]
                    for event_idx in range(sizes[j] - 1, -1, -1):
                        event_time = events[j, event_idx]
                        if event_time < first_time:
                            break
                        lag = current_time - event_time
                        intensity += _mixed_kernel_value(
                            ktype,
                            lag,
                            p1[i, j],
                            p2[i, j],
                            p3[i, j],
                            supports[i, j],
                            tf_t_values[i, j],
                            tf_y_values[i, j],
                            tf_sizes[i, j],
                            tf_inter_modes[i, j],
                            tf_border_types[i, j],
                            tf_border_values[i, j],
                        )
            if intensity < 0.0:
                if threshold_negative_intensity:
                    intensity = 0.0
                else:
                    return events, sizes, current_time, n_total_jumps, -1
            node_intensity[i] = intensity
            total_intensity += intensity

        if total_intensity <= 0.0 or np.random.random() * bound > total_intensity:
            continue
        if n_total_jumps >= capacity:
            return events, sizes, current_time, n_total_jumps, 1

        threshold = np.random.random() * total_intensity
        cumulative = 0.0
        node = n_nodes - 1
        for i in range(n_nodes):
            cumulative += node_intensity[i]
            if threshold < cumulative:
                node = i
                break
        events[node, sizes[node]] = current_time
        sizes[node] += 1
        n_total_jumps += 1

        for i in range(n_nodes):
            if kernel_types[i, node] == 1:
                exp_effects[i, node] += p1[i, node] * p2[i, node]

    return events, sizes, current_time, n_total_jumps, 0


def _sumexp_intensity_bound_numba_impl(t, events, sizes, baseline, adjacency, decays, include_current):
    total = 0.0
    n_nodes = baseline.size
    n_decays = decays.size
    for i in range(n_nodes):
        baseline_i = baseline[i]
        if baseline_i > 0.0:
            total += baseline_i
        for j in range(n_nodes):
            convolution = 0.0
            for u in range(n_decays):
                feature = 0.0
                decay = decays[u]
                for k in range(sizes[j]):
                    tk = events[j, k]
                    if tk > t or (tk == t and not include_current):
                        break
                    feature += decay * math.exp(-decay * (t - tk))
                convolution += adjacency[i, j, u] * feature
            if convolution > 0.0:
                total += convolution
    return total if total > 0.0 else 0.0


def _exp_loglik_loss_scan_numba_impl(events, sizes, end_time, baseline, adjacency, decays):
    n_nodes = baseline.size
    baseline_sum = 0.0
    for i in range(n_nodes):
        baseline_sum += baseline[i]
    value = (baseline_sum - n_nodes) * end_time

    for i in range(n_nodes):
        for j in range(n_nodes):
            primitive = 0.0
            decay = decays[i, j]
            if decay != 0.0:
                for k in range(sizes[j]):
                    tk = events[j, k]
                    if tk >= end_time:
                        break
                    primitive += 1.0 - math.exp(-decay * (end_time - tk))
            value += adjacency[i, j] * primitive

    for i in range(n_nodes):
        for k in range(sizes[i]):
            t = events[i, k]
            intensity = baseline[i]
            for j in range(n_nodes):
                feature = 0.0
                decay = decays[i, j]
                for m in range(sizes[j]):
                    tk = events[j, m]
                    if tk >= t:
                        break
                    feature += decay * math.exp(-decay * (t - tk))
                intensity += adjacency[i, j] * feature
            if intensity <= 0.0:
                return np.inf
            value -= math.log(intensity)
    return value


def _sumexp_loglik_loss_scan_numba_impl(events, sizes, end_time, baseline, adjacency, decays):
    n_nodes = baseline.size
    n_decays = decays.size
    baseline_sum = 0.0
    for i in range(n_nodes):
        baseline_sum += baseline[i]
    value = (baseline_sum - n_nodes) * end_time

    for i in range(n_nodes):
        for j in range(n_nodes):
            for u in range(n_decays):
                primitive = 0.0
                decay = decays[u]
                for k in range(sizes[j]):
                    tk = events[j, k]
                    if tk >= end_time:
                        break
                    primitive += 1.0 - math.exp(-decay * (end_time - tk))
                value += adjacency[i, j, u] * primitive

    for i in range(n_nodes):
        for k in range(sizes[i]):
            t = events[i, k]
            intensity = baseline[i]
            for j in range(n_nodes):
                for u in range(n_decays):
                    feature = 0.0
                    decay = decays[u]
                    for m in range(sizes[j]):
                        tk = events[j, m]
                        if tk >= t:
                            break
                        feature += decay * math.exp(-decay * (t - tk))
                    intensity += adjacency[i, j, u] * feature
            if intensity <= 0.0:
                return np.inf
            value -= math.log(intensity)
    return value


def _exp_loglik_grad_scan_numba_impl(events, sizes, end_time, baseline, adjacency, decays, grad_baseline, grad_adjacency):
    n_nodes = baseline.size
    for i in range(n_nodes):
        grad_baseline[i] = end_time
        for j in range(n_nodes):
            primitive = 0.0
            decay = decays[i, j]
            if decay != 0.0:
                for k in range(sizes[j]):
                    tk = events[j, k]
                    if tk >= end_time:
                        break
                    primitive += 1.0 - math.exp(-decay * (end_time - tk))
            grad_adjacency[i, j] = primitive

    for i in range(n_nodes):
        for k in range(sizes[i]):
            t = events[i, k]
            intensity = baseline[i]
            features = np.empty(n_nodes, dtype=np.float64)
            for j in range(n_nodes):
                feature = 0.0
                decay = decays[i, j]
                for m in range(sizes[j]):
                    tk = events[j, m]
                    if tk >= t:
                        break
                    feature += decay * math.exp(-decay * (t - tk))
                features[j] = feature
                intensity += adjacency[i, j] * feature
            if intensity <= 0.0:
                continue
            grad_baseline[i] -= 1.0 / intensity
            for j in range(n_nodes):
                grad_adjacency[i, j] -= features[j] / intensity


def _sumexp_loglik_grad_scan_numba_impl(events, sizes, end_time, baseline, adjacency, decays, grad_baseline, grad_adjacency):
    n_nodes = baseline.size
    n_decays = decays.size
    for i in range(n_nodes):
        grad_baseline[i] = end_time
        for j in range(n_nodes):
            for u in range(n_decays):
                primitive = 0.0
                decay = decays[u]
                for k in range(sizes[j]):
                    tk = events[j, k]
                    if tk >= end_time:
                        break
                    primitive += 1.0 - math.exp(-decay * (end_time - tk))
                grad_adjacency[i, j, u] = primitive

    for i in range(n_nodes):
        for k in range(sizes[i]):
            t = events[i, k]
            intensity = baseline[i]
            event_features = np.empty((n_nodes, n_decays), dtype=np.float64)
            for j in range(n_nodes):
                for u in range(n_decays):
                    feature = 0.0
                    decay = decays[u]
                    for m in range(sizes[j]):
                        tk = events[j, m]
                        if tk >= t:
                            break
                        feature += decay * math.exp(-decay * (t - tk))
                    event_features[j, u] = feature
                    intensity += adjacency[i, j, u] * feature
            if intensity <= 0.0:
                continue
            grad_baseline[i] -= 1.0 / intensity
            for j in range(n_nodes):
                for u in range(n_decays):
                    grad_adjacency[i, j, u] -= event_features[j, u] / intensity


def _feature_product_integral_numba_impl(end_time, timestamps_a, size_a, decay_a, timestamps_b, size_b, decay_b):
    if decay_a <= 0.0 or decay_b <= 0.0:
        return 0.0
    rate = decay_a + decay_b
    value = 0.0
    for a in range(size_a):
        ta = timestamps_a[a]
        if ta >= end_time:
            break
        for b in range(size_b):
            tb = timestamps_b[b]
            if tb >= end_time:
                break
            start = ta if ta >= tb else tb
            remaining = end_time - start
            if remaining <= 0.0:
                continue
            value += (
                decay_a
                * decay_b
                * math.exp(-decay_a * (start - ta) - decay_b * (start - tb))
                * (-math.expm1(-rate * remaining))
                / rate
            )
    return value


def _exp_ls_statistics_numba_impl(events, sizes, end_time, target_decays, target_node, feature_integrals, feature_products, event_feature_sums):
    n_nodes = sizes.size
    for j in range(n_nodes):
        primitive = 0.0
        decay = target_decays[j]
        if decay != 0.0:
            for k in range(sizes[j]):
                tk = events[j, k]
                if tk >= end_time:
                    break
                primitive += 1.0 - math.exp(-decay * (end_time - tk))
        feature_integrals[j] = primitive

        total = 0.0
        for k in range(sizes[target_node]):
            t = events[target_node, k]
            feature = 0.0
            for m in range(sizes[j]):
                tk = events[j, m]
                if tk >= t:
                    break
                feature += decay * math.exp(-decay * (t - tk))
            total += feature
        event_feature_sums[j] = total

    for j in range(n_nodes):
        for k in range(n_nodes):
            decay_a = target_decays[j]
            decay_b = target_decays[k]
            value = 0.0
            if decay_a > 0.0 and decay_b > 0.0:
                rate = decay_a + decay_b
                for a in range(sizes[j]):
                    ta = events[j, a]
                    if ta >= end_time:
                        break
                    for b in range(sizes[k]):
                        tb = events[k, b]
                        if tb >= end_time:
                            break
                        start = ta if ta >= tb else tb
                        remaining = end_time - start
                        if remaining <= 0.0:
                            continue
                        value += (
                            decay_a
                            * decay_b
                            * math.exp(-decay_a * (start - ta) - decay_b * (start - tb))
                            * (-math.expm1(-rate * remaining))
                            / rate
                        )
            feature_products[j, k] = value


def _sumexp_ls_integral_statistics_numba_impl(events, sizes, end_time, decays, feature_integrals, feature_products):
    n_nodes = sizes.size
    n_decays = decays.size
    for j in range(n_nodes):
        for u in range(n_decays):
            primitive = 0.0
            decay = decays[u]
            for k in range(sizes[j]):
                tk = events[j, k]
                if tk >= end_time:
                    break
                primitive += 1.0 - math.exp(-decay * (end_time - tk))
            feature_integrals[j, u] = primitive

    for j in range(n_nodes):
        for u in range(n_decays):
            left = j * n_decays + u
            for k in range(n_nodes):
                for v in range(n_decays):
                    right = k * n_decays + v
                    decay_a = decays[u]
                    decay_b = decays[v]
                    value = 0.0
                    if decay_a > 0.0 and decay_b > 0.0:
                        rate = decay_a + decay_b
                        for a in range(sizes[j]):
                            ta = events[j, a]
                            if ta >= end_time:
                                break
                            for b in range(sizes[k]):
                                tb = events[k, b]
                                if tb >= end_time:
                                    break
                                start = ta if ta >= tb else tb
                                remaining = end_time - start
                                if remaining <= 0.0:
                                    continue
                                value += (
                                    decay_a
                                    * decay_b
                                    * math.exp(-decay_a * (start - ta) - decay_b * (start - tb))
                                    * (-math.expm1(-rate * remaining))
                                    / rate
                                )
                    feature_products[left, right] = value


def _sumexp_ls_event_feature_sums_numba_impl(events, sizes, decays, target_node, event_feature_sums):
    n_nodes = sizes.size
    n_decays = decays.size
    for j in range(n_nodes):
        for u in range(n_decays):
            event_feature_sums[j, u] = 0.0
    for k in range(sizes[target_node]):
        t = events[target_node, k]
        for j in range(n_nodes):
            for u in range(n_decays):
                feature = 0.0
                decay = decays[u]
                for m in range(sizes[j]):
                    tk = events[j, m]
                    if tk >= t:
                        break
                    feature += decay * math.exp(-decay * (t - tk))
                event_feature_sums[j, u] += feature


def _time_function_left_border_value(border_type, border_value, y0):
    if border_type == 1:
        return border_value
    if border_type == 2:
        return y0
    return 0.0


def _time_function_right_border_value(border_type, border_value, yn):
    if border_type == 1:
        return border_value
    if border_type == 2:
        return yn
    return 0.0


def _time_function_value_numba_impl(
    x,
    t_values,
    y_values,
    border_type,
    inter_mode,
    border_value,
    is_constant,
    constant,
):
    if is_constant:
        return constant if x >= 0.0 else 0.0
    if x < 0.0:
        return 0.0

    t0 = t_values[0]
    tn = t_values[t_values.size - 1]
    if border_type == 3 and tn > t0:
        period = tn - t0
        x = ((x - t0) % period) + t0

    if x < t0:
        return _time_function_left_border_numba(border_type, border_value, y_values[0])
    if x > tn:
        return _time_function_right_border_numba(
            border_type, border_value, y_values[y_values.size - 1]
        )
    if x == tn:
        return y_values[y_values.size - 1]

    idx = np.searchsorted(t_values, x, side="right") - 1
    if idx < 0:
        idx = 0
    upper_idx = t_values.size - 2
    if idx > upper_idx:
        idx = upper_idx

    if inter_mode == 1:
        return y_values[idx + 1]
    if inter_mode == 2:
        return y_values[idx]

    t_left = t_values[idx]
    t_right = t_values[idx + 1]
    weight = (x - t_left) / (t_right - t_left)
    return (1.0 - weight) * y_values[idx] + weight * y_values[idx + 1]


def _time_function_future_bound_numba_impl(
    lag,
    t_values,
    y_values,
    border_type,
    inter_mode,
    border_value,
    is_constant,
    constant,
):
    if is_constant:
        return constant if constant > 0.0 else 0.0

    bound = 0.0
    if border_type == 3:
        for i in range(y_values.size):
            if y_values[i] > bound:
                bound = y_values[i]
        return bound if bound > 0.0 else 0.0

    for i in range(t_values.size):
        if t_values[i] >= lag and y_values[i] > bound:
            bound = y_values[i]

    right = _time_function_right_border_numba(
        border_type, border_value, y_values[y_values.size - 1]
    )
    if right > bound:
        bound = right

    current = _time_function_value_numba(
        lag,
        t_values,
        y_values,
        border_type,
        inter_mode,
        border_value,
        is_constant,
        constant,
    )
    if current > bound:
        bound = current
    return bound if bound > 0.0 else 0.0


def _timefunc_kernel_convolution_numba_impl(
    time,
    timestamps,
    t_values,
    y_values,
    border_type,
    inter_mode,
    border_value,
    is_constant,
    constant,
    support,
    include_current,
):
    start = 0
    if math.isfinite(support):
        lower = time - support
        while start < timestamps.size and timestamps[start] <= lower:
            start += 1

    value = 0.0
    for k in range(start, timestamps.size):
        tk = timestamps[k]
        if tk > time or (tk == time and not include_current):
            break
        value += _time_function_value_numba(
            time - tk,
            t_values,
            y_values,
            border_type,
            inter_mode,
            border_value,
            is_constant,
            constant,
        )
    return value


def _timefunc_kernel_future_bound_sum_numba_impl(
    time,
    timestamps,
    t_values,
    y_values,
    border_type,
    inter_mode,
    border_value,
    is_constant,
    constant,
    support,
    include_current,
):
    start = 0
    if math.isfinite(support):
        lower = time - support
        while start < timestamps.size and timestamps[start] <= lower:
            start += 1

    value = 0.0
    for k in range(start, timestamps.size):
        tk = timestamps[k]
        if tk > time or (tk == time and not include_current):
            break
        value += _time_function_future_bound_numba(
            time - tk,
            t_values,
            y_values,
            border_type,
            inter_mode,
            border_value,
            is_constant,
            constant,
        )
    return value


_exp_feature_at_time_numba = _compile(_exp_feature_at_time_numba_impl)
_exp_primitive_sum_numba = _compile(_exp_primitive_sum_numba_impl)
_sumexp_feature_at_time_numba = _compile(_sumexp_feature_at_time_numba_impl)
_sumexp_primitive_sum_numba = _compile(_sumexp_primitive_sum_numba_impl)
_exp_intensity_vector_numba = _compile(_exp_intensity_vector_numba_impl)
_sumexp_intensity_vector_numba = _compile(_sumexp_intensity_vector_numba_impl)
_exp_compensator_value_numba = _compile(_exp_compensator_value_numba_impl)
_sumexp_compensator_value_numba = _compile(_sumexp_compensator_value_numba_impl)
_homogeneous_poisson_events_numba = _compile(_homogeneous_poisson_events_numba_impl)
_exp_intensity_bound_numba = _compile(_exp_intensity_bound_numba_impl)
_simulate_exp_hawkes_numba = _compile(_simulate_exp_hawkes_numba_impl)
_mixed_timefunc_value = _compile(_mixed_timefunc_value)
_mixed_timefunc_future_bound = _compile(_mixed_timefunc_future_bound)
_mixed_kernel_value = _compile(_mixed_kernel_value)
_mixed_kernel_future_bound = _compile(_mixed_kernel_future_bound)
_simulate_mixed_hawkes_numba = _compile(_simulate_mixed_hawkes_numba_impl)
_sumexp_intensity_bound_numba = _compile(_sumexp_intensity_bound_numba_impl)
_exp_loglik_loss_scan_numba = _compile(_exp_loglik_loss_scan_numba_impl)
_sumexp_loglik_loss_scan_numba = _compile(_sumexp_loglik_loss_scan_numba_impl)
_exp_loglik_grad_scan_numba = _compile(_exp_loglik_grad_scan_numba_impl)
_sumexp_loglik_grad_scan_numba = _compile(_sumexp_loglik_grad_scan_numba_impl)
_feature_product_integral_numba = _compile(_feature_product_integral_numba_impl)
_exp_ls_statistics_numba = _compile(_exp_ls_statistics_numba_impl)
_sumexp_ls_integral_statistics_numba = _compile(_sumexp_ls_integral_statistics_numba_impl)
_sumexp_ls_event_feature_sums_numba = _compile(_sumexp_ls_event_feature_sums_numba_impl)
_time_function_left_border_numba = _compile(_time_function_left_border_value)
_time_function_right_border_numba = _compile(_time_function_right_border_value)
_time_function_value_numba = _compile(_time_function_value_numba_impl)
_time_function_future_bound_numba = _compile(_time_function_future_bound_numba_impl)
_timefunc_kernel_convolution_numba = _compile(_timefunc_kernel_convolution_numba_impl)
_timefunc_kernel_future_bound_sum_numba = _compile(
    _timefunc_kernel_future_bound_sum_numba_impl
)


def exp_feature_at_time(t: float, timestamps: np.ndarray, decay: float) -> float:
    timestamps = _float_array(timestamps)
    if NUMBA_AVAILABLE:
        return float(_exp_feature_at_time_numba(float(t), timestamps, float(decay), False))
    return exp_feature_at_time_reference(t, timestamps, decay)


def exp_primitive_sum(end_time: float, timestamps: np.ndarray, decay: float) -> float:
    timestamps = _float_array(timestamps)
    if NUMBA_AVAILABLE:
        return float(_exp_primitive_sum_numba(float(end_time), timestamps, float(decay)))
    return exp_primitive_sum_reference(end_time, timestamps, decay)


def sumexp_feature_at_time(
    t: float,
    timestamps: np.ndarray,
    decays: np.ndarray,
    out: np.ndarray,
) -> None:
    timestamps = _float_array(timestamps)
    decays = _float_array(decays)
    out_arr = np.asarray(out, dtype=np.float64)
    if NUMBA_AVAILABLE:
        _sumexp_feature_at_time_numba(float(t), timestamps, decays, out_arr, False)
    else:
        sumexp_feature_at_time_reference(t, timestamps, decays, out_arr)


def sumexp_primitive_sum(
    end_time: float,
    timestamps: np.ndarray,
    decays: np.ndarray,
    out: np.ndarray,
) -> None:
    timestamps = _float_array(timestamps)
    decays = _float_array(decays)
    out_arr = np.asarray(out, dtype=np.float64)
    if NUMBA_AVAILABLE:
        _sumexp_primitive_sum_numba(float(end_time), timestamps, decays, out_arr)
    else:
        sumexp_primitive_sum_reference(end_time, timestamps, decays, out_arr)


def exp_kernel_convolution(
    time: float,
    timestamps: np.ndarray,
    intensity: float,
    decay: float,
    include_current: bool = False,
) -> float:
    timestamps = _float_array(timestamps)
    if NUMBA_AVAILABLE:
        feature = _exp_feature_at_time_numba(
            float(time), timestamps, float(decay), bool(include_current)
        )
        return float(float(intensity) * feature)
    return exp_kernel_convolution_reference(time, timestamps, intensity, decay, include_current)


def exp_kernel_primitive_convolution(
    time: float,
    timestamps: np.ndarray,
    intensity: float,
    decay: float,
) -> float:
    timestamps = _float_array(timestamps)
    if NUMBA_AVAILABLE:
        primitive = _exp_primitive_sum_numba(float(time), timestamps, float(decay))
        return float(float(intensity) * primitive)
    return exp_kernel_primitive_convolution_reference(time, timestamps, intensity, decay)


def sumexp_kernel_convolution(
    time: float,
    timestamps: np.ndarray,
    intensities: np.ndarray,
    decays: np.ndarray,
    include_current: bool = False,
) -> float:
    timestamps = _float_array(timestamps)
    intensities = _float_array(intensities)
    decays = _float_array(decays)
    tmp = np.empty(decays.size, dtype=np.float64)
    if NUMBA_AVAILABLE:
        _sumexp_feature_at_time_numba(float(time), timestamps, decays, tmp, bool(include_current))
    else:
        sumexp_feature_at_time_reference(time, timestamps, decays, tmp, include_current)
    return float(np.dot(intensities, tmp))


def sumexp_kernel_primitive_convolution(
    time: float,
    timestamps: np.ndarray,
    intensities: np.ndarray,
    decays: np.ndarray,
) -> float:
    timestamps = _float_array(timestamps)
    intensities = _float_array(intensities)
    decays = _float_array(decays)
    tmp = np.empty(decays.size, dtype=np.float64)
    if NUMBA_AVAILABLE:
        _sumexp_primitive_sum_numba(float(time), timestamps, decays, tmp)
    else:
        sumexp_primitive_sum_reference(time, timestamps, decays, tmp)
    return float(np.dot(intensities, tmp))


def timefunc_kernel_convolution(
    time: float,
    timestamps: np.ndarray,
    t_values: np.ndarray,
    y_values: np.ndarray,
    border_type: int,
    inter_mode: int,
    border_value: float,
    is_constant: bool,
    constant: float,
    support: float,
    include_current: bool = False,
) -> float:
    """Evaluate a finite-support time-function kernel convolution with JIT."""

    return float(
        _timefunc_kernel_convolution_numba(
            float(time),
            _float_array(timestamps),
            _float_array(t_values),
            _float_array(y_values),
            int(border_type),
            int(inter_mode),
            float(border_value),
            bool(is_constant),
            float(constant),
            float(support),
            bool(include_current),
        )
    )


def timefunc_kernel_future_bound_sum(
    time: float,
    timestamps: np.ndarray,
    t_values: np.ndarray,
    y_values: np.ndarray,
    border_type: int,
    inter_mode: int,
    border_value: float,
    is_constant: bool,
    constant: float,
    support: float,
    include_current: bool = True,
) -> float:
    """Sum conservative future bounds for a time-function kernel with JIT."""

    return float(
        _timefunc_kernel_future_bound_sum_numba(
            float(time),
            _float_array(timestamps),
            _float_array(t_values),
            _float_array(y_values),
            int(border_type),
            int(inter_mode),
            float(border_value),
            bool(is_constant),
            float(constant),
            float(support),
            bool(include_current),
        )
    )


def exp_intensity_vector(
    t: float,
    events: np.ndarray,
    sizes: np.ndarray,
    baseline: np.ndarray,
    adjacency: np.ndarray,
    decays: np.ndarray,
    include_current: bool = False,
) -> np.ndarray:
    events = _float_array(events)
    sizes = _int_array(sizes)
    baseline = _float_array(baseline)
    adjacency = _float_array(adjacency)
    decays = _float_array(decays)
    if NUMBA_AVAILABLE:
        out = np.empty_like(baseline)
        _exp_intensity_vector_numba(
            float(t), events, sizes, baseline, adjacency, decays, bool(include_current), out
        )
        return out
    return exp_intensity_vector_reference(t, events, sizes, baseline, adjacency, decays, include_current)


def sumexp_intensity_vector(
    t: float,
    events: np.ndarray,
    sizes: np.ndarray,
    baseline: np.ndarray,
    adjacency: np.ndarray,
    decays: np.ndarray,
    include_current: bool = False,
) -> np.ndarray:
    events = _float_array(events)
    sizes = _int_array(sizes)
    baseline = _float_array(baseline)
    adjacency = _float_array(adjacency)
    decays = _float_array(decays)
    if NUMBA_AVAILABLE:
        out = np.empty_like(baseline)
        _sumexp_intensity_vector_numba(
            float(t), events, sizes, baseline, adjacency, decays, bool(include_current), out
        )
        return out
    return sumexp_intensity_vector_reference(t, events, sizes, baseline, adjacency, decays, include_current)


def exp_compensator_value(
    node: int,
    t: float,
    events: np.ndarray,
    sizes: np.ndarray,
    baseline: np.ndarray,
    adjacency: np.ndarray,
    decays: np.ndarray,
) -> float:
    events = _float_array(events)
    sizes = _int_array(sizes)
    baseline = _float_array(baseline)
    adjacency = _float_array(adjacency)
    decays = _float_array(decays)
    if NUMBA_AVAILABLE:
        return float(_exp_compensator_value_numba(int(node), float(t), events, sizes, baseline, adjacency, decays))
    return exp_compensator_value_reference(node, t, events, sizes, baseline, adjacency, decays)


def sumexp_compensator_value(
    node: int,
    t: float,
    events: np.ndarray,
    sizes: np.ndarray,
    baseline: np.ndarray,
    adjacency: np.ndarray,
    decays: np.ndarray,
) -> float:
    events = _float_array(events)
    sizes = _int_array(sizes)
    baseline = _float_array(baseline)
    adjacency = _float_array(adjacency)
    decays = _float_array(decays)
    if NUMBA_AVAILABLE:
        return float(_sumexp_compensator_value_numba(int(node), float(t), events, sizes, baseline, adjacency, decays))
    return sumexp_compensator_value_reference(node, t, events, sizes, baseline, adjacency, decays)


def homogeneous_poisson_events(
    start_time: float,
    end_time: float,
    intensities: np.ndarray,
    uniforms: np.ndarray,
    out_times: np.ndarray,
    out_nodes: np.ndarray,
) -> tuple[int, float, bool]:
    intensities = _float_array(intensities)
    uniforms = _float_array(uniforms)
    out_times_arr = np.asarray(out_times, dtype=np.float64)
    out_nodes_arr = np.asarray(out_nodes, dtype=np.int64)
    if NUMBA_AVAILABLE:
        count, stop_time, hit_end = _homogeneous_poisson_events_numba(
            float(start_time),
            float(end_time),
            intensities,
            uniforms,
            out_times_arr,
            out_nodes_arr,
        )
        return int(count), float(stop_time), bool(hit_end)
    return homogeneous_poisson_events_reference(
        start_time,
        end_time,
        intensities,
        uniforms,
        out_times_arr,
        out_nodes_arr,
    )


def exp_intensity_bound(
    t: float,
    events: np.ndarray,
    sizes: np.ndarray,
    baseline: np.ndarray,
    adjacency: np.ndarray,
    decays: np.ndarray,
    include_current: bool = True,
) -> float:
    events = _float_array(events)
    sizes = _int_array(sizes)
    baseline = _float_array(baseline)
    adjacency = _float_array(adjacency)
    decays = _float_array(decays)
    if NUMBA_AVAILABLE:
        return float(
            _exp_intensity_bound_numba(
                float(t),
                events,
                sizes,
                baseline,
                adjacency,
                decays,
                bool(include_current),
            )
        )
    return exp_intensity_bound_reference(t, events, sizes, baseline, adjacency, decays, include_current)


def simulate_exp_hawkes(
    baseline: np.ndarray,
    adjacency: np.ndarray,
    decays: np.ndarray,
    end_time: float,
    max_jumps: int,
    seed: int,
    threshold_negative_intensity: bool,
    capacity: int,
):
    """Simulate constant-baseline exponential Hawkes events with recursive state."""

    baseline = _float_array(baseline)
    adjacency = _float_array(adjacency)
    decays = _float_array(decays)
    events, sizes, stop_time, n_total_jumps, status = _simulate_exp_hawkes_numba(
        baseline,
        adjacency,
        decays,
        float(end_time),
        int(max_jumps),
        int(seed),
        bool(threshold_negative_intensity),
        int(capacity),
    )
    return events, sizes, float(stop_time), int(n_total_jumps), int(status)


def simulate_mixed_hawkes(
    baseline: np.ndarray,
    kernel_types: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    p3: np.ndarray,
    supports: np.ndarray,
    tf_t_values: np.ndarray,
    tf_y_values: np.ndarray,
    tf_sizes: np.ndarray,
    tf_inter_modes: np.ndarray,
    tf_border_types: np.ndarray,
    tf_border_values: np.ndarray,
    end_time: float,
    max_jumps: int,
    seed: int,
    threshold_negative_intensity: bool,
    capacity: int,
):
    """Simulate constant-baseline Hawkes events with compiled finite-kernel scans."""

    baseline = _float_array(baseline)
    kernel_types = _int_array(kernel_types)
    p1 = _float_array(p1)
    p2 = _float_array(p2)
    p3 = _float_array(p3)
    supports = _float_array(supports)
    tf_t_values = _float_array(tf_t_values)
    tf_y_values = _float_array(tf_y_values)
    tf_sizes = _int_array(tf_sizes)
    tf_inter_modes = _int_array(tf_inter_modes)
    tf_border_types = _int_array(tf_border_types)
    tf_border_values = _float_array(tf_border_values)
    events, sizes, stop_time, n_total_jumps, status = _simulate_mixed_hawkes_numba(
        baseline,
        kernel_types,
        p1,
        p2,
        p3,
        supports,
        tf_t_values,
        tf_y_values,
        tf_sizes,
        tf_inter_modes,
        tf_border_types,
        tf_border_values,
        float(end_time),
        int(max_jumps),
        int(seed),
        bool(threshold_negative_intensity),
        int(capacity),
    )
    return events, sizes, float(stop_time), int(n_total_jumps), int(status)


def sumexp_intensity_bound(
    t: float,
    events: np.ndarray,
    sizes: np.ndarray,
    baseline: np.ndarray,
    adjacency: np.ndarray,
    decays: np.ndarray,
    include_current: bool = True,
) -> float:
    events = _float_array(events)
    sizes = _int_array(sizes)
    baseline = _float_array(baseline)
    adjacency = _float_array(adjacency)
    decays = _float_array(decays)
    if NUMBA_AVAILABLE:
        return float(
            _sumexp_intensity_bound_numba(
                float(t),
                events,
                sizes,
                baseline,
                adjacency,
                decays,
                bool(include_current),
            )
        )
    return sumexp_intensity_bound_reference(t, events, sizes, baseline, adjacency, decays, include_current)


def exp_loglik_loss_scan(
    events: np.ndarray,
    sizes: np.ndarray,
    end_time: float,
    baseline: np.ndarray,
    adjacency: np.ndarray,
    decays: np.ndarray,
) -> float:
    events = _float_array(events)
    sizes = _int_array(sizes)
    baseline = _float_array(baseline)
    adjacency = _float_array(adjacency)
    decays = _float_array(decays)
    if NUMBA_AVAILABLE:
        return float(_exp_loglik_loss_scan_numba(events, sizes, float(end_time), baseline, adjacency, decays))
    return exp_loglik_loss_scan_reference(events, sizes, end_time, baseline, adjacency, decays)


def sumexp_loglik_loss_scan(
    events: np.ndarray,
    sizes: np.ndarray,
    end_time: float,
    baseline: np.ndarray,
    adjacency: np.ndarray,
    decays: np.ndarray,
) -> float:
    events = _float_array(events)
    sizes = _int_array(sizes)
    baseline = _float_array(baseline)
    adjacency = _float_array(adjacency)
    decays = _float_array(decays)
    if NUMBA_AVAILABLE:
        return float(_sumexp_loglik_loss_scan_numba(events, sizes, float(end_time), baseline, adjacency, decays))
    return sumexp_loglik_loss_scan_reference(events, sizes, end_time, baseline, adjacency, decays)


def exp_loglik_grad_scan(
    events: np.ndarray,
    sizes: np.ndarray,
    end_time: float,
    baseline: np.ndarray,
    adjacency: np.ndarray,
    decays: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    events = _float_array(events)
    sizes = _int_array(sizes)
    baseline = _float_array(baseline)
    adjacency = _float_array(adjacency)
    decays = _float_array(decays)
    grad_baseline = np.empty_like(baseline)
    grad_adjacency = np.empty_like(adjacency)
    if NUMBA_AVAILABLE:
        _exp_loglik_grad_scan_numba(
            events, sizes, float(end_time), baseline, adjacency, decays, grad_baseline, grad_adjacency
        )
        return grad_baseline, grad_adjacency
    return exp_loglik_grad_scan_reference(events, sizes, end_time, baseline, adjacency, decays)


def sumexp_loglik_grad_scan(
    events: np.ndarray,
    sizes: np.ndarray,
    end_time: float,
    baseline: np.ndarray,
    adjacency: np.ndarray,
    decays: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    events = _float_array(events)
    sizes = _int_array(sizes)
    baseline = _float_array(baseline)
    adjacency = _float_array(adjacency)
    decays = _float_array(decays)
    grad_baseline = np.empty_like(baseline)
    grad_adjacency = np.empty_like(adjacency)
    if NUMBA_AVAILABLE:
        _sumexp_loglik_grad_scan_numba(
            events, sizes, float(end_time), baseline, adjacency, decays, grad_baseline, grad_adjacency
        )
        return grad_baseline, grad_adjacency
    return sumexp_loglik_grad_scan_reference(events, sizes, end_time, baseline, adjacency, decays)


def feature_product_integral(
    end_time: float,
    timestamps_a: np.ndarray,
    decay_a: float,
    timestamps_b: np.ndarray,
    decay_b: float,
) -> float:
    timestamps_a = _float_array(timestamps_a)
    timestamps_b = _float_array(timestamps_b)
    if NUMBA_AVAILABLE:
        return float(
            _feature_product_integral_numba(
                float(end_time),
                timestamps_a,
                int(timestamps_a.size),
                float(decay_a),
                timestamps_b,
                int(timestamps_b.size),
                float(decay_b),
            )
        )
    return feature_product_integral_reference(end_time, timestamps_a, decay_a, timestamps_b, decay_b)


def exp_ls_statistics(
    events: np.ndarray,
    sizes: np.ndarray,
    end_time: float,
    target_decays: np.ndarray,
    target_node: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    events = _float_array(events)
    sizes = _int_array(sizes)
    target_decays = _float_array(target_decays)
    n_nodes = sizes.size
    feature_integrals = np.empty(n_nodes, dtype=np.float64)
    feature_products = np.empty((n_nodes, n_nodes), dtype=np.float64)
    event_feature_sums = np.empty(n_nodes, dtype=np.float64)
    if NUMBA_AVAILABLE:
        _exp_ls_statistics_numba(
            events,
            sizes,
            float(end_time),
            target_decays,
            int(target_node),
            feature_integrals,
            feature_products,
            event_feature_sums,
        )
        return feature_integrals, feature_products, event_feature_sums
    return exp_ls_statistics_reference(events, sizes, end_time, target_decays, target_node)


def sumexp_ls_integral_statistics(
    events: np.ndarray,
    sizes: np.ndarray,
    end_time: float,
    decays: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    events = _float_array(events)
    sizes = _int_array(sizes)
    decays = _float_array(decays)
    n_nodes = sizes.size
    n_decays = decays.size
    feature_integrals = np.empty((n_nodes, n_decays), dtype=np.float64)
    feature_products = np.empty((n_nodes * n_decays, n_nodes * n_decays), dtype=np.float64)
    if NUMBA_AVAILABLE:
        _sumexp_ls_integral_statistics_numba(
            events, sizes, float(end_time), decays, feature_integrals, feature_products
        )
        return feature_integrals, feature_products
    return sumexp_ls_integral_statistics_reference(events, sizes, end_time, decays)


def sumexp_ls_event_feature_sums(
    events: np.ndarray,
    sizes: np.ndarray,
    decays: np.ndarray,
    target_node: int,
) -> np.ndarray:
    events = _float_array(events)
    sizes = _int_array(sizes)
    decays = _float_array(decays)
    out = np.empty((sizes.size, decays.size), dtype=np.float64)
    if NUMBA_AVAILABLE:
        _sumexp_ls_event_feature_sums_numba(events, sizes, decays, int(target_node), out)
        return out
    return sumexp_ls_event_feature_sums_reference(events, sizes, decays, target_node)


def finite_difference_grad(func, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    grad = np.zeros_like(x)
    for idx in range(x.size):
        step = eps * max(1.0, abs(float(x[idx])))
        xp = x.copy()
        xm = x.copy()
        xp[idx] += step
        xm[idx] -= step
        grad[idx] = (func(xp) - func(xm)) / (2.0 * step)
    return grad


def finite_difference_hessian(func, x: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    n = x.size
    hess = np.zeros((n, n), dtype=float)
    for i in range(n):
        step = eps * max(1.0, abs(float(x[i])))
        xp = x.copy()
        xm = x.copy()
        xp[i] += step
        xm[i] -= step
        gp = finite_difference_grad(func, xp)
        gm = finite_difference_grad(func, xm)
        hess[:, i] = (gp - gm) / (2.0 * step)
    return 0.5 * (hess + hess.T)

