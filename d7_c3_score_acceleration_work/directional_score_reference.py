"""Exact complete directional score and low-variance summaries."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np


def _softmax(values: np.ndarray) -> np.ndarray:
    row = np.asarray(values, dtype=float)
    shifted = np.exp(row - float(np.max(row)))
    return shifted / float(np.sum(shifted))


def _weight_stats(values: np.ndarray) -> dict[str, float]:
    weights = _softmax(np.asarray(values, dtype=float))
    ordered = np.sort(weights)
    count = weights.size
    top_point_one = max(1, int(math.ceil(0.001 * count)))
    top_one = max(1, int(math.ceil(0.01 * count)))
    return {
        "absolute_ess": 1.0 / float(np.sum(weights * weights)),
        "ess_fraction": 1.0 / float(count * np.sum(weights * weights)),
        "maximum_weight_share": float(ordered[-1]),
        "top_point_one_percent_share": float(np.sum(ordered[-top_point_one:])),
        "top_one_percent_share": float(np.sum(ordered[-top_one:])),
    }


def evaluate_directional_archive(
    point: np.ndarray,
    direction: np.ndarray,
    *,
    context: Any,
    archive: Any,
    windows: Sequence[Mapping[str, Any]],
    adapter: Any,
    clipping_cap: float,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Evaluate one own-target archive and retain raw score ingredients.

    The proposal is held fixed.  For every trace/window pair this returns
    ``D_d log p_theta,C(X) + D_d log L_theta,w(X)`` and the exact log weight.
    The current vectorized likelihood is replayed for deterministic value
    parity on every production archive.
    """

    score_adapter = adapter.score_adapter
    analytic = score_adapter.analytic
    core = adapter.d7
    z = np.asarray(point, dtype=np.float64)
    d = np.asarray(direction, dtype=np.float64)
    law, kernel, _ = core.model_from_normalized(z, context)
    evaluator = analytic.RenewalScoreEvaluator(law)
    decode_jacobian = score_adapter.normalized_decode_jacobian(z, context)
    physical_direction = decode_jacobian @ d

    source_trace_log = np.asarray(
        [core.base.trace_log_density(trace, law, kernel) for trace in archive.traces],
        dtype=float,
    )
    proposal_log = np.asarray(archive.proposal_log_density, dtype=float)
    trace_ratio = source_trace_log - proposal_log
    identity_error = float(np.max(np.abs(trace_ratio)))
    trace_physical_scores = np.stack(
        [analytic.boundary_trace_score(trace, evaluator, kernel) for trace in archive.traces]
    )
    trace_directional_scores = trace_physical_scores @ physical_direction

    window_count = len(windows)
    sample_count = archive.size
    log_weights = np.empty((window_count, sample_count), dtype=float)
    complete_scores = np.empty_like(log_weights)
    conditional_scores = np.empty_like(log_weights)
    conditional_values = np.empty_like(log_weights)
    maximum_value_parity_error = 0.0
    clipped_count = 0
    log_cap = math.log(float(clipping_cap))

    excitation = core.base.boundary_excitation_batch(archive.traces, kernel)
    for window_index, window in enumerate(windows):
        values: list[float] = []
        scores: list[float] = []
        for trace in archive.traces:
            value, physical_score = analytic.conditional_log_density_and_gradient(
                window["events"], float(window["horizon"]), trace, evaluator, kernel
            )
            values.append(float(value))
            scores.append(float(np.asarray(physical_score, dtype=float) @ physical_direction))
        analytic_values = np.asarray(values, dtype=float)
        vectorized_values = core.base.conditional_log_density_batch(
            archive.traces, window, law, kernel, excitation=excitation
        )
        maximum_value_parity_error = max(
            maximum_value_parity_error,
            float(np.max(np.abs(analytic_values - vectorized_values))),
        )
        conditional_values[window_index] = analytic_values
        conditional_scores[window_index] = np.asarray(scores, dtype=float)
        log_weights[window_index] = trace_ratio + analytic_values
        complete_scores[window_index] = trace_directional_scores + conditional_scores[window_index]
        reference = core.base._window_reference_log_density(window, law, kernel)
        clipped = log_weights[window_index] - reference > log_cap
        clipped_count += int(np.count_nonzero(clipped))

    arrays = {
        "log_weights": log_weights,
        "complete_directional_scores": complete_scores,
        "trace_directional_scores": trace_directional_scores,
        "conditional_directional_scores": conditional_scores,
        "conditional_log_likelihood": conditional_values,
        "source_trace_log_density": source_trace_log,
        "proposal_log_density": proposal_log,
    }
    if any(np.any(~np.isfinite(row)) for row in arrays.values()):
        raise FloatingPointError("complete directional-score archive is nonfinite")
    diagnostics = {
        "archive_size": int(archive.size),
        "archive_seed": int(archive.seed),
        "trace_archive_sha256": score_adapter.trace_archive_sha256(archive),
        "proposal_identity_maximum_absolute_error": identity_error,
        "conditional_value_parity_maximum_absolute_error": maximum_value_parity_error,
        "clipped_count": int(clipped_count),
        "decode_jacobian_condition_number": float(np.linalg.cond(decode_jacobian)),
        "physical_direction_norm": float(np.linalg.norm(physical_direction)),
    }
    return arrays, diagnostics


def _ratio_cv_summary(log_weights: np.ndarray, complete: np.ndarray, trace: np.ndarray) -> dict[str, Any]:
    logw = np.asarray(log_weights, dtype=float)
    score = np.asarray(complete, dtype=float)
    tscore = np.asarray(trace, dtype=float)
    if logw.shape != score.shape or logw.shape[1] != tscore.size:
        raise ValueError("raw score arrays do not align")
    window_count, sample_count = logw.shape
    raw = np.empty(window_count, dtype=float)
    cv = np.empty(window_count, dtype=float)
    influence = np.empty_like(logw)
    jackknife_bias = np.empty(window_count, dtype=float)
    stats: dict[str, np.ndarray] = {
        key: np.empty(window_count, dtype=float)
        for key in (
            "absolute_ess",
            "ess_fraction",
            "maximum_weight_share",
            "top_point_one_percent_share",
            "top_one_percent_share",
        )
    }
    trace_mean = float(np.mean(tscore))
    trace_total = float(np.sum(tscore))
    for window in range(window_count):
        row = logw[window]
        maximum = float(np.max(row))
        unnormalized = np.exp(row - maximum)
        denominator = float(np.sum(unnormalized))
        numerator = float(np.sum(unnormalized * score[window]))
        raw[window] = numerator / denominator
        cv[window] = raw[window] - trace_mean
        q = unnormalized / (denominator / sample_count)
        influence[window] = q * (score[window] - raw[window]) - tscore
        loo_raw = (numerator - unnormalized * score[window]) / (denominator - unnormalized)
        loo_trace = (trace_total - tscore) / (sample_count - 1)
        loo_cv = loo_raw - loo_trace
        jackknife_bias[window] = (sample_count - 1) * (float(np.mean(loo_cv)) - cv[window])
        for key, value in _weight_stats(row).items():
            stats[key][window] = value
    window_se = np.std(influence, axis=1, ddof=1) / math.sqrt(sample_count)
    aggregate_influence = np.mean(influence, axis=0)
    aggregate = float(np.mean(cv))
    aggregate_raw = float(np.mean(raw))
    aggregate_se = float(np.std(aggregate_influence, ddof=1) / math.sqrt(sample_count))
    aggregate_bias = float(np.mean(jackknife_bias))
    trace_se = float(np.std(tscore, ddof=1) / math.sqrt(sample_count))
    return {
        "raw_window_scores": raw,
        "cv_window_scores": cv,
        "window_numerical_standard_errors": window_se,
        "window_jackknife_bias": jackknife_bias,
        "influence": influence,
        "aggregate_raw_score": aggregate_raw,
        "aggregate_cv_score": aggregate,
        "aggregate_numerical_standard_error": aggregate_se,
        "aggregate_jackknife_bias": aggregate_bias,
        "trace_score_mean": trace_mean,
        "trace_score_numerical_standard_error": trace_se,
        "weight_stats": stats,
        "sample_count": int(sample_count),
    }


def summarize_node(
    replicate_chunks: Sequence[Sequence[Mapping[str, np.ndarray]]],
    path_indices: np.ndarray,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Pool equal-size own-target chunks and preserve replicate diagnostics."""

    replicate_rows = []
    pooled_logw = []
    pooled_complete = []
    pooled_trace = []
    for chunks in replicate_chunks:
        logw = np.concatenate([np.asarray(row["log_weights"], dtype=float) for row in chunks], axis=1)
        complete = np.concatenate(
            [np.asarray(row["complete_directional_scores"], dtype=float) for row in chunks], axis=1
        )
        trace = np.concatenate([np.asarray(row["trace_directional_scores"], dtype=float) for row in chunks])
        replicate_rows.append(_ratio_cv_summary(logw, complete, trace))
        pooled_logw.append(logw)
        pooled_complete.append(complete)
        pooled_trace.append(trace)
    logw = np.concatenate(pooled_logw, axis=1)
    complete = np.concatenate(pooled_complete, axis=1)
    trace = np.concatenate(pooled_trace)
    pooled = _ratio_cv_summary(logw, complete, trace)
    indices = np.asarray(path_indices, dtype=int)
    window_scores = np.asarray(pooled["cv_window_scores"], dtype=float)
    raw_window_scores = np.asarray(pooled["raw_window_scores"], dtype=float)
    path_scores = np.asarray([np.mean(window_scores[indices == index]) for index in range(16)])
    raw_path_scores = np.asarray([np.mean(raw_window_scores[indices == index]) for index in range(16)])
    influence = np.asarray(pooled["influence"], dtype=float)
    path_influence = np.stack(
        [np.mean(influence[indices == index], axis=0) for index in range(16)]
    )
    path_mc_se = np.std(path_influence, axis=1, ddof=1) / math.sqrt(pooled["sample_count"])
    replicate_scores = np.asarray([row["aggregate_cv_score"] for row in replicate_rows])
    replicate_raw_scores = np.asarray([row["aggregate_raw_score"] for row in replicate_rows])
    weight_stats = pooled["weight_stats"]
    summary = {
        "score": pooled["aggregate_cv_score"],
        "unadjusted_score": pooled["aggregate_raw_score"],
        "numerical_standard_error": pooled["aggregate_numerical_standard_error"],
        "jackknife_bias": pooled["aggregate_jackknife_bias"],
        "trace_score_mean": pooled["trace_score_mean"],
        "trace_score_numerical_standard_error": pooled["trace_score_numerical_standard_error"],
        "trace_score_martingale_standardized_residual": (
            abs(pooled["trace_score_mean"]) / max(pooled["trace_score_numerical_standard_error"], 1e-300)
        ),
        "sample_count_total": pooled["sample_count"],
        "sample_count_per_replicate": int(replicate_chunks[0][0]["trace_directional_scores"].size * len(replicate_chunks[0])),
        "replicate_scores": replicate_scores,
        "replicate_raw_scores": replicate_raw_scores,
        "replicate_range": float(np.ptp(replicate_scores)),
        "replicate_standard_error": float(np.std(replicate_scores, ddof=1) / math.sqrt(len(replicate_scores))),
        "minimum_pooled_absolute_ess": float(np.min(weight_stats["absolute_ess"])),
        "minimum_pooled_ess_fraction": float(np.min(weight_stats["ess_fraction"])),
        "maximum_pooled_single_weight_share": float(np.max(weight_stats["maximum_weight_share"])),
        "maximum_pooled_top_point_one_percent_share": float(np.max(weight_stats["top_point_one_percent_share"])),
        "maximum_pooled_top_one_percent_share": float(np.max(weight_stats["top_one_percent_share"])),
        "minimum_replicate_absolute_ess": float(
            min(np.min(row["weight_stats"]["absolute_ess"]) for row in replicate_rows)
        ),
        "maximum_replicate_single_weight_share": float(
            max(np.max(row["weight_stats"]["maximum_weight_share"]) for row in replicate_rows)
        ),
        "maximum_replicate_top_point_one_percent_share": float(
            max(np.max(row["weight_stats"]["top_point_one_percent_share"]) for row in replicate_rows)
        ),
        "maximum_replicate_top_one_percent_share": float(
            max(np.max(row["weight_stats"]["top_one_percent_share"]) for row in replicate_rows)
        ),
        "maximum_absolute_window_jackknife_bias": float(np.max(np.abs(pooled["window_jackknife_bias"]))),
        "maximum_window_numerical_standard_error": float(np.max(pooled["window_numerical_standard_errors"])),
        "all_finite": True,
    }
    arrays = {
        "window_scores": window_scores,
        "unadjusted_window_scores": raw_window_scores,
        "window_numerical_standard_errors": np.asarray(pooled["window_numerical_standard_errors"]),
        "window_jackknife_bias": np.asarray(pooled["window_jackknife_bias"]),
        "path_scores": path_scores,
        "unadjusted_path_scores": raw_path_scores,
        "path_numerical_standard_errors": path_mc_se,
        "replicate_scores": replicate_scores,
        "replicate_raw_scores": replicate_raw_scores,
        "pooled_weight_absolute_ess": np.asarray(weight_stats["absolute_ess"]),
        "pooled_weight_ess_fraction": np.asarray(weight_stats["ess_fraction"]),
        "pooled_weight_maximum_share": np.asarray(weight_stats["maximum_weight_share"]),
        "pooled_weight_top_point_one_percent_share": np.asarray(weight_stats["top_point_one_percent_share"]),
        "pooled_weight_top_one_percent_share": np.asarray(weight_stats["top_one_percent_share"]),
    }
    return summary, arrays
