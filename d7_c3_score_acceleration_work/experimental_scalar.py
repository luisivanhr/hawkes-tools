"""Uncertified scalar boundary-law acceleration experiments.

This file is not installed into the evaluator.  It only consumes saved traces.
The baseline gap accumulation and math.log calls remain sequential.  Vector
BSpline calls replace repeated calls, but every coefficient contraction is still
the original one-row matrix/vector product.  NumPy powers keep one-element inputs
because scalar and vector ufunc implementations need not round identically.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
import time
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
from replay_support import *


def positive_scalar_gap_values(law, gaps, *, power_mode="one_element"):
    """Return h/H matching separate scalar calls, on positive finite gaps only.

    Non-body values use the untouched reference.  The caller must retain the
    original path for unsupported inputs, so this helper does not define new
    public validation or nonfinite behavior.
    """
    values = np.asarray(gaps, dtype=float)
    if values.ndim != 1 or np.any(~np.isfinite(values)) or np.any(values <= 0.):
        raise ValueError("prototype accepts only finite positive one-dimensional gaps")
    base = values / law.scale
    body = (base > 0.) & (base <= law.horizon)
    hazard = np.empty(values.size, dtype=float)
    cumulative = np.empty(values.size, dtype=float)
    selected = np.flatnonzero(body)
    ratios = base[body] / law.horizon
    if power_mode == "one_element":
        transformed = np.asarray([float((law.horizon * ratio ** law.origin_exponent)[0])
                                  for ratio in (ratios[i:i+1] for i in range(ratios.size))])
        warp = np.asarray([float((law.origin_exponent * ratio ** (law.origin_exponent - 1.))[0])
                           for ratio in (ratios[i:i+1] for i in range(ratios.size))])
    elif power_mode == "vector":
        transformed = law.horizon * ratios ** law.origin_exponent
        warp = law.origin_exponent * ratios ** (law.origin_exponent - 1.)
    else:
        raise ValueError(power_mode)
    basis = np.asarray(law.chart.basis(transformed), dtype=float)
    integrals = np.asarray(law.chart.integral_basis(transformed), dtype=float)
    for offset, index in enumerate(selected):
        coordinate = basis[offset:offset+1] @ law.coefficients
        hazard[index] = (coordinate * warp[offset:offset+1] / law.scale)[0]
        cumulative[index] = (integrals[offset:offset+1] @ law.coefficients)[0]
    for index in np.flatnonzero(~body):
        hazard[index] = law.hazard(float(values[index]))
        cumulative[index] = law.cumulative_hazard(float(values[index]))
    return hazard, cumulative


def renewal_trace_log_density_prototype(trace, law, *, power_mode="one_element"):
    """Fast path only for saved valid traces; retains exact Python accumulation."""
    value = -float(law.cumulative_hazard(trace.age)) - math.log(law.mean)
    gaps = np.asarray(trace.internal_gaps, dtype=float)
    if gaps.size:
        if gaps.ndim != 1 or np.any(~np.isfinite(gaps)) or np.any(gaps <= 0.):
            # Maintain baseline behavior and short-circuit order on unsupported
            # gaps; no preprocessing of later values may supersede that order.
            for gap in trace.internal_gaps:
                hazard = float(law.hazard(gap))
                if not math.isfinite(hazard) or hazard <= 0.:
                    return -math.inf
                value += math.log(hazard) - float(law.cumulative_hazard(gap))
        else:
            hazards, cumulative = positive_scalar_gap_values(law, gaps, power_mode=power_mode)
            for hazard, integral in zip(hazards, cumulative, strict=True):
                hazard = float(hazard)
                if not math.isfinite(hazard) or hazard <= 0.:
                    return -math.inf
                value += math.log(hazard) - float(integral)
    if trace.crossing_distance is not None:
        value -= float(law.cumulative_hazard(trace.crossing_distance))
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True)
    parser.add_argument("--raw", default="training_shared_S00_R0_C00_A0.npz")
    parser.add_argument("--output", default="evidence/scalar_prototype.json")
    parser.add_argument("--traces", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    common, adapter, baseline = setup(args.reference)
    inputs = load_role(common, "selection" if args.raw.startswith("selection") else "training")
    raw_path = Path(args.reference) / "raw" / args.raw
    archive, raw = restore_archive(raw_path, adapter)
    law, kernel, _ = adapter.d7.model_from_normalized(raw["point"], inputs["context"])
    traces = archive.traces[:args.traces]
    reference_density = adapter.d7.base.renewal_trace_log_density
    results = []
    gaps = np.concatenate([np.asarray(t.internal_gaps, dtype=float) for t in traces])
    reference_h = np.asarray([law.hazard(float(gap)) for gap in gaps])
    reference_H = np.asarray([law.cumulative_hazard(float(gap)) for gap in gaps])
    reference_log = np.asarray([reference_density(trace, law) for trace in traces])
    boundary_values = np.concatenate([
        np.asarray([np.nextafter(0., 1.), np.finfo(float).tiny, 1.e-15, 1.e-8]),
        law.scale * np.asarray(law.chart.breakpoints, dtype=float),
        np.asarray([np.nextafter(law.scale * law.horizon, 0.),
                    np.nextafter(law.scale * law.horizon, math.inf),
                    law.scale * (law.horizon + 1.), 100., 1536.]),
        np.asarray([float(t.age) for t in traces]),
        np.asarray([float(t.crossing_distance) for t in traces if t.crossing_distance is not None]),
    ])
    boundary_values = boundary_values[boundary_values > 0.]
    with np.errstate(all="ignore"):
        boundary_h = np.asarray([law.hazard(float(age)) for age in boundary_values])
        boundary_H = np.asarray([law.cumulative_hazard(float(age)) for age in boundary_values])
    for mode in ("vector", "one_element"):
        h, H = positive_scalar_gap_values(law, gaps, power_mode=mode)
        logs = np.asarray([renewal_trace_log_density_prototype(trace, law, power_mode=mode) for trace in traces])
        with np.errstate(all="ignore"):
            bh, bH = positive_scalar_gap_values(law, boundary_values, power_mode=mode)
        times = {"reference": [], "prototype": []}
        for _ in range(args.repeats):
            start = time.perf_counter()
            for trace in traces: reference_density(trace, law)
            times["reference"].append(time.perf_counter()-start)
            start = time.perf_counter()
            for trace in traces: renewal_trace_log_density_prototype(trace, law, power_mode=mode)
            times["prototype"].append(time.perf_counter()-start)
        results.append(dict(mode=mode,
                            checks={"saved_gap_h": array_check(h, reference_h),
                                    "saved_gap_H": array_check(H, reference_H),
                                    "trace_renewal_log_density": array_check(logs, reference_log),
                                    "boundary_h": array_check(bh, boundary_h),
                                    "boundary_H": array_check(bH, boundary_H)},
                            timings_seconds=times,
                            median_speedup=float(np.median(times["reference"])/np.median(times["prototype"]))))
    edge_checks = []
    def outcome(function, trace):
        try:
            with np.errstate(all="ignore"):
                value = function(trace, law)
            return {"value_bytes": np.asarray(value, dtype=np.float64).tobytes().hex()}
        except Exception as error:
            return {"exception": type(error).__name__, "message": str(error)}
    for label, edge_gaps in (("empty", ()), ("zero", (0.,)), ("negative_zero", (-0.,)),
                             ("negative", (-1.,)), ("nan", (math.nan,)),
                             ("infinity", (math.inf,)), ("short_circuit_zero_nan", (0., math.nan)),
                             ("short_circuit_zero_negative", (0., -1.)),
                             ("smallest_subnormal", (np.nextafter(0., 1.),))):
        trace = SimpleNamespace(age=traces[0].age, internal_gaps=edge_gaps,
                                crossing_distance=traces[0].crossing_distance)
        expected = outcome(reference_density, trace)
        actual = outcome(renewal_trace_log_density_prototype, trace)
        edge_checks.append(dict(fixture=label, expected=expected, actual=actual, exact=actual == expected))
    import scipy
    payload = dict(status="EXPERIMENTAL_NOT_CERTIFIED", source_sha256=sha256(__file__),
                   raw=args.raw, raw_sha256=sha256(raw_path), trace_count=len(traces),
                   gap_count=gaps.size, boundary_count=boundary_values.size,
                   point=raw["point"], results=results,
                   edge_checks=edge_checks,
                   runtime=dict(python=sys.version, numpy=np.__version__, scipy=scipy.__version__),
                   no_sampler=True, no_full_archive_evaluation=True,
                   rationale="One-element np.power and original one-row BLAS products, batched independent BSpline query points; untouched sequential math.log and scalar addition.")
    write_json(HERE / args.output, payload)
    print(json.dumps(ready(payload), indent=2))


if __name__ == "__main__":
    main()
