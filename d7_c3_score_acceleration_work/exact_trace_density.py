"""Batch independent spline queries, preserving scalar renewal-density arithmetic.

This deliberately does not vectorize the coefficient dot products, powers,
logarithms or the ordered sum over gaps. Those changes can alter binary64
rounding. Unsupported law/chart types and exceptional inputs use the original
implementation. No saved result or proposal density is consulted.
"""
from __future__ import annotations
import math
import warnings
import numpy as np


def _gap_values(law, gaps):
    base = gaps / law.scale
    body = (base > 0.) & (base <= law.horizon)
    hazard = np.empty(gaps.size, dtype=float)
    cumulative = np.empty(gaps.size, dtype=float)
    selected = np.flatnonzero(body)
    ratios = base[body] / law.horizon
    # Keep one-element ufunc calls. A larger np.power may dispatch to a
    # different SIMD implementation, even when algebraically identical.
    transformed = np.asarray([float((law.horizon * ratios[i:i+1] ** law.origin_exponent)[0])
                              for i in range(ratios.size)])
    warp = np.asarray([float((law.origin_exponent * ratios[i:i+1] ** (law.origin_exponent - 1.))[0])
                       for i in range(ratios.size)])
    # SciPy evaluates each query independently. Each following contraction
    # retains the original contiguous (1, coefficient_count) @ coefficients.
    basis = np.asarray(law.chart.basis(transformed), dtype=float)
    integrals = np.asarray(law.chart.integral_basis(transformed), dtype=float)
    for offset, index in enumerate(selected):
        coordinate = basis[offset:offset+1] @ law.coefficients
        hazard[index] = (coordinate * warp[offset:offset+1] / law.scale)[0]
        cumulative[index] = (integrals[offset:offset+1] @ law.coefficients)[0]
    for index in np.flatnonzero(~body):
        hazard[index] = law.hazard(float(gaps[index]))
        cumulative[index] = law.cumulative_hazard(float(gaps[index]))
    return hazard, cumulative


def trace_log_density(trace, law, kernel, base):
    """Same complete finite-root density as ``base.trace_log_density``.

The optimization is restricted to the existing fitted law and vector-spline
chart; generic optimizer/model interfaces and other model families are intact.
    """
    from fitted_renewal_quantiles import FittedRenewalLaw
    chart_type = type(law.chart) if isinstance(law, FittedRenewalLaw) else None
    supported = (type(law) is FittedRenewalLaw and chart_type.__name__ == 'BreakpointHazardChart'
                 and chart_type.__module__ == 'd7_blind_upstream_e79')
    # Under caller-defined floating-point traps, retain the reference call and
    # exception order instead of precomputing any later gap.
    if (not supported or any(mode not in ('ignore', 'warn') for mode in np.geterr().values())
            or any(rule[0] == 'error' for rule in warnings.filters)):
        return base.trace_log_density(trace, law, kernel)
    gaps = np.asarray(trace.internal_gaps, dtype=float)
    if gaps.ndim != 1 or not gaps.size or np.any(~np.isfinite(gaps)) or np.any(gaps <= 0.):
        return base.trace_log_density(trace, law, kernel)
    value = -float(law.cumulative_hazard(trace.age)) - math.log(law.mean)
    hazards, cumulative = _gap_values(law, gaps)
    # Exceptional values retain the original short-circuit and nonfinite
    # result behavior. This recomputes; it never substitutes archived outputs.
    if np.any(~np.isfinite(hazards)) or np.any(hazards <= 0.) or np.any(~np.isfinite(cumulative)):
        return base.trace_log_density(trace, law, kernel)
    for hazard, integral in zip(hazards, cumulative, strict=True):
        value += math.log(float(hazard)) - float(integral)
    if trace.crossing_distance is not None:
        value -= float(law.cumulative_hazard(trace.crossing_distance))
    return float(value + base.kernel_trace_log_density(trace, kernel))
