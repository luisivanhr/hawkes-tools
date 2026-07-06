"""RKHS-based univariate Hawkes kernel estimation."""

from __future__ import annotations

import math

import numpy as np
from scipy.integrate import cumulative_trapezoid, simpson
from scipy.interpolate import InterpolatedUnivariateSpline

from hawkes_tools.base import TimeFunction, normalize_events

from .inference import _LearnerBase
from .kernels import HawkesKernelTimeFunc


__all__ = ["RKHSHawkes"]


def _positive_float(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be positive")
    return value


def _nonnegative_float(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _positive_int(name: str, value: int, minimum: int = 1) -> int:
    value = int(value)
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _reproducing_kernel(x, y):
    """Return the W2[-1, 1] reproducing kernel from the thesis appendix."""

    x_arr, y_arr = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
    left = (
        7.0 / 3.0
        - y_arr**3 / 6.0
        + 0.5 * y_arr**2 * x_arr
        + 1.5 * y_arr
        + 2.0 * x_arr * y_arr
        + 1.5 * x_arr
    )
    right = (
        7.0 / 3.0
        - x_arr**3 / 6.0
        + 0.5 * x_arr**2 * y_arr
        + 1.5 * x_arr
        + 2.0 * x_arr * y_arr
        + 1.5 * y_arr
    )
    return np.where(y_arr <= x_arr, left, right)


def _tau_to_lag(tau, transform_scale: float):
    tau = np.asarray(tau, dtype=float)
    denom = 1.0 + tau
    out = np.full_like(tau, np.inf, dtype=float)
    mask = denom > 0.0
    out[mask] = transform_scale * (1.0 - tau[mask]) / denom[mask]
    return out


def _first_derivative_left(values: np.ndarray, spacing: float) -> np.ndarray:
    return (-3.0 * values[:, 0] + 4.0 * values[:, 1] - values[:, 2]) / (2.0 * spacing)


def _second_derivative(values: np.ndarray, domain: np.ndarray) -> np.ndarray:
    first = np.gradient(values, domain, axis=1, edge_order=2)
    return np.gradient(first, domain, axis=1, edge_order=2)


def _inner_product_matrix(functions: np.ndarray, domain: np.ndarray) -> np.ndarray:
    spacing = float(domain[1] - domain[0])
    left_values = functions[:, 0]
    left_derivatives = _first_derivative_left(functions, spacing)
    second = _second_derivative(functions, domain)
    curvature = simpson(second[:, None, :] * second[None, :, :], x=domain, axis=-1)
    return (
        left_values[:, None] * left_values[None, :]
        + left_derivatives[:, None] * left_derivatives[None, :]
        + curvature
    )


class RKHSHawkes(_LearnerBase):
    """Univariate non-parametric Hawkes kernel estimator using RKHS inversion.

    The estimator follows the thesis/notebook pipeline:

    1. estimate the normalized autocovariance ``Phi(t) = phi(t) / mean_intensity``;
    2. solve the Wiener-Hopf equation ``Phi = f + Phi * f`` in the RKHS
       ``W2[-1, 1]`` after the map ``t = gamma * (1 - tau) / (1 + tau)``;
    3. expose the fitted Hawkes kernel ``f = alpha h`` through the standard
       Hawkes learner accessors.

    The bandwidth is an explicit bin width. No optimal-bandwidth selection is
    performed.
    """

    def __init__(
        self,
        kernel_support: float,
        bandwidth: float,
        covariance_spline_order: int = 1,
        rkhs_grid_size: int = 401,
        rkhs_basis_size: int = 21,
        quadrature_size: int = 200,
        transform_scale: float | None = None,
        ridge: float = 1e-8,
        clip_negative: bool = True,
        verbose: bool = False,
        n_threads: int = 1,
    ):
        super().__init__(
            tol=0.0,
            max_iter=0,
            verbose=verbose,
            print_every=1,
            record_every=1,
            n_threads=n_threads,
        )
        self.kernel_support = _positive_float("kernel_support", kernel_support)
        self.bandwidth = _positive_float("bandwidth", bandwidth)
        if self.bandwidth >= self.kernel_support:
            raise ValueError("bandwidth must be smaller than kernel_support")
        self.covariance_spline_order = int(covariance_spline_order)
        if self.covariance_spline_order < 0 or self.covariance_spline_order > 5:
            raise ValueError("covariance_spline_order must be between 0 and 5")
        self.rkhs_grid_size = _positive_int("rkhs_grid_size", rkhs_grid_size, minimum=5)
        self.rkhs_basis_size = _positive_int("rkhs_basis_size", rkhs_basis_size)
        self.quadrature_size = _positive_int("quadrature_size", quadrature_size, minimum=2)
        self.transform_scale = (
            0.7 * self.kernel_support
            if transform_scale is None
            else _positive_float("transform_scale", transform_scale)
        )
        self.ridge = _nonnegative_float("ridge", ridge)
        self.clip_negative = bool(clip_negative)

        self.mean_intensity = None
        self.autocorrelation_lags = None
        self.autocorrelation = None
        self.kernel_lags = None
        self.kernel_values = None
        self.kernel_time_function = None
        self.kernels = None
        self.kernel = None
        self.baseline = None
        self.adjacency = None
        self.branching_ratio = None
        self._phi_x = None
        self._phi_y = None
        self._phi_spline = None
        self._kernel_primitive = None
        self._rkhs_domain = None
        self._rkhs_basis_points = None
        self._rkhs_basis_functions = None
        self._rkhs_gram = None
        self._rkhs_coeffs = None

    @property
    def kernel_discretization(self):
        if self.kernel_lags is None:
            raise ValueError("fit must be called first")
        return self.kernel_lags.copy()

    def fit(self, events, end_times=None):
        data, normalized_end_times, n_nodes = normalize_events(events, end_times)
        if n_nodes != 1:
            raise ValueError("RKHSHawkes currently supports univariate event data only")
        self.data = data
        self._end_times = normalized_end_times
        self._n_nodes = int(n_nodes)
        self._fitted = True

        total_events = int(sum(realization[0].size for realization in self.data))
        total_time = float(np.sum(self._end_times))
        if total_events < 2:
            raise ValueError("at least two events are required")
        if total_time <= 0.0:
            raise ValueError("total observation time must be positive")
        self.mean_intensity = np.asarray([total_events / total_time], dtype=float)

        self.autocorrelation_lags, self.autocorrelation = self._estimate_autocorrelation()
        self._build_phi_interpolator()
        self.kernel_lags, self.kernel_values = self._solve_rkhs()
        if self.clip_negative:
            self.kernel_values = np.maximum(self.kernel_values, 0.0)
        self._build_kernel_time_function()
        self._kernel_primitive = self._primitive_grid(self.kernel_lags, self.kernel_values)
        self.branching_ratio = float(self.kernels[0, 0].get_norm())
        self.adjacency = np.asarray([[self.branching_ratio]], dtype=float)
        self.baseline = np.asarray(
            [self.mean_intensity[0] * max(1.0 - self.branching_ratio, 0.0)],
            dtype=float,
        )
        self.kernel = self.kernel_values.reshape(1, 1, -1)
        return self

    def _build_kernel_time_function(self) -> None:
        lags = np.asarray(self.kernel_lags, dtype=float)
        values = np.asarray(self.kernel_values, dtype=float)
        if lags.size == 0:
            raise RuntimeError("RKHS solve returned an empty kernel grid")
        if lags[0] > 0.0:
            lags = np.concatenate(([0.0], lags))
            values = np.concatenate(([values[0]], values))
        else:
            lags = lags.copy()
            values = values.copy()
            lags[0] = 0.0
        if self.kernel_support - lags[-1] > 1e-12:
            lags = np.concatenate((lags, [self.kernel_support]))
            values = np.concatenate((values, [0.0]))
        else:
            lags[-1] = self.kernel_support
            values[-1] = 0.0

        self.kernel_lags = lags
        self.kernel_values = values
        self.kernel_time_function = TimeFunction(
            (self.kernel_lags, self.kernel_values),
            border_type=TimeFunction.Border0,
            inter_mode=TimeFunction.InterLinear,
        )
        self.kernels = np.asarray(
            [[HawkesKernelTimeFunc(time_function=self.kernel_time_function)]],
            dtype=object,
        )

    def _estimate_autocorrelation(self):
        edges = np.arange(0.0, self.kernel_support + self.bandwidth, self.bandwidth)
        if edges[-1] > self.kernel_support:
            edges[-1] = self.kernel_support
        if edges.size < 3:
            raise ValueError("kernel_support and bandwidth must define at least two bins")
        counts = np.zeros(edges.size - 1, dtype=float)
        exposure = np.zeros_like(counts)
        for realization, end_time in zip(self.data, self._end_times):
            timestamps = realization[0]
            counts += self._lag_counts(timestamps, edges)
            exposure += self._lag_exposure(timestamps, float(end_time), edges)
        density = np.divide(counts, exposure, out=np.zeros_like(counts), where=exposure > 0.0)
        normalized_covariance = density - float(self.mean_intensity[0])
        return 0.5 * (edges[:-1] + edges[1:]), normalized_covariance

    def _lag_counts(self, timestamps: np.ndarray, edges: np.ndarray) -> np.ndarray:
        counts = np.zeros(edges.size - 1, dtype=float)
        if timestamps.size < 2:
            return counts
        for i, start in enumerate(timestamps[:-1]):
            stop = np.searchsorted(timestamps, start + self.kernel_support, side="right")
            if stop <= i + 1:
                continue
            lags = timestamps[i + 1 : stop] - start
            counts += np.histogram(lags, bins=edges)[0]
        return counts

    def _lag_exposure(self, timestamps: np.ndarray, end_time: float, edges: np.ndarray) -> np.ndarray:
        remaining = np.asarray(end_time - timestamps, dtype=float)
        remaining = np.sort(remaining[remaining > 0.0])
        exposure = np.zeros(edges.size - 1, dtype=float)
        if remaining.size == 0:
            return exposure
        prefix = np.concatenate(([0.0], np.cumsum(remaining)))
        n_remaining = remaining.size
        for k, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
            idx_left = np.searchsorted(remaining, left, side="right")
            idx_right = np.searchsorted(remaining, right, side="left")
            middle_sum = prefix[idx_right] - prefix[idx_left]
            middle_count = idx_right - idx_left
            full_count = n_remaining - idx_right
            exposure[k] = (
                middle_sum
                - middle_count * left
                + full_count * (right - left)
            )
        return exposure

    def _build_phi_interpolator(self) -> None:
        x = np.concatenate(([0.0], self.autocorrelation_lags))
        y = np.concatenate(([self.autocorrelation[0]], self.autocorrelation))
        self._phi_x = x
        self._phi_y = y
        order = min(self.covariance_spline_order, x.size - 1)
        if order <= 0:
            self._phi_spline = None
        else:
            self._phi_spline = InterpolatedUnivariateSpline(x, y, k=order, ext=1)

    def _phi(self, lag) -> np.ndarray:
        lag_arr = np.asarray(lag, dtype=float)
        out = np.zeros_like(lag_arr, dtype=float)
        finite = np.isfinite(lag_arr)
        if not np.any(finite):
            return out
        query = np.abs(lag_arr[finite])
        if self._phi_spline is None:
            values = np.interp(query, self._phi_x, self._phi_y, left=self._phi_y[0], right=0.0)
        else:
            values = np.asarray(self._phi_spline(query), dtype=float)
            values[query > self.kernel_support] = 0.0
        out[finite] = values
        return out

    def _solve_rkhs(self):
        domain = np.linspace(-1.0, 1.0, self.rkhs_grid_size)
        basis_points = np.cos(np.pi * np.arange(self.rkhs_basis_size) / self.rkhs_basis_size)
        quad_nodes, quad_weights = np.polynomial.legendre.leggauss(self.quadrature_size)
        transformed_basis = _tau_to_lag(basis_points, self.transform_scale)
        transformed_quad = _tau_to_lag(quad_nodes, self.transform_scale)
        kernel_grid_quad = _reproducing_kernel(domain[:, None], quad_nodes[None, :])

        psi = np.empty((self.rkhs_basis_size, self.rkhs_grid_size), dtype=float)
        for i, (xi, t_xi) in enumerate(zip(basis_points, transformed_basis)):
            kernel_values = self._phi(t_xi - transformed_quad)
            weighted_kernel = quad_weights * kernel_values
            psi[i] = (
                (xi + 1.0) ** 2 * _reproducing_kernel(xi, domain)
                + 2.0 * self.transform_scale * (kernel_grid_quad @ weighted_kernel)
            )

        gram = _inner_product_matrix(psi, domain)
        gram = 0.5 * (gram + gram.T)
        rhs = self._phi(transformed_basis)
        system = gram + self.ridge * np.eye(gram.shape[0])
        try:
            coeffs = np.linalg.solve(system, rhs)
        except np.linalg.LinAlgError:
            coeffs = np.linalg.lstsq(system, rhs, rcond=None)[0]
        x_solution = coeffs @ psi
        y_solution = (domain + 1.0) ** 2 * x_solution
        lags = _tau_to_lag(domain, self.transform_scale)
        mask = np.isfinite(lags) & (lags >= 0.0) & (lags <= self.kernel_support)
        order = np.argsort(lags[mask])

        self._rkhs_domain = domain
        self._rkhs_basis_points = basis_points
        self._rkhs_basis_functions = psi.copy()
        self._rkhs_gram = gram
        self._rkhs_coeffs = coeffs
        return lags[mask][order], y_solution[mask][order]

    def recover_basis_functions(self, max_index: int, domain: str = "lag"):
        """Return the first RKHS basis functions up to ``max_index``.

        ``max_index`` is zero-based and inclusive. With ``domain="rkhs"``, the
        method returns the raw basis functions on the RKHS coordinate
        ``[-1, 1]``. With ``domain="lag"``, it applies the recovery
        ``(tau + 1)^2`` and maps the functions back to the Hawkes lag axis.
        """

        if self._rkhs_basis_functions is None or self._rkhs_domain is None:
            raise ValueError("fit must be called first")
        max_index = int(max_index)
        if max_index < 0:
            raise ValueError("max_index must be non-negative")
        if max_index >= self._rkhs_basis_functions.shape[0]:
            raise ValueError(
                f"max_index must be smaller than rkhs_basis_size={self._rkhs_basis_functions.shape[0]}"
            )

        values = self._rkhs_basis_functions[: max_index + 1].copy()
        domain_key = str(domain).lower()
        if domain_key == "rkhs":
            return self._rkhs_domain.copy(), values
        if domain_key != "lag":
            raise ValueError("domain must be 'lag' or 'rkhs'")

        lags = _tau_to_lag(self._rkhs_domain, self.transform_scale)
        mask = np.isfinite(lags) & (lags >= 0.0) & (lags <= self.kernel_support)
        order = np.argsort(lags[mask])
        recovered = ((self._rkhs_domain + 1.0) ** 2 * values)[:, mask]
        return lags[mask][order], recovered[:, order]

    def plot_basis_functions(
        self,
        max_index: int,
        domain: str = "lag",
        show: bool = True,
        ax=None,
        layout: str = "overlay",
    ):
        """Plot the first RKHS basis functions up to ``max_index``."""

        abscissa, values = self.recover_basis_functions(max_index, domain=domain)
        import matplotlib.pyplot as plt

        domain_key = str(domain).lower()
        x_label = "lag" if domain_key == "lag" else "rkhs coordinate"
        layout_key = str(layout).lower()
        if layout_key not in {"overlay", "grid"}:
            raise ValueError("layout must be 'overlay' or 'grid'")

        if layout_key == "grid":
            if ax is None:
                n_cols = min(2, values.shape[0])
                n_rows = int(math.ceil(values.shape[0] / n_cols))
                fig, axes = plt.subplots(
                    n_rows,
                    n_cols,
                    figsize=(4.0 * n_cols, 2.8 * n_rows),
                    sharex=True,
                )
            else:
                axes = np.asarray(ax, dtype=object)
                fig = axes.ravel()[0].figure
                show = False
            axes_flat = np.asarray(axes, dtype=object).ravel()
            if axes_flat.size < values.shape[0]:
                raise ValueError("ax must contain at least one axis per basis function")
            for index, basis_values in enumerate(values):
                axis = axes_flat[index]
                axis.plot(abscissa, basis_values)
                axis.set_title(f"basis {index}")
                axis.set_xlabel(x_label)
                axis.set_ylabel("basis value")
            for axis in axes_flat[values.shape[0] :]:
                axis.set_visible(False)
        else:
            if ax is None:
                fig, ax = plt.subplots(figsize=(5.0, 3.5))
            else:
                fig = ax.figure
                show = False
            for index, basis_values in enumerate(values):
                ax.plot(abscissa, basis_values, label=f"basis {index}")
            ax.set_xlabel(x_label)
            ax.set_ylabel("basis value")
            ax.legend(loc="best", fontsize=8)

        fig.tight_layout()
        if show:
            plt.show()
        return fig

    def _primitive_grid(self, lags: np.ndarray, values: np.ndarray) -> np.ndarray:
        primitive = np.zeros_like(values, dtype=float)
        if values.size > 1:
            primitive[1:] = cumulative_trapezoid(values, lags)
        return primitive

    def _primitive_values(self, x):
        x_arr = np.asarray(x, dtype=float)
        return np.interp(
            np.clip(x_arr, 0.0, self.kernel_support),
            self.kernel_lags,
            self._kernel_primitive,
            left=0.0,
            right=self._kernel_primitive[-1],
        )

    def get_kernel_supports(self):
        if not self._fitted:
            raise ValueError("fit must be called first")
        return np.asarray([[self.kernels[0, 0].get_support()]], dtype=float)

    def get_kernel_values(self, i, j, abscissa_array):
        if not self._fitted:
            raise ValueError("fit must be called first")
        if i != 0 or j != 0:
            raise IndexError("RKHSHawkes is univariate")
        return self.kernels[0, 0].get_values(abscissa_array)

    def _compute_primitive_kernel_values(self, i, j, abscissa_array):
        if i != 0 or j != 0:
            raise IndexError("RKHSHawkes is univariate")
        return self.kernels[0, 0].get_primitive_values(abscissa_array)

    def get_kernel_norms(self):
        if not self._fitted:
            raise ValueError("fit must be called first")
        return np.asarray([[self.kernels[0, 0].get_norm()]], dtype=float)

    def score(self, events=None, end_times=None, baseline=None, kernel_values=None, kernel_lags=None):
        if events is None and not self._fitted:
            raise ValueError("You must either call `fit` before `score` or provide events")
        if events is None and end_times is not None:
            raise ValueError("events must be provided when end_times is provided")
        if events is None:
            data = self.data
            end_times_arr = self._end_times
        else:
            data, end_times_arr, n_nodes = normalize_events(events, end_times)
            if n_nodes != 1:
                raise ValueError("RKHSHawkes scoring supports univariate event data only")

        baseline_value = float(self.baseline[0] if baseline is None else np.asarray(baseline, dtype=float).reshape(-1)[0])
        lags = self.kernel_lags if kernel_lags is None else np.asarray(kernel_lags, dtype=float)
        values = self.kernel_values if kernel_values is None else np.asarray(kernel_values, dtype=float)
        primitive = self._primitive_grid(lags, values)

        def kernel_at(x):
            return np.interp(x, lags, values, left=0.0, right=0.0)

        def primitive_at(x):
            return np.interp(np.clip(x, 0.0, lags[-1]), lags, primitive, left=0.0, right=primitive[-1])

        value = 0.0
        n_jumps = 0
        for realization, end_time in zip(data, end_times_arr):
            timestamps = realization[0]
            value += float(end_time)
            compensator = baseline_value * float(end_time)
            if timestamps.size:
                compensator += float(np.sum(primitive_at(float(end_time) - timestamps)))
            value -= compensator
            for idx, timestamp in enumerate(timestamps):
                n_jumps += 1
                intensity = baseline_value
                if idx:
                    intensity += float(np.sum(kernel_at(timestamp - timestamps[:idx])))
                if intensity <= 0.0:
                    return -np.inf
                value += math.log(intensity)
        return float(value / max(n_jumps, 1))
