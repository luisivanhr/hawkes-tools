"""Plotting helpers for point processes and Hawkes learners."""

from __future__ import annotations

import numpy as np

from hawkes_tools.base import TimeFunction

__all__ = [
    "plot_basis_kernels",
    "plot_estimated_intensity",
    "plot_hawkes_baseline_and_kernels",
    "plot_hawkes_kernel_norms",
    "plot_hawkes_kernels",
    "plot_history",
    "plot_point_process",
    "plot_timefunction",
    "qq_plots",
    "stems",
]


def _plt():
    import matplotlib.pyplot as plt

    return plt


def _stem_matplotlib(y, axis, title, x_range, y_range):
    axis.stem(np.asarray(y))
    if x_range is not None:
        axis.set_xlim(x_range)
    if y_range is not None:
        axis.set_ylim(y_range)
    if title is not None:
        axis.set_title(title, fontsize=16)
    return axis


def _stems_matplotlib(ys, titles, x_range, y_range, fig_size):
    plt = _plt()
    n_stems = len(ys)
    fig = plt.figure(figsize=(fig_size[0], fig_size[1] * n_stems))
    for idx, (y, title) in enumerate(zip(ys, titles)):
        axis = fig.add_subplot(n_stems, 1, idx + 1)
        _stem_matplotlib(y, axis, title, x_range, y_range)
    return fig


def _stem_bokeh(y, title, x_range, y_range, fig_size):
    from bokeh.plotting import figure

    y = np.asarray(y)
    dim = y.shape[0]
    x = np.arange(dim)
    plot_width, plot_height = fig_size
    try:
        fig = figure(
            width=plot_width,
            height=plot_height,
            x_range=x_range,
            y_range=y_range,
        )
    except AttributeError:  # pragma: no cover - older bokeh compatibility
        fig = figure(
            plot_width=plot_width,
            plot_height=plot_height,
            x_range=x_range,
            y_range=y_range,
        )
    fig.scatter(x, y, size=4, fill_alpha=0.5)
    fig.segment(x, np.zeros(dim), x, y)
    fig.title.text_font_size = "12pt"
    if title is not None:
        fig.title.text = title
    return fig


def _stems_bokeh(ys, titles, sync_axes, fig_size):
    from bokeh.plotting import gridplot

    figs = []
    x_range = None
    y_range = None
    for idx, (y, title) in enumerate(zip(ys, titles)):
        fig = _stem_bokeh(y, title=title, x_range=x_range, y_range=y_range, fig_size=fig_size)
        figs.append(fig)
        if idx == 0 and sync_axes:
            x_range = fig.x_range
            y_range = fig.y_range
    return gridplot([[fig] for fig in figs])


def stems(ys: list, titles: list | None = None, sync_axes: bool = True, rendering: str = "matplotlib", fig_size: tuple | None = None):
    """Plot several stem plots using matplotlib or bokeh rendering."""

    ys = [np.asarray(y) for y in ys]
    if titles is not None:
        if len(ys) != len(titles):
            raise ValueError("length of ``titles`` differs from the length of ``ys``")
    else:
        titles = len(ys) * [None]

    if rendering == "matplotlib":
        if fig_size is None:
            fig_size = (8, 2.5)
        if sync_axes and ys:
            x_range = (0, max(y.shape[0] for y in ys))
            y_min = min(float(y.min()) for y in ys)
            y_max = max(float(y.max()) for y in ys)
            y_range = (y_min * (1 - 5e-2), y_max * (1 + 5e-2))
        else:
            x_range = None
            y_range = None
        fig = _stems_matplotlib(ys=ys, titles=titles, x_range=x_range, y_range=y_range, fig_size=fig_size)
        fig.tight_layout()
        return fig
    if rendering == "bokeh":
        from bokeh.plotting import show as bk_show

        if fig_size is None:
            fig_size = (600, 200)
        fig = _stems_bokeh(ys=ys, titles=titles, sync_axes=sync_axes, fig_size=fig_size)
        bk_show(fig)
        return fig
    raise ValueError(f"Unknown rendering type. Expected 'matplotlib' or 'bokeh', got '{rendering}'")


def _history_for(obj):
    solver = getattr(obj, "_solver_obj", None)
    if solver is not None:
        return getattr(solver, "history", None)
    return getattr(obj, "history", None)


def _history_label_for(obj):
    solver = getattr(obj, "_solver_obj", None)
    label_obj = solver if solver is not None else obj
    return getattr(label_obj, "name", label_obj.__class__.__name__)


def plot_history(
    solvers,
    x: str = "n_iter",
    y: str = "obj",
    labels=None,
    show: bool = True,
    log_scale: bool = False,
    dist_min: bool = False,
    ax=None,
    **kwargs,
):
    """Plot tick-style optimization histories."""

    del kwargs
    plt = _plt()
    if not isinstance(solvers, (list, tuple)):
        solvers = [solvers]
    if labels is None:
        labels = [_history_label_for(solver) for solver in solvers]
    if len(labels) != len(solvers):
        raise ValueError("labels must have the same length as solvers")
    if ax is None:
        _, ax = plt.subplots(figsize=(6.6, 3.8))
    else:
        show = False

    series = []
    for solver, label in zip(solvers, labels):
        history = _history_for(solver)
        if history is None:
            raise ValueError(f"{solver.__class__.__name__} has no history")
        values = history.values
        if x not in values:
            raise ValueError(f"{label} has no history for {x}")
        if y not in values:
            raise ValueError(f"{label} has no history for {y}")
        x_values = np.asarray(values[x], dtype=float)
        y_values = np.asarray(values[y], dtype=float)
        if x_values.size != y_values.size:
            raise ValueError(f"{label} history arrays for {x} and {y} have inconsistent lengths")
        if x_values.size:
            series.append((label, x_values, y_values))

    minimum = None
    if dist_min and series:
        history_minima = []
        for _, _, y_values in series:
            finite = y_values[np.isfinite(y_values)]
            if finite.size:
                history_minima.append(float(np.min(finite)))
        minimum = min(history_minima) if history_minima else 0.0

    plotted_any = False
    for label, x_values, y_values in series:
        if dist_min:
            y_values = y_values - float(minimum)
        if log_scale:
            y_values = np.maximum(y_values, 1e-18)
        ax.plot(x_values, y_values, label=str(label))
        plotted_any = True

    ax.set_xlabel(x)
    ax.set_ylabel(f"{y} - min" if dist_min else y)
    if log_scale:
        ax.set_yscale("log")
    if plotted_any:
        ax.legend(loc="best")
    ax.figure.tight_layout()
    if show:
        plt.show()
    return ax.figure


def _as_axes(ax, shape, plt, figsize=None, sharex=False, sharey=False):
    if ax is None:
        _, axes = plt.subplots(*shape, squeeze=False, figsize=figsize, sharex=sharex, sharey=sharey)
        return axes, True
    axes = np.asarray(ax, dtype=object)
    if axes.shape == ():
        axes = axes.reshape((1, 1))
    elif axes.ndim == 1 and shape[1] == 1:
        axes = axes.reshape((shape[0], 1))
    elif axes.ndim == 1 and shape[0] == 1:
        axes = axes.reshape((1, shape[1]))
    if axes.shape != shape:
        raise ValueError(f"ax has shape {axes.shape}, expected {shape}")
    return axes, False


def _node_names(n_nodes, plot_nodes, node_names):
    if node_names is None:
        return [f"ticks #{node}" for node in plot_nodes]
    if len(node_names) != len(plot_nodes):
        raise ValueError(f"node_names must have length {len(plot_nodes)}, got {len(node_names)}")
    return list(node_names)


def _simulation_end_time(point_process):
    end_time = getattr(point_process, "end_time", None)
    if isinstance(end_time, (list, tuple, np.ndarray)):
        return float(np.max(end_time))
    if end_time is not None:
        return float(end_time)
    return float(getattr(point_process, "simulation_time", 0.0))


def _hawkes_n_nodes(obj):
    n_nodes = getattr(obj, "n_nodes", None)
    if isinstance(n_nodes, (list, tuple, np.ndarray)):
        return int(n_nodes[0])
    if n_nodes is not None:
        return int(n_nodes)
    return int(np.asarray(obj.kernels, dtype=object).shape[0])


def _kernel_supports(obj):
    if hasattr(obj, "get_kernel_supports"):
        return np.asarray(obj.get_kernel_supports(), dtype=float)
    kernels = np.asarray(obj.kernels, dtype=object)
    return np.vectorize(lambda kernel: kernel.get_plot_support(), otypes=[float])(kernels)


def _kernel_norms(obj):
    if hasattr(obj, "get_kernel_norms"):
        return np.asarray(obj.get_kernel_norms(), dtype=float)
    kernels = np.asarray(obj.kernels, dtype=object)
    return np.vectorize(lambda kernel: kernel.get_norm(), otypes=[float])(kernels)


def _kernel_values(obj, i, j, x):
    if hasattr(obj, "get_kernel_values"):
        return obj.get_kernel_values(i, j, x)
    return obj.kernels[i, j].get_values(x)


def _interval_mask(values, t_min, t_max):
    values = np.asarray(values, dtype=float)
    return (values >= t_min) & (values <= t_max)


def _extract_process_interval(
    plot_nodes,
    end_time,
    timestamps,
    intensities=None,
    intensity_times=None,
    t_min=None,
    t_max=None,
    max_jumps=None,
):
    """Extract a tick-style plotting interval from point-process arrays."""

    t_min_is_specified = t_min is not None
    if not t_min_is_specified:
        t_min = 0.0
    t_max_is_specified = t_max is not None
    if not t_max_is_specified:
        t_max = float(end_time)

    t_min = float(t_min)
    t_max = float(t_max)
    end_time = float(end_time)
    if t_min >= end_time:
        raise ValueError("`t_min` should be smaller than `end_time`")
    if t_max <= 0:
        raise ValueError("`t_max` should be positive")

    plot_nodes = list(plot_nodes)
    timestamps = [np.asarray(values, dtype=float) for values in timestamps]
    if max_jumps is not None:
        max_jumps = int(max_jumps)
        if max_jumps < 0:
            raise ValueError("max_jumps must be non-negative")
        if t_min_is_specified or not t_max_is_specified:
            for i in plot_nodes:
                timestamps_i = timestamps[i]
                i_t_min = np.searchsorted(timestamps_i, t_min, side="left")
                last_index = i_t_min + max_jumps - 1
                if last_index < 0:
                    t_max = 0.0
                elif last_index < len(timestamps_i) and timestamps_i[last_index] < t_max:
                    t_max = float(timestamps_i[last_index])
        elif t_max_is_specified:
            for i in plot_nodes:
                timestamps_i = timestamps[i]
                i_t_max = np.searchsorted(timestamps_i, t_max, side="left")
                first_index = i_t_max - max_jumps
                if first_index >= len(timestamps_i) - 1:
                    t_min = end_time
                elif first_index >= 0 and timestamps_i[first_index] > t_min:
                    t_min = float(timestamps_i[first_index])

    extracted_timestamps = [values[_interval_mask(values, t_min, t_max)] for values in timestamps]

    if intensity_times is None:
        return extracted_timestamps, None, None

    intensity_times = np.asarray(intensity_times, dtype=float)
    intensity_extracted_points = _interval_mask(intensity_times, t_min, t_max)
    extracted_intensity_times = intensity_times[intensity_extracted_points]
    extracted_intensities = [
        np.asarray(intensity, dtype=float)[intensity_extracted_points]
        for intensity in intensities
    ]
    return extracted_timestamps, extracted_intensity_times, extracted_intensities


def plot_point_process(
    point_process,
    plot_intensity=None,
    n_points: int = 10000,
    plot_nodes=None,
    node_names=None,
    t_min: float | None = None,
    t_max: float | None = None,
    max_jumps: int | None = None,
    show: bool = True,
    ax=None,
):
    """Plot point-process timestamps and, optionally, tracked intensities."""

    plt = _plt()
    n_nodes = int(getattr(point_process, "n_nodes", len(point_process.timestamps)))
    nodes = list(range(n_nodes)) if plot_nodes is None else list(plot_nodes)
    labels = _node_names(n_nodes, nodes, node_names)

    end_time = _simulation_end_time(point_process)
    t_min = 0.0 if t_min is None else float(t_min)
    t_max = end_time if t_max is None else float(t_max)
    if t_min >= end_time and end_time > 0:
        raise ValueError("t_min must be smaller than the process end time")
    if t_max <= 0:
        raise ValueError("t_max must be positive")
    if t_max <= t_min:
        raise ValueError("t_max must be greater than t_min")

    if plot_intensity is None:
        plot_intensity = bool(point_process.is_intensity_tracked())

    axes, created = _as_axes(
        ax,
        (len(nodes), 1),
        plt,
        figsize=(10, max(2.5, 2.4 * len(nodes))),
        sharex=True,
        sharey=False,
    )
    if not created:
        show = False

    timestamps = point_process.timestamps
    if plot_intensity and not point_process.is_intensity_tracked():
        step = (t_max - t_min) / max(int(n_points), 1)
        point_process.track_intensity(step)
        point_process.set_timestamps(timestamps, end_time=t_max)

    intensity_times = None
    intensities = None
    if plot_intensity:
        intensity_times = np.asarray(point_process.intensity_tracked_times, dtype=float)
        intensities = [np.asarray(values, dtype=float) for values in point_process.tracked_intensity]

    timestamps, intensity_times, intensities = _extract_process_interval(
        nodes,
        t_max,
        timestamps,
        intensities=intensities,
        intensity_times=intensity_times,
        t_min=t_min,
        t_max=t_max,
        max_jumps=max_jumps,
    )

    for row, (node, label) in enumerate(zip(nodes, labels)):
        axis = axes[row, 0]
        ts = np.asarray(timestamps[node], dtype=float)

        if plot_intensity:
            intensity = np.asarray(intensities[node], dtype=float)
            if intensity_times.size:
                x = np.linspace(float(intensity_times[0]), float(intensity_times[-1]), int(n_points))
                y = np.interp(x, intensity_times, intensity)
                axis.plot(x, y, label="intensity")
                if ts.size:
                    axis.scatter(ts, np.interp(ts, intensity_times, intensity), s=18, label="jumps")
            axis.set_ylabel(label)
        else:
            axis.vlines(ts, 0.0, 1.0, linewidth=0.9)
            axis.set_yticks([])
            axis.set_ylabel(label)
        axis.set_xlim(t_min, t_max)
        if plot_intensity:
            axis.legend(loc="best")

    axes[-1, 0].set_xlabel("time")
    if show:
        plt.show()
    return axes[0, 0].figure


def plot_hawkes_kernels(
    kernel_object,
    support=None,
    hawkes=None,
    n_points: int = 300,
    show: bool = True,
    log_scale: bool = False,
    min_support: float = 1e-4,
    ax=None,
):
    """Plot every entry of a Hawkes kernel matrix."""

    plt = _plt()
    supports = _kernel_supports(kernel_object)
    n_nodes = _hawkes_n_nodes(kernel_object)
    if support is None or support <= 0:
        finite_supports = supports[np.isfinite(supports) & (supports > 0)]
        support = float(np.max(finite_supports)) if finite_supports.size else 1.0
        support *= 1.2

    axes, created = _as_axes(
        ax,
        (n_nodes, n_nodes),
        plt,
        figsize=(3.2 * n_nodes, 2.7 * n_nodes),
        sharex=True,
        sharey=True,
    )
    if not created:
        show = False

    if log_scale:
        x = np.logspace(np.log10(min_support), np.log10(support), int(n_points))
    else:
        x = np.linspace(0.0, float(support), int(n_points))

    for i in range(n_nodes):
        for j in range(n_nodes):
            axis = axes[i, j]
            axis.plot(x, _kernel_values(kernel_object, i, j, x), label=f"kernel ({i}, {j})")
            if hawkes is not None:
                axis.plot(x, hawkes.kernels[i, j].get_values(x), linestyle="--", label=f"true ({i}, {j})")
            if i == n_nodes - 1:
                axis.set_xlabel("time")
            axis.set_ylabel(f"phi[{i},{j}]")
            if log_scale:
                axis.set_xscale("log")
                axis.set_yscale("log")
            axis.legend(loc="best", fontsize=8)

    if show:
        plt.show()
    return axes[0, 0].figure


def plot_hawkes_kernel_norms(
    kernel_object,
    show: bool = True,
    pcolor_kwargs=None,
    node_names=None,
    rotate_x_labels: float = 0.0,
    ax=None,
):
    """Plot the matrix of Hawkes kernel norms."""

    plt = _plt()
    norms = _kernel_norms(kernel_object)
    n_nodes = norms.shape[0]
    labels = list(range(n_nodes)) if node_names is None else list(node_names)
    if len(labels) != n_nodes:
        raise ValueError(f"node_names must have length {n_nodes}, got {len(labels)}")
    if pcolor_kwargs is None:
        pcolor_kwargs = {}
    if norms.size and np.nanmin(norms) < 0:
        vmax = float(np.nanmax(np.abs(norms)))
        pcolor_kwargs.setdefault("cmap", "RdBu")
        pcolor_kwargs.setdefault("vmin", -vmax)
        pcolor_kwargs.setdefault("vmax", vmax)
    else:
        pcolor_kwargs.setdefault("cmap", "Blues")

    if ax is None:
        _, ax = plt.subplots(figsize=(4.8, 4.2))
    else:
        show = False
    image = ax.imshow(norms, origin="upper", **pcolor_kwargs)
    ax.set_xticks(np.arange(n_nodes), [f"{label}" for label in labels], rotation=-rotate_x_labels)
    ax.set_yticks(np.arange(n_nodes), [f"{label}" for label in labels])
    ax.xaxis.tick_top()
    ax.set_xlabel("source")
    ax.set_ylabel("target")
    ax.figure.colorbar(image, ax=ax)
    if show:
        plt.show()
    return ax.figure


def _baseline_values(hawkes_object, node, t_values):
    if hasattr(hawkes_object, "get_baseline_values"):
        return hawkes_object.get_baseline_values(node, t_values)
    baseline = np.asarray(hawkes_object.baseline, dtype=float)
    if baseline.ndim == 1:
        return np.full_like(t_values, baseline[node], dtype=float)
    period_length = getattr(hawkes_object, "period_length", None)
    if period_length is None:
        raise ValueError("period_length is required for piecewise baseline arrays")
    idx = np.floor(((t_values % period_length) / period_length) * baseline.shape[1]).astype(int)
    idx = np.minimum(idx, baseline.shape[1] - 1)
    return baseline[node, idx]


def plot_hawkes_baseline_and_kernels(
    hawkes_object,
    kernel_support=None,
    hawkes=None,
    n_points: int = 300,
    show: bool = True,
    log_scale: bool = False,
    min_support: float = 1e-4,
    ax=None,
):
    """Plot Hawkes baselines in the first column and kernels beside them."""

    plt = _plt()
    n_nodes = int(hawkes_object.n_nodes)
    axes, created = _as_axes(
        ax,
        (n_nodes, n_nodes + 1),
        plt,
        figsize=(3.0 * (n_nodes + 1), 2.7 * n_nodes),
        sharex=False,
        sharey=False,
    )
    if not created:
        show = False

    kernel_axes = axes[:, 1:]
    plot_hawkes_kernels(
        hawkes_object,
        support=kernel_support,
        hawkes=hawkes,
        n_points=n_points,
        show=False,
        log_scale=log_scale,
        min_support=min_support,
        ax=kernel_axes,
    )

    period = getattr(hawkes_object, "period_length", None)
    if period is None:
        period = getattr(hawkes, "period_length", None) if hawkes is not None else None
    if period is None:
        period = kernel_support
    if period is None or period <= 0:
        supports = _kernel_supports(hawkes_object)
        finite_supports = supports[np.isfinite(supports) & (supports > 0)]
        period = float(np.max(finite_supports)) if finite_supports.size else 1.0

    t_values = np.linspace(0.0, float(period), int(n_points))
    for i in range(n_nodes):
        axis = axes[i, 0]
        axis.plot(t_values, _baseline_values(hawkes_object, i, t_values), label=f"baseline ({i})")
        if hawkes is not None:
            axis.plot(
                t_values,
                _baseline_values(hawkes, i, t_values),
                linestyle="--",
                label=f"true baseline ({i})",
            )
        axis.set_xlabel("time")
        axis.set_ylabel(f"mu[{i}]")
        axis.legend(loc="best", fontsize=8)

    if show:
        plt.show()
    return axes[0, 0].figure


def _normalize_functions(y_values_list, t_values):
    y_values = np.asarray(y_values_list, dtype=float)
    normalizations = []
    for values in y_values:
        integral = float(np.trapezoid(values, t_values))
        normalizations.append(1.0 / integral if abs(integral) > 1e-15 else 1.0)
    return (y_values.T * normalizations).T, np.asarray(normalizations, dtype=float)


def _find_best_match(diff_matrix):
    diff_matrix = np.asarray(diff_matrix, dtype=float).copy()
    matches = []
    for _ in range(diff_matrix.shape[0]):
        row, col = np.unravel_index(np.argmin(diff_matrix), diff_matrix.shape)
        matches.append((row, col))
        diff_matrix[row, :] = np.inf
        diff_matrix[:, col] = np.inf
    return matches


def _piecewise_step_xy(discretization, values):
    values = np.asarray(values, dtype=float)
    edges = np.asarray(discretization, dtype=float)
    return np.hstack((edges[0], np.repeat(edges[1:-1], 2), edges[-1])), np.repeat(values, 2)


def plot_basis_kernels(learner, support=None, basis_kernels=None, n_points: int = 300, show: bool = True, ax=None):
    """Plot basis kernels from a :class:`HawkesBasisKernels` learner."""

    plt = _plt()
    if support is None or support <= 0:
        support = learner.kernel_support
    axes, created = _as_axes(
        ax,
        (1, learner.n_basis),
        plt,
        figsize=(3.2 * learner.n_basis, 3.0),
        sharex=True,
        sharey=True,
    )
    if not created:
        show = False
    axes = axes[0]

    matches = [(i, i) for i in range(learner.n_basis)]
    true_values = None
    true_normalizations = None
    estimated_normalizations = np.ones(learner.n_basis)
    if basis_kernels is not None:
        if len(basis_kernels) != learner.n_basis:
            raise ValueError(f"learner has {learner.n_basis} basis kernels, got {len(basis_kernels)}")
        t_grid = learner.kernel_discretization[:-1]
        true_values = np.asarray([fn(t_grid) for fn in basis_kernels], dtype=float)
        normalized_true, true_normalizations = _normalize_functions(true_values, t_grid)
        normalized_estimated, estimated_normalizations = _normalize_functions(learner.basis_kernels, t_grid)
        diff = np.array(
            [
                [np.trapezoid(np.abs(est - true), t_grid) for true in normalized_true]
                for est in normalized_estimated
            ]
        )
        matches = _find_best_match(diff)

    dense_t = np.linspace(0.0, float(support), int(n_points))
    for estimated_index, basis_index in matches:
        axis = axes[basis_index]
        step_t, step_y = _piecewise_step_xy(learner.kernel_discretization, learner.basis_kernels[estimated_index])
        axis.step(step_t, step_y, where="post", label=f"estimated {estimated_index}")
        if basis_kernels is not None:
            scale = true_normalizations[basis_index] / estimated_normalizations[estimated_index]
            axis.plot(dense_t, basis_kernels[basis_index](dense_t) * scale, linestyle="--", label=f"true {basis_index}")
        axis.set_xlabel("time")
        axis.legend(loc="best", fontsize=8)

    axes[0].set_ylabel("basis value")
    if show:
        plt.show()
    return axes[0].figure


def plot_estimated_intensity(
    learner,
    events=None,
    intensity_track_step=None,
    end_time=None,
    t_min: float | None = None,
    t_max: float | None = None,
    plot_nodes=None,
    node_names=None,
    n_points: int = 1000,
    show: bool = True,
    ax=None,
):
    """Plot intensities implied by a fitted Hawkes learner and event history."""

    plt = _plt()
    if events is None:
        if getattr(learner, "data", None) is None or len(learner.data) != 1:
            raise ValueError("events must be provided unless learner has exactly one fitted realization")
        events = learner.data[0]
    if end_time is None:
        if getattr(learner, "_end_times", None) is not None and len(learner._end_times):
            end_time = float(learner._end_times[0])
        else:
            end_time = max((float(ts[-1]) for ts in events if len(ts)), default=0.0)
    if intensity_track_step is None:
        intensity_track_step = max(float(end_time), 1.0) / max(int(n_points), 1)

    intensities, times = learner.estimated_intensity(events, intensity_track_step, end_time=end_time)
    times = np.asarray(times, dtype=float)
    n_nodes = len(intensities)
    nodes = list(range(n_nodes)) if plot_nodes is None else list(plot_nodes)
    labels = _node_names(n_nodes, nodes, node_names)
    t_min = 0.0 if t_min is None else float(t_min)
    t_max = float(end_time) if t_max is None else float(t_max)
    mask = _interval_mask(times, t_min, t_max)

    axes, created = _as_axes(
        ax,
        (len(nodes), 1),
        plt,
        figsize=(10, max(2.5, 2.4 * len(nodes))),
        sharex=True,
        sharey=False,
    )
    if not created:
        show = False

    for row, (node, label) in enumerate(zip(nodes, labels)):
        axis = axes[row, 0]
        axis.plot(times[mask], np.asarray(intensities[node], dtype=float)[mask], label="estimated")
        ts = np.asarray(events[node], dtype=float)
        ts = ts[_interval_mask(ts, t_min, t_max)]
        if ts.size and np.any(mask):
            y = np.interp(ts, times[mask], np.asarray(intensities[node], dtype=float)[mask])
            axis.scatter(ts, y, s=18, label="jumps")
        axis.set_ylabel(label)
        axis.legend(loc="best")
    axes[-1, 0].set_xlabel("time")
    if show:
        plt.show()
    return axes[0, 0].figure


def qq_plots(
    point_process=None,
    residuals=None,
    plot_nodes=None,
    node_names=None,
    line: str = "45",
    show: bool = True,
    ax=None,
):
    """Plot exponential QQ diagnostics from compensators or residual arrays."""

    plt = _plt()
    if residuals is None:
        if point_process is None:
            raise ValueError("point_process or residuals must be provided")
        if not point_process.tracked_compensator:
            point_process.store_compensator_values()
        residuals = [np.diff(np.asarray(values, dtype=float)) for values in point_process.tracked_compensator]
    elif point_process is not None:
        raise ValueError("provide either point_process or residuals, not both")

    nodes = list(range(len(residuals))) if plot_nodes is None else list(plot_nodes)
    labels = _node_names(len(residuals), nodes, node_names)
    axes, created = _as_axes(
        ax,
        (len(nodes), 1),
        plt,
        figsize=(5.5, max(3.0, 2.8 * len(nodes))),
        sharex=True,
        sharey=True,
    )
    if not created:
        show = False

    for row, (node, label) in enumerate(zip(nodes, labels)):
        axis = axes[row, 0]
        values = np.sort(np.asarray(residuals[node], dtype=float))
        values = values[np.isfinite(values)]
        if values.size:
            probs = (np.arange(values.size) + 0.5) / values.size
            theoretical = -np.log1p(-probs)
            axis.scatter(theoretical, values, s=14)
            if line == "45":
                lim = max(float(np.max(theoretical)), float(np.max(values)))
                axis.plot([0.0, lim], [0.0, lim], color="black", linewidth=1)
        axis.set_title(label)
        axis.set_ylabel("empirical")
    axes[-1, 0].set_xlabel("theoretical exponential")
    if show:
        plt.show()
    return axes[0, 0].figure


def _extended_discrete_xaxis(x_axis, n_points=100, eps=0.10):
    x_axis = np.asarray(x_axis, dtype=float)
    min_value = float(np.min(x_axis))
    max_value = float(np.max(x_axis))
    distance = max_value - min_value
    if distance == 0:
        distance = 1.0
    return np.linspace(min_value - eps * distance, max_value + eps * distance, num=n_points)


def plot_timefunction(time_function: TimeFunction, labels=None, n_points: int = 300, show: bool = True, ax=None):
    """Plot a :class:`hawkes_tools.base.TimeFunction`."""

    plt = _plt()
    if ax is None:
        _, ax = plt.subplots(figsize=(4.5, 3.5))
    else:
        show = False

    if time_function.is_constant:
        if labels is None:
            labels = [f"value = {time_function.constant:.3g}"]
        t_values = np.arange(10, dtype=float)
        ax.plot(t_values, time_function.value(t_values), label=labels[0])
    else:
        if labels is None:
            interpolation = {
                TimeFunction.InterLinear: "linear",
                TimeFunction.InterConstLeft: "constant left",
                TimeFunction.InterConstRight: "constant right",
            }[time_function.inter_mode]
            border = {
                TimeFunction.Border0: "border zero",
                TimeFunction.BorderConstant: f"border constant {time_function.border_value:.3g}",
                TimeFunction.BorderContinue: "border continue",
                TimeFunction.Cyclic: "cyclic",
            }[time_function.border_type]
            labels = ["original points", f"{interpolation}, {border}"]
        original_t = time_function.original_t
        if time_function.border_type == TimeFunction.Cyclic:
            cycle_length = original_t[-1] - original_t[0]
            original_t = np.hstack((original_t, original_t + cycle_length, original_t + 2 * cycle_length))
        t_values = _extended_discrete_xaxis(original_t, n_points=n_points)
        ax.plot(time_function.original_t, time_function.original_y, linestyle="", marker="o", label=labels[0])
        ax.plot(t_values, time_function.value(t_values), label=labels[1])

    ax.set_xlabel("time")
    ax.legend(loc="best")
    if show:
        plt.show()
    return ax.figure

