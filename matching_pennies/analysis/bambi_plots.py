import numpy as np
import matplotlib.pyplot as plt
import arviz as az
from typing import Optional, Sequence, Tuple, List, Union

def plot_trace(
    idata: az.InferenceData,
    var_names: Optional[Sequence[str]] = None,
    *,
    combined: bool = True,
    compact: bool = False,
    legend: bool = True,
    zero_line: bool = True,
) -> Tuple[plt.Figure, List[plt.Axes]]:
    """
    Plot ArviZ trace and synchronize the x-limits of all posterior (KDE) panels
    to a common symmetric range around 0. Returns (figure, axes).

    Parameters
    ----------
    idata : az.InferenceData
        Output from Bambi/PyMC (e.g., from model.fit()).
    var_names : sequence of str, optional
        Variables to plot; defaults to all.
    combined : bool
        Passed to az.plot_trace.
    compact : bool
        Passed to az.plot_trace.
    legend : bool
        Passed to az.plot_trace.
    zero_line : bool
        If True, draw a vertical line at x=0 on posterior panels.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : list[matplotlib.axes.Axes]
        Flattened list of axes (posterior/trace interleaved, ArviZ style).
    """
    # Make the trace plot
    trace_out = az.plot_trace(
        idata,
        var_names=var_names,
        combined=combined,
        compact=compact,
        legend=legend,
    )

    # Normalize to a flat list of axes
    if hasattr(trace_out, "axes"):   # Matplotlib Figure
        fig: plt.Figure = trace_out
        axes_arr = np.array(fig.axes)
    else:  # already an array-like of Axes, ArviZ older/newer versions can do this
        axes_arr = np.array(trace_out).ravel()
        # Try to find the parent figure from the first axis if possible
        fig = axes_arr[0].get_figure() if len(axes_arr) else plt.gcf()

    axes: List[plt.Axes] = list(axes_arr)

    if not axes:
        return fig, axes  # nothing to sync

    # Posterior (KDE) panels are at even indices: [0, 2, 4, ...]
    post_axes = axes_arr[0::2]

    # Collect symmetric limits around 0 for each posterior panel
    xmin_list = []
    xmax_list = []
    for ax in post_axes:
        xmin, xmax = ax.get_xlim()
        max_abs = max(abs(xmin), abs(xmax))
        xmin_list.append(-max_abs)
        xmax_list.append(+max_abs)

    # Shared symmetric x-limits across all posterior panels
    global_xlim: Tuple[float, float] = (min(xmin_list), max(xmax_list))

    for ax in post_axes:
        ax.set_xlim(global_xlim)
        if zero_line:
            ax.axvline(0, linestyle="--", linewidth=1)

    plt.tight_layout()
    return fig, axes
