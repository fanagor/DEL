"""Publication heatmaps for checkpoint-specific length-generalization results.

Rows and columns retain the supplied experiment order; the first row is at the
bottom. Missing values are masked, and zero is a distinct under-range color on
the logarithmic scale. Ceiling annotations use explicit censoring flags
when available, rather than interpreting an unfinished experiment as a result.
"""

from math import ceil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from .plot_style import HEAT_CMAP, PAPER_RC, style_axes

MISSING_COLOR = "#d8d8d3"
ZERO_COLOR = "#faf9f4"


def draw_heatmap(
    ax,
    values,
    xlabels,
    ylabels,
    title,
    xlabel,
    ylabel,
    metric,
    cap=1000,
    *,
    at_cap=None,
):
    """Draw a 3-by-3 experiment grid and return its image mappable.

    ``values`` uses NaN for unavailable measurements. ``at_cap`` is an optional
    3-by-3 boolean matrix: True prints a >= annotation. Only
    explicitly flagged values are annotated as censored. A finite zero
    means a measured failure at the shortest positive length, never missing data.
    """
    values = np.asarray(values, dtype=float)
    if values.shape != (3, 3) or len(xlabels) != 3 or len(ylabels) != 3:
        raise ValueError("Heatmaps require a 3-by-3 matrix and three labels per axis")
    if not np.isfinite(cap) or cap <= 1:
        raise ValueError("The color-scale ceiling must exceed one")
    finite = np.isfinite(values)
    if np.isinf(values).any() or (values[finite] < 0).any():
        raise ValueError("Generalization lengths must be nonnegative or NaN")
    if (values[finite] > cap).any():
        raise ValueError("A measured value exceeds the color-scale ceiling")
    censored = (
        np.zeros_like(values, dtype=bool) if at_cap is None else np.asarray(at_cap, dtype=bool)
    )
    if censored.shape != values.shape:
        raise ValueError("Ceiling flags must have the same shape as the values")
    cmap = HEAT_CMAP.copy()
    cmap.set_bad(MISSING_COLOR)
    cmap.set_under(ZERO_COLOR)
    # LogNorm cannot represent zero: place measured zeros below its lower bound.
    colors = np.where(values == 0, 0.5, values)
    image = ax.imshow(
        np.ma.masked_invalid(colors),
        origin="lower",
        interpolation="nearest",
        cmap=cmap,
        norm=LogNorm(vmin=1, vmax=cap, clip=False),
        aspect="equal",
    )
    ax.set_xticks(range(3), [str(x) for x in xlabels])
    ax.set_yticks(range(3), [str(y) for y in ylabels])
    ax.set(xlabel=xlabel, ylabel=ylabel)
    ax.set_title(title, fontfamily="DejaVu Sans Mono", fontsize=13, pad=12)
    ax.tick_params(labelsize=11)
    ax.minorticks_off()
    style_axes(ax, grid=False)
    # Internal cell dividers remain subtle against the paper palette.
    ax.vlines([0.5, 1.5], -0.5, 2.5, colors="white", linewidth=1.0, alpha=0.75)
    ax.hlines([0.5, 1.5], -0.5, 2.5, colors="white", linewidth=1.0, alpha=0.75)
    for row, column in np.ndindex(values.shape):
        value = values[row, column]
        if not finite[row, column]:
            label, ink = "—", "#686863"
        else:
            number = f"{value:g}"
            label = ("≥" if censored[row, column] else "") + number
            rgba = cmap(image.norm(colors[row, column]))
            luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
            ink = "white" if luminance < 0.5 else "#30271f"
        ax.text(
            column,
            row,
            label,
            ha="center",
            va="center",
            fontsize=13,
            fontweight="normal",
            color=ink,
        )
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(-0.5, 2.5)
    return image


def plot_heatmap_group(panels, metric, ncols=4, *, cap=1000):
    """Return 3-by-3 heatmaps with shared or panel-specific logarithmic scales.

    Each panel supplies values, xlabels, ylabels, title, xlabel, ylabel and
    optionally at_cap and cap. Panel titles are literal monospace strings, e.g. a) add2.
    ``metric`` labels the colorbar (ULG, RLG, or Generalization). Panels with
    different ceilings receive individual colorbars; equal ceilings share one.
    """
    panels = list(panels)
    if not panels or not isinstance(ncols, int) or ncols < 1:
        raise ValueError("Supply at least one panel and a positive column count")
    ncols = min(ncols, len(panels))
    nrows = ceil(len(panels) / ncols)
    ceilings = [panel.get("cap", cap) for panel in panels]
    mixed_scales = len(set(ceilings)) > 1
    with plt.rc_context(PAPER_RC):
        fig, axes = plt.subplots(
            nrows,
            ncols,
            squeeze=False,
            figsize=((4.6 if mixed_scales else 3.7) * ncols + 0.65, 4.1 * nrows),
            layout="constrained",
        )
        active, images = [], []
        for ax, panel, ceiling in zip(axes.flat, panels, ceilings):
            settings = {key: value for key, value in panel.items() if key != "cap"}
            image = draw_heatmap(ax, metric=metric, cap=ceiling, **settings)
            active.append(ax)
            images.append(image)
        for ax in list(axes.flat)[len(panels) :]:
            ax.remove()

        def add_colorbar(image, panel_axes, ceiling):
            ticks = [tick for tick in (1, 10, 100, 1000, 10000) if tick <= ceiling]
            if ceiling not in ticks:
                ticks.append(ceiling)
            colorbar = fig.colorbar(image, ax=panel_axes, shrink=0.78, pad=0.025, ticks=ticks)
            colorbar.set_label(f"{metric} Length", fontsize=active[0].yaxis.label.get_size())
            colorbar.ax.set_yticklabels([f"{tick:g}" for tick in ticks])
            colorbar.ax.minorticks_off()

        if mixed_scales:
            for image, ax, ceiling in zip(images, active, ceilings):
                add_colorbar(image, ax, ceiling)
        else:
            add_colorbar(images[-1], active, ceilings[0])
    return fig


def save_figure(fig, prefix):
    """Save matching PDF, SVG and 300-DPI PNG exports; return their paths."""
    prefix = Path(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    paths = {}
    for extension in ("pdf", "svg", "png"):
        path = prefix.with_suffix(f".{extension}")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        paths[extension] = path
    return paths


def figure_output_dir(root, *parts):
    """Resolve the figure export root and create the requested subdirectory."""
    import os

    root = Path(root).resolve()
    directory = Path(
        os.environ.get("DEL_ANALYSIS_DIR", root / "analysis_outputs/paper_figures")
    ).expanduser()
    if not directory.is_absolute():
        directory = root / directory
    directory = directory.joinpath(*parts)
    directory.mkdir(parents=True, exist_ok=True)
    return directory
