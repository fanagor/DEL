"""Paper figure style: serif type, terracotta heatmaps, and gray plotting panels.

Palette and rc settings follow the original heatmaps and training-loss figures.
Current notebooks share this module.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler
from matplotlib.colors import LinearSegmentedColormap

BACKGROUND = "#ececea"
COLORS = ("#1f5d7a", "#a4503e", "#c98538", "#4a7c45")
HEAT_CMAP = LinearSegmentedColormap.from_list(
    "del_cream_terra", [BACKGROUND, "#e2a98c", "#b4452a", "#5a2310"]
)
PAPER_RC = {
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
    "mathtext.fontset": "cm",
    "axes.unicode_minus": False,
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "path",
    "axes.facecolor": BACKGROUND,
    "axes.edgecolor": "black",
    "axes.linewidth": 1.7,
    "axes.labelsize": 15,
    "axes.titlesize": 14,
    "axes.prop_cycle": cycler(color=COLORS),
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "xtick.major.size": 6,
    "ytick.major.size": 6,
    "xtick.major.width": 1.5,
    "ytick.major.width": 1.5,
    "legend.fontsize": 9,
    "legend.framealpha": 0.8,
    "legend.facecolor": "#f2f2f0",
    "legend.edgecolor": "#d0d0d0",
    "grid.color": "white",
    "grid.linewidth": 1.2,
    "lines.linewidth": 1.8,
    "lines.solid_capstyle": "round",
}


def apply_paper_style():
    """Set a consistent default for the active notebook's figures."""
    if HEAT_CMAP.name not in mpl.colormaps:
        mpl.colormaps.register(HEAT_CMAP)
    plt.rcParams.update(PAPER_RC)
    plt.rcParams["image.cmap"] = HEAT_CMAP.name


def style_axes(ax, *, grid=True):
    """Match the paper's training-loss panels without changing the data scale."""
    ax.set_facecolor(BACKGROUND)
    ax.set_axisbelow(True)
    if grid:
        ax.grid(True, which="major", color="white", linewidth=1.2)
        ax.grid(True, which="minor", color="white", linewidth=0.7, alpha=0.65)
    else:
        ax.grid(False, which="both")
    ax.tick_params(axis="both", direction="out", colors="black")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.7)


def index_ticks(size, maximum=5):
    """At most five integer tick positions, always including the endpoints."""
    if size < 1:
        return np.array([], dtype=int)
    return np.unique(np.linspace(0, size - 1, min(size, maximum)).round().astype(int))


def heatmap_ticks(ax, shape, *, xlabels=None, ylabels=None):
    """Use sparse row/column labels and no additional minor ticks."""
    rows, columns = shape
    xs, ys = index_ticks(columns), index_ticks(rows)
    ax.set_xticks(xs, [str(x) if xlabels is None else xlabels[x] for x in xs])
    ax.set_yticks(ys, [str(y) if ylabels is None else ylabels[y] for y in ys])
    ax.minorticks_off()
    style_axes(ax, grid=False)


def paper_curve(ax, x, y, *, color, label=None, marker=None, linestyle="-"):
    """Draw the original loss-curve halo and a crisp foreground line."""
    ax.plot(x, y, color=color, linewidth=2.8, alpha=0.16, linestyle=linestyle)
    return ax.plot(
        x,
        y,
        color=color,
        linewidth=1.8,
        alpha=0.98,
        label=label,
        marker=marker,
        markersize=4.5,
        markeredgecolor="white",
        markeredgewidth=0.7,
        linestyle=linestyle,
    )[0]


def plot_graph_stability(results, epsilon):
    """Plot at most ten prefixes; graph changes compare adjacent trajectories.

    All prefixes remain in the source results. Subsampling the display can omit
    transient changes, so this plot alone does not certify a stable interval.
    """
    import matplotlib.pyplot as plt

    if results.empty:
        raise ValueError("Stability results must contain at least one prefix")
    frame = results.sort_values("samples").iloc[index_ticks(len(results), maximum=10)]
    fig, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
    change_ax = ax.twinx()
    lines = []
    for metric, label, color, marker in (
        ("components", "Components", COLORS[1], "o"),
        ("edges", "Edges", COLORS[0], "s"),
    ):
        lines.extend(ax.plot(frame.samples, frame[metric], color=color, marker=marker, label=label))
    changes = frame.structure_changed.map(
        lambda value: float(value) if value is not None else float("nan")
    )
    lines.extend(
        change_ax.plot(
            frame.samples,
            changes,
            color=COLORS[2],
            marker="^",
            linestyle="--",
            label="Graph Changed",
        )
    )
    ax.set(
        xlabel="Sampled PTOs",
        ylabel="Count",
        title=rf"Epsilon-Chain Stability ($\epsilon = {epsilon:g}$)",
    )
    ax.set_xticks(frame.samples)
    change_ax.set(ylabel="Graph Changed", yticks=[0, 1], ylim=(-0.08, 1.08))
    style_axes(ax)
    change_ax.grid(False)
    ax.legend(lines, [line.get_label() for line in lines])
    return fig, ax, change_ax
