"""Publication plots of task-wise universal-controller length generalization."""

import math
from collections.abc import Mapping

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnnotationBbox, TextArea, VPacker
from matplotlib.patches import Rectangle
from matplotlib.ticker import FixedLocator, NullFormatter

from .plot_style import COLORS, PAPER_RC, style_axes

TASK_ORDER = (
    "scaling_A",
    "scaling_B",
    "shift_left",
    "shift_right",
    "parity",
    "flip",
    "cum_sum",
    "bit_delete",
    "reverse",
    "copy",
)
TASK_LABELS = (
    "Scaling-A",
    "Scaling-B",
    "Shift-Left",
    "Shift-Right",
    "Parity",
    "Flip",
    "Cum-Sum",
    "Bit-Delete",
    "Reverse",
    "Copy",
)


def _task_result(tasks, key):
    if key in tasks:
        return tasks[key]
    candidates = [value for name, value in tasks.items() if name.startswith(key + "_m")]
    if len(candidates) > 1:
        raise ValueError(f"Choose one evaluation scale for {key}; found multiple results")
    return candidates[0] if candidates else None


def _measurement(value):
    if value is None:
        return None
    if isinstance(value, Mapping):
        length, at_cap = value.get("length"), bool(value.get("at_cap", False))
    else:
        length, at_cap = value, False
    if length is None:
        return None
    length = float(length)
    if not math.isfinite(length) or length < 0:
        raise ValueError("Generalization lengths must be finite and nonnegative")
    return length, at_cap


def plot_universal_generalization(
    tasks,
    hidden_size,
    dataset_size,
    max_length=5,
    *,
    ax=None,
    show_title=True,
    parameter_count=None,
    show_model_size_label=True,
    show_training_set_label=True,
    annotation_fontsize=10,
):
    """Plot training tasks at the bottom with explicit measured length endpoints.

    ``tasks`` maps task names to ``{length, at_cap}`` or scalar lengths. An
    evaluation ceiling is labelled >=; missing measurements are never zero.
    Scaling task names may include a single ``_mN`` suffix. The x-axis starts at
    length one; a measured failure at zero is labelled 0 with a cross at that
    boundary, since zero has no logarithm. Seen-task lines start at the training
    boundary even for measurements below it.
    """
    if max_length <= 1 or hidden_size < 1 or dataset_size < 1:
        raise ValueError("Require positive dimensions and dataset size, and max_length > 1")
    values = [_measurement(_task_result(tasks, key)) for key in TASK_ORDER]
    maximum = max([float(max_length), *(v[0] for v in values if v is not None)])
    with plt.rc_context(PAPER_RC):
        if ax is None:
            fig, ax = plt.subplots(figsize=(9.4, 6.0), constrained_layout=True)
        else:
            fig = ax.figure
        ax.set_xscale("log")
        ax.set_xlim(1, max(12, maximum * 2.25))
        ax.set_ylim(-0.5, 9.6)
        style_axes(ax)
        ax.grid(False, axis="y", which="both")
        ax.add_patch(
            Rectangle(
                (1, -0.5),
                max_length - 1,
                5,
                facecolor="#E5D8BA",
                edgecolor="#B9A47A",
                linewidth=1.2,
                alpha=0.9,
                zorder=2,
            )
        )
        # Center the entire label group, including unequal-height text boxes.
        labels = []
        if parameter_count is not None:
            heading = "Model Size\n" if show_model_size_label else ""
            labels.append(f"{heading}{parameter_count:,}\nParameters")
        training_heading = "Training Set\n" if show_training_set_label else ""
        labels.append(f"{training_heading}{dataset_size:,} PTOs")
        boxes = [
            TextArea(
                label,
                textprops=dict(
                    ha="center",
                    multialignment="center",
                    fontsize=annotation_fontsize,
                    color="black",
                    bbox=dict(boxstyle="round,pad=0.38", facecolor="#F7EEDB", edgecolor="black"),
                ),
            )
            for label in labels
        ]
        group = VPacker(children=boxes, align="center", pad=0, sep=24)
        label_group = AnnotationBbox(
            group,
            (math.sqrt(max_length), 2),
            xycoords="data",
            box_alignment=(0.5, 0.5),
            frameon=False,
            zorder=4,
        )
        label_group.set_gid("training-region-labels")
        ax.add_artist(label_group)
        for y, value in enumerate(values):
            seen = y < 5
            color = COLORS[0] if seen else COLORS[3]
            start = max_length if seen else 1
            if value is None:
                ax.text(
                    start * 1.10,
                    y,
                    "Not Evaluated",
                    ha="left",
                    va="center",
                    fontsize=10,
                    color="#6E6C67",
                    zorder=5,
                )
                continue
            length, at_cap = value
            endpoint = max(1, length)
            ax.plot([start, endpoint], [y, y], color=color, linewidth=5.5, alpha=0.13, zorder=3)
            ax.plot([start, endpoint], [y, y], color=color, linewidth=2.2, zorder=4)
            ax.plot(
                endpoint,
                y,
                marker="x" if length == 0 else "o",
                color=color,
                markersize=5.5,
                markeredgecolor="white" if length else color,
                markeredgewidth=0.9,
                zorder=5,
            )
            number = f"{length:,.0f}" if length.is_integer() else f"{length:g}"
            label = f"≥ {number}" if at_cap else number
            ax.annotate(
                label,
                (endpoint, y),
                xytext=(9, 0),
                textcoords="offset points",
                ha="left",
                va="center",
                fontsize=10,
                color="white",
                weight="bold",
                zorder=6,
                bbox=dict(boxstyle="round,pad=0.22", facecolor=color, edgecolor="none"),
            )
        ticks = sorted(
            {1, max_length, *(10**p for p in range(1, int(math.log10(ax.get_xlim()[1])) + 1))}
        )
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.set_xticklabels([f"{tick:,g}" for tick in ticks])
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_yticks(range(10), TASK_LABELS)
        ax.tick_params(axis="y", length=0, pad=9)
        ax.set_xlabel("Input Length (Log Scale)")
        if show_title:
            ax.set_title(f"Hidden Dimensions {hidden_size}/ Dataset Size {dataset_size}", pad=16)
        else:
            ax.set_title("")
        ax.legend(
            handles=[
                Line2D([0], [0], color=COLORS[0], marker="o", label="Seen Tasks"),
                Line2D([0], [0], color=COLORS[3], marker="o", label="Unseen Tasks"),
            ],
            loc="upper center",
            bbox_to_anchor=(0.5, 1.0),
            ncol=2,
            frameon=False,
        )
    return fig
