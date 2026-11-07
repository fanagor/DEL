"""Final-checkpoint arithmetic and universal generalization panels."""

import math

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.ticker import FixedLocator, NullFormatter

from .figure_data import count_trainable_parameters
from .plot_style import PAPER_RC, style_axes
from .universal_figures import TASK_ORDER, _measurement, _task_result, plot_universal_generalization

ARITHMETIC_ORDER = ("add2", "add10", "mult2", "mult10")
BROWN = "#a4503e"


def plot_front_page(arithmetic, universal):
    """Create two untitled panels from saved evaluation results."""
    with plt.rc_context(PAPER_RC):
        fig, (left, right) = plt.subplots(1, 2, figsize=(14, 5.5), layout="constrained")
        boundaries, values = [], []
        for task in ARITHMETIC_ORDER:
            record = arithmetic[task]
            config = record["config"]
            boundary = int(config["data"]["max_length"])
            if boundary < 1:
                raise ValueError("Arithmetic training lengths must be positive")
            key = config["data"]["task"] + str(config["data"]["radix"])
            value = _measurement(record["evaluation"]["tasks"][key]["ulg"])
            if value is None:
                raise ValueError(f"Missing final ULG for {task}")
            boundaries.append(boundary)
            values.append(value)
        left.set_xscale("log")
        left.set_xlim(1, max(12, max([*boundaries, *(v[0] for v in values)]) * 4.0))
        left.set_ylim(-0.5, 3.5)
        style_axes(left)
        left.grid(False, axis="y", which="both")
        points = [(1, -0.5)]
        for y, boundary in enumerate(boundaries):
            points.extend([(boundary, y - 0.5), (boundary, y + 0.5)])
        points.append((1, 3.5))
        left.add_patch(
            Polygon(
                points,
                closed=True,
                facecolor="#E5D8BA",
                edgecolor="#B9A47A",
                linewidth=1.2,
                alpha=0.9,
                zorder=2,
            )
        )
        parameter_labels = [
            f"{count_trainable_parameters(arithmetic[task]):,} Parameters"
            for task in ARITHMETIC_ORDER
        ]
        for y, ((length, capped), boundary) in enumerate(zip(values, boundaries)):
            size = int(arithmetic[ARITHMETIC_ORDER[y]]["config"]["data"]["size"])
            left.annotate(
                f"{size:,} PTOs",
                (0, y),
                xycoords=left.get_yaxis_transform(),
                xytext=(-40, -18),
                textcoords="offset points",
                annotation_clip=False,
                ha="center",
                va="center",
                fontsize=11,
                color="black",
                zorder=4,
                bbox=dict(boxstyle="round,pad=0.25", facecolor="#F7EEDB", edgecolor="black"),
            )
            # Keep a common left edge just beyond the widest training region.
            left.annotate(
                parameter_labels[y],
                (max(boundaries), y + 0.27),
                xytext=(14, 0),
                textcoords="offset points",
                ha="left",
                va="center",
                fontsize=11,
                color="black",
                zorder=4,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#F7EEDB", edgecolor="black"),
            )
            endpoint = max(1, length)
            # A score below the training range is shown honestly inside it.
            if length >= boundary:
                left.plot(
                    [boundary, endpoint], [y, y], color=BROWN, linewidth=5.5, alpha=0.13, zorder=3
                )
                left.plot([boundary, endpoint], [y, y], color=BROWN, linewidth=2.2, zorder=4)
            left.plot(
                endpoint,
                y,
                marker="x" if length == 0 else "o",
                color=BROWN,
                markersize=5.5,
                markeredgecolor="white" if length else BROWN,
                zorder=5,
            )
            label = ("≥ " if capped else "") + f"{length:,.0f}"
            left.annotate(
                label,
                (endpoint, y),
                xytext=(9, 0),
                textcoords="offset points",
                va="center",
                color="white",
                fontsize=10,
                weight="bold",
                zorder=6,
                bbox=dict(boxstyle="round,pad=0.22", facecolor=BROWN, edgecolor="none"),
            )
        ticks = sorted(
            {1, *boundaries, *(10**i for i in range(1, int(math.log10(left.get_xlim()[1])) + 1))}
        )
        left.xaxis.set_major_locator(FixedLocator(ticks))
        left.set_xticklabels([f"{x:,}" for x in ticks])
        left.xaxis.set_minor_formatter(NullFormatter())
        left.set_yticks(range(4), ARITHMETIC_ORDER, fontfamily="DejaVu Sans Mono")
        left.tick_params(axis="y", length=0, pad=40)
        plt.setp(left.get_yticklabels(), ha="center")
        left.set_xlabel("Input Length (Log Scale)")
        scores = {
            task: item["generalization"]
            for task, item in universal["evaluation"]["tasks"].items()
            if "generalization" in item
        }
        if any(_task_result(scores, task) is None for task in TASK_ORDER):
            raise ValueError("Final universal evaluation must contain all ten tasks")
        config = universal["config"]
        plot_universal_generalization(
            scores,
            config["model"].get("hidden_size") or 1,
            config["data"]["size"],
            config["data"]["max_length"],
            ax=right,
            show_title=False,
            parameter_count=count_trainable_parameters(universal),
            show_model_size_label=False,
            show_training_set_label=False,
            annotation_fontsize=11,
        )
        right.text(
            math.sqrt(config["data"]["max_length"]),
            0.3,
            "universal",
            ha="center",
            va="center",
            fontfamily="DejaVu Sans Mono",
            fontsize=left.get_yticklabels()[0].get_fontsize(),
            color="black",
            zorder=4,
        )
        return fig
