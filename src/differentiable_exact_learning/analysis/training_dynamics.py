"""Read saved training histories and plot arithmetic loss, ULG and RLG trajectories."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from .plot_style import COLORS, PAPER_RC, paper_curve, style_axes

MODELS = ("DFST", "RNN", "GRU")
MODEL_COLORS = dict(zip(MODELS, (COLORS[1], COLORS[0], COLORS[2])))
MARKERS = dict(zip(MODELS, ("o", "s", "^")))
TASKS = {"add2": "addition2", "mult2": "multiplication2"}


def average_loss(steps, losses, window):
    """Average logged interval means over a trailing window of training updates.

    A row at step s describes updates since the preceding logged step (initially
    zero). Integrating that piecewise-constant history weights irregular logging
    intervals correctly. The initial window uses only observed updates.
    """
    steps, losses = np.asarray(steps, dtype=float), np.asarray(losses, dtype=float)
    if isinstance(window, bool) or not isinstance(window, int) or window < 1:
        raise ValueError("LOSS_WINDOW must be a positive integer")
    if (
        steps.ndim != 1
        or losses.shape != steps.shape
        or len(steps) == 0
        or not np.isfinite(steps).all()
        or not np.isfinite(losses).all()
        or (losses < 0).any()
        or steps[0] <= 0
        or (np.diff(steps) <= 0).any()
    ):
        raise ValueError(
            "Loss history requires increasing positive steps and finite nonnegative losses"
        )
    edges = np.r_[0, steps]
    left = np.maximum(0, steps - window)
    first = np.searchsorted(steps, left, side="right")
    averaged = np.empty_like(losses)
    # Sum within each window: subtracting global integrals erases tiny late losses.
    for end, start in enumerate(first):
        widths = np.diff(edges[start : end + 2]).copy()
        widths[0] = steps[start] - left[end]
        averaged[end] = np.dot(widths / (steps[end] - left[end]), losses[start : end + 1])
    return averaged


def _protocol(evaluation, config, score):
    options = evaluation.get("evaluation_config", config["evaluation"])
    # RLG hard cases do not affect the ULG protocol.
    return dict(
        protocol=evaluation.get("protocol", "teacher_forced_exact_match"),
        seed=evaluation.get("seed", options["seed"]),
        samples=evaluation.get("random_samples", options["samples"]),
        cap=score.get("cap", options["max_length"]),
        search=options["search"],
        scan=evaluation.get("scan", options.get("scan", "sequential")),
    )


def load_history(
    run_dir,
    *,
    model,
    task,
    loss_window=1000,
    evaluation_every=100000,
    evaluation_dir=None,
    metric="ulg",
):
    """Read one explicit metric without substituting another metric or synthetic values."""
    if metric not in {"ulg", "rlg"}:
        raise ValueError("metric must be ulg or rlg")
    if task not in TASKS or model not in MODELS:
        raise ValueError("Select add2 or mult2 and one supported architecture")
    if (
        isinstance(evaluation_every, bool)
        or not isinstance(evaluation_every, int)
        or evaluation_every < 1
    ):
        raise ValueError("evaluation_every must be a positive integer")
    run_dir = Path(run_dir).expanduser().resolve()
    config = yaml.safe_load((run_dir / "config.yaml").read_text())
    data = config["data"]
    if (
        config["model"]["name"] != model
        or data["radix"] != 2
        or data["task"] + str(data["radix"]) != TASKS[task]
    ):
        raise ValueError(f"Run does not match {model} / {task}: {run_dir}")
    path = run_dir / "metrics.jsonl"
    entries = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    entries = [row for row in entries if "loss" in row]
    steps, losses = [row["step"] for row in entries], [row["loss"] for row in entries]
    smoothed = average_loss(steps, losses, loss_window)
    loss = pd.DataFrame(dict(step=steps, loss=losses, averaged_loss=smoothed))
    planned = config["training"]["steps"]
    if loss.step.iloc[-1] > planned:
        raise ValueError(f"History exceeds the configured training budget: {run_dir}")
    last = int(loss.step.iloc[-1])
    expected = list(range(evaluation_every, last + 1, evaluation_every))
    if evaluation_dir is not None:
        directory = Path(evaluation_dir).expanduser().resolve()
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        paths = sorted(directory.glob("*.json"))
    else:
        paths = sorted(
            set(run_dir.glob("evaluation_*.json"))
            | set((run_dir / "evaluations").glob("step_*.json"))
        )
        if (run_dir / "evaluation.json").exists():
            paths.append(run_dir / "evaluation.json")
    observed, protocol = {}, None
    for path in paths:
        evaluation = json.loads(path.read_text())
        step = evaluation.get("step")
        if step not in expected:
            continue
        values = evaluation.get("tasks", {}).get(TASKS[task])
        if values is None or metric not in values:
            continue
        score = values[metric]
        current = _protocol(evaluation, config, score)
        if metric == "rlg":
            options = evaluation.get("evaluation_config", config["evaluation"])
            current["hard_cases"] = values.get("hard_cases", options.get("hard_cases"))
        if protocol is not None and current != protocol:
            raise ValueError(f"{metric.upper()} evaluation settings differ within the run: {path}")
        protocol = current
        length = score["length"]
        cap, at_cap = current["cap"], score.get("at_cap", False)
        if (
            not np.isfinite(length)
            or not 0 <= length <= cap
            or not isinstance(at_cap, bool)
            or (at_cap and length != cap)
        ):
            raise ValueError(f"Invalid {metric.upper()} value or ceiling flag: {path}")
        row = {"step": step, metric: length, "cap": cap, "at_cap": at_cap}
        if step in observed and observed[step] != row:
            raise ValueError(f"Conflicting {metric.upper()} results at step {step}: {path}")
        observed[step] = row
    scores = pd.DataFrame(
        [
            observed.get(step, {"step": step, metric: np.nan, "cap": np.nan, "at_cap": False})
            for step in expected
        ],
        columns=["step", metric, "cap", "at_cap"],
    )
    return dict(
        run=str(run_dir),
        model=model,
        task=task,
        config=config,
        loss=loss,
        metric=metric,
        **{metric: scores},
        protocol=protocol,
        missing_steps=[s for s in expected if s not in observed],
        planned_steps=planned,
        logged_steps=last,
        loss_window=loss_window,
        evaluation_every=evaluation_every,
    )


def _plot_task(axes, histories, task, metric, require_complete):
    if metric not in {"ulg", "rlg"}:
        raise ValueError("metric must be ulg or rlg")
    if not histories or not set(histories).issubset(MODELS):
        raise ValueError("Select at least one supported architecture: DFST, RNN or GRU")
    protocols, windows, intervals = [], set(), set()
    for model, history in histories.items():
        if (
            history["model"] != model
            or history["task"] != task
            or history.get("metric", "ulg") != metric
        ):
            raise ValueError(
                "History identity differs from the selected task, architecture or metric"
            )
        windows.add(history["loss_window"])
        intervals.add(history["evaluation_every"])
        if history["protocol"] is not None:
            protocols.append(history["protocol"])
        if require_complete and (
            history["missing_steps"]
            or history[metric].empty
            or history["logged_steps"] != history["planned_steps"]
        ):
            raise ValueError(
                f"{model}: incomplete training history or 100k-step {metric.upper()} evaluations"
            )
    if len(windows) != 1 or intervals != {100000}:
        raise ValueError("Use one loss window and evaluations every 100000 iterations")
    if protocols and any(p != protocols[0] for p in protocols):
        raise ValueError(f"Selected runs use different {metric.upper()} evaluation settings")
    for model in MODELS:
        if model not in histories:
            continue
        history = histories[model]
        loss, scores = history["loss"], history[metric]
        # Preserve zero in the exported data and leave a gap on the log axis.
        y = loss.averaged_loss.where(loss.averaged_loss > 0)
        paper_curve(axes[0], loss.step, y, color=MODEL_COLORS[model], label=model)
        paper_curve(
            axes[1],
            scores.step,
            scores[metric],
            color=MODEL_COLORS[model],
            label=model,
            marker=MARKERS[model],
        )
    losses = {h["config"]["training"].get("loss") for h in histories.values()}
    ylabel = "Training Loss (MSE)" if losses == {"mse"} else "Training Loss"
    axes[0].set(title=f"{task} · Averaged Training Loss", ylabel=ylabel, yscale="log")
    axes[1].set(
        title=f"{task} · {metric.upper()} Every 100k Iterations", ylabel=f"{metric.upper()} Length"
    )
    cap = (
        protocols[0]["cap"]
        if protocols
        else max(h[metric]["cap"].max() for h in histories.values())
    )
    if np.isfinite(cap):
        axes[1].set_ylim(0, cap * 1.10)
        axes[1].text(
            0.99,
            cap * 1.025,
            f"Evaluation Cap: {cap:g}",
            transform=axes[1].get_yaxis_transform(),
            ha="right",
            va="bottom",
            fontsize=10,
            color="#555555",
        )
    from matplotlib.ticker import FuncFormatter

    def iteration_label(value, position):
        if value == 0:
            return "0"
        return f"{value / 1000000:g}M" if value >= 1000000 else f"{value / 1000:g}k"

    end = max(h["planned_steps"] for h in histories.values())
    for ax in axes:
        ax.set_xlabel("Training Iteration")
        ax.set_xlim(-0.02 * end, 1.01 * end)
        ax.set_xticks(np.linspace(0, end, 6))
        ax.xaxis.set_major_formatter(FuncFormatter(iteration_label))
        style_axes(ax)
    axes[0].legend(loc="upper right", frameon=True)
    axes[1].legend(loc="lower right", frameon=True)


def plot_training_dynamics(histories, task, *, metric="ulg", require_complete=True):
    """Return loss and the selected generalization metric for one task."""
    with plt.rc_context(PAPER_RC):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
        try:
            _plot_task(axes, histories, task, metric, require_complete)
        except Exception:
            plt.close(fig)
            raise
    return fig
