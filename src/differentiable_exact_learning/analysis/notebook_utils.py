"""Shared, portable notebook setup. No network calls or private checkpoint paths."""

import json
import random
from pathlib import Path

import torch


def load_analysis_run(run_dir=None, *, task="addition", radix=2, softmax=False):
    """Load a requested run; absent path selects a labelled analytic demo.

    A supplied invalid run path is an error: never silently replace research
    results with a demo. Demo parameters encode the expert, not a training claim.
    """
    if run_dir is not None:
        from differentiable_exact_learning.training import load_run

        model, expert, config, checkpoint = load_run(Path(run_dir))
        model = model.cpu().eval()
        print(f"TRAINED RUN: {Path(run_dir).resolve()} (step {checkpoint.get('step', 'unknown')})")
        return model, expert, config, False
    from differentiable_exact_learning.config import load_config
    from differentiable_exact_learning.data import create_expert
    from differentiable_exact_learning.models import build_model

    config = load_config(
        overrides=[
            f"data.task={task}",
            f"data.radix={radix}",
            "data.max_length=2",
            "data.source=online",
            "model.scan=sequential",
            f"model.name={'DFST_Softmax' if softmax else 'DFST'}",
        ]
    )
    expert = create_expert(config["data"])
    model = build_model(
        config["model"], expert.input_size, expert.symbol_size, expert.motion_size, expert.n_states
    ).cpu()
    A, B, C = expert.get_tensors()
    with torch.no_grad():
        for parameter, target in zip((model.A, model.B, model.C), (A, B, C)):
            target = target.to(parameter)
            parameter.copy_(target.clamp_min(1e-12).log() if softmax else target)
        state = torch.zeros_like(model.h0)
        state[0] = 1
        model.h0.copy_(state.clamp_min(1e-12).log() if softmax else state)
    model.eval()
    print(
        "DEMO: fresh controller initialized from the expert tables; not a trained research result."
    )
    return model, expert, config, True


def sample_traces(expert, count=4, max_length=2, seed=42):
    rng = random.Random(seed)
    return [expert.sample(rng, max_length=max_length) for _ in range(count)]


@torch.no_grad()
def hidden_trajectories(model, traces, stop_index=None):
    """Teacher-forced trajectories with pre-transition output timing preserved.

    Returns the original research graph schema. For paper extraction prefer
    closed_loop_trajectories once an environment policy rollout is available.
    """
    if not hasattr(model, "effective_tensors"):
        raise TypeError("DFST tensor geometry requires a DFST or DFST_Softmax controller.")
    A, B, C, h0 = (v.detach().cpu() for v in model.effective_tensors())
    trajectories = []
    for trace in traces:
        h = h0.clone()
        h_seq, x_seq, sig_seq, halt_seq = [h.clone()], [], [], [False]
        for token in trace.inputs.tolist():
            symbol = int((B[token] @ h).argmax())
            motion = int((C[token] @ h).argmax())
            h = A[token] @ h
            h_seq.append(h.clone())
            x_seq.append(token)
            sig_seq.append((symbol, motion))
            halt_seq.append(motion == stop_index)
            if motion == stop_index:
                break
        trajectories.append(
            dict(
                h_seq=h_seq,
                x_seq=x_seq,
                sig_seq=sig_seq,
                halt_seq=halt_seq,
                metadata=trace.metadata,
                sampling="teacher_forced",
            )
        )
    return trajectories


def load_metrics(run_dir=None):
    """Read local training JSONL, or explicitly labelled synthetic demo series."""
    if run_dir is None:
        print("DEMO METRICS: synthetic curves used only to demonstrate plotting.")
        return [
            {
                "step": i * 10,
                "loss": 0.7**i + 1e-6,
                "accuracy": 1 - 0.7**i,
                "rlg": min(2 + i // 3, 8),
                "demo": True,
            }
            for i in range(21)
        ]
    path = Path(run_dir) / "metrics.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"Expected local training metrics: {path}")
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_dynamics_json(path):
    """Read a historical exported dynamics JSON without querying W&B."""
    return json.loads(Path(path).read_text())


def moving_average(values, window=1):
    """Causal average in recorded rows (not optimization steps)."""
    if window < 1:
        raise ValueError("window must be positive")
    values = list(values)
    return [
        sum(values[max(0, i - window + 1) : i + 1]) / min(i + 1, window) for i in range(len(values))
    ]


def plot_metrics(rows, *, smoothing=1, title="Training Diagnostics"):
    import matplotlib.pyplot as plt

    from .plot_style import COLORS, PAPER_RC, paper_curve, style_axes

    training = [row for row in rows if "loss" in row]
    with plt.rc_context(PAPER_RC):
        fig, ax = plt.subplots(figsize=(7, 4))
        paper_curve(
            ax,
            [row["step"] for row in training],
            moving_average([row["loss"] for row in training], smoothing),
            color=COLORS[1],
            label="Loss",
        )
        ax.set(xlabel="Iteration", ylabel="Training Loss", yscale="log", title=title)
        ax.minorticks_on()
        style_axes(ax)
        evaluated = [row for row in rows if "rlg" in row]
        if evaluated:
            secondary = ax.twinx()
            secondary.patch.set_alpha(0)
            secondary.plot(
                [row["step"] for row in evaluated],
                [row["rlg"] for row in evaluated],
                color=COLORS[1],
                linewidth=1.8,
                alpha=0.65,
            )
            secondary.scatter(
                [row["step"] for row in evaluated],
                [row["rlg"] for row in evaluated],
                s=42,
                color=COLORS[1],
                edgecolor="white",
                linewidth=0.9,
                zorder=3,
            )
            secondary.set_ylabel("RLG")
            secondary.spines["right"].set_linewidth(1.7)
        fig.tight_layout()
    return fig


@torch.no_grad()
def closed_loop_trajectories(model, expert, traces, max_steps=10000):
    """Let the controller write/move/read its own environment, as in paper extraction.

    Ground-truth traces supply initial instances only. Truncated/non-finite runs
    raise an error so a nonhalting trajectory cannot become false halt evidence.
    """
    if max_steps < 1:
        raise ValueError("max_steps must be positive")
    A, B, C, h0 = (v.detach().cpu() for v in model.effective_tensors())
    if not all(torch.isfinite(t).all() for t in (A, B, C, h0)):
        raise ValueError("Non-finite model tensors during closed-loop sampling")
    symbols = {v: k for k, v in expert.vocab.items()}
    motions = {v: k for k, v in expert.motion_vocab.items()}
    trajectories = []
    for trace in traces:
        memory = expert.make_environment(trace.metadata)
        h = h0.clone()
        h_seq, x_seq, sig_seq, halt_seq = [h.clone()], [], [], [False]
        for _ in range(max_steps):
            x = expert.vocab[memory.read()]
            symbol_scores, motion_scores = B[x] @ h, C[x] @ h
            if not (torch.isfinite(symbol_scores).all() and torch.isfinite(motion_scores).all()):
                raise ValueError("Non-finite readout during closed-loop sampling")
            symbol, motion = int(symbol_scores.argmax()), int(motion_scores.argmax())
            h = A[x] @ h
            if not torch.isfinite(h).all():
                raise ValueError("Non-finite hidden state during closed-loop sampling")
            memory.write(symbols[symbol])
            memory.move(motions[motion])
            halted = motions[motion] == "S"
            h_seq.append(h.clone())
            x_seq.append(x)
            sig_seq.append((symbol, motion))
            halt_seq.append(halted)
            if halted:
                break
        else:
            raise RuntimeError(
                f"Controller did not halt within {max_steps} steps; increase max_steps or inspect the run"
            )
        trajectories.append(
            dict(
                h_seq=h_seq,
                x_seq=x_seq,
                sig_seq=sig_seq,
                halt_seq=halt_seq,
                metadata=trace.metadata,
                sampling="closed_loop",
                cells=dict(memory.cells),
            )
        )
    return trajectories


def evaluation_history(run_dir, task=None):
    """Read one task's RLG/generalization history; never mix tasks implicitly."""
    records = []
    files = sorted(Path(run_dir).glob("evaluation_*.json"))
    final = Path(run_dir) / "evaluation.json"
    if final.is_file():
        files.append(final)
    seen = set()
    for path in files:
        saved = json.loads(path.read_text())
        tasks = saved.get("tasks", {})
        if task is None:
            if len(tasks) != 1:
                raise ValueError("Select task explicitly when evaluation contains multiple tasks")
            selected = next(iter(tasks))
        else:
            selected = task
        if selected not in tasks:
            continue
        outcome = tasks[selected].get("rlg", tasks[selected].get("generalization"))
        if outcome is None:
            continue
        step = saved.get("step")
        if step is None:
            raise ValueError(f"Evaluation has no optimization step: {path}")
        key = (step, selected)
        if key in seen:
            continue
        seen.add(key)
        records.append(
            {
                "step": step,
                "rlg": outcome["length"],
                "at_cap": outcome.get("at_cap", False),
                "task": selected,
            }
        )
    return sorted(records, key=lambda row: row["step"])


def sample_unique_grid_traces(expert, count, max_length, rng):
    """Sample distinct arithmetic initial grids using an explicit sampler state."""
    if expert.environment != "grid":
        raise ValueError("Distinct-grid sampling requires a grid expert")
    for name, value in (("count", count), ("max_length", max_length)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    capacity = sum(expert.radix**length for length in range(1, max_length + 1)) ** 2
    if count > capacity:
        raise ValueError(f"Requested {count} distinct grids, but only {capacity} exist")
    seen, traces = set(), []
    for _ in range(max(1000, 100 * count)):
        trace = expert.sample(rng, max_length=max_length)
        key = tuple(tuple(tokens) for tokens in trace.metadata["operand_tokens"])
        if key in seen:
            continue
        seen.add(key)
        traces.append(trace)
        if len(traces) == count:
            return traces
    raise ValueError(
        "Distinct-grid sampling exhausted its retries; reduce count or increase max_length"
    )


def sample_extraction_traces(expert, count, length, rng, *, tasks=None):
    """Sample grid inputs or uniformly chosen binary TMs with exact-length words."""
    if expert.environment == "grid":
        return sample_unique_grid_traces(expert, count, length, rng)
    if expert.environment != "teleporting":
        raise ValueError("Metric extraction supports grid and teleporting experts")
    from ..data.experts import DEFAULT_TP_TASKS, task_label

    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("count must be a positive integer")
    if isinstance(length, bool) or not isinstance(length, int) or length < 0:
        raise ValueError("Universal input length must be a nonnegative integer")
    selected = [task_label(task) for task in (DEFAULT_TP_TASKS if tasks is None else tasks)]
    if not selected or len(selected) != len(set(selected)):
        raise ValueError("Choose a nonempty list of distinct universal tasks")
    return [
        expert.sample(rng, max_length=length, lengths=length, task=rng.choice(selected))
        for _ in range(count)
    ]


def teaching_extraction_traces(expert):
    """Construct the full teaching core using the loaded expert's protocol.

    Explicit inputs preserve leading zeros and require no sampling from the
    extraction RNG. Expert traces supply initial instances for model rollouts.
    """
    from ..data.teaching import teaching_examples

    if expert.environment not in {"grid", "teleporting"}:
        raise ValueError("Extraction teaching inputs require a grid or universal expert")
    examples = teaching_examples(
        dict(
            environment=expert.environment,
            task=expert.task,
            radix=expert.radix,
            variant=expert.variant,
        )
    )
    rng = random.Random(0)  # Explicit operands consume no random draws.
    traces = []
    for first, second in examples:
        if expert.environment == "teleporting":
            trace = expert.sample(rng, max_length=len(second), task=first, operands=second)
        else:
            trace = expert.sample(
                rng, max_length=max(len(first), len(second)), operands=(first, second)
            )
        trace.metadata["extraction_group"] = "Teaching Set"
        traces.append(trace)
    return traces
