"""Saved evaluation data for paper figures."""

import hashlib
import json
from pathlib import Path

import numpy as np
import yaml


def load_latest_evaluation(root, run_path):
    """Read the highest recorded evaluation step without loading model weights.

    At the same step, prefer the training evaluation over an analysis evaluation.
    Modification times and checkpoint filenames do not determine recency.
    """
    import warnings

    run = Path(run_path).expanduser()
    if not run.is_absolute():
        run = Path(root) / run
    run = run.resolve()
    config = yaml.safe_load((run / "config.yaml").read_text())
    candidates = [(p, 0) for p in sorted((run / "analysis/evaluations").glob("step_*.json"))]
    candidates += [(p, 1) for p in sorted(run.glob("evaluation_*.json"))]
    if (run / "evaluation.json").exists():
        candidates.append((run / "evaluation.json", 2))
    selected = None
    for path, priority in candidates:
        try:
            data = json.loads(path.read_text())
            step = data.get("step")
            if (
                isinstance(step, bool)
                or not isinstance(step, int)
                or step < 0
                or not isinstance(data.get("tasks"), dict)
                or not data["tasks"]
            ):
                raise ValueError("Expected a nonnegative integer step and task results")
        except (OSError, ValueError, AttributeError) as error:
            warnings.warn(f"Skipping unreadable evaluation {path}: {error}", stacklevel=2)
            continue
        if selected is None or (step, priority) > selected[:2]:
            selected = (step, priority, data, path)
    step, result, source = (
        (None, None, None) if selected is None else (selected[0], selected[2], str(selected[3]))
    )
    checkpoint = None if step is None else run / "checkpoints" / f"step_{step:010d}.pt"
    return dict(
        run=str(run),
        checkpoint=str(checkpoint) if checkpoint else None,
        step=step,
        config=config,
        evaluation=result,
        evaluation_path=source,
        sha256=result.get("source_sha256") if result else None,
        checkpoint_available=checkpoint is not None and checkpoint.exists(),
        selection="latest_evaluation",
        final=step == config["training"]["steps"],
    )


def load_measurement(root, run_path, step, analysis_dir=None):
    """Load exact-step scores; disposable figure outputs are never input storage.

    An explicit latest selection captures bytes once so evaluation can use the
    same weights even if training replaces latest.pt. Saved scores can be
    plotted without retaining the checkpoint itself.
    """
    import io

    import torch

    run = Path(run_path)
    if not run.is_absolute():
        run = Path(root) / run
    latest = step == "latest"
    if not latest and (isinstance(step, bool) or not isinstance(step, int) or step < 0):
        raise ValueError("Checkpoint step must be a nonnegative integer or 'latest'")
    config = yaml.safe_load((run / "config.yaml").read_text())
    checkpoint = run / "checkpoints" / ("latest.pt" if latest else f"step_{step:010d}.pt")
    payload = checkpoint.read_bytes() if checkpoint.exists() else None
    if latest:
        if payload is None:
            raise FileNotFoundError(f"No latest checkpoint in {run / 'checkpoints'}")
        step = int(torch.load(io.BytesIO(payload), map_location="cpu", weights_only=False)["step"])
    digest = hashlib.sha256(payload).hexdigest() if payload is not None else None
    candidates = [
        run / f"evaluation_{step:010d}.json",
        run / "evaluation.json",
        run / "analysis/evaluations" / f"step_{step:010d}.json",
    ]
    result, source = None, None
    for path in candidates:
        if path.exists():
            data = json.loads(path.read_text())
            if data.get("step") == step and "tasks" in data:
                result, source = data, str(path)
                break
    if result is not None and digest is not None and result.get("source_sha256", digest) != digest:
        raise ValueError("Evaluation checkpoint hash does not match the configured checkpoint")
    if digest is None and result is not None:
        digest = result.get("source_sha256")
    return dict(
        run=str(run),
        checkpoint=str(checkpoint),
        step=step,
        config=config,
        evaluation=result,
        evaluation_path=source,
        sha256=digest,
        checkpoint_available=payload is not None,
        _checkpoint_bytes=payload,
        final=step == config["training"]["steps"],
    )


def panel_data(records, xs, ys, *, softmax=False, metric="ulg", title=""):
    values = np.full((len(ys), len(xs)), np.nan)
    capped = np.zeros(values.shape, dtype=bool)
    seen = set()
    for record in records:
        config = record["config"]
        x = config["model"]["hidden_size"] if softmax else config["training"]["warmup_steps"]
        y = config["data"]["size"] if softmax else config["training"]["steps"]
        if (x, y) in seen or x not in xs or y not in ys:
            raise ValueError(f"Duplicate or unexpected sweep coordinate {(x, y)}")
        seen.add((x, y))
        evaluation = record["evaluation"]
        if evaluation is None:
            continue
        task = config["data"]["task"] + str(config["data"]["radix"])
        outcome = evaluation["tasks"][task].get(metric)
        if outcome is None:
            continue
        i, j = ys.index(y), xs.index(x)
        values[i, j] = outcome["length"]
        capped[i, j] = outcome.get("at_cap", False)
    if len(seen) != len(xs) * len(ys):
        raise ValueError("Every sweep coordinate needs one configured run")

    def tick(value):
        return (
            f"{value // 1000000}M"
            if value >= 1000000
            else f"{value // 1000}k"
            if value >= 1000
            else str(value)
        )

    return dict(
        values=values,
        at_cap=capped,
        xlabels=[tick(v) for v in xs],
        ylabels=[tick(v) for v in ys],
        title=title,
        xlabel="Hidden Dimensions" if softmax else "Warmup Steps",
        ylabel="Dataset Size" if softmax else "Training Steps",
    )


def provenance_rows(groups):
    rows = []
    for group, records in groups.items():
        for record in records:
            base = {
                k: record[k]
                for k in ("run", "checkpoint", "step", "sha256", "final", "evaluation_path")
            }
            evaluation = record["evaluation"]
            if evaluation is None:
                rows.append(
                    dict(
                        base,
                        group=group,
                        status=(
                            "No Recorded Evaluation"
                            if record.get("selection") == "latest_evaluation"
                            else "Missing Evaluation"
                            if record["checkpoint_available"]
                            else "Missing Checkpoint and Evaluation"
                        ),
                    )
                )
                continue
            for task, metrics in evaluation["tasks"].items():
                for metric in ("ulg", "rlg", "generalization"):
                    if metric in metrics:
                        score = metrics[metric]
                        rows.append(
                            dict(
                                base,
                                group=group,
                                task=task,
                                metric=metric,
                                length=score["length"],
                                cap=score.get("cap"),
                                at_cap=score.get("at_cap", False),
                                scan=evaluation.get("scan"),
                                samples=evaluation.get("random_samples"),
                                hard_cases=metrics.get("hard_cases"),
                                status="Final" if record["final"] else "Interim",
                            )
                        )
    return rows


def count_trainable_parameters(record):
    """Count learned scalars from the saved architecture, without reading weights.

    The count excludes fixed one-hot embeddings and fixed initial states.
    """
    import torch

    from ..data import create_expert
    from ..models import build_model

    config = record["config"]
    expert = create_expert(config["data"])
    with torch.random.fork_rng(devices=[]):
        model = build_model(
            config["model"],
            expert.input_size,
            expert.symbol_size,
            expert.motion_size,
            expert.n_states,
        )
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
