"""One training loop for four controllers, three experts, and both data sources."""

import hashlib
import json
import random
import signal
import sys
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml

from .checkpoint import (
    atomic_json,
    atomic_save,
    capture_rng,
    latest_checkpoint,
    read_checkpoint,
    restore_rng,
    rewind_history,
)
from .config import experiment_config, validate_config
from .data import BatchSampler, build_static_dataset, create_expert
from .data.datasets import dataset_spec
from .initialization import initialize_from_checkpoint
from .models import build_model
from .objectives import action_counts, action_loss, exponential_lr
from .tracking import WandbTracker


def choose_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is unavailable")
    return device


def construct_model(config, expert, device):
    # Model initialization does not advance the training RNG.
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(config["run"]["model_seed"])
        model = build_model(
            config["model"],
            input_size=expert.input_size,
            symbol_size=expert.symbol_size,
            motion_size=expert.motion_size,
            expert_states=expert.n_states,
        )
    return model.to(device)


def load_run(path, checkpoint="latest.pt", device="cpu"):
    """Reconstruct the saved architecture; never guess a model from tensor keys."""
    path = Path(path)
    if path.is_file():
        saved = read_checkpoint(path)
    elif checkpoint == "latest.pt" and (path / "final_model_checkpoint.pt").exists():
        saved = read_checkpoint(path / "final_model_checkpoint.pt")
    elif checkpoint == "latest.pt":
        _, saved = latest_checkpoint(path)
    else:
        saved = read_checkpoint(path / "checkpoints" / checkpoint)
    config = deepcopy(saved["config"])
    _check_expert(saved, dataset_spec(config["data"])["expert_digest"])
    expert = create_expert(config["data"])
    model = construct_model(config, expert, choose_device(device))
    model.load_state_dict(saved["model"])
    model.eval()
    return model, expert, config, saved


def _check_expert(checkpoint, digest):
    """Reject different expert policies even when model tensor shapes still fit."""
    if checkpoint.get("expert_digest") != digest:
        raise ValueError(
            "Checkpoint expert fingerprint is missing or differs from the current expert. "
            "Start a fresh run for the updated expert; use the original code to inspect old runs."
        )


@contextmanager
def _termination_request(enabled):
    """Defer SIGTERM until an optimizer update and its bookkeeping finish."""
    requested = [False] if enabled else None
    if enabled:
        previous = signal.signal(signal.SIGTERM, lambda *_: requested.__setitem__(0, True))
    try:
        yield requested
    finally:
        if enabled:
            signal.signal(signal.SIGTERM, previous if previous is not None else signal.SIG_DFL)


def train(config, *, resume=False, run_dir=None, stop_after=None, handle_signals=False):
    """Train or resume; stop_after is an absolute-step interruption hook for tests.

    It does not change the configured schedule/budget. resume="auto" starts an
    empty run or restores a valid checkpoint under the same exclusive run lock.
    initialization.checkpoint applies only to a fresh run, never to a retry.
    handle_signals saves an initial checkpoint and handles SIGTERM at update
    boundaries, exiting with status 143 so a job scheduler can retry the run.
    """
    if not isinstance(resume, bool) and resume != "auto":
        raise ValueError("resume must be false, true, or 'auto'")
    config = validate_config(deepcopy(config))
    expert_digest = dataset_spec(config["data"])["expert_digest"]
    identity = hashlib.sha256(
        json.dumps(
            {"config": experiment_config(config), "expert_digest": expert_digest}, sort_keys=True
        ).encode()
    ).hexdigest()[:12]
    run_dir = Path(
        run_dir
        or Path(config["run"]["output_dir"]) / f"{config['run']['name'] or 'run'}-{identity}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    # A process-level lock prevents two writers corrupting the same logical run.
    import fcntl

    with (run_dir / ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Another process is training in {run_dir}") from error
        # Read only after acquiring the lock: a previous writer may have saved
        # a newer checkpoint while this process was starting.
        checkpoint = None
        existing = any(path.name != ".lock" for path in run_dir.iterdir())
        if resume is True or (resume == "auto" and existing):
            _, checkpoint = latest_checkpoint(run_dir)
            if experiment_config(checkpoint["config"]) != experiment_config(config):
                raise ValueError(
                    "Resume requires the same scientific configuration and training budget"
                )
            _check_expert(checkpoint, expert_digest)
        elif existing:
            raise FileExistsError(
                f"Run directory already contains files: {run_dir}. Use --resume or a new run.name."
            )
        with _termination_request(handle_signals) as termination:
            return _train_locked(
                config, run_dir, checkpoint, stop_after, expert_digest, termination
            )


def _train_locked(config, run_dir, checkpoint, stop_after, expert_digest, termination=None):
    d, t, e, r = (config[key] for key in ("data", "training", "evaluation", "run"))
    device = choose_device(r["device"])
    random.seed(r["seed"])
    np.random.seed(r["seed"] % 2**32)
    torch.manual_seed(r["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(r["seed"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    sampler = random.Random(d["seed"])
    expert = create_expert(d)
    model = construct_model(config, expert, device)
    optimizer_class = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW}[t["optimizer"]]
    optimizer = optimizer_class(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    curriculum = t["arithmetic_curriculum"]
    length = curriculum["start_length"] if curriculum["enabled"] else d["max_length"]
    step = 0
    aggregate = {"loss": 0.0, "correct": 0, "tokens": 0, "updates": 0}
    initialization = None
    if checkpoint:
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        step = checkpoint["step"]
        length = checkpoint["curriculum_length"]
        aggregate = checkpoint["aggregate"]
        restore_rng(checkpoint["rng"], sampler)
        rewind_history(run_dir / "metrics.jsonl", step)
        initialization = checkpoint.get("initialization")
    else:
        initialization = initialize_from_checkpoint(config, model, optimizer, expert)
    traces = build_static_dataset(d, cache_dir=d["cache_dir"]) if d["source"] == "static" else None
    batches = BatchSampler(expert, sampler, t["batch_size"], traces=traces)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    atomic_json(
        {
            "format_version": 1,
            "experiment_id": run_dir.name,
            "expert_digest": expert_digest,
            "torch": torch.__version__,
            "device": str(device),
            "started_at": datetime.now(timezone.utc).isoformat(),
            "resumed_from_step": step,
            "initialization": initialization,
        },
        run_dir / "execution.json",
    )
    tracking_state = checkpoint.get("tracking") if checkpoint else None
    tracker = WandbTracker(config, run_dir, tracking_state, step) if r["wandb"]["enabled"] else None
    end = min(t["steps"], stop_after) if stop_after is not None else t["steps"]
    model.train()

    def save(numbered=True):
        state = {
            "format_version": 1,
            "config": config,
            "expert_digest": expert_digest,
            "step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "curriculum_length": length,
            "aggregate": aggregate,
            "rng": capture_rng(sampler),
            "initialization": initialization,
            "tracking": tracker.state_dict() if tracker else tracking_state,
        }
        if numbered:
            atomic_save(state, run_dir / "checkpoints" / f"step_{step:010d}.pt")
        atomic_save(state, run_dir / "checkpoints" / "latest.pt")

    def tracking_call(function, *args):
        # Connecting, media rendering and logging must not advance training RNGs.
        rng = capture_rng(sampler)
        try:
            function(*args)
        finally:
            restore_rng(rng, sampler)

    def evaluate_and_log(path):
        evaluation_id = hashlib.sha256(json.dumps(e, sort_keys=True).encode()).hexdigest()
        key = f"evaluation:{step}:{evaluation_id}"
        if tracker and key in tracker.keys:
            return  # Already committed; retries must not append duplicate evaluation rows.
        from .evaluation import evaluate

        result = evaluate(model, expert, config, progress_dir=run_dir / "evaluation_progress")
        result["step"] = step
        atomic_json(result, path)
        if tracker:
            tracker.record("evaluation", result, key)
            save(numbered=False)
            tracking_call(tracker.publish)

    def exit_if_requested():
        if termination is not None and termination[0]:
            save()
            print(f"Saved step {step} after SIGTERM; exiting for scheduler recovery", flush=True)
            raise SystemExit(128 + signal.SIGTERM)

    try:
        if (termination is not None or tracker) and checkpoint is None:
            save()  # Step zero permits recovery before the first periodic save.
        if tracker:
            # Commit identity and history before any network operation, including
            # when tracking starts on an older, previously untracked checkpoint.
            save(numbered=False)
            tracking_call(tracker.connect, initialization)
            save(numbered=False)
            tracking_call(tracker.publish)
        if checkpoint and e["every"] and step and step % e["every"] == 0 and step < t["steps"]:
            evaluate_and_log(run_dir / f"evaluation_{step:010d}.json")
        while step < end:
            exit_if_requested()
            batch = batches.sample(length).to(device)
            lr = exponential_lr(
                step,
                total_steps=t["steps"],
                warmup_steps=t["warmup_steps"],
                constant_steps=t["constant_steps"],
                peak=t["lr"],
                floor=t["final_lr"],
            )
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            symbol, motion, _ = model(batch.inputs, mask=batch.mask)
            loss = action_loss(
                symbol,
                motion,
                batch.symbols,
                batch.motions,
                batch.mask,
                inputs=batch.inputs,
                inverse_freq_alpha=t["inverse_freq_alpha"],
                loss=t["loss"],
                output_kind=model.output_kind,
            )
            objective = loss + model.regularization_loss()
            if not torch.isfinite(objective):
                raise FloatingPointError(
                    f"Nonfinite loss at step {step + 1}; latest checkpoint remains intact"
                )
            objective.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), t["grad_clip"], error_if_nonfinite=True
            )
            optimizer.step()
            correct, tokens, _ = action_counts(
                symbol.detach(), motion.detach(), batch.symbols, batch.motions, batch.mask
            )
            step += 1
            aggregate["loss"] += float(loss.detach())
            aggregate["correct"] += correct
            aggregate["tokens"] += tokens
            aggregate["updates"] += 1
            if step % t["log_every"] == 0 or step == t["steps"]:
                accuracy = aggregate["correct"] / aggregate["tokens"]
                row = {
                    "step": step,
                    "loss": aggregate["loss"] / aggregate["updates"],
                    "accuracy": accuracy,
                    "lr": lr,
                }
                if curriculum["enabled"]:
                    row["curriculum_length"] = length
                with (run_dir / "metrics.jsonl").open("a") as stream:
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
                    stream.flush()
                print(
                    f"step={step}/{t['steps']} loss={row['loss']:.6g} accuracy={accuracy:.4%} lr={lr:.3g}"
                )
                if curriculum["enabled"] and accuracy >= curriculum["accuracy_threshold"]:
                    length = min(length + 1, d["max_length"])
                aggregate = {"loss": 0.0, "correct": 0, "tokens": 0, "updates": 0}
                if tracker:
                    tracker.record("training", row, f"training:{step}")
                    save(numbered=False)
                    tracking_call(tracker.publish)
            exit_if_requested()
            if step % t["checkpoint_every"] == 0:
                save()
            if e["every"] and step % e["every"] == 0 and step < t["steps"]:
                evaluate_and_log(run_dir / f"evaluation_{step:010d}.json")
        save()
        exit_if_requested()
        if step == t["steps"] and e["final"]:
            evaluate_and_log(run_dir / "evaluation.json")
        exit_if_requested()
        return run_dir
    finally:
        if tracker:
            tracker.finish(exit_code=0 if sys.exc_info()[0] is None else 1)
