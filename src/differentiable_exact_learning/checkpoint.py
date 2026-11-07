"""Atomic checkpoints and complete random-generator state for exact continuation."""

import json
import os
import pickle
import random
import tempfile
from pathlib import Path

import numpy as np
import torch


def atomic_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            torch.save(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_json(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def capture_rng(sampler):
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "sampler": sampler.getstate(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng(state, sampler):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    sampler.setstate(state["sampler"])
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([item.cpu() for item in state["cuda"]])


def read_checkpoint(path):
    """Load a trusted local checkpoint. PyTorch checkpoints must not be untrusted input."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("format_version") != 1:
        raise ValueError(
            "Unsupported checkpoint format; legacy checkpoints need explicit conversion"
        )
    return checkpoint


def latest_checkpoint(run_dir):
    """Try latest then fall back to the newest readable periodic snapshot."""
    root = Path(run_dir) / "checkpoints"
    candidates = [root / "latest.pt", *sorted(root.glob("step_*.pt"), reverse=True)]
    failures = []
    for path in candidates:
        if not path.exists():
            continue
        try:
            return path, read_checkpoint(path)
        except (
            OSError,
            RuntimeError,
            EOFError,
            ValueError,
            KeyError,
            pickle.UnpicklingError,
        ) as error:
            failures.append(f"{path.name}: {error}")
    if failures:
        raise RuntimeError(
            "No readable checkpoint; refusing to restart silently:\n" + "\n".join(failures)
        )
    raise FileNotFoundError(f"No checkpoint found in {root}")


def rewind_history(path, step):
    """Discard locally logged progress newer than the committed checkpoint."""
    path = Path(path)
    if not path.exists():
        return
    rows = []
    for line in path.read_text().splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue  # A preemption may leave a partial final line.
        if row["step"] <= step:
            rows.append(row)
    temporary = path.with_suffix(".jsonl.tmp")
    temporary.write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in rows))
    os.replace(temporary, path)
