"""Initialize a separate experiment from a trusted DEL checkpoint."""

import hashlib
from copy import deepcopy
from pathlib import Path

from .checkpoint import read_checkpoint
from .data import create_expert
from .data.datasets import dataset_spec
from .models import resolve_hidden_size


def _architecture(config, expert):
    """Compare effective structure, excluding initialization and training choices."""
    model = config["model"]
    dfst = model["name"] in {"DFST", "DFST_Softmax"}
    learnable = model["init"]["learnable_state"]
    structure = {
        "name": model["name"],
        "embed_dim": model["embed_dim"],
        "learnable_state": dfst if learnable is None else learnable,
    }
    structure["hidden_size"] = resolve_hidden_size(model, expert.n_states)
    if not dfst:
        structure["num_layers"] = model["num_layers"]
    return structure


def initialize_from_checkpoint(config, model, optimizer, expert):
    """Load weights and optionally optimizer accumulators, never training progress.

    Only call for an empty destination. A resumed child restores its own checkpoint
    and provenance without reopening the source (which may have moved or changed).
    """
    options = config["initialization"]
    if options["checkpoint"] is None:
        return None
    path = Path(options["checkpoint"]).expanduser().resolve(strict=True)
    # Hash and deserialize the same open file: atomic replacement of latest.pt
    # must not give provenance for different bytes than the loaded checkpoint.
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
        stream.seek(0)
        saved = read_checkpoint(stream)

    source = saved["config"]
    if saved.get("expert_digest") != dataset_spec(source["data"])["expert_digest"]:
        raise ValueError("Initialization checkpoint expert fingerprint is missing or incompatible")
    source_expert = create_expert(source["data"])
    if any(
        getattr(source_expert, key) != getattr(expert, key)
        for key in ("environment", "vocab", "motion_vocab")
    ):
        raise ValueError("Checkpoint initialization requires matching input/output encodings")
    previous = _architecture(source, source_expert)
    current = _architecture(config, expert)
    if previous != current:
        differences = sorted(
            key for key in previous.keys() | current.keys() if previous.get(key) != current.get(key)
        )
        raise ValueError(
            "Checkpoint initialization requires a matching model architecture; changed: "
            + ", ".join(differences)
        )
    if options["restore_optimizer"] and (
        source["training"]["optimizer"] != config["training"]["optimizer"]
    ):
        raise ValueError("Restoring optimizer state requires the same optimizer type")

    # Strict loading includes embeddings and the initial hidden-state tensor.
    model.load_state_dict(saved["model"], strict=True)
    if options["restore_optimizer"]:
        # load_state_dict also restores parameter-group options. Retain only the
        # accumulators and per-parameter step counters; new settings take priority.
        groups = [
            deepcopy({key: value for key, value in group.items() if key != "params"})
            for group in optimizer.param_groups
        ]
        optimizer.load_state_dict(saved["optimizer"])
        for group, fresh in zip(optimizer.param_groups, groups):
            parameters = group["params"]
            group.clear()
            group.update(fresh, params=parameters)

    return {
        "checkpoint": str(path),
        "sha256": digest.hexdigest(),
        "source_step": saved["step"],
        "source_run_name": source["run"]["name"],
        "restore_optimizer": options["restore_optimizer"],
    }
