"""A single strict configuration schema with small, composable recipe overlays."""

import math
from copy import deepcopy
from importlib.resources import files
from pathlib import Path

import yaml


def _merge(base, update, prefix=""):
    if not isinstance(update, dict):
        raise ValueError(f"{prefix or 'configuration'} must be a mapping")
    for key, value in update.items():
        path = f"{prefix}.{key}" if prefix else key
        if key not in base:
            raise ValueError(f"Unknown configuration field: {path}")
        if isinstance(base[key], dict):
            _merge(base[key], value, path)
        else:
            base[key] = value


def load_config(path=None, overrides=()):
    """Load package defaults, an optional YAML overlay, and strict dotted overrides."""
    config = yaml.safe_load(files(__package__).joinpath("config.yaml").read_text())
    if path:
        with Path(path).open() as stream:
            _merge(config, yaml.safe_load(stream) or {})
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Expected key=value, got {override!r}")
        key, value = override.split("=", 1)
        parts = key.split(".")
        target = config
        for part in parts[:-1]:
            if part not in target or not isinstance(target[part], dict):
                raise ValueError(f"Unknown configuration field: {key}")
            target = target[part]
        if parts[-1] not in target:
            raise ValueError(f"Unknown configuration field: {key}")
        if isinstance(target[parts[-1]], dict):
            _merge(target[parts[-1]], yaml.safe_load(value), key)
        else:
            target[parts[-1]] = yaml.safe_load(value)
    validate_config(config)
    return config


def validate_config(config):
    """Fail before dataset creation or a long run when a setting is inconsistent."""
    # Older checkpoints predate initialization settings. Keep those runs loadable.
    initialization = config.setdefault(
        "initialization", {"checkpoint": None, "restore_optimizer": False}
    )
    if not isinstance(initialization, dict):
        raise ValueError("initialization must be a mapping")
    checkpoint = initialization.get("checkpoint")
    if checkpoint is not None and (not isinstance(checkpoint, str) or not checkpoint.strip()):
        raise ValueError("initialization.checkpoint must be a nonempty path string or null")
    if not isinstance(initialization.get("restore_optimizer"), bool):
        raise ValueError("initialization.restore_optimizer must be a boolean")
    if initialization["restore_optimizer"] and checkpoint is None:
        raise ValueError("initialization.restore_optimizer requires initialization.checkpoint")
    d, m, t, e, r = (config[k] for k in ("data", "model", "training", "evaluation", "run"))
    # Checkpoints saved before evaluation scan options use the memory-bounded default.
    e.setdefault("scan", "sequential")
    # Configurations saved before the peak hold existed use none.
    t.setdefault("constant_steps", 0)
    if not isinstance(e["scan"], str):
        raise ValueError("evaluation.scan must be sequential or parallel")
    from .data.teaching import validate_teaching

    choices = {
        "data.environment": (d["environment"], {"grid", "tm", "teleporting"}),
        "data.variant": (d["variant"], {"default", "new"}),
        "data.source": (d["source"], {"static", "online"}),
        "data.batch_lengths": (d["batch_lengths"], {"fixed", "mixed"}),
        "model.name": (m["name"], {"DFST", "DFST_Softmax", "RNN", "GRU"}),
        "model.scan": (m["scan"], {"parallel", "sequential"}),
        "training.loss": (t["loss"], {"mse", "cross_entropy"}),
        "training.optimizer": (t["optimizer"], {"adam", "adamw"}),
        "evaluation.hard_cases": (e["hard_cases"], {"none", "diagonal", "all"}),
        "evaluation.search": (e["search"], {"binary", "exhaustive"}),
        "evaluation.scan": (e["scan"], {"sequential", "parallel"}),
    }
    for name, (value, allowed) in choices.items():
        if value not in allowed:
            raise ValueError(f"{name} must be one of {sorted(allowed)}, got {value!r}")
    for name, value in (
        ("data.teaching_set", d["teaching_set"]),
        ("training.arithmetic_curriculum.enabled", t["arithmetic_curriculum"]["enabled"]),
        ("evaluation.final", e["final"]),
        ("run.wandb.enabled", r["wandb"]["enabled"]),
    ):
        if not isinstance(value, bool):
            raise ValueError(f"{name} must be a boolean")
    if m["init"]["learnable_state"] is not None and not isinstance(
        m["init"]["learnable_state"], bool
    ):
        raise ValueError("model.init.learnable_state must be a boolean or null")
    for section, keys in (
        ("data", ["max_length", "size"]),
        ("model", ["num_layers"]),
        ("training", ["steps", "batch_size", "log_every", "checkpoint_every"]),
        ("evaluation", ["max_length", "samples", "batch_size", "chunk_size"]),
    ):
        for key in keys:
            value = config[section][key]
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{section}.{key} must be a positive integer")
    limit = d["max_steps"]
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit < 1):
        raise ValueError("data.max_steps must be null or a positive integer")
    for section, key in (
        ("data", "seed"),
        ("run", "seed"),
        ("run", "model_seed"),
        ("evaluation", "seed"),
        ("evaluation", "every"),
        ("training", "warmup_steps"),
        ("training", "constant_steps"),
    ):
        value = config[section][key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{section}.{key} must be a nonnegative integer")
    snapshot = e.get("snapshot", {"enabled": True, "m": 100, "d": 5})
    if not isinstance(snapshot["enabled"], bool):
        raise ValueError("evaluation.snapshot.enabled must be a boolean")
    for key in ("m", "d"):
        value = snapshot[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"evaluation.snapshot.{key} must be a positive integer")
    if snapshot["d"] > snapshot["m"]:
        raise ValueError("evaluation.snapshot.d must not exceed evaluation.snapshot.m")
    for key in ("lr", "final_lr", "grad_clip"):
        if not isinstance(t[key], (int, float)) or not math.isfinite(t[key]) or t[key] <= 0:
            raise ValueError(f"training.{key} must be finite and positive")
    if not 0 <= t["warmup_steps"] < t["steps"]:
        raise ValueError("training.warmup_steps must be smaller than training.steps")
    if t["warmup_steps"] + t["constant_steps"] >= t["steps"]:
        raise ValueError(
            "training.warmup_steps + training.constant_steps must be smaller than training.steps"
        )
    if t["final_lr"] > t["lr"]:
        raise ValueError("training.final_lr cannot exceed the peak training.lr")
    if not math.isfinite(t["weight_decay"]) or t["weight_decay"] < 0:
        raise ValueError("training.weight_decay must be finite and nonnegative")
    if (
        isinstance(t["inverse_freq_alpha"], bool)
        or not isinstance(t["inverse_freq_alpha"], (int, float))
        or not math.isfinite(t["inverse_freq_alpha"])
        or t["inverse_freq_alpha"] < 0
    ):
        raise ValueError("training.inverse_freq_alpha must be finite and nonnegative")
    if not math.isfinite(m["reg_lambda"]) or m["reg_lambda"] < 0:
        raise ValueError("model.reg_lambda must be finite and nonnegative")
    if m["reg_lambda"] and m["name"] != "DFST_Softmax":
        raise ValueError("model.reg_lambda applies only to DFST_Softmax")
    for key in ("hidden_size", "embed_dim"):
        if m[key] is not None and (
            isinstance(m[key], bool) or not isinstance(m[key], int) or m[key] < 1
        ):
            raise ValueError(f"model.{key} must be null or a positive integer")
    if m["name"] in {"DFST", "DFST_Softmax"} and m["num_layers"] != 1:
        raise ValueError(f"{m['name']} supports one layer; num_layers applies to RNN/GRU")
    c = t["arithmetic_curriculum"]
    if not 0 < c["accuracy_threshold"] <= 1:
        raise ValueError("Curriculum accuracy_threshold must be in (0, 1]")
    if not isinstance(c["start_length"], int) or not 1 <= c["start_length"] <= d["max_length"]:
        raise ValueError("Curriculum start_length must be in [1, data.max_length]")
    if d["environment"] == "teleporting":
        if c["enabled"]:
            raise ValueError("arithmetic_curriculum is unavailable for teleporting agents")
        if not d["train_tasks"] or not d["eval_tasks"]:
            raise ValueError("Teleporting agents require nonempty train_tasks and eval_tasks")
    else:
        if d["task"] not in {"addition", "multiplication"}:
            raise ValueError("Arithmetic task must be addition or multiplication")
        if not isinstance(d["radix"], int) or not 2 <= d["radix"] <= 10:
            raise ValueError("The arithmetic encodings support radix 2..10")
        if d["source"] == "static" and d["size"] < d["radix"] ** 2:
            raise ValueError("Static arithmetic data must include all radix² single-digit pairs")
    validate_teaching(d)
    if r["name"] is not None and (
        not isinstance(r["name"], str)
        or not r["name"]
        or Path(r["name"]).name != r["name"]
        or r["name"] in {".", ".."}
    ):
        raise ValueError("run.name must be null or a single nonempty directory name")
    return config


def experiment_config(config):
    """Scientific identity, excluding output locations, logging, and evaluation frequency."""
    result = deepcopy(config)
    # Null and explicit 64 represent the same recurrent default.
    if result["model"]["name"] not in {"DFST", "DFST_Softmax"}:
        if result["model"]["hidden_size"] is None:
            result["model"]["hidden_size"] = 64
    # Initialization without a source does not affect scientific identity.
    if result.get("initialization", {}).get("checkpoint") is None:
        result.pop("initialization", None)
    result["run"] = {key: result["run"][key] for key in ("seed", "model_seed")}
    result["data"].pop("cache_dir", None)
    if (
        result["data"].get("environment") != "grid"
        or result["data"].get("variant", "default") == "default"
    ):
        result["data"].pop("variant", None)
    if not result["data"].get("teaching_set"):
        result["data"].pop("teaching_set", None)
    # Without a peak hold the schedule, and so the identity, is unchanged.
    if result["training"].get("constant_steps", 0) == 0:
        result["training"].pop("constant_steps", None)
    if result["training"].get("inverse_freq_alpha", 0.0) == 0.0:
        result["training"].pop("inverse_freq_alpha", None)
    result.pop("evaluation")
    # Curriculum advancement uses the accuracy window at log_every boundaries.
    # Changing that window changes the training data, not just presentation.
    if not result["training"]["arithmetic_curriculum"]["enabled"]:
        result["training"].pop("log_every", None)
    result["training"].pop("checkpoint_every", None)
    return result
