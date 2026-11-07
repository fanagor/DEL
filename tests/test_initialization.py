"""Checkpoint forks start new experiments; retries continue the fork itself."""

import hashlib
import json
import os
import random
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest
import torch
import yaml
from test_training import small_config

from differentiable_exact_learning.checkpoint import atomic_save, read_checkpoint
from differentiable_exact_learning.config import experiment_config, load_config
from differentiable_exact_learning.training import train


def fork_config(source, checkpoint, restore_optimizer=False):
    config = deepcopy(source)
    config["run"]["name"] = "child"
    config["initialization"] = {
        "checkpoint": str(checkpoint),
        "restore_optimizer": restore_optimizer,
    }
    return config


@pytest.mark.parametrize("model", ["DFST", "DFST_Softmax", "RNN", "GRU"])
@pytest.mark.parametrize("embed_dim", [None, 3])
@pytest.mark.parametrize("restore_optimizer", [False, True])
def test_fork_loads_exact_weights_and_starts_new_training(
    tmp_path, model, embed_dim, restore_optimizer
):
    config = small_config(tmp_path, model)
    config["model"]["embed_dim"] = embed_dim
    source = train(config, stop_after=3)
    checkpoint = source / "checkpoints/latest.pt"
    before = checkpoint.read_bytes()
    saved = read_checkpoint(checkpoint)
    child = fork_config(config, checkpoint, restore_optimizer)
    child["run"].update(seed=123, model_seed=456)
    child["data"].update(seed=789, max_length=2, batch_lengths="mixed")
    child["training"].update(
        steps=5,
        warmup_steps=2,
        lr=0.002,
        final_lr=0.0001,
        weight_decay=0.1,
        batch_size=3,
        log_every=1,
        loss="cross_entropy",
    )
    child["training"]["arithmetic_curriculum"]["enabled"] = True
    path = train(child, stop_after=0)
    initial = read_checkpoint(path / "checkpoints/latest.pt")
    assert initial["step"] == 0
    assert initial["curriculum_length"] == 1
    assert initial["aggregate"] == {"loss": 0.0, "correct": 0, "tokens": 0, "updates": 0}
    assert initial["rng"]["sampler"] == random.Random(789).getstate()
    assert initial["rng"]["python"] == random.Random(123).getstate()
    assert torch.equal(initial["rng"]["torch"], torch.Generator().manual_seed(123).get_state())
    assert not (path / "metrics.jsonl").exists()
    for key in saved["model"]:
        torch.testing.assert_close(initial["model"][key], saved["model"][key], rtol=0, atol=0)
    states = initial["optimizer"]["state"]
    if restore_optimizer:
        assert states.keys() == saved["optimizer"]["state"].keys()
        for parameter, state in states.items():
            for key, value in state.items():
                torch.testing.assert_close(
                    value, saved["optimizer"]["state"][parameter][key], rtol=0, atol=0
                )
    else:
        assert states == {}
    group = initial["optimizer"]["param_groups"][0]
    assert group["lr"] == 0.002
    assert group["weight_decay"] == 0.1
    provenance = initial["initialization"]
    assert provenance["sha256"] == hashlib.sha256(before).hexdigest()
    assert provenance["source_step"] == 3
    assert provenance["checkpoint"] == str(checkpoint.resolve())
    assert provenance["restore_optimizer"] == restore_optimizer
    assert json.loads((path / "execution.json").read_text())["initialization"] == provenance
    train(child, resume=True, run_dir=path, stop_after=1)
    updated = read_checkpoint(path / "checkpoints/latest.pt")
    assert updated["step"] == 1
    assert all(
        int(state["step"]) == (4 if restore_optimizer else 1)
        for state in updated["optimizer"]["state"].values()
    )
    row = json.loads((path / "metrics.jsonl").read_text().splitlines()[0])
    assert row["lr"] == pytest.approx(0.0001)
    assert row["step"] == 1
    assert checkpoint.read_bytes() == before


@pytest.mark.parametrize("restore_optimizer", [False, True])
@pytest.mark.parametrize("model", ["DFST", "GRU"])
def test_retry_continues_child_without_source_checkpoint(tmp_path, restore_optimizer, model):
    config = small_config(tmp_path, model)
    source = train(config) / "checkpoints/latest.pt"
    child = fork_config(config, source, restore_optimizer)
    child["training"].update(steps=6, warmup_steps=3, lr=0.003)
    full = train(child, run_dir=tmp_path / "full")
    split = train(child, run_dir=tmp_path / "split", stop_after=2, handle_signals=True)
    source.unlink()
    train(child, run_dir=split, resume="auto", handle_signals=True)
    a = read_checkpoint(full / "checkpoints/latest.pt")
    b = read_checkpoint(split / "checkpoints/latest.pt")
    assert a["step"] == b["step"] == 6
    assert a["initialization"] == b["initialization"]
    for key in a["model"]:
        torch.testing.assert_close(a["model"][key], b["model"][key], rtol=0, atol=0)
    for parameter, state in a["optimizer"]["state"].items():
        for key, value in state.items():
            torch.testing.assert_close(
                value, b["optimizer"]["state"][parameter][key], rtol=0, atol=0
            )
    assert (full / "metrics.jsonl").read_bytes() == (split / "metrics.jsonl").read_bytes()
    assert a["rng"]["sampler"] == b["rng"]["sampler"]
    changed = deepcopy(child)
    changed["training"]["steps"] += 1
    with pytest.raises(ValueError, match="same scientific configuration"):
        train(changed, run_dir=split, resume="auto")


@pytest.mark.parametrize(
    "change",
    [
        {"name": "DFST_Softmax"},
        {"hidden_size": 5},
        {"embed_dim": 4},
        {"init": {"learnable_state": False}},
    ],
)
def test_incompatible_architecture_fails_before_saving_destination(tmp_path, change):
    config = small_config(tmp_path)
    source = train(config) / "checkpoints/latest.pt"
    child = fork_config(config, source)
    for key, value in change.items():
        if isinstance(value, dict):
            child["model"][key].update(value)
        else:
            child["model"][key] = value
    destination = tmp_path / "child"
    with pytest.raises(ValueError, match="matching model architecture"):
        train(child, run_dir=destination)
    assert not (destination / "config.yaml").exists()
    assert not (destination / "checkpoints").exists()


def test_encodings_and_expert_fingerprint_are_checked(tmp_path):
    config = small_config(tmp_path)
    source = train(config) / "checkpoints/latest.pt"
    child = fork_config(config, source)
    child["data"]["environment"] = "tm"
    with pytest.raises(ValueError, match="input/output encodings"):
        train(child)
    child["data"]["environment"] = "grid"
    broken = read_checkpoint(source)
    broken["expert_digest"] = "outdated"
    atomic_save(broken, source)
    with pytest.raises(ValueError, match="expert fingerprint"):
        train(child)


def test_new_tp_task_mixture_can_reuse_compatible_weights(tmp_path):
    config = small_config(tmp_path, environment="teleporting")
    source = train(config) / "checkpoints/latest.pt"
    child = fork_config(config, source)
    child["data"].update(train_tasks=["reverse"], eval_tasks=["reverse", "copy"])
    path = train(child)
    assert read_checkpoint(path / "checkpoints/latest.pt")["step"] == 4


@pytest.mark.parametrize("model", ["DFST", "DFST_Softmax"])
def test_new_curriculum_scan_and_regularization_settings_take_effect(tmp_path, model):
    config = small_config(tmp_path, model)
    config["data"]["max_length"] = 2
    config["training"]["arithmetic_curriculum"].update(enabled=True, start_length=2)
    source = train(config) / "checkpoints/latest.pt"
    assert read_checkpoint(source)["curriculum_length"] == 2
    child = fork_config(config, source, restore_optimizer=True)
    child["training"]["arithmetic_curriculum"]["start_length"] = 1
    child["model"]["scan"] = "sequential"
    if model == "DFST_Softmax":
        child["model"]["reg_lambda"] = 0.02
    path = train(child, stop_after=0)
    initial = read_checkpoint(path / "checkpoints/latest.pt")
    assert initial["curriculum_length"] == 1
    assert initial["config"]["model"] == child["model"]
    train(child, resume=True, run_dir=path)


@pytest.mark.parametrize("optimizer", ["adam", "adamw"])
def test_optimizer_type_must_match_only_when_restoring(tmp_path, optimizer):
    config = small_config(tmp_path)
    config["training"].update(optimizer=optimizer, weight_decay=0.2)
    source = train(config) / "checkpoints/latest.pt"
    child = fork_config(config, source, restore_optimizer=True)
    child["training"]["weight_decay"] = 0.01
    same = train(child, stop_after=0)
    state = read_checkpoint(same / "checkpoints/latest.pt")
    assert state["optimizer"]["param_groups"][0]["weight_decay"] == 0.01
    child["training"]["optimizer"] = "adamw" if optimizer == "adam" else "adam"
    with pytest.raises(ValueError, match="same optimizer type"):
        train(child)
    child["initialization"]["restore_optimizer"] = False
    path = train(child)
    assert read_checkpoint(path / "checkpoints/latest.pt")["step"] == 4


@pytest.mark.parametrize(
    "overrides",
    [
        ["initialization.restore_optimizer=true"],
        ["initialization.checkpoint=''"],
        ["initialization.checkpoint=42"],
        ["initialization.restore_optimizer=yes-please"],
    ],
)
def test_invalid_initialization_config(overrides):
    with pytest.raises(ValueError, match="initialization"):
        load_config(overrides=overrides)


def test_old_checkpoints_keep_identity_and_support_resume_and_fork(tmp_path):
    config = small_config(tmp_path)
    old_config = deepcopy(config)
    old_config.pop("initialization")
    assert experiment_config(old_config) == experiment_config(config)
    path = train(config, stop_after=2)
    source = path / "checkpoints/latest.pt"
    saved = read_checkpoint(source)
    saved["config"].pop("initialization")
    saved.pop("initialization")
    atomic_save(saved, source)
    train(config, resume=True, run_dir=path)
    # Fork the actual pre-feature snapshot, too.
    old_snapshot = tmp_path / "old.pt"
    atomic_save(saved, old_snapshot)
    child = fork_config(config, old_snapshot, restore_optimizer=True)
    destination = train(child, stop_after=0)
    assert read_checkpoint(destination / "checkpoints/latest.pt")["step"] == 0
    weights_only = deepcopy(child)
    weights_only["initialization"]["restore_optimizer"] = False
    assert experiment_config(child) != experiment_config(weights_only)
    assert experiment_config(child) != experiment_config(config)


def test_wandb_records_separate_run_and_source_provenance(tmp_path, fake_wandb):
    config = small_config(tmp_path)
    config["run"]["wandb"]["enabled"] = True
    source_run = train(config)
    child = fork_config(config, source_run / "checkpoints/latest.pt")
    child["run"]["wandb"]["enabled"] = True
    path = train(child)
    assert fake_wandb.starts[-1]["name"] == "child"
    assert fake_wandb.starts[-1]["group"] == path.name != source_run.name
    parent_id, child_id = [start["id"] for start in fake_wandb.starts]
    assert parent_id != child_id
    assert fake_wandb.runs[child_id]["summary"]["initialization"]["source_step"] == 4


def test_cli_entrypoint_forks_selected_snapshot_with_dotted_overrides(tmp_path):
    config = small_config(tmp_path)
    source_run = train(config)
    snapshot = source_run / "checkpoints/step_0000000002.pt"
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text(yaml.safe_dump(config))
    root = Path(__file__).resolve().parents[1]
    command = [
        sys.executable,
        "-m",
        "differentiable_exact_learning",
        "train",
        "--config",
        str(recipe),
        f"initialization.checkpoint={snapshot}",
        "initialization.restore_optimizer=true",
        "run.name=fork-via-command",
        "training.steps=2",
        "training.warmup_steps=0",
        "training.lr=0.003",
    ]
    env = dict(os.environ, PYTHONPATH=str(root / "src"), PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run(command, env=env, capture_output=True, text=True, check=True)
    path = Path(result.stdout.splitlines()[-1])
    state = read_checkpoint(path / "checkpoints/latest.pt")
    assert state["initialization"]["source_step"] == 2
    assert state["step"] == 2
    assert all(int(value["step"]) == 4 for value in state["optimizer"]["state"].values())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device unavailable")
def test_cpu_checkpoint_initializes_cuda_optimizer(tmp_path):
    config = small_config(tmp_path, "GRU")
    source = train(config) / "checkpoints/latest.pt"
    child = fork_config(config, source, restore_optimizer=True)
    child["run"]["device"] = "cuda"
    path = train(child)
    assert read_checkpoint(path / "checkpoints/latest.pt")["step"] == 4
