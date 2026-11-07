"""Final-only exports, checkpoint identity, and portable release loading."""

import hashlib
import json
from pathlib import Path

import pytest
import torch
import yaml

from differentiable_exact_learning.analysis.release import (
    CHECKPOINT,
    export_run,
    load_release_record,
)
from differentiable_exact_learning.config import load_config
from differentiable_exact_learning.data import create_expert
from differentiable_exact_learning.data.datasets import dataset_spec
from differentiable_exact_learning.models import build_model
from differentiable_exact_learning.training import load_run


@pytest.fixture
def source_run(tmp_path):
    config = load_config(overrides=["training.steps=20", "training.warmup_steps=0"])
    expert = create_expert(config["data"])
    model = build_model(
        config["model"], expert.input_size, expert.symbol_size, expert.motion_size, expert.n_states
    )
    saved = dict(
        format_version=1,
        config=config,
        model=model.state_dict(),
        step=20,
        expert_digest=dataset_spec(config["data"])["expert_digest"],
    )
    source = tmp_path / "outputs/source"
    (source / "checkpoints").mkdir(parents=True)
    (source / "config.yaml").write_text(yaml.safe_dump(config))
    (source / "wandb-run.json").write_text(json.dumps(dict(project="test", id="one")))
    entry = dict(
        path="runs/DFST/add2/warmup_0_steps_20",
        source="outputs/source",
        model="DFST",
        task="add2",
        wandb_project="test",
        wandb_id="one",
        wandb_url="https://example.invalid/one",
        expert_digest=saved["expert_digest"],
        expected=dict(
            steps=20,
            warmup_steps=0,
            hidden_size=None,
            data_size=config["data"]["size"],
            teaching_set=False,
        ),
    )
    return tmp_path, source, saved, entry


def save_checkpoint(source, saved, name="latest.pt"):
    path = source / "checkpoints" / name
    torch.save(saved, path)
    return path.read_bytes()


def write_evaluation(source, step, **extra):
    data = dict(
        step=step, tasks={"addition2": {"ulg": dict(length=10, cap=10, at_cap=True)}}, **extra
    )
    (source / "evaluation.json").write_text(json.dumps(data))


def test_interim_results_never_become_final(source_run):
    root, source, saved, entry = source_run
    saved["step"] = 7
    save_checkpoint(source, saved)
    write_evaluation(source, 7)
    row = export_run(root, entry)
    dest = root / entry["path"]
    assert row["status"] == "pending_training"
    assert sorted(p.name for p in dest.iterdir()) == ["config.yaml"]
    config = yaml.safe_load((dest / "config.yaml").read_text())
    assert config["model"]["hidden_size"] == saved["model"]["h0"].numel()
    assert config["model"]["init"]["learnable_state"] is True
    assert config["model"]["embed_dim"] is None  # Frozen embeddings must stay frozen.
    assert config["data"]["max_steps"] is None
    assert load_release_record(root, entry["path"])["evaluation"] is None


def test_final_export_preserves_weights_and_requires_final_evaluation(source_run):
    root, source, saved, entry = source_run
    save_checkpoint(source, saved)
    write_evaluation(source, 7)
    row = export_run(root, entry)
    dest = root / entry["path"]
    assert row["status"] == "pending_evaluation"
    released = torch.load(dest / CHECKPOINT, map_location="cpu", weights_only=False)
    assert released["config"]["run"]["wandb"] == dict(enabled=False, project="DEL", entity=None)
    assert "tracking" not in released
    for key, value in saved["model"].items():
        assert torch.equal(released["model"][key], value)
    assert not (dest / "evaluation.json").exists()
    write_evaluation(source, 20)
    assert export_run(root, entry)["status"] == "ready"
    result = load_release_record(root, entry["path"])
    assert (
        result["final"]
        and result["sha256"] == hashlib.sha256((dest / CHECKPOINT).read_bytes()).hexdigest()
    )
    loaded, _, _, checkpoint = load_run(dest, device="cpu")
    assert checkpoint["step"] == 20
    for key, value in loaded.state_dict().items():
        assert torch.equal(value, saved["model"][key])
    load_run(dest / CHECKPOINT, device="cpu")
    before = {p.name: p.read_bytes() for p in dest.iterdir()}
    export_run(root, entry)
    assert before == {p.name: p.read_bytes() for p in dest.iterdir()}


def test_wrong_evaluation_hash_rejected(source_run):
    root, source, saved, entry = source_run
    save_checkpoint(source, saved)
    write_evaluation(source, 20, source_sha256="wrong")
    with pytest.raises(ValueError, match="different checkpoint"):
        export_run(root, entry)
    assert not (root / entry["path"] / "evaluation.json").exists()


def test_existing_final_weights_are_not_silently_replaced(source_run):
    root, source, saved, entry = source_run
    save_checkpoint(source, saved)
    write_evaluation(source, 20)
    export_run(root, entry)
    saved["model"]["A"].add_(1)
    save_checkpoint(source, saved)
    with pytest.raises(ValueError, match="Final weights changed"):
        export_run(root, entry)


def test_changed_selection_is_rejected(source_run):
    root, source, saved, entry = source_run
    save_checkpoint(source, saved)
    entry["expected"]["data_size"] += 1
    with pytest.raises(ValueError, match="data_size changed"):
        export_run(root, entry)


def test_display_rejects_corrupted_final_bytes(source_run):
    root, source, saved, entry = source_run
    save_checkpoint(source, saved)
    write_evaluation(source, 20)
    export_run(root, entry)
    (root / entry["path"] / CHECKPOINT).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="does not match"):
        load_release_record(root, entry["path"])


def test_figure_selection_covers_requested_grids():
    root = Path(__file__).parents[1]
    groups = yaml.safe_load((root / "paper/figures.yaml").read_text())["groups"]
    paths = [path for group in groups.values() for path in group]
    assert len(paths) == len(set(paths)) == 69
    for name, paths in groups.items():
        assert len(paths) == (6 if name == "DFST_universal" else 9)
        configs = [yaml.safe_load((root / path / "config.yaml").read_text()) for path in paths]
        assert all(c["data"]["teaching_set"] is False for c in configs)
        if name.startswith("DFST_Softmax"):
            coordinates = {(c["model"]["hidden_size"], c["data"]["size"]) for c in configs}
        elif name.endswith("universal"):
            coordinates = {(c["training"]["steps"], c["data"]["size"]) for c in configs}
        else:
            coordinates = {(c["training"]["steps"], c["training"]["warmup_steps"]) for c in configs}
        assert len(coordinates) == len(paths)


def test_curator_keeps_compact_artifacts_and_proof_dependencies(tmp_path):
    import importlib.util

    script = Path(__file__).parents[1] / "scripts/curate_analyses.py"
    spec = importlib.util.spec_from_file_location("curate", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = tmp_path / "analysis_outputs/case"
    (source / "machines").mkdir(parents=True)
    (source / "proof").mkdir()
    (source / "intermediates").mkdir()
    (source / "intermediates/debug.json").write_text("{}")
    run = tmp_path / "runs/DFST/add2/example"
    run.mkdir(parents=True)
    (run / CHECKPOINT).write_bytes(b"checkpoint")
    (source / "config.yaml").write_text(
        yaml.safe_dump(dict(source_sha256=hashlib.sha256(b"checkpoint").hexdigest()))
    )
    for name in ["extracted", "expert"]:
        for ext in ["json", "csv", "pdf"]:
            (source / "machines" / f"{name}.{ext}").write_bytes(f"{name}.{ext}".encode())
    (source / "proof/dependency.json").write_text("{}")
    entry = dict(
        source="analysis_outputs/case",
        path="analyses/case",
        source_run="runs/DFST/add2/example",
        method="metric_extraction",
        title="Case",
    )
    dest = module.curate(tmp_path, entry)
    assert (dest / "proof/dependency.json").read_bytes() == (
        source / "proof/dependency.json"
    ).read_bytes()
    assert not (dest / "intermediates").exists()
    module.curate(tmp_path, entry)
    (source / "machines/extracted.json").write_text("changed")
    with pytest.raises(ValueError, match="differs"):
        module.curate(tmp_path, entry)


def test_public_export_removes_nested_tracking(source_run):
    root, source, saved, entry = source_run
    saved["tracking"] = {"identity": {"id": "private-id", "url": "https://private.invalid"}}
    saved["config"]["run"]["wandb"] = {
        "enabled": True,
        "project": "private-project",
        "entity": "private-entity",
    }
    saved["annealing"] = {
        "source_config": {"run": {"wandb": dict(saved["config"]["run"]["wandb"])}}
    }
    save_checkpoint(source, saved)
    write_evaluation(source, 20)
    row = export_run(root, entry)
    public = torch.load(root / entry["path"] / CHECKPOINT, map_location="cpu", weights_only=False)
    assert "tracking" not in public
    for config in (public["config"], public["annealing"]["source_config"]):
        assert config["run"]["wandb"] == {"enabled": False, "project": "DEL", "entity": None}
    assert "wandb_url" not in row
    assert json.loads((source / "wandb-run.json").read_text())["id"] == "one"


def test_public_metadata_clears_run_identity_without_mutating_source():
    from differentiable_exact_learning.analysis.public_metadata import public_metadata

    source = {
        "config": {
            "run": {
                "name": "private-experiment",
                "seed": 42,
                "wandb": {"enabled": True, "project": "private", "entity": "team"},
            }
        }
    }
    public = public_metadata(source)
    run = public["config"]["run"]
    assert run["name"] is None
    assert run["seed"] == 42
    assert run["wandb"] == {"enabled": False, "project": "DEL", "entity": None}
    assert source["config"]["run"]["name"] == "private-experiment"
