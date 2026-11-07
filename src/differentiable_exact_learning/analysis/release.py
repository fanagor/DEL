"""Read final paper artifacts and export selected runs without changing training outputs."""

from __future__ import annotations

import hashlib
import io
import json
from copy import deepcopy
from pathlib import Path

import torch
import yaml

from ..checkpoint import atomic_json
from ..config import validate_config
from ..data import create_expert
from ..models import build_model
from .public_metadata import public_checkpoint_bytes, public_metadata

CHECKPOINT = "final_model_checkpoint.pt"


def sha256(payload):
    return hashlib.sha256(payload).hexdigest()


def atomic_bytes(path, payload):
    import os
    import tempfile

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def safe_path(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Path leaves the repository: {relative}")
    return path


def read_catalog(root, manifest):
    """Read an explicitly supplied private export manifest."""
    catalog = yaml.safe_load(Path(manifest).read_text())
    entries = catalog["runs"]
    paths = [entry["path"] for entry in entries]
    if len(paths) != len(set(paths)):
        raise ValueError("Release paths must be unique")
    for entry in entries:
        safe_path(root, entry["path"])
        safe_path(root, entry["source"])
    return entries


def resolved_config(saved):
    """Resolve automatic defaults while retaining nulls with scientific meaning."""
    config = deepcopy(saved["config"])
    config["data"].setdefault("variant", "default")
    config["data"].setdefault("teaching_set", False)
    config["evaluation"].setdefault("scan", "sequential")
    config["training"].setdefault("inverse_freq_alpha", 0.0)
    config.setdefault("initialization", {"checkpoint": None, "restore_optimizer": False})
    model = config["model"]
    if model["hidden_size"] is None:
        # DFST h0 records the actual dimension even if the current expert changes.
        model["hidden_size"] = int(saved["model"]["h0"].shape[-1])
    dfst = model["name"] in {"DFST", "DFST_Softmax"}
    if model["init"]["state"] is None:
        model["init"]["state"] = "ones" if dfst else "zeros"
    if model["init"]["learnable_state"] is None:
        model["init"]["learnable_state"] = dfst
    validate_config(config)
    return config


def check_selection(entry, config, digest):
    d, m, t = config["data"], config["model"], config["training"]
    task = (
        "universal"
        if d["environment"] == "teleporting"
        else ("add" if d["task"] == "addition" else "mult") + str(d["radix"])
    )
    actual = dict(
        steps=t["steps"],
        warmup_steps=t["warmup_steps"],
        hidden_size=m["hidden_size"],
        data_size=d["size"],
        teaching_set=d.get("teaching_set", False),
    )
    if (entry["model"], entry["task"], entry["expert_digest"]) != (m["name"], task, digest):
        raise ValueError(f"Run identity changed: {entry['source']}")
    for key, value in entry["expected"].items():
        if key == "hidden_size" and value is None:
            continue
        if actual[key] != value:
            raise ValueError(f"Selected {key} changed: {entry['source']}")


def final_evaluation(source, step, digest):
    """Read only an evaluation at the planned final step, preferring training output."""
    paths = [
        source / "evaluation.json",
        source / f"evaluation_{step:010d}.json",
        source / "analysis/evaluations" / f"step_{step:010d}.json",
    ]
    for path in paths:
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        if data.get("step") != step:
            continue
        if not isinstance(data.get("tasks"), dict) or not data["tasks"]:
            raise ValueError(f"Empty final evaluation: {path}")
        if data.get("source_sha256", digest) != digest:
            raise ValueError(f"Evaluation belongs to different checkpoint bytes: {path}")
        return data, path
    return None, None


def export_run(root, entry):
    root = Path(root)
    source, dest = safe_path(root, entry["source"]), safe_path(root, entry["path"])
    source_config = yaml.safe_load((source / "config.yaml").read_text())
    planned = entry["expected"]["steps"]
    identity = json.loads((source / "wandb-run.json").read_text())
    if (identity["project"], identity["id"]) != (entry["wandb_project"], entry["wandb_id"]):
        raise ValueError(f"W&B identity changed: {source}")
    candidates = [
        source / "checkpoints" / f"step_{planned:010d}.pt",
        source / "checkpoints/latest.pt",
        source / "checkpoints/step_0000000000.pt",
    ]
    checkpoint = next((p for p in candidates if p.exists()), None)
    if checkpoint is None:
        raise FileNotFoundError(f"No checkpoint from which to resolve the configuration: {source}")
    payload = (
        checkpoint.read_bytes()
    )  # Capture latest once; a training write cannot change these bytes.
    saved = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=False)
    config = resolved_config(saved)
    check_selection(entry, config, saved["expert_digest"])
    check_selection(entry, source_config, saved["expert_digest"])
    step = saved["step"]
    if step > planned:
        raise ValueError(f"Checkpoint exceeds the selected budget: {source}")
    dest.mkdir(parents=True, exist_ok=True)
    config = public_metadata(config)
    config_bytes = (
        "# Automatic model defaults are resolved. Intentional nulls retain their meaning.\n"
        "# embed_dim: null = frozen one-hot; max_steps: null = unlimited; checkpoint: null = scratch.\n"
        + yaml.safe_dump(config, sort_keys=False)
    ).encode()
    row = dict(
        path=entry["path"],
        model=entry["model"],
        task=entry["task"],
        planned_steps=planned,
        available_step=step,
        status="pending_training",
        checkpoint_sha256="",
    )
    if step != planned:
        if (dest / CHECKPOINT).exists() or (dest / "evaluation.json").exists():
            raise ValueError(f"Refusing to replace a completed release with an interim run: {dest}")
        atomic_bytes(dest / "config.yaml", config_bytes)
        return row
    source_digest = sha256(payload)
    evaluation, evaluation_path = final_evaluation(source, step, source_digest)
    payload = public_checkpoint_bytes(saved)
    digest = sha256(payload)
    existing = dest / CHECKPOINT
    if existing.exists() and sha256(existing.read_bytes()) != digest:
        raise ValueError(f"Final weights changed; review before replacing {existing}")
    atomic_bytes(dest / "config.yaml", config_bytes)
    if not existing.exists():
        atomic_bytes(existing, payload)
    row.update(status="pending_evaluation", checkpoint_sha256=digest)
    if evaluation is not None:
        # Preserve the original result fields and bind them to the exact copied bytes.
        evaluation = public_metadata(evaluation)
        evaluation["source_sha256"] = digest
        evaluation["checkpoint"] = CHECKPOINT
        evaluation["final"] = True
        evaluation.setdefault("evaluation_config", deepcopy(config["evaluation"]))
        evaluation["expert_digest"] = saved["expert_digest"]
        atomic_json(evaluation, dest / "evaluation.json")
        row.update(status="ready")
    return row


def load_release_record(root, relative):
    root = Path(root)
    run = safe_path(root, relative)
    config = yaml.safe_load((run / "config.yaml").read_text())
    checkpoint = run / CHECKPOINT
    path = run / "evaluation.json"
    evaluation = None
    digest = None
    if path.exists():
        evaluation = json.loads(path.read_text())
        if not checkpoint.exists():
            raise ValueError(f"Evaluation has no final checkpoint: {run}")
        digest = sha256(checkpoint.read_bytes())
        if (
            evaluation.get("step") != config["training"]["steps"]
            or evaluation.get("source_sha256") != digest
            or evaluation.get("final") is not True
        ):
            raise ValueError(f"Final evaluation does not match the released checkpoint: {run}")
    return dict(
        run=str(run),
        run_id=relative,
        config=config,
        evaluation=evaluation,
        evaluation_path=str(path) if evaluation is not None else None,
        checkpoint=str(checkpoint),
        checkpoint_available=checkpoint.exists(),
        sha256=digest,
        step=evaluation["step"] if evaluation is not None else None,
        final=evaluation is not None,
        selection="release",
        status="ready"
        if evaluation is not None
        else ("pending_evaluation" if checkpoint.exists() else "pending_training"),
    )


def release_group(root, model, task):
    return figure_group(root, f"{model}_{task}")


def release_table(records):
    import pandas as pd

    rows = []
    for record in records:
        c = record["config"]
        base = dict(
            run=record["run_id"],
            status=record["status"],
            steps=c["training"]["steps"],
            warmup=c["training"]["warmup_steps"],
            hidden=c["model"]["hidden_size"],
            data_size=c["data"]["size"],
        )
        if record["evaluation"] is None:
            rows.append(base)
            continue
        for task, values in record["evaluation"]["tasks"].items():
            for metric in ("ulg", "rlg", "generalization"):
                if metric in values:
                    score = values[metric]
                    rows.append(
                        dict(
                            base,
                            task=task,
                            metric=metric,
                            length=score["length"],
                            cap=score.get("cap"),
                            at_cap=score.get("at_cap", False),
                        )
                    )
    return pd.DataFrame(rows)


def parameter_count(record):
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


def figure_group(root, name):
    settings = yaml.safe_load((Path(root) / "paper/figures.yaml").read_text())
    return [load_release_record(root, path) for path in settings["groups"][name]]


def analysis_cases(root, method):
    settings = yaml.safe_load((Path(root) / "paper/analyses.yaml").read_text())
    return [entry for entry in settings["analyses"] if entry["method"] == method]


def show_analysis_case(root, case):
    """Display saved case results without rerunning extraction or annealing."""
    import pandas as pd
    from IPython.display import Image, Markdown, display

    path = safe_path(root, case["path"])
    display(Markdown(f"### {case['title']}\n\nSource run: `{case['source_run']}`"))
    for name in ("graph_stability.png", "sharpness_evolution.png"):
        image = path / "figures" / name
        if image.exists():
            display(Image(filename=str(image), width=900))
    machine_path = path / "machines/extracted.json"
    machine = json.loads(machine_path.read_text())
    display(
        pd.Series(
            {
                "states": len(machine["states"]),
                "defined_transitions": len(machine["transitions"]),
                "proof_scope": case["proof_scope"],
            },
            name="Selected controller",
        )
    )
    display(pd.read_csv(path / "machines/extracted.csv", keep_default_na=False))
    check = path / "checks/pto_comparison.json"
    if check.exists():
        result = json.loads(check.read_text())
        display(
            pd.Series(
                {k: v for k, v in result.items() if isinstance(v, (str, int, float, bool))},
                name="Saved PTO comparison",
            )
        )
    report_path = case.get("proof_report", "proof/equivalence_verification.json")
    report = json.loads(safe_path(path, report_path).read_text())
    display(
        pd.Series(
            {k: v for k, v in report.items() if isinstance(v, (str, int, float, bool))},
            name="Saved verification report",
        )
    )
    # Notebook links resolve relative to notebooks/ in Jupyter and GitHub.
    prefix = "../" + case["path"]
    display(
        Markdown(
            f"[Case notes]({prefix}/README.md) · [Machine]({prefix}/machines/extracted.json) · "
            f"[Transition table]({prefix}/machines/extracted.pdf) · "
            f"[Proof]({prefix}/proof/equivalence_proof.pdf) · "
            f"[Checker]({prefix}/proof/verify_equivalence.py)"
        )
    )


def final_results_table(records):
    """Validate final evaluations and return measured results without release status."""
    records = list(records)
    for record in records:
        if record["evaluation"] is None:
            raise FileNotFoundError(f"Final evaluation required: {record['run_id']}")
    return release_table(records).drop(columns="status", errors="ignore")
