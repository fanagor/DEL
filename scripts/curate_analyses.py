#!/usr/bin/env python
"""Import compact analysis outputs and optional independent proof bundles."""

import argparse
import shutil
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from differentiable_exact_learning.analysis.public_metadata import public_metadata  # noqa: E402
from differentiable_exact_learning.analysis.release import CHECKPOINT, safe_path, sha256  # noqa: E402


def curate(root, entry):
    source, dest = safe_path(root, entry["source"]), safe_path(root, entry["path"])
    if source == dest:
        raise ValueError("Import source and destination must differ")
    settings = yaml.safe_load((source / "config.yaml").read_text())
    checkpoint = safe_path(root, entry["source_run"]) / CHECKPOINT
    if sha256(checkpoint.read_bytes()) != settings["source_sha256"]:
        raise ValueError("Analysis source does not match the released checkpoint")
    required = [
        source / "machines" / (name + extension)
        for name in ("extracted", "expert")
        for extension in (".json", ".csv", ".pdf")
    ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    files = []
    for folder in ("machines", "figures", "data", "checks", "checkpoints", "proof"):
        for path in (source / folder).rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
                target = dest / path.relative_to(source)
                if target.exists() and target.read_bytes() != path.read_bytes():
                    raise ValueError(f"Published artifact differs: {target}")
                files.append((path, target))
    settings = public_metadata(settings)
    settings.update(
        method=entry["method"],
        source_run=entry["source_run"],
        source_checkpoint=CHECKPOINT,
        scope="extracted_discrete_controller",
    )
    config_payload = yaml.safe_dump(settings, sort_keys=False)
    if (dest / "config.yaml").exists() and (dest / "config.yaml").read_text() != config_payload:
        raise ValueError("Published configuration differs")
    dest.mkdir(parents=True, exist_ok=True)
    for path, target in files:
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy2(path, target)
    (dest / "config.yaml").write_text(config_payload)
    (dest / "README.md").write_text(
        f"# {entry['title']}\n\n"
        f"Source run: `{entry['source_run']}`. Method settings appear in `config.yaml`.\n\n"
        "`machines/` contains JSON, readable CSV and PDF transition maps. "
        "`figures/` contains main plots, `data/` contains supporting measurements, "
        "and `checks/` contains sampled PTO checks. Annealing checkpoints live in `checkpoints/`.\n\n"
        "Independent proofs, when supplied, live in `proof/`. "
        "Sampled PTO agreement alone does not establish equivalence on all inputs.\n"
    )
    return dest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Import manifest with source, path and source_run",
    )
    args = parser.parse_args()
    for entry in yaml.safe_load(args.manifest.read_text())["analyses"]:
        print(curate(args.root, entry), flush=True)


if __name__ == "__main__":
    main()
