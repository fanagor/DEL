#!/usr/bin/env python
"""Plot final-checkpoint arithmetic ULG and universal task generalization.

Example:
  python scripts/plot_front_page.py --add2 outputs/ADD2_RUN --add10 outputs/ADD10_RUN \
    --mult2 outputs/MULT2_RUN --mult10 outputs/MULT10_RUN \
    --universal-tp outputs/UNIVERSAL_RUN --output-prefix analysis_outputs/paper_figures/front_page_summary/front_page

Final means training.steps, not the latest available checkpoint. Missing final
scores raise an error unless --evaluate-missing is supplied. Evaluations retain
run settings and disable loss snapshots; computing long PTOs can take time.
"""

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import yaml  # noqa: E402

from differentiable_exact_learning.analysis.figure_data import load_measurement, provenance_rows  # noqa: E402
from differentiable_exact_learning.analysis.front_page import ARITHMETIC_ORDER, plot_front_page  # noqa: E402
from differentiable_exact_learning.analysis.paper_heatmaps import save_figure  # noqa: E402


def load_final(run_path, label, *, evaluate_missing=False, device="cpu"):
    run = Path(run_path).expanduser().resolve()
    config = yaml.safe_load((run / "config.yaml").read_text())
    data = config["data"]
    if label == "universal_tp":
        if data["environment"] != "teleporting":
            raise ValueError(f"{run} is not a universal teleporting run")
    else:
        task = "addition" if label.startswith("add") else "multiplication"
        radix = 10 if label.endswith("10") else 2
        if (data["environment"], data["task"], data["radix"]) != ("grid", task, radix):
            raise ValueError(f"{run} does not match {label}")
    step = int(config["training"]["steps"])
    record = load_measurement(ROOT, run, step)
    if record["evaluation"] is None:
        if not evaluate_missing:
            raise ValueError(
                f"{label}: no evaluation at final step {step}. Wait for evaluation or use --evaluate-missing."
            )
        if not record["checkpoint_available"]:
            raise FileNotFoundError(
                f"{label}: final checkpoint does not exist: {record['checkpoint']}"
            )
        from differentiable_exact_learning.checkpoint import atomic_json
        from differentiable_exact_learning.evaluation import evaluate
        from differentiable_exact_learning.training import load_run

        model, expert, config, saved = load_run(record["checkpoint"], device=device)
        assert saved["step"] == step
        config = deepcopy(config)
        config["evaluation"]["snapshot"] = {"enabled": False}
        result = evaluate(model, expert, config)
        result.update(
            step=step, source_sha256=record["sha256"], evaluation_config=config["evaluation"]
        )
        destination = run / "analysis/evaluations" / f"step_{step:010d}.json"
        atomic_json(result, destination)
        record.update(evaluation=result, evaluation_path=str(destination))
    assert record["evaluation"]["step"] == step
    return record


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    for task in ARITHMETIC_ORDER:
        parser.add_argument(f"--{task}", required=True, type=Path)
    parser.add_argument("--universal-tp", required=True, type=Path)
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("analysis_outputs/paper_figures/front_page_summary/front_page"),
    )
    parser.add_argument("--evaluate-missing", action="store_true")
    parser.add_argument(
        "--device", default="cpu", help="Device for explicitly requested missing evaluations"
    )
    args = parser.parse_args()
    arithmetic = {
        task: load_final(
            getattr(args, task), task, evaluate_missing=args.evaluate_missing, device=args.device
        )
        for task in ARITHMETIC_ORDER
    }
    universal = load_final(
        args.universal_tp,
        "universal_tp",
        evaluate_missing=args.evaluate_missing,
        device=args.device,
    )
    fig = plot_front_page(arithmetic, universal)
    paths = save_figure(fig, args.output_prefix)
    records = {key: [value] for key, value in arithmetic.items()} | {"universal_tp": [universal]}
    path = args.output_prefix.with_suffix(".json")
    path.write_text(json.dumps(provenance_rows(records), indent=2) + "\n")
    for kind, destination in paths.items():
        print(f"{kind.upper()}: {destination.resolve()}")
    print(f"Provenance: {path.resolve()}")


if __name__ == "__main__":
    main()
