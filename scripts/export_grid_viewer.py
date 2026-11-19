"""Export inference-only weights and expert programs for the static DEL grid viewer.

Run from an installed DEL checkout:
    python scripts/export_grid_viewer.py /path/to/site/del/grid/data

Each binary file contains little-endian float32 A, B, C, h0 in C order.
The leading axis of A, B and C follows the alphabet in catalog.json.
Softmax runs export the effective stochastic matrices and initial distribution.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from differentiable_exact_learning.data.experts import DEFAULT_TP_TASKS
from differentiable_exact_learning.training import load_run
from differentiable_exact_learning.viewer.controller import ControllerSource, Session


def export_expert(expert):
    encoded = {
        "environment": expert.environment,
        "alphabet": expert.alphabet,
        "motions": [expert.motion_vocab_rev[i] for i in range(expert.motion_size)],
        "halt": expert.halt_state,
        "transitions": [],
    }
    for state in range(expert.n_states):
        row = []
        for symbol in expert.alphabet:
            successor, write, motion = expert.transitions[state, symbol]
            row.append([successor, expert.vocab[write], expert.motion_vocab[motion]])
        encoded["transitions"].append(row)
    if expert.environment == "teleporting":
        names = dict.fromkeys([*DEFAULT_TP_TASKS, *expert.train_tasks, *expert.eval_tasks])
        encoded["programs"] = {
            name: expert.encode_program(expert.program_transitions(name), "")[:-1] for name in names
        }
    else:
        encoded.update(radix=expert.radix, operation=expert.task)
    return encoded


def reference_trace(expert, model, run, program=None):
    """Use the local viewer for an independent check of the website implementation."""
    if expert.environment == "teleporting" and program is None:
        program = "parity"
    text = "10" if program else "1+1" if expert.task == "addition" else "1x1"
    source = ControllerSource(expert, "reference", model, program)
    session = Session(source, text, task=program)
    steps = []
    while len(steps) < 200:
        step = session.advance()
        if step is None:
            break
        if "motion" not in step:
            raise ValueError(step["message"])
        frame = {key: step[key] for key in ("read", "write", "motion", "head", "status", "state")}
        if model is not None and len(steps) < 3:
            frame["hiddenAfter"] = session.state.flatten().tolist()
        steps.append(frame)
    cells = sorted([*session.position(pos), sym] for pos, sym in session.memory.cells.items())
    return dict(run=run, input=text, program=program, steps=steps, cells=cells)


@torch.inference_mode()
def export(destination, runs, parity):
    checkpoints = sorted(runs.glob("*/*/*/final_model_checkpoint.pt"))
    checkpoints = [
        p for p in checkpoints if p.relative_to(runs).parts[0] in {"DFST", "DFST_Softmax"}
    ]
    if not checkpoints:
        raise ValueError(f"No released DFST checkpoints found in {runs}")
    destination.mkdir(parents=True, exist_ok=True)
    catalog = {"experts": {}, "runs": [], "programs": {}}
    program_overrides = {}
    examples = []
    for path in checkpoints:
        family, task, run_id, _ = path.relative_to(runs).parts
        model, expert, config, saved = load_run(path, device="cpu")
        if config["model"]["name"] != family or expert.environment not in {"grid", "teleporting"}:
            raise ValueError(f"Unexpected architecture or environment in {path}")
        key = (task, saved["expert_digest"])
        if key not in program_overrides:
            encoded = export_expert(expert)
            program_overrides[key] = {}
            if task not in catalog["experts"]:
                catalog["experts"][task] = encoded
                if parity:
                    examples.append(dict(task=task, **reference_trace(expert, None, None)))
            else:
                base = catalog["experts"][task]
                programs = encoded.pop("programs", {})
                if encoded != {k: v for k, v in base.items() if k != "programs"}:
                    raise ValueError(f"Incompatible expert controller for {path}")
                for name, rows in programs.items():
                    if rows != base.get("programs", {}).get(name):
                        variant = f"{name}_{saved['expert_digest'][:12]}"
                        catalog["programs"][variant] = rows
                        program_overrides[key][name] = variant
        relative = Path(family) / task / f"{run_id}.bin"
        filename = destination / relative
        filename.parent.mkdir(parents=True, exist_ok=True)
        with filename.open("wb") as stream:
            for tensor in model.effective_tensors():
                data = tensor.contiguous().cpu().numpy().astype("<f4", copy=False)
                if not np.isfinite(data).all():
                    raise ValueError(f"Non-finite model parameters in {path}")
                stream.write(data.tobytes())
        source = f"runs/{family}/{task}/{run_id}"
        catalog["runs"].append(
            dict(
                family=family,
                task=task,
                id=run_id,
                hidden=model.hidden_size,
                path=f"data/{relative.as_posix()}",
                source=source,
                **({"programs": program_overrides[key]} if program_overrides[key] else {}),
            )
        )
        if parity:
            examples.append(dict(task=task, **reference_trace(expert, model, source)))
            for program in program_overrides[key]:
                if program.endswith("_m5"):
                    examples.append(
                        dict(task=task, **reference_trace(expert, model, source, program))
                    )
    (destination / "catalog.json").write_text(json.dumps(catalog, separators=(",", ":")) + "\n")
    if parity:
        parity.write_text(json.dumps(examples, separators=(",", ":")) + "\n")
    print(f"Exported {len(catalog['runs'])} checkpoints and {len(catalog['experts'])} experts.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--runs", type=Path, default=Path(__file__).resolve().parents[1] / "runs")
    parser.add_argument("--parity", type=Path, help="Also write reference executions for testing.")
    args = parser.parse_args()
    export(args.destination, args.runs, args.parity)
