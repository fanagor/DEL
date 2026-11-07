"""Compact analysis artifacts for indexed notebook views and reproducible checks."""

import csv
import json
import shutil
import tempfile
from pathlib import Path

import yaml

from ..checkpoint import atomic_json
from .extraction import Transducer, export_machine, transition_rows
from .public_metadata import public_metadata
from .transition_table import render_transition_table


def prepare_case(directory, save_intermediates=False):
    if not isinstance(save_intermediates, bool):
        raise TypeError("SAVE_INTERMEDIATES must contain True or False")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    for name in ("machines", "figures", "data", "checks"):
        (directory / name).mkdir()
    if save_intermediates:
        (directory / "intermediates").mkdir()
    return directory


def save_case_config(directory, settings):
    directory = Path(directory)
    settings = public_metadata(settings)
    inputs = settings.pop("extraction_instances", None)
    if inputs is not None:
        atomic_json(inputs, directory / "data/inputs.json")
        settings["inputs"] = "data/inputs.json"
    (directory / "config.yaml").write_text(yaml.safe_dump(settings, sort_keys=False))


def read_machine(path):
    raw = json.loads(Path(path).read_text())
    return Transducer(
        {
            (row["state"], row["input"]): (row["next_state"], row["symbol"], row["motion"])
            for row in raw["transitions"]
        },
        raw["initial_state"],
        tuple(raw["alphabet"]),
        frozenset(raw["final_states"]),
    )


def write_readable_csv(machine, expert, path):
    rows = transition_rows(machine)
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["state", "read", "write", "move", "next_state"])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                dict(
                    state=row["state"],
                    next_state=row["next_state"],
                    read=expert.vocab_rev[row["input"]] or "λ",
                    write=expert.vocab_rev[row["symbol"]] or "λ",
                    move=expert.motion_vocab_rev[row["motion"]],
                )
            )


def save_machine(machine, expert, directory, name):
    """Save canonical JSON, readable CSV and PDF; return SVG panels for inline display."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="del-transition-table-") as temporary:
        prefix = Path(temporary) / name
        export_machine(machine, prefix)
        shutil.copy2(prefix.with_suffix(".json"), directory / f"{name}.json")
        tables = render_transition_table(machine, expert, prefix)
        shutil.copy2(tables["pdf"], directory / f"{name}.pdf")
        panels = [path.read_text() for path in tables["svg"]]
    write_readable_csv(machine, expert, directory / f"{name}.csv")
    return dict(
        json=directory / f"{name}.json",
        csv=directory / f"{name}.csv",
        pdf=directory / f"{name}.pdf",
        panels=panels,
    )


def select_case(cases, index):
    if isinstance(index, bool) or not isinstance(index, int):
        raise ValueError("CASE_INDEX must contain an integer")
    by_index = {entry["case_index"]: entry for entry in cases}
    if len(by_index) != len(cases):
        raise ValueError("Analysis case indices must be unique")
    if index not in by_index:
        raise ValueError(f"Unknown CASE_INDEX: {index}. Available indices: {sorted(by_index)}")
    return by_index[index]
