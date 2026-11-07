"""Compact exports preserve controllers and reject ambiguous notebook selections."""

import json

import pandas as pd
import pytest

from differentiable_exact_learning.analysis.case_artifacts import (
    prepare_case,
    read_machine,
    save_case_config,
    save_machine,
    select_case,
)
from differentiable_exact_learning.analysis.extraction import Transducer
from differentiable_exact_learning.config import load_config
from differentiable_exact_learning.data import create_expert


def test_machine_formats_share_transitions(tmp_path):
    expert = create_expert(load_config()["data"])
    machine = Transducer({(0, 0): (1, 0, 0), (1, 0): (1, 0, 0)}, 0, (0,), frozenset({1}))
    result = save_machine(machine, expert, tmp_path, "extracted")
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "extracted.csv",
        "extracted.json",
        "extracted.pdf",
    ]
    assert read_machine(result["json"]) == machine
    rows = pd.read_csv(result["csv"])
    assert rows.state.tolist() == [0, 1] and rows.next_state.tolist() == [1, 1]
    assert result["pdf"].read_bytes().startswith(b"%PDF")
    assert result["panels"] and "<svg" in result["panels"][0]


def test_selection_rejects_unknown_negative_and_duplicate_indices():
    cases = [dict(case_index=0, title="A"), dict(case_index=4, title="B")]
    assert select_case(cases, 4)["title"] == "B"
    for index in [-1, 1, True, "0"]:
        with pytest.raises(ValueError):
            select_case(cases, index)
    with pytest.raises(ValueError):
        select_case(cases + cases, 0)


def test_output_modes_and_separate_inputs(tmp_path):
    for flag in [False, True]:
        root = prepare_case(tmp_path / str(flag), flag)
        assert (root / "intermediates").exists() == flag
        save_case_config(root, dict(samples=1, extraction_instances=[{"input": "01"}]))
        assert json.loads((root / "data/inputs.json").read_text()) == [{"input": "01"}]
        assert "extraction_instances" not in (root / "config.yaml").read_text()
        with pytest.raises(FileExistsError):
            prepare_case(root, flag)
    with pytest.raises(TypeError):
        prepare_case(tmp_path / "invalid", "yes")
