import json

import numpy as np
import pytest
import yaml

from differentiable_exact_learning.analysis.figure_data import load_measurement, panel_data


def test_measurement_requires_exact_step_and_hash(tmp_path):
    run = tmp_path / "run"
    (run / "checkpoints").mkdir(parents=True)
    (run / "config.yaml").write_text(yaml.safe_dump({"training": {"steps": 20}}))
    (run / "checkpoints/step_0000000010.pt").write_bytes(b"checkpoint")
    (run / "evaluation.json").write_text(json.dumps({"step": 20, "tasks": {}}))
    measured = load_measurement(tmp_path, "run", 10, tmp_path / "analysis")
    assert measured["evaluation"] is None and not measured["final"]
    (run / "evaluation.json").write_text(
        json.dumps({"step": 10, "tasks": {}, "source_sha256": "wrong"})
    )
    with pytest.raises(ValueError, match="hash"):
        load_measurement(tmp_path, "run", 10, tmp_path / "analysis")


def test_panel_preserves_orientation_zero_missing_and_censoring():
    records = []
    for y in [100, 200, 300]:
        for x in [1, 2, 3]:
            records.append(
                {
                    "config": {
                        "training": {"steps": y, "warmup_steps": x},
                        "data": {"task": "addition", "radix": 2},
                    },
                    "evaluation": {
                        "tasks": {"addition2": {"ulg": {"length": y + x, "at_cap": x == 3}}}
                    },
                }
            )
    records[0]["evaluation"]["tasks"]["addition2"]["ulg"]["length"] = 0
    records[1]["evaluation"] = None
    panel = panel_data(records, [1, 2, 3], [100, 200, 300])
    assert panel["values"][0, 0] == 0 and np.isnan(panel["values"][0, 1])
    assert panel["values"][2, 2] == 303 and panel["at_cap"][2, 2]
    with pytest.raises(ValueError, match="Duplicate"):
        panel_data(records + records[:1], [1, 2, 3], [100, 200, 300])


def test_saved_evaluation_does_not_require_weights_or_figure_folder(tmp_path):
    run = tmp_path / "run"
    (run / "analysis/evaluations").mkdir(parents=True)
    (run / "config.yaml").write_text(yaml.safe_dump({"training": {"steps": 20}}))
    result = {"step": 10, "tasks": {}, "source_sha256": "preserved-source-hash"}
    (run / "analysis/evaluations/step_0000000010.json").write_text(json.dumps(result))
    measured = load_measurement(tmp_path, "run", 10)
    assert measured["evaluation"] == result
    assert not measured["checkpoint_available"]
    assert measured["sha256"] == "preserved-source-hash"


def test_latest_capture_uses_actual_step_and_survives_replacement(tmp_path):
    import io

    import torch

    run = tmp_path / "run"
    (run / "checkpoints").mkdir(parents=True)
    (run / "config.yaml").write_text(yaml.safe_dump({"training": {"steps": 20}}))
    torch.save({"step": 7}, run / "checkpoints/latest.pt")
    measured = load_measurement(tmp_path, "run", "latest")
    torch.save({"step": 8}, run / "checkpoints/latest.pt")
    assert measured["step"] == 7
    assert torch.load(io.BytesIO(measured["_checkpoint_bytes"]), weights_only=False)["step"] == 7
    assert load_measurement(tmp_path, "run", "latest")["step"] == 8


def test_latest_evaluation_selects_recorded_step_without_weights(tmp_path):
    from differentiable_exact_learning.analysis.figure_data import load_latest_evaluation

    run = tmp_path / "run"
    (run / "analysis/evaluations").mkdir(parents=True)
    (run / "config.yaml").write_text(yaml.safe_dump({"training": {"steps": 30}}))

    def save(name, step, length):
        result = {"step": step, "tasks": {"addition2": {"ulg": {"length": length}}}}
        (run / name).write_text(json.dumps(result))

    save("evaluation.json", 10, 100)
    save("evaluation_0000000999.json", 20, 200)
    save("analysis/evaluations/step_0000000025.json", 25, 20)
    measured = load_latest_evaluation(tmp_path, "run")
    assert measured["step"] == 25 and not measured["final"]
    assert not measured["checkpoint_available"]
    assert measured["evaluation_path"].endswith("step_0000000025.json")
    save("evaluation.json", 30, 300)
    save("analysis/evaluations/step_0000000030.json", 30, 20)
    measured = load_latest_evaluation(tmp_path, run)
    assert measured["step"] == 30 and measured["final"]
    assert measured["evaluation"]["tasks"]["addition2"]["ulg"]["length"] == 300
    assert measured["evaluation_path"].endswith("/evaluation.json")


def test_latest_evaluation_reports_missing_and_skips_incomplete_files(tmp_path):
    from differentiable_exact_learning.analysis.figure_data import (
        load_latest_evaluation,
        provenance_rows,
    )

    run = tmp_path / "run"
    run.mkdir()
    (run / "config.yaml").write_text(yaml.safe_dump({"training": {"steps": 30}}))
    measured = load_latest_evaluation(tmp_path, "run")
    assert measured["evaluation"] is None and measured["step"] is None
    assert provenance_rows({"add2": [measured]})[0]["status"] == "No Recorded Evaluation"
    (run / "evaluation.json").write_text("{")
    (run / "evaluation_0000000010.json").write_text(
        json.dumps({"step": 10, "tasks": {"addition2": {"ulg": {"length": 0}}}})
    )
    with pytest.warns(UserWarning, match="Skipping unreadable evaluation"):
        measured = load_latest_evaluation(tmp_path, run)
    assert measured["step"] == 10
    assert measured["evaluation"]["tasks"]["addition2"]["ulg"]["length"] == 0
