"""Evaluation snapshots preserve trajectory metrics across chunks and batches."""

import json
import random

import numpy as np
import pytest
import torch
from test_training import small_config

from differentiable_exact_learning import evaluation
from differentiable_exact_learning.data import create_expert, pad_traces
from differentiable_exact_learning.evaluation import evaluate, search_lengths, snapshot_metrics
from differentiable_exact_learning.objectives import action_counts, action_loss
from differentiable_exact_learning.tracking import log_evaluation, snapshot_figure
from differentiable_exact_learning.training import construct_model, train


@pytest.mark.parametrize("name", ["DFST", "DFST_Softmax", "RNN", "GRU"])
@pytest.mark.parametrize("loss", ["mse", "cross_entropy"])
def test_snapshot_matches_full_trajectory_loss_and_accuracy(tmp_path, name, loss):
    config = small_config(tmp_path, name)
    if name == "DFST":
        config["model"]["init"].update(symbol="random", motion="random")
    expert = create_expert(config["data"])
    model = construct_model(config, expert, torch.device("cpu")).eval()
    samples = [(n, expert.sample(random.Random(n), n, lengths=(n, n))) for n in (1, 2, 3)]
    expected = []
    with torch.no_grad():
        for _, trace in samples:
            batch = pad_traces([trace])
            symbols, motions, _ = model(batch.inputs)
            expected.append(
                (
                    float(
                        action_loss(
                            symbols,
                            motions,
                            batch.symbols,
                            batch.motions,
                            batch.mask,
                            loss=loss,
                            output_kind=model.output_kind,
                        )
                    ),
                    action_counts(symbols, motions, batch.symbols, batch.motions, batch.mask)[0]
                    / int(batch.mask.sum()),
                )
            )
    for batch_size, chunk_size in ((1, 1), (2, 3), (100, 256)):
        points = snapshot_metrics(
            model, iter(samples), batch_size=batch_size, chunk_size=chunk_size, loss=loss
        )
        assert [p["length"] for p in points] == [1, 2, 3]
        for point, (value, accuracy) in zip(points, expected):
            assert point["loss"] == pytest.approx(value, rel=2e-5, abs=1e-6)
            assert point["accuracy"] == accuracy


@pytest.mark.parametrize("environment", ["grid", "tm", "teleporting"])
def test_snapshot_lengths_tasks_and_determinism(tmp_path, environment):
    config = small_config(tmp_path, "GRU", environment)
    config["evaluation"]["snapshot"] = {"enabled": True, "m": 7, "d": 2}
    config["data"]["eval_tasks"] = ["parity", {"name": "scaling_A", "m": 10}]
    expert = create_expert(config["data"])
    model = construct_model(config, expert, torch.device("cpu"))
    state = torch.get_rng_state()
    first = evaluate(model, expert, config)
    assert first == evaluate(model, expert, config)
    assert torch.equal(state, torch.get_rng_state())
    assert model.training
    assert len(first["tasks"]) == (2 if environment == "teleporting" else 1)
    for metrics in first["tasks"].values():
        assert [p["length"] for p in metrics["snapshot"]["points"]] == [2, 4, 6]
    if environment == "teleporting":
        assert first["tasks"]["scaling_A_m10"]["snapshot"]["axis"] == "input_and_machine_size"


def test_snapshot_scaling_uses_candidate_machine_size(tmp_path, monkeypatch):
    config = small_config(tmp_path, "GRU", "teleporting")
    config["data"]["eval_tasks"] = ["scaling_B_m10"]
    expert = create_expert(config["data"])
    model = construct_model(config, expert, torch.device("cpu"))
    original = expert.sample_stream
    calls = []

    def sample(rng, length, **kwargs):
        calls.append((length, kwargs["task"], kwargs["lengths"]))
        return original(rng, length, **kwargs)

    monkeypatch.setattr(expert, "sample_stream", sample)
    evaluate(model, expert, config)
    assert calls[-2:] == [(2, "scaling_B_m2", 2), (4, "scaling_B_m4", 4)]


def test_nonfinite_snapshot_is_json_safe(tmp_path):
    config = small_config(tmp_path, "RNN")
    expert = create_expert(config["data"])
    model = construct_model(config, expert, torch.device("cpu"))
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(float("nan"))
    trace = expert.sample(random.Random(1), 1, lengths=(1, 1))
    points = snapshot_metrics(model, [(1, trace)], batch_size=1, chunk_size=2, loss="mse")
    assert points[0]["loss"] is None
    assert points[0]["loss_finite"] is False
    assert points[0]["accuracy"] == 0
    json.dumps(points, allow_nan=False)


def test_binary_returns_immediately_when_cap_passes():
    calls = []

    def probe(length):
        calls.append(length)
        return True

    assert search_lengths(probe, 8)["length"] == 8
    assert calls == [8]


def test_binary_searches_below_failed_cap_without_retesting_it():
    calls = []

    def probe(length):
        calls.append(length)
        return length <= 5

    result = search_lengths(probe, 8)
    assert result["length"] == 5
    assert not result["at_cap"]
    assert calls == [8, 4, 6, 5]
    assert result["tested"] == [
        {"length": 4, "passed": True},
        {"length": 5, "passed": True},
        {"length": 6, "passed": False},
        {"length": 8, "passed": False},
    ]


@pytest.mark.parametrize("method", ["binary", "exhaustive"])
def test_length_progress_is_printed_before_and_after_each_probe(capsys, method):
    output = []
    lengths = []

    def probe(length):
        output.append(capsys.readouterr().out)
        assert output[-1].endswith(f"multiplication10 ULG length={length}: START\n")
        lengths.append(length)
        return length <= 5

    result = search_lengths(probe, 8, method, label="multiplication10 ULG")
    output.append(capsys.readouterr().out)
    text = "".join(output)
    assert result["length"] == 5
    assert text.count(": START") == len(lengths)
    for length in lengths:
        outcome = "PASS" if length <= 5 else "FAIL"
        assert f"multiplication10 ULG length={length}: {outcome} (elapsed=" in text
    assert "largest passing tested length=5" in text


@pytest.mark.parametrize("error", [MemoryError("out of memory"), RuntimeError("probe interrupted")])
def test_length_progress_keeps_resource_errors_distinct_from_failures(capsys, error):
    def probe(length):
        raise error

    with pytest.raises(type(error)) as raised:
        search_lengths(probe, 8, label="multiplication10 RLG")
    assert raised.value is error
    output = capsys.readouterr().out
    assert "multiplication10 RLG length=8: START" in output
    assert f"multiplication10 RLG length=8: ERROR {type(error).__name__}: {error}" in output
    assert ": FAIL" not in output
    assert "largest passing" not in output


@pytest.mark.parametrize("environment", ["grid", "tm"])
def test_arithmetic_progress_labels_ulg_and_rlg_without_repeating_cached_work(
    tmp_path, monkeypatch, capsys, environment
):
    config = small_config(tmp_path, "GRU", environment)
    config["evaluation"].update(max_length=4, hard_cases="all", search="binary")
    config["evaluation"]["snapshot"]["enabled"] = False
    expert = create_expert(config["data"])
    model = construct_model(config, expert, torch.device("cpu"))
    # ULG(4) passes; RLG(4) reuses that random probe and fails its hard cases.
    # RLG(2) passes both checks; RLG(3) fails its random check.
    results = iter([True, False, True, True, False])
    monkeypatch.setattr(evaluation, "traces_match", lambda *args, **kwargs: next(results))
    metrics = evaluate(model, expert, config)["tasks"]["addition2"]
    assert metrics["ulg"]["length"] == 4
    assert metrics["rlg"]["length"] == 2
    assert next(results, None) is None
    output = capsys.readouterr().out
    for metric, length, outcome in [
        ("ULG", 4, "PASS"),
        ("RLG", 4, "FAIL"),
        ("RLG", 2, "PASS"),
        ("RLG", 3, "FAIL"),
    ]:
        assert f"addition2 {metric} length={length}: {outcome}" in output
    assert output.count("addition2 ULG length=4: START") == 1


@pytest.mark.parametrize("environment", ["grid", "tm"])
@pytest.mark.parametrize("task", ["addition", "multiplication"])
@pytest.mark.parametrize("hard_cases", ["all", "diagonal"])
def test_rlg_starts_at_measured_ulg_below_configured_cap_and_reuses_progress(
    tmp_path, monkeypatch, capsys, environment, task, hard_cases
):
    config = small_config(tmp_path, "GRU", environment)
    config["data"]["task"] = task
    config["evaluation"].update(max_length=8, hard_cases=hard_cases, search="binary")
    config["evaluation"]["snapshot"]["enabled"] = False
    expert = create_expert(config["data"])
    model = construct_model(config, expert, torch.device("cpu"))
    monkeypatch.setattr(expert, "sample_stream", lambda rng, n, **kw: (n, kw))
    calls = []

    def check(model, traces, **kwargs):
        probes = list(traces)
        length, options = probes[0]
        hard = "operands" in options
        calls.append(("hard" if hard else "random", length))
        if hard:
            assert [kw["operands"] for _, kw in probes] == [
                (a * length, b * length)
                for a in "01"
                for b in "01"
                if hard_cases == "all" or a == b
            ]
        return length <= (3 if hard else 5)

    monkeypatch.setattr(evaluation, "traces_match", check)
    directory = tmp_path / "progress"
    result = evaluate(model, expert, config, progress_dir=directory)
    metrics = result["tasks"][f"{task}2"]
    assert metrics["ulg"]["length"] == 5
    assert metrics["rlg"]["length"] == 3
    assert metrics["rlg"]["cap"] == 8 and not metrics["rlg"]["at_cap"]
    assert calls == [
        ("random", 8),
        ("random", 4),
        ("random", 6),
        ("random", 5),
        ("hard", 5),
        ("random", 2),
        ("hard", 2),
        ("random", 3),
        ("hard", 3),
        ("hard", 4),
    ]
    starts = [line for line in capsys.readouterr().out.splitlines() if "RLG" in line]
    assert starts[0] == f"[evaluation] {task}2 RLG length=5: START"
    calls.clear()
    assert evaluate(model, expert, config, progress_dir=directory) == result
    assert calls == []


def test_zero_length_search_is_reported_as_skipped(capsys):
    def probe(length):
        pytest.fail("No positive lengths should be probed")

    result = search_lengths(probe, 0, label="addition2 RLG")
    assert result["tested"] == []
    assert "addition2 RLG: skipped (no positive lengths to test)" in capsys.readouterr().out


def test_teleporting_progress_identifies_each_task(tmp_path, monkeypatch, capsys):
    config = small_config(tmp_path, "GRU", "teleporting")
    config["data"]["eval_tasks"] = ["parity", "scaling_A_m10"]
    config["evaluation"].update(max_length=4, search="binary")
    config["evaluation"]["snapshot"]["enabled"] = False
    expert = create_expert(config["data"])
    model = construct_model(config, expert, torch.device("cpu"))
    monkeypatch.setattr(evaluation, "traces_match", lambda *args, **kwargs: True)
    evaluate(model, expert, config)
    output = capsys.readouterr().out
    for task in config["data"]["eval_tasks"]:
        assert f"{task} generalization length=4: START" in output
        assert f"{task} generalization length=4: PASS" in output


def test_snapshot_progress_reports_lengths_and_metrics(tmp_path, capsys):
    config = small_config(tmp_path, "GRU")
    expert = create_expert(config["data"])
    model = construct_model(config, expert, torch.device("cpu"))
    samples = [(n, expert.sample_stream(random.Random(n), n, lengths=(n, n))) for n in (1, 2)]
    points = snapshot_metrics(
        model, samples, batch_size=2, chunk_size=4, loss="mse", label="addition2"
    )
    output = capsys.readouterr().out
    for point in points:
        prefix = f"addition2 snapshot length={point['length']}:"
        assert f"{prefix} START" in output
        assert f"{prefix} DONE accuracy={point['accuracy']:.4%} loss={point['loss']}" in output


@pytest.mark.parametrize("environment", ["grid", "tm", "teleporting"])
def test_training_logs_periodic_and_final_evaluation(tmp_path, fake_wandb, environment):
    config = small_config(tmp_path, "GRU", environment)
    config["run"]["wandb"]["enabled"] = True
    config["evaluation"].update(every=2, final=True)
    path = train(config)
    rows = next(iter(fake_wandb.runs.values()))["rows"]
    logged = [row for row in rows if any(k.startswith("evaluation/") for k in row)]
    assert [row["step"] for row in logged] == [2, 4]
    saved = json.loads((path / "evaluation.json").read_text())
    assert (path / "evaluation_0000000002.json").exists()
    expected_keys = {"step", "_step"}
    for task, metrics in saved["tasks"].items():
        prefix = f"evaluation/{task}"
        for name in ("generalization",) if environment == "teleporting" else ("ulg", "rlg"):
            assert logged[-1][f"{prefix}/{name}"] == metrics[name]["length"]
            expected_keys.add(f"{prefix}/{name}")
            assert "cap" in metrics[name] and "at_cap" in metrics[name]
        assert len(metrics["snapshot"]["points"]) == 2
        expected_keys.add(f"{prefix}/snapshot")
        pixels = logged[-1][f"{prefix}/snapshot"]["pixels"]
        assert pixels.ndim == 3 and np.ptp(pixels) > 0
    assert all(set(row) == expected_keys for row in logged)
    snapshot = next(iter(saved["tasks"].values()))["snapshot"]
    figure = snapshot_figure(snapshot, "test", 4)
    assert len(figure.axes) == 2
    assert figure.axes[0].lines[0].get_color() != figure.axes[1].lines[0].get_color()
    assert figure.axes[0].get_ylabel() == "Test loss (mse)"
    figure.clear()


def test_no_tracking_import_when_disabled():
    log_evaluation(None, {})


@pytest.mark.parametrize(
    "override",
    [
        "evaluation.snapshot.m=0",
        "evaluation.snapshot.d=-1",
        "evaluation.snapshot.d=101",
        "evaluation.snapshot.enabled=1",
    ],
)
def test_snapshot_validation(override):
    from differentiable_exact_learning.config import load_config

    with pytest.raises(ValueError, match="evaluation.snapshot"):
        load_config(overrides=[override])
