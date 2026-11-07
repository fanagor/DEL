"""Interrupted evaluations preserve completed probes and snapshots."""

import json
from copy import deepcopy

import pytest
import torch
from test_training import small_config

from differentiable_exact_learning import evaluation
from differentiable_exact_learning.data import create_expert
from differentiable_exact_learning.evaluation_progress import EvaluationProgress
from differentiable_exact_learning.training import construct_model, train


def setup(tmp_path, environment="grid"):
    config = small_config(tmp_path, "GRU", environment)
    config["evaluation"].update(max_length=4, hard_cases="all")
    config["evaluation"]["snapshot"]["enabled"] = False
    expert = create_expert(config["data"])
    model = construct_model(config, expert, torch.device("cpu"))
    return model, expert, config


@pytest.mark.parametrize("environment", ["grid", "tm", "teleporting"])
@pytest.mark.parametrize("search", ["binary", "exhaustive"])
def test_interrupted_search_resumes_without_repeating_probes(
    tmp_path, monkeypatch, capsys, environment, search
):
    model, expert, config = setup(tmp_path, environment)
    config["evaluation"]["search"] = search
    monkeypatch.setattr(expert, "sample_stream", lambda rng, n, **kw: (n, kw))
    calls = []
    interrupt = True

    def check(model, traces, **kwargs):
        length, options = next(iter(traces))
        calls.append((length, options))
        if interrupt and len(calls) == 3:
            raise MemoryError("interrupted")
        return length <= (2 if "operands" in options else 3)

    monkeypatch.setattr(evaluation, "traces_match", check)
    directory = tmp_path / "progress"
    with pytest.raises(MemoryError):
        evaluation.evaluate(model, expert, config, progress_dir=directory)
    saved = json.loads(next(directory.glob("*.json")).read_text())
    first = next(iter(saved["probes"].values()))
    assert first == ({"4": False, "2": True} if search == "binary" else {"1": True, "2": True})
    assert model.training
    interrupt = False
    calls.clear()
    result = evaluation.evaluate(model, expert, config, progress_dir=directory)
    assert calls[0][0] == 3
    assert "RESUMED PASS" in capsys.readouterr().out
    # The entire final result, including tested lengths, matches uninterrupted work.
    expected = evaluation.evaluate(model, expert, config)
    assert result == expected
    calls.clear()
    assert evaluation.evaluate(model, expert, config, progress_dir=directory) == expected
    assert calls == []


def test_random_subprobe_survives_interrupted_rlg(tmp_path, monkeypatch):
    model, expert, config = setup(tmp_path)
    calls = []

    def check(model, traces, **kwargs):
        calls.append(True)
        if len(calls) == 2:
            raise RuntimeError("hard-case probe interrupted")
        return True

    monkeypatch.setattr(evaluation, "traces_match", check)
    directory = tmp_path / "progress"
    with pytest.raises(RuntimeError, match="interrupted"):
        evaluation.evaluate(model, expert, config, progress_dir=directory)
    progress = EvaluationProgress(model, expert, config, directory)
    assert progress.results("addition2 ULG") == {4: True}
    assert progress.results("addition2 RLG") == {}
    result = evaluation.evaluate(model, expert, config, progress_dir=directory)
    assert len(calls) == 3  # Only the unfinished hard cases ran again.
    assert result["tasks"]["addition2"]["rlg"]["length"] == 4


def test_snapshot_saves_shorter_finished_length_before_batch_finishes(tmp_path, monkeypatch):
    model, expert, config = setup(tmp_path)
    config["evaluation"]["snapshot"] = {"enabled": True, "m": 4, "d": 2}
    config["evaluation"]["chunk_size"] = 4
    monkeypatch.setattr(evaluation, "traces_match", lambda *a, **kw: True)
    expected = evaluation.evaluate(model, expert, config)
    original = EvaluationProgress.record_snapshot

    def interrupt(progress, label, point):
        original(progress, label, point)
        raise RuntimeError("interrupted after first snapshot")

    directory = tmp_path / "progress"
    monkeypatch.setattr(EvaluationProgress, "record_snapshot", interrupt)
    with pytest.raises(RuntimeError, match="first snapshot"):
        evaluation.evaluate(model, expert, config, progress_dir=directory)
    progress = EvaluationProgress(model, expert, config, directory)
    assert list(progress.snapshots("addition2")) == ["2"]
    monkeypatch.setattr(EvaluationProgress, "record_snapshot", original)
    sample = expert.sample_stream
    lengths = []

    def track(rng, length, **kwargs):
        lengths.append(length)
        return sample(rng, length, **kwargs)

    monkeypatch.setattr(expert, "sample_stream", track)
    resumed = evaluation.evaluate(model, expert, config, progress_dir=directory)
    assert lengths == [4]
    a = resumed["tasks"]["addition2"]["snapshot"]["points"]
    b = expected["tasks"]["addition2"]["snapshot"]["points"]
    for point, reference in zip(a, b):
        assert point == pytest.approx(reference)


@pytest.mark.parametrize(
    "change", ["weights", "seed", "samples", "batch_size", "chunk_size", "expert"]
)
def test_different_model_or_protocol_cannot_reuse_progress(tmp_path, change):
    model, expert, config = setup(tmp_path)
    directory = tmp_path / "progress"
    first = EvaluationProgress(model, expert, config, directory)
    first.record("addition2 ULG", 4, True)
    config = deepcopy(config)
    if change == "weights":
        with torch.no_grad():
            next(model.parameters()).add_(1)
    elif change == "expert":
        config["data"]["radix"] = 3
        expert = create_expert(config["data"])
    else:
        config["evaluation"][change] += 1
    second = EvaluationProgress(model, expert, config, directory)
    assert first.path != second.path
    assert second.results("addition2 ULG") == {}
    assert json.loads(first.path.read_text())["probes"]["addition2 ULG"] == {"4": True}


def test_scheduling_changes_do_not_invalidate_progress(tmp_path):
    model, expert, config = setup(tmp_path)
    first = EvaluationProgress(model, expert, config, tmp_path)
    first.record("addition2 ULG", 4, True)
    config["evaluation"].update(every=123, final=True)
    second = EvaluationProgress(model, expert, config, tmp_path)
    assert second.path == first.path
    assert second.results("addition2 ULG") == {4: True}


def test_invalid_progress_fails_instead_of_becoming_a_scientific_failure(tmp_path):
    model, expert, config = setup(tmp_path)
    progress = EvaluationProgress(model, expert, config, tmp_path)
    progress.document["probes"] = {"addition2 ULG": {"4": "false"}}
    progress.save()
    with pytest.raises(ValueError, match="Invalid evaluation progress"):
        evaluation.evaluate(model, expert, config, progress_dir=tmp_path)


def test_training_retry_reuses_final_evaluation_progress(tmp_path, monkeypatch):
    _, _, config = setup(tmp_path)
    config["evaluation"]["final"] = True
    calls = []

    def check(*args, **kwargs):
        calls.append(True)
        if len(calls) == 2:
            raise MemoryError("interrupted final evaluation")
        return True

    monkeypatch.setattr(evaluation, "traces_match", check)
    directory = tmp_path / "run"
    with pytest.raises(MemoryError):
        train(config, run_dir=directory)
    train(config, run_dir=directory, resume=True)
    assert len(calls) == 3
    assert json.loads((directory / "evaluation.json").read_text())["step"] == 4
    assert len(list((directory / "evaluation_progress").glob("*.json"))) == 1
