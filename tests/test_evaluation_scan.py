"""Independent evaluation scans, bounded chunks, and old-checkpoint compatibility."""

import json
from copy import deepcopy

import pytest
import torch
from test_models import make_model
from test_training import small_config

from differentiable_exact_learning import inference
from differentiable_exact_learning.checkpoint import atomic_save, read_checkpoint
from differentiable_exact_learning.cli import main
from differentiable_exact_learning.config import experiment_config, load_config, validate_config
from differentiable_exact_learning.data import create_expert
from differentiable_exact_learning.evaluation import evaluate
from differentiable_exact_learning.evaluation_progress import EvaluationProgress
from differentiable_exact_learning.inference import forward_chunk
from differentiable_exact_learning.training import construct_model, train


@pytest.mark.parametrize("name", ["DFST", "DFST_Softmax"])
@pytest.mark.parametrize("embed_dim", [None, 5])
@pytest.mark.parametrize("training_scan", ["sequential", "parallel"])
@pytest.mark.parametrize("evaluation_scan", ["sequential", "parallel"])
def test_evaluation_scan_matches_raw_forward_across_chunks(
    name, embed_dim, training_scan, evaluation_scan
):
    torch.manual_seed(173)
    model = make_model(name, embed_dim=embed_dim, scan=training_scan).eval()
    tokens = torch.randint(3, (2, 11))
    with torch.no_grad():
        expected_symbols, expected_motions, expected_state = model(tokens)
        for size in (1, 3, 20):
            state = None
            pieces = []
            for start in range(0, tokens.shape[1], size):
                symbols, motions, state = forward_chunk(
                    model, tokens[:, start : start + size], state, scan=evaluation_scan
                )
                pieces.append((symbols, motions))
                if name == "DFST_Softmax":
                    for value in (symbols, motions, state):
                        assert bool((value >= 0).all())
                        torch.testing.assert_close(value.sum(-1), torch.ones_like(value.sum(-1)))
            torch.testing.assert_close(torch.cat([p[0] for p in pieces], 1), expected_symbols)
            torch.testing.assert_close(torch.cat([p[1] for p in pieces], 1), expected_motions)
            torch.testing.assert_close(state, expected_state)
    assert model.scan == training_scan


@pytest.mark.parametrize("name", ["DFST", "DFST_Softmax"])
@pytest.mark.parametrize("scan", ["sequential", "parallel"])
def test_evaluation_scan_empty_input_and_invalid_state(name, scan):
    model = make_model(name)
    tokens = torch.empty(2, 0, dtype=torch.long)
    state = torch.randn(2, model.hidden_size, dtype=torch.float64)
    symbols, motions, final = forward_chunk(model, tokens, state, scan=scan)
    assert symbols.shape == (2, 0, 4) and motions.shape == (2, 0, 2)
    torch.testing.assert_close(final, state)
    with pytest.raises(ValueError, match="state must have shape"):
        forward_chunk(model, tokens, state[:, :1], scan=scan)


def test_softmax_sequential_avoids_dense_scan_and_preserves_probabilities(monkeypatch):
    model = make_model("DFST_Softmax", hidden_size=93).eval()

    def forbidden(*args, **kwargs):
        raise AssertionError("Sequential inference must not allocate the dense scan intermediates")

    monkeypatch.setattr(model, "forward", forbidden)
    monkeypatch.setattr(inference, "matrix_prefix_scan", forbidden)
    with torch.no_grad():
        result = forward_chunk(
            model, torch.zeros(2, 257, dtype=torch.long), stabilize=True, scan="sequential"
        )
    for value in result:
        assert bool(torch.isfinite(value).all())
        torch.testing.assert_close(value.sum(-1), torch.ones_like(value.sum(-1)))


def test_sequential_stabilization_is_retained_for_plain_dfst():
    model = make_model("DFST", hidden_size=2).float().eval()
    with torch.no_grad():
        model.A.copy_(10 * torch.eye(2).expand_as(model.A))
        tokens = torch.zeros(1, 1000, dtype=torch.long)
        sequential = forward_chunk(model, tokens, stabilize=True, scan="sequential")
        parallel = forward_chunk(model, tokens, stabilize=True, scan="parallel")
    assert all(bool(torch.isfinite(v).all()) for v in sequential)
    assert not all(bool(torch.isfinite(v).all()) for v in parallel)


@pytest.mark.parametrize("name", ["DFST", "DFST_Softmax"])
@pytest.mark.parametrize("scan", ["sequential", "parallel"])
def test_configured_scan_controls_searches_and_snapshots_with_bounded_chunks(
    tmp_path, monkeypatch, name, scan
):
    config = small_config(tmp_path, name)
    training_scan = "parallel" if scan == "sequential" else "sequential"
    config["model"].update(hidden_size=None, scan=training_scan)
    config["evaluation"].update(scan=scan, chunk_size=3)
    expert = create_expert(config["data"])
    model = construct_model(config, expert, torch.device("cpu"))
    with torch.no_grad():
        for key, tensor in zip(("A", "B", "C"), expert.get_tensors()):
            value = tensor if name == "DFST" else torch.where(tensor > 0, 40.0, -40.0)
            getattr(model, key).copy_(value)
        model.h0.fill_(0 if name == "DFST" else -40)
        model.h0[0] = 1 if name == "DFST" else 40
    calls, scan_shapes = [], []
    original_chunk, original_scan = inference.forward_chunk, inference.matrix_prefix_scan

    def chunk(current_model, tokens, state=None, **kwargs):
        calls.append((tokens.shape, state is not None, kwargs))
        return original_chunk(current_model, tokens, state, **kwargs)

    def prefix(matrices):
        scan_shapes.append(matrices.shape)
        return original_scan(matrices)

    monkeypatch.setattr("differentiable_exact_learning.evaluation.forward_chunk", chunk)
    monkeypatch.setattr(inference, "matrix_prefix_scan", prefix)
    result = evaluate(model, expert, config)
    metrics = result["tasks"]["addition2"]
    assert result["scan"] == scan
    assert metrics["ulg"]["length"] == metrics["rlg"]["length"] == 2
    assert len(metrics["snapshot"]["points"]) == 2
    assert calls and all(shape[1] <= 3 and kw["scan"] == scan for shape, _, kw in calls)
    assert any(continuing for _, continuing, _ in calls)
    assert {kw["stabilize"] for _, _, kw in calls} == {True, False}
    assert bool(scan_shapes) == (scan == "parallel")
    assert all(shape[1] <= 3 for shape in scan_shapes)
    assert model.scan == training_scan and model.training


@pytest.mark.parametrize("value", ["auto", "invalid", "null", "1", "[parallel]"])
def test_invalid_evaluation_scan_is_rejected(value):
    with pytest.raises(ValueError, match="evaluation.scan"):
        load_config(overrides=[f"evaluation.scan={value}"])


def test_evaluation_scan_default_and_training_identity():
    config = load_config()
    assert config["evaluation"]["scan"] == "parallel"
    identity = experiment_config(config)
    config["evaluation"]["scan"] = "sequential"
    assert experiment_config(config) == identity
    del config["evaluation"]["scan"]
    assert experiment_config(config) == identity
    validate_config(config)
    assert config["evaluation"]["scan"] == "sequential"


def test_scan_changes_do_not_reuse_evaluation_progress(tmp_path):
    config = small_config(tmp_path)
    config["evaluation"]["scan"] = "sequential"
    expert = create_expert(config["data"])
    model = construct_model(config, expert, torch.device("cpu"))
    sequential = EvaluationProgress(model, expert, config, tmp_path)
    sequential.record("addition2 ULG", 2, True)
    legacy = deepcopy(config)
    del legacy["evaluation"]["scan"]
    assert EvaluationProgress(model, expert, legacy, tmp_path).path == sequential.path
    config["evaluation"]["scan"] = "parallel"
    parallel = EvaluationProgress(model, expert, config, tmp_path)
    assert parallel.path != sequential.path
    assert parallel.results("addition2 ULG") == {}


def test_evaluate_cli_overrides_scan_for_an_old_checkpoint(tmp_path):
    config = small_config(tmp_path)
    path = train(config)
    checkpoint_path = path / "checkpoints/latest.pt"
    checkpoint = read_checkpoint(checkpoint_path)
    del checkpoint["config"]["evaluation"]["scan"]
    atomic_save(checkpoint, checkpoint_path)
    output = tmp_path / "evaluation.json"
    main(["evaluate", str(path), "evaluation.scan=parallel", "--output", str(output)])
    result = json.loads(output.read_text())
    assert result["scan"] == "parallel" and result["step"] == 4
