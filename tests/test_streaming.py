"""Streaming equivalence, continuation state, and bounded generation regressions."""

import random

import pytest
import torch
from test_models import make_model
from test_training import small_config

from differentiable_exact_learning.data import StepLimitExceeded, Trace, create_expert, pad_traces
from differentiable_exact_learning.evaluation import evaluate, snapshot_metrics, traces_match
from differentiable_exact_learning.objectives import action_counts, action_loss
from differentiable_exact_learning.training import construct_model


def split_trace(trace, sizes=(1, 4, 2)):
    offset = 0
    index = 0
    while offset < len(trace):
        end = offset + sizes[index % len(sizes)]
        yield Trace(*(getattr(trace, key)[offset:end] for key in ("inputs", "symbols", "motions")))
        offset = end
        index += 1


@pytest.mark.parametrize(
    "environment,task,radix",
    [
        ("grid", "addition", 2),
        ("grid", "addition", 10),
        ("grid", "multiplication", 2),
        ("grid", "multiplication", 10),
        ("tm", "addition", 2),
        ("tm", "multiplication", 2),
        ("teleporting", "parity", 2),
        ("teleporting", "copy", 2),
        ("teleporting", "reverse", 2),
        ("teleporting", "scaling_A_m3", 2),
    ],
)
@pytest.mark.parametrize("chunk_size", [1, 7, 256])
def test_stream_matches_full_trace_metadata_and_rng(environment, task, radix, chunk_size):
    expert = create_expert(dict(environment=environment, task=task, radix=radix))
    full_rng, stream_rng = random.Random(73), random.Random(73)
    args = {"task": task} if environment == "teleporting" else {}
    expected = [expert.sample(full_rng, 3, **args) for _ in range(2)]
    streams = [expert.sample_stream(stream_rng, 3, chunk_size=chunk_size, **args) for _ in range(2)]
    assert full_rng.getstate() == stream_rng.getstate()
    # Input draws must not depend on when or in which order streams advance.
    for stream, trace in reversed(list(zip(streams, expected))):
        chunks = list(stream)
        assert all(0 < len(chunk) <= chunk_size for chunk in chunks)
        assert all(chunk.metadata == trace.metadata for chunk in chunks)
        for key in ("inputs", "symbols", "motions"):
            actual = torch.cat([getattr(chunk, key) for chunk in chunks])
            assert torch.equal(actual, getattr(trace, key))
    assert full_rng.getstate() == stream_rng.getstate()


def test_halt_and_finite_limit_apply_across_chunks():
    expert = create_expert({"environment": "grid"})
    rows = [list("11+1")]
    expected = expert.simulate(rows).trace
    for chunk_size in (1, len(expected), len(expected) + 1):
        chunks = list(expert.iter_simulate(rows, max_steps=len(expected), chunk_size=chunk_size))
        assert sum(map(len, chunks)) == len(expected)
        assert torch.equal(torch.cat([c.inputs for c in chunks]), expected.inputs)
        with pytest.raises(StepLimitExceeded):
            list(expert.iter_simulate(rows, max_steps=len(expected) - 1, chunk_size=chunk_size))


@pytest.mark.parametrize("name", ["DFST", "DFST_Softmax", "RNN", "GRU"])
@pytest.mark.parametrize("embed_dim", [None, 5])
def test_chunk_continuation_and_exact_match_with_mixed_lengths(name, embed_dim):
    torch.manual_seed(91)
    model = make_model(name, embed_dim=embed_dim).eval()
    traces = []
    with torch.no_grad():
        for length in (2, 11, 5, 7):
            tokens = torch.randint(3, (1, length))
            symbols, motions, expected_state = model(tokens)
            state = None
            pieces = []
            for start in range(0, length, 3):
                s, m, state = model.forward_chunk(tokens[:, start : start + 3], state)
                pieces.append((s, m))
            torch.testing.assert_close(state, expected_state)
            torch.testing.assert_close(torch.cat([p[0] for p in pieces], dim=1), symbols)
            torch.testing.assert_close(torch.cat([p[1] for p in pieces], dim=1), motions)
            # Use the existing full-token inference interface as the reference
            # for exact-match labels, including ordinary DFST stabilization.
            chunks = list(model.iter_outputs(tokens, chunk_size=length, stabilize=True))
            traces.append(
                Trace(
                    tokens[0],
                    torch.cat([c[0] for c in chunks], dim=1)[0].argmax(-1),
                    torch.cat([c[1] for c in chunks], dim=1)[0].argmax(-1),
                )
            )
    for batch_size, chunk_size in ((1, 1), (3, 4), (100, 8)):
        assert traces_match(
            model,
            (split_trace(t) for t in traces),
            batch_size=batch_size,
            chunk_size=chunk_size,
        )
    # Require correctness through the final, partial chunk of the longest row.
    traces[1].motions[-1] = (traces[1].motions[-1] + 1) % model.motion_size
    assert not traces_match(model, (split_trace(t) for t in traces), batch_size=3, chunk_size=4)


@pytest.mark.parametrize("name", ["DFST", "DFST_Softmax", "RNN", "GRU"])
@pytest.mark.parametrize("embed_dim", [None, 3])
@pytest.mark.parametrize("loss", ["mse", "cross_entropy"])
def test_streamed_snapshots_match_full_forward(tmp_path, name, embed_dim, loss):
    config = small_config(tmp_path, name)
    config["model"]["embed_dim"] = embed_dim
    if name == "DFST":
        config["model"]["init"].update(symbol="random", motion="random")
    expert = create_expert(config["data"])
    model = construct_model(config, expert, torch.device("cpu")).eval()
    expected = []
    with torch.no_grad():
        for length in (1, 3, 2):
            trace = expert.sample(random.Random(length), length, lengths=(length, length))
            batch = pad_traces([trace])
            symbols, motions, _ = model(batch.inputs)
            value = action_loss(
                symbols,
                motions,
                batch.symbols,
                batch.motions,
                batch.mask,
                loss=loss,
                output_kind=model.output_kind,
            )
            correct, total, _ = action_counts(
                symbols, motions, batch.symbols, batch.motions, batch.mask
            )
            expected.append((float(value), correct / total, total))
    for batch_size, chunk_size in ((1, 2), (3, 7)):
        samples = (
            (
                length,
                expert.sample_stream(
                    random.Random(length), length, lengths=(length, length), chunk_size=3
                ),
            )
            for length in (1, 3, 2)
        )
        actual = snapshot_metrics(
            model, samples, batch_size=batch_size, chunk_size=chunk_size, loss=loss
        )
        for point, (value, accuracy, total) in zip(actual, expected):
            assert point["loss"] == pytest.approx(value, rel=2e-5, abs=1e-6)
            assert point["accuracy"] == accuracy
            assert point["timesteps"] == total


@pytest.mark.parametrize("environment", ["grid", "tm", "teleporting"])
def test_evaluation_uses_streaming_for_search_and_snapshots(tmp_path, monkeypatch, environment):
    config = small_config(tmp_path, "GRU", environment)
    config["evaluation"]["chunk_size"] = 4
    expert = create_expert(config["data"])
    model = construct_model(config, expert, torch.device("cpu")).eval()

    def forbidden(*args, **kwargs):
        raise AssertionError("Evaluation must not materialize an expert execution")

    monkeypatch.setattr(expert, "sample", forbidden)
    monkeypatch.setattr(expert, "simulate", forbidden)
    result = evaluate(model, expert, config)
    for task in result["tasks"].values():
        assert len(task["snapshot"]["points"]) == 2
        assert all(point["timesteps"] > 0 for point in task["snapshot"]["points"])


def test_5000_digit_search_stops_generation_at_first_failing_chunk(tmp_path, monkeypatch):
    config = small_config(tmp_path, "GRU")
    config["evaluation"].update(
        max_length=5000, samples=10, batch_size=100, chunk_size=64, hard_cases="none"
    )
    config["evaluation"]["snapshot"]["enabled"] = False
    assert config["data"]["max_steps"] is None
    expert = create_expert(config["data"])
    model = construct_model(config, expert, torch.device("cpu")).eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()  # Predict blank/L; addition starts with a digit/R.
    original_steps, original_chunk = expert._steps, model.forward_chunk
    generated, shapes = [], []

    def steps(memory, max_steps):
        count = 0
        try:
            for action in original_steps(memory, max_steps):
                count += 1
                yield action
        finally:
            generated.append(count)

    def forward(tokens, state=None, **kwargs):
        shapes.append(tuple(tokens.shape))
        return original_chunk(tokens, state, **kwargs)

    def forbidden(*args, **kwargs):
        raise AssertionError("A 5000-digit trace must not be generated in full")

    monkeypatch.setattr(expert, "_steps", steps)
    monkeypatch.setattr(expert, "sample", forbidden)
    monkeypatch.setattr(expert, "simulate", forbidden)
    monkeypatch.setattr(model, "forward_chunk", forward)
    result = evaluate(model, expert, config)["tasks"]["addition2"]["ulg"]
    assert result["cap"] == 5000
    assert {"length": 5000, "passed": False} in result["tested"]
    assert result["length"] == 0
    assert len(shapes) == len(result["tested"])
    assert all(batch <= 10 and length <= 64 for batch, length in shapes)
    assert generated and max(generated) <= 64
    assert sum(generated) <= 64 * 10 * len(result["tested"])


def test_stream_errors_propagate_and_model_mode_is_restored(tmp_path):
    config = small_config(tmp_path)
    config["model"]["hidden_size"] = None
    config["evaluation"]["chunk_size"] = 2
    config["data"]["max_steps"] = 4
    expert = create_expert(config["data"])
    model = construct_model(config, expert, torch.device("cpu"))
    with torch.no_grad():
        for key, tensor in zip(("A", "B", "C"), expert.get_tensors()):
            getattr(model, key).copy_(tensor)
        model.h0.zero_()
        model.h0[0] = 1
    with pytest.raises(StepLimitExceeded):
        evaluate(model, expert, config)
    assert model.training


@pytest.mark.parametrize("environment", ["grid", "tm", "teleporting"])
def test_complete_evaluation_matches_materialized_expert_reference(
    tmp_path, monkeypatch, environment
):
    config = small_config(tmp_path, environment=environment)
    config["model"]["hidden_size"] = None
    expert = create_expert(config["data"])
    model = construct_model(config, expert, torch.device("cpu"))
    with torch.no_grad():
        for key, tensor in zip(("A", "B", "C"), expert.get_tensors()):
            getattr(model, key).copy_(tensor)
        model.h0.zero_()
        model.h0[0] = 1
    streamed = evaluate(model, expert, config)

    def materialized(rng, max_length, *, chunk_size, **kwargs):
        return iter([expert.sample(rng, max_length, **kwargs)])

    monkeypatch.setattr(expert, "sample_stream", materialized)
    assert streamed == evaluate(model, expert, config)
    for metrics in streamed["tasks"].values():
        measurement = metrics["generalization"] if environment == "teleporting" else metrics["ulg"]
        assert measurement["length"] == config["evaluation"]["max_length"]
        assert measurement["at_cap"]


def test_empty_stream_does_not_count_as_correct(tmp_path):
    config = small_config(tmp_path)
    expert = create_expert(config["data"])
    model = construct_model(config, expert, torch.device("cpu"))
    with pytest.raises(ValueError, match="empty"):
        traces_match(model, [iter(())])
