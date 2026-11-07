"""Autonomous playback, sparse rune updates, model loading, and local HTTP contracts."""

import json
from copy import deepcopy

import pytest
import torch

from differentiable_exact_learning.checkpoint import atomic_save
from differentiable_exact_learning.config import load_config
from differentiable_exact_learning.data import create_expert
from differentiable_exact_learning.data.datasets import dataset_spec
from differentiable_exact_learning.models import build_model
from differentiable_exact_learning.viewer.controller import Session, initial_memory, load_controller
from differentiable_exact_learning.viewer.server import make_server


def finish(session):
    events = []
    while event := session.advance():
        events.append(event)
    return events


@pytest.mark.parametrize(
    "selector,text",
    [
        ("grid:add2", "0011+001"),
        ("grid:add10", "999+1"),
        ("grid:mult2", "11x101"),
        ("grid:mult10", "12×34"),
        ("grid_new:add2", "0011+001"),
        ("grid_new:add10", "999+1"),
        ("grid_new:mult2", "11x101"),
        ("grid_new:mult10", "12×34"),
        ("tm:add2", "11+101"),
        ("tm:mult2", "11*101"),
        ("tp:parity", "101"),
        ("tp:copy", "101"),
        ("tp:reverse", ""),
    ],
)
def test_expert_events_match_real_simulator_and_replay_every_cell(selector, text):
    source = load_controller(selector)
    session = Session(source, text)
    initial = session.snapshot()
    assert session.step == 0  # Preview never advances the machine.
    cells = {(x, y): s for x, y, s in initial["cells"]}
    memory, _, _ = initial_memory(source, text)
    original = dict(memory.cells)
    # The existing expert accepts rows, so reconstruct the initial rectangle.
    ys = [p[1] for p in original] + [memory.head[1]]
    if memory.teleporting:
        rows = [
            [original.get((x, -y), "") for x in range(max(p[0] for p in original) + 1)]
            for y in range(-min(ys) + 1)
        ]
    else:
        rows = [[original.get((x, 0), "") for x in range(max(p[0] for p in original) + 1)]]
    expected = source.expert.simulate(rows, memory.head)
    events = finish(session)
    assert session.status == "halted"
    assert len(events) == len(expected.trace)
    for i, event in enumerate(events):
        assert event["read"] == source.expert.vocab_rev[expected.trace.inputs[i].item()]
        assert event["write"] == source.expert.vocab_rev[expected.trace.symbols[i].item()]
        assert event["motion"] == source.expert.motion_vocab_rev[expected.trace.motions[i].item()]
        for x, y, s in event["patches"]:
            if s:
                cells[x, y] = s
            else:
                cells.pop((x, y), None)
        if event["jump"] and event["jump"]["available"]:
            assert event["head"] == event["runes"][event["jump"]["rune"]]["position"]
    final = {tuple(session.position(p)): s for p, s in expected.cells.items()}
    assert cells == final
    assert session.memory.cells == expected.cells
    assert session.memory.head == expected.head


@pytest.mark.parametrize("task", ["parity", "flip", "shift_left", "shift_right", "copy", "reverse"])
def test_direct_binary_tm_catalog(task):
    session = Session(load_controller("tm:" + task), "101")
    finish(session)
    assert session.status == "halted"
    assert all(y == 0 for x, y in session.memory.cells)
    if task == "parity":
        assert session.memory.cells == {(3, 0): "0"}
    if task == "flip":
        assert session.memory.cells == {(0, 0): "0", (1, 0): "1", (2, 0): "0"}


def model_config(name, embedded=False):
    config = load_config(
        overrides=[
            f"model.name={name}",
            "data.source=online",
            "data.teaching_set=false",
            "model.hidden_size=5",
        ]
    )
    config["model"]["init"].update(transition="random", symbol="random", motion="random")
    if embedded:
        config["model"]["embed_dim"] = 4
    if name in {"RNN", "GRU"}:
        config["model"]["num_layers"] = 2
    return config


@pytest.mark.parametrize("name", ["DFST", "DFST_Softmax", "RNN", "GRU"])
@pytest.mark.parametrize("embedded", [False, True])
@pytest.mark.parametrize(
    "environment,text", [("grid", "101+11"), ("tm", "101+11"), ("teleporting", "101")]
)
def test_all_model_families_follow_own_outputs_and_preserve_opaque_state(
    name, embedded, environment, text
):
    config = model_config(name, embedded)
    config["data"]["environment"] = environment
    expert = create_expert(config["data"])
    torch.manual_seed(42)
    model = build_model(
        config["model"], expert.input_size, expert.symbol_size, expert.motion_size, expert.n_states
    ).eval()
    source = load_controller(model, config=config)
    session = Session(source, text, max_steps=8)
    reference = deepcopy(model)
    state = None
    with torch.inference_mode():
        for _ in range(8):
            if session.status in {"halted", "limit"}:
                break
            token = expert.vocab[session.memory.read()]
            symbols, motions, state = reference(torch.tensor([[token]]), state)
            event = session.advance()
            assert event["status"] != "error"
            assert event["write"] == expert.vocab_rev[int(symbols.argmax())]
            assert event["motion"] == expert.motion_vocab_rev[int(motions.argmax())]
            if not model.supports_state_rescaling:
                torch.testing.assert_close(session.state, state)
    assert model.training is False


@pytest.mark.parametrize("softmax", [False, True])
@pytest.mark.parametrize(
    "environment,text", [("grid", "111+1"), ("tm", "111+1"), ("teleporting", "101")]
)
def test_exact_model_closed_loop_matches_expert(softmax, environment, text):
    config = load_config(
        overrides=[
            "data.source=online",
            "data.teaching_set=false",
            f"model.name={'DFST_Softmax' if softmax else 'DFST'}",
            f"data.environment={environment}",
        ]
    )
    expert = create_expert(config["data"])
    model = build_model(
        config["model"], expert.input_size, expert.symbol_size, expert.motion_size, expert.n_states
    )
    with torch.no_grad():
        for parameter, matrix in zip((model.A, model.B, model.C), expert.get_tensors()):
            parameter.copy_(matrix.clamp_min(1e-30).log() if softmax else matrix)
        h = torch.zeros_like(model.h0)
        h[0] = 1
        model.h0.copy_(h.clamp_min(1e-30).log() if softmax else h)
    actual = Session(load_controller(model, config=config), text)
    expected = Session(load_controller(expert), text)
    a, b = finish(actual), finish(expected)
    assert [(e["read"], e["write"], e["motion"], e["head"]) for e in a] == [
        (e["read"], e["write"], e["motion"], e["head"]) for e in b
    ]


@pytest.mark.parametrize("name", ["DFST", "DFST_Softmax", "RNN", "GRU"])
def test_raw_weights_and_saved_checkpoints(tmp_path, name):
    config = model_config(name, embedded=True)
    expert = create_expert(config["data"])
    model = build_model(
        config["model"], expert.input_size, expert.symbol_size, expert.motion_size, expert.n_states
    )
    raw = tmp_path / "weights.pt"
    torch.save(model.state_dict(), raw)
    loaded = load_controller(raw, config=config)
    assert loaded.model.embedding.width == 4
    run = tmp_path / "run"
    atomic_save(
        {
            "format_version": 1,
            "config": config,
            "model": model.state_dict(),
            "expert_digest": dataset_spec(config["data"])["expert_digest"],
        },
        run / "checkpoints" / "latest.pt",
    )
    assert load_controller(run).label.startswith(name)
    assert load_controller(run / "checkpoints" / "latest.pt").label.startswith(name)
    with pytest.raises(ValueError, match="Raw weights require"):
        load_controller(model.state_dict())


def test_nonfinite_predictions_do_not_write_and_limits_do_not_claim_halt():
    config = model_config("DFST")
    expert = create_expert(config["data"])
    model = build_model(
        config["model"], expert.input_size, expert.symbol_size, expert.motion_size, expert.n_states
    )
    with torch.no_grad():
        model.B.fill_(float("nan"))
    session = Session(load_controller(model, config=config), "1+1")
    initial = dict(session.memory.cells)
    assert session.advance()["status"] == "error"
    assert session.step == 0 and session.memory.cells == initial
    limited = Session(load_controller("grid:add2"), "1+1", max_steps=1)
    assert limited.advance()["status"] == "limit"
    assert limited.advance() is None


@pytest.mark.parametrize(
    "source,text",
    [
        ("grid:add2", "102+1"),
        ("grid:add2", "1x1"),
        ("grid:mult10", ""),
        ("tp:parity", "102"),
        ("tm:copy", "a"),
    ],
)
def test_invalid_input_fails_before_execution(source, text):
    with pytest.raises(ValueError):
        Session(load_controller(source), text)


def test_local_server_protocol_without_network_sockets():
    # Test the real request handler over in-memory streams, so ordinary pytest
    # also runs in sandboxes that prohibit loopback socket binding.
    import io
    from unittest.mock import patch

    class FakeServer:
        server_port = 54321

        def __init__(self, address, handler):
            self.handler = handler

    with patch("differentiable_exact_learning.viewer.server.ThreadingHTTPServer", FakeServer):
        server = make_server(load_controller("tp:copy"), "101")

    def request(method, path, payload=None, token=None, host="127.0.0.1:54321"):
        body = json.dumps(payload).encode() if payload is not None else b""
        raw = f"{method} {path} HTTP/1.0\r\nHost: {host}\r\nContent-Length: {len(body)}\r\n"
        if token:
            raw += f"X-Viewer-Token: {token}\r\n"

        class Connection:
            def makefile(self, *args):
                return io.BytesIO(raw.encode() + b"\r\n" + body)

            def sendall(self, data):
                self.output.write(data)

            output = io.BytesIO()

        connection = Connection()
        server.handler(connection, ("127.0.0.1", 1234), server)
        return connection.output.getvalue()

    page = request("GET", "/")
    assert b"200 OK" in page and b"Tape & grid" in page
    token = page.split(b'"token": "')[1].split(b'"')[0].decode()
    assert b"403 Forbidden" in request("POST", "/api/reset", {"input": "101"})
    assert b"403 Forbidden" in request("GET", "/", host="external.example")
    response = request("POST", "/api/reset", {"input": "101"}, token)
    data = json.loads(response.split(b"\r\n\r\n", 1)[1])
    assert data["step"] == 0 and data["head"][1] == data["tape_row"]
    result = request("POST", "/api/step", {"session": data["session"], "count": 3}, token)
    assert len(json.loads(result.split(b"\r\n\r\n", 1)[1])["events"]) == 3
    assert b"400 Bad Request" in request(
        "POST", "/api/step", {"session": data["session"], "count": 65}, token
    )
    assert b"404 Not Found" in request("GET", "/../../config.yaml")
    assert b"200 OK" in request("GET", "/camera.mjs")
