"""Recovery at the checkpoint/network boundary with a stateful fake W&B server."""

import json
import os
import signal
from copy import deepcopy

import pytest
import torch
from test_training import small_config

from differentiable_exact_learning.checkpoint import read_checkpoint
from differentiable_exact_learning.training import train


def tracked_config(tmp_path):
    config = small_config(tmp_path, "GRU")
    config["run"]["wandb"]["enabled"] = True
    config["training"].update(steps=6, log_every=2, checkpoint_every=2)
    return config


def record(service, path):
    identity = json.loads((path / "wandb-run.json").read_text())
    return service.runs[identity["id"]]


def assert_weights_equal(a, b):
    for key in a["model"]:
        torch.testing.assert_close(a["model"][key], b["model"][key], rtol=0, atol=0)


def test_resume_reuses_id_and_matches_uninterrupted_training(tmp_path, fake_wandb):
    config = tracked_config(tmp_path)
    full = train(config, run_dir=tmp_path / "full")
    split = train(config, run_dir=tmp_path / "split", stop_after=3)
    identifier = json.loads((split / "wandb-run.json").read_text())["id"]
    train(config, run_dir=split, resume=True)
    assert fake_wandb.starts[-1]["id"] == identifier
    assert fake_wandb.starts[-1]["resume"] == "must"
    assert fake_wandb.starts[-1]["entity"] == "default-team"
    assert len(fake_wandb.runs) == 2  # Two experiments, not three executions.
    assert record(fake_wandb, full)["rows"] == record(fake_wandb, split)["rows"]
    assert (full / "metrics.jsonl").read_bytes() == (split / "metrics.jsonl").read_bytes()
    a, b = [read_checkpoint(p / "checkpoints/latest.pt") for p in (full, split)]
    assert_weights_equal(a, b)
    assert a["rng"]["sampler"] == b["rng"]["sampler"]
    before = list(record(fake_wandb, split)["rows"])
    train(config, run_dir=split, resume=True)
    assert record(fake_wandb, split)["rows"] == before


def test_tracking_does_not_change_training_randomness(tmp_path, fake_wandb):
    config = tracked_config(tmp_path)
    tracked = train(config, run_dir=tmp_path / "tracked")
    config["run"]["wandb"]["enabled"] = False
    plain = train(config, run_dir=tmp_path / "plain")
    assert_weights_equal(*(read_checkpoint(p / "checkpoints/latest.pt") for p in (tracked, plain)))
    assert (tracked / "metrics.jsonl").read_bytes() == (plain / "metrics.jsonl").read_bytes()


def test_every_upload_has_a_durable_checkpoint(tmp_path, fake_wandb):
    config = tracked_config(tmp_path)
    config["training"]["checkpoint_every"] = 5
    path = tmp_path / "run"
    observed = []

    def check(run, row, index):
        saved = read_checkpoint(path / "checkpoints/latest.pt")
        assert saved["step"] >= row["step"]
        assert saved["tracking"]["identity"]["id"] == run.id
        assert saved["tracking"]["cursor"] > index
        observed.append(index)

    fake_wandb.before_log = check
    train(config, run_dir=path)
    assert observed == [0, 1, 2, 3]
    assert sorted(p.name for p in (path / "checkpoints").glob("step_*.pt")) == [
        "step_0000000000.pt",
        "step_0000000005.pt",
        "step_0000000006.pt",
    ]


def test_network_failure_before_first_connection_can_retry(tmp_path, fake_wandb):
    config = tracked_config(tmp_path)
    path = tmp_path / "run"

    def fail(arguments):
        assert read_checkpoint(path / "checkpoints/latest.pt")["step"] == 0
        raise ConnectionError("unavailable")

    fake_wandb.before_init = fail
    with pytest.raises(ConnectionError):
        train(config, run_dir=path, resume="auto")
    identifier = json.loads((path / "wandb-run.json").read_text())["id"]
    fake_wandb.before_init = None
    train(config, run_dir=path, resume="auto")
    assert fake_wandb.starts[-1]["id"] == identifier
    assert len(fake_wandb.runs) == 1


def test_committed_but_unuploaded_history_replays_once(tmp_path, fake_wandb):
    config = tracked_config(tmp_path)
    fake_wandb.accept_logs = False  # SDK accepted log(), but no upload reached the server.
    path = train(config)
    assert record(fake_wandb, path)["rows"] == []
    fake_wandb.accept_logs = True
    train(config, run_dir=path, resume=True)
    assert [row["step"] for row in record(fake_wandb, path)["rows"]] == [0, 2, 4, 6]
    train(config, run_dir=path, resume=True)
    assert len(record(fake_wandb, path)["rows"]) == 4


@pytest.mark.parametrize("after_ack", [False, True])
def test_failed_log_call_recovers_with_or_without_server_ack(tmp_path, fake_wandb, after_ack):
    config = tracked_config(tmp_path)
    path = tmp_path / "run"

    def fail(run, row, index):
        if row["step"] == 4:
            raise ConnectionError("ambiguous upload")

    attribute = "after_log" if after_ack else "before_log"
    setattr(fake_wandb, attribute, fail)
    with pytest.raises(ConnectionError):
        train(config, run_dir=path)
    assert read_checkpoint(path / "checkpoints/latest.pt")["step"] == 4
    setattr(fake_wandb, attribute, None)
    train(config, run_dir=path, resume="auto")
    assert [row["step"] for row in record(fake_wandb, path)["rows"]] == [0, 2, 4, 6]


def test_crash_between_journal_and_checkpoint_discards_uncommitted_tail(
    tmp_path, fake_wandb, monkeypatch
):
    import differentiable_exact_learning.training as training

    config = tracked_config(tmp_path)
    path = tmp_path / "run"
    original = training.atomic_save

    def fail(state, target):
        if state["step"] == 4:
            raise OSError("disk interrupted")
        original(state, target)

    with monkeypatch.context() as patch:
        patch.setattr(training, "atomic_save", fail)
        with pytest.raises(OSError):
            train(config, run_dir=path)
    assert [row["step"] for row in record(fake_wandb, path)["rows"]] == [0, 2]
    with (path / "wandb-history.jsonl").open("ab") as stream:
        stream.write(b'{"index":')
    train(config, run_dir=path, resume="auto")
    assert [row["step"] for row in record(fake_wandb, path)["rows"]] == [0, 2, 4, 6]
    assert fake_wandb.rewinds == []


def test_older_checkpoint_rewinds_same_id_and_replays_evaluations(tmp_path, fake_wandb):
    config = tracked_config(tmp_path)
    config["evaluation"].update(every=2, final=True)
    config["evaluation"]["snapshot"]["enabled"] = False
    path = train(config)
    expected = deepcopy(record(fake_wandb, path)["rows"])
    identifier = fake_wandb.starts[-1]["id"]
    for filename in ("latest.pt", "step_0000000006.pt"):
        (path / "checkpoints" / filename).write_bytes(b"corrupt")
    train(config, run_dir=path, resume="auto")
    assert len(fake_wandb.runs) == 1
    assert fake_wandb.rewinds == [(identifier, 3)]
    assert record(fake_wandb, path)["rows"] == expected
    train(config, run_dir=path, resume="auto")
    assert record(fake_wandb, path)["rows"] == expected


def test_rollback_to_step_zero_keeps_valid_rewind_anchor(tmp_path, fake_wandb):
    config = tracked_config(tmp_path)
    config["training"]["checkpoint_every"] = 100
    path = train(config)
    expected = deepcopy(record(fake_wandb, path)["rows"])
    for filename in ("latest.pt", "step_0000000006.pt"):
        (path / "checkpoints" / filename).unlink()
    train(config, run_dir=path, resume="auto")
    assert fake_wandb.rewinds[-1][1] == 0
    assert record(fake_wandb, path)["rows"] == expected


def test_rewind_failure_stops_without_creating_another_run(tmp_path, fake_wandb):
    config = tracked_config(tmp_path)
    path = train(config)
    expected = deepcopy(record(fake_wandb, path)["rows"])
    for filename in ("latest.pt", "step_0000000006.pt"):
        (path / "checkpoints" / filename).unlink()
    fake_wandb.reject_rewind = True
    with pytest.raises(RuntimeError, match="rewind failed"):
        train(config, run_dir=path, resume=True)
    assert len(fake_wandb.runs) == 1
    assert record(fake_wandb, path)["rows"] == expected
    assert read_checkpoint(path / "checkpoints/latest.pt")["step"] == 4
    fake_wandb.reject_rewind = False
    train(config, run_dir=path, resume=True)
    assert record(fake_wandb, path)["rows"] == expected


def test_sigterm_recovers_same_run_without_duplicate_metrics(tmp_path, fake_wandb, monkeypatch):
    config = tracked_config(tmp_path)
    path = tmp_path / "run"
    original = torch.optim.Adam.step
    killed = []

    def update(optimizer, *args, **kwargs):
        result = original(optimizer, *args, **kwargs)
        if not killed:
            killed.append(True)
            os.kill(os.getpid(), signal.SIGTERM)
        return result

    with monkeypatch.context() as patch:
        patch.setattr(torch.optim.Adam, "step", update)
        with pytest.raises(SystemExit) as stopped:
            train(config, run_dir=path, resume="auto", handle_signals=True)
    assert stopped.value.code == 143
    assert fake_wandb.finishes[-1][1] == 1
    train(config, run_dir=path, resume="auto", handle_signals=True)
    assert len(fake_wandb.runs) == 1
    assert [row["step"] for row in record(fake_wandb, path)["rows"]] == [0, 2, 4, 6]


def test_offline_runs_keep_one_identity_and_upload_after_restart(tmp_path, fake_wandb, monkeypatch):
    config = tracked_config(tmp_path)
    monkeypatch.setenv("WANDB_MODE", "offline")
    path = train(config, stop_after=3)
    identifier = json.loads((path / "wandb-run.json").read_text())["id"]
    train(config, run_dir=path, resume=True)
    assert fake_wandb.starts == []
    monkeypatch.setenv("WANDB_MODE", "online")
    train(config, run_dir=path, resume=True)
    assert fake_wandb.starts[0]["id"] == identifier
    assert [row["step"] for row in record(fake_wandb, path)["rows"]] == [0, 2, 4, 6]


def test_offline_rollback_remembers_obsolete_remote_history(tmp_path, fake_wandb, monkeypatch):
    config = tracked_config(tmp_path)
    path = train(config)
    for filename in ("latest.pt", "step_0000000006.pt"):
        (path / "checkpoints" / filename).unlink()
    record(fake_wandb, path)["rows"][-1]["obsolete"] = True
    monkeypatch.setenv("WANDB_MODE", "offline")
    train(config, run_dir=path, resume=True)
    # New local history has caught up in length, but the old remote tail still
    # belongs to the discarded execution and must be removed before replay.
    assert fake_wandb.rewinds == []
    monkeypatch.setenv("WANDB_MODE", "online")
    train(config, run_dir=path, resume=True)
    assert fake_wandb.rewinds[-1][1] == 2
    assert not any("obsolete" in row for row in record(fake_wandb, path)["rows"])


def test_checkpoint_before_first_tracked_step_fails_safely(tmp_path, fake_wandb):
    config = tracked_config(tmp_path)
    config["run"]["wandb"]["enabled"] = False
    path = train(config, stop_after=4)
    old = (path / "checkpoints/step_0000000002.pt").read_bytes()
    config["run"]["wandb"]["enabled"] = True
    train(config, run_dir=path, resume=True)
    (path / "checkpoints/latest.pt").write_bytes(old)
    with pytest.raises(ValueError, match="predates"):
        train(config, run_dir=path, resume=True)
    assert len(fake_wandb.starts) == 1


@pytest.mark.parametrize("field,value", [("entity", "other-team"), ("project", "other-project")])
def test_resume_cannot_redirect_existing_identity(tmp_path, fake_wandb, field, value):
    config = tracked_config(tmp_path)
    path = train(config)
    config["run"]["wandb"][field] = value
    with pytest.raises(ValueError, match="same project/entity"):
        train(config, run_dir=path, resume=True)
    assert len(fake_wandb.starts) == 1


def test_missing_remote_run_is_not_silently_recreated(tmp_path, fake_wandb):
    config = tracked_config(tmp_path)
    path = train(config)
    fake_wandb.runs.clear()
    with pytest.raises(RuntimeError, match="Remote run missing"):
        train(config, run_dir=path, resume=True)
    assert fake_wandb.runs == {}


def test_checkpoint_recovers_missing_identity_sidecar(tmp_path, fake_wandb):
    config = tracked_config(tmp_path)
    path = train(config)
    identifier = fake_wandb.starts[0]["id"]
    (path / "wandb-run.json").unlink()
    train(config, run_dir=path, resume=True)
    assert json.loads((path / "wandb-run.json").read_text())["id"] == identifier
    assert len(fake_wandb.runs) == 1


@pytest.mark.parametrize("contents", [None, b'{"index":'])
def test_missing_committed_history_fails_before_network_access(tmp_path, fake_wandb, contents):
    config = tracked_config(tmp_path)
    path = train(config)
    journal = path / "wandb-history.jsonl"
    if contents is None:
        journal.unlink()
    else:
        journal.write_bytes(contents)
    with pytest.raises(ValueError, match="[Hh]istory"):
        train(config, run_dir=path, resume=True)
    assert len(fake_wandb.starts) == 1


@pytest.mark.parametrize("change", ["valid_json", "missing_newline"])
def test_committed_history_checksum_detects_changed_branch(tmp_path, fake_wandb, change):
    config = tracked_config(tmp_path)
    path = train(config, stop_after=0 if change == "missing_newline" else None)
    journal = path / "wandb-history.jsonl"
    if change == "missing_newline":
        journal.write_bytes(journal.read_bytes().rstrip(b"\n"))
    else:
        lines = journal.read_text().splitlines()
        last = json.loads(lines[-1])
        last["data"]["loss"] += 1
        lines[-1] = json.dumps(last)
        journal.write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match="checksum"):
        train(config, run_dir=path, resume=True)
    assert len(fake_wandb.starts) == 1


def test_temporarily_disabled_tracking_keeps_identity(tmp_path, fake_wandb):
    config = tracked_config(tmp_path)
    path = train(config, stop_after=2)
    config["run"]["wandb"]["enabled"] = False
    train(config, run_dir=path, resume=True, stop_after=4)
    config["run"]["wandb"]["enabled"] = True
    train(config, run_dir=path, resume=True)
    assert len(fake_wandb.runs) == 1
    assert [row["step"] for row in record(fake_wandb, path)["rows"]] == [0, 2, 6]


def test_tracking_can_start_on_an_untracked_checkpoint(tmp_path, fake_wandb):
    config = tracked_config(tmp_path)
    config["run"]["wandb"]["enabled"] = False
    path = train(config, stop_after=2)
    config["run"]["wandb"]["enabled"] = True
    train(config, run_dir=path, resume=True)
    assert [row["step"] for row in record(fake_wandb, path)["rows"]] == [2, 4, 6]
