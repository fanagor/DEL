"""Integration checks for actual training, masking, checkpoint continuation and evaluation."""

import json
import random
from copy import deepcopy
from pathlib import Path

import pytest
import torch

from differentiable_exact_learning.checkpoint import atomic_save, latest_checkpoint, read_checkpoint
from differentiable_exact_learning.config import experiment_config, load_config
from differentiable_exact_learning.data import create_expert
from differentiable_exact_learning.evaluation import evaluate, search_lengths
from differentiable_exact_learning.objectives import action_loss, exponential_lr
from differentiable_exact_learning.training import construct_model, load_run, train

torch.set_num_threads(1)


def small_config(tmp_path, name="DFST", environment="grid", source="online"):
    config = load_config(
        overrides=[
            f"model.name={name}",
            "model.hidden_size=4",
            f"data.environment={environment}",
            f"data.source={source}",
            "data.max_length=1",
            "data.size=4",
            "data.train_tasks=[parity,flip]",
            "data.eval_tasks=[parity,flip]",
            "training.steps=4",
            "training.batch_size=2",
            "training.log_every=2",
            "training.checkpoint_every=2",
            "training.warmup_steps=1",
            "evaluation.final=false",
            "evaluation.max_length=2",
            "evaluation.samples=2",
            "evaluation.batch_size=2",
            "evaluation.chunk_size=3",
            "evaluation.snapshot.m=4",
            "evaluation.snapshot.d=2",
            "run.device=cpu",
        ]
    )
    config["data"]["cache_dir"] = str(tmp_path / "cache")
    config["run"]["output_dir"] = str(tmp_path / "outputs")
    if name == "DFST_Softmax":
        config["model"]["init"].update(transition="random", symbol="random", motion="random")
        config["model"]["reg_lambda"] = 0.01
    return config


@pytest.mark.parametrize("name", ["DFST", "DFST_Softmax", "RNN", "GRU"])
@pytest.mark.parametrize("environment", ["grid", "tm", "teleporting"])
@pytest.mark.parametrize("source", ["online", "static"])
@pytest.mark.parametrize("embedded", [False, True])
@pytest.mark.parametrize("loss", ["mse", "cross_entropy"])
def test_train_every_supported_combination(tmp_path, name, environment, source, embedded, loss):
    config = small_config(tmp_path, name, environment, source)
    config["model"]["embed_dim"] = 3 if embedded else None
    config["data"]["batch_lengths"] = "mixed" if embedded else "fixed"
    config["training"]["loss"] = loss
    config["training"]["steps"] = 2
    path = train(config)
    model, expert, saved_config, checkpoint = load_run(path)
    assert checkpoint["step"] == 2
    assert saved_config["model"]["name"] == name
    assert expert.environment == environment
    assert all(torch.isfinite(value).all() for value in model.state_dict().values())
    metrics = [json.loads(line) for line in (path / "metrics.jsonl").read_text().splitlines()]
    assert metrics[-1]["step"] == 2
    assert 0 <= metrics[-1]["accuracy"] <= 1


@pytest.mark.parametrize("name", ["DFST", "DFST_Softmax", "RNN", "GRU"])
@pytest.mark.parametrize("source", ["online", "static"])
@pytest.mark.parametrize("environment", ["grid", "tm", "teleporting"])
def test_resume_matches_uninterrupted_training_exactly(tmp_path, name, source, environment):
    config = small_config(tmp_path, name, environment=environment, source=source)
    full = train(config, run_dir=tmp_path / "full")
    split = train(config, run_dir=tmp_path / "split", stop_after=3)
    # Simulate logs written after a checkpoint, before preemption.
    with (split / "metrics.jsonl").open("a") as stream:
        stream.write('{"step": 999, "loss": 999}\n')
    train(config, run_dir=split, resume=True)
    a = read_checkpoint(full / "checkpoints" / "latest.pt")
    b = read_checkpoint(split / "checkpoints" / "latest.pt")
    assert a["step"] == b["step"] == 4
    for key in a["model"]:
        torch.testing.assert_close(a["model"][key], b["model"][key], rtol=0, atol=0)
    assert a["rng"]["sampler"] == b["rng"]["sampler"]
    assert (full / "metrics.jsonl").read_text() == (split / "metrics.jsonl").read_text()


def test_checkpoint_fallback_and_configuration_guard(tmp_path):
    config = small_config(tmp_path)
    path = train(config)
    (path / "checkpoints" / "latest.pt").write_bytes(b"truncated")
    found, checkpoint = latest_checkpoint(path)
    assert found.name == "step_0000000004.pt"
    assert checkpoint["step"] == 4
    changed = deepcopy(config)
    changed["training"]["lr"] *= 2
    with pytest.raises(ValueError, match="same scientific"):
        train(changed, run_dir=path, resume=True)


def test_padding_does_not_change_loss_or_gradients():
    for kind in ("mse", "cross_entropy"):
        symbol = torch.randn(1, 2, 3, requires_grad=True)
        motion = torch.randn(1, 2, 2, requires_grad=True)
        labels = torch.tensor([[1, 0]])
        mask = torch.tensor([[True, False]])
        actual = action_loss(symbol, motion, labels, labels, mask, loss=kind)
        expected = action_loss(
            symbol[:, :1], motion[:, :1], labels[:, :1], labels[:, :1], mask[:, :1], loss=kind
        )
        torch.testing.assert_close(actual, expected)
        actual.backward()
        assert torch.equal(symbol.grad[:, 1], torch.zeros_like(symbol.grad[:, 1]))
        assert torch.equal(motion.grad[:, 1], torch.zeros_like(motion.grad[:, 1]))


def test_ce_probabilities_and_logits_are_equivalent():
    symbol, motion = torch.randn(2, 3, 4), torch.randn(2, 3, 2)
    ts, tm = torch.randint(4, (2, 3)), torch.randint(2, (2, 3))
    mask = torch.ones(2, 3, dtype=torch.bool)
    logits = action_loss(symbol, motion, ts, tm, mask, loss="cross_entropy")
    probs = action_loss(
        symbol.softmax(-1),
        motion.softmax(-1),
        ts,
        tm,
        mask,
        loss="cross_entropy",
        output_kind="probabilities",
    )
    torch.testing.assert_close(logits, probs)


def test_mse_matches_paper_equation():
    result = action_loss(
        torch.zeros(1, 1, 9),
        torch.zeros(1, 1, 5),
        torch.zeros(1, 1, dtype=torch.long),
        torch.zeros(1, 1, dtype=torch.long),
        torch.ones(1, 1, dtype=torch.bool),
    )
    assert result.item() == 1.0  # half of one squared unit error per head


def test_inverse_freq_weighting_normalizes_observable_triples_and_ignores_padding():
    # Triple (x=1, s=2, m=0) appears 4 times; triple (x=0, s=1, m=1) appears once; 6th step is pad.
    inputs = torch.tensor([[1, 1, 1, 1, 0, 1]])
    ts = torch.tensor([[2, 2, 2, 2, 1, 2]])
    tm = torch.tensor([[0, 0, 0, 0, 1, 0]])
    mask = torch.tensor([[True, True, True, True, True, False]])
    symbol = torch.zeros(1, 6, 4)
    motion = torch.zeros(1, 6, 3)
    # Put non-zero error ONLY on the unique step (index 4) so unweighted vs alpha=1 is exact:
    # At index 4, make symbol perfect (one-hot) on 0..3 and 0 on 4 (squared error = 1, m=0 -> 1).
    symbol[0, :4] = torch.nn.functional.one_hot(ts[0, :4], 4).to(torch.float32)
    motion[0, :4] = torch.nn.functional.one_hot(tm[0, :4], 3).to(torch.float32)
    # On index 4, all-zeros prediction gives MSE sum(-1) = 1.0 on both heads -> step loss = 1.0;
    # steps 0..3 have loss 0.0.
    unweighted = action_loss(symbol, motion, ts, tm, mask, inputs=inputs, inverse_freq_alpha=0.0)
    assert unweighted.item() == pytest.approx(1.0 / 5.0)
    # At alpha=1.0: the 4 repeated steps each get weight 1/4 (sum=1), unique step gets 1 (total=2).
    # So the unique step receives normalized weight 1 / 2 = 0.5!
    weighted = action_loss(symbol, motion, ts, tm, mask, inputs=inputs, inverse_freq_alpha=1.0)
    assert weighted.item() == pytest.approx(0.5)
    # Unpadded slice produces identical loss even when pad holds identical triple tokens.
    unpadded = action_loss(
        symbol[:, :5],
        motion[:, :5],
        ts[:, :5],
        tm[:, :5],
        mask[:, :5],
        inputs=inputs[:, :5],
        inverse_freq_alpha=1.0,
    )
    torch.testing.assert_close(weighted, unpadded)

    cfg0 = load_config()
    cfg_pos = load_config(overrides=["training.inverse_freq_alpha=0.5"])
    from differentiable_exact_learning.config import experiment_config

    assert "inverse_freq_alpha" not in experiment_config(cfg0)["training"]
    assert experiment_config(cfg_pos)["training"]["inverse_freq_alpha"] == 0.5


def test_mixed_batch_lengths_respect_curriculum_bound():
    for source in ("static", "online"):
        config = load_config(
            overrides=[
                f"data.source={source}",
                "data.batch_lengths=mixed",
                "data.max_length=4",
                "data.size=60",
                "training.arithmetic_curriculum.enabled=true",
                "training.arithmetic_curriculum.start_length=1",
            ]
        )
        from differentiable_exact_learning.data import BatchSampler, build_static_dataset

        traces = build_static_dataset(config["data"]) if source == "static" else None
        sampler = BatchSampler(create_expert(config["data"]), random.Random(11), 16, traces=traces)
        for cur_len in (1, 2, 3, 4):
            seen_tuples = set()
            for _ in range(6):
                batch = sampler.sample(cur_len)
                for meta in batch.metadata:
                    lengths = tuple(meta["lengths"])
                    assert max(lengths) <= cur_len
                    seen_tuples.add(lengths)
            if cur_len >= 2:
                assert len(seen_tuples) > 1


def test_schedule_endpoints():
    args = dict(total_steps=100, warmup_steps=20, peak=1e-2, floor=1e-7)
    assert exponential_lr(0, **args) == pytest.approx(1e-7)
    assert exponential_lr(20, **args) == pytest.approx(1e-2)
    assert exponential_lr(100, **args) == pytest.approx(1e-7)
    args["warmup_steps"] = 0
    assert exponential_lr(0, **args) == pytest.approx(1e-2)


def test_schedule_constant_steps_hold_the_peak():
    args = dict(total_steps=100, warmup_steps=20, peak=1e-2, floor=1e-7)
    for step in range(101):
        assert exponential_lr(step, **args, constant_steps=0) == exponential_lr(step, **args)
    held = dict(args, constant_steps=30)
    assert exponential_lr(10, **held) == exponential_lr(10, **args)
    assert exponential_lr(20, **held) == exponential_lr(49, **held) == pytest.approx(1e-2)
    assert exponential_lr(75, **held) == pytest.approx((1e-2 * 1e-7) ** 0.5)
    assert exponential_lr(100, **held) == pytest.approx(1e-7)
    with pytest.raises(ValueError):
        exponential_lr(0, **dict(args, constant_steps=80))


def test_constant_steps_validation_and_identity():
    assert "constant_steps" not in experiment_config(load_config())["training"]
    held = load_config(overrides=["training.warmup_steps=10", "training.constant_steps=20"])
    assert experiment_config(held)["training"]["constant_steps"] == 20
    steps = held["training"]["steps"]
    with pytest.raises(ValueError, match="constant_steps"):
        load_config(overrides=["training.warmup_steps=10", f"training.constant_steps={steps - 10}"])
    with pytest.raises(ValueError, match="constant_steps"):
        load_config(overrides=["training.constant_steps=-1"])


def test_search_boundaries_and_nonmonotonic_exhaustive():
    assert search_lengths(lambda n: True, 1)["length"] == 1
    assert search_lengths(lambda n: False, 1)["length"] == 0
    for threshold in range(9):
        result = search_lengths(lambda n: n <= threshold, 8)
        assert result["length"] == threshold
        assert result["at_cap"] == (threshold == 8)
    visited = []

    def probe(n):
        visited.append(n)
        return n in {1, 3}

    result = search_lengths(probe, 4, "exhaustive")
    assert result["length"] == 1
    assert visited == [1, 2]
    assert result["tested"] == [{"length": 1, "passed": True}, {"length": 2, "passed": False}]
    assert search_lengths(lambda n: False, 4, "exhaustive")["length"] == 0
    assert search_lengths(lambda n: True, 4, "exhaustive")["at_cap"]


def test_evaluation_matches_expert_and_preserves_rng(tmp_path):
    config = small_config(tmp_path)
    config["model"]["hidden_size"] = None
    expert = create_expert(config["data"])
    model = construct_model(config, expert, torch.device("cpu"))
    with torch.no_grad():
        for name, tensor in zip(("A", "B", "C"), expert.get_tensors()):
            getattr(model, name).copy_(tensor)
        model.h0.zero_()
        model.h0[0] = 1
    before_python, before_torch = random.getstate(), torch.get_rng_state()
    first = evaluate(model, expert, config)
    second = evaluate(model, expert, config)
    assert first == second
    assert first["tasks"]["addition2"]["rlg"]["length"] == 2
    assert first["tasks"]["addition2"]["rlg"]["at_cap"]
    assert random.getstate() == before_python
    assert torch.equal(torch.get_rng_state(), before_torch)
    assert model.training


@pytest.mark.parametrize(
    "override",
    [
        "training.mask_loss=false",
        "training.optimizer=sgd",
        "training.loss=nll",
        "training.warmup_steps=1000000",
        "model.softmax_out=true",
        "data.environment=fsa",
        "training.batch_size=0",
    ],
)
def test_reject_unsupported_or_invalid_configuration(override):
    with pytest.raises(ValueError):
        load_config(overrides=[override])


def test_no_teleporting_curriculum():
    with pytest.raises(ValueError, match="unavailable"):
        load_config(
            overrides=[
                "data.environment=teleporting",
                "training.arithmetic_curriculum.enabled=true",
            ]
        )


def test_curriculum_window_is_part_of_experiment_identity():
    config = load_config(overrides=["training.arithmetic_curriculum.enabled=true"])
    changed = deepcopy(config)
    changed["training"]["log_every"] *= 2
    assert experiment_config(config) != experiment_config(changed)


def test_resume_reads_checkpoint_under_run_lock(tmp_path, monkeypatch):
    import fcntl

    import differentiable_exact_learning.training as training

    config = small_config(tmp_path)
    path = train(config, stop_after=2)
    original = training.latest_checkpoint
    checked = []

    def assert_locked(run_dir):
        with (run_dir / ".lock").open("a") as other:
            with pytest.raises(BlockingIOError):
                fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
        checked.append(True)
        return original(run_dir)

    monkeypatch.setattr(training, "latest_checkpoint", assert_locked)
    train(config, run_dir=path, resume=True)
    assert checked == [True]


def test_second_writer_and_accidental_overwrite_are_rejected(tmp_path):
    import fcntl

    config = small_config(tmp_path)
    path = train(config)
    with pytest.raises(FileExistsError, match="already contains files"):
        train(config, run_dir=path)
    with (path / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="Another process"):
            train(config, run_dir=path, resume=True)


def test_evaluation_cli_accepts_options_between_overrides(tmp_path):
    from differentiable_exact_learning.cli import main

    config = small_config(tmp_path)
    path = train(config)
    output = tmp_path / "evaluation.json"
    main(
        [
            "evaluate",
            str(path),
            "evaluation.max_length=2",
            "--device",
            "cpu",
            "evaluation.max_length=1",
            "--output",
            str(output),
            "evaluation.samples=1",
        ]
    )
    result = json.loads(output.read_text())
    assert result["step"] == 4
    assert result["tasks"]["addition2"]["ulg"]["cap"] == 1
    assert result["random_samples"] == 1


def test_cli_rejects_unknown_flags():
    from differentiable_exact_learning.cli import main

    with pytest.raises(SystemExit) as error:
        main(["train", "--bogus", "training.steps=1"])
    assert error.value.code == 2


def test_cluster_cli_starts_and_resumes_completed_run(tmp_path):
    from differentiable_exact_learning.cli import main

    args = [
        "train",
        "--resume-auto",
        "--handle-signals",
        "--run-dir",
        str(tmp_path),
        "data.source=online",
        "data.max_length=1",
        "training.steps=2",
        "training.batch_size=2",
        "evaluation.final=false",
        "run.device=cpu",
    ]
    main(args)
    _, first = latest_checkpoint(tmp_path)
    main(args)
    _, resumed = latest_checkpoint(tmp_path)
    assert first["step"] == resumed["step"] == 2
    assert json.loads((tmp_path / "execution.json").read_text())["resumed_from_step"] == 2
    for key in first["model"]:
        assert torch.equal(first["model"][key], resumed["model"][key])


@pytest.mark.parametrize("name", ["DFST", "RNN"])
def test_expert_changes_create_new_runs_and_reject_old_checkpoints(tmp_path, monkeypatch, name):
    import differentiable_exact_learning.data.experts as experts

    config = small_config(tmp_path, name=name)
    path = train(config, stop_after=2)
    old = read_checkpoint(path / "checkpoints" / "latest.pt")
    original = experts.add_grid

    def changed_program(radix):
        transitions = original(radix)
        halt = max(state for state, _ in transitions)
        # Preserve all dimensions and executed traces. Even this otherwise
        # invisible table change must receive a distinct experiment identity.
        transitions[halt, ""] = (halt, "0", "S")
        return transitions

    monkeypatch.setattr(experts, "add_grid", changed_program)
    with pytest.raises(ValueError, match="expert fingerprint"):
        load_run(path)
    with pytest.raises(ValueError, match="expert fingerprint"):
        train(config, run_dir=path, resume=True)
    replacement = train(config, stop_after=2)
    assert replacement != path
    fresh = read_checkpoint(replacement / "checkpoints" / "latest.pt")
    assert fresh["expert_digest"] != old["expert_digest"]
    assert old["model"].keys() == fresh["model"].keys()


def test_checkpoints_without_expert_identity_require_explicit_migration(tmp_path):
    config = small_config(tmp_path)
    path = train(config, stop_after=2)
    checkpoint_path = path / "checkpoints" / "latest.pt"
    old = read_checkpoint(checkpoint_path)
    old.pop("expert_digest")
    atomic_save(old, checkpoint_path)
    with pytest.raises(ValueError, match="expert fingerprint is missing"):
        load_run(path)
    with pytest.raises(ValueError, match="expert fingerprint is missing"):
        train(config, run_dir=path, resume=True)


@pytest.mark.parametrize("source", ["static", "online"])
def test_tm_multiplication_trains_and_reloads(tmp_path, source):
    config = small_config(tmp_path, environment="tm", source=source)
    config["data"].update(task="multiplication", max_length=2)
    config["training"].update(steps=2, lr=1e-4)
    config["model"]["hidden_size"] = None
    path = train(config)
    model, expert, _, checkpoint = load_run(path)
    assert expert.environment == "tm" and expert.task == "multiplication"
    assert model.hidden_size == expert.n_states
    assert checkpoint["step"] == config["training"]["steps"]


@pytest.mark.parametrize(
    "recipe,states", [("add2", 11), ("add10", 19), ("mult2", 32), ("mult10", 108)]
)
def test_dfst_arithmetic_recipes_train_and_reload(tmp_path, recipe, states):
    config = load_config(
        Path(__file__).parents[1] / "configs" / "DFST" / f"{recipe}.yaml",
        overrides=[
            "data.source=online",
            "data.max_length=2",
            "training.steps=2",
            "training.warmup_steps=1",
            "training.batch_size=2",
            "training.log_every=1",
            "training.checkpoint_every=1",
            "evaluation.final=false",
            "run.device=cpu",
        ],
    )
    path = train(config, run_dir=tmp_path / recipe)
    model, expert, _, checkpoint = load_run(path)
    assert model.hidden_size == expert.n_states == states
    assert checkpoint["step"] == 2


def test_auto_resume_starts_empty_runs_and_continues_under_lock(tmp_path, monkeypatch):
    import fcntl

    import differentiable_exact_learning.training as training

    config = small_config(tmp_path)
    path = train(config, resume="auto", stop_after=2)
    original = training.latest_checkpoint
    checked = []

    def read_while_locked(folder):
        with (folder / ".lock").open("a") as lock:
            with pytest.raises(BlockingIOError):
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        checked.append(True)
        return original(folder)

    monkeypatch.setattr(training, "latest_checkpoint", read_while_locked)
    train(config, resume="auto")
    assert checked == [True]
    assert read_checkpoint(path / "checkpoints/latest.pt")["step"] == 4
    before = (path / "metrics.jsonl").read_bytes()
    train(config, resume="auto")  # Completed training adds no optimizer updates.
    assert (path / "metrics.jsonl").read_bytes() == before


def test_auto_resume_refuses_unknown_or_corrupt_existing_runs(tmp_path):
    config = small_config(tmp_path)
    path = tmp_path / "occupied"
    path.mkdir()
    (path / "unrelated.txt").write_text("keep")
    with pytest.raises(FileNotFoundError, match="No checkpoint"):
        train(config, resume="auto", run_dir=path)
    assert (path / "unrelated.txt").read_text() == "keep"
    path = train(config, run_dir=tmp_path / "corrupt", stop_after=2)
    for saved in (path / "checkpoints").glob("*.pt"):
        saved.write_bytes(b"truncated")
    with pytest.raises(RuntimeError, match="No readable checkpoint"):
        train(config, resume="auto", run_dir=path)


def test_auto_resume_keeps_scientific_configuration_guard(tmp_path):
    config = small_config(tmp_path)
    path = train(config, stop_after=2)
    changed = deepcopy(config)
    changed["training"]["lr"] /= 2
    with pytest.raises(ValueError, match="same scientific configuration"):
        train(changed, run_dir=path, resume="auto")


@pytest.mark.parametrize("model_name", ["DFST", "GRU"])
def test_sigterm_saves_complete_update_and_resumes_exactly(tmp_path, monkeypatch, model_name):
    import os
    import signal

    config = small_config(tmp_path, model_name)
    full = train(config, run_dir=tmp_path / "full")
    original = torch.optim.Adam.step
    previous_handler = signal.getsignal(signal.SIGTERM)
    killed = []

    def update_then_terminate(optimizer, *args, **kwargs):
        result = original(optimizer, *args, **kwargs)
        if not killed:
            killed.append(True)
            os.kill(os.getpid(), signal.SIGTERM)
        return result

    split = tmp_path / "split"
    with monkeypatch.context() as patch:
        patch.setattr(torch.optim.Adam, "step", update_then_terminate)
        with pytest.raises(SystemExit) as stop:
            train(config, resume="auto", run_dir=split, handle_signals=True)
    assert stop.value.code == 143
    assert signal.getsignal(signal.SIGTERM) == previous_handler
    assert read_checkpoint(split / "checkpoints/step_0000000000.pt")["step"] == 0
    interrupted = read_checkpoint(split / "checkpoints/latest.pt")
    assert interrupted["step"] == 1
    assert interrupted["aggregate"]["updates"] == 1
    train(config, resume="auto", run_dir=split, handle_signals=True)
    a = read_checkpoint(full / "checkpoints/latest.pt")
    b = read_checkpoint(split / "checkpoints/latest.pt")
    for key in a["model"]:
        torch.testing.assert_close(a["model"][key], b["model"][key], rtol=0, atol=0)
    assert a["rng"]["sampler"] == b["rng"]["sampler"]
    assert (full / "metrics.jsonl").read_bytes() == (split / "metrics.jsonl").read_bytes()


def test_null_run_name_trains_and_reloads(tmp_path):
    config = small_config(tmp_path)
    config["run"]["name"] = None
    path = train(config, stop_after=2)
    assert path.name.startswith("run-")
    _, _, restored, checkpoint = load_run(path)
    assert restored["run"]["name"] is None
    assert checkpoint["step"] == 2
