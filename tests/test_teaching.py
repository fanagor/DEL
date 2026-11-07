"""Teaching cores cover all 8 expert variants and enforce static/size invariants."""

import hashlib
import json
import random
from copy import deepcopy

import pytest
import torch
from test_data import (
    _check_grid_addition_result,
    _check_grid_multiplication_result,
    _check_new_grid_multiplication_result,
    _result_value,
)

from differentiable_exact_learning.cli import main
from differentiable_exact_learning.config import experiment_config, load_config
from differentiable_exact_learning.data import (
    BatchSampler,
    build_static_dataset,
    create_expert,
    dataset_fingerprint,
    dataset_spec,
    task_transitions,
)
from differentiable_exact_learning.data.teaching import (
    TP_TEACHING_SPEC,
    addition_pairs,
    multiplication_pairs,
    teaching_manifest,
    teaching_set_size,
    teleporting_teaching_inputs,
)
from differentiable_exact_learning.training import load_run, train


def _events(expert, execution):
    return {
        (q, expert.vocab_rev[x]) for q, x in zip(execution.states, execution.trace.inputs.tolist())
    }


@pytest.mark.parametrize("radix", range(2, 11))
def test_grid_default_addition_core_matches_main_and_covers_all_reachable_rules(radix):
    pairs = addition_pairs(radix, environment="grid", variant="default")
    assert len(pairs) == len(set(pairs)) == radix**2 + 2 * radix
    assert set(pairs) == {(b, a) for a, b in pairs}
    assert max(len(w) for pair in pairs for w in pair) == 2
    expert = create_expert(
        dict(environment="grid", variant="default", task="addition", radix=radix)
    )
    covered = set()
    for a, b in pairs:
        ex = expert.simulate([list(a + "+" + b)])
        _check_grid_addition_result(ex, a, b, radix)
        covered |= _events(expert, ex)
    intended = {k for k, v in expert.transitions.items() if v != (expert.halt_state, k[1], "S")}
    intended |= {(radix + 7, str(d)) for d in range(radix)}
    assert covered == intended
    assert len(covered) == radix**2 + 9 * radix + 11
    h = str(radix - 1)
    expected = {(str(a), str(b)) for a in range(radix) for b in range(radix)}
    expected |= {(str(d) + h, "1") for d in range(radix)}
    expected |= {("1", str(d) + h) for d in range(radix)}
    assert set(pairs) == expected


@pytest.mark.parametrize("radix", range(2, 11))
def test_grid_new_addition_core_is_symmetric_3digit_and_covers_all_reachable_rules(radix):
    pairs = addition_pairs(radix, environment="grid", variant="new")
    assert len(pairs) == len(set(pairs)) == radix**2 + 8 * radix + 1
    assert set(pairs) == {(b, a) for a, b in pairs}
    assert max(len(w) for pair in pairs for w in pair) == 3
    # Redundant 0^m + 0^n grid is excluded (e.g. ("00", "00") is not included).
    assert ("00", "00") not in set(pairs)
    expert = create_expert(dict(environment="grid", variant="new", task="addition", radix=radix))
    covered = set()
    for a, b in pairs:
        ex = expert.simulate([list(a + "+" + b)])
        _check_grid_addition_result(ex, a, b, radix)
        covered |= _events(expert, ex)
    intended = {k for k, v in expert.transitions.items() if v != (expert.halt_state, k[1], "S")}
    intended |= {(expert.halt_state - 1, str(d)) for d in range(radix)}
    assert len(covered) == 11 * radix + 14 and covered == intended


@pytest.mark.parametrize("radix", range(2, 11))
def test_tm_addition_core_is_symmetric_3digit_and_covers_all_reachable_rules(radix):
    pairs = addition_pairs(radix, environment="tm", variant="default")
    assert len(pairs) == len(set(pairs))
    assert set(pairs) == {(b, a) for a, b in pairs}
    assert max(len(w) for pair in pairs for w in pair) == 3
    expert = create_expert(dict(environment="tm", task="addition", radix=radix))
    covered = set()
    for a, b in pairs:
        ex = expert.simulate([list(a) + ["+"] + list(b)])
        assert _result_value(ex, radix) == int(a, radix) + int(b, radix)
        covered |= _events(expert, ex)
    assert len(covered) == 3 * radix**2 + 8 * radix + 5


@pytest.mark.parametrize("radix", range(2, 11))
@pytest.mark.parametrize("variant", ["default", "new"])
def test_grid_multiplication_transition_core_is_symmetric_5digit_and_covers_all_rules(
    radix, variant
):
    pairs = multiplication_pairs(radix, environment="grid", variant=variant)
    count = (
        radix**3 + 6 * radix**2 + radix * (radix - 1) // 2 if variant == "default" else 7 * radix**2
    )
    assert len(pairs) == len(set(pairs)) == count
    assert set(pairs) == {(b, a) for a, b in pairs}
    assert max(len(w) for pair in pairs for w in pair) == 5
    assert {(str(a), str(b)) for a in range(radix) for b in range(radix)} <= set(pairs)
    expert = create_expert(
        dict(environment="grid", variant=variant, task="multiplication", radix=radix)
    )
    covered = set()
    check_result = (
        _check_grid_multiplication_result
        if variant == "default"
        else _check_new_grid_multiplication_result
    )
    for a, b in pairs:
        ex = expert.simulate([list(a + "x" + b)])
        check_result(ex, a, b, radix)
        covered |= _events(expert, ex)
    intended = {k for k, v in expert.transitions.items() if v != (expert.halt_state, k[1], "S")}
    if variant == "default":
        sum_entry = 2 * radix + 11 + radix * (radix + 1) // 2
        # A lower partial-product leading carry is <= R-2; add_value[R] never reads blank.
        intended -= {(sum_entry + 2, str(radix - 1)), (sum_entry + radix + 9, "")}
        intended.add((expert.halt_state - 1, ""))  # Explicit terminal S after returning up.
        expected_rules = (radix**3 + 6 * radix**2 + 43 * radix + 40) // 2
    else:
        # Explicit preserve-digit halts look like fallback entries but are reachable.
        intended |= {(expert.halt_state - 1, str(d)) for d in range(1, radix)}
        expected_rules = 2 * radix**2 + 21 * radix + 13
    assert len(covered) == expected_rules
    assert covered == intended


@pytest.mark.parametrize("radix", range(2, 11))
def test_default_grid_multiplication_teaches_both_mixed_later_erasure_contexts(radix):
    pairs = set(multiplication_pairs(radix, environment="grid", variant="default"))
    new_variant = set(multiplication_pairs(radix, environment="grid", variant="new"))
    expert = create_expert(dict(environment="grid", task="multiplication", radix=radix))
    covered = set()
    for a in range(radix):
        for b in range(a + 1, radix):
            word = f"{a}{b}{b}{a}"
            assert (word, word) in pairs
            assert (word, word) not in new_variant
            execution = expert.simulate([list(word + "x" + word)])
            _check_grid_multiplication_result(execution, word, word, radix)
            previous = None
            run_length = 0
            contexts = set()
            for state, token in zip(execution.states, execution.trace.inputs.tolist()):
                read = expert.vocab_rev[token]
                factor = state - (2 * radix + 7)
                if 0 <= factor < radix and read.isdigit():
                    run_length = run_length + 1 if state == previous else 1
                    previous = state
                    if run_length >= 2 and int(read) != factor:
                        contexts.add((factor, int(read)))
                else:
                    previous, run_length = None, 0
            assert contexts == {(a, b), (b, a)}
            covered |= contexts
    assert covered == {(f, d) for f in range(radix) for d in range(radix) if f != d}


@pytest.mark.parametrize(
    "builder,radix,digest",
    [
        (addition_pairs, 2, "995ac7706781db07cbc647e905cc428224282007d7d855c1a8f14eebf2207d7e"),
        (addition_pairs, 10, "34e4b2aecc9e13aad10edd1dc7ad5196e0dee464279ef362c3ae04c12fde18e3"),
        (
            multiplication_pairs,
            2,
            "0f30d6a2218a8be28c330f60a3fef8e085d7b869844328f1f16d564b409cc824",
        ),
        (
            multiplication_pairs,
            10,
            "3b3c4ede271e13dfce606232ebb4192f18cb4376e46b327a37e422c36433c8e3",
        ),
    ],
)
def test_default_teaching_pairs_are_frozen_for_experiment_reproducibility(builder, radix, digest):
    pairs = builder(radix, environment="grid", variant="default")
    encoded = json.dumps(pairs, separators=(",", ":")).encode()
    assert hashlib.sha256(encoded).hexdigest() == digest


@pytest.mark.parametrize("radix,count,coverage", [(2, 21, 209), (10, 481, 4357)])
def test_tm_multiplication_core_3digit_and_covers_all_reachable_rules(radix, count, coverage):
    pairs = multiplication_pairs(radix, environment="tm", variant="default")
    assert len(pairs) == len(set(pairs)) == count
    assert set(pairs) == {(b, a) for a, b in pairs}
    assert max(len(w) for pair in pairs for w in pair) == 3
    expert = create_expert(
        dict(environment="tm", task="multiplication", radix=radix, max_steps=300_000)
    )
    covered = set()
    for a, b in pairs:
        ex = expert.simulate([list(a) + ["x"] + list(b)])
        assert _result_value(ex, radix) == int(a, radix) * int(b, radix)
        covered |= _events(expert, ex)
    assert len(covered) == coverage


@pytest.mark.parametrize("variant", ["default", "new"])
def test_tp_real_tm_teaching_set_covers_all_reachable_rules(variant):
    tasks_used = {t for t, _ in TP_TEACHING_SPEC}
    assert tasks_used == {"scaling_A_m5", "scaling_B_m5", "shift_left", "shift_right", "parity"}
    config = dict(
        environment="teleporting",
        variant=variant,
        source="static",
        max_length=3,
        size=80,
        teaching_set=True,
        train_tasks=["parity"],
        seed=7,
    )
    inputs = teleporting_teaching_inputs(config)
    assert len(inputs) == len(set(inputs)) == 75
    words = {""} | {format(i, f"0{n}b") for n in range(1, 4) for i in range(2**n)}
    assert set(inputs) == {(task, word) for task in tasks_used for word in words}
    expert = create_expert(config)
    assert expert.variant == "default"
    covered = set()
    for task_name, word in inputs:
        rows = expert.encode_program(task_transitions(task_name), word)
        ex = expert.simulate(rows, (1, 1 - len(rows)))
        covered |= _events(expert, ex)
    intended = {k for k, v in expert.transitions.items() if v != (expert.halt_state, k[1], "S")}
    assert len(covered) == 67 and covered == intended

    traces = build_static_dataset(config)
    assert len(traces) == 80
    assert [(t.metadata["task"], t.metadata["input"]) for t in traces[:75]] == list(inputs)
    # Remaining 5 slots are sampled from configured data.train_tasks (here ["parity"]).
    assert {t.metadata["task"] for t in traces[75:]} == {"parity"}
    sampler = BatchSampler(expert, random.Random(0), 2, traces)
    sampled_tasks = {sampler.sample(3).metadata[0]["task"] for _ in range(100)}
    assert sampled_tasks == tasks_used


@pytest.mark.parametrize(
    "recipe,variant,count",
    [
        ("add10", "default", 120),
        ("add10", "new", 181),
        ("mult10", "default", 1645),
        ("mult10", "new", 700),
    ],
)
def test_recipes_cache_sample_train_and_resume(tmp_path, recipe, variant, count):
    config = load_config(
        f"configs/DFST/{recipe}.yaml",
        overrides=[
            f"data.variant={variant}",
            f"data.size={count}",
            "data.teaching_set=true",
            "training.steps=2",
            "training.warmup_steps=0",
            "training.batch_size=1",
            "training.checkpoint_every=1",
            "training.log_every=1",
            "evaluation.final=false",
            "model.hidden_size=4",
            "run.device=cpu",
        ],
    )
    config["data"]["cache_dir"] = str(tmp_path / "cache")
    traces = build_static_dataset(config["data"])
    assert len(traces) == count
    cached = build_static_dataset(config["data"])
    assert all(
        torch.equal(a.inputs, b.inputs) and a.metadata == b.metadata for a, b in zip(traces, cached)
    )
    sampler = BatchSampler(create_expert(config["data"]), random.Random(2), 2, traces)
    for length in range(1, config["data"]["max_length"] + 1):
        assert all(max(m["lengths"]) <= length for m in sampler.sample(length).metadata)
    path = train(config, run_dir=tmp_path / "run", stop_after=1)
    train(config, run_dir=path, resume=True)
    _, _, saved, checkpoint = load_run(path)
    assert checkpoint["step"] == 2 and saved["data"]["teaching_set"] is True


def test_teaching_off_keeps_old_dataset_and_experiment_identities():
    config = load_config(overrides=["data.teaching_set=false"])
    old = deepcopy(config)
    del old["data"]["teaching_set"]
    assert experiment_config(config) == experiment_config(old)
    assert dataset_spec(config["data"]) == dataset_spec(old["data"])
    config["data"].update(teaching_set=True, size=25)
    baseline = deepcopy(config)
    baseline["data"]["teaching_set"] = False
    assert experiment_config(config) != experiment_config(baseline)
    assert dataset_spec(config["data"]) != dataset_spec(baseline["data"])


def test_changed_teaching_examples_invalidate_cache_without_changing_expert(monkeypatch, tmp_path):
    import differentiable_exact_learning.data.datasets as datasets

    config = dict(
        environment="grid",
        task="addition",
        radix=2,
        teaching_set=True,
        max_length=3,
        size=12,
        source="static",
    )
    before = dataset_spec(config)
    without_teaching = {**config, "teaching_set": False}
    old_plain_fingerprint = dataset_fingerprint(without_teaching)
    build_static_dataset(config, tmp_path)
    original = datasets.arithmetic_teaching_pairs

    def changed_pairs(cfg):
        return (*original(cfg), ("00", "1"))

    monkeypatch.setattr(datasets, "arithmetic_teaching_pairs", changed_pairs)
    after = dataset_spec(config)
    assert before["expert_digest"] == after["expert_digest"]
    assert before["teaching_digest"] != after["teaching_digest"]
    assert dataset_fingerprint(without_teaching) == old_plain_fingerprint
    traces = build_static_dataset(config, tmp_path)
    assert tuple(traces[8].metadata["operands"]) == ("00", "1")
    assert len(list(tmp_path.glob("*.pt"))) == 2


@pytest.mark.parametrize(
    "overrides",
    [
        ["data.teaching_set=none"],
        ["data.teaching_set=add"],
        ["data.teaching_set=mult"],
        ["data.teaching_set=tp"],
        ["data.size=null"],
        ["data.teaching_set=true", "data.source=online"],
        ["data.teaching_set=true", "data.variant=default", "data.max_length=1"],
        [
            "data.teaching_set=true",
            "data.variant=default",
            "data.size=7",
        ],  # grid default add2 requires 8
        [
            "data.teaching_set=true",
            "data.variant=default",
            "data.task=multiplication",
            "data.max_length=4",
            "data.size=50",
        ],
        [
            "data.teaching_set=true",
            "data.variant=default",
            "data.task=multiplication",
            "data.max_length=5",
            "data.size=32",
        ],  # grid default mult2 requires 33
        ["data.teaching_set=true", "data.variant=new", "data.max_length=2"],
        ["data.teaching_set=true", "data.variant=new", "data.size=20"],  # grid new add2 requires 21
        [
            "data.teaching_set=true",
            "data.variant=new",
            "data.task=multiplication",
            "data.max_length=4",
            "data.size=100",
        ],
        [
            "data.teaching_set=true",
            "data.variant=new",
            "data.task=multiplication",
            "data.max_length=5",
            "data.size=27",
        ],  # grid new mult2 requires 28
        ["data.teaching_set=true", "data.environment=tm", "data.variant=default", "data.size=18"],
        ["data.teaching_set=true", "data.environment=teleporting", "data.size=74"],
    ],
)
def test_invalid_teaching_configuration_fails_early(overrides):
    with pytest.raises(ValueError):
        load_config(overrides=overrides)


def test_cli_exports_reviewable_core_for_grid_and_tp(capsys):
    main(
        [
            "teaching-set",
            "--config",
            "configs/DFST/add10.yaml",
            "data.variant=default",
            "data.teaching_set=true",
        ]
    )
    manifest = json.loads(capsys.readouterr().out)
    assert manifest["count"] == 120 and manifest["covered_transitions"] == 201
    assert manifest["examples"][0] == {"operands": ["0", "0"]}
    assert teaching_manifest({"teaching_set": False})["examples"] == []
    assert teaching_set_size({"teaching_set": False}) == 0

    main(
        [
            "teaching-set",
            "--config",
            "configs/DFST/add10.yaml",
            "data.variant=new",
            "data.teaching_set=true",
        ]
    )
    new_add_manifest = json.loads(capsys.readouterr().out)
    assert new_add_manifest["count"] == 181 and new_add_manifest["covered_transitions"] == 124

    main(
        [
            "teaching-set",
            "--config",
            "configs/DFST/mult10.yaml",
            "data.variant=default",
            "data.teaching_set=true",
        ]
    )
    mult_manifest = json.loads(capsys.readouterr().out)
    assert mult_manifest["count"] == 1645 and mult_manifest["covered_transitions"] == 1035

    main(
        [
            "teaching-set",
            "--config",
            "configs/DFST/mult10.yaml",
            "data.variant=new",
            "data.teaching_set=true",
            "data.size=700",
        ]
    )
    new_mult_manifest = json.loads(capsys.readouterr().out)
    assert new_mult_manifest["count"] == 700 and new_mult_manifest["covered_transitions"] == 423

    main(
        [
            "teaching-set",
            "--config",
            "configs/DFST/universal.yaml",
            "data.variant=default",
            "data.source=static",
            "data.size=75",
            "data.max_length=3",
            "data.teaching_set=true",
        ]
    )
    tp_manifest = json.loads(capsys.readouterr().out)
    assert tp_manifest["count"] == 75 and tp_manifest["covered_transitions"] == 67
