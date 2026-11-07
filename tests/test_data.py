"""Expert correctness, finite generation, masking, and cache/resume regressions."""

import itertools
import random

import pytest
import torch

from differentiable_exact_learning.config import experiment_config, load_config
from differentiable_exact_learning.data import (
    DEFAULT_TP_TASKS,
    BatchSampler,
    OnlineStream,
    StepLimitExceeded,
    build_static_dataset,
    create_expert,
    dataset_fingerprint,
    dataset_spec,
    encode_tm,
    pad_traces,
    task_transitions,
)
from differentiable_exact_learning.data.transitions import (
    add_grid_new,
    mult_grid_new,
)


def _words(radix, maximum):
    return [
        list(word)
        for n in range(1, maximum + 1)
        for word in itertools.product([str(i) for i in range(radix)], repeat=n)
    ]


def _value(word, radix):
    answer = 0
    for digit in word:
        answer = answer * radix + int(digit)
    return answer


def _result_digits(execution, radix):
    cells = execution.cells
    assert cells, "An arithmetic result must contain at least one digit"
    assert len({y for x, y in cells}) == 1, cells
    xs = sorted(x for x, y in cells)
    assert xs == list(range(xs[0], xs[-1] + 1)), cells
    digits = [symbol for _, symbol in sorted(cells.items())]
    assert all(symbol in set(map(str, range(radix))) for symbol in digits), cells
    return digits


def _result_value(execution, radix):
    return _value(_result_digits(execution, radix), radix)


def _encode(value, radix):
    if value == 0:
        return "0"
    result = []
    while value:
        value, digit = divmod(value, radix)
        result.append(str(digit))
    return "".join(reversed(result))


def _check_grid_addition_result(execution, a, b, radix):
    digits = _encode(_value(a, radix) + _value(b, radix), radix)
    digits = digits.zfill(max(len(a), len(b) + 1))
    assert "".join(_result_digits(execution, radix)) == digits
    left = len(a) - len(digits)
    assert execution.cells == {(left + i, 0): digit for i, digit in enumerate(digits)}
    assert execution.head == (len(a) - len(b) - 1, 0)


def _check_new_grid_multiplication_result(execution, a, b, radix):
    digits = _encode(_value(a, radix) * _value(b, radix), radix).zfill(len(a) + len(b))
    assert "".join(_result_digits(execution, radix)) == digits
    assert execution.cells == {(-len(b) + i, 1): digit for i, digit in enumerate(digits)}
    assert execution.head == (-len(b), 1)


def _check_grid_multiplication_result(execution, a, b, radix):
    digits = _encode(_value(a, radix) * _value(b, radix), radix).zfill(len(a) + len(b))
    assert "".join(_result_digits(execution, radix)) == digits
    row = len(b) - 1
    assert execution.cells == {(-len(b) + i, row): digit for i, digit in enumerate(digits)}
    assert execution.head == (-len(b) - 1, row)


@pytest.mark.parametrize("environment", ["tm", "grid"])
@pytest.mark.parametrize("task", ["addition", "multiplication"])
@pytest.mark.parametrize("radix,maximum", [(2, 4), (3, 2), (10, 1)])
def test_exhaustive_arithmetic(environment, task, radix, maximum):
    expert = create_expert(dict(environment=environment, task=task, radix=radix, max_steps=100_000))
    operator = "+" if task == "addition" else "x"
    for a, b in itertools.product(_words(radix, maximum), repeat=2):
        execution = expert.simulate([a + [operator] + b])
        expected = (
            _value(a, radix) + _value(b, radix)
            if task == "addition"
            else _value(a, radix) * _value(b, radix)
        )
        assert _result_value(execution, radix) == expected, (a, b, execution.cells)
        if environment == "grid":
            if task == "addition":
                _check_grid_addition_result(execution, a, b, radix)
            else:
                _check_grid_multiplication_result(execution, a, b, radix)


@pytest.mark.parametrize("a,b", [("11", "1"), ("100", "0"), ("111", "1")])
def test_grid_addition_preserves_significant_zero(a, b):
    expert = create_expert(dict(environment="grid", task="addition", radix=2))
    execution = expert.simulate([list(a + "+" + b)])
    assert _result_value(execution, 2) == int(a, 2) + int(b, 2)
    if (a, b) == ("11", "1"):
        assert execution.cells == {(-1, 0): "1", (0, 0): "0", (1, 0): "0"}
    assert expert.n_states == 11


@pytest.mark.parametrize(
    "a,b,expected,head",
    [
        ("100", "0", "100", (1, 0)),
        ("99", "1", "100", (0, 0)),
        ("1", "1", "02", (-1, 0)),
        ("0", "0", "00", (-1, 0)),
        ("0000", "0000", "00000", (-1, 0)),
        ("001", "0", "001", (1, 0)),
        ("0", "001", "0001", (-3, 0)),
    ],
)
def test_grid_addition_retains_padding_and_halts_above_final_marker(a, b, expected, head):
    expert = create_expert(dict(environment="grid", task="addition", radix=10))
    execution = expert.simulate([list(a + "+" + b)])
    assert "".join(_result_digits(execution, 10)) == expected
    assert execution.head == head
    assert expert.motion_vocab_rev[int(execution.trace.motions[-1])] == "S"


@pytest.mark.parametrize("radix", [2, 10])
@pytest.mark.parametrize("n,m", list(itertools.product([1, 4, 16], repeat=2)))
def test_grid_addition_zero_result_preserves_padding_and_units_digit(radix, n, m):
    expert = create_expert(dict(environment="grid", task="addition", radix=radix))
    execution = expert.simulate([list("0" * n + "+" + "0" * m)])
    units = (n - 1, 0)
    assert execution.cells == {(x, 0): "0" for x in range(min(0, n - m - 1), n)}
    assert execution.head == (n - m - 1, 0)

    # Replay the actual writes: cleanup must retain the existing units zero,
    # rather than erase it and reconstruct an equivalent-looking final result.
    head = (0, 0)
    moves = {"L": (-1, 0), "R": (1, 0), "U": (0, -1), "D": (0, 1), "S": (0, 0)}
    for written, motion in zip(execution.trace.symbols, execution.trace.motions):
        if head == units:
            assert expert.vocab_rev[written.item()] == "0"
        dx, dy = moves[expert.motion_vocab_rev[motion.item()]]
        head = (head[0] + dx, head[1] + dy)


@pytest.mark.parametrize("radix", range(2, 11))
@pytest.mark.parametrize("task", ["addition", "multiplication"])
def test_grid_state_counts_and_complete_halt(task, radix):
    expert = create_expert(dict(environment="grid", task=task, radix=radix))
    expected = radix + 9 if task == "addition" else radix * (radix + 1) // 2 + 3 * radix + 23
    assert expert.n_states == expected
    assert expert.halt_state == expected - 1
    assert all(
        move == "S"
        for target, _, move in expert.transitions.values()
        if target == expert.halt_state
    )
    assert {state for state, _ in expert.transitions} == set(range(expected))
    assert len(expert.transitions) == expected * (radix + 2)
    for symbol in expert.alphabet:
        assert expert.transitions[expert.halt_state, symbol] == (expert.halt_state, symbol, "S")


@pytest.mark.parametrize("radix", range(2, 11))
@pytest.mark.parametrize("task", ["addition", "multiplication"])
def test_grid_results_with_leading_zeros_and_unequal_lengths(task, radix):
    expert = create_expert(dict(environment="grid", task=task, radix=radix))
    maximum = str(radix - 1)
    cases = [
        ("0", "0"),
        ("000000", "00"),
        ("0", "0001"),
        ("0001", "0"),
        ("00100", "000"),
        ("000", "00100"),
        (maximum * 8, "1"),
        ("1", maximum * 8),
        ("000" + maximum * 5, "00" + maximum * 3),
        ("10001", "00101"),
    ]
    rng = random.Random(310 + radix)
    cases.extend(
        (
            "".join(str(rng.randrange(radix)) for _ in range(rng.randrange(1, 13))),
            "".join(str(rng.randrange(radix)) for _ in range(rng.randrange(1, 13))),
        )
        for _ in range(30)
    )
    operator = "+" if task == "addition" else "x"
    for a, b in cases:
        result = expert.simulate([list(a + operator + b)])
        if task == "addition":
            _check_grid_addition_result(result, a, b, radix)
        else:
            _check_grid_multiplication_result(result, a, b, radix)


@pytest.mark.parametrize("radix", [2, 10])
def test_grid_multiplication_carry_at_radix_and_past_top_row(radix):
    expert = create_expert(dict(environment="grid", task="multiplication", radix=radix))
    operand = str(radix - 1) * 5
    execution = expert.simulate([list(operand + "x" + operand)])
    # Confirm this end-to-end case exercises both subtle carry branches, rather
    # than merely testing a transition entry that the algorithm never reaches.
    sum_entry = 2 * radix + 6 + radix * (radix + 1) // 2 + 2 + 3
    read_with_carry = sum_entry + radix + 6
    events = {
        (state, expert.vocab_rev[token.item()])
        for state, token in zip(execution.states, execution.trace.inputs)
    }
    assert (read_with_carry, str(radix - 1)) in events  # top digit + carry == radix
    assert (read_with_carry, "") in events  # carry extends beyond the upper row
    assert "".join(_result_digits(execution, radix)) == _encode(int(operand, radix) ** 2, radix)


@pytest.mark.parametrize("length,expected_steps", [(5, 431), (20, 6251)])
def test_decimal_multiplication_trace_length(length, expected_steps):
    expert = create_expert(dict(environment="grid", task="multiplication", radix=10))
    operand = "9" * length
    execution = expert.simulate([list(operand + "x" + operand)])
    assert len(execution.trace) == expected_steps
    assert expert.motion_vocab_rev[int(execution.trace.motions[-1])] == "S"
    assert "".join(_result_digits(execution, 10)) == str(int(operand) ** 2)


@pytest.mark.parametrize("environment", ["grid", "tm"])
@pytest.mark.parametrize("task", ["addition", "multiplication"])
def test_decimal_carries(environment, task):
    expert = create_expert(dict(environment=environment, task=task, radix=10))
    for a, b in [("999", "1"), ("100", "0"), ("109", "991"), ("000", "000")]:
        result = expert.simulate([list(a) + ["+" if task == "addition" else "x"] + list(b)])
        assert _result_value(result, 10) == (
            int(a) + int(b) if task == "addition" else int(a) * int(b)
        )


def _direct_tm(transitions, word, limit=100_000):
    tape = {i: digit for i, digit in enumerate(word)}
    head, state = 0, 0
    halt = max(s for s, _ in transitions)
    for _ in range(limit):
        state, written, move = transitions[state, tape.get(head, "")]
        if written:
            tape[head] = written
        else:
            tape.pop(head, None)
        head += {"L": -1, "R": 1, "S": 0}[move]
        if state == halt:
            return tape
    raise AssertionError("Underlying task TM exceeded limit")


@pytest.mark.parametrize("task", DEFAULT_TP_TASKS)
def test_universal_simulates_underlying_tm(task):
    expert = create_expert(dict(environment="teleporting", max_steps=100_000))
    transitions = task_transitions(task)
    for n in range(4):
        for bits in itertools.product("01", repeat=n):
            rows = encode_tm(transitions, bits)
            execution = expert.simulate(rows, (1, 1 - len(rows)))
            tape = {
                x - 1: symbol for (x, y), symbol in execution.cells.items() if y == 1 - len(rows)
            }
            assert tape == _direct_tm(transitions, bits), (task, bits, tape)


@pytest.mark.parametrize(
    "task",
    ["parity", "flip", "cum_sum", "bit_delete", "shift_left", "shift_right", "reverse", "copy"],
)
def test_binary_task_semantics(task):
    transitions = task_transitions(task)
    for n in range(5):
        for bits in itertools.product("01", repeat=n):
            tape = _direct_tm(transitions, bits)
            if task == "parity":
                expected = {n: str(sum(map(int, bits)) % 2)}
            elif task == "flip":
                expected = {i: str(1 - int(b)) for i, b in enumerate(bits)}
            elif task == "cum_sum":
                expected = {i: str(sum(map(int, bits[: i + 1])) % 2) for i in range(n)}
            elif task == "bit_delete":
                expected = {i: b for i, b in enumerate(bits) if b == str(n % 2)}
            elif task in {"shift_left", "shift_right"}:
                shift = -1 if task == "shift_left" else 1
                expected = {i + shift: b for i, b in enumerate(bits)}
            elif task == "reverse":
                # Reverse relocates the string to the left while reversing it.
                expected_word = list(reversed(bits))
                assert [b for _, b in sorted(tape.items())] == expected_word
                continue
            else:
                expected = {i: b for i, b in enumerate(bits)}
                expected.update({i + n + 1: b for i, b in enumerate(bits)})
            assert tape == expected, (task, bits, tape, expected)


@pytest.mark.parametrize(
    "environment,task,radix",
    [
        ("grid", "addition", 2),
        ("grid", "multiplication", 2),
        ("grid", "addition", 10),
        ("grid", "multiplication", 10),
        ("tm", "addition", 2),
        ("tm", "multiplication", 2),
        ("teleporting", "parity", 2),
    ],
)
def test_exact_expert_tensor_replay(environment, task, radix):
    expert = create_expert(dict(environment=environment, task=task, radix=radix))
    trace = expert.sample(
        random.Random(9),
        3,
        task=task if environment == "teleporting" else None,
        operands=("999", "999") if radix == 10 else None,
    )
    A, B, C = expert.get_tensors()
    hidden = torch.zeros(expert.n_states)
    hidden[0] = 1
    for x, symbol, motion in zip(trace.inputs, trace.symbols, trace.motions):
        assert torch.equal(
            B[x] @ hidden, torch.nn.functional.one_hot(symbol, expert.symbol_size).float()
        )
        assert torch.equal(
            C[x] @ hidden, torch.nn.functional.one_hot(motion, expert.motion_size).float()
        )
        hidden = A[x] @ hidden
    assert hidden.argmax().item() == expert.halt_state


def test_no_duplicate_halt_and_step_guard():
    expert = create_expert(dict(environment="grid", task="addition"))
    execution = expert.simulate([list("1+1")])
    assert len(execution.trace) == len(execution.states) - 1
    assert execution.states[-2] != expert.halt_state
    with pytest.raises(StepLimitExceeded):
        expert.simulate([list("1+1")], max_steps=1)


@pytest.mark.parametrize("environment", ["grid", "tm", "teleporting"])
def test_unlimited_expert_execution_and_exact_finite_boundary(environment):
    config = load_config(overrides=[f"data.environment={environment}", "data.teaching_set=false"])[
        "data"
    ]
    assert config["max_steps"] is None

    def sample(expert):
        return expert.sample(
            random.Random(9), 3, task="parity" if environment == "teleporting" else None
        )

    expected = sample(create_expert(config))
    # Direct construction without a configuration entry also runs until halt.
    omitted = {key: value for key, value in config.items() if key != "max_steps"}
    for limit_config in (omitted, dict(config, max_steps=len(expected))):
        actual = sample(create_expert(limit_config))
        assert actual.metadata == expected.metadata
        for key in ("inputs", "symbols", "motions"):
            assert torch.equal(getattr(actual, key), getattr(expected, key))
    with pytest.raises(StepLimitExceeded):
        sample(create_expert(dict(config, max_steps=len(expected) - 1)))


def test_null_cli_override_removes_expert_execution_limit():
    options = ["data.max_steps=1"]
    limited = create_expert(load_config(overrides=options)["data"])
    with pytest.raises(StepLimitExceeded):
        limited.simulate([list("1+1")])
    unlimited = create_expert(load_config(overrides=[*options, "data.max_steps=null"])["data"])
    execution = unlimited.simulate([list("1+1")])
    assert execution.states[-1] == unlimited.halt_state
    assert len(execution.trace) > 1


def test_padding_and_local_rng():
    expert = create_expert(dict(environment="grid"))
    rng1, rng2 = random.Random(8), random.Random(8)
    before = random.getstate()
    first = [expert.sample(rng1, 3) for _ in range(8)]
    second = [expert.sample(rng2, 3) for _ in range(8)]
    assert random.getstate() == before
    for a, b in zip(first, second):
        assert a.metadata == b.metadata
        assert torch.equal(a.inputs, b.inputs)
    batch = pad_traces(first).to("cpu")
    assert batch.mask.dtype == torch.bool
    assert batch.mask.sum().item() == sum(map(len, first))
    for row, trace in enumerate(first):
        assert torch.equal(batch.inputs[row, : len(trace)], trace.inputs)
        assert not batch.mask[row, len(trace) :].any()


def test_static_exact_size_unique_single_digits_and_cache(tmp_path):
    config = dict(environment="grid", task="addition", radix=2, max_length=3, size=25, seed=19)
    first = build_static_dataset(config, tmp_path)
    second = build_static_dataset(config, tmp_path)
    pairs = [tuple(t.metadata["operands"]) for t in first]
    assert len(first) == len(set(pairs)) == 25
    assert set(itertools.product("01", repeat=2)).issubset(pairs)
    assert any(a.startswith("0") and len(a) > 1 for a, b in pairs)
    for a, b in zip(first, second):
        assert a.metadata == b.metadata
        assert torch.equal(a.inputs, b.inputs)
    assert dataset_fingerprint(config) != dataset_fingerprint({**config, "seed": 20})
    cached = next(tmp_path.glob("*.pt"))
    payload = torch.load(cached, weights_only=True)
    payload["traces"].pop()
    torch.save(payload, cached)
    with pytest.raises(ValueError, match="Invalid/incomplete"):
        build_static_dataset(config, tmp_path)


@pytest.mark.parametrize(
    "task,builder", [("addition", "add_grid"), ("multiplication", "mult_grid")]
)
def test_expert_transition_changes_invalidate_dataset_fingerprint(monkeypatch, task, builder):
    import differentiable_exact_learning.data.experts as experts

    config = dict(environment="grid", task=task, radix=2, max_length=2, size=4, seed=19)
    original_fingerprint = dataset_fingerprint(config)
    original_builder = getattr(experts, builder)

    def altered_builder(radix):
        table = original_builder(radix)
        successor, _, motion = table[0, "0"]
        table[0, "0"] = (successor, "1", motion)
        return table

    monkeypatch.setattr(experts, builder, altered_builder)
    assert dataset_fingerprint(config) != original_fingerprint


def test_static_bounds_and_full_capacity():
    cfg = dict(environment="grid", radix=2, max_length=1)
    for size in [0, 3, 5]:
        with pytest.raises(ValueError):
            build_static_dataset({**cfg, "size": size})
    traces = build_static_dataset({**cfg, "max_length": 2, "size": 36})
    assert len({tuple(t.metadata["operands"]) for t in traces}) == 36


def test_static_curriculum_pool():
    cfg = dict(environment="grid", radix=2, max_length=3, size=20)
    traces = build_static_dataset(cfg)
    expert = create_expert(cfg)
    sampler = BatchSampler(expert, random.Random(1), 8, traces)
    batch = sampler.sample(1)
    assert all(max(meta["lengths"]) == 1 for meta in batch.metadata)
    with pytest.raises(ValueError, match="No static"):
        sampler.sample(0)


def test_tp_static_task_coverage_and_uniform_task_batches():
    cfg = dict(
        environment="teleporting",
        train_tasks=["parity", "flip"],
        eval_tasks=["parity"],
        max_length=1,
        size=4,
        seed=2,
    )
    traces = build_static_dataset(cfg)
    assert {t.metadata["task"] for t in traces} == {"parity", "flip"}
    sampler = BatchSampler(create_expert(cfg), random.Random(5), 4, traces)
    seen = set()
    for _ in range(20):
        batch = sampler.sample(1)
        labels = {m["task"] for m in batch.metadata}
        assert len(labels) == 1
        seen.update(labels)
    assert seen == {"parity", "flip"}
    with pytest.raises(ValueError):
        build_static_dataset({**cfg, "size": 1})


def test_online_resume_and_exact_lengths():
    expert = create_expert(dict(environment="grid"))
    stream = OnlineStream(expert, 2, 3, 8)
    next(stream)
    state = stream.state_dict()
    expected = next(stream)
    stream.load_state_dict(state)
    actual = next(stream)
    assert torch.equal(expected.inputs, actual.inputs)
    assert expected.metadata == actual.metadata
    assert expert.sample(random.Random(1), 4, lengths=(4, 4)).metadata["lengths"] == [4, 4]
    tp_expert = create_expert(dict(environment="teleporting"))
    assert (
        tp_expert.sample(random.Random(1), 4, lengths=0, task="scaling_A_m4").metadata["input"]
        == ""
    )


def test_environment_reconstruction():
    for config in [
        dict(environment="grid"),
        dict(environment="grid", variant="new"),
        dict(environment="grid", task="multiplication", variant="new"),
        dict(environment="tm"),
        dict(environment="teleporting"),
    ]:
        expert = create_expert(config)
        trace = expert.sample(random.Random(2), 2)
        memory = expert.make_environment(trace.metadata)
        for x, s, m in zip(trace.inputs, trace.symbols, trace.motions):
            assert expert.vocab[memory.read()] == x.item()
            memory.write(expert.vocab_rev[s.item()])
            memory.move(expert.motion_vocab_rev[m.item()])


@pytest.mark.parametrize("environment", ["grid", "tm", "teleporting"])
@pytest.mark.parametrize("source", ["online", "static"])
@pytest.mark.parametrize("mode", ["fixed", "mixed"])
def test_batch_length_serving_modes(environment, source, mode):
    config = dict(
        environment=environment,
        task="addition",
        radix=2,
        train_tasks=["parity", "flip"],
        eval_tasks=["parity"],
        max_length=3,
        size=40,
        seed=3,
        batch_lengths=mode,
    )
    traces = build_static_dataset(config) if source == "static" else None
    sampler = BatchSampler(create_expert(config), random.Random(6), 16, traces)
    observed_mixed = False
    for _ in range(8):
        batch = sampler.sample(3)
        keys = {tuple(meta["lengths"]) for meta in batch.metadata}
        if mode == "fixed":
            assert len(keys) == 1
        observed_mixed |= len(keys) > 1
        assert len({meta["task"] for meta in batch.metadata}) == 1
    if mode == "mixed":
        assert observed_mixed
    if source == "static":
        cached = dict(sampler._pools)
        sampler.sample(3)
        assert all(sampler._pools[key] is pool for key, pool in cached.items())


def test_static_buckets_are_uniform_not_weighted_by_example_count():
    from collections import Counter

    config = dict(environment="grid", max_length=2, size=36, batch_lengths="fixed")
    traces = build_static_dataset(config)
    sampler = BatchSampler(create_expert(config), random.Random(19), 1, traces)
    counts = Counter(tuple(sampler.sample(2).metadata[0]["lengths"]) for _ in range(1000))
    assert set(counts) == {(1, 1), (1, 2), (2, 1), (2, 2)}
    # Bucket sizes are 4/8/8/16. Uniform-example sampling would yield roughly
    # 111/222/222/444; uniform bucket sampling yields roughly 250 each.
    assert all(190 < count < 310 for count in counts.values()), counts


def test_batch_lengths_default_and_cache_independence():
    config = dict(environment="grid", max_length=2, size=8)
    assert BatchSampler(create_expert(config), random.Random(1), 2).batch_lengths == "fixed"
    assert dataset_fingerprint({**config, "batch_lengths": "fixed"}) == dataset_fingerprint(
        {**config, "batch_lengths": "mixed"}
    )
    with pytest.raises(ValueError, match="batch_lengths"):
        BatchSampler(create_expert({**config, "batch_lengths": "invalid"}), random.Random(1), 2)


@pytest.mark.parametrize("task", ["addition", "multiplication"])
@pytest.mark.parametrize("radix,maximum", [(2, 4), (3, 2), (10, 1)])
def test_exhaustive_new_grid_arithmetic_and_state_counts(task, radix, maximum):
    expert = create_expert(
        dict(environment="grid", variant="new", task=task, radix=radix, max_steps=100_000)
    )
    expected_states = radix + 13 if task == "addition" else 2 * radix + 22
    assert expert.n_states == expected_states
    assert expert.halt_state == expected_states - 1
    assert len(expert.transitions) == expected_states * (radix + 2)
    builder = add_grid_new if task == "addition" else mult_grid_new
    assert expert.transitions == builder(radix)
    operator = "+" if task == "addition" else "x"
    for a, b in itertools.product(_words(radix, maximum), repeat=2):
        execution = expert.simulate([a + [operator] + b])
        expected = (
            _value(a, radix) + _value(b, radix)
            if task == "addition"
            else _value(a, radix) * _value(b, radix)
        )
        assert _result_value(execution, radix) == expected, (a, b, execution.cells)
        if task == "addition":
            _check_grid_addition_result(execution, a, b, radix)
        else:
            _check_new_grid_multiplication_result(execution, a, b, radix)


@pytest.mark.parametrize("radix", range(2, 11))
@pytest.mark.parametrize("task", ["addition", "multiplication"])
def test_new_grid_results_with_leading_zeros_carries_and_reachable_rules(task, radix):
    expert = create_expert(dict(environment="grid", variant="new", task=task, radix=radix))
    maximum = str(radix - 1)
    cases = [
        ("0", "0"),
        ("000000", "00"),
        ("0", "0001"),
        ("0001", "0"),
        ("00100", "000"),
        ("000", "00100"),
        (maximum * 6, "1"),
        ("1", maximum * 6),
        (maximum * 3, maximum * 3),
        ("000" + maximum * 4, "00" + maximum * 3),
        ("10001", "00101"),
    ]
    rng = random.Random(510 + radix)
    cases.extend(
        (
            "".join(str(rng.randrange(radix)) for _ in range(rng.randrange(1, 5))),
            "".join(str(rng.randrange(radix)) for _ in range(rng.randrange(1, 5))),
        )
        for _ in range(120)
    )
    # Include all 2-digit pairs to verify exact reachable-rule count.
    if radix <= 3:
        two_digit = [
            "".join(w) for n in (1, 2) for w in itertools.product(map(str, range(radix)), repeat=n)
        ]
        cases.extend(itertools.product(two_digit, repeat=2))
    operator = "+" if task == "addition" else "x"
    covered = set()
    for a, b in cases:
        result = expert.simulate([list(a + operator + b)])
        if task == "addition":
            _check_grid_addition_result(result, a, b, radix)
        else:
            _check_new_grid_multiplication_result(result, a, b, radix)
        covered.update(
            (q, expert.vocab_rev[x]) for q, x in zip(result.states, result.trace.inputs.tolist())
        )
    if radix <= 3:
        expected_rules = 11 * radix + 14 if task == "addition" else 2 * radix**2 + 21 * radix + 13
        intended = {k for k, v in expert.transitions.items() if v != (expert.halt_state, k[1], "S")}
        if task == "addition":
            intended |= {(expert.halt_state - 1, str(d)) for d in range(radix)}
        else:
            intended |= {(expert.halt_state - 1, str(d)) for d in range(1, radix)}
        assert len(intended) == expected_rules
        assert covered == intended


def test_variant_default_preserves_identities_and_new_changes_them():
    default_cfg = load_config(overrides=["data.variant=default", "data.teaching_set=false"])
    omitted = load_config(overrides=["data.variant=default", "data.teaching_set=false"])
    del omitted["data"]["variant"]
    assert experiment_config(default_cfg) == experiment_config(omitted)
    assert dataset_spec(default_cfg["data"]) == dataset_spec(omitted["data"])

    new_cfg = load_config(overrides=["data.variant=new", "data.teaching_set=false"])
    assert experiment_config(new_cfg) != experiment_config(default_cfg)
    assert dataset_spec(new_cfg["data"]) != dataset_spec(default_cfg["data"])

    # Both tm and teleporting ignore data.variant completely.
    for env in ("tm", "teleporting"):
        env_default = load_config(
            overrides=[f"data.environment={env}", "data.variant=default", "data.teaching_set=false"]
        )
        env_new = load_config(
            overrides=[f"data.environment={env}", "data.variant=new", "data.teaching_set=false"]
        )
        assert (
            create_expert(env_new["data"]).transitions
            == create_expert(env_default["data"]).transitions
        )
        assert create_expert(env_new["data"]).variant == "default"
        assert experiment_config(env_new) == experiment_config(env_default)
        assert dataset_spec(env_new["data"]) == dataset_spec(env_default["data"])
