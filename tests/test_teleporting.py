"""The 15-state no-rewind controller preserves each simulated TM step."""

import itertools
import random

import pytest

from differentiable_exact_learning.checkpoint import atomic_save
from differentiable_exact_learning.config import load_config
from differentiable_exact_learning.data import (
    DEFAULT_TP_TASKS,
    create_expert,
    encode_tm,
    task_transitions,
)
from differentiable_exact_learning.data.datasets import dataset_spec
from differentiable_exact_learning.data.experts import Memory
from differentiable_exact_learning.data.transitions.teleporting import universal_tp
from differentiable_exact_learning.training import construct_model, load_run
from differentiable_exact_learning.viewer import Session, load_controller


def _check_simulated_steps(program, word, count=None, rows=None, alphabet=("", "0", "1")):
    """Compare every completed simulated step to an independent sparse tape.

    Also uncover the bookmarks and compare the entire program table, so an
    apparently correct tape cannot hide instruction corruption during execution.
    The encoded S instruction defines termination (including a halt self-loop
    after an instruction that moves the head).
    """
    rows = encode_tm(program, word) if rows is None else rows
    tape_row = 1 - len(rows)
    memory = Memory(rows, (1, tape_row), True)
    pristine = {p: s for p, s in memory.cells.items() if p[1] != tape_row and s != "X"}
    rules = universal_tp(set(alphabet))
    halt = max(q for q, _ in rules)
    controller_state = tm_state = head = steps = microsteps = 0
    tape = dict(enumerate(word))
    while count is None or steps < count:
        tm_state, written, motion = program[tm_state, tape.get(head, "")]
        if written:
            tape[head] = written
        else:
            tape.pop(head, None)
        head += {"L": -1, "R": 1, "S": 0}[motion]
        for _ in range(10000):
            before = memory.head
            controller_state, output, move = rules[controller_state, memory.read()]
            memory.write(output)
            memory.move(move)
            microsteps += 1
            if before[1] == tape_row and move in "LRS":
                break
            assert controller_state != halt, "Halted in the middle of decoding an instruction"
        else:
            pytest.fail("Failed to complete one simulated TM step within 10,000 actions")
        steps += 1
        assert memory.head == (head + 1, tape_row)
        assert {x - 1: s for (x, y), s in memory.cells.items() if y == tape_row} == tape
        assert (controller_state == halt) == (motion == "S")
        # Between simulated steps, X sits at column 1 (covering blank) of a next-state row.
        (x, y), under = memory.runes["X"]
        assert x == 1 and -y // len(alphabet) == tm_state and under == ""
        uncovered = {p: s for p, s in memory.cells.items() if p[1] != tape_row}
        for position, hidden in memory.runes.values():
            if hidden:
                uncovered[position] = hidden
            else:
                uncovered.pop(position, None)
        assert uncovered == pristine
        if motion == "S":
            return steps, microsteps
        assert steps < 10000, "Underlying task exceeded the test's finite budget"
    return steps, microsteps


def test_binary_controller_has_15_states_135_rules_and_67_reachable_rules():
    expert = create_expert({"environment": "teleporting"})
    assert expert.n_states == 15 and expert.halt_state == 14
    assert expert.alphabet == ["", "0", "1", "L", "R", "S", "X", "Y", "Z"]
    assert expert.motion_vocab == dict(zip("LRDUXYZS", range(8)))
    assert len(expert.transitions) == 135
    for symbol in expert.alphabet:
        assert expert.transitions[14, symbol] == (14, symbol, "S")
    config = load_config(overrides=["data.environment=teleporting", "data.teaching_set=false"])
    assert construct_model(config, expert, "cpu").A.shape[-2:] == (15, 15)

    covered = set()
    for task in [
        "parity",
        "shift_left",
        "shift_right",
        "bit_delete",
        "scaling_A_m5",
        "scaling_B_m5",
    ]:
        program = task_transitions(task)
        for n in range(4):
            for word in itertools.product("01", repeat=n):
                rows = encode_tm(program, word)
                ex = expert.simulate(rows, (1, 1 - len(rows)))
                covered.update(
                    (q, expert.vocab_rev[x]) for q, x in zip(ex.states, ex.trace.inputs.tolist())
                )
    intended = {k for k, v in expert.transitions.items() if v != (expert.halt_state, k[1], "S")}
    assert len(intended) == 67
    assert covered == intended


@pytest.mark.parametrize("task", [*DEFAULT_TP_TASKS, "scaling_A_m10", "scaling_B_m10"])
def test_all_binary_inputs_through_length_six_at_every_tm_step(task):
    for length in range(7):
        for word in itertools.product("01", repeat=length):
            _check_simulated_steps(task_transitions(task), word)


@pytest.mark.parametrize("states", [2, 3, 4, 8, 9, 10, 26, 27, 28, 80])
def test_generated_terminating_machines_across_ternary_width_boundaries(states):
    rng = random.Random(239 + states)
    for _ in range(10):
        program = {
            (q, s): (rng.randrange(q + 1, states), rng.choice(["", "0", "1"]), rng.choice("LR"))
            for q in range(states - 1)
            for s in ("", "0", "1")
        }
        for s in ("", "0", "1"):
            program[states - 1, s] = (states - 1, s, "S")
        for length in range(4):
            _check_simulated_steps(program, [rng.choice("01") for _ in range(length)])


@pytest.mark.parametrize("states", [2, 4, 10, 28, 80])
def test_cyclic_machines_preserve_tape_head_and_program_at_every_step(states):
    rng = random.Random(987 + states)
    for _ in range(12):
        program = {
            (q, s): (rng.randrange(states - 1), rng.choice(["", "0", "1"]), rng.choice("LR"))
            for q in range(states - 1)
            for s in ("", "0", "1")
        }
        for s in ("", "0", "1"):
            program[states - 1, s] = (states - 1, s, "S")
        _check_simulated_steps(program, [rng.choice("01") for _ in range(8)], count=60)


@pytest.mark.parametrize("m", [2, 4, 5, 7, 9])
def test_general_tape_alphabet_keeps_state_sharing_unambiguous(m):
    alphabet = ["", *"abcdefgh"[: m - 1]]
    delim = alphabet[1]
    program = {
        (q, s): (min(q + 1, 3), alphabet[(i + q) % m], "RLS"[min(q, 2)])
        for q in range(4)
        for i, s in enumerate(alphabet)
    }

    def trits(q):
        return ["LRS"[q // 3], "LRS"[q % 3]]

    word = alphabet[1:] * 2
    rows = [
        [s, "", *trits(q), delim, *trits(n), delim, w, move]
        for (q, s), (n, w, move) in program.items()
    ]
    rows[0][1] = "X"
    rows.append(["", *word])
    _check_simulated_steps(program, word, rows=rows, alphabet=alphabet)


@pytest.mark.parametrize("alphabet", [set(), {"0", "1"}, {""}, {"", "X"}, {"", "L"}, {"", 1}])
def test_invalid_alphabets_raise_value_error(alphabet):
    with pytest.raises(ValueError):
        universal_tp(alphabet)


@pytest.mark.parametrize(
    "task,word",
    [
        ("parity", "101101"),
        ("reverse", "101101"),
        ("copy", "0110"),
        ("scaling_B_m10", "101101"),
    ],
)
def test_viewer_halt_matches_expert(task, word):
    _, expected_steps = _check_simulated_steps(task_transitions(task), word)
    source = load_controller("tp:" + task)
    session = Session(source, word)
    while session.advance() is not None:
        pass
    assert session.status == "halted" and session.step == expected_steps
    assert session.expert_state == 14


@pytest.mark.parametrize("name", ["DFST", "RNN"])
def test_old_checkpoint_digest_cannot_load_even_when_model_dimensions_fit(tmp_path, name):
    # Historical policy digest, with the default binary training-task mixture.
    old_digest = "39063187b87f4f1313a4137fe974b55dc69a09e4533467cc29bf4df89d7648f3"
    config = load_config(
        overrides=[
            "data.environment=teleporting",
            "data.teaching_set=false",
            f"model.name={name}",
            "model.hidden_size=4",
        ]
    )
    assert dataset_spec(config["data"])["expert_digest"] != old_digest
    expert = create_expert(config["data"])
    model = construct_model(config, expert, "cpu")
    path = tmp_path / "old-policy.pt"
    atomic_save(
        {
            "format_version": 1,
            "config": config,
            "model": model.state_dict(),
            "expert_digest": old_digest,
        },
        path,
    )
    with pytest.raises(ValueError, match="expert fingerprint"):
        load_run(path)


@pytest.mark.parametrize("m", [1, 2, 5, 10])
def test_scaling_a_explicit_terminal_stop(m):
    from differentiable_exact_learning.data.transitions.teleporting import scaling_A_tm

    table = scaling_A_tm(m)
    halt = max(q for q, _ in table)
    assert all(move == "S" for target, _, move in table.values() if target == halt)
    for word in ("", "0", "1", "01101"):
        tape = dict(enumerate(word))
        state, head = 0, 0
        actions = []
        while state != halt:
            state, symbol, motion = table[state, tape.get(head, "")]
            tape[head] = symbol
            head += {"L": -1, "R": 1, "S": 0}[motion]
            actions.append(motion)
            assert len(actions) <= m + 1
        assert len(actions) == m + 1
        assert actions[-1] == "S" and "S" not in actions[:-1]
