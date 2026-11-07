"""Exact minimization and fail-fast PTO checks for the annealing workflow."""

import random
from copy import deepcopy

import pytest

from differentiable_exact_learning.analysis.extraction import (
    Transducer,
    compare_closed_loop,
    compare_exact_ptos,
    extract_argmax,
    hopcroft_minimize,
    minimize,
    number_states,
    verify_quotient,
)
from differentiable_exact_learning.analysis.notebook_utils import load_analysis_run
from differentiable_exact_learning.analysis.transition_table import (
    render_transition_table,
    transition_matrix,
)


def test_hopcroft_matches_independent_refinement():
    rng = random.Random(12042)
    for trial in range(200):
        n, width = rng.randrange(1, 20), rng.randrange(1, 5)
        transitions = {}
        for q in range(n):
            for x in range(width):
                if rng.random() < 0.8:
                    transitions[q, x] = (rng.randrange(n), rng.randrange(2) if trial % 2 else 0, 0)
            if q + 1 < n:
                transitions[q, 0] = (q + 1, 0, 0)
        machine = Transducer(transitions, 0, tuple(range(width)), frozenset([n - 1]))
        reduced, mapping = hopcroft_minimize(machine)
        reference, _ = minimize(machine)
        assert verify_quotient(machine, reduced, mapping)
        assert len(reduced.states) == len(reference.states)


def test_hopcroft_removes_unreachable_states_and_preserves_missing_edges():
    machine = Transducer(
        {(0, 0): (1, 2, 3), (1, 0): (1, 2, 3), (99, 1): (99, 0, 0)},
        0,
        (0, 1),
        frozenset([1, 99]),
    )
    reduced, mapping = hopcroft_minimize(machine)
    assert set(mapping) == {0, 1}
    assert verify_quotient(machine, reduced, mapping)
    assert len(reduced.states) == 2
    assert reduced.predict([0, 1]) == ([(2, 3)], False)


@pytest.fixture
def exact_controller():
    model, expert, _, _ = load_analysis_run(softmax=True)
    return hopcroft_minimize(extract_argmax(model))[0], expert


def test_streamed_ptos_match_autonomous_execution(exact_controller):
    machine, expert = exact_controller
    report = compare_exact_ptos(machine, expert, 8, 5, random.Random(42))
    traces = [expert.sample(random.Random(i), max_length=5, lengths=(5, 5)) for i in range(8)]
    assert report["all_exact"] and report["exact_matches"] == 8
    assert compare_closed_loop(machine, expert, traces)["exact_matches"] == 8
    assert all(row["metadata"]["lengths"] == [5, 5] for row in report["results"])


@pytest.mark.parametrize("corruption", ["unknown", "symbol", "early_stop"])
def test_pto_reports_first_mismatch(exact_controller, corruption):
    machine, expert = exact_controller
    machine = deepcopy(machine)
    for x in machine.alphabet:
        key = machine.initial_state, x
        target, symbol, motion = machine.transitions[key]
        if corruption == "unknown":
            del machine.transitions[key]
        elif corruption == "symbol":
            machine.transitions[key] = target, (symbol + 1) % expert.symbol_size, motion
        else:
            machine.transitions[key] = target, symbol, expert.motion_vocab["S"]
    report = compare_exact_ptos(machine, expert, 4, 3, random.Random(42))
    assert report["failures"] == 4 and not report["all_exact"]
    for row in report["results"]:
        assert row["failure"]["step"] == 1
        assert row["checked_steps"] == 1
        assert row["metadata"]["lengths"] == [3, 3]
        assert row["failure"]["actual"] != row["failure"]["expected"]


def test_pto_rejects_wrong_final_halt(exact_controller):
    machine, expert = exact_controller
    machine = deepcopy(machine)
    for key, (target, symbol, motion) in list(machine.transitions.items()):
        if motion == expert.motion_vocab["S"]:
            machine.transitions[key] = target, symbol, expert.motion_vocab["R"]
    report = compare_exact_ptos(machine, expert, 4, 3, random.Random(42))
    assert report["failures"] == 4
    assert all(
        row["failure"]["expected"][1] == expert.motion_vocab["S"] for row in report["results"]
    )


def test_numbering_keeps_start_first_halt_last_and_preserves_behavior():
    machine = Transducer({(5, 0): (3, 1, 2), (3, 0): (-1, 0, 4)}, 5, (0,), frozenset([-1]))
    numbered, mapping = number_states(machine)
    assert numbered.states == [0, 1, 2]
    assert numbered.initial_state == 0 and numbered.final_states == frozenset([2])
    assert verify_quotient(machine, numbered, mapping)


def test_transition_matrix_has_read_columns_and_exact_triples(exact_controller):
    _, expert = exact_controller
    machine = Transducer({(0, 0): (1, 1, 0)}, 0, (0, 1), frozenset([1]))
    rows = transition_matrix(machine, expert)
    assert rows[0]["state"] == 0 and rows[0]["role"] == "initial"
    assert rows[0]["cells"] == [(1, expert.vocab_rev[1] or "λ", expert.motion_vocab_rev[0]), None]
    assert rows[1]["role"] == "halt" and rows[1]["cells"] == [None, None]


def test_transition_table_exports_fonts_rows_and_unknowns(exact_controller, tmp_path):
    import csv
    import xml.etree.ElementTree as ET

    pytest.importorskip("reportlab")
    _, expert = exact_controller
    machine = Transducer(
        {
            (q, expert.vocab[x]): (target, expert.vocab[symbol], expert.motion_vocab[motion])
            for (q, x), (target, symbol, motion) in expert.transitions.items()
        },
        0,
        tuple(range(expert.input_size)),
        frozenset([expert.halt_state]),
    )
    machine, _ = number_states(machine)
    paths = render_transition_table(machine, expert, tmp_path / "table", rows_per_page=4)
    assert len(paths["svg"]) == (len(machine.states) + 3) // 4
    for path in paths["svg"]:
        svg = path.read_text()
        ET.fromstring(svg)
        assert "@font-face" in svg and "data:font/ttf;base64," in svg
        assert "DELState" in svg and "DELSymbol" in svg
    assert paths["pdf"].read_bytes().startswith(b"%PDF")
    tex = paths["tex"].read_text()
    assert r"\texttt{0}" in tex and r"\rowcolor{DELInitial}" in tex
    assert r"\rowcolor{DELFinal}" in tex
    with paths["csv"].open() as stream:
        rows = list(csv.reader(stream))
    assert len(rows) == len(machine.states) + 1
    assert rows[0] == ["state", "role", *[expert.vocab_rev[x] or "λ" for x in machine.alphabet]]
    for row, entry in zip(rows[1:], transition_matrix(machine, expert)):
        assert row[0] == str(entry["state"])
        assert row[2:] == [
            "" if cell is None else " | ".join(map(str, cell)) for cell in entry["cells"]
        ]


def test_explicit_halt_preserves_stopping_outputs_and_unknowns():
    from differentiable_exact_learning.analysis.extraction import with_halt_state

    original = Transducer({(0, 0): (1, 1, 0), (1, 0): (0, 1, 2)}, 0, (0, 1))
    normalized = with_halt_state(original, 2)
    (halt,) = normalized.final_states
    assert normalized.transitions[1, 0] == (halt, 1, 2)
    assert normalized.transitions[0, 0] == original.transitions[0, 0]
    assert (1, 1) not in normalized.transitions
    assert normalized.predict([0, 0]) == original.predict([0, 0])
    assert all(normalized.transitions[halt, x] == (halt, x, 2) for x in (0, 1))
    assert with_halt_state(normalized, 2) == normalized
    reduced, mapping = hopcroft_minimize(normalized)
    reduced, numbering = number_states(reduced)
    assert verify_quotient(normalized, reduced, {q: numbering[v] for q, v in mapping.items()})
    assert reduced.final_states == {max(reduced.states)}
    assert original.final_states == frozenset()


def test_explicit_halt_preserves_expert_ptos(exact_controller):
    from differentiable_exact_learning.analysis.extraction import with_halt_state

    original, expert = exact_controller
    normalized = with_halt_state(original, expert.motion_vocab["S"])
    reduced, mapping = hopcroft_minimize(normalized)
    assert verify_quotient(normalized, reduced, mapping)
    assert compare_exact_ptos(reduced, expert, 16, 5, random.Random(42))["all_exact"]


def test_explicit_halt_can_be_retained_without_stopping_edges():
    from differentiable_exact_learning.analysis.extraction import with_halt_state

    original = with_halt_state(Transducer({(0, 0): (0, 0, 0)}, 0, (0, 1)), 2)
    reduced, mapping = hopcroft_minimize(original)
    assert not reduced.final_states
    reduced = with_halt_state(reduced, 2)
    assert verify_quotient(original, reduced, mapping)
    assert len(reduced.final_states) == 1
    assert (0, 1) not in reduced.transitions
