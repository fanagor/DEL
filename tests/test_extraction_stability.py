"""Epsilon-chain extraction, graph stability, and exact minimization."""

import random

import pytest
import torch

from differentiable_exact_learning.analysis.clustering import cluster_trajectories_epschain
from differentiable_exact_learning.analysis.extraction import (
    Transducer,
    from_merge_graph,
    hopcroft_minimize,
    minimize,
    verify_quotient,
)
from differentiable_exact_learning.analysis.stability import (
    IncrementalGraphCC,
    canonical_graph,
    extract_snapshot,
)


@pytest.fixture(scope="module")
def helpers():
    return dict(
        canonical_graph=canonical_graph,
        extract_snapshot=extract_snapshot,
        hopcroft_minimize=hopcroft_minimize,
        verify_quotient=verify_quotient,
    )


def cascade_traces(conflict=False):
    traces = []
    for branch, states in enumerate(([0.0, 10.0, 20.0, 40.0, 0.0], [0.0, 10.05, 30.0, 50.0, 0.0])):
        traces.append(
            dict(
                h_seq=[torch.tensor([s]) for s in states],
                x_seq=[branch, 2, 3, 4],
                sig_seq=[(0, 0), (0, 0), (branch if conflict else 1, 0), (0, 4)],
                halt_seq=[False, False, False, False, True],
            )
        )
    return traces


def test_successor_closure_cascades_until_fixed_point(helpers):
    traces = cascade_traces()
    incremental = IncrementalGraphCC(0.1)
    incremental.add(traces[:1])
    incremental.compute_graph()
    incremental.add(traces[1:])
    machine, _ = helpers["extract_snapshot"](incremental, range(5))
    assert len(machine.states) == 5  # start, geometric component, two forced blocks, halt
    for trajectory in traces:
        assert machine.predict(trajectory["x_seq"]) == (trajectory["sig_seq"], True)
    ok, graph = cluster_trajectories_epschain(traces, 4, 5, eps=0.1, min_eps=0.1, verbose=False)
    assert ok
    batch = from_merge_graph(graph)
    batch.transitions = {
        k: v for k, v in batch.transitions.items() if k[0] not in batch.final_states
    }
    assert helpers["canonical_graph"](batch) == helpers["canonical_graph"](machine)


def test_conflicts_after_forced_merges_reject_fixed_epsilon(helpers):
    builder = IncrementalGraphCC(0.1)
    builder.add(cascade_traces(conflict=True))
    with pytest.raises(ValueError, match="Fixed epsilon"):
        helpers["extract_snapshot"](builder, range(5))
    assert builder.eps == 0.1


def test_graph_identity_ignores_ids_but_detects_equal_size_changes(helpers):
    a = Transducer({(5, 0): (8, 0, 0), (8, 0): (8, 1, 0)}, 5, (0, 1))
    b = Transducer({(1, 0): (2, 0, 0), (2, 0): (2, 1, 0)}, 1, (0, 1))
    assert helpers["canonical_graph"](a) == helpers["canonical_graph"](b)
    b.transitions[2, 0] = (2, 0, 0)
    assert len(a.states) == len(b.states) and len(a.transitions) == len(b.transitions)
    assert helpers["canonical_graph"](a) != helpers["canonical_graph"](b)


def test_hopcroft_preserves_unknowns_and_halt_status(helpers):
    machine = Transducer(
        {(0, 0): (1, 0, 0), (0, 1): (2, 0, 0), (1, 0): (1, 1, 0)}, 0, (0, 1), frozenset([2])
    )
    small, mapping = helpers["hopcroft_minimize"](machine)
    assert helpers["verify_quotient"](machine, small, mapping)
    assert len(small.states) == 3
    assert small.predict([0, 1]) == ([(0, 0)], False)
    assert mapping[2] in small.final_states


def test_hopcroft_matches_independent_refinement_on_random_partial_machines(helpers):
    rng = random.Random(12042)
    for trial in range(200):
        n, width = rng.randrange(1, 20), rng.randrange(1, 5)
        alphabet, transitions = tuple(range(width)), {}
        for q in range(n):
            for x in alphabet:
                if rng.random() < 0.8:
                    symbol = rng.randrange(2) if trial % 2 else 0
                    transitions[q, x] = (rng.randrange(n), symbol, 0)
            if q + 1 < n:
                transitions[q, 0] = (q + 1, 0, 0)  # all states reachable
        final = frozenset(q for q in range(n) if rng.random() < 0.1)
        machine = Transducer(transitions, 0, alphabet, final)
        small, mapping = helpers["hopcroft_minimize"](machine)
        reference, _ = minimize(machine)
        assert helpers["verify_quotient"](machine, small, mapping)
        assert len(small.states) == len(reference.states), trial


def test_distinct_grid_sampling_reuses_sampler_state_and_checks_capacity():
    from differentiable_exact_learning.analysis.notebook_utils import (
        load_analysis_run,
        sample_unique_grid_traces,
    )

    _, expert, _, _ = load_analysis_run()
    traces = sample_unique_grid_traces(expert, 8, 2, random.Random(42))
    again = sample_unique_grid_traces(expert, 8, 2, random.Random(42))
    keys = [tuple(tuple(x) for x in t.metadata["operand_tokens"]) for t in traces]
    assert len(set(keys)) == 8
    assert [t.metadata for t in traces] == [t.metadata for t in again]
    with pytest.raises(ValueError, match="only 4 exist"):
        sample_unique_grid_traces(expert, 5, 1, random.Random(42))


def test_closed_loop_rejects_nonfinite_readout_before_argmax():
    from types import SimpleNamespace

    from differentiable_exact_learning.analysis.notebook_utils import (
        closed_loop_trajectories,
        load_analysis_run,
        sample_traces,
    )

    model, expert, _, _ = load_analysis_run()
    a, b, c, h0 = (t.detach().clone() for t in model.effective_tensors())
    b.fill_(torch.finfo(b.dtype).max)
    h0.fill_(2.0)
    finite_tensors = SimpleNamespace(effective_tensors=lambda: (a, b, c, h0))
    with pytest.raises(ValueError, match="Non-finite readout"):
        closed_loop_trajectories(finite_tensors, expert, sample_traces(expert, count=1))


def test_select_full_sample_then_sweep_every_trajectory():
    from differentiable_exact_learning.analysis.stability import (
        select_valid_epsilon,
        trajectory_stability,
    )

    traces = cascade_traces(conflict=True)
    epsilon, full, attempts = select_valid_epsilon(traces, tuple(range(5)), 0.1)
    assert epsilon < 0.1
    assert attempts[-1]["status"] == "accepted"
    assert len(attempts) > 1
    snapshots = list(trajectory_stability(traces, tuple(range(5)), epsilon))
    assert [row["samples"] for row, _ in snapshots] == [1, 2]
    assert all(row["epsilon"] == epsilon for row, _ in snapshots)
    assert snapshots[0][0]["edge_changes"] == snapshots[0][0]["edges"]
    assert canonical_graph(snapshots[-1][1]) == canonical_graph(full)
    for trajectory in traces:
        assert full.predict(trajectory["x_seq"]) == (trajectory["sig_seq"], True)


def test_select_epsilon_stops_on_irreducible_conflict():
    from differentiable_exact_learning.analysis.stability import select_valid_epsilon

    traces = cascade_traces(conflict=True)
    traces[1]["h_seq"][1] = traces[0]["h_seq"][1].clone()
    with pytest.raises(ValueError, match="epsilon zero"):
        select_valid_epsilon(traces, tuple(range(5)), 0)
