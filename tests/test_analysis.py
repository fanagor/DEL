"""Scientific checks for notebook analysis, independent of external checkpoints."""

import json

import pytest
import torch

from differentiable_exact_learning.analysis.extraction import (
    Transducer,
    compare_traces,
    export_machine,
    extract_argmax,
    from_merge_graph,
    minimize,
    observed_submachine,
)
from differentiable_exact_learning.analysis.merging import (
    cluster_trajectories_epschain,
    greedy_metric_merging,
)
from differentiable_exact_learning.analysis.notebook_utils import (
    closed_loop_trajectories,
    evaluation_history,
    hidden_trajectories,
    load_analysis_run,
    load_metrics,
    sample_traces,
)
from differentiable_exact_learning.analysis.stability import IncrementalGraphCC


@pytest.mark.parametrize("softmax", [False, True])
def test_exact_demo_tensor_extraction_preserves_prestate_outputs(softmax):
    model, expert, _, _ = load_analysis_run(softmax=softmax)
    traces = sample_traces(expert, count=8, max_length=3)
    machine = extract_argmax(model)
    assert compare_traces(machine, traces)["exact_matches"] == len(traces)
    smaller, _ = minimize(machine)
    assert compare_traces(smaller, traces)["exact_matches"] == len(traces)
    assert len(smaller.states) <= len(machine.states)


def test_hidden_trajectory_matches_model_forward():
    model, expert, _, _ = load_analysis_run()
    trace = sample_traces(expert, count=1)[0]
    trajectory = hidden_trajectories(model, [trace], expert.motion_vocab["S"])[0]
    with torch.no_grad():
        symbol, motion, state = model(trace.inputs[None, :])
    assert trajectory["sig_seq"] == list(
        zip(symbol[0].argmax(-1).tolist(), motion[0].argmax(-1).tolist())
    )
    assert torch.allclose(trajectory["h_seq"][-1], state[0])


def test_closed_loop_exact_controller_reproduces_expert_observations():
    model, expert, _, _ = load_analysis_run()
    traces = sample_traces(expert, count=3)
    trajectories = closed_loop_trajectories(model, expert, traces)
    for trajectory, trace in zip(trajectories, traces):
        assert trajectory["x_seq"] == trace.inputs.tolist()
        assert trajectory["sig_seq"] == list(zip(trace.symbols.tolist(), trace.motions.tolist()))
        assert trajectory["halt_seq"][-1]


def test_closed_loop_step_limit_is_not_silently_halt():
    model, expert, _, _ = load_analysis_run()
    with pytest.raises(RuntimeError, match="did not halt"):
        closed_loop_trajectories(model, expert, sample_traces(expert, count=1), max_steps=1)


def test_epsilon_chaining_and_merging_keep_observed_policy():
    model, expert, _, _ = load_analysis_run()
    traces = sample_traces(expert, count=5)
    trajectories = closed_loop_trajectories(model, expert, traces)
    ok, graph = cluster_trajectories_epschain(
        trajectories, expert.motion_vocab["S"], expert.input_size, 0.125, verbose=False
    )
    assert ok
    _, _, _, merged = greedy_metric_merging(graph, 0.125, verbose=False, run_phase1=False)
    assert compare_traces(from_merge_graph(merged), traces)["exact_matches"] == len(traces)
    assert merged.n_nodes() <= graph.n_nodes()


def test_incremental_stability_repeated_samples_preserve_edge_counts():
    model, expert, _, _ = load_analysis_run()
    trajectories = closed_loop_trajectories(model, expert, sample_traces(expert, count=2))
    graph = IncrementalGraphCC(0.125)
    graph.add(trajectories)
    edges, n, conflict = graph.compute_graph()
    assert not conflict
    graph.add(trajectories)
    again, n_again, conflict = graph.compute_graph()
    assert not conflict
    assert edges == again and n == n_again


def test_partial_minimization_preserves_unknowns_and_output_labels():
    machine = Transducer({(0, 0): (1, 0, 0), (1, 0): (1, 1, 0)}, 0, (0, 1))
    smaller, _ = minimize(machine)
    assert len(smaller.states) == 2
    assert smaller.predict([0, 1]) == ([(0, 0)], False)
    assert smaller.predict([0, 0]) == ([(0, 0), (1, 0)], True)


def test_exports_and_coverage(tmp_path):
    model, expert, _, _ = load_analysis_run()
    traces = sample_traces(expert, count=2)
    machine = observed_submachine(extract_argmax(model), traces)
    artifact = export_machine(machine, tmp_path / "machine")
    assert json.loads((tmp_path / "machine.json").read_text()) == artifact
    assert (tmp_path / "machine.csv").is_file() and (tmp_path / "machine.dot").is_file()
    assert compare_traces(machine, traces)["covered"] == 2


def test_missing_actual_metrics_does_not_fall_back_to_demo(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_metrics(tmp_path)


def test_evaluation_history_keeps_task_and_censoring(tmp_path):
    data = {"step": 20, "tasks": {"addition": {"rlg": {"length": 30, "at_cap": True}}}}
    (tmp_path / "evaluation_0000000020.json").write_text(json.dumps(data))
    (tmp_path / "evaluation.json").write_text(json.dumps(data))
    rows = evaluation_history(tmp_path)
    assert rows == [{"step": 20, "rlg": 30, "at_cap": True, "task": "addition"}]


def test_closed_loop_transducer_unknowns_and_halt_are_failures():
    from differentiable_exact_learning.analysis.extraction import compare_closed_loop

    model, expert, _, _ = load_analysis_run()
    traces = sample_traces(expert, count=2)
    machine = extract_argmax(model)
    assert compare_closed_loop(machine, expert, traces)["exact_matches"] == 2
    empty = Transducer({}, 0, machine.alphabet)
    result = compare_closed_loop(empty, expert, traces)
    assert result["covered"] == 0 and result["halted"] == 0 and result["exact_matches"] == 0


def test_observed_basis_never_includes_zero_or_more_columns_than_dimension():
    from differentiable_exact_learning.analysis.geometry import build_orthogonal_basis

    h0 = torch.tensor([1.0])
    basis = build_orthogonal_basis([torch.zeros(1), h0], h0)
    assert basis.shape == (1, 1)
    assert torch.equal(build_orthogonal_basis([], h0), h0[:, None])
    with pytest.raises(ValueError, match="nonzero"):
        build_orthogonal_basis([], torch.zeros(1))


@pytest.mark.parametrize("method", ["epsilon_chain", "epsilon_net", "kmeans"])
def test_clustering_retries_incompatible_outputs_without_losing_evidence(method):
    from differentiable_exact_learning.analysis.clustering import (
        cluster_trajectories_epschain,
        cluster_trajectories_epsnet,
        cluster_trajectories_kmeans,
    )

    trajectory = {
        "h_seq": [torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0]), torch.tensor([0.0, 0.0])],
        "x_seq": [0, 0],
        "sig_seq": [(0, 0), (1, 1)],
        "halt_seq": [False, False, True],
    }
    if method == "epsilon_chain":
        ok, graph = cluster_trajectories_epschain([trajectory], 1, 2, eps=2.0, verbose=False)
    elif method == "epsilon_net":
        ok, graph = cluster_trajectories_epsnet([trajectory], 1, 2, delta=2.0, verbose=False)
    else:
        ok, graph = cluster_trajectories_kmeans(
            [trajectory], 1, 2, k_start=1, k_step=1, verbose=False
        )
    assert ok
    assert from_merge_graph(graph).predict([0, 0]) == ([(0, 0), (1, 1)], True)
    _, _, _, merged = greedy_metric_merging(graph, 0.1, verbose=False, run_phase1=False)
    assert from_merge_graph(merged).predict([0, 0]) == ([(0, 0), (1, 1)], True)
