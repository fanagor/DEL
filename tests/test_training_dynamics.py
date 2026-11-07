"""Scientific data handling in the merged training-dynamics workflow."""

import json

import matplotlib.pyplot as plt
import numpy as np
import pytest
import yaml

from differentiable_exact_learning.analysis.training_dynamics import (
    MODEL_COLORS,
    MODELS,
    average_loss,
    load_history,
    plot_training_dynamics,
)


def make_run(tmp_path, model="DFST", task="add2"):
    root = tmp_path / model / task
    root.mkdir(parents=True)
    config = dict(
        model=dict(name=model),
        data=dict(task="addition" if task == "add2" else "multiplication", radix=2),
        training=dict(steps=300000),
        evaluation=dict(seed=42, samples=10, max_length=1000, search="binary", scan="sequential"),
    )
    (root / "config.yaml").write_text(yaml.safe_dump(config))
    (root / "metrics.jsonl").write_text(
        "\n".join(
            json.dumps(dict(step=step, loss=loss))
            for step, loss in [(100000, 1.0), (200000, 0.1), (300000, 0.01)]
        )
    )
    label = "addition2" if task == "add2" else "multiplication2"
    for step, length in [(100000, 0), (200000, 60), (300000, 1000)]:
        value = dict(
            step=step,
            tasks={
                label: dict(
                    ulg=dict(length=length, cap=1000, at_cap=length == 1000), rlg=dict(length=1)
                )
            },
        )
        (root / f"evaluation_{step:010d}.json").write_text(json.dumps(value))
    return root


def test_loss_average_weights_updates_not_rows():
    np.testing.assert_allclose(average_loss([100, 300, 400], [2, 8, 4], 200), [2, 8, 6])
    np.testing.assert_allclose(average_loss([100, 300], [2, 8], 1000), [2, 6])


@pytest.mark.parametrize(
    "steps,losses",
    [([100, 100], [1, 2]), ([100, 90], [1, 2]), ([100], [float("nan")]), ([100], [-1])],
)
def test_invalid_history_rejected(steps, losses):
    with pytest.raises(ValueError):
        average_loss(steps, losses, 100)


def test_load_uses_ulg_and_preserves_missing_and_zero(tmp_path):
    root = make_run(tmp_path)
    (root / "evaluation_0000200000.json").unlink()
    h = load_history(root, model="DFST", task="add2")
    assert h["missing_steps"] == [200000]
    assert h["ulg"].ulg.iloc[0] == 0 and np.isnan(h["ulg"].ulg.iloc[1])
    assert h["ulg"].ulg.iloc[2] == 1000 and h["ulg"].at_cap.iloc[2]
    with pytest.raises(ValueError):
        load_history(root, model="RNN", task="add2")


def test_duplicate_and_protocol_conflicts_rejected(tmp_path):
    root = make_run(tmp_path)
    data = json.loads((root / "evaluation_0000300000.json").read_text())
    data["tasks"]["addition2"]["ulg"]["length"] = 42
    data["tasks"]["addition2"]["ulg"]["at_cap"] = False
    (root / "evaluation.json").write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Conflicting"):
        load_history(root, model="DFST", task="add2")
    (root / "evaluation.json").unlink()
    data["seed"] = 777
    (root / "evaluation_0000300000.json").write_text(json.dumps(data))
    with pytest.raises(ValueError, match="settings differ"):
        load_history(root, model="DFST", task="add2")


def test_rlg_only_result_does_not_become_ulg(tmp_path):
    root = make_run(tmp_path)
    p = root / "evaluation_0000200000.json"
    d = json.loads(p.read_text())
    del d["tasks"]["addition2"]["ulg"]
    p.write_text(json.dumps(d))
    assert load_history(root, model="DFST", task="add2")["missing_steps"] == [200000]


def test_two_panels_supported_models_and_no_bridge_across_missing(tmp_path):
    histories = {m: load_history(make_run(tmp_path, m), model=m, task="add2") for m in MODELS}
    fig = plot_training_dynamics(histories, "add2")
    assert len(fig.axes) == 2 and fig.axes[0].get_yscale() == "log"
    for ax in fig.axes:
        lines = {line.get_label(): line for line in ax.lines if line.get_label() in MODELS}
        assert set(lines) == set(MODELS)
        assert {m: line.get_color() for m, line in lines.items()} == MODEL_COLORS
    plt.close(fig)
    histories["DFST"]["missing_steps"] = [200000]
    histories["DFST"]["ulg"].loc[1, "ulg"] = np.nan
    with pytest.raises(ValueError, match="incomplete"):
        plot_training_dynamics(histories, "add2")
    fig = plot_training_dynamics(histories, "add2", require_complete=False)
    line = next(line for line in fig.axes[1].lines if line.get_label() == "DFST")
    assert np.isnan(line.get_ydata()[1])
    plt.close(fig)
    histories["RNN"]["protocol"]["seed"] = 111
    with pytest.raises(ValueError, match="different ULG"):
        plot_training_dynamics(histories, "add2", require_complete=False)


@pytest.mark.parametrize("models", [("DFST",), ("DFST", "GRU"), ("RNN", "GRU")])
def test_optional_architectures_keep_colors_and_validate_data(tmp_path, models):
    histories = {m: load_history(make_run(tmp_path, m), model=m, task="add2") for m in models}
    fig = plot_training_dynamics(histories, "add2")
    for ax in fig.axes:
        lines = {line.get_label(): line for line in ax.lines if line.get_label() in MODELS}
        assert set(lines) == set(models)
        assert {m: line.get_color() for m, line in lines.items()} == {
            m: MODEL_COLORS[m] for m in models
        }
        assert [text.get_text() for text in ax.get_legend().get_texts()] == list(models)
    plt.close(fig)
    histories[models[0]]["missing_steps"] = [200000]
    with pytest.raises(ValueError, match="incomplete"):
        plot_training_dynamics(histories, "add2")


def test_empty_or_unknown_architectures_rejected():
    for histories in ({}, {"unknown": {}}):
        with pytest.raises(ValueError, match="supported architecture"):
            plot_training_dynamics(histories, "add2")


@pytest.mark.parametrize("metric", ["ulg", "rlg"])
@pytest.mark.parametrize("task", ["add2", "mult2"])
def test_task_layout_preserves_metrics_and_legends(tmp_path, metric, task):
    histories = {
        model: load_history(make_run(tmp_path, model, task), model=model, task=task, metric=metric)
        for model in ("DFST", "GRU")
    }
    fig = plot_training_dynamics(histories, task, metric=metric)
    assert len(fig.axes) == 2 and fig._suptitle is None
    left, right = fig.axes
    assert left.get_yscale() == "log"
    assert left.get_title().startswith(task) and right.get_title().startswith(task)
    assert right.get_ylabel() == metric.upper() + " Length"
    assert not right.collections
    for ax in fig.axes:
        assert [text.get_text() for text in ax.get_legend().get_texts()] == ["DFST", "GRU"]
    for line in right.lines:
        expected = [0, 60, 1000] if metric == "ulg" else [1, 1, 1]
        np.testing.assert_allclose(line.get_ydata(), expected)
    assert any(text.get_text() == "Evaluation Cap: 1000" for text in right.texts)
    plt.close(fig)


def test_loss_average_preserves_small_late_values():
    result = average_loss([100, 200, 300], [1.0, 1e-20, 2e-20], 100)
    np.testing.assert_allclose(result, [1.0, 1e-20, 2e-20], rtol=1e-12, atol=0)


def test_rlg_missing_values_and_hard_case_protocol(tmp_path):
    root = make_run(tmp_path)
    p = root / "evaluation_0000200000.json"
    data = json.loads(p.read_text())
    del data["tasks"]["addition2"]["rlg"]
    p.write_text(json.dumps(data))
    assert load_history(root, model="DFST", task="add2", metric="rlg")["missing_steps"] == [200000]
    assert load_history(root, model="DFST", task="add2")["missing_steps"] == []
    data["tasks"]["addition2"].update(rlg=dict(length=1), hard_cases="diagonal")
    p.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="RLG evaluation settings differ"):
        load_history(root, model="DFST", task="add2", metric="rlg")
