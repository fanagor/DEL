import json
import runpy
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
import yaml

from differentiable_exact_learning.analysis.front_page import ARITHMETIC_ORDER, plot_front_page
from differentiable_exact_learning.analysis.universal_figures import TASK_ORDER


def example_records():
    arithmetic = {}
    for name, maximum in zip(ARITHMETIC_ORDER, [3, 5, 4, 7]):
        task, radix = (
            ("addition" if name.startswith("add") else "multiplication"),
            (10 if name.endswith("10") else 2),
        )
        arithmetic[name] = {
            "config": {
                "model": {"name": "DFST"},
                "data": {"task": task, "radix": radix, "max_length": maximum, "size": 25},
            },
            "evaluation": {"tasks": {task + str(radix): {"ulg": {"length": 500, "at_cap": True}}}},
        }
    universal = {
        "config": {
            "model": {"name": "DFST_Softmax", "hidden_size": 15},
            "data": {
                "environment": "teleporting",
                "task": "universal",
                "size": 75,
                "max_length": 5,
            },
        },
        "evaluation": {
            "tasks": {t: {"generalization": {"length": 500, "at_cap": True}} for t in TASK_ORDER}
        },
    }
    return arithmetic, universal


def test_front_page_order_training_boundary_and_no_titles():
    arithmetic, universal = example_records()
    fig = plot_front_page(arithmetic, universal)
    assert len(fig.axes) == 2
    assert all(not ax.get_title() for ax in fig.axes)
    assert [t.get_text() for t in fig.axes[0].get_yticklabels()] == list(ARITHMETIC_ORDER)
    vertices = fig.axes[0].patches[0].get_xy()
    np.testing.assert_equal(vertices[1:9:2, 0], [3, 5, 4, 7])
    assert all(ax.get_xscale() == "log" for ax in fig.axes)
    plt.close(fig)


def test_final_loader_does_not_substitute_interim_scores(tmp_path):
    module = runpy.run_path(str(Path(__file__).parents[1] / "scripts/plot_front_page.py"))
    config = {
        "data": {"environment": "grid", "task": "addition", "radix": 2},
        "training": {"steps": 100},
    }
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(config))
    (tmp_path / "evaluation.json").write_text(json.dumps({"step": 50, "tasks": {}}))
    with pytest.raises(ValueError, match="final step 100"):
        module["load_final"](tmp_path, "add2")
    result = {"step": 100, "tasks": {"addition2": {"ulg": {"length": 80, "at_cap": False}}}}
    (tmp_path / "evaluation.json").write_text(json.dumps(result))
    assert module["load_final"](tmp_path, "add2")["evaluation"] == result
