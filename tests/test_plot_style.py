"""Probability diagnostics and sparse heatmap labels used by research notebooks."""

import ast
import json
from pathlib import Path

import pytest
import torch


def sharpness_function():
    path = Path(__file__).parents[1] / "notebooks/anneal_dfst_softmax_to_fst.ipynb"
    for cell in json.loads(path.read_text())["cells"]:
        if cell["cell_type"] != "code":
            continue
        for node in ast.parse("".join(cell["source"])).body:
            if isinstance(node, ast.FunctionDef) and node.name == "sharpness_rows":
                namespace = {"torch": torch}
                exec(
                    compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace
                )
                return namespace["sharpness_rows"]
    raise AssertionError("Missing sharpness_rows")


def test_probability_diagnostics_use_the_correct_worst_columns_and_entropy():
    p = torch.tensor([[[0.4, 0.6], [0.35, 0.4], [0.25, 0.0]]], dtype=torch.float64)
    rows = sharpness_function()({"A": p, "h0": torch.tensor([1.0, 0.0])}, 7)
    expected_keys = ["step", "tensor", "columns", "worst_max", "worst_second", "mean_entropy"]
    assert list(rows[0]) == expected_keys
    assert rows[0]["step"] == 7 and rows[0]["columns"] == 2
    assert rows[0]["worst_max"] == pytest.approx(0.4)
    assert rows[0]["worst_second"] == pytest.approx(0.4)
    assert rows[0]["mean_entropy"] == pytest.approx(
        float(-torch.special.xlogy(p, p).sum(-2).mean())
    )
    assert rows[1] == dict(
        step=7, tensor="h_0", columns=1, worst_max=1.0, worst_second=0.0, mean_entropy=0.0
    )


@pytest.mark.parametrize("shape", [(1, 1), (4, 4), (5, 22), (22, 22), (128, 256)])
def test_heatmap_axes_have_at_most_five_ticks(shape):
    plt = pytest.importorskip("matplotlib.pyplot")
    from differentiable_exact_learning.analysis.plot_style import heatmap_ticks

    fig, ax = plt.subplots()
    ax.imshow(torch.zeros(shape).numpy())
    heatmap_ticks(ax, shape)
    assert 1 <= len(ax.get_xticks()) <= 5
    assert 1 <= len(ax.get_yticks()) <= 5
    assert ax.get_xticks()[0] == 0
    assert ax.get_xticks()[-1] == shape[1] - 1
    assert ax.get_yticks()[0] == 0 and ax.get_yticks()[-1] == shape[0] - 1
    assert not ax.get_xticks(minor=True).size and not ax.get_yticks(minor=True).size
    assert not any(line.get_visible() for line in ax.get_xgridlines() + ax.get_ygridlines())
    plt.close(fig)
