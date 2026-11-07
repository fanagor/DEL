"""Teaching inputs supplement extraction without changing random sampling budgets."""

from copy import deepcopy

import pytest

from differentiable_exact_learning.analysis.notebook_utils import teaching_extraction_traces
from differentiable_exact_learning.data import create_expert
from differentiable_exact_learning.data.teaching import teaching_examples


@pytest.mark.parametrize(
    "task,radix,variant,count",
    [
        ("addition", 2, "default", 8),
        ("multiplication", 2, "default", 33),
        ("multiplication", 2, "new", 28),
        ("multiplication", 10, "default", 1645),
    ],
)
def test_complete_arithmetic_core_ignores_training_budgets(task, radix, variant, count):
    data = dict(
        environment="grid",
        task=task,
        radix=radix,
        variant=variant,
        source="online",
        size=1,
        max_length=1,
        teaching_set=False,
    )
    original = deepcopy(data)
    expert = create_expert(data)
    traces = teaching_extraction_traces(expert)
    assert data == original
    assert len(traces) == count
    pairs = [tuple(trace.metadata["operands"]) for trace in traces]
    assert len(set(pairs)) == count
    assert set(pairs) == set(teaching_examples(data))
    assert max(max(map(len, pair)) for pair in pairs) > data["max_length"]
    assert all(trace.metadata["extraction_group"] == "Teaching Set" for trace in traces)
    assert all(int(trace.motions[-1]) == expert.motion_vocab["S"] for trace in traces)


def test_universal_core_includes_all_five_teaching_programs():
    data = dict(
        environment="teleporting",
        task="addition",
        radix=2,
        source="online",
        size=1,
        max_length=1,
        teaching_set=False,
        train_tasks=["copy"],
    )
    traces = teaching_extraction_traces(create_expert(data))
    assert len(traces) == 75
    assert {t.metadata["task"] for t in traces} == {
        "parity",
        "shift_left",
        "shift_right",
        "scaling_A_m5",
        "scaling_B_m5",
    }
    assert {t.metadata["lengths"][0] for t in traces} == {0, 1, 2, 3}
    assert len({(t.metadata["task"], t.metadata["input"]) for t in traces}) == 75
