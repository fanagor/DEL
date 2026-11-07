"""Generate the README overview from binary-addition PTOs and a curated FST."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

from differentiable_exact_learning.analysis.plot_style import BACKGROUND, COLORS, PAPER_RC
from differentiable_exact_learning.data import Memory, create_expert

ROOT = Path(__file__).resolve().parents[1]
CASE = ROOT / "analyses/metric_extraction/DFST/add2/warmup_50000_steps_250000"


def main():
    expert = create_expert(dict(environment="grid", task="addition", radix=2))
    execution = expert.simulate([list("101+11")])
    pto = list(zip(execution.trace.inputs, execution.trace.symbols, execution.trace.motions))
    decoded = [
        (expert.vocab_rev[int(r)], expert.vocab_rev[int(w)], expert.motion_vocab_rev[int(m)])
        for r, w, m in pto
    ]
    with (CASE / "machines/extracted.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    transitions = {(int(r["state"]), r["read"].replace("λ", "")): r for r in rows}
    memory, state, replay = Memory([list("101+11")], (0, 0), False), 0, []
    for _ in range(1000):
        read = memory.read()
        row = transitions[state, read]
        write, move = row["write"].replace("λ", ""), row["move"]
        replay.append((read, write, move))
        memory.write(write)
        memory.move(move)
        state = int(row["next_state"])
        if move == "S":
            break
    assert replay == decoded and memory.cells == execution.cells
    assert "".join(execution.cells[x, 0] for x in range(-1, 3)) == "1000"
    assert transitions[0, "1"]["next_state"] == "1"
    assert transitions[1, "0"]["next_state"] == "1"
    assert transitions[1, "+"]["next_state"] == "2"
    blue, terra, _, green = COLORS
    ink = "#252525"
    io_color = "#6B5876"
    with plt.rc_context(PAPER_RC):
        fig, ax = plt.subplots(figsize=(14, 3.4))
        fig.patch.set_facecolor("white")
        ax.set(xlim=(0, 1400), ylim=(0, 340))
        ax.axis("off")
        fig.subplots_adjust(0, 0, 1, 1)

        def text(x, y, value, size=14, color=ink):
            ax.text(x, y, value, fontsize=size, color=color, va="center", ha="center")

        def arrow(a, b, color=ink, **kw):
            ax.add_patch(
                FancyArrowPatch(
                    a, b, arrowstyle="-|>", mutation_scale=15, linewidth=1.8, color=color, **kw
                )
            )

        for x, color, title in [
            (20, blue, "Observe"),
            (490, terra, "Learn Differentiably"),
            (960, green, "Extract"),
        ]:
            ax.add_patch(
                FancyBboxPatch(
                    (x, 20),
                    420,
                    300,
                    boxstyle="round,pad=0,rounding_size=14",
                    facecolor=BACKGROUND,
                    edgecolor="none",
                )
            )
            text(x + 210, 284, title, 21, color)
        arrow((451, 174), (479, 174), ink)
        arrow((921, 174), (949, 174), ink)

        # Initial input row, with the first observed cell highlighted.
        for i, symbol in enumerate("101+11"):
            x = 65 + i * 55
            ax.add_patch(
                Rectangle(
                    (x, 202),
                    50,
                    48,
                    facecolor="white",
                    edgecolor=blue if i == 0 else "#d2d2cf",
                    linewidth=2 if i == 0 else 1,
                )
            )
            text(x + 25, 226, symbol, 20, blue)
        text(140, 165, "read", 17, io_color)
        text(320, 165, "write · move", 17, io_color)
        ax.plot([92, 188], [146, 146], color=io_color, linewidth=1.2)
        ax.plot([245, 395], [146, 146], color=io_color, linewidth=1.2)
        text(140, 128, "environment input", 12, io_color)
        text(320, 128, "expert action", 12, io_color)
        text(230, 79, "PTO", 16, blue)
        text(230, 42, "Policy-Trajectory Observation", 12)

        text(700, 250, "environment input", 14, io_color)
        arrow((700, 232), (700, 200), terra)
        ax.add_patch(
            FancyBboxPatch(
                (550, 142),
                300,
                56,
                boxstyle="round,pad=0,rounding_size=9",
                facecolor="white",
                edgecolor=terra,
                linewidth=2,
            )
        )
        text(700, 182, "State-Tracking", 17, terra)
        text(700, 158, "Neural Network", 17, terra)
        arrow((700, 140), (700, 108), terra)
        text(700, 90, "expert action", 14, io_color)
        text(700, 42, "Extreme Length Generalization", 12, terra)

        # Topology of the exported 0 -> 1 -> 2 fragment, with the loop at state 1.
        # State identifiers and transition labels stay in the linked case artifacts.
        for x in (1070, 1170, 1270):
            ax.add_patch(Circle((x, 174), 20, facecolor="white", edgecolor=green, linewidth=2))
        arrow((1093, 174), (1147, 174), green)
        arrow((1193, 174), (1247, 174), green)
        arrow((1158, 191), (1182, 191), green, connectionstyle="arc3,rad=-1.65")
        text(1170, 100, "a discrete controller", 14)

        text(1170, 42, "Provably Equivalent to Expert", 12, green)

        output = ROOT / "docs/assets"
        output.mkdir(parents=True, exist_ok=True)
        fig.savefig(output / "binary_addition_overview.svg", metadata={"Date": None})
        fig.savefig(output / "binary_addition_overview.png", dpi=160)
        plt.close(fig)


if __name__ == "__main__":
    main()
