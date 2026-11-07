"""Coverage-aware deterministic Mealy transducers extracted from DFST tensors."""

import csv
import json
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Transducer:
    """Each (state,input) maps to (next_state,symbol,motion); missing means unknown."""

    transitions: dict
    initial_state: int
    alphabet: tuple
    final_states: frozenset = frozenset()

    @property
    def states(self):
        return sorted(
            {self.initial_state}
            | {q for q, _ in self.transitions}
            | {edge[0] for edge in self.transitions.values()}
        )

    def predict(self, tokens):
        state = self.initial_state
        outputs = []
        for x in tokens:
            edge = self.transitions.get((state, int(x)))
            if edge is None:
                return outputs, False
            state, symbol, motion = edge
            outputs.append((symbol, motion))
        return outputs, True


def extract_argmax(model):
    """Round columns, then retain states reachable from argmax(h0).

    This discretization is a hypothesis, not an equivalence certificate. It is
    most interpretable for stochastic/near-one-hot controllers. B/C read the
    PRE-transition state, matching model.forward exactly.
    """
    A, B, C, h0 = (x.detach().cpu() for x in model.effective_tensors())
    initial = int(h0.argmax())
    alphabet = tuple(range(A.shape[0]))
    todo, seen, transitions = deque([initial]), {initial}, {}
    while todo:
        q = todo.popleft()
        for x in alphabet:
            dest = int(A[x, :, q].argmax())
            transitions[q, x] = (dest, int(B[x, :, q].argmax()), int(C[x, :, q].argmax()))
            if dest not in seen:
                seen.add(dest)
                todo.append(dest)
    return Transducer(transitions, initial, alphabet)


def with_halt_state(machine, stop_motion):
    """Make stopping explicit without changing PTOs up to and including S.

    Reuse a sole existing final state, otherwise add a dedicated one. Only
    stopping edges are redirected; their written symbols are preserved. Halt
    loops are a convention after execution has stopped, not learned evidence.
    Missing transitions outside the halt state remain unknown.
    """
    halt = (
        next(iter(machine.final_states))
        if len(machine.final_states) == 1
        else max(machine.states, default=-1) + 1
    )
    transitions = {
        (q, x): (halt if motion == stop_motion else target, symbol, motion)
        for (q, x), (target, symbol, motion) in machine.transitions.items()
    }
    transitions.update({(halt, x): (halt, x, stop_motion) for x in machine.alphabet})
    return Transducer(transitions, machine.initial_state, machine.alphabet, frozenset({halt}))


def minimize(machine):
    """Partition refinement preserving all outputs, definedness and final status.

    Undefined transitions remain undefined; this is conservative for partial
    automata. Geometric compatibility merging is provided separately in merging.
    """
    states = machine.states
    blocks = [set(states)]
    while True:
        block_of = {q: i for i, block in enumerate(blocks) for q in block}
        groups = {}
        for q in states:
            signature = (
                q in machine.final_states,
                tuple(
                    None
                    if (q, x) not in machine.transitions
                    else (machine.transitions[q, x][1:], block_of[machine.transitions[q, x][0]])
                    for x in machine.alphabet
                ),
            )
            groups.setdefault(signature, set()).add(q)
        refined = list(groups.values())
        if {frozenset(b) for b in refined} == {frozenset(b) for b in blocks}:
            break
        blocks = refined
    blocks.sort(key=lambda b: (machine.initial_state not in b, min(b)))
    index = {q: i for i, block in enumerate(blocks) for q in block}
    transitions = {}
    for i, block in enumerate(blocks):
        representative = min(block)
        for x in machine.alphabet:
            if (representative, x) in machine.transitions:
                target, symbol, motion = machine.transitions[representative, x]
                transitions[i, x] = (index[target], symbol, motion)
    return Transducer(
        transitions,
        index[machine.initial_state],
        machine.alphabet,
        frozenset(index[q] for q in machine.final_states),
    ), index


def from_merge_graph(graph, complete=False):
    """Export a MergeGraph; completion is an explicitly requested halt heuristic."""
    edges = graph.complete_and_export() if complete else graph.edges
    transitions = {
        (q, x): (int(e["target"]), int(e["sym"]), int(e["mot"]))
        for (q, x), e in edges.items()
        if q in graph.active
    }
    return Transducer(
        transitions, graph.initial_id, tuple(range(graph.d_vocab)), frozenset([graph.halt_id])
    )


def observed_submachine(machine, traces):
    """Keep only transitions traversed by the supplied input traces."""
    kept = {}
    for trace in traces:
        q = machine.initial_state
        for x in trace.inputs.tolist():
            if (q, x) not in machine.transitions:
                break
            kept[q, x] = machine.transitions[q, x]
            q = kept[q, x][0]
    return Transducer(kept, machine.initial_state, machine.alphabet, machine.final_states)


def compare_traces(machine, traces):
    """Teacher-forced exact-match and coverage; unknowns count as failures."""
    correct = covered = 0
    for trace in traces:
        output, complete = machine.predict(trace.inputs.tolist())
        expected = list(zip(trace.symbols.tolist(), trace.motions.tolist()))
        covered += int(complete)
        correct += int(complete and output == expected)
    n = len(traces)
    return {
        "sequences": n,
        "covered": covered,
        "exact_matches": correct,
        "accuracy": correct / n if n else float("nan"),
    }


def transition_rows(machine):
    return [
        {"state": q, "input": x, "next_state": nxt, "symbol": s, "motion": m}
        for (q, x), (nxt, s, m) in sorted(machine.transitions.items())
    ]


def export_machine(machine, prefix):
    """Portable canonical JSON, long-form CSV, and Graphviz DOT (no renderer needed)."""
    prefix = Path(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    rows = transition_rows(machine)
    artifact = {
        "initial_state": machine.initial_state,
        "final_states": sorted(machine.final_states),
        "alphabet": list(machine.alphabet),
        "states": machine.states,
        "transitions": rows,
    }
    prefix.with_suffix(".json").write_text(json.dumps(artifact, indent=2) + "\n")
    with prefix.with_suffix(".csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["state", "input", "next_state", "symbol", "motion"]
        )
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "digraph FST {",
        "rankdir=LR;",
        "start [shape=point];",
        f"start -> q{machine.initial_state};",
    ]
    for row in rows:
        lines.append(
            f'q{row["state"]} -> q{row["next_state"]} [label="{row["input"]}/{row["symbol"]},{row["motion"]}"];'
        )
    lines.append("}")
    prefix.with_suffix(".dot").write_text("\n".join(lines) + "\n")
    return artifact


def compare_closed_loop(machine, expert, traces, max_steps=10000):
    """Compare autonomous FST executions with expert PTOs from identical instances.

    The extracted machine controls its own writes and moves. Unknown transitions,
    nontermination, and an early halt count as failures; they are not dropped.
    """
    if max_steps < 1:
        raise ValueError("max_steps must be positive")
    symbols = {value: key for key, value in expert.vocab.items()}
    motions = {value: key for key, value in expert.motion_vocab.items()}
    exact = halted = covered = 0
    for trace in traces:
        memory = expert.make_environment(trace.metadata)
        state = machine.initial_state
        observed = []
        complete = True
        did_halt = False
        for _ in range(max_steps):
            token = expert.vocab[memory.read()]
            transition = machine.transitions.get((state, token))
            if transition is None:
                complete = False
                break
            state, symbol, motion = transition
            observed.append((token, symbol, motion))
            memory.write(symbols[symbol])
            memory.move(motions[motion])
            if motions[motion] == "S":
                did_halt = True
                break
        expected = list(zip(trace.inputs.tolist(), trace.symbols.tolist(), trace.motions.tolist()))
        covered += int(complete)
        halted += int(did_halt)
        exact += int(complete and did_halt and observed == expected)
    n = len(traces)
    return {
        "sequences": n,
        "covered": covered,
        "halted": halted,
        "exact_matches": exact,
        "accuracy": exact / n if n else float("nan"),
    }


def hopcroft_minimize(machine):
    """Minimize the reachable Mealy transducer with Hopcroft worklist refinement."""
    # Restrict to states reachable from the designated start.
    reachable, queue = {machine.initial_state}, deque([machine.initial_state])
    while queue:
        q = queue.popleft()
        for x in machine.alphabet:
            edge = machine.transitions.get((q, x))
            if edge is not None and edge[0] not in reachable:
                reachable.add(edge[0])
                queue.append(edge[0])
    groups = defaultdict(set)
    predecessors = {x: defaultdict(set) for x in machine.alphabet}
    for q in sorted(reachable):
        signature = (
            q in machine.final_states,
            tuple(
                None if (q, x) not in machine.transitions else machine.transitions[q, x][1:]
                for x in machine.alphabet
            ),
        )
        groups[signature].add(q)
        for x in machine.alphabet:
            if (q, x) in machine.transitions:
                predecessors[x][machine.transitions[q, x][0]].add(q)
    blocks = dict(enumerate(groups.values()))
    block_of = {q: b for b, members in blocks.items() for q in members}
    work, queued = deque(blocks), set(blocks)
    while work:
        splitter_id = work.popleft()
        queued.remove(splitter_id)
        splitter = set(blocks[splitter_id])
        for x in machine.alphabet:
            incoming = set()
            for target in splitter:
                incoming.update(predecessors[x].get(target, ()))
            touched = defaultdict(set)
            for q in incoming:
                touched[block_of[q]].add(q)
            for b, intersection in touched.items():
                remainder = blocks[b] - intersection
                if not remainder:
                    continue
                new_b = len(blocks)
                blocks[b], blocks[new_b] = intersection, remainder
                for q in remainder:
                    block_of[q] = new_b
                if b in queued:
                    work.append(new_b)
                    queued.add(new_b)
                else:
                    smaller = b if len(intersection) <= len(remainder) else new_b
                    if smaller not in queued:
                        work.append(smaller)
                        queued.add(smaller)
    ordered = sorted(
        blocks.values(), key=lambda block: (machine.initial_state not in block, min(block))
    )
    mapping = {q: i for i, block in enumerate(ordered) for q in block}
    transitions = {}
    for i, block in enumerate(ordered):
        q = min(block)
        for x in machine.alphabet:
            if (q, x) in machine.transitions:
                target, symbol, motion = machine.transitions[q, x]
                transitions[i, x] = (mapping[target], symbol, motion)
    quotient = Transducer(
        transitions,
        mapping[machine.initial_state],
        machine.alphabet,
        frozenset(mapping[q] for q in reachable if q in machine.final_states),
    )
    return quotient, mapping


def verify_quotient(original, quotient, mapping):
    """Check the exact state-map certificate, including missing edges and finals."""
    # Exact finite certificate: outputs, missing edges, start/halt and successors.
    assert mapping[original.initial_state] == quotient.initial_state
    assert original.alphabet == quotient.alphabet
    for q, reduced in mapping.items():
        assert (q in original.final_states) == (reduced in quotient.final_states)
        for x in original.alphabet:
            before, after = original.transitions.get((q, x)), quotient.transitions.get((reduced, x))
            if before is None:
                assert after is None
            else:
                target, symbol, motion = before
                assert after == (mapping[target], symbol, motion)
    return True


def compare_exact_ptos(machine, expert, count, length, rng, *, tasks=None):
    """Stream expert PTOs and reject at the first differing action.

    Equal writes/moves from the same initial memory imply equal subsequent reads.
    Thus full action agreement, including halt, establishes exact autonomous PTO
    agreement on that input. A mismatch already establishes failure, so no
    independent learned rollout or artificial rollout cap is needed.
    """
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("Input count must be a positive integer")
    minimum = 0 if expert.environment == "teleporting" else 1
    if isinstance(length, bool) or not isinstance(length, int) or length < minimum:
        raise ValueError(f"Input length must be an integer >= {minimum}")
    if tasks is not None:
        from ..data.experts import task_label

        tasks = [task_label(task) for task in tasks]
        if expert.environment != "teleporting" or not tasks or len(tasks) != len(set(tasks)):
            raise ValueError("Explicit tasks require a nonempty distinct universal task list")
    exact_length = length if expert.environment == "teleporting" else (length, length)
    results = []
    for index in range(count):
        state, checked, failure, metadata = machine.initial_state, 0, None, None
        stream = expert.sample_stream(
            rng,
            max_length=length,
            lengths=exact_length,
            task=rng.choice(tasks) if tasks is not None else None,
        )
        for chunk in stream:
            metadata = chunk.metadata
            for x, symbol, motion in zip(
                chunk.inputs.tolist(), chunk.symbols.tolist(), chunk.motions.tolist()
            ):
                checked += 1
                edge = machine.transitions.get((state, x))
                if edge is None or edge[1:] != (symbol, motion):
                    failure = dict(
                        step=checked,
                        state=state,
                        input=x,
                        expected=[symbol, motion],
                        actual=list(edge[1:]) if edge is not None else None,
                        reason="unknown transition" if edge is None else "action mismatch",
                    )
                    break
                state = edge[0]
            if failure is not None:
                break
        stream.close()
        results.append(
            dict(
                input_index=index + 1,
                exact_match=failure is None,
                checked_steps=checked,
                metadata=metadata,
                failure=failure,
            )
        )
    matches = sum(row["exact_match"] for row in results)
    return dict(
        inputs=count,
        input_length=length,
        exact_matches=matches,
        failures=count - matches,
        all_exact=matches == count,
        results=results,
    )


def number_states(machine):
    """Number states consecutively, with the start first and other halt states last."""
    order = [machine.initial_state]
    order.extend(
        q for q in machine.states if q != machine.initial_state and q not in machine.final_states
    )
    order.extend(
        q for q in machine.states if q != machine.initial_state and q in machine.final_states
    )
    mapping = {q: index for index, q in enumerate(order)}
    numbered = Transducer(
        {
            (mapping[q], x): (mapping[target], symbol, motion)
            for (q, x), (target, symbol, motion) in machine.transitions.items()
        },
        0,
        machine.alphabet,
        frozenset(mapping[q] for q in machine.final_states if q in mapping),
    )
    return numbered, mapping
