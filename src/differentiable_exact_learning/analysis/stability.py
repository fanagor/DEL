"""Incremental epsilon-chain graph stability, retained from the paper notebooks."""

from collections import defaultdict, deque

import torch

from .clustering import UnionFind
from .extraction import Transducer


class IncrementalGraphCC:
    """
    Incremental ε-chain graph builder.

    Speed notes
    -----------
    * _trans_by_src is a dict-of-dicts  {state_idx: {x: (sym, mot, tgt_idx)}}
      so each (state, x) pair is stored at most once.  For a deterministic
      DFST the same state always produces the same output on the same input,
      so the first observation wins.  This stops _trans_by_src from growing
      proportionally to the total number of trajectory steps.

    * Phase-A cluster membership (_pa_members, _pa_of_state) is built once
      inside add() after all union operations complete.  compute_graph() copies
      these dicts in O(K + n) instead of rebuilding by calling _find(i) for
      every state again.

    Phase A — ε-connected components (smaller index wins).
    Phase B — queue-based consistency & merging (mirrors cluster_trajectories_epschain):
      signature check → break on inconsistency,
      multi-target merge → union successor clusters, re-enqueue.
    """

    def __init__(self, eps, chunk=512):
        self.eps = eps
        self.chunk = chunk
        self._all_h = []
        self._halt_h = []
        # Deduplicated: state_idx -> {x: (sym, mot, tgt_idx)}
        self._trans_by_src = defaultdict(dict)
        self._sets = UnionFind()

        # Phase-A cache (rebuilt in add())
        self._pa_members = {}  # pa_rep -> set of state indices
        self._pa_of_state = {}  # state_idx -> pa_rep

        # Stored by compute_graph() for the hard-check cell
        self._cluster_of_state = {}
        self._final_members = {}

    # ── Phase-A union-find (smaller index wins) ───────────────────────────
    def _find(self, i):
        return self._sets.find(i)

    def _union(self, a, b):
        self._sets.union(a, b)

    def _rebuild_pa_cache(self):
        pa_members = defaultdict(set)
        pa_of_state = {}
        for i in range(len(self._all_h)):
            r = self._find(i)
            pa_members[r].add(i)
            pa_of_state[i] = r
        self._pa_members = dict(pa_members)
        self._pa_of_state = pa_of_state

    # ── ingest trajectories ───────────────────────────────────────────────
    def add(self, new_trajs):
        old_n = len(self._all_h)

        for traj in new_trajs:
            h_seq, x_seq = traj["h_seq"], traj["x_seq"]
            sig_seq, halt_seq = traj["sig_seq"], traj["halt_seq"]
            step_idx = []
            for h, is_halt in zip(h_seq, halt_seq):
                if is_halt:
                    step_idx.append(-1)
                    self._halt_h.append(h.float())
                else:
                    step_idx.append(len(self._all_h))
                    self._all_h.append(h.float())
            for t, (x, (sym, mot)) in enumerate(zip(x_seq, sig_seq)):
                src = step_idx[t]
                if src == -1:
                    continue
                # Only store first observation of (state, x) — deterministic DFST
                if x not in self._trans_by_src[src]:
                    self._trans_by_src[src][x] = (sym, mot, step_idx[t + 1])

        new_n = len(self._all_h)
        if new_n == old_n:
            self._rebuild_pa_cache()
            return

        self._sets.parent.update({i: i for i in range(old_n, new_n)})
        all_h_t = torch.stack(self._all_h).float()

        if old_n == 0:
            for s in range(0, new_n, self.chunk):
                chunk = all_h_t[s : s + self.chunk]
                D = torch.cdist(chunk, all_h_t)
                rs, cs = (D <= self.eps).nonzero(as_tuple=True)
                for r, c in zip(rs.tolist(), cs.tolist()):
                    a, b = s + int(r), int(c)
                    if a < b:
                        self._union(a, b)
        else:
            new_h_t = all_h_t[old_n:]
            for s in range(0, new_h_t.shape[0], self.chunk):
                chunk = new_h_t[s : s + self.chunk]
                D = torch.cdist(chunk, all_h_t)
                rs, cs = (D <= self.eps).nonzero(as_tuple=True)
                for r, c in zip(rs.tolist(), cs.tolist()):
                    a, b = old_n + s + int(r), int(c)
                    if a != b:
                        self._union(a, b)

        self._rebuild_pa_cache()

    # ── Phase B + edge extraction ─────────────────────────────────────────
    def compute_graph(self):
        """
        Returns (edges, n_comp, incon_msg).
          edges     : frozenset of (src_rep, dst_rep_or_"halt", x, sym, mot)
          n_comp    : final non-halt clusters + 1 (halt)
          incon_msg : "" if consistent; description on first inconsistency
        """
        n = len(self._all_h)
        if n == 0:
            return frozenset(), 1, ""

        # Start Phase B from the cached Phase-A state (O(K + n) copy)
        cluster_members = {k: set(v) for k, v in self._pa_members.items()}
        cluster_of_state = dict(self._pa_of_state)

        Q = deque(sorted(cluster_members.keys()))

        while Q:
            k = Q.popleft()
            if k not in cluster_members:
                continue

            sig_for_x = {}
            tgt_for_x = defaultdict(list)
            sig_incon = None

            for state_i in cluster_members[k]:
                for x, (sym, mot, tgt_idx) in self._trans_by_src.get(state_i, {}).items():
                    if x in sig_for_x:
                        if sig_for_x[x] != (sym, mot):
                            sig_incon = (k, x, {sig_for_x[x], (sym, mot)})
                            break
                    else:
                        sig_for_x[x] = (sym, mot)
                    tgt_c = -1 if tgt_idx < 0 else cluster_of_state[tgt_idx]
                    tgt_for_x[x].append(tgt_c)
                if sig_incon:
                    break

            if sig_incon:
                k_, x_, sigs_ = sig_incon
                msg = f"[sig incon]      comp={k_}  x={x_}  signatures={sigs_}"
                print(msg)
                return frozenset(), 0, msg

            target_sets = UnionFind()
            _fu, _uu = target_sets.find, target_sets.union

            halt_nonhalt = None
            for x, tgts in tgt_for_x.items():
                unique_t = set(tgts)
                if len(unique_t) <= 1:
                    continue
                real_t = {t for t in unique_t if t >= 0}
                if -1 in unique_t and real_t:
                    halt_nonhalt = (k, x)
                    break
                real_list = sorted(real_t)
                for t in real_list[1:]:
                    _uu(real_list[0], t)

            if halt_nonhalt:
                k_, x_ = halt_nonhalt
                msg = f"[halt/non-halt]  comp={k_}  x={x_}"
                print(msg)
                return frozenset(), 0, msg

            involved = {t for tgts in tgt_for_x.values() for t in tgts if t >= 0}
            if involved:
                groups = defaultdict(set)
                for t in involved:
                    groups[_fu(t)].add(t)

                for group in groups.values():
                    if len(group) <= 1:
                        continue
                    new_cid = min(group)
                    merged = set()
                    for t in group:
                        merged |= cluster_members.pop(t)
                    cluster_members[new_cid] = merged
                    for si in merged:
                        cluster_of_state[si] = new_cid
                    Q = deque(c for c in Q if c not in group)
                    Q.append(new_cid)

        edges = set()
        for state_i, x_dict in self._trans_by_src.items():
            src_c = cluster_of_state[state_i]
            for x, (sym, mot, tgt_idx) in x_dict.items():
                dst_c = "halt" if tgt_idx < 0 else cluster_of_state[tgt_idx]
                edges.add((src_c, dst_c, x, sym, mot))

        self._cluster_of_state = cluster_of_state
        self._final_members = dict(cluster_members)
        n_comp = len(cluster_members) + 1
        return frozenset(edges), n_comp, ""


def canonical_graph(machine):
    """Canonical rooted graph identity, preserving labels and missing edges."""
    # Exact rooted-graph identity by input-ordered BFS, independent of component IDs.
    # This does not minimize the graph.
    ids, queue = {machine.initial_state: 0}, deque([machine.initial_state])
    rows, finals = [], []
    while queue:
        q = queue.popleft()
        if q in machine.final_states:
            finals.append(ids[q])
        for x in machine.alphabet:
            edge = machine.transitions.get((q, x))
            if edge is None:
                continue
            target, symbol, motion = edge
            if target not in ids:
                ids[target] = len(ids)
                queue.append(target)
            rows.append((ids[q], x, ids[target], symbol, motion))
    if set(ids) != set(machine.states):
        raise ValueError("Unexpected unreachable state in sampled graph")
    return (tuple(machine.alphabet), len(ids), tuple(sorted(finals)), tuple(rows))


def extract_snapshot(builder, alphabet):
    """Export a fixed-epsilon partial FST after recursive successor closure."""
    edges, components, conflict = builder.compute_graph()
    if conflict:
        raise ValueError(
            f"Fixed epsilon {builder.eps} rejected: {conflict}. Lower epsilon and restart the sweep."
        )
    initial = builder._cluster_of_state[0]  # state index 0 is the first h0
    transitions = {}
    for src, dst, x, symbol, motion in edges:
        key = (int(src), int(x))
        edge = (-1 if dst == "halt" else int(dst), int(symbol), int(motion))
        if key in transitions and transitions[key] != edge:
            raise AssertionError("Successor closure did not produce a deterministic graph")
        transitions[key] = edge
    machine = Transducer(transitions, initial, tuple(alphabet), frozenset([-1]))
    if len(machine.states) != components:
        raise AssertionError("Component count disagrees with exported graph")
    return machine, edges


def select_valid_epsilon(trajectories, alphabet, epsilon):
    """Select epsilon on the full sample before measuring prefix stability."""
    import math

    if not trajectories or not math.isfinite(epsilon) or epsilon < 0:
        raise ValueError("Provide trajectories and a finite, nonnegative epsilon")
    attempts = []
    epsilon = float(epsilon)
    for attempt in range(66):
        builder = IncrementalGraphCC(eps=epsilon)
        builder.add(trajectories)
        try:
            machine, _ = extract_snapshot(builder, alphabet)
        except ValueError as error:
            if not str(error).startswith("Fixed epsilon "):
                raise
            attempts.append(dict(epsilon=epsilon, error=str(error)))
            if epsilon == 0:
                raise ValueError("Full-sample extraction conflicts even at epsilon zero") from error
            epsilon = epsilon / 2 if attempt < 64 else 0.0
        else:
            attempts.append(dict(epsilon=epsilon, status="accepted"))
            return epsilon, machine, attempts


def trajectory_stability(trajectories, alphabet, epsilon):
    """Yield one fixed-epsilon graph and diagnostic row per added trajectory.

    Edge changes count the symmetric difference of labelled edges using the
    builder's component representatives, as in the historical stability plots.
    The canonical identity additionally tests structure independently of IDs.
    """
    import hashlib
    import json

    builder = IncrementalGraphCC(eps=epsilon)
    previous_edges, previous_signature = frozenset(), None
    for count, trajectory in enumerate(trajectories, 1):
        builder.add([trajectory])
        machine, edges = extract_snapshot(builder, alphabet)
        signature = canonical_graph(machine)
        row = dict(
            samples=count,
            epsilon=epsilon,
            components=len(machine.states),
            edges=len(edges),
            edge_changes=len(edges.symmetric_difference(previous_edges)),
            structure_changed=None
            if previous_signature is None
            else signature != previous_signature,
            graph_sha256=hashlib.sha256(json.dumps(signature).encode()).hexdigest(),
        )
        yield row, machine
        previous_edges, previous_signature = edges, signature
