"""Geometric FST extraction algorithms retained from the original research notebooks.

Phase A clusters sampled hidden states. Phase B merges states only when recursively
forced successor merges do not produce conflicting output labels. This preserves
empirical consistency, not a proof of unobserved input behavior. Optional k-means
requires scikit-learn. Completing unknown transitions to halt is an explicit
heuristic; prefer the partial graph for coverage-aware evaluation.
"""

import os
import random

import torch
from tqdm.auto import tqdm

from .clustering import (
    cluster_trajectories_epschain,
    cluster_trajectories_epsnet,
    cluster_trajectories_kmeans,
)


class MergeGraph:
    """
    Directed graph of DFST hidden states, supporting greedy merge operations.

    Nodes
    -----
    Each active node stores:
        h        : (d_state,) running-mean hidden-state vector (None for halt node)
        h_count  : number of original vectors merged into h
        is_halt  : True only for the special halt node

    Edges
    -----
    Partial map: (src_id, x) -> {'target': id, 'sym': int, 'mot': int}
    The halt node has full self-loop coverage (all x) set at construction.
    """

    def __init__(self, d_vocab, stop_mot_idx):
        self.d_vocab = d_vocab
        self.stop_mot_idx = stop_mot_idx

        self._next_id = 0
        self.nodes = {}  # id -> {'h', 'h_count', 'is_halt'}
        self.edges = {}  # (src_id, x) -> {'target', 'sym', 'mot'}
        self.active = set()
        self.merge_into = {}  # deactivated_id -> active_id it was merged into

        self.halt_id = None
        self.initial_id = None

        # Create halt node with absorbing self-loops
        self.halt_id = self._alloc(h=None, is_halt=True)
        for x in range(d_vocab):
            self.edges[(self.halt_id, x)] = {"target": self.halt_id, "sym": x, "mot": stop_mot_idx}

    # ── Internal helpers ────────────────────────────────────────────────

    def _alloc(self, h, is_halt=False):
        nid = self._next_id
        self._next_id += 1
        self.nodes[nid] = {
            "h": h.clone() if h is not None else None,
            "h_count": 1 if h is not None else 0,
            "is_halt": is_halt,
        }
        self.active.add(nid)
        return nid

    def _find_active(self, nid):
        """Follow merge_into chain to reach the current active representative."""
        while nid in self.merge_into:
            nid = self.merge_into[nid]
        return nid

    def _non_halt(self):
        return [nid for nid in self.active if not self.nodes[nid]["is_halt"]]

    def n_nodes(self):
        return len(self.active)

    # ── Building the graph ───────────────────────────────────────────────

    def add_trajectory(self, h_seq, x_seq, sig_seq, halt_seq, tol=1e-9):
        """
        Add one trajectory to the graph.

        h_seq   : list[Tensor]       (L+1)  hidden states
        x_seq   : list[int]          (L)    input tokens
        sig_seq : list[(int,int)]    (L)    (sym, mot) output signatures
        halt_seq: list[bool]         (L+1)  True for halt states
        tol     : deduplication tolerance (merge states within this L2 distance)
        """
        nids = []
        for t, (h, is_halt) in enumerate(zip(h_seq, halt_seq)):
            if is_halt:
                nids.append(self.halt_id)
                continue

            # h_0 always maps to the fixed initial node
            if t == 0 and self.initial_id is not None:
                nids.append(self.initial_id)
                continue

            # Find existing node within dedup tolerance
            found = None
            for nid in self.active:
                if self.nodes[nid]["is_halt"]:
                    continue
                if (self.nodes[nid]["h"] - h).norm().item() < tol:
                    found = nid
                    break

            if found is not None:
                # Update running mean
                n = self.nodes[found]["h_count"]
                self.nodes[found]["h"] = (self.nodes[found]["h"] * n + h) / (n + 1)
                self.nodes[found]["h_count"] += 1
                nids.append(found)
            else:
                nids.append(self._alloc(h))

        if self.initial_id is None:
            self.initial_id = nids[0]

        # Add edges, rejecting conflicting observed outputs instead of hiding them.
        for t, (x, (sym, mot)) in enumerate(zip(x_seq, sig_seq)):
            src = nids[t]
            if self.nodes[src]["is_halt"]:
                continue
            previous = self.edges.get((src, x))
            if previous is not None and (previous["sym"], previous["mot"]) != (sym, mot):
                raise ValueError(
                    "Deduplication merged hidden states with conflicting output signatures"
                )
            self.edges.setdefault((src, x), {"target": nids[t + 1], "sym": sym, "mot": mot})

    # ── Merge operation ──────────────────────────────────────────────────

    def _do_merge(self, a, b):
        """Merge active non-halt nodes a and b. Returns (new_id, cascade_pairs)."""
        na, nb = self.nodes[a], self.nodes[b]
        ca, cb = na["h_count"], nb["h_count"]
        c = self._alloc((na["h"] * ca + nb["h"] * cb) / (ca + cb))
        self.nodes[c]["h_count"] = ca + cb

        cascade = []
        for x in range(self.d_vocab):
            ea = self.edges.get((a, x))
            eb = self.edges.get((b, x))

            # Remap self-references: targets pointing to a or b become c
            def remap(t):
                return c if t in (a, b) else t

            if ea and eb:
                ta, tb = remap(ea["target"]), remap(eb["target"])
                self.edges[(c, x)] = {"target": ta, "sym": ea["sym"], "mot": ea["mot"]}
                if ta != tb:
                    cascade.append((ta, tb))
            elif ea:
                self.edges[(c, x)] = {
                    "target": remap(ea["target"]),
                    "sym": ea["sym"],
                    "mot": ea["mot"],
                }
            elif eb:
                self.edges[(c, x)] = {
                    "target": remap(eb["target"]),
                    "sym": eb["sym"],
                    "mot": eb["mot"],
                }

        # Re-point all incoming edges targeting a or b → c
        for key, e in list(self.edges.items()):
            if e["target"] in (a, b):
                self.edges[key] = dict(e)
                self.edges[key]["target"] = c

        # Remove outgoing edges of a and b
        for x in range(self.d_vocab):
            self.edges.pop((a, x), None)
            self.edges.pop((b, x), None)

        # Deactivate a and b
        self.active -= {a, b}
        self.merge_into[a] = c
        self.merge_into[b] = c
        if self.initial_id in (a, b):
            self.initial_id = c

        return c, cascade

    def merge(self, a, b):
        """Merge a and b with full cascade. Returns number of nodes removed."""
        n0 = len(self.active)
        q = [(a, b)]
        while q:
            u, v = q.pop()
            u, v = self._find_active(u), self._find_active(v)
            if (
                u == v
                or u not in self.active
                or v not in self.active
                or self.nodes[u]["is_halt"]
                or self.nodes[v]["is_halt"]
            ):
                continue
            _, cascade = self._do_merge(u, v)
            q.extend(cascade)
        return n0 - len(self.active)

    # ── Distinguishability ───────────────────────────────────────────────

    def compute_dist_table(self):
        """
        Myhill-Nerode distinguishability for all pairs of active nodes.
        Missing edges are treated as 'don't care' (partial IST semantics).

        Returns dict (min_id, max_id) -> bool.
        """
        nodes = list(self.active)
        idx = {nid: i for i, nid in enumerate(nodes)}
        n = len(nodes)
        dist = [[False] * n for _ in range(n)]

        # Halt node is always distinguished from all non-halt nodes
        hi = idx[self.halt_id]
        for j, nj in enumerate(nodes):
            if nj != self.halt_id:
                dist[hi][j] = dist[j][hi] = True

        # Iterative Myhill-Nerode marking
        changed = True
        while changed:
            changed = False
            for i, ni in enumerate(nodes):
                if self.nodes[ni]["is_halt"]:
                    continue
                for j, nj in enumerate(nodes):
                    if j <= i or dist[i][j] or self.nodes[nj]["is_halt"]:
                        continue
                    for x in range(self.d_vocab):
                        ei = self.edges.get((ni, x))
                        ej = self.edges.get((nj, x))
                        if ei is None or ej is None:
                            continue  # don't-care
                        if ei["sym"] != ej["sym"] or ei["mot"] != ej["mot"]:
                            dist[i][j] = dist[j][i] = True
                            changed = True
                            break
                        ti, tj = ei["target"], ej["target"]
                        if ti != tj and dist[idx[ti]][idx[tj]]:
                            dist[i][j] = dist[j][i] = True
                            changed = True
                            break

        return {
            (min(nodes[i], nodes[j]), max(nodes[i], nodes[j])): dist[i][j]
            for i in range(n)
            for j in range(i + 1, n)
        }

    # ── Score computation ────────────────────────────────────────────────

    def simulate_merge_cascade(self, a, b, dist_table=None):
        """
        Simulate merging a and b (no graph modification).
        Returns number of nodes that would be removed.
        Returns 0 if a and b are Myhill-Nerode distinguishable.

        dist_table: precomputed result of compute_dist_table(); computed
                    internally if not provided (expensive for many calls).
        """
        if dist_table is None:
            dist_table = self.compute_dist_table()
        key = (min(a, b), max(a, b))
        if dist_table.get(key, False):
            return 0

        # Build per-node outgoing-target cache
        out = {}
        for nid in self.active:
            out[nid] = {
                x: self.edges[(nid, x)]["target"]
                for x in range(self.d_vocab)
                if (nid, x) in self.edges
            }

        parent = {nid: nid for nid in self.active}
        # group_edges[rep] = combined outgoing targets for the group
        group_edges = {nid: dict(out[nid]) for nid in self.active}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        q = [(a, b)]
        while q:
            u, v = q.pop()
            ru, rv = find(u), find(v)
            if ru == rv:
                continue
            # Detect cascade BEFORE merging
            ru_e = group_edges.get(ru, {})
            rv_e = group_edges.get(rv, {})
            for x in range(self.d_vocab):
                tu = ru_e.get(x)
                tv = rv_e.get(x)
                if tu is not None and tv is not None:
                    rtu, rtv = find(tu), find(tv)
                    if rtu != rtv:
                        # Never cascade-merge halt with non-halt
                        if not (self.nodes[rtu]["is_halt"] or self.nodes[rtv]["is_halt"]):
                            if dist_table is not None:
                                cascade_key = (min(rtu, rtv), max(rtu, rtv))
                                if dist_table.get(cascade_key, False):
                                    return 0  # cascade would force merging distinguished nodes
                            q.append((rtu, rtv))
            # Union: rv absorbs ru
            parent[ru] = rv
            for x, t in ru_e.items():
                group_edges[rv].setdefault(x, t)
            del group_edges[ru]

        return len(self.active) - len(set(find(nid) for nid in self.active))

    # ── Export ───────────────────────────────────────────────────────────

    def complete_and_export(self):
        """Return a complete transition map; missing edges → halt node."""
        fmap = {}
        for nid in self.active:
            for x in range(self.d_vocab):
                fmap[(nid, x)] = self.edges.get((nid, x)) or {
                    "target": self.halt_id,
                    "sym": x,
                    "mot": self.stop_mot_idx,
                }
        return fmap

    def complete(self):
        """
        Return a copy of this graph with every missing (non-halt node, x) edge
        filled in as a halt transition: target=halt_id, sym=x, mot=stop_mot_idx.

        The original graph is not modified.
        """
        g = self.copy()
        for nid in g.active:
            if g.nodes[nid]["is_halt"]:
                continue
            for x in range(g.d_vocab):
                if (nid, x) not in g.edges:
                    g.edges[(nid, x)] = {"target": g.halt_id, "sym": x, "mot": g.stop_mot_idx}
        return g

    def copy(self):
        """Return a deep copy of this MergeGraph."""
        import copy as _copy

        return _copy.deepcopy(self)

    # ── Persistence ─────────────────────────────────────────────────────

    def save(self, path):
        """
        Serialise the entire MergeGraph to a single .pt file.

        All node tensors, edge dicts, active set, merge_into chain,
        and scalar metadata are stored so the graph can be
        reconstructed exactly with MergeGraph.load(path).
        """
        state = {
            "d_vocab": self.d_vocab,
            "stop_mot_idx": self.stop_mot_idx,
            "_next_id": self._next_id,
            "halt_id": self.halt_id,
            "initial_id": self.initial_id,
            "active": list(self.active),
            "merge_into": dict(self.merge_into),
            "nodes": {
                nid: {"h": info["h"], "h_count": info["h_count"], "is_halt": info["is_halt"]}
                for nid, info in self.nodes.items()
            },
            "edges": dict(self.edges),
        }
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(state, path)
        print(f"MergeGraph saved → {path}  ({self.n_nodes()} nodes, {len(self.edges)} edges)")

    @classmethod
    def load(cls, path):
        """Reconstruct a MergeGraph previously saved with save()."""
        state = torch.load(path, map_location="cpu", weights_only=False)
        g = cls.__new__(cls)
        g.d_vocab = state["d_vocab"]
        g.stop_mot_idx = state["stop_mot_idx"]
        g._next_id = state["_next_id"]
        g.halt_id = state["halt_id"]
        g.initial_id = state["initial_id"]
        g.active = set(state["active"])
        g.merge_into = dict(state["merge_into"])
        g.nodes = state["nodes"]
        g.edges = state["edges"]
        print(f"MergeGraph loaded ← {path}  ({g.n_nodes()} nodes, {len(g.edges)} edges)")
        return g

    def summary(self, agent=None):
        nh = self._non_halt()
        n_edges = sum(
            1 for (nid, _) in self.edges if nid in self.active and not self.nodes[nid]["is_halt"]
        )
        print(
            f"MergeGraph: {len(self.active)} nodes "
            f"({len(nh)} non-halt + 1 halt),  "
            f"{n_edges} defined non-halt edges out of {len(nh) * self.d_vocab} possible"
        )
        print(f"  initial_id={self.initial_id},  halt_id={self.halt_id}")


def build_model_graph(
    trajectories, A, C_sym, C_mot, h_0, S_idx, d_vocab, tol, dedup_tol=1e-9, verbose=True
):
    """
    Build a partial MergeGraph from DFST model parameters.

    1. Deduplicate all non-halt hidden states from trajectories (within dedup_tol),
       with h_0 always as the first (initial) node.
    2. For every deduplicated state h and every vocabulary symbol x:
       - Compute the predicted next state  next_h = A[x] @ h.
       - Find h' = nearest neighbour of next_h among the deduplicated states.
       - If ||h' - next_h|| <= tol : draw edge  h -> h'  with label
           (argmax C_sym[x] @ h,  argmax C_mot[x] @ h).
       - If ||h' - next_h|| >  tol : leave (h, x) undefined (no edge drawn).

    The returned graph is *partial*: out-of-tolerance (h, x) pairs have no
    outgoing edge and are NOT defaulted to the halt node.

    Parameters
    ----------
    trajectories : list of trajectory dicts (h_seq, x_seq, sig_seq, halt_seq)
    A            : (d_vocab, d_state, d_state) transition matrices
    C_sym        : (d_vocab, d_sym,  d_state) symbol output classifier
    C_mot        : (d_vocab, d_mot,  d_state) motion output classifier
    h_0          : (d_state,) initial hidden state
    S_idx        : int   stop-motion index
    d_vocab      : int   vocabulary size
    tol          : float L2 distance threshold for nearest-neighbour lookup
    dedup_tol    : float tolerance for deduplicating hidden states
    verbose      : bool

    Returns
    -------
    MergeGraph  (partial — missing edges for out-of-tolerance transitions)
    """
    import time

    def _t(t0):
        return f"{time.time() - t0:.2f}s"

    # ── Step 1: Collect and deduplicate hidden states ─────────────────────
    t0 = time.time()
    all_h_raw = [h_0.float()]
    for traj in trajectories:
        for h, is_halt in zip(traj["h_seq"], traj["halt_seq"]):
            if not is_halt:
                all_h_raw.append(h.float())

    # Greedy deduplication: keep h_0 first; add h only if farther than
    # dedup_tol from every already-kept state (vectorized inner step).
    unique_list = [all_h_raw[0]]
    U = all_h_raw[0].unsqueeze(0)  # (1, d_state)
    for h in tqdm(all_h_raw[1:], desc="Deduplicating states", leave=False, disable=not verbose):
        if (U - h).norm(dim=1).min().item() >= dedup_tol:
            unique_list.append(h)
            U = torch.cat([U, h.unsqueeze(0)], dim=0)

    H = torch.stack(unique_list)  # (n_states, d_state)
    n_states = H.shape[0]
    if verbose:
        print(f"build_model_graph | dedup: {n_states} unique states  [{_t(t0)}]")

    # ── Step 2: Build partial transition map ──────────────────────────────
    t1 = time.time()
    new_g = MergeGraph(d_vocab=d_vocab, stop_mot_idx=S_idx)

    # Allocate one node per unique state; node 0 (after halt) = initial
    node_ids = []
    for h in H:
        nid = new_g._alloc(h.to(H.dtype))
        node_ids.append(nid)
    new_g.initial_id = node_ids[0]

    n_within = 0
    for x in tqdm(range(d_vocab), desc="Building transitions", leave=False, disable=not verbose):
        # Predicted next states: A[x] @ h  for each h in H  (vectorized)
        next_H = H @ A[x].T  # (n_states, d_state)
        D = torch.cdist(next_H, H)  # (n_states, n_states)
        nn_idx = D.argmin(dim=1)  # (n_states,)
        nn_dist = D[torch.arange(n_states), nn_idx]  # (n_states,)

        # Output signatures from current state h  (vectorized)
        sym_out = (H @ C_sym[x].T).argmax(dim=1).tolist()  # list[int]
        mot_out = (H @ C_mot[x].T).argmax(dim=1).tolist()  # list[int]

        within = (nn_dist <= tol).tolist()
        for i in range(n_states):
            if within[i]:
                new_g.edges[(node_ids[i], x)] = {
                    "target": node_ids[int(nn_idx[i])],
                    "sym": sym_out[i],
                    "mot": mot_out[i],
                }
                n_within += 1

    n_total = n_states * d_vocab
    if verbose:
        print(
            f"build_model_graph | {n_within}/{n_total} transitions drawn "
            f"({100 * n_within / n_total:.1f}% within tol={tol})  [{_t(t1)}]"
        )

    return new_g


def greedy_metric_merging(
    graph, epsilon, verbose=True, phase1_extra_greedy=False, run_phase1=True, run_phase2=True
):
    """
    Three-phase greedy FST extraction from a MergeGraph.

    The input graph is never mutated; a deep copy is made internally.

    Parameters
    ----------
    graph               : MergeGraph
    epsilon             : L2 distance threshold for Phase 1
    verbose             : print merge events
    phase1_extra_greedy : if True, Phase 1 merges every epsilon-close,
                          non-distinguished pair immediately without
                          computing merge scores (faster, less selective).
                          If False (default), always picks the pair with
                          the highest cascade score.
    run_phase1          : if False, skip Phase 1 entirely (default True).
    run_phase2          : if False, skip Phase 2 entirely (default True).

    Returns
    -------
    full_map     : complete transition map  (graph node IDs as state IDs)
    metric_ptmap : partial transition map after Phase 1
    metric_graph : MergeGraph snapshot after Phase 1 (use this with metric_ptmap)
    graph        : internal MergeGraph after all phases (copy of input)
    """
    graph = graph.copy()  # never mutate the caller's graph

    def _score_candidates(candidates, dt, desc):
        """Score all candidates; return (best_score, best_pair)."""
        best_score, best_pair = 0, None
        for a, b in tqdm(candidates, desc=desc, leave=False, disable=not verbose):
            s = graph.simulate_merge_cascade(a, b, dt)
            if s > best_score:
                best_score, best_pair = s, (a, b)
        return best_score, best_pair

    # ── Phase 1: distance-bounded merging ───────────────────────────────
    if run_phase1:
        mode_label = "extra-greedy" if phase1_extra_greedy else "score-based"
        phase1_merges = 0
        pbar1 = tqdm(desc=f"Phase 1 ε={epsilon} [{mode_label}]", unit="merge", disable=not verbose)
        while True:
            nh = graph._non_halt()
            if len(nh) < 2:
                break

            if phase1_extra_greedy:
                # Compute dist table for correctness, then pick the first epsilon-close
                # non-distinguished pair for which simulate_merge_cascade confirms the
                # full cascade (including transitive) raises no distinguishability
                # violations.  "Extra-greedy" = first valid pair (random order), not
                # highest-scoring.
                dt = graph.compute_dist_table()
                candidates = [
                    (a, b)
                    for i, a in enumerate(nh)
                    for j, b in enumerate(nh)
                    if j > i
                    and (graph.nodes[a]["h"] - graph.nodes[b]["h"]).norm().item() <= epsilon
                    and not dt.get((min(a, b), max(a, b)), False)
                ]
                random.shuffle(candidates)
                pbar1.set_postfix({"nodes": graph.n_nodes(), "candidates": len(candidates)})

                merged = False
                for a, b in candidates:
                    if a not in graph.active or b not in graph.active:
                        continue
                    if graph.simulate_merge_cascade(a, b, dt) > 0:
                        removed = graph.merge(a, b)
                        phase1_merges += 1
                        pbar1.update(1)
                        pbar1.set_postfix({"nodes": graph.n_nodes(), "last_removed": removed})
                        if verbose:
                            tqdm.write(
                                f"  [P1 merge {phase1_merges}] ({a},{b}) "
                                f"removed={removed} "
                                f"→ {graph.n_nodes()} nodes"
                            )
                        merged = True
                        break
                if not merged:
                    break  # no valid pair within epsilon

            else:
                # Pick the pair with the highest cascade score among non-distinguished
                # epsilon-close pairs.
                dt = graph.compute_dist_table()
                candidates = [
                    (a, b)
                    for i, a in enumerate(nh)
                    for j, b in enumerate(nh)
                    if j > i
                    and (graph.nodes[a]["h"] - graph.nodes[b]["h"]).norm().item() <= epsilon
                    and not dt.get((min(a, b), max(a, b)), False)
                ]
                pbar1.set_postfix({"nodes": graph.n_nodes(), "candidates": len(candidates)})
                if not candidates:
                    break

                best_score, best_pair = _score_candidates(
                    candidates, dt, desc=f"P1 scoring ({len(candidates)} pairs)"
                )
                if best_pair is None:
                    break

                a, b = best_pair
                removed = graph.merge(a, b)
                phase1_merges += 1
                pbar1.update(1)
                pbar1.set_postfix(
                    {"nodes": graph.n_nodes(), "last_removed": removed, "score": best_score}
                )
                if verbose:
                    tqdm.write(
                        f"  [P1 merge {phase1_merges}] ({a},{b}) "
                        f"score={best_score} removed={removed} "
                        f"→ {graph.n_nodes()} nodes"
                    )

        pbar1.close()
        if verbose:
            print(f"Phase 1 done: {phase1_merges} merges, {graph.n_nodes()} nodes")

    # Snapshot of the graph and partial transition map after Phase 1.
    # metric_graph must be kept alongside metric_ptmap because Phase 2 will
    # create new node IDs — testing metric_ptmap against the Phase-3 graph
    # would fail since its node IDs are from Phase 1.
    metric_graph = graph.copy()
    metric_ptmap = {key: dict(e) for key, e in graph.edges.items() if key[0] in graph.active}

    # ── Phase 2: score-only merging (any distance) ───────────────────────
    if run_phase2:
        phase2_merges = 0
        pbar2 = tqdm(desc="Phase 2 (any dist)", unit="merge", disable=not verbose)
        while True:
            nh = graph._non_halt()
            if len(nh) < 2:
                break
            dt = graph.compute_dist_table()

            candidates = [
                (a, b)
                for i, a in enumerate(nh)
                for j, b in enumerate(nh)
                if j > i and not dt.get((min(a, b), max(a, b)), False)
            ]
            pbar2.set_postfix({"nodes": graph.n_nodes(), "candidates": len(candidates)})

            best_score, best_pair = _score_candidates(
                candidates, dt, desc=f"P2 scoring ({len(candidates)} pairs)"
            )
            if best_score == 0:
                break

            a, b = best_pair
            removed = graph.merge(a, b)
            phase2_merges += 1
            pbar2.update(1)
            pbar2.set_postfix(
                {"nodes": graph.n_nodes(), "last_removed": removed, "score": best_score}
            )
            if verbose:
                tqdm.write(
                    f"  [P2 merge {phase2_merges}] ({a},{b}) "
                    f"score={best_score} removed={removed} "
                    f"→ {graph.n_nodes()} nodes"
                )
        pbar2.close()
        if verbose:
            print(f"Phase 2 done: {phase2_merges} merges, {graph.n_nodes()} nodes")

    # ── Phase 3: complete missing transitions ────────────────────────────
    full_map = graph.complete_and_export()
    n_missing = sum(
        1 for nid in graph.active for x in range(graph.d_vocab) if (nid, x) not in graph.edges
    )
    if verbose:
        print(f"Phase 3: filled {n_missing} missing transitions → halt")
        print(f"Final graph: {graph.n_nodes()} nodes")

    return full_map, metric_ptmap, metric_graph, graph


__all__ = [
    "MergeGraph",
    "cluster_trajectories_epschain",
    "cluster_trajectories_epsnet",
    "cluster_trajectories_kmeans",
    "build_model_graph",
    "greedy_metric_merging",
]
