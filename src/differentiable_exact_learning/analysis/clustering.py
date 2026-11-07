"""Shared cluster-consistency closure for epsilon-chain, epsilon-net and k-means."""

from collections import defaultdict

import torch


class UnionFind:
    """Path-compressed disjoint sets with deterministic minimum-index roots."""

    def __init__(self, items=()):
        self.parent = {item: item for item in items}

    def find(self, item):
        self.parent.setdefault(item, item)
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left, right):
        a, b = self.find(left), self.find(right)
        if a != b:
            low, high = min(a, b), max(a, b)
            self.parent[high] = low
        return self.find(left)


def _observations(trajectories):
    hidden, transitions = [], []
    for trajectory in trajectories:
        hs, xs, sigs, halts = (trajectory[k] for k in ("h_seq", "x_seq", "sig_seq", "halt_seq"))
        if len(hs) != len(xs) + 1 or len(halts) != len(hs) or len(sigs) != len(xs):
            raise ValueError(
                "Invalid trajectory: expected L inputs/outputs and L+1 states/halt flags"
            )
        indices = []
        for h, halt in zip(hs, halts):
            indices.append(-1 if halt else len(hidden))
            if not halt:
                hidden.append(h.detach().cpu())
        for t, (x, (s, m)) in enumerate(zip(xs, sigs)):
            if indices[t] >= 0:
                transitions.append((indices[t], int(x), int(s), int(m), indices[t + 1]))
    if not hidden:
        raise ValueError("No non-halt hidden states in trajectories")
    H = torch.stack(hidden)
    if not torch.isfinite(H).all():
        raise ValueError("Non-finite hidden states")
    return H, transitions


def _consistent_graph(H, transitions, labels, stop, vocab):
    """Close successor equivalences, rejecting any output/halt conflict."""
    from .merging import MergeGraph

    sets = UnionFind(set(labels))
    while True:
        signatures, targets = {}, defaultdict(set)
        for source, x, s, m, target in transitions:
            key = (sets.find(labels[source]), x)
            if key in signatures and signatures[key] != (s, m):
                return None
            signatures[key] = (s, m)
            targets[key].add(-1 if target < 0 else sets.find(labels[target]))
        changed = False
        for successors in targets.values():
            if -1 in successors and len(successors) > 1:
                return None
            real = sorted(q for q in successors if q >= 0)
            for q in real[1:]:
                if sets.find(real[0]) != sets.find(q):
                    sets.union(real[0], q)
                    changed = True
        if not changed:
            break
    members = defaultdict(list)
    for i, label in enumerate(labels):
        members[sets.find(label)].append(i)
    initial = sets.find(labels[0])
    order = [initial] + sorted(q for q in members if q != initial)
    graph = MergeGraph(vocab, stop)
    node = {}
    for q in order:
        node[q] = graph._alloc(H[members[q]].mean(0))
        graph.nodes[node[q]]["h_count"] = len(members[q])
    graph.initial_id = node[initial]
    for source, x, s, m, target in transitions:
        src = node[sets.find(labels[source])]
        dst = graph.halt_id if target < 0 else node[sets.find(labels[target])]
        graph.edges[src, x] = {"target": dst, "sym": s, "mot": m}
    return graph


def _epsilon_labels(H, epsilon, chunk_size):
    sets = UnionFind(range(len(H)))
    for start in range(0, len(H), chunk_size):
        distance = torch.cdist(H[start : start + chunk_size].float(), H.float())
        rows, columns = (distance <= epsilon).nonzero(as_tuple=True)
        for row, column in zip(rows.tolist(), columns.tolist()):
            sets.union(start + row, column)
    return [sets.find(i) for i in range(len(H))]


def cluster_trajectories_epschain(
    trajectories, S_idx, d_vocab, eps, min_eps=1e-9, chunk_size=512, verbose=True
):
    """Epsilon-connected components, followed by shared successor consistency closure."""
    if min_eps <= 0 or chunk_size < 1 or eps < 0:
        raise ValueError("Require nonnegative eps and positive min_eps/chunk_size")
    H, transitions = _observations(trajectories)
    while eps >= min_eps:
        graph = _consistent_graph(
            H, transitions, _epsilon_labels(H, eps, chunk_size), S_idx, d_vocab
        )
        if graph is not None:
            if verbose:
                print(f"Epsilon-chain: {graph.n_nodes()} states at epsilon={eps:g}")
            return True, graph
        eps /= 2
    return False, "No output-consistent epsilon-chain clustering at the allowed tolerances"


def cluster_trajectories_epsnet(trajectories, S_idx, d_vocab, delta, min_delta=1e-9, verbose=True):
    """Greedy running-centroid epsilon net plus the same consistency closure."""
    if min_delta <= 0 or delta < 0:
        raise ValueError("Require nonnegative delta and positive min_delta")
    H, transitions = _observations(trajectories)
    while delta >= min_delta:
        centroids, counts, labels = [], [], []
        for h in H:
            distances = torch.stack([(h - c).norm() for c in centroids]) if centroids else None
            if distances is None or float(distances.min()) > delta:
                labels.append(len(centroids))
                centroids.append(h.clone())
                counts.append(1)
            else:
                label = int(distances.argmin())
                labels.append(label)
                centroids[label] = (centroids[label] * counts[label] + h) / (counts[label] + 1)
                counts[label] += 1
        graph = _consistent_graph(H, transitions, labels, S_idx, d_vocab)
        if graph is not None:
            if verbose:
                print(f"Epsilon-net: {graph.n_nodes()} states at delta={delta:g}")
            return True, graph
        delta /= 10
    return False, "No output-consistent epsilon-net clustering at the allowed tolerances"


def cluster_trajectories_kmeans(trajectories, S_idx, d_vocab, k_start, k_step, verbose=True):
    """Deterministic k-means; increase k on output/target conflicts."""
    from sklearn.cluster import KMeans

    if k_start < 1 or k_step < 1:
        raise ValueError("k_start and k_step must be positive")
    H, transitions = _observations(trajectories)
    for k in range(k_start, len(H) + 1, k_step):
        labels = (
            KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(H.float().numpy()).tolist()
        )
        graph = _consistent_graph(H, transitions, labels, S_idx, d_vocab)
        if graph is not None:
            if verbose:
                print(f"K-means: {graph.n_nodes()} states at k={k}")
            return True, graph
    return False, "No output-consistent k-means clustering at the tested cluster counts"
