"""Hidden-state basis and geometric diagnostics."""

import torch


def _sample_basis(all_h, h0, epsilon, delta=None):
    if not 0 < epsilon <= 1:
        raise ValueError("epsilon must lie in (0, 1]")
    if delta is not None and delta < 0:
        raise ValueError("delta must be nonnegative")
    h0 = h0.detach().float()
    if h0.ndim != 1 or not torch.isfinite(h0).all() or h0.norm() <= 1e-8:
        raise ValueError("The pinned initial state must be a finite nonzero vector")
    selected = [h0]
    for h in all_h:
        if len(selected) == h0.numel():
            break
        h = h.detach().to(h0)
        norm = h.norm()
        if not torch.isfinite(h).all() or norm <= 1e-8:
            continue
        if delta is not None and not 1 - delta <= norm <= 1 + delta:
            continue
        normalized = torch.stack([v / v.norm() for v in selected])
        if (normalized @ (h / norm)).abs().max() < epsilon:
            candidate = torch.stack(selected + [h], dim=1)
            if torch.linalg.matrix_rank(candidate) == candidate.shape[1]:
                selected.append(h)
    return torch.stack(selected, dim=1)


def build_orthogonal_basis(all_h, h_0, epsilon=0.1):
    """Greedily select near-orthogonal observations, pinning h0 as column zero.

    This is an observed-state basis, not Gram-Schmidt. Zero vectors and linearly
    dependent candidates are excluded. At most hidden_size columns are returned.
    """
    return _sample_basis(all_h, h_0, epsilon)


def build_orthonormal_basis(all_h, h_0, epsilon=0.1, delta=0.1):
    """Observed-state basis with a unit-norm filter; pinned h0 bypasses the filter.

    Historical notebook name retained. Columns are selected, not normalized, so
    their norms lie within delta of one rather than being forced to exactly one.
    """
    return _sample_basis(all_h, h_0, epsilon, delta)


def project_tensors(model, basis):
    """Represent token transitions/output heads in a sampled hidden-state basis."""
    A, B, C, h0 = model.effective_tensors()
    basis = basis.to(A)
    inverse = torch.linalg.pinv(basis)
    return inverse @ A @ basis, B @ basis, C @ basis, inverse @ h0


def cluster_geometry(states, labels):
    """Per-label diameters and minimum inter-cluster distances; CPU scalar results."""
    groups = {q: states[labels == q] for q in sorted(set(labels.tolist()))}
    diameters = {q: float(torch.cdist(h, h).max()) for q, h in groups.items()}
    separation = {
        (a, b): float(torch.cdist(groups[a], groups[b]).min())
        for a in groups
        for b in groups
        if a < b
    }
    return {"diameters": diameters, "separation": separation}
