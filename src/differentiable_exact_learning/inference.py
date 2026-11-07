"""Evaluation scan selection without changing the model's training recurrence."""

import torch

from .models.dfst import DFST, matrix_prefix_scan


@torch.no_grad()
def forward_chunk(model, tokens, state=None, *, scan="sequential", stabilize=False):
    """Evaluate a chunk and return symbol scores, motion scores, and continuation state.

    DFST variants can use either scan independently of ``model.scan``. Sequential
    inference avoids per-timestep matrix tensors. Only ordinary DFST rescales
    states when stabilization is requested; SoftmaxDFST retains probabilities.
    Parallel inference uses raw matrix products and may overflow on long chunks.
    Other controllers retain their native chunk implementation.

    This dispatch is separate from the model methods so lazy evaluation imports
    also work with models already loaded by an existing training process.
    """
    if scan not in ("sequential", "parallel"):
        raise ValueError("evaluation.scan must be sequential or parallel")
    if not isinstance(model, DFST):
        return model.forward_chunk(tokens, state, stabilize=stabilize)

    model._check_tokens(tokens)
    A, B, C, h0 = model.effective_tensors()
    batch, length = tokens.shape
    h = h0.unsqueeze(0).expand(batch, -1) if state is None else state
    if h.shape != (batch, model.hidden_size):
        raise ValueError(f"DFST state must have shape ({batch}, {model.hidden_size})")
    if length == 0:
        return (
            h.new_empty((batch, 0, model.symbol_size)),
            h.new_empty((batch, 0, model.motion_size)),
            h,
        )

    if scan == "parallel":
        prefixes = matrix_prefix_scan(A[tokens])
        states = (prefixes @ h[:, None, :, None]).squeeze(-1)
        previous = torch.cat((h[:, None], states[:, :-1]), dim=1)
        symbols = (B[tokens] @ previous.unsqueeze(-1)).squeeze(-1)
        motions = (C[tokens] @ previous.unsqueeze(-1)).squeeze(-1)
        return symbols, motions, states[:, -1]

    rescale = stabilize and model.supports_state_rescaling

    def normalize(value):
        if not rescale:
            return value
        scale = value.abs().amax(dim=-1, keepdim=True)
        return value / torch.where(scale > 0, scale, torch.ones_like(scale))

    h = normalize(h)
    symbols, motions = [], []
    for t in range(length):
        token = tokens[:, t]
        symbols.append((B[token] @ h.unsqueeze(-1)).squeeze(-1))
        motions.append((C[token] @ h.unsqueeze(-1)).squeeze(-1))
        h = normalize((A[token] @ h.unsqueeze(-1)).squeeze(-1))
    return torch.stack(symbols, dim=1), torch.stack(motions, dim=1), h
