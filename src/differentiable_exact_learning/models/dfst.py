"""Differentiable finite-state transducers with a logarithmic-depth scan."""

import math

import torch
from torch import Tensor, nn

from .base import Controller, TokenEmbedding, initial_vector, register_initial_state


def matrix_prefix_scan(matrices: Tensor) -> Tensor:
    """Return A_t @ ... @ A_0 for every t, in O(log T) parallel stages.

    Input and output have shape (B,T,H,H). Multiplication order matters: each
    later block multiplies its earlier block on the right. Every stage reads
    the previous tensor, so autograd sees no in-place changes or detached state.
    The work is O(T log T); the computational depth is O(log T).
    """
    if matrices.ndim != 4 or matrices.shape[-1] != matrices.shape[-2]:
        raise ValueError("matrices must have shape (batch, time, state, state)")
    result = matrices
    stride = 1
    while stride < matrices.shape[1]:
        result = torch.cat((result[:, :stride], result[:, stride:] @ result[:, :-stride]), dim=1)
        stride *= 2
    return result


def _matrix(shape: tuple[int, int, int], scheme: str) -> Tensor:
    if scheme in {"zero", "zeros"}:
        return torch.zeros(shape)
    if scheme == "identity":
        return torch.eye(shape[1], shape[2]).expand(shape[0], -1, -1).clone()
    if scheme in {"random", "randn"}:
        return torch.randn(shape) / math.sqrt(shape[-1])
    raise ValueError(
        f"Unknown DFST matrix initialization {scheme!r}; use identity, zero, or random"
    )


class DFST(Controller):
    """h_t=A(x_t)h_{t-1}, y_t=B(x_t)h_{t-1}, z_t=C(x_t)h_{t-1}.

    A, B, C are indexed by the embedding coordinate. With the default fixed
    one-hot embedding this coordinate is exactly the input-token ID.
    """

    supports_state_rescaling = True

    def __init__(
        self,
        input_size: int,
        symbol_size: int,
        motion_size: int,
        hidden_size: int,
        *,
        embed_dim: int | None = None,
        scan: str = "parallel",
        transition_init: str = "identity",
        symbol_init: str = "zero",
        motion_init: str = "zero",
        state_init: str = "ones",
        learnable_state: bool = True,
    ):
        super().__init__()
        if scan not in {"parallel", "sequential"}:
            raise ValueError("scan must be 'parallel' or 'sequential'")
        self.scan = scan
        self.hidden_size = hidden_size
        self.symbol_size = symbol_size
        self.motion_size = motion_size
        self.embedding = TokenEmbedding(
            input_size, embed_dim, simplex=self.output_kind == "probabilities"
        )
        width = self.embedding.width
        self.A = nn.Parameter(_matrix((width, hidden_size, hidden_size), transition_init))
        self.B = nn.Parameter(_matrix((width, symbol_size, hidden_size), symbol_init))
        self.C = nn.Parameter(_matrix((width, motion_size, hidden_size), motion_init))
        initial = initial_vector((hidden_size,), state_init)
        if state_init in {"random", "randn"}:
            # Preserve the DFST paper implementation's variance 1 / hidden_size.
            initial = initial / math.sqrt(hidden_size)
        register_initial_state(self, initial, learnable_state)

    def effective_parameters(self) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Return transition/readout tensors in embedding coordinates and h0."""
        return self.A, self.B, self.C, self.h0

    def effective_tensors(self) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Materialize A, B, C per input token, including a learned embedding."""
        A, B, C, h0 = self.effective_parameters()
        ids = torch.arange(self.embedding.input_size, device=A.device)
        embedding = self.embedding(ids)
        return (
            torch.einsum("ve,ehk->vhk", embedding, A),
            torch.einsum("ve,esh->vsh", embedding, B),
            torch.einsum("ve,emh->vmh", embedding, C),
            h0,
        )

    def forward_chunk(self, tokens: Tensor, state: Tensor | None = None, *, stabilize=False):
        """Continue inference; optional DFST stabilization preserves argmax only.

        Stabilized inference uses matrix-vector recurrence and rescales the state
        at every timestep. This avoids both unbounded state magnitudes within a
        chunk and prefix-matrix overflow in directions the actual state never
        visits. It changes score magnitudes, so training should use ``forward``
        or the default unstabilized chunks. SoftmaxDFST never takes this path.
        """
        if not stabilize or not self.supports_state_rescaling:
            return super().forward_chunk(tokens, state)
        self._check_tokens(tokens)
        A, B, C, h0 = self.effective_tensors()

        def normalize(h: Tensor) -> Tensor:
            scale = h.abs().amax(dim=-1, keepdim=True)
            return h / torch.where(scale > 0, scale, torch.ones_like(scale))

        batch, length = tokens.shape
        h = normalize(h0.unsqueeze(0).expand(batch, -1)) if state is None else state
        if h.shape != (batch, self.hidden_size):
            raise ValueError(f"DFST state must have shape ({batch}, {self.hidden_size})")
        if length == 0:
            return (
                h.new_empty((batch, 0, self.symbol_size)),
                h.new_empty((batch, 0, self.motion_size)),
                h,
            )
        symbols, motions = [], []
        for t in range(length):
            token = tokens[:, t]
            symbols.append((B[token] @ h.unsqueeze(-1)).squeeze(-1))
            motions.append((C[token] @ h.unsqueeze(-1)).squeeze(-1))
            h = normalize((A[token] @ h.unsqueeze(-1)).squeeze(-1))
        return torch.stack(symbols, dim=1), torch.stack(motions, dim=1), h

    def forward(self, tokens: Tensor, state: Tensor | None = None, mask: Tensor | None = None):
        self._check_tokens(tokens)
        self._check_mask(tokens, mask)
        A, B, C, h0 = self.effective_parameters()
        batch, length = tokens.shape
        h = h0.unsqueeze(0).expand(batch, -1) if state is None else state
        if h.shape != (batch, self.hidden_size):
            raise ValueError(f"DFST state must have shape ({batch}, {self.hidden_size})")
        if length == 0:
            return (
                h.new_empty((batch, 0, self.symbol_size)),
                h.new_empty((batch, 0, self.motion_size)),
                h,
            )
        embedding = self.embedding(tokens)
        if mask is not None:
            embedding = embedding.masked_fill(~mask.unsqueeze(-1), 0)
        transitions = torch.einsum("bte,ehk->bthk", embedding, A)
        symbol_maps = torch.einsum("bte,esh->btsh", embedding, B)
        motion_maps = torch.einsum("bte,emh->btmh", embedding, C)
        if mask is not None:
            valid = mask[:, :, None, None]
            identity = torch.eye(self.hidden_size, device=A.device, dtype=A.dtype)
            # Mask before the scan: masking only the loss leaves 0*inf paths
            # in backward when a short row's padded recurrence overflows.
            transitions = torch.where(valid, transitions, identity)
            symbol_maps = symbol_maps.masked_fill(~valid, 0)
            motion_maps = motion_maps.masked_fill(~valid, 0)
        if self.scan == "parallel":
            prefixes = matrix_prefix_scan(transitions)
            states = (prefixes @ h[:, None, :, None]).squeeze(-1)
            previous = torch.cat((h[:, None], states[:, :-1]), dim=1)
            symbols = (symbol_maps @ previous.unsqueeze(-1)).squeeze(-1)
            motions = (motion_maps @ previous.unsqueeze(-1)).squeeze(-1)
            return symbols, motions, states[:, -1]
        symbols, motions = [], []
        for t in range(length):
            symbols.append((symbol_maps[:, t] @ h.unsqueeze(-1)).squeeze(-1))
            motions.append((motion_maps[:, t] @ h.unsqueeze(-1)).squeeze(-1))
            h = (transitions[:, t] @ h.unsqueeze(-1)).squeeze(-1)
        return torch.stack(symbols, dim=1), torch.stack(motions, dim=1), h


class SoftmaxDFST(DFST):
    """Column-stochastic transitions and readouts, with optional peaking loss."""

    output_kind = "probabilities"
    supports_state_rescaling = False

    def __init__(self, *args, reg_lambda: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        if reg_lambda < 0:
            raise ValueError("reg_lambda must be nonnegative")
        self.reg_lambda = float(reg_lambda)

    def effective_parameters(self) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        return self.A.softmax(-2), self.B.softmax(-2), self.C.softmax(-2), self.h0.softmax(-1)

    def reg_loss(self) -> Tensor:
        if self.reg_lambda == 0:
            return self.A.new_zeros(())
        A, B, C, h0 = self.effective_parameters()
        peaking = sum((1 - tensor.square().sum(-2)).mean() for tensor in (A, B, C))
        return self.reg_lambda * (peaking + 1 - h0.square().sum())
