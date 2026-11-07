"""The common dual-head controller interface and token representations."""

from collections.abc import Iterator

import torch
from torch import Tensor, nn


class TokenEmbedding(nn.Module):
    """Fixed one-hot tokens, or learned vectors (simplex mixtures if requested)."""

    def __init__(self, input_size: int, embed_dim: int | None, *, simplex: bool = False):
        super().__init__()
        self.input_size = input_size
        self.width = input_size if embed_dim is None else int(embed_dim)
        if self.width <= 0:
            raise ValueError("embed_dim must be positive or null")
        self.simplex = simplex and embed_dim is not None
        if embed_dim is None:
            self.table = nn.Embedding.from_pretrained(torch.eye(input_size), freeze=True)
        else:
            self.table = nn.Embedding(input_size, self.width)

    def forward(self, tokens: Tensor) -> Tensor:
        vectors = self.table(tokens)
        return vectors.softmax(dim=-1) if self.simplex else vectors


class Controller(nn.Module):
    """forward(tokens, state, mask) returns symbol scores, motion scores, and state.

    States are opaque to callers: DFST uses (B,H), recurrent stacks use
    (layers,B,H). Carry the entire returned state.
    An optional boolean mask identifies a nonempty contiguous prefix per row.
    Padded outputs are zero and returned states stop at each row's last real token.
    """

    output_kind = "logits"
    supports_state_rescaling = False

    def reg_loss(self) -> Tensor:
        return next(self.parameters()).new_zeros(())

    def regularization_loss(self) -> Tensor:
        """Uniform trainer hook; only SoftmaxDFST implements a nonzero reg_loss."""
        return self.reg_loss()

    def forward_chunk(self, tokens: Tensor, state=None, *, stabilize=False):
        """Evaluate the next token chunk and return scores plus continuation state.

        Call under no_grad() for inference. Stabilization only affects ordinary
        DFST; all other controllers retain their usual recurrence.
        """
        return self(tokens, state)

    def iter_outputs(
        self, tokens: Tensor, chunk_size: int = 256, stabilize: bool = False
    ) -> Iterator[tuple[Tensor, Tensor]]:
        """Stream contiguous chunks without changing the recurrent computation.

        This method does not disable gradients. Inference callers should use
        ``torch.no_grad()`` or ``torch.inference_mode()``. Optional stabilization
        rescales only homogeneous, unconstrained DFST states. It preserves their
        future argmax predictions, but not their raw scores or training loss.
        """
        self._check_tokens(tokens)
        if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer")
        state = None
        for start in range(0, tokens.shape[1], chunk_size):
            symbol, motion, state = self.forward_chunk(
                tokens[:, start : start + chunk_size], state, stabilize=stabilize
            )
            yield symbol, motion

    @staticmethod
    def _check_tokens(tokens: Tensor) -> None:
        if tokens.ndim != 2:
            raise ValueError("tokens must have shape (batch, time)")

    @staticmethod
    def _check_mask(tokens: Tensor, mask: Tensor | None) -> Tensor | None:
        """Validate right padding and return lengths for packed recurrence."""
        if mask is None:
            return None
        if mask.dtype != torch.bool or mask.shape != tokens.shape:
            raise ValueError("mask must be boolean with the same shape as tokens")
        if mask.device != tokens.device:
            raise ValueError("mask and tokens must be on the same device")
        if tokens.shape[1]:
            if not bool(mask[:, 0].all()):
                raise ValueError("Each mask row must contain at least one real token")
            if bool((mask[:, 1:] & ~mask[:, :-1]).any()):
                raise ValueError("Each mask row must be a contiguous prefix of real tokens")
        return mask.sum(dim=1)


def register_initial_state(module: nn.Module, value: Tensor, learnable: bool) -> None:
    if learnable:
        module.register_parameter("h0", nn.Parameter(value))
    else:
        module.register_buffer("h0", value)


def initial_vector(shape: tuple[int, ...], scheme: str) -> Tensor:
    """Build vector states; the final axis is the state dimension."""
    if scheme in {"zero", "zeros"}:
        return torch.zeros(shape)
    if scheme in {"one", "ones"}:
        return torch.ones(shape)
    if scheme in {"random", "randn"}:
        return torch.randn(shape)
    if scheme == "one_hot":
        result = torch.zeros(shape)
        result[..., 0] = 1
        return result
    raise ValueError(f"Unknown initial state {scheme!r}; use zeros, ones, randn, or one_hot")
