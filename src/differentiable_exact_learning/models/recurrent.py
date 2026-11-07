"""RNN and GRU controllers using the same dual-head contract."""

import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .base import Controller, TokenEmbedding, initial_vector, register_initial_state


def _init_heads(heads: tuple[nn.Linear, ...], scheme: str) -> None:
    if scheme not in {"default", "zero"}:
        raise ValueError("head initialization must be 'default' or 'zero'")
    if scheme == "zero":
        for head in heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)


class RecurrentController(Controller):
    """A recurrent stack. All layers' states survive streaming boundaries.

    ``recurrent_init='default'`` preserves PyTorch's weights and biases.
    Orthogonal/identity initialization uses Xavier input weights and zero biases.
    ``identity_zero_input`` is identity recurrence with zero input weights and
    zero biases. An explicit GRU update-gate bias is applied after any of these.
    """

    def __init__(
        self,
        name: str,
        input_size: int,
        symbol_size: int,
        motion_size: int,
        hidden_size: int = 64,
        *,
        num_layers: int = 1,
        embed_dim: int | None = None,
        recurrent_init: str = "orthogonal",
        head_init: str = "default",
        state_init: str = "zeros",
        learnable_state: bool = False,
        gate_bias: float = 0.0,
    ):
        super().__init__()
        if name not in {"RNN", "GRU"}:
            raise ValueError("RecurrentController name must be RNN or GRU")
        if num_layers < 1:
            raise ValueError("num_layers must be positive")
        self.name = name
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.symbol_size, self.motion_size = symbol_size, motion_size
        self.embedding = TokenEmbedding(input_size, embed_dim)
        core = nn.RNN if name == "RNN" else nn.GRU
        self.core = core(self.embedding.width, hidden_size, num_layers, batch_first=True)
        self.B = nn.Linear(hidden_size, symbol_size)
        self.C = nn.Linear(hidden_size, motion_size)
        self._initialize_recurrence(recurrent_init)
        if name == "GRU" and gate_bias != 0:
            with torch.no_grad():
                for layer in range(num_layers):
                    getattr(self.core, f"bias_ih_l{layer}")[hidden_size : 2 * hidden_size].fill_(
                        gate_bias
                    )
        _init_heads((self.B, self.C), head_init)
        register_initial_state(
            self, initial_vector((num_layers, hidden_size), state_init), learnable_state
        )

    def _initialize_recurrence(self, scheme: str) -> None:
        if scheme not in {"default", "orthogonal", "identity", "identity_zero_input", "zero"}:
            raise ValueError(
                "recurrent initialization must be default, orthogonal, identity, "
                "identity_zero_input, or zero"
            )
        if scheme == "default":
            return
        H = self.hidden_size
        gates = 3 if self.name == "GRU" else 1
        with torch.no_grad():
            for layer in range(self.num_layers):
                w_in = getattr(self.core, f"weight_ih_l{layer}")
                w_rec = getattr(self.core, f"weight_hh_l{layer}")
                if scheme == "zero":
                    w_in.zero_()
                    w_rec.zero_()
                else:
                    if scheme == "identity_zero_input":
                        w_in.zero_()
                    else:
                        nn.init.xavier_uniform_(w_in)
                    if scheme == "orthogonal":
                        for gate in range(gates):
                            nn.init.orthogonal_(w_rec[gate * H : (gate + 1) * H])
                    else:
                        w_rec.zero_()
                        # GRU's candidate-state block is the third gate.
                        w_rec[(gates - 1) * H : gates * H].copy_(torch.eye(H))
                getattr(self.core, f"bias_ih_l{layer}").zero_()
                getattr(self.core, f"bias_hh_l{layer}").zero_()

    def forward(self, tokens: Tensor, state: Tensor | None = None, mask: Tensor | None = None):
        self._check_tokens(tokens)
        lengths = self._check_mask(tokens, mask)
        batch, length = tokens.shape
        h = self.h0[:, None].expand(-1, batch, -1).contiguous() if state is None else state
        if h.shape != (self.num_layers, batch, self.hidden_size):
            raise ValueError(f"{self.name} state must have shape (layers, batch, hidden_size)")
        if length == 0:
            return (
                h.new_empty((batch, 0, self.symbol_size)),
                h.new_empty((batch, 0, self.motion_size)),
                h,
            )
        embedding = self.embedding(tokens)
        if mask is None:
            output, final = self.core(embedding, h)
        else:
            packed = pack_padded_sequence(
                embedding, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            packed_output, final = self.core(packed, h)
            output, _ = pad_packed_sequence(packed_output, batch_first=True, total_length=length)
        symbols, motions = self.B(output), self.C(output)
        if mask is not None:
            symbols = symbols.masked_fill(~mask.unsqueeze(-1), 0)
            motions = motions.masked_fill(~mask.unsqueeze(-1), 0)
        return symbols, motions, final
