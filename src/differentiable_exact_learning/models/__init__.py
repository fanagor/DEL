"""Build any supported controller from one small model configuration."""

from torch import nn

from .base import Controller
from .dfst import DFST, SoftmaxDFST, matrix_prefix_scan
from .recurrent import RecurrentController

MODEL_NAMES = ("DFST", "DFST_Softmax", "RNN", "GRU")


def resolve_hidden_size(config: dict, expert_states: int) -> int:
    """Resolve the one capacity setting using the selected controller's default."""
    hidden_size = config.get("hidden_size")
    if hidden_size is None:
        hidden_size = expert_states if config["name"] in {"DFST", "DFST_Softmax"} else 64
    if isinstance(hidden_size, bool) or not isinstance(hidden_size, int) or hidden_size < 1:
        raise ValueError("model.hidden_size must be null or a positive integer")
    return hidden_size


def build_model(
    config: dict, input_size: int, symbol_size: int, motion_size: int, expert_states: int
) -> nn.Module:
    """Build a dual-head controller. Null initial-state settings use family defaults.

    DFST defaults to a learnable all-ones h0; RNN/GRU default to fixed
    zeros. ``hidden_size=None`` uses the expert state count for DFST variants
    and 64 for RNN and GRU.
    """
    name = config.get("name", "DFST")
    if name not in MODEL_NAMES:
        raise ValueError(f"Unknown model {name!r}; choose one of {MODEL_NAMES}")
    if min(input_size, symbol_size, motion_size, expert_states) <= 0:
        raise ValueError("Vocabulary sizes and expert_states must be positive")
    init = config.get("init") or {}
    dfst = name in {"DFST", "DFST_Softmax"}
    state_init = init.get("state") or ("ones" if dfst else "zeros")
    learnable = init.get("learnable_state")
    if learnable is None:
        learnable = dfst
    common = {
        "input_size": input_size,
        "symbol_size": symbol_size,
        "motion_size": motion_size,
        "embed_dim": config.get("embed_dim"),
        "state_init": state_init,
        "learnable_state": bool(learnable),
    }
    hidden_size = resolve_hidden_size({**config, "name": name}, expert_states)
    if dfst:
        kwargs = dict(
            **common,
            hidden_size=hidden_size,
            scan=config.get("scan", "parallel"),
            transition_init=init.get("transition", "identity"),
            symbol_init=init.get("symbol", "zero"),
            motion_init=init.get("motion", "zero"),
        )
        return (
            SoftmaxDFST(**kwargs, reg_lambda=config.get("reg_lambda", 0.0))
            if name == "DFST_Softmax"
            else DFST(**kwargs)
        )
    kwargs = dict(
        **common,
        hidden_size=hidden_size,
        num_layers=int(config.get("num_layers", 1)),
        recurrent_init=init.get("recurrent", "orthogonal"),
        head_init=init.get("head", "default"),
    )
    return RecurrentController(name, **kwargs, gate_bias=float(init.get("gate_bias", 0.0)))


__all__ = [
    "DFST",
    "MODEL_NAMES",
    "Controller",
    "RecurrentController",
    "SoftmaxDFST",
    "build_model",
    "resolve_hidden_size",
    "matrix_prefix_scan",
]
