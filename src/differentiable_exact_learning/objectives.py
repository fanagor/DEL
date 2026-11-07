"""Masked two-head objectives and the paper's exponential learning-rate schedule."""

import math

import torch
import torch.nn.functional as F


def _step_weights(
    inputs, target_symbol, target_motion, mask, *, symbol_size, motion_size, alpha, dtype
):
    """Per-PTO observable (x, s, m) inverse-frequency weights n^(-alpha), normalized over mask."""
    if inputs is None:
        raise ValueError("inputs must be provided when inverse_freq_alpha > 0")
    if inputs.shape != mask.shape:
        raise ValueError("inputs must match mask shape")
    m2 = mask.unsqueeze(0) if mask.ndim == 1 else mask.reshape(-1, mask.shape[-1])
    x2 = (inputs.unsqueeze(0) if mask.ndim == 1 else inputs.reshape(-1, mask.shape[-1])).where(
        m2, 0
    )
    s2 = (
        target_symbol.unsqueeze(0) if mask.ndim == 1 else target_symbol.reshape(-1, mask.shape[-1])
    ).where(m2, 0)
    t2 = (
        target_motion.unsqueeze(0) if mask.ndim == 1 else target_motion.reshape(-1, mask.shape[-1])
    ).where(m2, 0)
    if (x2 < 0).any() or (s2 < 0).any() or (t2 < 0).any():
        raise ValueError("Token indices on masked steps must be nonnegative")
    input_size = max(int(x2.max().item()) + 1, symbol_size)
    triple = (x2 * symbol_size + s2) * motion_size + t2
    counts = torch.zeros(
        m2.shape[0],
        input_size * symbol_size * motion_size,
        dtype=dtype,
        device=mask.device,
    )
    counts.scatter_add_(1, triple, m2.to(dtype))
    weights = counts.gather(1, triple)[m2].pow(-float(alpha))
    return weights / weights.sum()


def action_loss(
    symbol,
    motion,
    target_symbol,
    target_motion,
    mask,
    *,
    inputs=None,
    inverse_freq_alpha=0.0,
    loss="mse",
    output_kind="logits",
):
    """Normalized (weighted) mean over real timesteps; the two heads receive equal weight.

    MSE sums squared errors within each vocabulary, as in paper Eq. (2).
    Cross-entropy takes logits for ordinary controllers and probabilities for
    DFST_Softmax. Applying softmax before torch's cross_entropy would be incorrect.
    Padding is excluded before computing either loss, including nonfinite padding.
    When ``inverse_freq_alpha > 0``, each valid step is weighted by ``n**(-alpha)``,
    where ``n`` is the count of its observable ``(x, s, m)`` triple within its PTO,
    normalized to sum to 1 across valid steps.
    """
    if mask.dtype != torch.bool or mask.shape != target_symbol.shape:
        raise ValueError("mask must be boolean and match the target shape")
    if not mask.any():
        raise ValueError("A training/evaluation batch must contain real timesteps")
    if (
        isinstance(inverse_freq_alpha, bool)
        or not isinstance(inverse_freq_alpha, (int, float))
        or not math.isfinite(inverse_freq_alpha)
        or inverse_freq_alpha < 0
    ):
        raise ValueError("inverse_freq_alpha must be finite and nonnegative")
    s, m, ts, tm = symbol[mask], motion[mask], target_symbol[mask], target_motion[mask]
    if loss == "mse":
        s_step = (s - F.one_hot(ts, s.shape[-1]).to(s.dtype)).square().sum(-1)
        m_step = (m - F.one_hot(tm, m.shape[-1]).to(m.dtype)).square().sum(-1)
    elif loss == "cross_entropy":
        if output_kind == "probabilities":
            # This is categorical cross-entropy on already-normalized predictions.
            # Clamp only at dtype tiny; do not normalize a second time.
            s_step = F.nll_loss(s.clamp_min(torch.finfo(s.dtype).tiny).log(), ts, reduction="none")
            m_step = F.nll_loss(m.clamp_min(torch.finfo(m.dtype).tiny).log(), tm, reduction="none")
        elif output_kind == "logits":
            s_step = F.cross_entropy(s, ts, reduction="none")
            m_step = F.cross_entropy(m, tm, reduction="none")
        else:
            raise ValueError(f"Unknown output_kind: {output_kind}")
    else:
        raise ValueError(f"Unknown loss: {loss}")
    if inverse_freq_alpha == 0:
        s_loss, m_loss = s_step.mean(), m_step.mean()
    else:
        weights = _step_weights(
            inputs,
            target_symbol,
            target_motion,
            mask,
            symbol_size=s.shape[-1],
            motion_size=m.shape[-1],
            alpha=inverse_freq_alpha,
            dtype=s.dtype,
        )
        s_loss = (s_step * weights).sum()
        m_loss = (m_step * weights).sum()
    return (s_loss + m_loss) / 2


def action_counts(symbol, motion, target_symbol, target_motion, mask):
    """Joint action accuracy and whole-trace exact match, excluding padding."""
    correct = (symbol.argmax(-1) == target_symbol) & (motion.argmax(-1) == target_motion)
    finite = torch.isfinite(symbol).all(-1) & torch.isfinite(motion).all(-1)
    correct = correct & finite
    return int((correct & mask).sum()), int(mask.sum()), (correct | ~mask).all(-1)


def exponential_lr(step, *, total_steps, warmup_steps, peak, floor, constant_steps=0):
    """Paper EWEC schedule at update indices 0..T-1, with an endpoint at T.

    With zero warmup, update zero uses the peak. Otherwise update zero uses
    the floor and update `warmup_steps` uses the peak. The peak then holds for
    `constant_steps` updates before the cooldown; zero gives the paper schedule.
    The final endpoint at T equals the floor. Compute from the absolute step,
    so resume needs no mutable scheduler state.
    """
    if (
        not 0 <= warmup_steps
        or not 0 <= constant_steps
        or not warmup_steps + constant_steps < total_steps
        or not 0 < floor <= peak
    ):
        raise ValueError(
            "Require 0 <= warmup_steps, 0 <= constant_steps, "
            "warmup_steps + constant_steps < total_steps and 0 < floor <= peak"
        )
    step = min(max(step, 0), total_steps)
    if warmup_steps and step < warmup_steps:
        fraction = step / warmup_steps
        return math.exp(math.log(floor) * (1 - fraction) + math.log(peak) * fraction)
    decay_start = warmup_steps + constant_steps
    if step < decay_start:
        return peak
    fraction = (step - decay_start) / (total_steps - decay_start)
    return math.exp(math.log(peak) * (1 - fraction) + math.log(floor) * fraction)
