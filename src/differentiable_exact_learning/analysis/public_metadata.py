"""Remove tracking metadata while retaining scientific checkpoint state."""

import io
from copy import deepcopy

import torch

PRIVATE_KEYS = {
    "tracking",
    "wandb_id",
    "wandb_project",
    "wandb_url",
    "source_wandb_id",
    "source_wandb_url",
    "evaluation_wandb",
    "wandb_run_id",
    "wandb_run_url",
}


def public_metadata(value):
    """Return a sanitized copy; tensors, optimizer state and RNG state remain intact."""
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if key in PRIVATE_KEYS:
                continue
            if key == "wandb":
                result[key] = {"enabled": False, "project": "DEL", "entity": None}
            elif key == "run" and isinstance(item, dict):
                result[key] = public_metadata(item)
                result[key]["name"] = None
            elif key in {"entity", "project"}:
                continue
            else:
                result[key] = public_metadata(item)
        return result
    if isinstance(value, list):
        return [public_metadata(item) for item in value]
    if isinstance(value, tuple):
        return tuple(public_metadata(item) for item in value)
    return deepcopy(value)


def public_checkpoint_bytes(saved):
    stream = io.BytesIO()
    torch.save(public_metadata(saved), stream)
    return stream.getvalue()
