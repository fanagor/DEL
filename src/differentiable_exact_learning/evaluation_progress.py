"""Atomic, model-specific evaluation progress across interrupted probes."""

import hashlib
import json
from pathlib import Path

import torch

from .checkpoint import atomic_json
from .data.datasets import dataset_spec


def _identity(model, expert, config):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        tensor = tensor.detach().cpu().contiguous()
        digest.update(json.dumps([name, str(tensor.dtype), list(tensor.shape)]).encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return {
        "protocol_version": 1,
        "model_digest": digest.hexdigest(),
        "model_class": f"{type(model).__module__}.{type(model).__qualname__}",
        "model_config": config["model"],
        "expert_digest": dataset_spec(config["data"])["expert_digest"],
        "controller_digest": hashlib.sha256(
            repr(sorted(expert.transitions.items())).encode()
        ).hexdigest(),
        "data": {
            **{
                key: config["data"].get(key)
                for key in ("environment", "task", "radix", "eval_tasks", "max_steps")
            },
            **(
                {"variant": config["data"]["variant"]}
                if config["data"].get("environment") == "grid"
                and config["data"].get("variant", "default") != "default"
                else {}
            ),
        },
        "evaluation": {
            **{
                key: value
                for key, value in config["evaluation"].items()
                if key not in {"every", "final"}
            },
            "scan": config["evaluation"].get("scan", "sequential"),
        },
        "loss": config["training"]["loss"],
        "runtime": {
            "torch": str(torch.__version__),
            "device": next(model.parameters()).device.type,
            "matmul_precision": torch.get_float32_matmul_precision(),
            "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        },
    }


class EvaluationProgress:
    """Only completed probes are committed; different evaluation identities never mix."""

    def __init__(self, model, expert, config, directory=None):
        self.path = None
        self.document = {"probes": {}, "snapshots": {}}
        self._results = {}
        if directory is not None:
            identity = _identity(model, expert, config)
            digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
            self.path = Path(directory) / f"{digest}.json"
            self.document["identity"] = identity
            if self.path.exists():
                saved = json.loads(self.path.read_text())
                if saved.get("identity") != identity:
                    raise ValueError(f"Evaluation progress identity mismatch: {self.path}")
                self.document = saved
                # Reject malformed outcomes instead of silently treating them as failures.
                for values in saved["probes"].values():
                    for length, passed in values.items():
                        if (
                            str(int(length)) != length
                            or int(length) < 1
                            or type(passed) is not bool
                        ):
                            raise ValueError(f"Invalid evaluation progress: {self.path}")

    def save(self):
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            atomic_json(self.document, self.path)

    def results(self, label):
        if label not in self._results:
            self._results[label] = {
                int(length): passed
                for length, passed in self.document["probes"].get(label, {}).items()
            }
        return self._results[label]

    def record(self, label, length, passed):
        if type(length) is not int or length < 1 or type(passed) is not bool:
            raise ValueError("A completed probe needs a positive length and a boolean result")
        results = self.results(label)
        if length in results and results[length] != passed:
            raise ValueError(f"Conflicting evaluation results for {label} length={length}")
        self.document["probes"].setdefault(label, {})[str(length)] = passed
        results[length] = passed
        self.save()

    def snapshots(self, label):
        return self.document["snapshots"].setdefault(label, {})

    def record_snapshot(self, label, point):
        self.snapshots(label)[str(point["length"])] = point
        self.save()
