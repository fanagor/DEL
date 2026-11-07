"""Static and online PTO data share the same expert and padded-batch interface."""

from __future__ import annotations

import hashlib
import json
import os
import random
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from .experts import Batch, Expert, Trace, create_expert, pad_traces, task_label
from .teaching import (
    TP_TEACHING_SPEC,
    arithmetic_teaching_pairs,
    teleporting_teaching_inputs,
    validate_teaching,
)

SCHEMA_VERSION = 1


def dataset_spec(config: Mapping[str, Any]) -> dict:
    validate_teaching(config)
    expert = create_expert(config)
    spec = {
        "schema": SCHEMA_VERSION,
        "environment": expert.environment,
        "task": expert.task if expert.environment != "teleporting" else "universal",
        "radix": expert.radix,
        "max_length": int(config.get("max_length", 3)),
        "size": int(config.get("size", 100)),
        "seed": int(config.get("seed", 0)),
        "max_steps": expert.max_steps,
    }
    if expert.variant != "default":
        spec["variant"] = expert.variant
    if expert.environment == "teleporting":
        spec["train_tasks"] = expert.train_tasks
    if config.get("teaching_set", False):
        spec["teaching_set"] = True
        # Teaching selection can change without changing the expert controller.
        examples = (
            teleporting_teaching_inputs(config)
            if expert.environment == "teleporting"
            else arithmetic_teaching_pairs(config)
        )
        spec["teaching_digest"] = hashlib.sha256(
            json.dumps(examples, separators=(",", ":")).encode()
        ).hexdigest()
    if spec["size"] < 1:
        raise ValueError("Dataset size must be positive")
    if spec["max_length"] < (0 if expert.environment == "teleporting" else 1):
        raise ValueError("Invalid maximum input length")
    # A change to an expert program invalidates cached traces automatically.
    relevant = [expert.transitions]
    if expert.environment == "teleporting":
        from .experts import task_transitions

        relevant.extend(task_transitions(t) for t in expert.train_tasks)
        if spec.get("teaching_set"):
            existing = set(expert.train_tasks)
            relevant.extend(task_transitions(t) for t, _ in TP_TEACHING_SPEC if t not in existing)
    serialized = repr([sorted(table.items()) for table in relevant]).encode()
    spec["expert_digest"] = hashlib.sha256(serialized).hexdigest()
    return spec


def dataset_fingerprint(config: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(dataset_spec(config), sort_keys=True).encode()).hexdigest()[
        :24
    ]


def _arithmetic_pairs(spec, rng):
    """Source protocol: uniform lengths, then uniform digit strings, with deduplication.

    Leading zeros remain significant. The paper's phrase "uniformly at random"
    does not specify this length stratification; this preserves the experiment
    code's protocol while requiring exactly N samples and all R² single digits.
    """
    radix, maximum, count = spec["radix"], spec["max_length"], spec["size"]
    capacity = sum(radix**length for length in range(1, maximum + 1)) ** 2
    if count > capacity:
        raise ValueError(f"Requested {count} unique pairs, but only {capacity} exist")
    mandatory = []
    seen = set()

    def add(a, b):
        pair = (tuple(a), tuple(b))
        if pair not in seen:
            seen.add(pair)
            mandatory.append(pair)

    for a in range(radix):
        for b in range(radix):
            add((str(a),), (str(b),))
    for a, b in arithmetic_teaching_pairs(spec):
        add(a, b)
    if count < len(mandatory):
        raise ValueError(
            f"size={count} is smaller than the {len(mandatory)} mandatory "
            "single-digit/teaching pairs"
        )
    yield from mandatory
    # Near-full finite datasets must not rely on increasingly unlikely rejection
    # draws. Weighted sampling without replacement over remaining pairs retains
    # the source's length-then-digit proposal distribution.
    if capacity <= 100_000 and count > capacity // 2:
        import itertools

        digits = [str(d) for d in range(radix)]
        words = [w for n in range(1, maximum + 1) for w in itertools.product(digits, repeat=n)]
        candidates = []
        for a in words:
            for b in words:
                if (a, b) not in seen:
                    weight = radix ** (-len(a) - len(b))
                    # Exponential keys implement weighted sampling without replacement.
                    import math

                    key = -math.log(1.0 - rng.random()) / weight
                    candidates.append((key, a, b))
        candidates.sort(key=lambda item: item[0])
        for _, a, b in candidates[: count - len(mandatory)]:
            yield a, b
        return
    while len(seen) < count:
        m, n = rng.randint(1, maximum), rng.randint(1, maximum)
        pair = (
            tuple(str(rng.randrange(radix)) for _ in range(m)),
            tuple(str(rng.randrange(radix)) for _ in range(n)),
        )
        if pair not in seen:
            seen.add(pair)
            yield pair


def _generate(config, spec):
    expert, rng = create_expert(config), random.Random(spec["seed"])
    if expert.environment == "teleporting":
        if spec.get("teaching_set"):
            inputs = list(teleporting_teaching_inputs(spec))
        else:
            if spec["size"] < len(expert.train_tasks):
                raise ValueError(
                    "Teleporting static size must cover every configured training task"
                )
            # Include the empty input once per task so every curriculum stage has
            # support; remaining examples use uniform tasks and lengths in [0,q].
            inputs = [(t, "") for t in expert.train_tasks]
        if spec["size"] < len(inputs):
            raise ValueError(
                f"size={spec['size']} is smaller than the {len(inputs)} mandatory TP examples"
            )
        traces = [expert.sample(rng, spec["max_length"], task=t, operands=w) for t, w in inputs]
        traces.extend(
            expert.sample(rng, spec["max_length"]) for _ in range(spec["size"] - len(traces))
        )
        return traces
    return [
        expert.sample(rng, spec["max_length"], operands=pair)
        for pair in _arithmetic_pairs(spec, rng)
    ]


def _payload(traces, spec):
    return {
        "spec": spec,
        "traces": [
            {"inputs": t.inputs, "symbols": t.symbols, "motions": t.motions, "metadata": t.metadata}
            for t in traces
        ],
    }


def _read(path, spec, expert):
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("spec") != spec or len(payload["traces"]) != spec["size"]:
            raise ValueError("Dataset manifest/size does not match the requested specification")
        traces = [Trace(**item) for item in payload["traces"]]
        for trace in traces:
            for tensor, size in (
                (trace.inputs, expert.input_size),
                (trace.symbols, expert.symbol_size),
                (trace.motions, expert.motion_size),
            ):
                if tensor.min() < 0 or tensor.max() >= size:
                    raise ValueError("Dataset contains an out-of-vocabulary token")
            if (
                not trace.metadata.get("lengths")
                or max(trace.metadata["lengths"]) > spec["max_length"]
            ):
                raise ValueError("Dataset metadata contains invalid lengths")
            if trace.metadata.get("environment") != spec["environment"]:
                raise ValueError("Dataset environment does not match")
        return traces
    except Exception as exc:
        raise ValueError(f"Invalid/incomplete cached dataset {path}: {exc}") from exc


def build_static_dataset(config: Mapping[str, Any], cache_dir=None) -> list[Trace]:
    """Generate exactly size traces, or read a fully validated matching cache.

    Writes are atomic. Existing malformed files raise an explicit error; they
    are never silently trusted, truncated, or replaced with a different dataset.
    """
    spec = dataset_spec(config)
    destination = cache_dir if cache_dir is not None else config.get("cache_dir")
    if destination is None:
        return _generate(config, spec)
    directory = Path(destination)
    directory.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:24]
    path = directory / f"pto-{digest}.pt"
    if path.exists():
        return _read(path, spec, create_expert(config))
    traces = _generate(config, spec)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "wb") as stream:
            torch.save(_payload(traces, spec), stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return traces


class BatchSampler:
    """Uniform length-bucket sampling, with reusable curriculum indexes.

    ``batch_lengths=fixed`` selects one length tuple for the whole minibatch;
    ``mixed`` independently selects a tuple per example. Within a selected
    static bucket examples are uniform, regardless of other bucket sizes.
    Teleporting tasks are selected uniformly once per minibatch in both modes.
    """

    def __init__(
        self,
        expert: Expert,
        rng: random.Random,
        batch_size: int,
        traces: Sequence[Trace] | None = None,
    ):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.expert, self.rng, self.batch_size = expert, rng, batch_size
        self.batch_lengths = expert.config.get("batch_lengths", "fixed")
        if self.batch_lengths not in {"fixed", "mixed"}:
            raise ValueError("batch_lengths must be fixed or mixed")
        self.traces = tuple(traces) if traces is not None else None
        self._buckets = {}
        self._pools = {}
        if self.traces is not None:
            for trace in self.traces:
                key = (trace.metadata["task"], tuple(trace.metadata["lengths"]))
                self._buckets.setdefault(key, []).append(trace)
            self._buckets = {key: tuple(pool) for key, pool in self._buckets.items()}

    def _pool(self, max_length, label):
        key = (max_length, label)
        if key not in self._pools:
            self._pools[key] = tuple(
                pool
                for (task, lengths), pool in self._buckets.items()
                if max(lengths) <= max_length and (label is None or task == label)
            )
        return self._pools[key]

    def _draw_lengths(self, max_length):
        if self.expert.environment == "teleporting":
            return self.rng.randint(0, max_length)
        return (self.rng.randint(1, max_length), self.rng.randint(1, max_length))

    def sample(self, max_length: int, task=None) -> Batch:
        label = (
            task_label(task)
            if task is not None and self.expert.environment == "teleporting"
            else task
        )
        if self.traces is not None:
            if self.expert.environment == "teleporting" and label is None:
                extra_tasks = [
                    t
                    for t in dict.fromkeys(task for task, _ in self._buckets)
                    if t not in set(self.expert.train_tasks)
                ]
                available = [
                    t for t in [*self.expert.train_tasks, *extra_tasks] if self._pool(max_length, t)
                ]
                if not available:
                    raise ValueError(f"No static traces available for max_length={max_length}")
                label = self.rng.choice(available)
            buckets = self._pool(max_length, label)
            if not buckets:
                raise ValueError(
                    f"No static traces available for max_length={max_length}, task={label!r}"
                )
            if self.batch_lengths == "fixed":
                bucket = self.rng.choice(buckets)
                selected = [self.rng.choice(bucket) for _ in range(self.batch_size)]
            else:
                selected = [
                    self.rng.choice(self.rng.choice(buckets)) for _ in range(self.batch_size)
                ]
            return pad_traces(selected)
        if label is None and self.expert.environment == "teleporting":
            label = self.rng.choice(self.expert.train_tasks)
        lengths = self._draw_lengths(max_length) if self.batch_lengths == "fixed" else None
        return pad_traces(
            [
                self.expert.sample(self.rng, max_length, task=label, lengths=lengths)
                for _ in range(self.batch_size)
            ]
        )


def sample_batch(
    expert: Expert,
    rng: random.Random,
    batch_size: int,
    max_length: int,
    traces: Sequence[Trace] | None = None,
    task=None,
) -> Batch:
    """Convenience wrapper; use BatchSampler to cache static curriculum pools."""
    return BatchSampler(expert, rng, batch_size, traces).sample(max_length, task)


class OnlineStream:
    """Reproducible CPU minibatch stream with explicit serializable RNG state."""

    def __init__(self, expert: Expert, batch_size: int, max_length: int, seed: int = 0):
        self.expert, self.batch_size, self.max_length = expert, batch_size, max_length
        self.rng = random.Random(seed)
        self.sampler = BatchSampler(expert, self.rng, batch_size)

    def __iter__(self):
        return self

    def __next__(self):
        return self.sampler.sample(self.max_length)

    def state_dict(self):
        return {"rng": self.rng.getstate(), "max_length": self.max_length}

    def load_state_dict(self, state):
        self.rng.setstate(state["rng"])
        self.max_length = state["max_length"]
