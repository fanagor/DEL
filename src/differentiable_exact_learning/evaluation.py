"""Deterministic teacher-forced exact-match length generalization.

Only predictions on expert observations are evaluated. Resource errors propagate;
they must not be recorded as scientific generalization failures.
"""

import hashlib
import math
import random
import re
from itertools import islice
from time import perf_counter

import torch

from .data import Trace, pad_traces
from .evaluation_progress import EvaluationProgress
from .inference import forward_chunk
from .objectives import action_counts, action_loss


def search_lengths(probe, cap, method="binary", *, label=None, completed=None, on_result=None):
    """Return measured passes, all probes, and explicit ceiling information.

    Binary search probes the cap first, then uses the monotonic-pass assumption
    below a failing cap. It cannot rule out failures at untested shorter lengths.
    Exhaustive stops at the first failure.
    A label enables flushed terminal progress before and after each probe.
    """
    if cap < 1:
        if label:
            print(f"[evaluation] {label}: skipped (no positive lengths to test)", flush=True)
        return {"length": 0, "at_cap": False, "cap": cap, "search": method, "tested": []}
    tested = {}

    def check(length):
        if length not in tested:
            prefix = f"[evaluation] {label} length={length}"
            if completed is not None and length in completed:
                tested[length] = completed[length]
                if label:
                    outcome = "PASS" if tested[length] else "FAIL"
                    print(f"{prefix}: RESUMED {outcome}", flush=True)
                return tested[length]
            if label:
                print(f"{prefix}: START", flush=True)
            started = perf_counter()
            try:
                tested[length] = bool(probe(length))
                if on_result is not None:
                    on_result(length, tested[length])
            except Exception as error:
                if label:
                    print(
                        f"{prefix}: ERROR {type(error).__name__}: {error} "
                        f"(elapsed={perf_counter() - started:.2f}s)",
                        flush=True,
                    )
                raise
            if label:
                outcome = "PASS" if tested[length] else "FAIL"
                print(
                    f"{prefix}: {outcome} (elapsed={perf_counter() - started:.2f}s)",
                    flush=True,
                )
        return tested[length]

    best = 0
    if method == "exhaustive":
        for length in range(1, cap + 1):
            if not check(length):
                break
            best = length
    elif method == "binary":
        if check(cap):
            best = cap
        else:
            low, high = 1, cap - 1
            while low <= high:
                middle = (low + high) // 2
                if check(middle):
                    best = middle
                    low = middle + 1
                else:
                    high = middle - 1
    else:
        raise ValueError(f"Unknown search method: {method}")
    if label:
        print(f"[evaluation] {label}: largest passing tested length={best}", flush=True)
    return {
        "length": best,
        "at_cap": best == cap,
        "cap": cap,
        "search": method,
        "tested": [
            {"length": length, "passed": passed} for length, passed in sorted(tested.items())
        ],
    }


def _rng(seed, task, length):
    digest = hashlib.sha256(f"{seed}:{task}:{length}".encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def _groups(items, batch_size):
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    items = iter(items)
    while group := list(islice(items, batch_size)):
        yield group


def _trace_chunks(source, chunk_size):
    """Accept a stored trace or a stream, reblocking without collecting the trajectory."""
    sources = (source,) if isinstance(source, Trace) else source
    parts = [[], [], []]
    size = 0
    for trace in sources:
        offset = 0
        while offset < len(trace):
            end = min(len(trace), offset + chunk_size - size)
            for buffer, values in zip(parts, (trace.inputs, trace.symbols, trace.motions)):
                buffer.append(values[offset:end])
            size += end - offset
            offset = end
            if size == chunk_size:
                chunk = Trace(*(p[0] if len(p) == 1 else torch.cat(p) for p in parts))
                parts, size = [[], [], []], 0
                yield chunk
    if size:
        yield Trace(*(p[0] if len(p) == 1 else torch.cat(p) for p in parts))


def _chunk_batches(sources, chunk_size):
    """Keep fixed batch slots; completed rows receive only masked padding.

    Every nonfinal chunk contains chunk_size transitions. Consequently a row
    with padding never receives more real tokens, so padded recurrent states
    cannot affect later predictions. Slots never get reused within a group.
    """
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    streams = [iter(_trace_chunks(source, chunk_size)) for source in sources]
    padding = Trace(*(torch.zeros(1, dtype=torch.long) for _ in range(3)))
    seen = [False] * len(streams)
    while True:
        chunks = [next(stream, None) for stream in streams]
        for i, chunk in enumerate(chunks):
            if chunk is not None:
                seen[i] = True
            elif not seen[i]:
                raise ValueError("Cannot evaluate an empty trace stream")
        if all(chunk is None for chunk in chunks):
            return
        batch = pad_traces([chunk if chunk is not None else padding for chunk in chunks])
        for i, chunk in enumerate(chunks):
            if chunk is None:
                batch.mask[i] = False
        yield batch


@torch.no_grad()
def traces_match(model, traces, *, batch_size=10, chunk_size=256, scan="sequential"):
    """Check stored traces or lazy chunk streams; stop at the first failing chunk."""
    device = next(model.parameters()).device
    count = 0
    for group in _groups(traces, batch_size):
        count += len(group)
        state = None
        for batch in _chunk_batches(group, chunk_size):
            batch = batch.to(device)
            symbols, motions, state = forward_chunk(
                model, batch.inputs, state, stabilize=True, scan=scan
            )
            correct = (
                (symbols.argmax(-1) == batch.symbols)
                & (motions.argmax(-1) == batch.motions)
                & torch.isfinite(symbols).all(-1)
                & torch.isfinite(motions).all(-1)
            )
            if not bool((correct | ~batch.mask).all()):
                return False
    if not count:
        raise ValueError("Cannot evaluate an empty probe set")
    return True


@torch.no_grad()
def snapshot_metrics(
    model, samples, *, batch_size, chunk_size, loss, scan="sequential", label=None, on_point=None
):
    """Accumulate one loss/accuracy point per input without retaining full trajectories.

    Scores remain unscaled to match the training objective. Nonfinite loss
    becomes null, with nonfinite predictions counted as incorrect.
    """
    device = next(model.parameters()).device
    points = []
    for group in _groups(samples, batch_size):
        if label:
            for length, _ in group:
                print(f"[evaluation] {label} snapshot length={length}: START", flush=True)
        totals = [0.0] * len(group)
        correct = [0] * len(group)
        counts = [0] * len(group)
        finished = {}

        def finish(i):
            if i in finished or not counts[i]:
                return
            length = group[i][0]
            value = totals[i] / counts[i]
            point = {
                "length": length,
                "loss": value if math.isfinite(value) else None,
                "loss_finite": math.isfinite(value),
                "accuracy": correct[i] / counts[i],
                "timesteps": counts[i],
            }
            if on_point is not None:
                on_point(point)
            finished[i] = point
            if label:
                print(
                    f"[evaluation] {label} snapshot length={length}: DONE "
                    f"accuracy={point['accuracy']:.4%} loss={point['loss']}",
                    flush=True,
                )

        state = None
        for batch in _chunk_batches([source for _, source in group], chunk_size):
            batch = batch.to(device)
            symbols, motions, state = forward_chunk(
                model, batch.inputs, state, stabilize=False, scan=scan
            )
            for i in range(len(group)):
                mask = batch.mask[i : i + 1]
                count = int(mask.sum())
                if not count:
                    finish(i)
                    continue
                ts, tm = batch.symbols[i : i + 1], batch.motions[i : i + 1]
                s, m = symbols[i : i + 1], motions[i : i + 1]
                value = action_loss(s, m, ts, tm, mask, loss=loss, output_kind=model.output_kind)
                totals[i] += float(value) * count
                correct[i] += action_counts(s, m, ts, tm, mask)[0]
                counts[i] += count
                if count < chunk_size:
                    finish(i)
        for i in range(len(group)):
            finish(i)
            points.append(finished[i])
    return points


def evaluate(model, expert, config, *, progress_dir=None):
    options, data = config["evaluation"], config["data"]
    scan = options.get("scan", "sequential")
    if scan not in ("sequential", "parallel"):
        raise ValueError("evaluation.scan must be sequential or parallel")
    progress = EvaluationProgress(model, expert, config, progress_dir)
    was_training = model.training
    model.eval()
    try:

        def sample(rng, length, **kwargs):
            return expert.sample_stream(rng, length, chunk_size=options["chunk_size"], **kwargs)

        def check(traces):
            return traces_match(
                model,
                traces,
                batch_size=options["batch_size"],
                chunk_size=options["chunk_size"],
                scan=scan,
            )

        cap, method = options["max_length"], options["search"]
        result = {
            "protocol": "teacher_forced_exact_match",
            "scan": scan,
            "seed": options["seed"],
            "random_samples": options["samples"],
            "tasks": {},
        }
        if data["environment"] != "teleporting":
            task, radix = data["task"], data["radix"]
            label = f"{task}{radix}"
            random_results = progress.results(f"{label} ULG")

            def random_passes(length):
                if length not in random_results:
                    rng = _rng(options["seed"], task, length)
                    passed = check(
                        sample(rng, length, lengths=(length, length))
                        for _ in range(options["samples"])
                    )
                    progress.record(f"{label} ULG", length, passed)
                return random_results[length]

            ulg = search_lengths(
                random_passes, cap, method, label=f"{label} ULG", completed=random_results
            )
            metrics = {"ulg": ulg, "hard_cases": options["hard_cases"]}
            if options["hard_cases"] != "none":

                def robust_passes(length):
                    if not random_passes(length):
                        return False
                    digits = [
                        (a, b)
                        for a in range(radix)
                        for b in range(radix)
                        if options["hard_cases"] == "all" or a == b
                    ]
                    rng = _rng(options["seed"], task, length)
                    return check(
                        sample(rng, length, operands=(str(a) * length, str(b) * length))
                        for a, b in digits
                    )

                metrics["rlg"] = search_lengths(
                    robust_passes,
                    ulg["length"],
                    method,
                    label=f"{label} RLG",
                    completed=progress.results(f"{label} RLG"),
                    on_result=lambda length, passed: progress.record(
                        f"{label} RLG", length, passed
                    ),
                )
                # Reaching the ULG limit is not necessarily reaching the requested ceiling.
                metrics["rlg"]["at_cap"] = metrics["rlg"]["length"] == cap
                metrics["rlg"]["cap"] = cap
            result["tasks"][label] = metrics
        else:
            for task in data["eval_tasks"]:
                if isinstance(task, dict):
                    label = task["name"] + (f"_m{task['m']}" if "m" in task else "")
                else:
                    label = task
                family = re.match(r"^(scaling_[AB])(?:_m\d+)?$", label)

                def probe(length):
                    current_task = f"{family.group(1)}_m{length}" if family else task
                    rng = _rng(options["seed"], label, length)
                    return check(
                        sample(rng, length, task=current_task, lengths=length)
                        for _ in range(options["samples"])
                    )

                result["tasks"][label] = {
                    "generalization": search_lengths(
                        probe,
                        cap,
                        method,
                        label=f"{label} generalization",
                        completed=progress.results(f"{label} generalization"),
                        on_result=lambda length, passed: progress.record(
                            f"{label} generalization", length, passed
                        ),
                    ),
                    "axis": "input_and_machine_size" if family else "input_length",
                }
        snapshot = options.get("snapshot", {"enabled": True, "m": 100, "d": 5})
        if snapshot["enabled"]:
            for label, metrics in result["tasks"].items():
                family = re.match(r"^(scaling_[AB])(?:_m\d+)?$", label)
                completed = progress.snapshots(label)

                def samples():
                    for length in range(snapshot["d"], snapshot["m"] + 1, snapshot["d"]):
                        if str(length) in completed:
                            print(
                                f"[evaluation] {label} snapshot length={length}: RESUMED",
                                flush=True,
                            )
                            continue
                        rng = _rng(options["seed"], f"snapshot:{label}", length)
                        if data["environment"] == "teleporting":
                            task = f"{family.group(1)}_m{length}" if family else label
                            trace = sample(rng, length, task=task, lengths=length)
                        else:
                            trace = sample(rng, length, lengths=(length, length))
                        yield length, trace

                snapshot_metrics(
                    model,
                    samples(),
                    batch_size=options["batch_size"],
                    chunk_size=options["chunk_size"],
                    scan=scan,
                    loss=config["training"]["loss"],
                    label=label,
                    on_point=lambda point: progress.record_snapshot(label, point),
                )
                metrics["snapshot"] = {
                    "m": snapshot["m"],
                    "d": snapshot["d"],
                    "loss_kind": config["training"]["loss"],
                    "axis": metrics.get("axis", "operand_length"),
                    "points": [
                        completed[str(length)]
                        for length in range(snapshot["d"], snapshot["m"] + 1, snapshot["d"])
                    ],
                }
        return result
    finally:
        model.train(was_training)
