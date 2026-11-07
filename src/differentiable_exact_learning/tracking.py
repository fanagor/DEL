"""Persistent W&B identity, checkpoint-aligned history, and evaluation plots."""

import hashlib
import json
import os
import uuid
from copy import deepcopy

import numpy as np

from .checkpoint import atomic_json


class WandbTracker:
    """Journal events before checkpointing; publish only after the checkpoint commits.

    W&B's internal step indexes journal entries, not optimizer updates. The
    checkpoint cursor counts committed entries. Reconnection replays missing
    uploads; an older checkpoint rewinds the same remote run to that cursor.
    """

    def __init__(self, config, run_dir, saved, training_step):
        self.config = config
        self.run_dir = run_dir
        self.path = run_dir / "wandb-run.json"
        self.journal = run_dir / "wandb-history.jsonl"
        self.run = None
        self.sent = 0
        options = config["run"]["wandb"]
        disk = json.loads(self.path.read_text()) if self.path.exists() else None
        self.identity = deepcopy(saved["identity"] if saved else disk)
        if self.identity is None:
            self.identity = {
                "id": uuid.uuid4().hex,
                "entity": options["entity"],
                "project": options["project"],
                "established": False,
                "rewind_to": None,
            }
        if disk:
            for key in ("id", "entity", "project"):
                a, b = self.identity[key], disk[key]
                if a is not None and b is not None and a != b:
                    raise ValueError("W&B checkpoint identity conflicts with wandb-run.json")
            # The sidecar may contain the resolved entity from a connection that
            # completed just after the restored checkpoint was saved.
            for key in ("id", "entity", "project"):
                if disk[key] is not None:
                    self.identity[key] = disk[key]
            self.identity["established"] |= disk["established"]
        pending = [
            value
            for value in (self.identity.get("rewind_to"), disk.get("rewind_to") if disk else None)
            if value is not None
        ]
        self.identity["rewind_to"] = min(pending) if pending else None
        if options["project"] != self.identity["project"] or (
            options["entity"] is not None
            and self.identity["entity"] is not None
            and options["entity"] != self.identity["entity"]
        ):
            raise ValueError(
                "An existing W&B run requires the same project/entity; use a new run.name"
            )
        self.identity["entity"] = self.identity["entity"] or options["entity"]
        self.events = []
        self.history_hash = hashlib.sha256()
        cursor = saved["cursor"] if saved else 0
        if saved is None and self.identity["established"]:
            # A checkpoint saved before tracking began has no event cursor.
            # Preserve the initial tracked training state when resuming.
            if not self.journal.exists():
                raise ValueError("Missing committed W&B history: retain wandb-history.jsonl")
            with self.journal.open() as stream:
                anchor = json.loads(stream.readline())
            if anchor["data"]["step"] > training_step:
                raise ValueError(
                    "Checkpoint predates this W&B run's history; choose a tracked checkpoint"
                )
            cursor = 1
        if isinstance(cursor, bool) or not isinstance(cursor, int) or cursor < (1 if saved else 0):
            raise ValueError("Invalid W&B checkpoint history cursor")
        if cursor:
            if not self.journal.exists():
                raise ValueError("Missing committed W&B history: retain wandb-history.jsonl")
            with self.journal.open("rb") as stream:
                for index in range(cursor):
                    try:
                        raw = stream.readline()
                        event = json.loads(raw)
                        if (
                            event["index"] != index
                            or not 0 <= event["data"]["step"] <= training_step
                            or event["kind"] not in {"anchor", "training", "evaluation"}
                            or not isinstance(event["key"], str)
                        ):
                            raise ValueError("History index or training step differs")
                        self.events.append(event)
                        self.history_hash.update(raw)
                    except (ValueError, KeyError, TypeError) as error:
                        raise ValueError(
                            "Committed W&B history is corrupt or incomplete"
                        ) from error
                boundary = stream.tell()
        else:
            boundary = 0
        if saved and saved.get("history_sha256") != self.history_hash.hexdigest():
            raise ValueError("Committed W&B history checksum differs from the checkpoint")
        if self.identity["established"] and self.journal.stat().st_size > boundary:
            # Remember rollback even if this execution stays offline and later
            # grows a new history beyond the old server cursor. Length alone can
            # no longer distinguish those branches on the next online restart.
            previous = self.identity["rewind_to"]
            self.identity["rewind_to"] = min(previous, cursor) if previous is not None else cursor
        # Discard only the uncommitted tail, including a partial final line.
        with self.journal.open("a+b") as stream:
            stream.truncate(boundary)
            stream.flush()
            os.fsync(stream.fileno())
        self.keys = {event["key"] for event in self.events}
        if not self.events:
            # A durable row at _step=0 provides a valid rewind anchor even for
            # rollback to the initial model checkpoint (W&B disallows _step=-1).
            self.record("anchor", {"step": training_step}, "anchor")
        atomic_json(self.identity, self.path)

    def state_dict(self):
        return {
            "identity": deepcopy(self.identity),
            "cursor": len(self.events),
            "history_sha256": self.history_hash.hexdigest(),
        }

    def record(self, kind, data, key):
        if key in self.keys:
            return False
        event = {"index": len(self.events), "kind": kind, "key": key, "data": data}
        encoded = (json.dumps(event, allow_nan=False) + "\n").encode()
        with self.journal.open("ab") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        self.events.append(event)
        self.history_hash.update(encoded)
        self.keys.add(key)
        return True

    def connect(self, initialization):
        # Offline executions keep a DEL journal instead of separate W&B offline
        # sessions, which cannot resume server history safely across restarts.
        if os.environ.get("WANDB_MODE", "online").lower() in {"offline", "dryrun", "disabled"}:
            return
        try:
            import wandb
        except ImportError as error:
            raise RuntimeError(
                "Install differentiable-exact-learning[tracking] to enable W&B"
            ) from error
        settings = wandb.Settings(init_timeout=60, mode="online")
        arguments = dict(
            entity=self.identity["entity"],
            project=self.identity["project"],
            id=self.identity["id"],
            group=self.run_dir.name,
            name=self.config["run"]["name"],
            config=self.config,
            dir=str(self.run_dir),
            settings=settings,
        )
        self.run = wandb.init(
            **arguments, resume="must" if self.identity["established"] else "allow"
        )
        self.identity.update(entity=self.run.entity, project=self.run.project, established=True)
        atomic_json(self.identity, self.path)
        self.sent = self.run.step
        limit = min(len(self.events), self.identity["rewind_to"] or len(self.events))
        if self.sent > limit:
            # Ordinary resume cannot remove invalid newer metrics. Finish the
            # connection before rewind, which requires a terminal remote run.
            self.finish()
            arguments.update(entity=self.identity["entity"], project=self.identity["project"])
            try:
                self.run = wandb.init(
                    **arguments, resume_from=f"{self.identity['id']}?_step={limit - 1}"
                )
                self.sent = self.run.step
                if not 0 <= self.sent <= limit:
                    raise ValueError("W&B did not rewind to the committed history cursor")
            except Exception as error:
                self.finish(exit_code=1)
                raise RuntimeError(
                    "W&B history exceeds the restored checkpoint and rewind failed. "
                    "Training stopped to avoid mixing histories; retry with W&B rewind support. "
                    "No replacement run was created."
                ) from error
        if not isinstance(self.sent, int) or self.sent < 0:
            raise ValueError("Invalid W&B remote history cursor")
        self.identity["rewind_to"] = None
        atomic_json(self.identity, self.path)
        self.run.define_metric("step", hidden=True)
        self.run.define_metric("*", step_metric="step")
        if initialization is not None:
            self.run.summary["initialization"] = initialization

    def publish(self):
        """Call only after saving a checkpoint containing state_dict()."""
        if self.run is None:
            return
        while self.sent < len(self.events):
            event = self.events[self.sent]
            row = evaluation_row(event["data"]) if event["kind"] == "evaluation" else event["data"]
            self.run.log(row, step=self.sent, commit=True)
            self.sent += 1

    def finish(self, exit_code=0):
        if self.run is not None:
            run, self.run = self.run, None
            run.finish(exit_code=exit_code)


def snapshot_figure(snapshot, task, step):
    """Render without a GUI backend; null losses appear as gaps."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(8, 4.5), layout="constrained")
    FigureCanvasAgg(figure)
    left = figure.subplots()
    right = left.twinx()
    points = snapshot["points"]
    lengths = [point["length"] for point in points]
    left.plot(
        lengths,
        [np.nan if p["loss"] is None else p["loss"] for p in points],
        color="tab:blue",
        marker="o",
        markersize=3,
        label="Test loss",
    )
    right.plot(
        lengths,
        [p["accuracy"] for p in points],
        color="tab:orange",
        marker="o",
        markersize=3,
        label="Test accuracy",
    )
    labels = {
        "operand_length": "Digits per operand",
        "input_length": "Input length (symbols)",
        "input_and_machine_size": "Input length and machine parameter m",
    }
    left.set_xlabel(labels[snapshot["axis"]])
    left.set_ylabel(f"Test loss ({snapshot['loss_kind']})", color="tab:blue")
    right.set_ylabel("Joint action accuracy", color="tab:orange")
    left.tick_params(axis="y", labelcolor="tab:blue")
    right.tick_params(axis="y", labelcolor="tab:orange")
    right.set_ylim(0, 1.02)
    left.grid(alpha=0.2)
    left.set_title(f"{task} · training step {step}\nOne random example per length")
    lines = left.lines + right.lines
    left.legend(lines, [line.get_label() for line in lines], loc="best")
    if any(not point["loss_finite"] for point in points):
        figure.text(
            0.5, 0.005, "Nonfinite raw losses are omitted (see evaluation JSON).", ha="center"
        )
    return figure


def log_evaluation(writer, result):
    """Log generalization vs optimizer step and a loss/accuracy image per task."""
    if writer is None:
        return
    writer.log(evaluation_row(result))


def evaluation_row(result):
    """Rebuild W&B plots; retain detailed metadata in the durable results only."""
    import wandb

    step = result["step"]
    row = {"step": step}
    for task, metrics in result["tasks"].items():
        prefix = f"evaluation/{task}"
        for name in ("ulg", "rlg", "generalization"):
            if name in metrics:
                measurement = metrics[name]
                row[f"{prefix}/{name}"] = measurement["length"]
        if "snapshot" in metrics:
            snapshot = metrics["snapshot"]
            figure = snapshot_figure(snapshot, task, step)
            try:
                figure.canvas.draw()
                row[f"{prefix}/snapshot"] = wandb.Image(
                    np.asarray(figure.canvas.buffer_rgba()).copy(),
                    caption=f"{task}: test loss (left) and accuracy (right), step {step}",
                )
            finally:
                figure.clear()
    return row
