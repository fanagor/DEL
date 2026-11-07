"""Live, autonomous controller execution; never substitute expert predictions."""

import re
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import torch

from ..config import load_config
from ..data import create_expert
from ..data.experts import DEFAULT_TP_TASKS, Expert, Memory, task_label, task_transitions
from ..training import choose_device, construct_model, load_run


@dataclass
class ControllerSource:
    expert: Expert
    label: str
    model: torch.nn.Module | None = None
    task: str | None = None
    binary_tape: bool = False

    def tasks(self):
        if self.expert.environment != "teleporting":
            return []
        return list(
            dict.fromkeys(
                [self.task, *self.expert.train_tasks, *self.expert.eval_tasks, *DEFAULT_TP_TASKS]
            )
        )


def _from_config(config, model=None, task=None, label=None):
    expert = create_expert(config["data"])
    if expert.environment == "teleporting":
        task = task_label(task or expert.train_tasks[0])
    elif task is not None:
        raise ValueError("--task selects a program only for a teleporting controller")
    kind = config["model"]["name"] if model is not None else "Expert"
    return ControllerSource(expert, label or f"{kind} · {expert.environment}", model, task)


def load_controller(controller, *, config=None, task=None, device="cpu"):
    """Accept an Expert/model object, builtin selector, recipe, run, or weights.

    Bare model objects and state dictionaries require the complete experiment
    configuration. Run checkpoints carry architecture/environment metadata.
    """
    if isinstance(config, (str, Path)):
        config = load_config(config)
    if isinstance(controller, Expert):
        selected = (
            task_label(task or controller.train_tasks[0])
            if (controller.environment == "teleporting")
            else None
        )
        return ControllerSource(controller, f"Expert · {controller.environment}", task=selected)
    if isinstance(controller, torch.nn.Module):
        if config is None:
            raise ValueError(
                "A model object requires config with the model and environment settings"
            )
        model = deepcopy(controller).to(choose_device(device)).eval()
        return _from_config(config, model, task)
    if isinstance(controller, dict):
        if config is None:
            raise ValueError("Raw weights require --config with the architecture and environment")
        source = _from_config(config, task=task)
        source.model = construct_model(config, source.expert, choose_device(device)).eval()
        source.model.load_state_dict(controller, strict=True)
        source.label = f"{config['model']['name']} · {source.expert.environment}"
        return source
    name = str(controller)
    match = re.fullmatch(r"(grid|grid_new|tm):(add|mult)([2-9]|10)", name)
    if match:
        environment, operation, radix = match.groups()
        variant = "new" if environment == "grid_new" else "default"
        env_name = "grid" if environment == "grid_new" else environment
        recipe = load_config(
            overrides=[
                f"data.environment={env_name}",
                f"data.variant={variant}",
                f"data.task={'addition' if operation == 'add' else 'multiplication'}",
                f"data.radix={radix}",
                "data.source=online",
                "data.teaching_set=false",
            ]
        )
        return _from_config(recipe, task=task, label=f"{operation}{radix} · {environment} expert")
    if name in {"tp", "tp_agent", "universal_tp"} or name.startswith("tp:"):
        selected = task or (name[3:] if name.startswith("tp:") else "parity")
        recipe = load_config(
            overrides=[
                "data.environment=teleporting",
                "data.source=online",
                "data.teaching_set=false",
            ]
        )
        return _from_config(recipe, task=selected, label="Universal · teleporting expert")
    if name.startswith("tm:"):
        selected = task_label(name[3:])
        # The binary program catalog also runs directly on a one-dimensional tape.
        expert = create_expert({"environment": "tm", "task": "addition", "radix": 2})
        expert.transitions = task_transitions(selected)
        expert.n_states = 1 + max(q for q, _ in expert.transitions)
        expert.halt_state = expert.n_states - 1
        # Arithmetic TM alphabets contain an operator; replace the entire codec.
        expert.alphabet = ["", "0", "1"]
        expert.vocab = {s: i for i, s in enumerate(expert.alphabet)}
        expert.vocab_rev = dict(enumerate(expert.alphabet))
        expert.input_size = expert.symbol_size = 3
        return ControllerSource(
            expert, f"{selected} · tape expert", task=selected, binary_tape=True
        )
    path = Path(name).expanduser()
    if not path.exists():
        raise ValueError(
            "Choose grid:add2, grid:mult10, tm:add2, tm:parity, tp:copy, a recipe, or a checkpoint"
        )
    if path.suffix in {".yaml", ".yml"}:
        return _from_config(load_config(path), task=task)
    if config is not None:
        # Tensor-only weights use the restricted loader. Full run checkpoints use
        # the repository's existing trusted-local-checkpoint loading contract.
        weights = torch.load(path, map_location="cpu", weights_only=True)
        return load_controller(weights, config=config, task=task, device=device)
    model, expert, saved, _ = load_run(path, device=device)
    return _from_config(saved, model, task)


def initial_memory(source, text, task=None):
    """Parse one editable input without executing any controller transitions."""
    if not isinstance(text, str) or len(text) > 4096:
        raise ValueError("Input must contain at most 4096 characters")
    text = "".join(text.split())
    expert = source.expert
    if expert.environment == "teleporting" or source.binary_tape:
        if any(s not in "01" for s in text):
            raise ValueError(
                "Enter a binary string containing only 0 and 1; empty input also works"
            )
        selected = task_label(task or source.task)
        if expert.environment == "teleporting":
            rows = expert.encode_program(task_transitions(selected), text)
            return Memory(rows, (1, 1 - len(rows)), True), 1 - len(rows), selected
        return Memory([list(text)], (0, 0), False), 0, selected
    operator = "+" if expert.task == "addition" else "x"
    text = text.replace("×", "x").replace("*", "x").lower()
    parts = text.split(operator)
    if len(parts) != 2 or any(not p or not p.isascii() or not p.isdigit() for p in parts):
        raise ValueError(f"Enter two base-{expert.radix} operands separated by {operator}")
    if any(int(d) >= expert.radix for part in parts for d in part):
        raise ValueError(f"Base {expert.radix} accepts digits 0–{expert.radix - 1}")
    return Memory([list(text)], (0, 0), False), 0, expert.task


def _finite(state):
    if isinstance(state, (tuple, list)):
        return all(_finite(s) for s in state)
    return state is None or bool(torch.isfinite(state).all())


class Session:
    """A bounded execution with sparse cell deltas and opaque recurrent state."""

    def __init__(self, source, text, *, task=None, max_steps=100000):
        if not isinstance(max_steps, int) or isinstance(max_steps, bool) or max_steps < 1:
            raise ValueError("max_steps must be a positive integer")
        self.source = source
        self.memory, self.tape_row, self.task = initial_memory(source, text, task)
        self.max_steps, self.step, self.state = max_steps, 0, None
        self.expert_state = 0
        self.status, self.message = "ready", "Ready"
        self.tensors = None
        model = source.model
        if model is not None and model.supports_state_rescaling:
            with torch.inference_mode():
                self.tensors = tuple(t.detach() for t in model.effective_tensors())
                self.state = self._normalize(self.tensors[-1].clone())

    @staticmethod
    def _normalize(state):
        scale = state.abs().amax(dim=-1, keepdim=True)
        return state / torch.where(scale > 0, scale, torch.ones_like(scale))

    def position(self, point):
        x, y = point
        return [x, -y if self.memory.teleporting else y]

    def runes(self):
        return {
            r: {"position": self.position(p), "under": s} for r, (p, s) in self.memory.runes.items()
        }

    def snapshot(self):
        return {
            "cells": [[*self.position(p), s] for p, s in self.memory.cells.items()],
            "head": self.position(self.memory.head),
            "runes": self.runes(),
            "step": self.step,
            "state": self.expert_state if self.source.model is None else None,
            "status": self.status,
            "message": self.message,
            "tape_row": self.position((0, self.tape_row))[1],
            "task": self.task,
        }

    @torch.inference_mode()
    def advance(self):
        if self.status in {"halted", "limit", "error"}:
            return None
        expert, model = self.source.expert, self.source.model
        read, before = self.memory.read(), self.memory.head
        try:
            if model is None:
                successor, written, motion = expert.transitions[self.expert_state, read]
                self.expert_state = successor
                halted = successor == expert.halt_state
            else:
                token = expert.vocab[read]
                if self.tensors is not None:
                    A, B, C, _ = self.tensors
                    symbols, motions = B[token] @ self.state, C[token] @ self.state
                    self.state = self._normalize(A[token] @ self.state)
                else:
                    device = next(model.parameters()).device
                    inputs = torch.tensor([[token]], dtype=torch.long, device=device)
                    symbols, motions, self.state = model(inputs, self.state)
                if not _finite(self.state) or not _finite(symbols) or not _finite(motions):
                    raise ValueError("Non-finite model output; execution stopped before writing")
                written = expert.vocab_rev[int(symbols.reshape(-1).argmax())]
                motion = expert.motion_vocab_rev[int(motions.reshape(-1).argmax())]
                halted = motion == "S"
            affected = {before}
            if written in self.memory.runes:
                affected.add(self.memory.runes[written][0])
            previous = {p: self.memory.cells.get(p, "") for p in affected}
            self.memory.write(written)
            jump = None
            if self.memory.teleporting and motion in "XYZ":
                jump = {"rune": motion, "available": motion in self.memory.runes}
            self.memory.move(motion)
            self.step += 1
            self.status = (
                "halted" if halted else "limit" if self.step >= self.max_steps else "running"
            )
            self.message = {
                "halted": "Halted",
                "limit": "Step limit reached",
                "running": "Running",
            }[self.status]
            return {
                "step": self.step,
                "from": self.position(before),
                "head": self.position(self.memory.head),
                "read": read,
                "write": written,
                "motion": motion,
                "jump": jump,
                "runes": self.runes(),
                "patches": [
                    [*self.position(p), self.memory.cells.get(p, "")]
                    for p in sorted(affected)
                    if previous[p] != self.memory.cells.get(p, "")
                ],
                "state": self.expert_state if model is None else None,
                "status": self.status,
                "message": self.message,
            }
        except (KeyError, ValueError, RuntimeError) as error:
            self.status, self.message = "error", str(error)
            return {
                "step": self.step,
                "head": self.position(before),
                "status": "error",
                "message": self.message,
                "patches": [],
                "runes": self.runes(),
            }
