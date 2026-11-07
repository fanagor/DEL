"""Guarded expert execution and reproducible policy-trajectory observations.

Transition tables retain the research implementation's state and vocabulary
ordering. A trace records each executed transition exactly once: unlike the old
``fast_input_output`` helpers, no synthetic duplicate halt action is appended.
"""

from __future__ import annotations

import random
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from itertools import count
from typing import Any, Mapping, Sequence

import torch

from .transitions import teleporting as tp
from .transitions.grid import add_grid, add_grid_new, mult_grid, mult_grid_new
from .transitions.tm import add_tm, mult_tm


@dataclass
class Trace:
    inputs: torch.Tensor
    symbols: torch.Tensor
    motions: torch.Tensor
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        tensors = (self.inputs, self.symbols, self.motions)
        if any(t.ndim != 1 or t.dtype != torch.long or t.device.type != "cpu" for t in tensors):
            raise ValueError("Trace tensors must be one-dimensional CPU torch.long tensors")
        if len(self.inputs) == 0 or len({len(t) for t in tensors}) != 1:
            raise ValueError("A trace must have equally sized, nonempty tensors")

    def __len__(self):
        return len(self.inputs)


@dataclass
class Batch:
    inputs: torch.Tensor
    symbols: torch.Tensor
    motions: torch.Tensor
    mask: torch.Tensor
    metadata: list[dict[str, Any]] = field(default_factory=list)

    def to(self, device):
        return Batch(
            *(t.to(device) for t in (self.inputs, self.symbols, self.motions, self.mask)),
            self.metadata,
        )


def pad_traces(traces: Sequence[Trace]) -> Batch:
    """Last-value padding, excluded from loss/evaluation by the boolean mask."""
    if not traces:
        raise ValueError("Cannot pad an empty trace collection")
    length = max(map(len, traces))
    tensors = [torch.empty(len(traces), length, dtype=torch.long) for _ in range(3)]
    mask = torch.zeros(len(traces), length, dtype=torch.bool)
    for row, trace in enumerate(traces):
        for output, values in zip(tensors, (trace.inputs, trace.symbols, trace.motions)):
            output[row].fill_(values[-1].item())
            output[row, : len(trace)] = values
        mask[row, : len(trace)] = True
    return Batch(*tensors, mask, [t.metadata for t in traces])


class StepLimitExceeded(RuntimeError):
    """Expert execution exceeded its explicit finite step budget."""


@dataclass
class Execution:
    trace: Trace
    states: list[int]
    cells: dict[tuple[int, int], str]
    head: tuple[int, int]


class Memory:
    def __init__(self, rows, attention, teleporting):
        self.cells = {
            (x, -y): str(value)
            for y, row in enumerate(rows)
            for x, value in enumerate(row)
            if str(value) != ""
        }
        self.head = tuple(attention)
        self.teleporting = teleporting
        self.runes = {}
        if teleporting:
            for pos, value in self.cells.items():
                if value in "XYZ":
                    if value in self.runes:
                        raise ValueError(f"Duplicate initial rune {value}")
                    self.runes[value] = (pos, "")

    def read(self):
        return self.cells.get(self.head, "")

    def write(self, value):
        if self.teleporting and value in {"X", "Y", "Z"}:
            if value in self.runes:
                old, hidden = self.runes.pop(value)
                if hidden:
                    self.cells[old] = hidden
                else:
                    self.cells.pop(old, None)
            current = self.read()
            if current in self.runes:
                self.runes.pop(current)
            self.runes[value] = (self.head, self.read())
            self.cells[self.head] = value
        else:
            old = self.cells.pop(self.head, None)
            self.runes.pop(old, None)
            if value:
                self.cells[self.head] = value

    def move(self, motion):
        dx, dy = {"L": (-1, 0), "R": (1, 0), "D": (0, 1), "U": (0, -1), "S": (0, 0)}.get(
            motion, (0, 0)
        )
        # The source TP encoding uses upward-positive coordinates; the arithmetic
        # grid simulator uses downward-positive coordinates. Preserve both.
        if self.teleporting:
            dy = -dy
        if motion in self.runes:
            self.head = self.runes[motion][0]
        else:
            self.head = (self.head[0] + dx, self.head[1] + dy)


TASK_BUILDERS = {
    "parity": tp.parity_tm,
    "flip": tp.flip_tm,
    "cum_sum": tp.cum_sum_tm,
    "bit_delete": tp.bit_delete_tm,
    "shift_left": tp.shift_left_tm,
    "shift_right": tp.shift_right_tm,
    "reverse": tp.reverse_tm,
    "copy": tp.copy_tm,
}
DEFAULT_TP_TASKS = [*TASK_BUILDERS, "scaling_A_m5", "scaling_B_m5"]


def task_label(task) -> str:
    if isinstance(task, Mapping):
        name = str(task.get("name", ""))
        if name.lower() in {"scaling_a", "scaling_b"}:
            m = task.get("m")
            if isinstance(m, bool) or not isinstance(m, int) or m < 1:
                raise ValueError("Scaling tasks require a positive integer m")
            return f"scaling_{name[-1].upper()}_m{m}"
        if name.lower() in TASK_BUILDERS and set(task) == {"name"}:
            return name.lower()
        raise ValueError(f"Unknown teleporting task specification: {task!r}")
    if not isinstance(task, str):
        raise ValueError(f"Invalid teleporting task: {task!r}")
    name = task.lower()
    if name in TASK_BUILDERS:
        return name
    match = re.fullmatch(r"scaling_([ab])_m([1-9][0-9]*)", name)
    if match:
        return f"scaling_{match[1].upper()}_m{int(match[2])}"
    raise ValueError(f"Unknown teleporting task: {task!r}")


def task_transitions(task):
    name = task_label(task)
    if name in TASK_BUILDERS:
        return TASK_BUILDERS[name]()
    family, count = name.rsplit("_m", 1)
    return (tp.scaling_A_tm if family == "scaling_A" else tp.scaling_B_tm)(int(count))


def encode_tm(transitions, word: Sequence[str]):
    """No-rewind ternary transition-table encoding paired with ``universal_tp``.

    Each table row has layout ``[symbol, '', *state_trits, '0', *next_trits, '0', written, motion]``
    with initial rune ``'X'`` at ``(1, 0)``. Placing ``symbol`` at column 0 and
    delimiters after both state fields eliminates the 3-state silent rewind walk.
    """
    states = sorted({state for state, _ in transitions})
    alphabet = sorted({symbol for _, symbol in transitions})
    if states != list(range(len(states))) or not states:
        raise ValueError("Encoded TM states must be contiguous from zero")
    if set(alphabet) != {"", "0", "1"}:
        raise ValueError("Universal tasks must use the binary tape alphabet")
    delim = alphabet[1]
    width = 1
    while 3**width < len(states):
        width += 1

    def state_digits(state):
        digits = ["L"] * width
        for i in range(width - 1, -1, -1):
            digits[i] = "LRS"[state % 3]
            state //= 3
        return digits

    rows = []
    for state in states:
        for symbol in alphabet:
            successor, written, motion = transitions[(state, symbol)]
            rows.append(
                [symbol, ""]
                + state_digits(state)
                + [delim]
                + state_digits(successor)
                + [delim, written, motion]
            )
    rows[0][1] = "X"
    rows.append([""] + list(word))
    return rows


class Expert:
    def __init__(self, config: Mapping[str, Any]):
        self.config = dict(config)
        self.environment = config.get("environment", "grid")
        raw_variant = config.get("variant", "default")
        self.radix = int(config.get("radix", 2))
        self.max_steps = config.get("max_steps")
        if self.max_steps is not None and (
            isinstance(self.max_steps, bool)
            or not isinstance(self.max_steps, int)
            or self.max_steps < 1
        ):
            raise ValueError("max_steps must be null or a positive integer")
        if self.environment not in {"grid", "tm", "teleporting"}:
            raise ValueError(f"Unknown environment: {self.environment}")
        if raw_variant not in {"default", "new"}:
            raise ValueError(f"Unknown expert variant: {raw_variant}")
        self.variant = raw_variant if self.environment == "grid" else "default"
        self.task = config.get("task", "addition")
        self._task_cache = {}
        if self.environment == "teleporting":
            self.train_tasks = [task_label(t) for t in config.get("train_tasks", DEFAULT_TP_TASKS)]
            self.eval_tasks = [task_label(t) for t in config.get("eval_tasks", DEFAULT_TP_TASKS)]
            if not self.train_tasks or not self.eval_tasks:
                raise ValueError("train_tasks and eval_tasks must be nonempty")
            self.transitions = tp.universal_tp({"", "0", "1"})
            motions = "LRDUXYZS"
        else:
            if self.radix < 2:
                raise ValueError("radix must be at least two")
            aliases = {"add": "addition", "multiply": "multiplication", "mult": "multiplication"}
            self.task = aliases.get(self.task, self.task)
            if self.task not in {"addition", "multiplication"}:
                raise ValueError(f"Unknown arithmetic task: {self.task}")
            if self.environment == "grid":
                builder = (
                    {"addition": add_grid_new, "multiplication": mult_grid_new}
                    if self.variant == "new"
                    else {"addition": add_grid, "multiplication": mult_grid}
                )[self.task]
            else:
                builder = {"addition": add_tm, "multiplication": mult_tm}[self.task]
            revision = config.get("expert_revision")
            if revision is not None:
                if (revision, self.environment, self.variant, self.task, self.radix) != (
                    "grid_20261020", "grid", "default", "multiplication", 10
                ):
                    raise ValueError("Unsupported historical expert revision")
                from .transitions.grid_20261020 import mult_grid as builder
            self.transitions = builder(self.radix)
            motions = "LRDUS" if self.environment == "grid" else "LRS"
        states = {s for s, _ in self.transitions} | {v[0] for v in self.transitions.values()}
        if states != set(range(len(states))):
            raise ValueError("Expert states must be contiguous from zero")
        self.n_states = len(states)
        self.halt_state = self.n_states - 1
        self.alphabet = sorted(
            {x for _, x in self.transitions} | {v[1] for v in self.transitions.values()}
        )
        self.vocab = {x: i for i, x in enumerate(self.alphabet)}
        self.vocab_rev = dict(enumerate(self.alphabet))
        self.motion_vocab = {x: i for i, x in enumerate(motions)}
        self.motion_vocab_rev = dict(enumerate(motions))
        self.input_size = self.symbol_size = len(self.alphabet)
        self.motion_size = len(motions)
        for state in states:
            for symbol in self.alphabet:
                if (state, symbol) not in self.transitions:
                    raise ValueError(f"Incomplete expert transition table at {(state, symbol)}")

    def get_tensors(self):
        """Exact column-oriented transition/output tensors (A, B, C)."""
        A = torch.zeros(self.input_size, self.n_states, self.n_states)
        B = torch.zeros(self.input_size, self.symbol_size, self.n_states)
        C = torch.zeros(self.input_size, self.motion_size, self.n_states)
        for (state, read), (successor, write, move) in self.transitions.items():
            x = self.vocab[read]
            A[x, successor, state] = 1
            B[x, self.vocab[write], state] = 1
            C[x, self.motion_vocab[move], state] = 1
        return A, B, C

    def simulate(
        self, input_rows, initial_attention=(0, 0), metadata=None, max_steps=None
    ) -> Execution:
        """Run until halt, returning actions and final sparse memory.

        An omitted per-call limit inherits the expert's configured max_steps;
        a configured None allows unlimited transitions.
        """
        memory = self._memory(input_rows, initial_attention)
        xs, ss, ms, states = [], [], [], [0]
        for x, symbol, motion, successor in self._steps(memory, max_steps):
            xs.append(x)
            ss.append(symbol)
            ms.append(motion)
            states.append(successor)
        trace = Trace(
            *(torch.tensor(values, dtype=torch.long) for values in (xs, ss, ms)),
            metadata or {},
        )
        return Execution(trace, states, dict(memory.cells), memory.head)

    def iter_simulate(
        self, input_rows, initial_attention=(0, 0), metadata=None, max_steps=None, *, chunk_size=256
    ) -> Iterator[Trace]:
        """Yield bounded PTO chunks, retaining only the current memory and state.

        Each transition, including halt, appears exactly once. A finite limit
        applies to the entire execution, not separately to each chunk.
        """
        if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1:
            raise ValueError("chunk_size must be a positive integer")
        memory = self._memory(input_rows, initial_attention)
        xs, ss, ms = [], [], []
        for x, symbol, motion, _ in self._steps(memory, max_steps):
            xs.append(x)
            ss.append(symbol)
            ms.append(motion)
            if len(xs) == chunk_size:
                yield Trace(
                    *(torch.tensor(values, dtype=torch.long) for values in (xs, ss, ms)),
                    metadata or {},
                )
                xs, ss, ms = [], [], []
        if xs:
            yield Trace(
                *(torch.tensor(values, dtype=torch.long) for values in (xs, ss, ms)),
                metadata or {},
            )

    def _memory(self, input_rows, initial_attention):
        if self.environment == "tm" and (not input_rows or isinstance(input_rows[0], str)):
            input_rows = [input_rows]
        return Memory(input_rows, initial_attention, self.environment == "teleporting")

    def _steps(self, memory, max_steps):
        limit = self.max_steps if max_steps is None else max_steps
        if limit is not None and (
            isinstance(limit, bool) or not isinstance(limit, int) or limit < 1
        ):
            raise ValueError("max_steps must be null or a positive integer")
        state = 0
        for _ in count() if limit is None else range(limit):
            read = memory.read()
            try:
                successor, written, motion = self.transitions[(state, read)]
            except KeyError as exc:
                raise ValueError(
                    f"No expert transition for state {state}, symbol {read!r}"
                ) from exc
            memory.write(written)
            memory.move(motion)
            state = successor
            yield self.vocab[read], self.vocab[written], self.motion_vocab[motion], successor
            if successor == self.halt_state:
                return
        raise StepLimitExceeded(
            f"{self.environment} expert did not halt within {limit} transitions"
        )

    def encode_program(self, transitions, word: Sequence[str]):
        """Encode a binary TM program table and input word for ``universal_tp``."""
        return encode_tm(transitions, word)

    def make_environment(self, metadata: Mapping[str, Any]) -> Memory:
        """Reconstruct a mutable environment for independent analysis policies.

        This constructs memory only; learned-controller rollout belongs to the
        analysis layer. Call read/write/move with decoded symbol/motion strings.
        """
        if self.environment == "teleporting":
            rows = self.encode_program(task_transitions(metadata["task"]), metadata["input"])
            return Memory(rows, (1, 1 - len(rows)), True)
        a, b = metadata["operand_tokens"]
        row = a + ["+" if self.task == "addition" else "x"] + b
        return Memory([row], (0, 0), False)

    def _digits(self, operand):
        if isinstance(operand, int):
            if operand < 0:
                raise ValueError("Operands must be nonnegative")
            tokens = []
            while operand:
                tokens.append(str(operand % self.radix))
                operand //= self.radix
            return list(reversed(tokens)) or ["0"]
        tokens = list(operand) if isinstance(operand, str) else [str(d) for d in operand]
        if not tokens or any(
            d not in self.vocab or not d.isdigit() or int(d) >= self.radix for d in tokens
        ):
            raise ValueError(f"Invalid radix-{self.radix} operand: {operand!r}")
        return tokens

    def sample(
        self, rng: random.Random, max_length: int, task=None, lengths=None, operands=None
    ) -> Trace:
        return self.simulate(*self._sample_input(rng, max_length, task, lengths, operands)).trace

    def sample_stream(
        self,
        rng: random.Random,
        max_length: int,
        task=None,
        lengths=None,
        operands=None,
        *,
        chunk_size=256,
    ) -> Iterator[Trace]:
        """Sample the same input as sample(), then execute lazily in bounded chunks.

        Input sampling consumes RNG draws immediately; advancing a stream uses
        no RNG, so interleaving executions never changes subsequent samples.
        """
        return self.iter_simulate(
            *self._sample_input(rng, max_length, task, lengths, operands), chunk_size=chunk_size
        )

    def _sample_input(self, rng, max_length, task, lengths, operands):
        if not isinstance(rng, random.Random):
            raise TypeError("Provide an explicit random.Random instance")
        if self.environment == "teleporting":
            if max_length < 0:
                raise ValueError("max_length must be nonnegative")
            label = task_label(task) if task is not None else rng.choice(self.train_tasks)
            if label not in self._task_cache:
                self._task_cache[label] = task_transitions(label)
            transitions = self._task_cache[label]
            if operands is not None:
                word = list(operands if isinstance(operands, str) else operands[0])
                if any(d not in {"0", "1"} for d in word):
                    raise ValueError("Teleporting inputs must be binary strings")
            else:
                n = (
                    int(lengths if isinstance(lengths, int) else lengths[0])
                    if lengths is not None
                    else rng.randint(0, max_length)
                )
                if n < 0:
                    raise ValueError("Input length must be nonnegative")
                word = [str(rng.randrange(2)) for _ in range(n)]
            rows = self.encode_program(transitions, word)
            meta = {
                "environment": self.environment,
                "task": label,
                "input": "".join(word),
                "lengths": [len(word)],
                "tape_row": 1 - len(rows),
            }
            return rows, (1, 1 - len(rows)), meta
        if max_length < 1:
            raise ValueError("Arithmetic max_length must be positive")
        if task is not None and task != self.task:
            raise ValueError("Use a separate expert for a different arithmetic task")
        if operands is None:
            if lengths is None:
                lengths = (rng.randint(1, max_length), rng.randint(1, max_length))
            if len(lengths) != 2 or min(lengths) < 1:
                raise ValueError("Arithmetic lengths must be two positive integers")
            operands = [[str(rng.randrange(self.radix)) for _ in range(n)] for n in lengths]
        if len(operands) != 2:
            raise ValueError("Arithmetic requires two operands")
        a, b = map(self._digits, operands)
        row = a + ["+" if self.task == "addition" else "x"] + b
        meta = {
            "environment": self.environment,
            "task": self.task,
            "radix": self.radix,
            "operands": ["".join(a), "".join(b)],
            "operand_tokens": [a, b],
            "lengths": [len(a), len(b)],
        }
        return [row], (0, 0), meta


def create_expert(data_config: Mapping[str, Any]) -> Expert:
    return Expert(data_config)
