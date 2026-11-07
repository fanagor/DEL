"""Deterministic mandatory teaching sets paired to each expert implementation.

When ``data.teaching_set`` is ``True`` (supported only for ``data.source="static"``),
static dataset generation injects the mandatory coverage core paired to the
configured ``(environment, task, variant, radix)`` expert before filling any
remaining ``data.size - len(teaching_set)`` slots with random samples:

- Grid addition (``default``): symmetric <=2-digit ``R^2 + 2R`` core (120 decimal,
  8 binary).
- Grid addition (``new``): symmetric <=3-digit pairs without the redundant
  ``0^m + 0^n`` grid (181 decimal, 21 binary).
- TM addition (``default``): symmetric <=3-digit cover of dotted/undotted tape
  transitions (253 decimal, 19 binary).
- Grid multiplication (``default``): explicit symmetric <=5-digit transition
  and mixed-erasure core of ``R^3 + 6R^2 + R(R-1)/2`` pairs (1645 decimal, 33 binary).
- Grid multiplication (``new``): explicit symmetric <=5-digit transition core
  of ``7R^2`` pairs (700 decimal, 28 binary). Both grid multipliers retain zeros.
- TM multiplication (``default``): symmetric <=3-digit cover including
  leading-zero trim anchors (481 decimal, 21 binary).
- Universal teleporting (``default`` and ``new``): 75 ``(tm, word)`` examples
  (all words length <=3) across the 5 TMs ``parity``, ``shift_left``,
  ``shift_right``, ``scaling_A_m5``, and ``scaling_B_m5``, covering
  state-trit widths ``w in {1, 2, 3}`` and
  100% of reachable universal-controller transitions.
"""

from __future__ import annotations

import itertools
import random
from functools import lru_cache

MOVES = {"L": (-1, 0), "R": (1, 0), "D": (0, 1), "U": (0, -1), "S": (0, 0)}
ALIASES = {"add": "addition", "multiply": "multiplication", "mult": "multiplication"}

TP_TEACHING_SPEC: tuple[tuple[str, int], ...] = (
    ("parity", 3),
    ("shift_left", 3),
    ("shift_right", 3),
    ("scaling_A_m5", 3),
    ("scaling_B_m5", 3),
)


def _check_radix(radix):
    if isinstance(radix, bool) or not isinstance(radix, int) or not 2 <= radix <= 10:
        raise ValueError("Teaching sets support radix 2..10")


def _pair_order(pair):
    return max(map(len, pair)), sum(map(len, pair)), pair


def _symmetric(pairs):
    return tuple(sorted({p for a, b in pairs for p in ((a, b), (b, a))}, key=_pair_order))


def _simulate_features(transitions, halt, row_tokens):
    """Fast sparse-grid/tape execution returning visited rules, loop exits, and PTO steps."""
    cells = {(x, 0): tok for x, tok in enumerate(row_tokens) if tok}
    hx = hy = 0
    state = 0
    prev_st = prev_succ = -1
    prev_read = None
    prev_run = 0
    features = set()
    steps = 0
    for _ in range(300_000):
        read = cells.get((hx, hy), "")
        succ, written, move = transitions[state, read]
        features.add(("transition", state, read))
        same = state == prev_st == prev_succ
        if same and read != prev_read:
            features.add(("loop_exit", state, 3 if prev_run >= 3 else prev_run))
        prev_run = prev_run + 1 if (same and read == prev_read) else 1
        prev_st = state
        prev_read = read
        prev_succ = succ
        if written:
            cells[hx, hy] = written
        else:
            cells.pop((hx, hy), None)
        dx, dy = MOVES[move]
        hx += dx
        hy += dy
        state = succ
        steps += 1
        if state == halt:
            return frozenset(features), steps
    raise RuntimeError(f"Teaching witness did not halt: {row_tokens}")


def multiplication_features(expert, pair):
    """Visited rules plus exits after 1, 2, or >=3 identical self-loop reads."""
    transitions, halt = expert.transitions, expert.halt_state
    a, b = pair
    features = set()
    for x1, x2 in ((a, b), (b, a)) if a != b else ((a, b),):
        fs, _ = _simulate_features(transitions, halt, list(x1 + "x" + x2))
        features |= fs
    return frozenset(features)


def _greedy_symmetric_cover(expert, operator, anchors, pool, *, prefer_short_pto: bool = True):
    """Deterministic greedy cover followed by reverse pruning."""
    transitions, halt = expert.transitions, expert.halt_state
    features_list = []
    pto_max = []
    for a, b in pool:
        combined = set()
        longest = 0
        for x1, x2 in ((a, b), (b, a)) if a != b else ((a, b),):
            fs, steps = _simulate_features(transitions, halt, list(x1 + operator + x2))
            combined |= fs
            if steps > longest:
                longest = steps
        features_list.append(frozenset(combined))
        pto_max.append(longest)
    target = sorted(set().union(*features_list))
    keys = {feat: i for i, feat in enumerate(target)}
    masks = [sum(1 << keys[feat] for feat in fs) for fs in features_list]
    selected = [i for i, pair in enumerate(pool) if pair in anchors]
    covered = 0
    for i in selected:
        covered |= masks[i]
    full = (1 << len(keys)) - 1
    while covered != full:
        remaining = full ^ covered
        if prefer_short_pto:
            best = max(
                range(len(pool)),
                key=lambda i: (
                    (masks[i] & remaining).bit_count(),
                    -pto_max[i],
                    -sum(map(len, pool[i])),
                    -i,
                ),
            )
        else:
            best = max(
                range(len(pool)),
                key=lambda i: (
                    (masks[i] & remaining).bit_count(),
                    -sum(map(len, pool[i])),
                    -i,
                ),
            )
        selected.append(best)
        covered |= masks[best]
    for i in selected[::-1]:
        if pool[i] in anchors:
            continue
        other = 0
        for j in selected:
            if j != i:
                other |= masks[j]
        if other == full:
            selected.remove(i)
    return _symmetric(pool[i] for i in selected)


@lru_cache(maxsize=None)
def addition_pairs(radix: int, *, environment: str = "grid", variant: str = "default"):
    """Symmetric addition core paired to grid (default <=2 digits, new <=3 digits) or tm."""
    _check_radix(radix)
    if environment not in {"grid", "tm"}:
        raise ValueError(f"Unsupported arithmetic environment: {environment}")
    if variant not in {"default", "new"}:
        raise ValueError(f"Unsupported variant {variant!r} for {environment}")
    if environment == "tm":
        variant = "default"
    h = str(radix - 1)
    if environment == "grid":
        pairs = [(str(a), str(b)) for a in range(radix) for b in range(radix)]
        pairs += [(str(d) + h, "1") for d in range(radix)]
        if variant == "default":
            return _symmetric(pairs)
        pairs += [(str(d) + h + h, "1") for d in range(radix)]
        pairs += [("1", "000"), ("0", "001"), (h * 3, h * 3)]
        for d in range(1, radix):
            ds = str(d)
            pairs.append((ds, ds + "0" + ds))
            pairs.append((ds + h, str(radix - d) + "0" + ds))
        return _symmetric(pairs)

    from .experts import create_expert

    expert = create_expert({"environment": "tm", "task": "addition", "radix": radix})
    digits = list(map(str, range(radix)))
    anchors = set()
    pool_set = set()

    def add_anchor(a, b):
        p = tuple(sorted((a, b)))
        anchors.add(p)
        pool_set.add(p)

    def add(a, b):
        pool_set.add(tuple(sorted((a, b))))

    for a in range(radix):
        for b in range(radix):
            add_anchor(str(a), str(b))
    for d in range(radix):
        add_anchor(str(d) + h, "1")
        add_anchor(str(d) + h + h, "1")
    for d in range(1, radix):
        add_anchor("0", str(d) + "0")
    add_anchor("1", "000")
    add_anchor("0", "001")
    add_anchor(h * 3, h * 3)
    words2 = ["".join(w) for w in itertools.product(digits, repeat=2)]
    for a, b in itertools.combinations_with_replacement(words2, 2):
        add(a, b)
    for d in digits:
        add("0", d + "00")
        add(d, "0" + h + h)
    pool = tuple(sorted(pool_set, key=_pair_order))
    return _greedy_symmetric_cover(expert, "+", anchors, pool)


def _grid_multiplication_pairs(radix: int, variant: str):
    """Algebraic cores: R^3+6R^2+R(R-1)/2 default pairs, 7R^2 new pairs.

    Concatenation denotes digit words and h=R-1. The shared, disjoint families
    are (a^k,b^k), k=1..5; (ab,11), a!=b; and (hbhhh,11), with both orders.
    The default controller also needs (a s,f), where s=ceil(R*c/f), for every
    factor 1<=f<R, carry 0<=c<f, and digit a. Reading s first establishes c.
    Squares (abba,abba), a<b, exercise both mixed later-erasure contexts
    (factor a, read b) and (factor b, read a), beyond the first prefix erasure.
    No simulation, random witness pool, or greedy selection is needed.
    """
    digits = list(map(str, range(radix)))
    h = digits[-1]

    # 5R^2 ordered pairs: arithmetic and scans at each length from one to five.
    pairs = [(a * k, b * k) for k in range(1, 6) for a in digits for b in digits]
    # 2R(R-1) after symmetrization: mixed adjacent digits; equal digits are above.
    pairs += [(a + b, "11") for a in digits for b in digits if a != b]
    # 2R: generate and propagate carry through h, then encounter each digit b.
    pairs += [(h + b + h * 3, "11") for b in digits]

    if variant == "default":
        # R^2(R-1) after symmetrization: each (factor, carry, digit) transition.
        for factor in range(1, radix):
            for carry in range(factor):
                suffix = str((radix * carry + factor - 1) // factor)
                pairs += [(a + suffix, str(factor)) for a in digits]
        # One four-digit square per unordered digit pair covers both directions.
        # Equal operands preserve symmetry; strings retain leading zeros.
        for a in range(radix):
            for b in range(a + 1, radix):
                word = f"{a}{b}{b}{a}"
                pairs.append((word, word))
    return _symmetric(pairs)


def _tm_multiplication_anchors(radix: int):
    """Behavioral anchors for the canonical TM multiplier, including trimming."""
    h = str(radix - 1)
    pairs = [(str(a), str(b)) for a in range(radix) for b in range(radix)]
    pairs += [("0" * m, "0" * n) for m in range(1, 4) for n in range(1, 4)]
    pairs += [("000", str(d) * 3) for d in range(1, radix)]
    pairs += [("1", str(d) * 3) for d in range(1, radix)]
    pairs += [("101", h * 3)]
    pairs += [(h * 2, h * 2), (h * 3, h * 3), ("101", "101")]
    return {tuple(sorted(p)) for p in pairs}


def _tm_multiplication_candidates(radix: int):
    """Bounded TM witness pool; seed 1741+R is fixed for reproducibility."""
    digits = list(map(str, range(radix)))
    pairs = set(_tm_multiplication_anchors(radix))

    def add(a, b):
        pairs.add(tuple(sorted((a, b))))

    for m, n in itertools.product(range(1, 4), repeat=2):
        for a, b in itertools.product(digits, repeat=2):
            add(a * m, b * n)
    for factor in range(1, radix):
        for carry in range(factor):
            suffix = str(next(d for d in range(radix) if factor * d // radix == carry))
            for digit in digits:
                for repeat in (1, 2):
                    add(digit * repeat + suffix, str(factor))
    words2 = ["".join(w) for w in itertools.product(digits, repeat=2)]
    for a, b in itertools.combinations_with_replacement(words2, 2):
        add(a, b)
    for zeros in (1, 2):
        for d in range(1, radix):
            add("0" * zeros + "1", str(d))
            add("0" * zeros + str(d), "1")
    rng = random.Random(1741 + radix)
    for _ in range(512):
        add(
            "".join(str(rng.randrange(radix)) for _ in range(rng.randint(1, 3))),
            "".join(str(rng.randrange(radix)) for _ in range(rng.randint(1, 3))),
        )
    return tuple(sorted(pairs, key=_pair_order))


@lru_cache(maxsize=None)
def multiplication_pairs(radix: int, *, environment: str = "grid", variant: str = "default"):
    """Symmetric multiplication core: explicit <=5-digit grid sets or a <=3-digit TM cover."""
    _check_radix(radix)
    if environment not in {"grid", "tm"}:
        raise ValueError(f"Unsupported arithmetic environment: {environment}")
    if variant not in {"default", "new"}:
        raise ValueError(f"Unsupported variant {variant!r} for {environment}")
    if environment == "grid":
        return _grid_multiplication_pairs(radix, variant)
    variant = "default"
    from .experts import create_expert

    expert = create_expert(
        {
            "environment": environment,
            "variant": variant,
            "task": "multiplication",
            "radix": radix,
            "max_steps": 300_000,
        }
    )
    anchors = _tm_multiplication_anchors(radix)
    pool = _tm_multiplication_candidates(radix)
    return _greedy_symmetric_cover(expert, "x", anchors, pool)


@lru_cache(maxsize=1)
def _tp_pairs() -> tuple[tuple[str, str], ...]:
    examples = []
    for task_name, max_len in TP_TEACHING_SPEC:
        for n in range(max_len + 1):
            for bits in itertools.product("01", repeat=n):
                examples.append((task_name, "".join(bits)))
    return tuple(examples)


def _raw_examples(config):
    environment = config.get("environment", "grid")
    variant = config.get("variant", "default")
    if environment == "teleporting":
        if variant not in {"default", "new"}:
            raise ValueError(f"Unsupported variant {variant!r} for teleporting")
        return _tp_pairs()
    task = ALIASES.get(config.get("task", "addition"), config.get("task", "addition"))
    if task not in {"addition", "multiplication"}:
        raise ValueError(f"Unsupported arithmetic task: {task}")
    radix = config.get("radix", 2)
    builder = addition_pairs if task == "addition" else multiplication_pairs
    return builder(radix, environment=environment, variant=variant)


def teaching_examples(config):
    """Return the complete teaching core for an expert configuration.

    Extraction can append this core independently of a training dataset's size,
    length budget, source mode, or teaching_set flag. Training dataset builders
    retain validate_teaching and the corresponding capacity checks.
    """
    return _raw_examples(config)


def _required_max_length(environment: str, variant: str, task: str) -> int:
    if environment == "grid":
        if task == "multiplication":
            return 5
        if variant == "default":
            return 2
    return 3


def validate_teaching(config):
    enabled = config.get("teaching_set", False)
    if not isinstance(enabled, bool):
        raise ValueError("data.teaching_set must be a boolean")
    if not enabled:
        return
    if config.get("source", "static") != "static":
        raise ValueError("data.teaching_set=true requires data.source='static'")
    environment = config.get("environment", "grid")
    if environment not in {"grid", "tm", "teleporting"}:
        raise ValueError(f"Unknown environment: {environment}")
    variant = config.get("variant", "default")
    if variant not in {"default", "new"}:
        raise ValueError(f"Unsupported variant {variant!r} for {environment}")
    if environment != "grid":
        variant = "default"
    task = ALIASES.get(config.get("task", "addition"), config.get("task", "addition"))
    if environment != "teleporting" and task not in {"addition", "multiplication"}:
        raise ValueError(f"Unsupported arithmetic task: {task}")
    min_length = _required_max_length(environment, variant, task)
    maximum = config.get("max_length", 3)
    if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum < min_length:
        raise ValueError(f"data.teaching_set=true requires data.max_length >= {min_length}")
    required = len(_raw_examples(config))
    size = config.get("size", 100)
    if isinstance(size, bool) or not isinstance(size, int) or size < required:
        raise ValueError(f"data.size={size} is smaller than the teaching set size ({required})")


def teaching_set_size(config) -> int:
    if not config.get("teaching_set", False):
        return 0
    return len(_raw_examples(config))


def arithmetic_teaching_pairs(config):
    validate_teaching(config)
    if not config.get("teaching_set", False) or config.get("environment", "grid") == "teleporting":
        return ()
    return _raw_examples(config)


def teleporting_teaching_inputs(config):
    validate_teaching(config)
    if not config.get("teaching_set", False) or config.get("environment", "grid") != "teleporting":
        return ()
    return _raw_examples(config)


def teaching_manifest(config):
    """Reviewable mandatory teaching set and expert transition coverage."""
    from .experts import create_expert, task_transitions

    validate_teaching(config)
    enabled = config.get("teaching_set", False)
    if not enabled:
        return {"teaching_set": False, "count": 0, "examples": []}
    expert = create_expert(config)
    covered = set()
    if expert.environment == "teleporting":
        pairs = teleporting_teaching_inputs(config)
        examples = [{"task": t, "input": w} for t, w in pairs]
        for t, w in pairs:
            rows = expert.encode_program(task_transitions(t), w)
            execution = expert.simulate(rows, (1, 1 - len(rows)))
            covered.update(
                (q, expert.vocab_rev[x])
                for q, x in zip(execution.states, execution.trace.inputs.tolist())
            )
    else:
        pairs = arithmetic_teaching_pairs(config)
        examples = [{"operands": list(pair)} for pair in pairs]
        operator = "+" if expert.task == "addition" else "x"
        for a, b in pairs:
            execution = expert.simulate([list(a + operator + b)], (0, 0))
            covered.update(
                (q, expert.vocab_rev[x])
                for q, x in zip(execution.states, execution.trace.inputs.tolist())
            )
    return {
        "teaching_set": True,
        "environment": expert.environment,
        "variant": expert.variant,
        "task": "universal" if expert.environment == "teleporting" else expert.task,
        "count": len(examples),
        "covered_transitions": len(covered),
        "expert_states": expert.n_states,
        "transition_table_entries": len(expert.transitions),
        "examples": examples,
    }
