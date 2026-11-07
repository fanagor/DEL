"""Teleporting experts and binary Turing-machine programs.

The universal controller uses 15 states for the binary tape alphabet (including
blank). Binary task builders retain the research programs, with the documented
shift-right correction. Source: data/teleporting_agent/tp_agent_tansitions.py.
"""


def universal_tp(tape_alphabet):
    """Build the no-rewind universal controller; 15 states / 135 rules for binary tape.

    Paired with ``encode_tm``, whose program-table rows are ordered
    lexicographically by ``(state_trits, read_symbol)`` with layout:
    ``[read_symbol, blank, *state_trits, delim, *next_state_trits, delim, write_symbol, motion]``,
    where ``delim`` is the smallest nonempty tape symbol (``'0'`` for binary).

    Rune invariant: Z marks the simulated tape head; Y bookmarks the instruction
    field being decoded; X sits at column 1 (covering blank) of a row belonging
    to the current simulated state between simulated steps. Reading the tape
    jumps to X and steps left to column 0 to match ``read_symbol`` vertically
    within the current state's block; a single ``skip_q`` state crosses the
    current state's trits to ``delim``; ``find_trit`` matches ``next_state_trits``;
    and a single ``reset_x`` state rewinds X across ``next_state_trits`` back to
    column 1 upon reaching the second ``delim``, eliminating the 3-state silent
    rewind and ``find_symbol`` trit-overwrite walks.
    """
    symbols = set(tape_alphabet)
    if any(not isinstance(symbol, str) for symbol in symbols):
        raise ValueError("Tape alphabet must contain strings")
    if "" not in symbols or len(symbols) < 2:
        raise ValueError("Tape alphabet must contain blank and a nonempty symbol")
    trits = "LRS"
    reserved = set("XYZ") | set(trits)
    if reserved.intersection(symbols):
        raise ValueError("Tape alphabet cannot contain the reserved symbols X/Y/Z/L/R/S")
    symbols = sorted(symbols)
    delim = symbols[1]
    m = len(symbols)
    alphabet = sorted(set(symbols) | reserved)
    dispatch = 0
    find_symbol = dict(zip(symbols, range(1, m + 1)))
    skip_q = m + 1
    next_trit = m + 2
    find_trit = dict(zip(trits, range(m + 3, m + 6)))
    reset_x = m + 6
    read_write = m + 7
    read_motion = dict(zip(symbols, range(m + 8, 2 * m + 8)))

    # States 1..m+7 only inspect program-table cells and never read Z, so up to
    # m+7 (write_symbol, L/R) pairs share those state IDs unambiguously.
    write_slots = [
        *find_symbol.values(),
        skip_q,
        next_trit,
        *find_trit.values(),
        reset_x,
        read_write,
    ][: 2 * m]
    extra = max(0, 2 * m - len(write_slots))
    write_slots.extend(range(2 * m + 8, 2 * m + 8 + extra))
    halt = 2 * m + 8 + extra
    write_move = dict(zip(((s, move) for s in symbols for move in "LR"), write_slots))
    transitions = {}

    # 1. Read the tape, leave Z, jump to X at column 1, and step left to column 0
    # to select the matching read-symbol row within the current state block.
    for symbol, state in find_symbol.items():
        transitions[dispatch, symbol] = (state, "Z", "X")
        transitions[state, "X"] = (state, "X", "L")
        for observed in symbols:
            transitions[state, observed] = (
                (skip_q, observed, "R")
                if observed == symbol
                else (state, observed, "U" if observed > symbol else "D")
            )

    # 2. Cross column 1 (X or blank) and current-state trits until delim.
    transitions[skip_q, "X"] = (skip_q, "X", "R")
    transitions[skip_q, ""] = (skip_q, "", "R")
    for trit in trits:
        transitions[skip_q, trit] = (skip_q, trit, "R")
    transitions[skip_q, delim] = (next_trit, delim, "R")

    # 3. Match next-state trits one trit at a time, advancing X from column 1 to w+1.
    for trit, state in find_trit.items():
        transitions[next_trit, trit] = (state, "Y", "X")
        transitions[state, "X"] = (state, "X", "R")
        for observed in trits:
            transitions[state, observed] = (
                (state, "X", "Y")
                if observed == trit
                else (state, observed, "U" if observed > trit else "D")
            )
        transitions[state, "Y"] = (next_trit, "Y", "R")

    # 4. At the second delim, rewind X in a single state back to blank at column 1.
    transitions[next_trit, delim] = (reset_x, "Y", "X")
    for observed in ("X", *trits):
        transitions[reset_x, observed] = (reset_x, observed, "L")
    transitions[reset_x, ""] = (reset_x, "X", "Y")
    transitions[reset_x, "Y"] = (read_write, "Y", "R")

    # 5. Read write_symbol and motion, then apply to Z on the simulated tape.
    for symbol, state in read_motion.items():
        transitions[read_write, symbol] = (state, symbol, "R")
        for motion in "LR":
            writer = write_move[symbol, motion]
            transitions[state, motion] = (writer, "Y", "Z")
            transitions[writer, "Z"] = (dispatch, symbol, motion)
        transitions[state, "S"] = (state, "Y", "Z")
        transitions[state, "Z"] = (halt, symbol, "S")

    for state in range(halt + 1):
        for symbol in alphabet:
            transitions.setdefault((state, symbol), (halt, symbol, "S"))
    return transitions


def parity_tm():
    """
    Creates the transition diagram of a Turing machine
    with 3 states and 9 transitions that computes the
    parity of a binary string. The input is a binary
    string, and the output is '0' for even parity and
    '1' for odd parity.
    """
    transitions = {
        (0, "0"): (0, "", "R"),
        (0, "1"): (1, "", "R"),
        (0, ""): (2, "0", "S"),
        (1, "0"): (1, "", "R"),
        (1, "1"): (0, "", "R"),
        (1, ""): (2, "1", "S"),
        (2, "0"): (2, "0", "S"),
        (2, "1"): (2, "1", "S"),
        (2, ""): (2, "", "S"),
    }

    return transitions


def flip_tm():
    """
    Creates the transition diagram of a Turing machine
    with 2 states and 6 transitions that flips a binary
    string. The input is a binary string, and the output
    is the flipped string.
    """
    transitions = {
        (0, "0"): (0, "1", "R"),
        (0, "1"): (0, "0", "R"),
        (0, ""): (1, "", "S"),
        (1, "0"): (1, "0", "S"),
        (1, "1"): (1, "1", "S"),
        (1, ""): (1, "", "S"),
    }

    return transitions


def cum_sum_tm():
    """
    Creates the transition diagram of a Turing machine
    with 3 states and 9 transitions that computes the
    cumsum of a binary string. The input is a binary string,
    and the output is the cumsum of that string.
    """
    transitions = {
        (0, "0"): (0, "0", "R"),
        (0, "1"): (1, "1", "R"),
        (0, ""): (2, "", "S"),
        (1, "0"): (1, "1", "R"),
        (1, "1"): (0, "0", "R"),
        (1, ""): (2, "", "S"),
        (2, "0"): (2, "0", "S"),
        (2, "1"): (2, "1", "S"),
        (2, ""): (2, "", "S"),
    }

    return transitions


def bit_delete_tm():
    """
    Creates the transition diagram of a Turing machine
    with 5 states and 15 transitions that computes the
    parity of the length of a binary string. Then, it goes
    back and deletes all 0s or all 1s depending on whether
    the parity is 0 or 1. The input is a binary string, and
    the output is a string of 0s for even lengths and 1s for
    odd lengths (riddled with holes).
    """

    transitions = {
        (0, "0"): (1, "0", "R"),
        (0, "1"): (1, "1", "R"),
        (0, ""): (2, "", "L"),
        (1, "0"): (0, "0", "R"),
        (1, "1"): (0, "1", "R"),
        (1, ""): (3, "", "L"),
        (2, "0"): (2, "0", "L"),
        (2, "1"): (2, "", "L"),
        (2, ""): (4, "", "S"),
        (3, "0"): (3, "", "L"),
        (3, "1"): (3, "1", "L"),
        (3, ""): (4, "", "S"),
        (4, "0"): (4, "0", "S"),
        (4, "1"): (4, "1", "S"),
        (4, ""): (4, "", "S"),
    }

    return transitions


def shift_left_tm():
    """
    Creates the transition diagram of a Turing machine
    with 5 states and 15 transitions that shifts the
    binary string one cell to the left and halts.
    """

    transitions = {
        (0, "0"): (1, "", "L"),
        (0, "1"): (2, "", "L"),
        (0, ""): (4, "", "S"),
        (1, "0"): (4, "0", "S"),
        (1, "1"): (4, "1", "S"),
        (1, ""): (3, "0", "R"),
        (2, "0"): (4, "0", "S"),
        (2, "1"): (4, "1", "S"),
        (2, ""): (3, "1", "R"),
        (3, "0"): (4, "0", "S"),
        (3, "1"): (4, "1", "S"),
        (3, ""): (0, "", "R"),
        (4, "0"): (4, "0", "S"),
        (4, "1"): (4, "1", "S"),
        (4, ""): (4, "", "S"),
    }

    return transitions


def shift_right_tm():
    """
    Creates the transition diagram of a Turing machine
    with 7 states and 21 transitions that shifts the
    binary string one cell to the right and halts.
    """

    transitions = {
        (0, "0"): (5, "0", "R"),
        (0, "1"): (5, "1", "R"),
        (0, ""): (6, "", "S"),
        # Carry the current bit right, then return left to the next bit.
        # The original table erased zeroes and moved left instead.
        (1, "0"): (2, "", "R"),
        (1, "1"): (3, "", "R"),
        (1, ""): (6, "", "S"),
        (2, "0"): (6, "0", "S"),
        (2, "1"): (6, "1", "S"),
        (2, ""): (4, "0", "L"),
        (3, "0"): (6, "0", "S"),
        (3, "1"): (6, "1", "S"),
        (3, ""): (4, "1", "L"),
        (4, "0"): (6, "0", "S"),
        (4, "1"): (6, "1", "S"),
        (4, ""): (1, "", "L"),
        (5, "0"): (5, "0", "R"),
        (5, "1"): (5, "1", "R"),
        (5, ""): (1, "", "L"),
        (6, "0"): (6, "0", "S"),
        (6, "1"): (6, "1", "S"),
        (6, ""): (6, "", "S"),
    }

    return transitions


def scaling_A_tm(m):
    """
    Creates the transition diagram of a Turing machine
    with m+2 states and 3m+6 transitions with the following
    description: When the TM is in state i and reads a 0,
    it writes a 1 and moves left. If it reads a 1, it writes
    a 0 and moves right. If it reads a blank symbol, it writes
    (i+1)%2 and moves left if i%2=0 and right if i%2=1. The last
    state is the halting state. Note that no matter the input
    length, the TM always performs m computational steps followed by one symbol-preserving S step.
    """
    transitions = {}

    for i in range(m):
        transitions[(i, "0")] = (i + 1, "1", "L")
        transitions[(i, "1")] = (i + 1, "0", "R")
        transitions[(i, "")] = (i + 1, str((i + 1) % 2), "L" if i % 2 == 0 else "R")

    for symbol in ("0", "1", ""):
        transitions[(m, symbol)] = (m + 1, symbol, "S")
        transitions[(m + 1, symbol)] = (m + 1, symbol, "S")

    return transitions


def scaling_B_tm(m):
    """
    Creates the transition diagram of a Turing machine
    with 4*m+1 states and 12*m+3 transitions
    that computes a complicated parity-inspired process.
    """
    transitions = {}

    # Handling the Right-Even states
    for i in range(m - 1):
        transitions[(i, "0")] = (i + 1, "0", "R")
        transitions[(i, "1")] = (i + m + 1, "1", "R")
        transitions[(i, "")] = (i + 2 * m + 1, "0", "R")
    transitions[(m - 1, "0")] = (4 * m, "0", "S")
    transitions[(m - 1, "1")] = (4 * m, "1", "S")
    transitions[(m - 1, "")] = (4 * m, "0", "S")

    # Handling the Right-Odd states
    for i in range(m, 2 * m - 1):
        transitions[(i, "0")] = (i + 1, "1", "R")
        transitions[(i, "1")] = (i - m + 1, "0", "R")
        transitions[(i, "")] = (i + 2 * m + 1, "1", "R")
    transitions[(2 * m - 1, "0")] = (4 * m, "1", "S")
    transitions[(2 * m - 1, "1")] = (4 * m, "0", "S")
    transitions[(2 * m - 1, "")] = (4 * m, "1", "S")

    # Handling the Left-Even states
    for i in range(2 * m, 3 * m - 1):
        transitions[(i, "0")] = (i + 1, "0", "L")
        transitions[(i, "1")] = (i + m + 1, "1", "L")
        transitions[(i, "")] = (i - 2 * m + 1, "0", "L")
    transitions[(3 * m - 1, "0")] = (4 * m, "0", "S")
    transitions[(3 * m - 1, "1")] = (4 * m, "1", "S")
    transitions[(3 * m - 1, "")] = (4 * m, "0", "S")

    # Handling the Left-Odd states
    for i in range(3 * m, 4 * m - 1):
        transitions[(i, "0")] = (i + 1, "1", "L")
        transitions[(i, "1")] = (i - m + 1, "0", "L")
        transitions[(i, "")] = (i - 2 * m + 1, "1", "L")
    transitions[(4 * m - 1, "0")] = (4 * m, "1", "S")
    transitions[(4 * m - 1, "1")] = (4 * m, "0", "S")
    transitions[(4 * m - 1, "")] = (4 * m, "1", "S")

    transitions[(4 * m, "0")] = (4 * m, "0", "S")
    transitions[(4 * m, "1")] = (4 * m, "1", "S")
    transitions[(4 * m, "")] = (4 * m, "", "S")

    return transitions


def reverse_tm():
    """
    Creates the transition diagram of a Turing machine
    with 16 states and 48 transitions that reverses a
    binary string. The input is a binary string, and
    the output is the reversed string.
    """
    transitions = {
        (0, "0"): (1, "0", "R"),
        (0, "1"): (1, "1", "R"),
        (1, "0"): (2, "0", "R"),
        (1, "1"): (2, "1", "R"),
        (2, ""): (4, "", "L"),
        (2, "0"): (3, "0", "L"),
        (2, "1"): (3, "1", "L"),
        (3, "0"): (5, "", "L"),
        (3, "1"): (6, "", "L"),
        (4, "0"): (7, "", "L"),
        (4, "1"): (8, "", "L"),
        (5, ""): (5, "", "L"),
        (5, "0"): (9, "0", "L"),
        (5, "1"): (9, "1", "L"),
        (6, ""): (6, "", "L"),
        (6, "0"): (10, "0", "L"),
        (6, "1"): (10, "1", "L"),
        (7, ""): (7, "", "L"),
        (7, "0"): (11, "0", "L"),
        (7, "1"): (11, "1", "L"),
        (8, ""): (8, "", "L"),
        (8, "0"): (12, "0", "L"),
        (8, "1"): (12, "1", "L"),
        (9, ""): (13, "0", "R"),
        (9, "0"): (9, "0", "L"),
        (9, "1"): (9, "1", "L"),
        (10, ""): (13, "1", "R"),
        (10, "0"): (10, "0", "L"),
        (10, "1"): (10, "1", "L"),
        (11, ""): (15, "0", "S"),
        (11, "0"): (11, "0", "L"),
        (11, "1"): (11, "1", "L"),
        (12, ""): (15, "1", "S"),
        (12, "0"): (12, "0", "L"),
        (12, "1"): (12, "1", "L"),
        (13, ""): (14, "", "R"),
        (13, "0"): (13, "0", "R"),
        (13, "1"): (13, "1", "R"),
        (14, ""): (14, "", "R"),
        (14, "0"): (2, "0", "R"),
        (14, "1"): (2, "1", "R"),
    }

    # Add missing transitions
    tape_alphabet = {"0", "1", ""}
    states = set()
    for state, _ in transitions:
        states.add(state)
    for new_state, _, _ in transitions.values():
        states.add(new_state)

    for state in states:
        for symbol in tape_alphabet:
            if (state, symbol) not in transitions:
                transitions[(state, symbol)] = (len(states) - 1, symbol, "S")

    return transitions


def copy_tm():
    """
    Creates the transition diagram of a Turing machine
    with 10 states and 30 transitions that copies a
    binary string. The input is a binary string 'w', and the
    output is 'w w'.
    """
    transitions = {
        (0, "0"): (1, "", "R"),
        (0, "1"): (2, "", "R"),
        (0, ""): (9, "", "S"),
        (1, "0"): (1, "0", "R"),
        (1, "1"): (1, "1", "R"),
        (1, ""): (3, "", "R"),
        (2, "0"): (2, "0", "R"),
        (2, "1"): (2, "1", "R"),
        (2, ""): (4, "", "R"),
        (3, "0"): (3, "0", "R"),
        (3, "1"): (3, "1", "R"),
        (3, ""): (5, "0", "L"),
        (4, "0"): (4, "0", "R"),
        (4, "1"): (4, "1", "R"),
        (4, ""): (6, "1", "L"),
        (5, "0"): (5, "0", "L"),
        (5, "1"): (5, "1", "L"),
        (5, ""): (7, "", "L"),
        (6, "0"): (6, "0", "L"),
        (6, "1"): (6, "1", "L"),
        (6, ""): (8, "", "L"),
        (7, "0"): (7, "0", "L"),
        (7, "1"): (7, "1", "L"),
        (7, ""): (0, "0", "R"),
        (8, "0"): (8, "0", "L"),
        (8, "1"): (8, "1", "L"),
        (8, ""): (0, "1", "R"),
    }

    # Add missing transitions
    tape_alphabet = {"0", "1", ""}
    states = set()
    for state, _ in transitions:
        states.add(state)
    for new_state, _, _ in transitions.values():
        states.add(new_state)

    for state in states:
        for symbol in tape_alphabet:
            if (state, symbol) not in transitions:
                transitions[(state, symbol)] = (len(states) - 1, symbol, "S")

    return transitions
