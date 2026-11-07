"""Grid arithmetic with explicit, contiguous controller states.

Inputs are two nonempty radix-R digit strings separated by '+' or 'x'. All
controllers leave a contiguous result with no work markers and retain leading
zeros, including the padding introduced during arithmetic.
"""


def _complete(transitions, radix, operator, halt):
    """Complete unreachable state/symbol pairs by preserving the symbol and halting."""
    alphabet = ["", operator, *map(str, range(radix))]
    for state in range(halt + 1):
        for symbol in alphabet:
            transitions.setdefault((state, symbol), (halt, symbol, "S"))
    return transitions


def add_grid(R):
    """Digit-wise addition with R+9 states, including halt (11 binary, 19 decimal).

    A '+' below the first operand tracks the next place to add. Initialization
    and marker placement share one state. Cleanup removes both markers, then
    preserves the digit above the last marker and halts, treating zero like any
    other digit. Leading zeros, including those introduced by marker extension,
    remain. For operand lengths m,n, the result ends at (m-1,0), has at least
    max(m,n+1) digits, and the final head is at (m-n-1,0), possibly inside it.
    """
    if not isinstance(R, int) or isinstance(R, bool) or R < 2:
        raise ValueError("radix must be an integer of at least two")
    find_separator, place_marker, scan_right, take_digit = range(4)
    add_digit = [4 + digit for digit in range(R)]
    (
        carry,
        find_marker,
        resume_scan,
        finish_marker,
        halt,
    ) = range(R + 4, R + 9)
    transitions = {}

    # Place the first marker below the first operand's least significant digit.
    for digit in map(str, range(R)):
        transitions[find_separator, digit] = (find_separator, digit, "R")
        transitions[place_marker, digit] = (place_marker, digit, "D")
        transitions[scan_right, digit] = (scan_right, digit, "R")
        transitions[resume_scan, digit] = (scan_right, digit, "R")
    transitions[find_separator, "+"] = (place_marker, "+", "L")
    transitions[place_marker, ""] = (resume_scan, "+", "U")
    transitions[scan_right, "+"] = (scan_right, "+", "R")
    transitions[scan_right, ""] = (take_digit, "", "L")

    # Erase one second-operand digit and carry its value to the marked column.
    for digit, state in enumerate(add_digit):
        transitions[take_digit, str(digit)] = (state, "", "D")
        transitions[state, ""] = (state, "", "L")
        transitions[state, "+"] = (state, "+", "U")
        for other in range(R):
            total = digit + other
            transitions[state, str(other)] = (
                (find_marker, str(total), "D") if total < R else (carry, str(total - R), "L")
            )
    for digit in range(R - 1):
        transitions[carry, str(digit)] = (find_marker, str(digit + 1), "D")
    transitions[carry, str(R - 1)] = (carry, "0", "L")
    transitions[carry, ""] = (find_marker, "1", "D")

    # Move the marker one column left and extend a short first operand with zero.
    transitions[find_marker, ""] = (find_marker, "", "R")
    transitions[find_marker, "+"] = (place_marker, "", "L")
    transitions[resume_scan, ""] = (scan_right, "0", "R")

    # Remove both '+' signs, preserving every result digit including zero.
    transitions[take_digit, "+"] = (finish_marker, "", "D")
    transitions[finish_marker, ""] = (finish_marker, "", "L")
    transitions[finish_marker, "+"] = (finish_marker, "", "U")
    for digit in map(str, range(R)):
        transitions[finish_marker, digit] = (halt, digit, "S")
    return _complete(transitions, R, "+", halt)


def add_grid_new(R):
    """Breadcrumb-counter addition with R+13 states, including halt (15 binary, 23 decimal).

    Preserves the exact input/output/padding/final-head contract of ``add_grid(R)``
    while eliminating silent state transport and the R x R digit-addition table:
    each second-operand digit d is transported left on row -1 while writing a
    contiguous breadcrumb trail of ``str(d)`` (making ``carry_digit[d]`` horizon-1
    observable at every step), then added into the marked column of the first
    operand via a local vertical +/-1 counter loop before erasing the trail rightwards
    back to the next digit of the second operand.
    """
    if not isinstance(R, int) or isinstance(R, bool) or R < 2:
        raise ValueError("radix must be an integer of at least two")
    find_separator, place_marker, init_scan, take_digit = range(4)
    carry_digit = [4 + digit for digit in range(R)]
    (
        local_inc,
        return_marker,
        move_marker,
        ensure_digit,
        erase_trail,
        step_back_first,
        step_back_second,
        finish_marker,
        halt,
    ) = range(R + 4, R + 13)
    digits = [str(digit) for digit in range(R)]
    transitions = {}

    # Place the initial '+' marker below the first operand's units digit and scan right.
    for digit in digits:
        transitions[find_separator, digit] = (find_separator, digit, "R")
        transitions[place_marker, digit] = (place_marker, digit, "D")
        transitions[init_scan, digit] = (init_scan, digit, "R")
    transitions[find_separator, "+"] = (place_marker, "+", "L")
    transitions[place_marker, ""] = (init_scan, "+", "U")
    transitions[init_scan, "+"] = (init_scan, "+", "R")
    transitions[init_scan, ""] = (take_digit, "", "L")

    # Erase one second-operand digit and leave an observable breadcrumb trail on row -1.
    for digit, state in enumerate(carry_digit):
        transitions[take_digit, str(digit)] = (state, "", "D")
        transitions[state, ""] = (state, str(digit), "L")
        if digit == 0:
            transitions[state, "+"] = (move_marker, "0", "L")
        else:
            transitions[state, "+"] = (local_inc, str(digit - 1), "U")

    # Increment the marked column of the first operand via a local +/-1 counter loop.
    for digit in range(R - 1):
        transitions[local_inc, str(digit)] = (return_marker, str(digit + 1), "D")
    transitions[local_inc, str(R - 1)] = (local_inc, "0", "L")
    transitions[local_inc, ""] = (return_marker, "1", "D")

    # Return right across blanks if ripple carry moved left; deposited counter <= R-2.
    transitions[return_marker, ""] = (return_marker, "", "R")
    for digit in range(1, R - 1):
        transitions[return_marker, str(digit)] = (local_inc, str(digit - 1), "U")
    transitions[return_marker, "0"] = (move_marker, "0", "L")

    # Shift '+' one column left, zero-extend if needed, and erase the breadcrumb trail.
    transitions[move_marker, ""] = (ensure_digit, "+", "U")
    for digit in digits:
        transitions[ensure_digit, digit] = (erase_trail, digit, "D")
    transitions[ensure_digit, ""] = (erase_trail, "0", "D")

    transitions[erase_trail, "+"] = (erase_trail, "+", "R")
    for digit in digits:
        transitions[erase_trail, digit] = (erase_trail, "", "R")
    transitions[erase_trail, ""] = (step_back_first, "", "L")
    transitions[step_back_first, ""] = (step_back_second, "", "L")
    transitions[step_back_second, ""] = (take_digit, "", "U")

    # Remove both '+' signs, preserving every result digit including zero.
    transitions[take_digit, "+"] = (finish_marker, "", "D")
    transitions[finish_marker, ""] = (finish_marker, "", "L")
    transitions[finish_marker, "+"] = (finish_marker, "", "U")
    for digit in digits:
        transitions[finish_marker, digit] = (halt, digit, "S")
    return _complete(transitions, R, "+", halt)


def mult_grid(R):
    """Multiplication with R(R+1)/2+3R+23 states (32 binary, 108 decimal).

    Build shifted partial-product rows, then add neighboring rows right to left.
    The addition phase stores carry in the controller instead of immediately
    propagating it through the lower row. The combined top digit plus carry can
    equal R, so it has R+1 dedicated states. A carry beyond the top row is added
    to the next lower column before summation finishes. Emit a final S action when no
    lower row remains, preserving the full (m+n)-digit product on row n-1 at
    columns [-n, m-1]. The final head is on the blank at (-n-1, n-1).
    """
    if not isinstance(R, int) or isinstance(R, bool) or R < 2:
        raise ValueError("radix must be an integer of at least two")
    scan_input, skip_last, copy_digit = range(3)
    descend_digit = [3 + digit for digit in range(R)]
    descend_operator = R + 3
    write_digit = [R + 4 + digit for digit in range(R)]
    write_operator, read_copy = 2 * R + 4, 2 * R + 5
    read_factor = 2 * R + 6
    erase_factor = [read_factor + 1 + digit for digit in range(R)]
    zero_product = read_factor + R + 1
    product_carry = {
        (digit, carry): read_factor + R + 2 + digit * (digit - 1) // 2 + carry
        for digit in range(1, R)
        for carry in range(digit)
    }
    product_end = read_factor + R * (R + 1) // 2 + 2
    inspect_above, scan_above = product_end + 1, product_end + 2
    sum_entry = product_end + 3
    sum_left, inspect_below, lower_end = sum_entry + 1, sum_entry + 2, sum_entry + 3
    read_top = [sum_entry + 4, sum_entry + R + 6]
    return_top = [sum_entry + 5, sum_entry + R + 7]
    align_top = sum_entry + R + 8
    add_value = [sum_entry + 6 + value for value in range(R)] + [sum_entry + R + 9]
    finish = sum_entry + R + 10
    halt = finish + 1
    transitions = {}
    digits = list(map(str, range(R)))

    # Copy the input repeatedly one row down/one column left, omitting its last
    # digit each time. The resulting rows align the partial-product place values.
    for digit in digits:
        transitions[scan_input, digit] = (scan_input, digit, "R")
        transitions[skip_last, digit] = (copy_digit, digit, "L")
    transitions[scan_input, "x"] = (scan_input, "x", "R")
    transitions[scan_input, ""] = (skip_last, "", "L")
    for digit in range(R):
        transitions[copy_digit, str(digit)] = (descend_digit[digit], str(digit), "D")
        for symbol in [*digits, "x", ""]:
            transitions[descend_digit[digit], symbol] = (write_digit[digit], symbol, "L")
        transitions[write_digit[digit], ""] = (read_copy, str(digit), "U")
        transitions[descend_operator, str(digit)] = (write_operator, str(digit), "L")
        transitions[read_copy, str(digit)] = (descend_digit[digit], str(digit), "D")
    transitions[write_operator, ""] = (read_copy, "x", "U")
    transitions[read_copy, "x"] = (descend_operator, "x", "D")
    transitions[read_copy, ""] = (scan_input, "", "D")

    # Form one partial product per row, including a leading carry digit.
    transitions[copy_digit, "x"] = (read_factor, "x", "R")
    for digit in range(R):
        transitions[read_factor, str(digit)] = (erase_factor[digit], "", "L")
        for symbol in digits:
            transitions[erase_factor[digit], symbol] = (erase_factor[digit], "", "L")
    transitions[erase_factor[0], "x"] = (zero_product, "", "L")
    for digit in digits:
        transitions[zero_product, digit] = (zero_product, "0", "L")
    transitions[zero_product, ""] = (product_end, "0", "R")
    for digit in range(1, R):
        transitions[erase_factor[digit], "x"] = (product_carry[digit, 0], "", "L")
        for carry in range(digit):
            state = product_carry[digit, carry]
            for other in range(R):
                next_carry, written = divmod(digit * other + carry, R)
                transitions[state, str(other)] = (
                    product_carry[digit, next_carry],
                    str(written),
                    "L",
                )
            transitions[state, ""] = (product_end, str(carry), "R")
    for digit in digits:
        transitions[product_end, digit] = (product_end, digit, "R")
        transitions[inspect_above, digit] = (scan_above, digit, "R")
        transitions[scan_above, digit] = (scan_above, digit, "R")
    transitions[product_end, ""] = (inspect_above, "", "U")
    transitions[scan_above, "x"] = (scan_above, "x", "R")
    transitions[scan_above, ""] = (read_factor, "", "L")
    transitions[inspect_above, ""] = (sum_entry, "", "D")

    # Locate the next pair of rows and align their right ends.
    transitions[sum_entry, ""] = (sum_left, "", "L")
    for digit in digits:
        transitions[sum_left, digit] = (sum_left, digit, "L")
        transitions[inspect_below, digit] = (lower_end, digit, "R")
        transitions[lower_end, digit] = (lower_end, digit, "R")
        transitions[align_top, digit] = (align_top, digit, "R")
    transitions[sum_left, ""] = (inspect_below, "", "D")
    transitions[lower_end, ""] = (align_top, "", "U")
    transitions[align_top, ""] = (read_top[0], "", "L")

    # Consume one upper digit and its incoming carry. The lower update writes
    # one digit and chooses the next carry, without a separate carry cascade.
    for carry in (0, 1):
        for digit in range(R):
            transitions[read_top[carry], str(digit)] = (add_value[digit + carry], "", "D")
        transitions[return_top[carry], ""] = (read_top[carry], "", "L")
    transitions[read_top[0], ""] = (sum_left, "", "D")
    transitions[read_top[1], ""] = (add_value[1], "", "D")
    for value, state in enumerate(add_value):
        for symbol in ["", *digits]:
            next_carry, written = divmod(value + (int(symbol) if symbol else 0), R)
            transitions[state, symbol] = (return_top[next_carry], str(written), "U")

    # Only the final row remains. Keep its leading zeros and halt to its left.
    transitions[inspect_below, ""] = (finish, "", "U")
    for symbol in ["", *digits, "x"]:
        transitions[finish, symbol] = (halt, symbol, "S")
    return _complete(transitions, R, "x", halt)


def mult_grid_new(R):
    """2-row aligned shift-and-add multiplication with 2R+22 states (26 binary, 42 decimal).

    Row 0 holds the shifted first operand ``a``, any blank gap, fixed ``'x'``, and
    the remaining prefix of ``b``. Row 1 holds a dense running accumulator ``acc``
    initialized with ``'0'`` directly below ``a[0..m-1]``.

    - While ``b``'s rightmost digit ``d > 0``, decrement ``d`` and add ``a``
      vertically into ``acc`` right-to-left (``add_acc[0..R]`` is horizon-1
      observable at every step).
    - When ``b``'s rightmost digit is ``'0'``, erase it; if more digits of ``b``
      remain, shift ``a`` one cell left in a single pass (``shift_digit[0..R-1]``
      is horizon-1 observable) and ensure ``acc`` has a digit under ``a``'s new
      MSB; if ``'x'`` is reached, erase ``'x'`` and ``a``, step down to ``(-n, 1)``,
      write ``'0'`` if blank (or preserve the leading carry digit ``1..R-1``), and
      halt on the MSB of the untrimmed ``(m+n)``-digit product at ``[-n, m-1]``.
    """
    if not isinstance(R, int) or isinstance(R, bool) or R < 2:
        raise ValueError("radix must be an integer of at least two")
    (
        init_check,
        init_write,
        init_advance,
        to_x,
        scan_b,
        dec_b,
        check_more_b,
        back_to_a_add,
        find_a_add,
        back_to_a_shift,
        find_a_shift,
    ) = range(11)
    read_a = [11, 12]
    add_acc = [13 + value for value in range(R + 1)]
    step_left_a = [14 + R, 15 + R]
    prop_acc_carry = 16 + R
    shift_digit = [17 + R + digit for digit in range(R)]
    ensure_acc_msb = 17 + 2 * R
    erase_gap_to_a = 18 + 2 * R
    erase_a = 19 + 2 * R
    finish_msb = 20 + 2 * R
    halt = 21 + 2 * R
    digits = [str(digit) for digit in range(R)]
    transitions = {}

    # 1. Initialize row -1 with '0' below a[0..m-1], ending at 'x' -> scan_b.
    for digit in digits:
        transitions[init_check, digit] = (init_write, digit, "D")
        transitions[init_advance, digit] = (init_check, digit, "R")
    transitions[init_write, ""] = (init_advance, "0", "U")
    transitions[init_check, "x"] = (scan_b, "x", "R")

    # 2. Navigate to b's rightmost remaining digit.
    for digit in digits:
        transitions[to_x, digit] = (to_x, digit, "R")
        transitions[scan_b, digit] = (scan_b, digit, "R")
    transitions[to_x, ""] = (to_x, "", "R")
    transitions[to_x, "x"] = (scan_b, "x", "R")
    transitions[scan_b, ""] = (dec_b, "", "L")

    # 3. Decrement b's rightmost digit (add a), or erase '0' (shift a or finish).
    for digit in range(1, R):
        transitions[dec_b, str(digit)] = (back_to_a_add, str(digit - 1), "L")
    transitions[dec_b, "0"] = (check_more_b, "", "L")

    for digit in digits:
        transitions[back_to_a_add, digit] = (back_to_a_add, digit, "L")
        transitions[check_more_b, digit] = (back_to_a_shift, digit, "L")
        transitions[back_to_a_shift, digit] = (back_to_a_shift, digit, "L")
    transitions[back_to_a_add, "x"] = (find_a_add, "x", "L")
    transitions[find_a_add, ""] = (find_a_add, "", "L")
    for digit in range(R):
        transitions[find_a_add, str(digit)] = (add_acc[digit], str(digit), "D")

    transitions[check_more_b, "x"] = (erase_gap_to_a, "", "L")
    transitions[back_to_a_shift, "x"] = (find_a_shift, "x", "L")
    transitions[find_a_shift, ""] = (find_a_shift, "", "L")
    for digit in range(R):
        transitions[find_a_shift, str(digit)] = (shift_digit[digit], "", "L")

    # 4. Vertically aligned acc += a (right-to-left across a).
    # Because init and ensure_acc_msb keep acc dense under a, add_acc reads only digits;
    # and since acc <= (R^m - 1)(R^(j+1) - 1) < R^(m+j+1), carries past a's MSB never
    # overflow column x_l - 1, so prop_acc_carry reads only "" and 1..R-2.
    for carry in (0, 1):
        for digit in range(R):
            transitions[step_left_a[carry], str(digit)] = (read_a[carry], str(digit), "L")
            transitions[read_a[carry], str(digit)] = (add_acc[digit + carry], str(digit), "D")
    transitions[read_a[0], ""] = (to_x, "", "R")
    transitions[read_a[1], ""] = (prop_acc_carry, "", "D")

    for value, state in enumerate(add_acc):
        for symbol in digits:
            next_carry, written = divmod(value + int(symbol), R)
            transitions[state, symbol] = (step_left_a[next_carry], str(written), "U")

    for digit in range(1, R - 1):
        transitions[prop_acc_carry, str(digit)] = (to_x, str(digit + 1), "U")
    transitions[prop_acc_carry, ""] = (to_x, "1", "U")

    # 5. Shift a one cell left in one pass and ensure acc has a digit under a's new MSB.
    for digit in range(R):
        state = shift_digit[digit]
        for other in range(R):
            transitions[state, str(other)] = (shift_digit[other], str(digit), "L")
        transitions[state, ""] = (ensure_acc_msb, str(digit), "D")

    for digit in range(1, R):
        transitions[ensure_acc_msb, str(digit)] = (to_x, str(digit), "U")
    transitions[ensure_acc_msb, ""] = (to_x, "0", "U")

    # 6. Erase gap + a on row 0 and finalize the untrimmed (m+n)-digit product at (-n, 1).
    transitions[erase_gap_to_a, ""] = (erase_gap_to_a, "", "L")
    for digit in digits:
        transitions[erase_gap_to_a, digit] = (erase_a, "", "L")
        transitions[erase_a, digit] = (erase_a, "", "L")
    transitions[erase_a, ""] = (finish_msb, "", "D")
    transitions[finish_msb, ""] = (halt, "0", "S")
    for digit in range(1, R):
        transitions[finish_msb, str(digit)] = (halt, str(digit), "S")
    return _complete(transitions, R, "x", halt)
