"""Expert transition tables preserved from the original research implementation.

Source: data/turing_machine/tm_transitions.py; generated/random task families omitted.
"""


def add_tm(R):
    """
    Creates the transition diagram of a Turing machine
    with 2R+5 states and (2R+5)(2R+2) transitions that
    adds two numbers in radix-R. The alphabet includes
    2R digits, the '+' sign, and the blank symbol.
    When R>10, the input x should be presented as a list
    of strings from '0' to str(R-1).
    """
    transitions = {}

    # Add initial state, which goes to the end of the second number
    for i in range(R):
        transitions[(0, str(i))] = (0, str(i), "R")
        transitions[(0, "⋅" + str(i))] = (0, "⋅" + str(i), "R")
    transitions[(0, "+")] = (0, "+", "R")
    transitions[(0, "")] = (1, "", "L")

    # Store information about the last digit of the second number
    for i in range(R):
        # Delete last digit of the second number
        transitions[(1, str(i))] = (i + 2, "", "L")

        # Go left to the '+' sign
        for j in range(R):
            transitions[(i + 2, str(j))] = (i + 2, str(j), "L")
        transitions[(i + 2, "+")] = (i + R + 2, "+", "L")

        # Go left to the active digit of the first number
        for j in range(R):
            transitions[(i + R + 2, "⋅" + str(j))] = (i + R + 2, "⋅" + str(j), "L")
            # Reach the active digit j of the first number
            k = i + j
            if k < R:  # no carry, go back at the start
                transitions[(i + R + 2, str(j))] = (0, "⋅" + str(k), "R")
            else:  # carry, add 1 to the next digit with state 2R+2
                transitions[(i + R + 2, str(j))] = (2 * R + 2, "⋅" + str(k - R), "L")

        # If the scanned symbol is blank
        transitions[(i + R + 2, "")] = (0, "⋅" + str(i), "R")

    # Fix the carry state 2R+2
    transitions[(2 * R + 2, "")] = (0, "1", "R")
    for digit in range(R - 1):
        transitions[(2 * R + 2, str(digit))] = (0, str(digit + 1), "R")
    # Handle cascading carry for digit R+1
    transitions[(2 * R + 2, str(R - 1))] = (2 * R + 2, "0", "L")

    # When the second number is empty, rewrite the first one and halt
    transitions[(1, "+")] = (2 * R + 3, "", "L")
    for i in range(R):
        transitions[(2 * R + 3, "⋅" + str(i))] = (2 * R + 3, str(i), "L")

    # Create the halt state
    transitions[(2 * R + 4, "")] = (2 * R + 4, "", "S")

    # Add missing transitions
    tape_alphabet = {""}
    for _, symbol in transitions.keys():
        tape_alphabet.add(symbol)
    for _, symbol, _ in transitions.values():
        tape_alphabet.add(symbol)

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


def mult_tm(R):
    """
    Creates the transition diagram of a Turing machine
    with 3R^2+8R+31 states and (3R^2+8R+31)(2R+3)
    transitions that multiplies two numbers in radix-R.
    When R>10, the input x should be presented as a list
    of strings from '0' to str(R-1).
    """
    transitions = {}

    # Phase 0: Add a '=0+0' at the end to ensure 2 summands
    for i in range(R):
        transitions[(0, str(i))] = (0, str(i), "R")
    transitions[(0, "x")] = (0, "x", "R")
    transitions[(0, "")] = (1, "=", "R")
    transitions[(1, "")] = (2, "0", "R")
    transitions[(2, "")] = (3, "+", "R")
    transitions[(3, "")] = (4, "0", "L")
    for i in range(R):
        transitions[(4, str(i))] = (4, str(i), "L")
    transitions[(4, "+")] = (4, "+", "L")
    transitions[(4, "=")] = (4, "=", "L")
    transitions[(4, "x")] = (4, "x", "L")
    transitions[(4, "")] = (5, "", "R")

    # Phase 1: Add trailing zeroes based on the position of the multiplying digit of the first number
    START = 5
    for i in range(R):
        transitions[(START, str(i))] = (START + 1, str(i), "R")
        transitions[(START + 1, str(i))] = (START + 2, "⋅" + str(i), "R")
        transitions[(START + 2, str(i))] = (START + 2, str(i), "R")
    transitions[(START + 2, "x")] = (START + 2, "x", "R")
    transitions[(START + 2, "=")] = (START + 2, "=", "R")
    transitions[(START + 2, "+")] = (START + 3, "+", "R")
    transitions[(START + 3, "⋅0")] = (START + 3, "⋅0", "R")
    transitions[(START + 3, "0")] = (START + 3, "⋅0", "R")
    transitions[(START + 3, "")] = (START + 4, "0", "L")
    for i in range(R):
        transitions[(START + 4, str(i))] = (START + 4, str(i), "L")
    transitions[(START + 4, "+")] = (START + 4, "+", "L")
    transitions[(START + 4, "⋅0")] = (START + 4, "⋅0", "L")
    transitions[(START + 4, "=")] = (START + 5, "=", "L")
    for i in range(R):
        transitions[(START + 5, str(i))] = (START + 5, str(i), "L")
    transitions[(START + 5, "x")] = (START + 5, "x", "L")
    for i in range(R):
        transitions[(START + 5, "⋅" + str(i))] = (START + 1, "⋅" + str(i), "R")

    # Phase 2: Memorize the first digit i and start 1-digit multiplication
    MEM = START + 6
    transitions[(START + 1, "x")] = (MEM, "x", "L")
    for i in range(R):
        transitions[(MEM, "⋅" + str(i))] = (MEM, str(i), "L")
        transitions[(MEM, str(i))] = (MEM + i + 1, "", "R")
        for j in range(R):
            transitions[(MEM + i + 1, str(j))] = (MEM + i + 1, str(j), "R")
        transitions[(MEM + i + 1, "x")] = (MEM + i + 1, "x", "R")
        transitions[(MEM + i + 1, "=")] = (MEM + R + i + 1, "=", "L")

        # Memorize the second digit j
        for j in range(R):
            transitions[(MEM + R + i + 1, str(j))] = (MEM + (i + 2) * R + j + 1, "⋅" + str(j), "R")
            transitions[(MEM + (i + 2) * R + j + 1, "=")] = (MEM + (i + 2) * R + j + 1, "=", "R")
            transitions[(MEM + (i + 2) * R + j + 1, "+")] = (MEM + (i + 2) * R + j + 1, "+", "R")
            for k in range(R):
                transitions[(MEM + (i + 2) * R + j + 1, "⋅" + str(k))] = (
                    MEM + (i + 2) * R + j + 1,
                    "⋅" + str(k),
                    "R",
                )
                transitions[(MEM + (i + 2) * R + j + 1, str(k))] = (
                    MEM + (i + 2) * R + j + 1,
                    str(k),
                    "R",
                )
            transitions[(MEM + (i + 2) * R + j + 1, "")] = (
                MEM + R**2 + (i + 2) * R + j + 1,
                "",
                "L",
            )

            iCARRY = MEM + 2 * R**2 + 2 * R + i * (R + 1) + 1
            # Carry
            for k in range(R - 1):
                transitions[(MEM + R**2 + (i + 2) * R + j + 1, str(k))] = (
                    iCARRY + (i * j + k) // R,
                    "⋅" + str((i * j + k) % R),
                    "R",
                )
                # Future carry
                transitions[(iCARRY + k, "")] = (iCARRY + R - 1, str(k), "L")

        # Go back and memorize the next j
        for j in range(R):
            transitions[(iCARRY + R - 1, str(j))] = (iCARRY + R - 1, str(j), "L")
            transitions[(iCARRY + R - 1, "⋅" + str(j))] = (iCARRY + R - 1, "⋅" + str(j), "L")
        transitions[(iCARRY + R - 1, "+")] = (iCARRY + R - 1, "+", "L")
        transitions[(iCARRY + R - 1, "=")] = (iCARRY + R, "=", "L")
        for j in range(R):
            transitions[(iCARRY + R, "⋅" + str(j))] = (iCARRY + R, "⋅" + str(j), "L")
            transitions[(iCARRY + R, str(j))] = (MEM + (i + 2) * R + j + 1, "⋅" + str(j), "R")
        transitions[(iCARRY + R, "x")] = (MEM + 3 * R**2 + 3 * R + 1, "x", "R")

    # Phase 3: Clean up and sum
    for i in range(R):
        transitions[(MEM + 3 * R**2 + 3 * R + 1, str(i))] = (
            MEM + 3 * R**2 + 3 * R + 1,
            str(i),
            "R",
        )
        transitions[(MEM + 3 * R**2 + 3 * R + 1, "⋅" + str(i))] = (
            MEM + 3 * R**2 + 3 * R + 1,
            str(i),
            "R",
        )
    transitions[(MEM + 3 * R**2 + 3 * R + 1, "=")] = (MEM + 3 * R**2 + 3 * R + 1, "=", "R")
    transitions[(MEM + 3 * R**2 + 3 * R + 1, "+")] = (MEM + 3 * R**2 + 3 * R + 1, "+", "R")
    transitions[(MEM + 3 * R**2 + 3 * R + 1, "")] = (MEM + 3 * R**2 + 3 * R + 2, "", "L")
    for i in range(R):
        transitions[(MEM + 3 * R**2 + 3 * R + 2, str(i))] = (
            MEM + 3 * R**2 + 3 * R + 2,
            str(i),
            "L",
        )
    transitions[(MEM + 3 * R**2 + 3 * R + 2, "+")] = (MEM + 3 * R**2 + 3 * R + 2, "+", "L")
    transitions[(MEM + 3 * R**2 + 3 * R + 2, "=")] = (MEM + 3 * R**2 + 3 * R + 3, "=", "R")

    ADD = MEM + 3 * R**2 + 3 * R + 3  # 3R^2+3R+14
    # Remember the last digit of the first number
    for i in range(R):
        # Delete last digit of the first number
        transitions[(ADD, str(i))] = (ADD + i + 1, "", "R")

        # Go right to the '+' sign
        for j in range(R):
            transitions[(ADD + i + 1, str(j))] = (ADD + i + 1, str(j), "R")
        transitions[(ADD + i + 1, "+")] = (ADD + i + R + 1, "+", "R")

        # Go right to the active digit of the second number
        for j in range(R):
            transitions[(ADD + i + R + 1, "⋅" + str(j))] = (ADD + i + R + 1, "⋅" + str(j), "R")
            # Reach the active digit j of the second number
            k = i + j
            if k < R:  # no carry, go back at the start
                transitions[(ADD + i + R + 1, str(j))] = (ADD + 2 * R + 1, "⋅" + str(k), "L")
            else:  # carry, add 1 to the next digit with state 2R+2
                transitions[(ADD + i + R + 1, str(j))] = (ADD + 2 * R + 2, "⋅" + str(k - R), "R")

        # If the scanned symbol is blank
        transitions[(ADD + i + R + 1, "")] = (ADD + 2 * R + 1, "⋅" + str(i), "L")

    # Fix the carry state 2R+2
    transitions[(ADD + 2 * R + 2, "")] = (ADD + 2 * R + 1, "1", "L")
    for digit in range(R - 1):
        transitions[(ADD + 2 * R + 2, str(digit))] = (ADD + 2 * R + 1, str(digit + 1), "L")
    # Handle cascading carry for digit R+1
    transitions[(ADD + 2 * R + 2, str(R - 1))] = (ADD + 2 * R + 2, "0", "R")

    # Add a state, which goes to the beginning of the first number
    for i in range(R):
        transitions[(ADD + 2 * R + 1, str(i))] = (ADD + 2 * R + 1, str(i), "L")
        transitions[(ADD + 2 * R + 1, "⋅" + str(i))] = (ADD + 2 * R + 1, "⋅" + str(i), "L")
    transitions[(ADD + 2 * R + 1, "+")] = (ADD + 2 * R + 1, "+", "L")
    transitions[(ADD + 2 * R + 1, "")] = (ADD, "", "R")

    # When the first number is empty, rewrite the second one and halt
    transitions[(ADD, "+")] = (ADD + 2 * R + 3, "", "R")
    for i in range(R):
        transitions[(ADD + 2 * R + 3, "⋅" + str(i))] = (ADD + 2 * R + 3, str(i), "R")
        transitions[(ADD + 2 * R + 3, str(i))] = (ADD + 2 * R + 3, str(i), "R")

    # Add '+' and '=' at the end
    transitions[(ADD + 2 * R + 3, "")] = (ADD + 2 * R + 4, "+", "L")

    # Go to the start of the sum
    for i in range(R):
        transitions[(ADD + 2 * R + 4, str(i))] = (ADD + 2 * R + 4, str(i), "L")
    transitions[(ADD + 2 * R + 4, "")] = (ADD + 2 * R + 5, "", "R")

    # Phase 4: Shift the sum to the right of '='
    SHIFT = ADD + 2 * R + 5  # 3R^2+5R+19
    for i in range(R):
        transitions[(SHIFT, str(i))] = (SHIFT + i + 1, "", "L")
        transitions[(SHIFT + i + 1, "")] = (SHIFT + i + 1, "", "L")
        transitions[(SHIFT + i + 1, "=")] = (SHIFT + R + i + 1, "=", "R")
        for j in range(R):
            transitions[(SHIFT + i + 1, str(j))] = (SHIFT + R + i + 1, str(j), "R")
        transitions[(SHIFT + R + i + 1, "")] = (SHIFT, str(i), "R")
    transitions[(SHIFT, "")] = (SHIFT, "", "R")
    transitions[(SHIFT, "+")] = (SHIFT + 2 * R + 1, "", "L")
    transitions[(SHIFT + 2 * R + 1, "")] = (SHIFT + 2 * R + 1, "", "L")
    for i in range(R):
        transitions[(SHIFT + 2 * R + 1, str(i))] = (SHIFT + 2 * R + 2, str(i), "R")
    transitions[(SHIFT + 2 * R + 2, "")] = (SHIFT + 2 * R + 3, "+", "R")
    transitions[(SHIFT + 2 * R + 3, "")] = (SHIFT + 2 * R + 4, "0", "L")

    # Phase 5: Go back and collect the next digit for multiplication
    transitions[(SHIFT + 2 * R + 4, "+")] = (SHIFT + 2 * R + 4, "+", "L")
    transitions[(SHIFT + 2 * R + 4, "=")] = (SHIFT + 2 * R + 4, "=", "L")
    for i in range(R):
        transitions[(SHIFT + 2 * R + 4, str(i))] = (SHIFT + 2 * R + 4, str(i), "L")
    transitions[(SHIFT + 2 * R + 4, "x")] = (SHIFT + 2 * R + 5, "x", "L")
    for i in range(R):
        transitions[(SHIFT + 2 * R + 5, str(i))] = (SHIFT + 2 * R + 5, str(i), "L")
    transitions[(SHIFT + 2 * R + 5, "")] = (START, "", "R")

    # Initiate end protocol
    transitions[(START, "x")] = (SHIFT + 2 * R + 6, "", "R")
    for i in range(R):
        transitions[(SHIFT + 2 * R + 6, str(i))] = (SHIFT + 2 * R + 6, "", "R")
    transitions[(SHIFT + 2 * R + 6, "=")] = (SHIFT + 2 * R + 7, "", "R")

    # Phase 6: Reverse the multiplication, delete the '+0' and useless 0s
    REVERSE = SHIFT + 2 * R + 7  # 3R^2+7R+26
    for i in range(R):
        transitions[(REVERSE, str(i))] = (REVERSE + i + 1, "x", "L")
        transitions[(REVERSE + i + 1, "x")] = (REVERSE + i + 1, "x", "L")
        for j in range(R):
            transitions[(REVERSE + i + 1, str(j))] = (REVERSE + i + 1, str(j), "L")
        transitions[(REVERSE + i + 1, "")] = (REVERSE + R + 1, str(i), "R")
    for i in range(R):
        transitions[(REVERSE + R + 1, str(i))] = (REVERSE + R + 1, str(i), "R")
    transitions[(REVERSE + R + 1, "x")] = (REVERSE, "x", "R")
    transitions[(REVERSE, "x")] = (REVERSE, "x", "R")
    transitions[(REVERSE, "+")] = (REVERSE + R + 2, "", "R")
    transitions[(REVERSE + R + 2, "0")] = (REVERSE + R + 2, "", "L")
    transitions[(REVERSE + R + 2, "")] = (REVERSE + R + 3, "", "L")
    transitions[(REVERSE + R + 3, "x")] = (REVERSE + R + 3, "", "L")
    for i in range(R):
        transitions[(REVERSE + R + 3, str(i))] = (REVERSE + R + 3, str(i), "L")

    # Delete useless 0s
    transitions[(REVERSE + R + 3, "")] = (REVERSE + R + 4, "", "R")
    transitions[(REVERSE + R + 4, "0")] = (REVERSE + R + 4, "", "R")

    # If all there is is 0s, leave one 0 and halt
    transitions[(REVERSE + R + 4, "")] = (REVERSE + R + 5, "0", "S")

    # Add missing transitions
    tape_alphabet = {""}
    for _, symbol in transitions.keys():
        tape_alphabet.add(symbol)
    for _, symbol, _ in transitions.values():
        tape_alphabet.add(symbol)

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
