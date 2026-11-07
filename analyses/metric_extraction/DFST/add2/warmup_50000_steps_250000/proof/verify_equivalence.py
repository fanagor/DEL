#!/usr/bin/env python3
"""Finite certificate and independent grid tests for the adjacent controller.

Run with Python 3; no third-party packages are needed. The finite check proves
action agreement on the observation protocol. The accompanying LaTeX proof
establishes that every valid addition grid follows that protocol and terminates.
"""
from pathlib import Path
import argparse
import collections
import csv
import hashlib
import importlib.util
import itertools
import json
import random

ROOT = Path(__file__).resolve().parent
ALPHABET = ['', '+', '0', '1']
MOVES = 'LRDUS'
DELTA = {'L': (-1, 0), 'R': (1, 0), 'D': (0, 1), 'U': (0, -1), 'S': (0, 0)}
PROTOCOL = {'start': {'0': 'seek', '1': 'seek'},
 'seek': {'0': 'seek', '1': 'seek', '+': 'initial_down'},
 'initial_down': {'0': 'place', '1': 'place'},
 'place': {'': 'ensure'},
 'ensure': {'0': 'left_scan', '1': 'left_scan', '': 'adjacent'},
 'adjacent': {'0': 'left_scan', '1': 'left_scan'},
 'left_scan': {'0': 'left_scan', '1': 'left_scan', '+': 'right_start'},
 'right_start': {'0': 'right_scan', '1': 'right_scan', '': 'take_separator'},
 'right_scan': {'0': 'right_scan', '1': 'right_scan', '': 'take_digit'},
 'take_digit': {'0': 'zero_first', '1': 'one_first'},
 'zero_first': {'': 'zero_second'},
 'zero_second': {'': 'zero_walk'},
 'zero_walk': {'': 'zero_walk', '+': 'zero_add'},
 'zero_add': {'0': 'at_marker', '1': 'at_marker'},
 'one_first': {'': 'one_second'},
 'one_second': {'': 'one_walk'},
 'one_walk': {'': 'one_walk', '+': 'one_add'},
 'one_add': {'0': 'at_marker', '1': 'carry'},
 'carry': {'1': 'carry', '0': 'return_first', '': 'return_first'},
 'return_first': {'': 'return_walk'},
 'return_walk': {'': 'return_walk', '+': 'place'},
 'at_marker': {'+': 'place'},
 'take_separator': {'+': 'clean_first'},
 'clean_first': {'': 'clean_walk'},
 'clean_walk': {'': 'clean_walk', '+': 'stop'},
 'stop': {'0': 'halt', '1': 'halt'},
 'halt': {}}
GROUND_STATES = {'start': 0,
 'seek': 0,
 'initial_down': 1,
 'place': 1,
 'ensure': 8,
 'adjacent': 2,
 'left_scan': 2,
 'right_start': 2,
 'right_scan': 2,
 'take_digit': 3,
 'zero_first': 4,
 'zero_second': 4,
 'zero_walk': 4,
 'zero_add': 4,
 'one_first': 5,
 'one_second': 5,
 'one_walk': 5,
 'one_add': 5,
 'carry': 6,
 'return_first': 7,
 'return_walk': 7,
 'at_marker': 7,
 'take_separator': 3,
 'clean_first': 9,
 'clean_walk': 9,
 'stop': 9,
 'halt': 10}
MACHINE_JSON = 'minimized_partial.json'
EXPECTED_COUNTS = {'halt': 21, 'pairs': 28, 'edges': 50}

REQUIRED_FILES = ('expert_transition_table.csv', 'minimized_partial.json', 'minimized_transition_table.csv', 'minimized_transition_table.pdf', 'proof_reference_grid.py', 'verify_equivalence.py')

def verify_artifacts():
    """Validate the exact artifact set before checking the finite certificate."""
    manifest = json.loads((ROOT / "fingerprints.json").read_text())
    if set(manifest) != set(REQUIRED_FILES):
        raise ValueError("Proof manifest must list exactly the required artifacts")
    for name in REQUIRED_FILES:
        path = ROOT / name
        if not path.is_file():
            raise FileNotFoundError(f"Missing proof artifact: {name}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != manifest[name]:
            raise ValueError(f"Proof artifact checksum mismatch: {name}")
    return manifest


def table(path):
    result = {}
    with path.open() as stream:
        for row in csv.DictReader(stream):
            for symbol in ALPHABET:
                raw = row[symbol or 'λ']
                if raw:
                    q, written, motion = raw.split(' | ')
                    key = (int(row['state']), symbol)
                    assert key not in result
                    result[key] = (int(q), '' if written == 'λ' else written, motion)
    return result

def run_grid(machine, a, b):
    cells = {(x, 0): value for x, value in enumerate(a + '+' + b)}
    head = (0, 0)
    state = 0
    bound = 100 * (len(a) + len(b) + 2) ** 2
    for step in range(bound):
        read = cells.get(head, '')
        successor, written, motion = machine[state, read]
        if written:
            cells[head] = written
        else:
            cells.pop(head, None)
        dx, dy = DELTA[motion]
        head = (head[0] + dx, head[1] + dy)
        state = successor
        yield (read, written, motion, head)
        if motion == 'S':
            return cells, head
    raise AssertionError(('step limit', len(a), len(b)))

def compare_grid(machine, expert, a, b):
    learned = run_grid(machine, a, b)
    ground = run_grid(expert, a, b)
    steps = 0
    while True:
        try:
            left = next(learned)
        except StopIteration as stop:
            learned_result = stop.value
            try:
                next(ground)
            except StopIteration as end:
                assert learned_result == end.value
                break
            raise AssertionError('early learned halt')
        try:
            right = next(ground)
        except StopIteration:
            raise AssertionError('late learned halt')
        assert left == right, (a, b, steps, left, right)
        steps += 1
    cells, head = learned_result
    m, n = len(a), len(b)
    value = int(a, 2) + int(b, 2)
    width = max(m, n + 1, max(1, value.bit_length()))
    expected = {(m - width + i, 0): d for i, d in enumerate(format(value, f'0{width}b'))}
    assert cells == expected
    assert head == (m - n - 1, 0)
    return steps

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true', help='only finite certificate and source checks')
    parser.add_argument('--report', type=Path, help='optional JSON report destination')
    args = parser.parse_args()
    verify_artifacts()
    expert = table(ROOT / 'expert_transition_table.csv')
    machine = table(ROOT / 'minimized_transition_table.csv')
    spec = importlib.util.spec_from_file_location('reference', ROOT / 'proof_reference_grid.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert expert == module.add_grid(2)
    exported = json.loads((ROOT / MACHINE_JSON).read_text())
    decoded = {(t['state'], ALPHABET[t['input']]):
               (t['next_state'], ALPHABET[t['symbol']], MOVES[t['motion']])
               for t in exported['transitions']}
    assert machine == decoded
    assert exported['initial_state'] == 0
    assert exported['final_states'] == [EXPECTED_COUNTS['halt']]
    assert table(ROOT / 'expert_transition_table.csv') == expert
    seen = {('start', 0)}
    todo = collections.deque(seen)
    checks = []
    while todo:
        phase, state = todo.popleft()
        for read, next_phase in PROTOCOL[phase].items():
            gnext, written, motion = expert[GROUND_STATES[phase], read]
            assert gnext == GROUND_STATES[next_phase]
            assert (state, read) in machine, (phase, state, read, 'missing transition')
            snext, written2, motion2 = machine[state, read]
            assert (written, motion) == (written2, motion2), (phase, state, read)
            assert (snext in exported['final_states']) == (next_phase == 'halt')
            assert (motion == 'S') == (next_phase == 'halt')
            target = (next_phase, snext)
            checks.append([phase, state, read, next_phase, snext, written, motion])
            if target not in seen:
                seen.add(target)
                todo.append(target)
    assert len(seen) == EXPECTED_COUNTS['pairs']
    assert len(checks) == EXPECTED_COUNTS['edges']
    report = {
        'controller': ROOT.name,
        'source_hashes_match': True,
        'csv_matches_json': True,
        'expert_equals_add_grid_2': True,
        'protocol_phases': len(PROTOCOL),
        'reachable_phase_state_pairs': len(seen),
        'checked_protocol_edges': len(checks),
        'covered_nonhalt_table_entries': len({(s, x) for p, s, x, np, ns, w, m in checks}),
        'phase_states': {p: sorted(s for pp, s in seen if pp == p) for p in PROTOCOL},
        'checked_edges': checks,
    }
    if not args.quick:
        words = [''.join(t) for n in range(1, 7) for t in itertools.product('01', repeat=n)]
        small_steps = sum(compare_grid(machine, expert, a, b) for a, b in itertools.product(words, repeat=2))
        directed = []
        for n in [1, 2, 3, 8, 32, 128, 512]:
            directed += [('1' * n, '1'), ('0', '1' * n), ('0' * n, '0' * n),
                         ('1' * n, '1' * n), ('0' * n + '1', '1' * n),
                         ('1' + '0' * n, '1')]
        directed += [('1' * 1024, '1'), ('0' * 1024, '1')]
        rng = random.Random(20261102)
        random_inputs = [(''.join(rng.choice('01') for _ in range(rng.randint(1, 64))),
                          ''.join(rng.choice('01') for _ in range(rng.randint(1, 64)))) for _ in range(128)]
        long_steps = sum(compare_grid(machine, expert, a, b) for a, b in directed)
        random_steps = sum(compare_grid(machine, expert, a, b) for a, b in random_inputs)
        report['grid_tests'] = {
            'exhaustive_pairs_lengths_1_through_6': len(words) ** 2,
            'directed_cases': len(directed),
            'random_cases': len(random_inputs),
            'max_operand_length': 1024,
            'compared_transitions': small_steps + long_steps + random_steps,
            'full_trace_final_grid_and_final_head_agree': True,
        }
    if args.report:
        args.report.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k not in {'checked_edges', 'phase_states'}}, indent=2))

if __name__ == '__main__':
    main()
