"""Recheck the finite simulation certificate; optionally replay saved inputs.

Usage: python3 verify_equivalence.py [--quick]
Only the Python standard library is required. The accompanying mathematical
proof establishes that every valid grid execution follows this finite protocol.
"""
from pathlib import Path
from collections import deque
import argparse
import ast
import csv
import hashlib
import json
import time

ROOT = Path(__file__).resolve().parent
SYMBOLS = ['', *map(str, range(10)), 'x']
MOTIONS = 'LRDUS'


REQUIRED_FILES = ('expert.csv', 'expert.json', 'fresh_pto_comparison.json', 'minimized_partial.csv', 'minimized_partial.json', 'observation_protocol.py', 'replay_inputs.json', 'simulation_certificate.json', 'source_grid.py', 'verify_equivalence.py')

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


def load_machine(name):
    raw = json.loads((ROOT / (name + '.json')).read_text())
    rows = raw['transitions']
    with (ROOT / (name + '.csv')).open(newline='') as stream:
        csv_rows = [{k: int(v) for k, v in row.items()} for row in csv.DictReader(stream)]
    key = lambda row: (row['state'], row['input'])
    assert sorted(rows, key=key) == sorted(csv_rows, key=key)
    machine = {(r['state'], SYMBOLS[r['input']]):
               (r['next_state'], SYMBOLS[r['symbol']], MOTIONS[r['motion']]) for r in rows}
    assert len(machine) == len(rows)
    return raw, machine


def replay(a, b, controller, expert):
    # Separate sparse grids and heads; no teacher-forced observations.
    original = {(i, 0): x for i, x in enumerate(a + 'x' + b)}
    grids = [dict(original), dict(original)]
    heads, states = [(0, 0), (0, 0)], [0, 0]
    motion = {'L': (-1, 0), 'R': (1, 0), 'D': (0, 1), 'U': (0, -1), 'S': (0, 0)}
    for step in range(1, 100 * (len(a) + len(b) + 2) ** 2):
        reads = [grids[k].get(heads[k], '') for k in range(2)]
        actions = [machine[states[k], reads[k]] for k, machine in enumerate((controller, expert))]
        assert reads[0] == reads[1] and actions[0][1:] == actions[1][1:], (a, b, step)
        for k, (next_state, write, move) in enumerate(actions):
            if write:
                grids[k][heads[k]] = write
            else:
                grids[k].pop(heads[k], None)
            dx, dy = motion[move]
            heads[k] = (heads[k][0] + dx, heads[k][1] + dy)
            states[k] = next_state
        assert heads[0] == heads[1]
        assert (states[0] == 181) == (states[1] == 107)
        if actions[0][2] == 'S':
            assert states == [181, 107]
            assert heads[0] == (-len(b) - 1, len(b) - 1)
            value = str(int(a) * int(b)).zfill(len(a) + len(b))
            expected = {(i - len(b), len(b) - 1): d for i, d in enumerate(value)}
            assert grids[0] == grids[1] == expected, (a, b, 'final grid')
            return step
    raise AssertionError('Execution exceeded diagnostic step limit')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true', help='Check the full finite certificate without trajectory replay')
    args = parser.parse_args()
    start = time.monotonic()
    manifest = verify_artifacts()
    from observation_protocol import P, G
    raw, controller = load_machine('minimized_partial')
    eraw, expert = load_machine('expert')
    assert raw['initial_state'] == eraw['initial_state'] == 0
    assert raw['final_states'] == [181] and eraw['final_states'] == [107]
    assert len(raw['states']) == 182 and len(eraw['states']) == 108
    nodes = [node for node in ast.parse((ROOT / 'source_grid.py').read_text()).body
             if isinstance(node, ast.FunctionDef) and node.name in {'_complete', 'mult_grid'}]
    namespace = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), 'source_grid.py', 'exec'), namespace)
    assert namespace['mult_grid'](10) == expert
    names = list(P)
    index = {p: i for i, p in enumerate(names)}
    certificate = json.loads((ROOT / 'simulation_certificate.json').read_text())
    assert len(certificate) == len(P) == 207
    relation = {}
    for i, row in enumerate(certificate):
        p = names[i]
        assert row['i'] == i and row['name'] == p and row['g'] == G[p]
        assert row['edges'] == {x: index[p1] for x, p1 in P[p].items()}
        assert row['Q'] and len(row['Q']) == len(set(row['Q']))
        relation[p] = set(row['Q'])
    queue = deque([('start', 0)])
    seen, checked, checks = set(queue), set(), 0
    mapping = {}
    while queue:
        p, q = queue.popleft()
        assert q in relation[p]
        assert mapping.setdefault(q, G[p]) == G[p]
        assert (q == 181) == (p == 'halt')
        for read, successor in P[p].items():
            en, ew, em = expert[G[p], read]
            cn, cw, cm = controller[q, read]
            assert en == G[successor] and (ew, em) == (cw, cm)
            assert cn in relation[successor]
            assert (cn == 181) == (successor == 'halt') == (cm == 'S')
            checks += 1
            checked.add((q, read))
            pair = (successor, cn)
            if pair not in seen:
                seen.add(pair)
                queue.append(pair)
    assert seen == {(p, q) for p, qs in relation.items() for q in qs}
    assert set(mapping) == set(raw['states'])
    assert checked == {(q, x) for q, x in controller if q != 181}
    for (q, x), (qn, w, m) in controller.items():
        assert expert[mapping[q], x] == (mapping[qn], w, m)
    assert len(seen) == 208 and checks == 1509 and len(checked) == 1259
    undefined = len(raw['states']) * len(SYMBOLS) - len(controller)
    assert undefined == 913
    report = dict(controller_sha256=manifest['minimized_partial.json'], phases=len(P),
                  related_phase_state_pairs=len(seen), local_obligations=checks,
                  checked_nonterminal_entries=len(checked), defined_entries=len(controller),
                  unreachable_undefined_entries=undefined, all_finite_checks_pass=True)
    if not args.quick:
        extraction = json.loads((ROOT / 'replay_inputs.json').read_text())
        long_cases = json.loads((ROOT / 'fresh_pto_comparison.json').read_text())['results']
        groups = [('extraction', [r['operands'] for r in extraction]),
                  ('hundred_digit', [r['metadata']['operands'] for r in long_cases])]
        for name, cases in groups:
            steps = sum(replay(a, b, controller, expert) for a, b in cases)
            report[name] = dict(exact_matches=len(cases), steps=steps, final_products_checked=True)
            print(name, report[name], flush=True)
        assert report['extraction']['steps'] == 180510
        assert report['hundred_digit']['steps'] == 15114148
    report['seconds'] = round(time.monotonic() - start, 3)
    name = 'verification_quick.json' if args.quick else 'verification_report.json'
    (ROOT / name).write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
