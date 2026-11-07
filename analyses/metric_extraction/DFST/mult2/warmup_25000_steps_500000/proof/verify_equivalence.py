#!/usr/bin/env python3
"""Recompute the mult2 equivalence certificate and independent grid checks."""
from pathlib import Path
import argparse
import ast
import collections
import hashlib
import itertools
import json
import random
from equivalence_protocol import P,G,NAMES,read_table,check

ROOT=Path(__file__).resolve().parent

REQUIRED_FILES = ('equivalence_protocol.py', 'expert_transition_table.csv', 'minimized_partial.json', 'minimized_transition_table.csv', 'minimized_transition_table.pdf', 'proof_reference_experts.py', 'proof_reference_grid.py', 'verify_equivalence.py')

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


def definitions():
    env={}
    for filename,names in [('proof_reference_grid.py',{'_complete','mult_grid'}),
                           ('proof_reference_experts.py',{'Memory'})]:
        tree=ast.parse((ROOT/filename).read_text())
        nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
        assert {n.name for n in nodes}==names
        exec(compile(ast.Module(body=nodes,type_ignores=[]),filename,'exec'),env)
    return env['Memory'],env['mult_grid']

def compare(machine,expert,Memory,a,b):
    left=Memory([a+'x'+b],(0,0),False)
    right=Memory([a+'x'+b],(0,0),False)
    q=g=0;p='start';used=set()
    for k in range(100*(len(a)+len(b)+2)**2):
        x=left.read();assert x==right.read()
        assert g==G[p] and x in P[p],(a,b,p,g,x)
        p1=P[p][x]
        q1,w,m=machine[q,x];g1,ww,mm=expert[g,x]
        assert (w,m)==(ww,mm),(a,b,k,g,q,x)
        used.add((q,x))
        left.write(w);left.move(m);right.write(ww);right.move(mm)
        assert left.head==right.head
        assert (q1==60)==(g1==31)==(m=='S')
        q,g,p=q1,g1,p1
        if m=='S':break
    else:raise AssertionError('step limit')
    assert left.cells==right.cells
    width=len(a)+len(b)
    expected={(i-len(b),len(b)-1):d for i,d in enumerate(format(int(a,2)*int(b,2),f'0{width}b'))}
    assert left.cells==expected,(a,b,left.cells,expected)
    assert left.head==(-len(b)-1,len(b)-1)
    assert p=='halt'
    return k+1,used

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--quick',action='store_true')
    parser.add_argument('--report',type=Path)
    args=parser.parse_args()
    verify_artifacts()
    Memory,builder=definitions()
    expert=read_table(ROOT/'expert_transition_table.csv')
    machine=read_table(ROOT/'minimized_transition_table.csv')
    assert expert==builder(2)
    exported=json.loads((ROOT/'minimized_partial.json').read_text())
    alphabet=['','0','1','x'];motions='LRDUS'
    decoded={(t['state'],alphabet[t['input']]):(t['next_state'],alphabet[t['symbol']],motions[t['motion']]) for t in exported['transitions']}
    assert machine==decoded and exported['initial_state']==0 and exported['final_states']==[60]
    seen,checks=check(ROOT)
    assert (len(P),len(seen),len(checks))==(67,67,130)
    used={(q,x) for p,q,x,*_ in checks}
    assert len(used)==119
    assert used=={(q,x) for q,x in machine if q!=60}
    report={'source_hashes_match':True,'csv_matches_json':True,'expert_equals_mult_grid_2':True,
            'reference_state_count':32,'controller_state_count':61,
            'protocol_phases':67,'phase_controller_pairs':67,'local_checks':130,
            'all_119_defined_nonterminal_entries_checked':True,
            'phase_states':[{'phase':p,'meaning':NAMES[p],'ground':G[p],
                'controller':next(q for pp,q in seen if pp==p)} for p in P],
            'checked_edges':checks}
    if not args.quick:
        totals=collections.Counter();observed=set()
        def case(a,b,category):
            steps,entries=compare(machine,expert,Memory,a,b)
            totals[category]+=1;totals['compared_transitions']+=steps;observed.update(entries)
        words=[''.join(w) for n in range(1,7) for w in itertools.product('01',repeat=n)]
        for a,b in itertools.product(words,repeat=2):case(a,b,'exhaustive_pairs_lengths_1_through_6')
        for n in [1,2,3,8,32,64,128]:
            for a,b in [('1'*n,'1'*n),('0'*n,'1'*n),('1'*n,'0'*n),
                        ('1','1'*n),('1'*n,'1'),('0'*n+'1','1'*n)]:case(a,b,'directed_cases')
        for a,b in [('1'*1024,'1'),('0','1'*256),('1'*256,'0')]:case(a,b,'directed_cases')
        rng=random.Random(20261102)
        for _ in range(128):
            a=''.join(rng.choice('01') for _ in range(rng.randint(1,48)))
            b=''.join(rng.choice('01') for _ in range(rng.randint(1,48)))
            case(a,b,'random_cases')
        assert observed==used
        report['execution_checks']={**totals,'max_operand_length':1024,
            'all_defined_nonterminal_entries_exercised':True,
            'full_traces_final_grids_and_heads_agree':True,
            'arithmetic_output_and_padding_match':True}
    if args.report:args.report.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in {'phase_states','checked_edges'}},indent=2))

if __name__=='__main__':main()
