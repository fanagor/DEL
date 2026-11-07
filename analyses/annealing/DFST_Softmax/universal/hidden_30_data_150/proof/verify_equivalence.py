#!/usr/bin/env python3
"""Check the universal-controller certificate and independent encoded-TM runs.

The human proof establishes protocol coverage. This script checks all finite
local obligations and supplementary executions; bounded runs are not evidence
that an arbitrary simulated TM will eventually halt.
"""
from pathlib import Path
import argparse
import ast
import collections
import hashlib
import itertools
import json
import random
from equivalence_protocol import spec, read_table, check, SYMBOLS, TRITS

ROOT=Path(__file__).resolve().parent

REQUIRED_FILES = ('equivalence_protocol.py', 'expert_transition_table.csv', 'minimized_fst.json', 'minimized_transition_table.csv', 'minimized_transition_table.pdf', 'proof_reference_experts.py', 'proof_reference_teleporting.py', 'verify_equivalence.py')

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


def load_definitions():
    env={'Sequence':list}
    for filename,names in [('proof_reference_experts.py',{'Memory','encode_tm'}),
                            ('proof_reference_teleporting.py',{'universal_tp'})]:
        tree=ast.parse((ROOT/filename).read_text())
        selected=[node for node in tree.body if isinstance(node,(ast.ClassDef,ast.FunctionDef)) and node.name in names]
        assert {n.name for n in selected}==names
        exec(compile(ast.Module(body=selected,type_ignores=[]),filename,'exec'),env)
    return env['Memory'],env['encode_tm'],env['universal_tp']

def case(machine,expert,Memory,encode,program,word,limit):
    rows=encode(program,word)
    n=len(program)//3
    base={(x,-j):v for j,row in enumerate(rows[:-1]) for x,v in enumerate(row) if v and v!='X'}
    head=(1,-3*n)
    left=Memory(rows,head,True);right=Memory(rows,head,True)
    q=g=0
    p=('dispatch',0)
    tape={i+1:d for i,d in enumerate(word)}
    tmstate=0;tmhead=1;simulated=0;micro=0
    while simulated<limit:
        read=tape.get(tmhead,'')
        next_tm,written_tm,motion_tm=program[tmstate,read]
        budget=30*(3*n+20*len(rows[0])+20)
        for _ in range(budget):
            x=left.read(); assert x==right.read()
            assert spec(p)[0]==g and x in spec(p)[1],(p,x,n,word)
            p1=spec(p)[1][x]
            q1,w,m=machine[q,x];g1,ww,mm=expert[g,x]
            assert (w,m)==(ww,mm),(n,word,micro,q,g,x)
            left.write(w);left.move(m)
            right.write(ww);right.move(mm)
            assert left.head==right.head and left.runes==right.runes
            q,g,p=q1,g1,p1;micro+=1
            assert (q==27)==(g==14)==(m=='S')
            if g in {0,14}:break
        else:raise AssertionError('simulated step exceeded its conservative bound')
        if written_tm:tape[tmhead]=written_tm
        else:tape.pop(tmhead,None)
        tmhead+= {'L':-1,'R':1,'S':0}[motion_tm]
        tmstate=next_tm;simulated+=1
        assert left.cells==right.cells
        assert left.head==(tmhead,-3*n)
        actual={x:s for (x,y),s in left.cells.items() if y==-3*n}
        assert actual==tape,(n,word,actual,tape)
        xp,hidden=left.runes['X']
        row=-xp[1]
        assert xp[0]==1 and row//3==tmstate and row%3 in {0,2} and hidden==''
        assert 'Z' not in left.runes
        logical={pos:s for pos,s in left.cells.items() if pos[1]!=-3*n}
        for rune,(pos,hidden) in left.runes.items():
            logical.pop(pos,None)
            if hidden:logical[pos]=hidden
        assert logical==base
        if motion_tm=='S':
            assert g==14
            return micro,simulated,True
        assert g==0 and p==('dispatch',row%3)
    return micro,simulated,False

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--quick',action='store_true')
    parser.add_argument('--report',type=Path)
    args=parser.parse_args()
    verify_artifacts()
    Memory,encode,builder=load_definitions()
    expert=read_table(ROOT/'expert_transition_table.csv')
    machine=read_table(ROOT/'minimized_transition_table.csv')
    assert expert==builder({'','0','1'})
    exported=json.loads((ROOT/'minimized_fst.json').read_text())
    alphabet=['','0','1','L','R','S','X','Y','Z'];motions='LRDUXYZS'
    decoded={(t['state'],alphabet[t['input']]):(t['next_state'],alphabet[t['symbol']],motions[t['motion']]) for t in exported['transitions']}
    assert machine==decoded and exported['initial_state']==0 and exported['final_states']==[27]
    seen,edges=check(ROOT)
    assert (len({p for p,q in seen}),len(seen),len(edges))==(91,93,162)
    used={(q,x) for p,q,x,p1,q1,w,m in edges}
    assert len(used)==81
    report={'source_hashes_match':True,'csv_matches_json':True,'expert_equals_universal_tp':True,
            'protocol_phases':91,'phase_controller_pairs':93,'checked_protocol_edges':162,
            'covered_nonterminal_entries':81,
            'phase_states':[{'phase':p,'ground':spec(p)[0],'controller_states':sorted(q for pp,q in seen if pp==p)} for p in sorted({p for p,q in seen},key=repr)],
            'checked_edges':edges}
    if not args.quick:
        totals=collections.Counter()
        def test(program,word,limit,category):
            steps,simulated,halted=case(machine,expert,Memory,encode,program,word,limit)
            totals[category]+=1;totals['controller_transitions']+=steps
            totals['simulated_TM_steps']+=simulated
            totals['halted_cases' if halted else 'bounded_prefix_cases']+=1
        words=[''.join(w) for n in range(4) for w in itertools.product('01',repeat=n)]
        actions=list(itertools.product(SYMBOLS,TRITS))
        for rules in itertools.product(actions,repeat=3):
            program={(0,s):(0,w,m) for s,(w,m) in zip(SYMBOLS,rules)}
            for word in words:test(program,word,8,'exhaustive_one_state_cases')
        rng=random.Random(20261102)
        for n in [2,3,4,8,9,10,26,27,28,80,81,82,243,244,729,730]:
            for _ in range(8):
                program={(q,s):(rng.randrange(n),rng.choice(SYMBOLS),rng.choice(TRITS)) for q in range(n) for s in SYMBOLS}
                word=''.join(rng.choice('01') for _ in range(rng.randrange(65)))
                test(program,word,8,'random_cases')
        for n in [3,4,9,10,27,28,81,82,243,244,729,730,2187,2188]:
            program={(q,s):(0,s,'R') for q in range(n) for s in SYMBOLS}
            for s in SYMBOLS:
                program[0,s]=(n-1,'1','R')
                program[n-1,s]=(n//2,'0','L')
                program[n//2,s]=(0,'','R')
            for word in ['', '01', '101']:test(program,word,6,'directed_boundary_cases')
        report['execution_checks']={**totals,'largest_state_count':2188,'largest_ternary_width':8,
            'independent_grid_head_runes_and_direct_TM_checked':True,
            'bounded_prefix_note':'No eventual-halting claim is inferred from a bounded execution.'}
    if args.report:args.report.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in {'phase_states','checked_edges'}},indent=2))

if __name__=='__main__':main()
