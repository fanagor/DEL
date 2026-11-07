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
import re
from equivalence_protocol import spec, read_table, check, SYMBOLS, TRITS

ROOT=Path(__file__).resolve().parent

REQUIRED_FILES = ('equivalence_protocol.py', 'expert_transition_table.csv', 'minimized_partial.json', 'minimized_transition_table.csv', 'minimized_transition_table.pdf', 'proof_reference_experts.py', 'proof_reference_teleporting.py', 'verify_equivalence.py')

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
    assert word, "The original extracted controller requires nonempty input"
    rows=encode(program,word)
    n=len(program)//3
    base={(x,-j):v for j,row in enumerate(rows[:-1]) for x,v in enumerate(row) if v and v!='X'}
    head=(1,-3*n)
    left=Memory(rows,head,True);right=Memory(rows,head,True)
    q=g=0;used=set()
    p=('initial',)
    tape={i+1:d for i,d in enumerate(word)}
    tmstate=0;tmhead=1;simulated=0;micro=0
    while simulated<limit:
        read=tape.get(tmhead,'')
        next_tm,written_tm,motion_tm=program[tmstate,read]
        budget=30*(3*n+20*len(rows[0])+20)
        for _ in range(budget):
            x=left.read(); assert x==right.read()
            used.add((q,x))
            assert spec(p)[0]==g and x in spec(p)[1],(p,x,n,word)
            p1=spec(p)[1][x]
            q1,w,m=machine[q,x];g1,ww,mm=expert[g,x]
            assert (w,m)==(ww,mm),(n,word,micro,q,g,x)
            left.write(w);left.move(m)
            right.write(ww);right.move(mm)
            assert left.head==right.head and left.runes==right.runes
            q,g,p=q1,g1,p1;micro+=1
            assert (q==43)==(g==14)==(m=='S')
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
            return micro,simulated,True,used
        assert g==0 and p==('dispatch',row%3)
    return micro,simulated,False,used

def empty_input_counterexample(machine,expert,Memory,encode):
    program={(0,s):(0,s,'S') for s in SYMBOLS}
    rows=encode(program,'');memory=Memory(rows,(1,-3),True)
    assert memory.read()=='' and (0,'') not in machine
    assert expert[0,'']==(1,'Z','X')
    g=0;trace=[]
    for _ in range(100):
        x=memory.read();gn,w,m=expert[g,x]
        trace.append([x,w,m]);memory.write(w);memory.move(m);g=gn
        if m=='S':break
    else:raise AssertionError('counterexample expert did not halt')
    assert g==14 and len(trace)==17
    return {'states':1,'input':'','initial_head':[1,-3],
            'program':[{'state':0,'read':s,'next_state':0,'write':s,'motion':'S'} for s in SYMBOLS],
            'encoded_rows':rows,'first_read':'','extracted_initial_state':0,
            'extracted_first_transition':None,'expert_first_transition':[1,'Z','X'],
            'expert_pto':trace,'expert_halts_after_controller_steps':len(trace),
            'failure':'The original partial controller has no transition for state 0 and blank.'}

def check_pdf(machine):
    from pypdf import PdfReader
    alphabet=['','0','1','L','R','S','X','Y','Z'];result={};seen_rows=set();missing=0
    for page in PdfReader(ROOT/'minimized_transition_table.pdf').pages:
        for line in page.extract_text(extraction_mode='layout').splitlines():
            state=re.match(r'^\s*(\d+)\s+',line)
            if not state or '|' not in line:continue
            state=int(state.group(1));assert state not in seen_rows;seen_rows.add(state)
            row=re.sub(r'^\s*\d+\s+(?:initial\s+|halt\s+)?','',line)
            cells=re.findall(r'(\d+)\s*\|\s*([λ01LRSXYZ])\s*\|\s*([LRDUXYZS])|([-–—−])',row)
            assert len(cells)==9,(state,cells,line)
            for x,(qn,w,m,dash) in zip(alphabet,cells):
                if dash:missing+=1
                else:result[state,x]=(int(qn),'' if w=='λ' else w,m)
    assert seen_rows==set(range(44)) and result==machine
    return {'rows':44,'defined_entries':len(result),'undefined_entries':missing,'all_396_cells_match':True}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--quick',action='store_true')
    parser.add_argument('--check-pdf',action='store_true')
    parser.add_argument('--counterexample',type=Path)
    parser.add_argument('--report',type=Path)
    args=parser.parse_args()
    verify_artifacts()
    Memory,encode,builder=load_definitions()
    expert=read_table(ROOT/'expert_transition_table.csv')
    machine=read_table(ROOT/'minimized_transition_table.csv')
    assert expert==builder({'','0','1'})
    exported=json.loads((ROOT/'minimized_partial.json').read_text())
    alphabet=['','0','1','L','R','S','X','Y','Z'];motions='LRDUXYZS'
    decoded={(t['state'],alphabet[t['input']]):(t['next_state'],alphabet[t['symbol']],motions[t['motion']]) for t in exported['transitions']}
    assert machine==decoded and exported['initial_state']==0 and exported['final_states']==[43]
    seen,edges=check(ROOT)
    assert (len({p for p,q in seen}),len(seen),len(edges))==(92,96,160)
    used={(q,x) for p,q,x,p1,q1,w,m in edges}
    assert len(used)==73
    assert used=={(q,x) for q,x in machine if q!=43}
    report={'source_hashes_match':True,'csv_matches_json':True,'expert_equals_universal_tp':True,
            'scope':'All correctly encoded binary TMs with nonempty binary inputs',
            'controller_states':44,'expert_states':15,
            'protocol_phases':92,'phase_controller_pairs':96,'checked_protocol_edges':160,
            'all_73_defined_nonterminal_entries_checked':True,
            'phase_states':[{'phase':p,'ground':spec(p)[0],'controller_states':sorted(q for pp,q in seen if pp==p)} for p in sorted({p for p,q in seen},key=repr)],
            'checked_edges':edges}
    witness=empty_input_counterexample(machine,expert,Memory,encode)
    report['unrestricted_claim']= 'False: empty input has no initial transition'
    report['empty_input_counterexample']=witness
    if args.counterexample:args.counterexample.write_text(json.dumps(witness,indent=2)+'\n')
    # Verify a hypothetical one-edge repair in memory; do not edit any export.
    augmented=dict(machine);augmented[0,'']=(18,'Z','X')
    repaired_seen,repaired_edges=check(ROOT,include_empty=True,transitions=augmented)
    report['hypothetical_repair']={'added_entry':[0,'',18,'Z','X'],
        'original_controller_files_modified':False,'phase_state_pairs':len(repaired_seen),
        'local_checks':len(repaired_edges),'full_input_domain_certificate_passes':True}
    if args.check_pdf:report['printed_pdf_check']=check_pdf(machine)
    if not args.quick:
        totals=collections.Counter();observed=set()
        def test(program,word,limit,category):
            steps,simulated,halted,entries=case(machine,expert,Memory,encode,program,word,limit)
            observed.update(entries)
            totals[category]+=1;totals['controller_transitions']+=steps
            totals['simulated_TM_steps']+=simulated
            totals['halted_cases' if halted else 'bounded_prefix_cases']+=1
        words=[''.join(w) for n in range(1,4) for w in itertools.product('01',repeat=n)]
        actions=list(itertools.product(SYMBOLS,TRITS))
        for rules in itertools.product(actions,repeat=3):
            program={(0,s):(0,w,m) for s,(w,m) in zip(SYMBOLS,rules)}
            for word in words:test(program,word,8,'exhaustive_one_state_cases')
        rng=random.Random(20261103)
        for n in [2,3,4,8,9,10,26,27,28,80,81,82,243,244,729,730]:
            for _ in range(8):
                program={(q,s):(rng.randrange(n),rng.choice(SYMBOLS),rng.choice(TRITS)) for q in range(n) for s in SYMBOLS}
                word=''.join(rng.choice('01') for _ in range(rng.randrange(1,65)))
                test(program,word,8,'random_cases')
        for n in [3,4,9,10,27,28,81,82,243,244,729,730,2187,2188]:
            program={(q,s):(0,s,'R') for q in range(n) for s in SYMBOLS}
            for s in SYMBOLS:
                program[0,s]=(n-1,'1','R')
                program[n-1,s]=(n//2,'0','L')
                program[n//2,s]=(0,'','R')
            for word in ['0', '01', '101']:test(program,word,6,'directed_boundary_cases')
        assert observed==used
        report['execution_checks']={'all_73_nonterminal_entries_exercised':True,**totals,'largest_state_count':2188,'largest_ternary_width':8,
            'independent_grid_head_runes_and_direct_TM_checked':True,
            'bounded_prefix_note':'No eventual-halting claim is inferred from a bounded execution.'}
    if args.report:args.report.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in {'phase_states','checked_edges','empty_input_counterexample'}},indent=2))

if __name__=='__main__':main()
