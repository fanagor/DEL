#!/usr/bin/env python3
"""Reproduce the finite add10 certificate and independent grid-execution checks.

The default check uses only the Python standard library. --check-pdf additionally
compares all printed table cells, including undefined entries, using pypdf.
The human proof establishes protocol coverage for arbitrary operand lengths.
"""
from pathlib import Path
import argparse,ast,collections,hashlib,itertools,json,random,re
from equivalence_protocol import P,G,NAMES,read_table,check,HALT,DIGITS,ALPHABET
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
    for filename,names in [('proof_reference_grid.py',{'_complete','add_grid'}),('proof_reference_experts.py',{'Memory'})]:
        tree=ast.parse((ROOT/filename).read_text())
        nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
        assert {n.name for n in nodes}==names
        exec(compile(ast.Module(body=nodes,type_ignores=[]),filename,'exec'),env)
    return env['Memory'],env['add_grid']

def compare(machine,expert,Memory,a,b):
    left=Memory([a+'+'+b],(0,0),False);right=Memory([a+'+'+b],(0,0),False)
    q=g=0;p='start';used=set()
    for k in range(100*(len(a)+len(b)+2)**2):
        x=left.read();assert x==right.read()
        assert g==G[p] and x in P[p],(a,b,p,g,x)
        p1=P[p][x];q1,w,m=machine[q,x];g1,ww,mm=expert[g,x]
        assert (w,m)==(ww,mm),(a,b,k,g,q,x)
        used.add((q,x));left.write(w);left.move(m);right.write(ww);right.move(mm)
        assert left.head==right.head
        assert (q1==HALT)==(g1==18)==(m=='S')
        q,g,p=q1,g1,p1
        if m=='S':break
    else:raise AssertionError('step limit')
    assert left.cells==right.cells
    value=str(int(a)+int(b));width=max(len(a),len(b)+1,len(value))
    expected={(len(a)-width+i,0):d for i,d in enumerate(value.zfill(width))}
    assert left.cells==expected,(a,b,left.cells,expected)
    assert left.head==(len(a)-len(b)-1,0) and p=='halt'
    return k+1,used

def check_pdf(machine):
    from pypdf import PdfReader
    result={};seen_rows=set();missing=0
    for page in PdfReader(ROOT/'minimized_transition_table.pdf').pages:
        for line in page.extract_text(extraction_mode='layout').splitlines():
            state=re.match(r'^\s*(\d+)\s+',line)
            if not state or '|' not in line:continue
            state=int(state.group(1));assert state not in seen_rows;seen_rows.add(state)
            row=re.sub(r'^\s*\d+\s+(?:initial\s+|halt\s+)?','',line)
            cells=re.findall(r'(\d+)\s*\|\s*([λ+0-9])\s*\|\s*([LRDUS])|([-–—−])',row)
            assert len(cells)==12,(state,cells,line)
            for x,(qn,w,m,dash) in zip(ALPHABET,cells):
                if dash:missing+=1
                else:result[state,x]=(int(qn),'' if w=='λ' else w,m)
    assert seen_rows==set(range(52)) and result==machine
    return {'rows':len(seen_rows),'defined_entries':len(result),'undefined_entries':missing,'all_624_cells_match':True}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--quick',action='store_true')
    parser.add_argument('--check-pdf',action='store_true');parser.add_argument('--report',type=Path);args=parser.parse_args()
    verify_artifacts()
    Memory,builder=definitions();expert=read_table(ROOT/'expert_transition_table.csv');machine=read_table(ROOT/'minimized_transition_table.csv')
    assert expert==builder(10)
    exported=json.loads((ROOT/'minimized_partial.json').read_text());motions='LRDUS'
    decoded={(t['state'],ALPHABET[t['input']]):(t['next_state'],ALPHABET[t['symbol']],motions[t['motion']]) for t in exported['transitions']}
    assert machine==decoded and exported['initial_state']==0 and exported['final_states']==[HALT]
    assert exported['states']==list(range(52))
    seen,edges=check(ROOT);assert (len(P),len(seen),len(edges))==(59,60,266)
    used={(q,x) for p,q,x,*_ in edges};assert len(used)==223
    assert used=={(q,x) for q,x in machine if q!=HALT}
    report={'source_hashes_match':True,'csv_matches_json':True,'expert_equals_add_grid_10':True,
            'reference_state_count':19,'controller_state_count':52,'protocol_phases':59,
            'phase_state_pairs':60,'local_checks':266,'all_223_defined_nonterminal_entries_checked':True,
            'phase_states':[{'phase':p,'meaning':NAMES[p],'ground':G[p],'controller':sorted(q for pp,q in seen if pp==p)} for p in P],
            'checked_edges':edges}
    if args.check_pdf:report['printed_pdf_check']=check_pdf(machine)
    if not args.quick:
        totals=collections.Counter();observed=set()
        def case(a,b,category):
            steps,entries=compare(machine,expert,Memory,a,b);totals[category]+=1;totals['compared_pto_steps']+=steps;observed.update(entries)
        words=[''.join(w) for n in (1,2) for w in itertools.product(DIGITS,repeat=n)]
        for a,b in itertools.product(words,repeat=2):case(a,b,'exhaustive_pairs_lengths_1_through_2')
        for n in [1,2,3,8,32,128,512]:
            for a,b in [('9'*n,'1'),('0','9'*n),('0'*n,'0'*n),('9'*n,'9'*n),('0'*n+'1','9'*n),('1'+'0'*n,'1')]:case(a,b,'directed_cases')
        for a,b in [('9'*1024,'1'),('0'*1024,'1')]:case(a,b,'directed_cases')
        rng=random.Random(20261103)
        for _ in range(256):
            a=''.join(rng.choice(DIGITS) for _ in range(rng.randint(1,64)))
            b=''.join(rng.choice(DIGITS) for _ in range(rng.randint(1,64)));case(a,b,'random_cases')
        assert observed==used
        report['execution_checks']={**totals,'max_operand_length':1024,'all_223_nonterminal_entries_exercised':True,
                                    'complete_ptos_final_grids_heads_and_padded_decimal_sums_agree':True}
    if args.report:args.report.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in {'phase_states','checked_edges'}},indent=2))
if __name__=='__main__':main()
