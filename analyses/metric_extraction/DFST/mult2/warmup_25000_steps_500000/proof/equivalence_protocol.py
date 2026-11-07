"""Finite observation protocol for the exported 32-state mult_grid(2)."""
from pathlib import Path
from collections import deque
import csv

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT
P={};G={};NAMES={}
def phase(name,g,edges,description):
    P[name]=edges;G[name]=g;NAMES[name]=description
def bits(target):return {'0':target,'1':target}

phase('start',0,bits('scan_a'),'First left-operand digit')
phase('scan_a',0,{**bits('scan_a'),'x':'first_b'},'Scan left operand')
phase('first_b',0,bits('scan_b'),'First right-operand digit')
phase('scan_b',0,{**bits('scan_b'),'':'skip_last'},'Scan right operand')
phase('skip_last',1,bits('copy_decision'),'Skip last right-operand digit')
phase('copy_decision',2,{'0':'first_down_0','1':'first_down_1','x':'last_factor'},'Copy more, or start products')
for d in (0,1):
    phase(f'first_down_{d}',3+d,{'':f'first_write_{d}'},f'First copied {d}: descend')
    phase(f'first_write_{d}',6+d,{'':'copy_after_first'},f'First copied {d}: write')
phase('copy_after_first',9,{'0':'regular_down_0','1':'regular_down_1','x':'operator_down'},'Continue after first copy')
for d in (0,1):
    phase(f'regular_down_{d}',3+d,bits(f'regular_write_{d}'),f'Copy {d} above a lower digit')
    phase(f'regular_write_{d}',6+d,{'':'copy_loop'},f'Write copied {d}')
phase('copy_loop',9,{'0':'regular_down_0','1':'regular_down_1','x':'operator_down','':'copy_restart'},'Continue the copy')
phase('operator_down',5,bits('operator_write'),'Descend below operator')
phase('operator_write',8,{'':'first_a_copy'},'Write shifted operator')
phase('first_a_copy',9,{'0':'operator_neighbor_0','1':'operator_neighbor_1'},'Copy last left-operand digit')
for d in (0,1):
    phase(f'operator_neighbor_{d}',3+d,{'x':f'regular_write_{d}'},f'Copied {d} above lower operator')
phase('copy_restart',0,bits('scan_a'),'Begin scan on copied row')

phase('last_factor',10,{'0':'erase_single_0','1':'erase_single_1'},'Read the last remaining factor bit')
for d in (0,1):
    phase(f'erase_single_{d}',11+d,{'x':f'product_first_{d}'},f'Erase operator for factor {d}')
    phase(f'product_first_{d}',13+d,bits(f'product_rest_{d}'),f'First digit of product by {d}')
    phase(f'product_rest_{d}',13+d,{**bits(f'product_rest_{d}'),'':'product_scan'},f'Continue product by {d}')
phase('product_scan',15,{**bits('product_scan'),'':'inspect_above'},'Scan completed partial product')
phase('inspect_above',16,{**bits('upper_operator'),'':'sum_entry'},'Inspect the row above')
phase('upper_operator',17,{'x':'upper_first_b'},'Cross upper operator')
phase('upper_first_b',17,bits('upper_scan_b'),'First bit in upper factor prefix')
phase('upper_scan_b',17,{**bits('upper_scan_b'),'':'upper_factor'},'Scan upper factor prefix')
phase('upper_factor',10,{'0':'erase_first_0','1':'erase_first_1'},'Read upper row factor bit')
for d in (0,1):
    phase(f'erase_first_{d}',11+d,bits(f'erase_rest_{d}'),f'First remaining bit to erase ({d})')
    phase(f'erase_rest_{d}',11+d,{**bits(f'erase_rest_{d}'),'x':f'product_first_{d}'},f'Erase factor prefix ({d})')

phase('sum_entry',18,{'':'initial_left'},'Start summing the product rows')
phase('initial_left',19,{'0':'left_after_0','1':'left_after_1'},'First digit of initial left scan')
phase('left_after_0',19,{'0':'left_after_0','1':'left_after_1','':'inspect_below'},'Left scan after reading zero')
phase('left_after_1',19,{'0':'left_after_0','1':'left_after_1'},'Left scan after reading one')
phase('inspect_below',20,{'0':'lower_first','':'finish'},'Inspect fresh lower leading cell')
phase('lower_first',21,bits('lower_scan'),'First digit after lower leading zero')
phase('lower_scan',21,{**bits('lower_scan'),'':'align_first'},'Scan the lower partial product')
phase('align_first',28,bits('align_next'),'First upper alignment digit')
phase('align_next',28,{**bits('align_rest'),'':'top_first'},'Next upper alignment cell')
phase('align_rest',28,{**bits('align_rest'),'':'top_first'},'Finish aligning the upper row')
phase('top_first',22,{'0':'first_add_0','1':'first_add_1'},'Read upper units digit')
for d in (0,1):
    phase(f'first_add_{d}',24+d,{'':'return_0'},f'Write units digit {d} below')
phase('return_0',23,{'':'top_0'},'Return above, carry zero')
phase('top_0',22,{'0':'add_0','1':'add_1','':'leading_zero'},'Read upper bit, carry zero')
phase('add_0',24,{'':'return_0','0':'return_0','1':'return_0'},'Add value zero below')
phase('add_1',25,{'':'return_0','0':'return_0','1':'return_1'},'Add value one below')
phase('return_1',27,{'':'top_1'},'Return above, carry one')
phase('top_1',26,{'0':'add_1_nonblank','1':'add_2_nonblank','':'carry_extension'},'Read upper bit, carry one')
phase('add_1_nonblank',25,{'0':'return_0','1':'return_1'},'Add one inside lower digit block')
phase('add_2_nonblank',29,{'0':'return_1','1':'return_1'},'Add two inside lower digit block')
phase('leading_zero',19,{'0':'left_of_sum'},'Preserve unneeded leading zero')
phase('left_of_sum',19,{'':'inspect_below'},'Blank to the left of merged row')
phase('carry_extension',25,{'0':'extension_return'},'Apply final carry to leading zero')
phase('extension_return',23,{'':'extension_top'},'Return after final carry')
phase('extension_top',22,{'':'left_of_sum'},'Step down left of merged row')
phase('finish',30,{'':'halt'},'Emit final stop')
phase('halt',31,{},'Halted')

def read_table(path):
    result={}
    for row in csv.DictReader(path.open()):
        for x in ['', '0','1','x']:
            if row[x or 'λ']:
                q,w,m=row[x or 'λ'].split(' | ')
                result[int(row['state']),x]=(int(q),'' if w=='λ' else w,m)
    return result

def check(source=SOURCE):
    expert=read_table(source/'expert_transition_table.csv')
    machine=read_table(source/'minimized_transition_table.csv')
    seen={('start',0)};todo=deque(seen);checked=[];parents={('start',0):None}
    while todo:
        p,q=todo.popleft()
        for x,p1 in P[p].items():
            gn,w,m=expert[G[p],x]
            cn,ww,mm=machine.get((q,x),(-1,'?','?'))
            if (gn,w,m)!=(G[p1],ww,mm):
                trace=[];node=p,q
                while parents[node] is not None:
                    node,last=parents[node];trace.append(last)
                raise AssertionError((p,q,x,(gn,w,m),(cn,ww,mm),list(reversed(trace))+[x]))
            assert (cn==60)==(p1=='halt')
            checked.append((p,q,x,p1,cn,w,m))
            node=p1,cn
            if node not in seen:seen.add(node);todo.append(node);parents[node]=((p,q),x)
    return seen,checked

if __name__=='__main__':
    seen,edges=check()
    print('phases',len(P),'pairs',len(seen),'checks',len(edges),'entries',len({(q,x) for p,q,x,*_ in edges}))
    for p in P:print(p,G[p],sorted(q for pp,q in seen if pp==p))
