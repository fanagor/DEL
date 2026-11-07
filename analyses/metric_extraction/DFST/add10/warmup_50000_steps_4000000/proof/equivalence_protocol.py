"""Finite valid-grid observation protocol for add_grid(10)."""
from pathlib import Path
from collections import deque
import csv
ROOT=Path(__file__).resolve().parent
SOURCE=ROOT
DIGITS=tuple('0123456789'); ALPHABET=('', '+', *DIGITS)
HALT=51
P={};G={};NAMES={}
def phase(name,g,edges,meaning):P[name]=edges;G[name]=g;NAMES[name]=meaning
def digits(target):return {d:target for d in DIGITS}
phase('start',0,digits('seek'),'Read first left-operand digit')
phase('seek',0,{**digits('seek'),'+':'initial_down'},'Find the input separator')
phase('initial_down',1,digits('place'),'Descend below the units digit')
phase('place',1,{'':'ensure'},'Place the working marker')
phase('ensure',16,{**digits('left_scan'),'':'adjacent'},'Ensure a digit above the marker')
phase('adjacent',2,digits('left_scan'),'Read the adjacent existing digit')
phase('left_scan',2,{**digits('left_scan'),'+':'right_start'},'Scan the accumulator to the separator')
phase('right_start',2,{**digits('right_scan'),'':'take_separator'},'Inspect the remaining right operand')
phase('right_scan',2,{**digits('right_scan'),'':'take_digit'},'Scan the remaining right operand')
phase('take_digit',3,{d:'first_'+d for d in DIGITS},'Select the rightmost remaining digit')
for d in DIGITS:
    g=4+int(d)
    phase('first_'+d,g,{'':'second_'+d},'Transport '+d+': first blank')
    phase('second_'+d,g,{'':'walk_'+d},'Transport '+d+': second blank')
    phase('walk_'+d,g,{'':'walk_'+d,'+':'add_'+d},'Transport '+d+': reach the marker')
    phase('add_'+d,g,{e:('at_marker' if int(d)+int(e)<10 else 'carry') for e in DIGITS},'Add digit '+d+' above the marker')
phase('carry',14,{'9':'carry',**{d:'return_first' for d in DIGITS[:-1]},'':'return_first'},'Propagate a decimal carry')
phase('return_first',15,{'':'return_walk'},'First return cell after a carry')
phase('return_walk',15,{'':'return_walk','+':'place'},'Return to the working marker')
phase('at_marker',15,{'+':'place'},'Erase and shift the working marker')
phase('take_separator',3,{'+':'clean_first'},'Erase the input separator')
phase('clean_first',17,{'':'clean_walk'},'Begin cleanup below the separator')
phase('clean_walk',17,{'':'clean_walk','+':'stop'},'Erase the final working marker')
phase('stop',17,digits('halt'),'Preserve the final digit and stop')
phase('halt',18,{},'Halted')

def read_table(path):
    result={}
    for row in csv.DictReader(path.open()):
        for x in ALPHABET:
            cell=row[x or 'λ']
            if cell:
                q,w,m=cell.split(' | ');key=int(row['state']),x
                assert key not in result
                result[key]=int(q),'' if w=='λ' else w,m
    return result

def check(source=SOURCE):
    expert=read_table(source/'expert_transition_table.csv');machine=read_table(source/'minimized_transition_table.csv')
    seen={('start',0)};todo=deque(seen);edges=[];parents={('start',0):None}
    while todo:
        p,q=todo.popleft()
        for x,p1 in P[p].items():
            gn,w,m=expert[G[p],x];cn,ww,mm=machine.get((q,x),(-1,'?','?'))
            if (gn,w,m)!=(G[p1],ww,mm):
                trace=[];node=p,q
                while parents[node] is not None:node,last=parents[node];trace.append(last)
                raise AssertionError((p,q,x,(gn,w,m),(cn,ww,mm),list(reversed(trace))+[x]))
            assert (cn==HALT)==(p1=='halt')==(m=='S')
            edges.append((p,q,x,p1,cn,w,m));node=p1,cn
            if node not in seen:seen.add(node);todo.append(node);parents[node]=((p,q),x)
    return seen,edges

if __name__=='__main__':
    seen,edges=check();print('phases',len(P),'pairs',len(seen),'checks',len(edges),'entries',len({(q,x) for p,q,x,*_ in edges}))
    for p in P:print(p,G[p],sorted(q for pp,q in seen if pp==p))
