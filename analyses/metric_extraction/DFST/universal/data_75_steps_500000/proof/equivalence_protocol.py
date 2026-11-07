"""Finite observation overapproximation for a correctly encoded binary TM."""
import collections
import csv
from pathlib import Path

SYMBOLS = ('', '0', '1')
TRITS = ('L', 'R', 'S')
ROOT = Path(__file__).resolve().parent
SOURCE = ROOT

def read_table(path):
    result = {}
    with path.open() as stream:
        for row in csv.DictReader(stream):
            for symbol in ['', '0', '1', 'L', 'R', 'S', 'X', 'Y', 'Z']:
                raw = row[symbol or 'λ']
                if raw:
                    q,w,m=raw.split(' | ')
                    result[int(row['state']),symbol]=(int(q),'' if w=='λ' else w,m)
    return result

def spec(p, include_empty=False):
    kind,*v=p
    if kind=='initial':
        return 0,{s:('select_jump',0,i) for i,s in enumerate(SYMBOLS) if s or include_empty}
    if kind=='dispatch':
        e,=v
        return 0,{s:('select_jump',e,i) for i,s in enumerate(SYMBOLS)}
    if kind=='select_jump':
        e,r=v
        return 1+r,{'X':('select',e,r,e)}
    if kind=='select':
        e,r,j=v
        dest=('cross',e,r) if j==r else ('select',e,r,j+(1 if j<r else -1))
        return 1+r,{SYMBOLS[j]:dest}
    if kind=='cross':
        e,r=v
        return 4,{'X' if e==r else '':('skip_first',e)}
    if kind=='skip_first':
        e,=v
        return 4,{t:('skip_rest',e) for t in TRITS}
    if kind=='skip_rest':
        e,=v
        return 4,{**{t:('skip_rest',e) for t in TRITS},'0':('next_first',e)}
    if kind in {'next_first','next'}:
        e,=v
        edges={t:('match_jump',e,i) for i,t in enumerate(TRITS)}
        if kind=='next':edges['0']=('reset_jump',e)
        return 5,edges
    if kind=='match_jump':
        e,t=v
        return 6+t,{'X':('compare_first',e,t)}
    if kind=='compare_first':
        e,t=v
        return 6+t,{TRITS[j]:('resume',e,t) if j==t else
                    ('compare_down' if j<t else 'compare_up',t,j)
                    for j in range(3)}
    if kind in {'compare_down','compare_up'}:
        t,last=v
        step=1 if kind=='compare_down' else -1
        endpoint=0 if step==1 else 2
        return 6+t,{TRITS[j]:('resume',endpoint,t) if j==t else (kind,t,j)
                    for j in (last,last+step)}
    if kind=='resume':
        e,t=v
        return 6+t,{'Y':('next',e)}
    if kind=='reset_jump':
        e,=v
        return 9,{'X':('reset_back',e)}
    if kind=='reset_back':
        e,=v
        return 9,{**{t:('reset_back',e) for t in TRITS},'':('reset_resume',e)}
    if kind=='reset_resume':
        e,=v
        return 9,{'Y':('read_write',e)}
    if kind=='read_write':
        e,=v
        return 10,{s:('read_motion',e,i) for i,s in enumerate(SYMBOLS)}
    if kind=='read_motion':
        e,w=v
        return 11+w,{m:('apply',e,w,m) for m in TRITS}
    if kind=='apply':
        e,w,m=v
        ground=11+w if m=='S' else 1+2*w+(m=='R')
        return ground,{'Z':('halt',) if m=='S' else ('dispatch',e)}
    if kind=='halt':return 14,{}
    raise AssertionError(p)

def check(source=SOURCE, *, include_empty=False, transitions=None):
    expert=read_table(source/'expert_transition_table.csv')
    machine=read_table(source/'minimized_transition_table.csv') if transitions is None else transitions
    initial=(('initial',),0)
    todo=collections.deque([initial]); seen={initial}; parents={initial:None};checked=[]
    while todo:
        p,q=todo.popleft();g,edges=spec(p,include_empty)
        for x,p1 in edges.items():
            gn,w,m=expert[g,x]
            q1,w1,m1=machine.get((q,x),(-1,'?','?'))
            if (gn,w,m)!=(spec(p1,include_empty)[0],w1,m1):
                trace=[]; cur=(p,q)
                while parents[cur] is not None:
                    cur,xprev=parents[cur];trace.append(xprev)
                raise AssertionError({'phase':p,'state':q,'read':x,'ground':(gn,w,m),
                                      'candidate':(q1,w1,m1),'protocol_trace':list(reversed(trace))+[x]})
            assert (q1==43)==(p1==('halt',))==(m=='S')
            checked.append((p,q,x,p1,q1,w,m))
            target=(p1,q1)
            if target not in seen:
                seen.add(target);todo.append(target);parents[target]=((p,q),x)
    return seen,checked

if __name__=='__main__':
    seen,edges=check()
    print('phases',len({p for p,q in seen}),'pairs',len(seen),'edges',len(edges))
    print('entries',len({(q,x) for p,q,x,*_ in edges}))
    groups=collections.defaultdict(set)
    for p,q in seen:groups[p[0],spec(p)[0]].add(q)
    for (kind,g),qs in sorted(groups.items()):print(kind,g,sorted(qs))
