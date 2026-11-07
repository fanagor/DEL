"""Finite grid protocol and simulation certificate for the annealed mult2 controller.

Each phase restricts only the read symbol and names the next phase.
G is the exact reference state. The checker explores (phase, controller state)
and verifies shared write/move and the prescribed reference successor.
The accompanying proof establishes coverage on every valid multiplication.
"""
from pathlib import Path
from collections import deque
import csv

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT
HALT = 51
P = {}; G = {}; NAMES = {}
def phase(name, ground, edges, description):
    P[name] = edges; G[name] = ground; NAMES[name] = description

# Copying
phase('start', 0, {'0': 'scan_a', '1': 'scan_a'}, 'First left-operand digit')
phase('scan_a', 0, {'0': 'scan_a', '1': 'scan_a', 'x': 'first_b'}, 'Scan left operand')
phase('first_b', 0, {'0': 'scan_b_0', '1': 'scan_b_1'}, 'First right-operand digit')
phase('scan_b_0', 0, {'0': 'scan_b_0', '1': 'scan_b_1', '': 'skip_last_0'}, 'Scan factor; last bit 0')
phase('scan_b_1', 0, {'0': 'scan_b_0', '1': 'scan_b_1', '': 'skip_last_1'}, 'Scan factor; last bit 1')
phase('skip_last_0', 1, {'0': 'copy_decision_0'}, 'Revisit last factor bit 0')
phase('skip_last_1', 1, {'1': 'copy_decision_1'}, 'Revisit last factor bit 1')
phase('copy_decision_0', 2, {'0': 'first_down_0', '1': 'first_down_1', 'x': 'last_factor_0'}, 'Copy or use remembered bit 0')
phase('copy_decision_1', 2, {'0': 'first_down_0', '1': 'first_down_1', 'x': 'last_factor_1'}, 'Copy or use remembered bit 1')
phase('first_down_0', 3, {'': 'first_write_0'}, 'First copied 0: descend')
phase('first_write_0', 6, {'': 'copy_after_first'}, 'First copied 0: write')
phase('first_down_1', 4, {'': 'first_write_1'}, 'First copied 1: descend')
phase('first_write_1', 7, {'': 'copy_after_first'}, 'First copied 1: write')
phase('copy_after_first', 9, {'0': 'regular_down_0', '1': 'regular_down_1', 'x': 'operator_down'}, 'Continue after first copy')
phase('regular_down_0', 3, {'0': 'regular_write_0', '1': 'regular_write_0'}, 'Copy 0 above a lower digit')
phase('regular_write_0', 6, {'': 'copy_loop'}, 'Write copied 0')
phase('regular_down_1', 4, {'0': 'regular_write_1', '1': 'regular_write_1'}, 'Copy 1 above a lower digit')
phase('regular_write_1', 7, {'': 'copy_loop'}, 'Write copied 1')
phase('copy_loop', 9, {'0': 'regular_down_0', '1': 'regular_down_1', 'x': 'operator_down', '': 'copy_restart'}, 'Continue the copy')
phase('operator_down', 5, {'0': 'operator_write', '1': 'operator_write'}, 'Descend below operator')
phase('operator_write', 8, {'': 'first_a_copy'}, 'Write shifted operator')
phase('first_a_copy', 9, {'0': 'operator_neighbor_0', '1': 'operator_neighbor_1'}, 'Copy last left-operand digit')
phase('operator_neighbor_0', 3, {'x': 'regular_write_0'}, 'Copied 0 above lower operator')
phase('operator_neighbor_1', 4, {'x': 'regular_write_1'}, 'Copied 1 above lower operator')
phase('copy_restart', 0, {'0': 'scan_a', '1': 'scan_a'}, 'Begin scan on copied row')

# Products
phase('last_factor_0', 10, {'0': 'erase_single_0'}, 'Select remembered factor bit 0')
phase('last_factor_1', 10, {'1': 'erase_single_1'}, 'Select remembered factor bit 1')
phase('erase_single_0', 11, {'x': 'product_first_0'}, 'Erase operator for factor 0')
phase('erase_single_1', 12, {'x': 'product_first_1'}, 'Erase operator for factor 1')
phase('product_first_0', 13, {'0': 'product_rest_0', '1': 'product_rest_0'}, 'First digit of product by 0')
phase('product_rest_0', 13, {'0': 'product_rest_0', '1': 'product_rest_0', '': 'product_scan_first'}, 'Continue product by 0')
phase('product_first_1', 14, {'0': 'product_rest_1', '1': 'product_rest_1'}, 'First digit of product by 1')
phase('product_rest_1', 14, {'0': 'product_rest_1', '1': 'product_rest_1', '': 'product_scan_first'}, 'Continue product by 1')
phase('product_scan_first', 15, {'0': 'product_scan_one_0', '1': 'product_scan_one_1'}, 'First digit after product padding')
phase('product_scan_one_0', 15, {'0': 'product_scan_0', '1': 'product_scan_1', '': 'inspect_above_one_0'}, 'After one product digit 0')
phase('product_scan_one_1', 15, {'0': 'product_scan_0', '1': 'product_scan_1', '': 'inspect_above_one_1'}, 'After one product digit 1')
phase('product_scan_0', 15, {'0': 'product_scan_0', '1': 'product_scan_1', '': 'inspect_above_0'}, 'Product scan; last digit 0')
phase('product_scan_1', 15, {'0': 'product_scan_0', '1': 'product_scan_1', '': 'inspect_above_1'}, 'Product scan; last digit 1')
phase('inspect_above_one_0', 16, {'0': 'upper_operator', '1': 'upper_operator', '': 'sum_entry_one_0'}, 'Inspect above one-digit product 0')
phase('inspect_above_one_1', 16, {'0': 'upper_operator', '1': 'upper_operator', '': 'sum_entry_one_1'}, 'Inspect above one-digit product 1')
phase('inspect_above_0', 16, {'0': 'upper_operator', '1': 'upper_operator', '': 'sum_entry_0'}, 'Inspect above; last digit 0')
phase('inspect_above_1', 16, {'0': 'upper_operator', '1': 'upper_operator', '': 'sum_entry_1'}, 'Inspect above; last digit 1')
phase('upper_operator', 17, {'x': 'upper_first_b'}, 'Cross upper operator')
phase('upper_first_b', 17, {'0': 'upper_scan_b_0', '1': 'upper_scan_b_1'}, 'First bit in upper factor prefix')
phase('upper_scan_b_0', 17, {'0': 'upper_scan_b_0', '1': 'upper_scan_b_1', '': 'upper_factor_0'}, 'Scan factor; last bit 0')
phase('upper_scan_b_1', 17, {'0': 'upper_scan_b_0', '1': 'upper_scan_b_1', '': 'upper_factor_1'}, 'Scan factor; last bit 1')
phase('upper_factor_0', 10, {'0': 'erase_first_0'}, 'Select remembered factor bit 0')
phase('upper_factor_1', 10, {'1': 'erase_first_1'}, 'Select remembered factor bit 1')
phase('erase_first_0', 11, {'0': 'erase_rest_0', '1': 'erase_rest_0'}, 'First remaining bit to erase (0)')
phase('erase_rest_0', 11, {'0': 'erase_rest_0', '1': 'erase_rest_0', 'x': 'product_first_0'}, 'Erase factor prefix (0)')
phase('erase_first_1', 12, {'0': 'erase_rest_1', '1': 'erase_rest_1'}, 'First remaining bit to erase (1)')
phase('erase_rest_1', 12, {'0': 'erase_rest_1', '1': 'erase_rest_1', 'x': 'product_first_1'}, 'Erase factor prefix (1)')

# Initial sum and alignment
phase('sum_entry_one_0', 18, {'': 'initial_left_one_0'}, 'Begin one-digit sum; last bit 0')
phase('sum_entry_one_1', 18, {'': 'initial_left_one_1'}, 'Begin one-digit sum; last bit 1')
phase('sum_entry_0', 18, {'': 'initial_left_0'}, 'Begin sum; remember last digit 0')
phase('sum_entry_1', 18, {'': 'initial_left_1'}, 'Begin sum; remember last digit 1')
phase('initial_left_one_0', 19, {'0': 'initial_left_padding'}, 'Revisit sole product digit 0')
phase('initial_left_one_1', 19, {'1': 'initial_left_padding'}, 'Revisit sole product digit 1')
phase('initial_left_0', 19, {'0': 'initial_left_second'}, 'Revisit product units digit 0')
phase('initial_left_1', 19, {'1': 'initial_left_second'}, 'Revisit product units digit 1')
phase('initial_left_padding', 19, {'0': 'initial_left_boundary'}, 'Read the sole leading zero')
phase('initial_left_boundary', 19, {'': 'inspect_below'}, 'Left boundary after sole digit')
phase('initial_left_second', 19, {'0': 'initial_left_third', '1': 'initial_left_third'}, 'Second digit of initial left scan')
phase('initial_left_third', 19, {'0': 'left_after_0', '1': 'left_after_1'}, 'Third digit of initial left scan')
phase('left_after_0', 19, {'0': 'left_after_0', '1': 'left_after_1', '': 'inspect_below'}, 'Left scan after reading zero')
phase('left_after_1', 19, {'0': 'left_after_0', '1': 'left_after_1'}, 'Left scan after reading one')
phase('inspect_below', 20, {'0': 'lower_first', '': 'finish'}, 'Inspect fresh lower leading cell')
phase('lower_first', 21, {'0': 'lower_scan', '1': 'lower_scan'}, 'First digit after lower leading zero')
phase('lower_scan', 21, {'0': 'lower_scan', '1': 'lower_scan', '': 'align_first'}, 'Scan the lower partial product')
phase('align_first', 28, {'0': 'align_next', '1': 'align_next'}, 'First upper alignment digit')
phase('align_next', 28, {'0': 'align_rest', '1': 'align_rest', '': 'top_first'}, 'Next upper alignment cell')
phase('align_rest', 28, {'0': 'align_rest', '1': 'align_rest', '': 'top_first'}, 'Finish aligning the upper row')

# Column addition and stopping
phase('top_first', 22, {'0': 'first_add_0', '1': 'first_add_1'}, 'Read upper units digit')
phase('first_add_0', 24, {'': 'first_return'}, 'Write units digit 0 below')
phase('first_add_1', 25, {'': 'first_return'}, 'Write units digit 1 below')
phase('return_0', 23, {'': 'top_0'}, 'Return above, carry zero')
phase('top_0', 22, {'0': 'add_0', '1': 'add_1', '': 'leading_zero'}, 'Read upper bit, carry zero')
phase('add_0', 24, {'': 'gap_return', '0': 'return_0', '1': 'return_0'}, 'Add value zero below')
phase('add_1', 25, {'': 'gap_return', '0': 'return_0', '1': 'return_1'}, 'Add value one below')
phase('return_1', 27, {'': 'top_1'}, 'Return above, carry one')
phase('top_1', 26, {'0': 'add_1_nonblank', '1': 'add_2_nonblank', '': 'carry_extension'}, 'Read upper bit, carry one')
phase('add_1_nonblank', 25, {'0': 'return_0', '1': 'return_1'}, 'Add one inside lower digit block')
phase('add_2_nonblank', 29, {'0': 'return_1', '1': 'return_1'}, 'Add two inside lower digit block')
phase('leading_zero', 19, {'0': 'left_of_sum'}, 'Preserve unneeded leading zero')
phase('left_of_sum', 19, {'': 'inspect_below'}, 'Blank to the left of merged row')
phase('carry_extension', 25, {'0': 'extension_return'}, 'Apply final carry to leading zero')
phase('extension_return', 23, {'': 'extension_top'}, 'Return after final carry')
phase('extension_top', 22, {'': 'left_of_sum'}, 'Step down left of merged row')
phase('first_return', 23, {'': 'top_second'}, 'Return after units digit')
phase('top_second', 22, {'0': 'add_0', '1': 'add_1'}, 'Read the second upper digit')
phase('gap_return', 23, {'': 'top_gap'}, 'Return from the lower blank gap')
phase('top_gap', 22, {'0': 'add_0', '1': 'add_1'}, 'Upper digit after lower blank gap')
phase('finish', 30, {'': 'halt'}, 'Emit final stop')
phase('halt', 31, {}, 'Halted')

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
            assert (cn==HALT)==(p1=='halt')
            checked.append((p,q,x,p1,cn,w,m))
            node=p1,cn
            if node not in seen:seen.add(node);todo.append(node);parents[node]=((p,q),x)
    return seen,checked

if __name__=='__main__':
    seen,edges=check()
    print('phases',len(P),'pairs',len(seen),'checks',len(edges),'entries',len({(q,x) for p,q,x,*_ in edges}))
    for p in P:print(p,G[p],sorted(q for pp,q in seen if pp==p))
