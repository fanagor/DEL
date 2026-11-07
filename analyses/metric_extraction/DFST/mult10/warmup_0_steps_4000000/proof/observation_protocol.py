"""Finite observation protocol for valid decimal grid multiplications."""
from collections import deque
D=list(map(str,range(10)))
P={};G={};NAMES={}
def phase(name,g,edges,desc=''):
 assert name not in P
 P[name]=edges;G[name]=g;NAMES[name]=desc or name

def digits(target):return {d:target for d in D}
phase('start',0,digits('scan_a'))
phase('scan_a',0,{**digits('scan_a'),'x':'first_b'})
phase('first_b',0,digits('scan_b'))
phase('scan_b',0,{**digits('scan_b'),'':'skip_last'})
phase('skip_last',1,digits('copy_decision'))
phase('copy_decision',2,{**{d:f'first_down_{d}' for d in D},'x':'last_factor'})
for d in D:
 phase(f'first_down_{d}',3+int(d),{'':f'first_write_{d}'})
 phase(f'first_write_{d}',14+int(d),{'':'copy_after_first'})
phase('copy_after_first',25,{**{d:f'regular_down_{d}' for d in D},'x':'operator_down'})
for d in D:
 phase(f'regular_down_{d}',3+int(d),digits(f'regular_write_{d}'))
 phase(f'regular_write_{d}',14+int(d),{'':'copy_loop'})
phase('copy_loop',25,{**{d:f'regular_down_{d}' for d in D},'x':'operator_down','':'copy_restart'})
phase('operator_down',13,digits('operator_write'))
phase('operator_write',24,{'':'first_a_copy'})
phase('first_a_copy',25,{d:f'operator_neighbor_{d}' for d in D})
for d in D:phase(f'operator_neighbor_{d}',3+int(d),{'x':f'regular_write_{d}'})
phase('copy_restart',0,digits('scan_a'))
phase('last_factor',26,{d:f'erase_single_{d}' for d in D})
for d in D:
 phase(f'erase_single_{d}',27+int(d),{'x':f'product_first_{d}'})
 if d=='0':
  phase('product_first_0',37,digits('product_rest_0'))
  phase('product_rest_0',37,{**digits('product_rest_0'),'':'product_scan'})
 else:
  f=int(d)
  phase(f'product_first_{d}',38+f*(f-1)//2,{x:f'product_{f}_{f*int(x)//10}' for x in D})
  for c in range(f):
   phase(f'product_{f}_{c}',38+f*(f-1)//2+c,{**{x:f'product_{f}_{(f*int(x)+c)//10}' for x in D},'':'product_scan'})
phase('product_scan',83,{**digits('product_scan'),'':'inspect_above'})
phase('inspect_above',84,{**digits('upper_operator'),'':'sum_entry'})
phase('upper_operator',85,{'x':'upper_first_b'})
phase('upper_first_b',85,digits('upper_scan_b'))
phase('upper_scan_b',85,{**digits('upper_scan_b'),'':'upper_factor'})
phase('upper_factor',26,{d:f'erase_first_{d}' for d in D})
for d in D:
 phase(f'erase_first_{d}',27+int(d),digits(f'erase_rest_{d}'))
 phase(f'erase_rest_{d}',27+int(d),{**digits(f'erase_rest_{d}'),'x':f'product_first_{d}'})
phase('sum_entry',86,{'':'initial_left'})
phase('initial_left',87,digits('sum_left_scan'))
phase('sum_left_scan',87,{**digits('sum_left_scan'),'':'inspect_below'})
phase('inspect_below',88,{**{d:'lower_first' for d in D[:9]},'':'finish'})
phase('lower_first',89,digits('lower_scan'))
phase('lower_scan',89,{**digits('lower_scan'),'':'align_first'})
phase('align_first',104,digits('align_rest'))
phase('align_rest',104,{**digits('align_rest'),'':'top_first'})
phase('top_first',90,{d:f'first_add_{d}' for d in D})
for d in D:phase(f'first_add_{d}',92+int(d),{'':'return_empty'})
phase('return_empty',91,{'':'top_nonblank'})
phase('top_nonblank',90,{d:f'add_{d}' for d in D})
phase('return_0',91,{'':'top_0'})
phase('top_0',90,{**{d:f'add_{d}' for d in D},'':'leading_digit'})
for v in range(10):
 phase(f'add_{v}',92+v,{'':'return_empty',**{x:f'return_{int(v+int(x)>=10)}' for x in D}})
phase('return_1',103,{'':'top_1'})
phase('top_1',102,{**{d:f'add_nonblank_{int(d)+1}' for d in D},'':'carry_extension'})
for v in range(1,11):
 phase(f'add_nonblank_{v}',92+v if v<10 else 105,{x:f'return_{int(v+int(x)>=10)}' for x in D})
phase('leading_digit',87,{d:'left_of_sum' for d in D[:9]})
phase('left_of_sum',87,{'':'inspect_below'})
phase('carry_extension',93,{d:'extension_return' for d in D[:9]})
phase('extension_return',91,{'':'extension_top'})
phase('extension_top',90,{'':'left_of_sum'})
phase('finish',106,{'':'halt'})
phase('halt',107,{})
