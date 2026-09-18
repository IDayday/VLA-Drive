"""Wait for the already-running bounded diagnostic; atomically publish its comparison."""
import json
import os
from pathlib import Path
import time
import torch
from scripts.cluster_flow_grpo.boundary_evidence import compare
root = Path('runs/resource_reallocation_v3')
control = root/'f16_split_resume_control/result.json'
deadline = time.monotonic()+3600
while not control.exists():
    if time.monotonic()>deadline: raise TimeoutError('running diagnostic exceeded comparison deadline')
    time.sleep(3)
receipt = json.loads(control.read_text())
if receipt.get('status')!='PASS' or receipt.get('exit_codes')!=[0,0,0]:
    raise ValueError('bounded diagnostic failed; no numerical comparison or release')
output = root/'f16_split_resume_comparison.json'
temporary = output.with_suffix('.pending.json')
if output.exists() or temporary.exists(): raise FileExistsError('comparison attempt already exists')
torch.set_num_threads(8)
try:
    result = compare(root/'projected_expected/update_000002',
                     root/'f16_split_resume/checkpoints/update_000002', temporary)
finally:
    if temporary.exists(): os.replace(temporary, output)
print(json.dumps({'status':result['status'],'files':len(result['files'])}),flush=True)
