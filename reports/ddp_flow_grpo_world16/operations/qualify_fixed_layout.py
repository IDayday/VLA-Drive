"""Bounded fixed-layout repeat/resume checks; preserves cross-layout FAIL."""
import json
import os
from pathlib import Path
import time
import torch
from scripts.cluster_flow_grpo.cluster import run
from scripts.cluster_flow_grpo.boundary_evidence import compare
root=Path('runs/resource_reallocation_v3')
def wait(name):
    path=root/(name+'_control/result.json');deadline=time.monotonic()+2400
    while not path.exists():
        if time.monotonic()>deadline:raise TimeoutError(name)
        time.sleep(3)
    row=json.loads(path.read_text())
    if row.get('status')!='PASS' or row.get('exit_codes')!=[0,0,0]:raise RuntimeError(row)
def check(a,b,name):
    torch.set_num_threads(8)
    final=root/(name+'.json');pending=final.with_suffix('.pending.json')
    if final.exists() or pending.exists():raise FileExistsError(final)
    try:result=compare(a,b,pending)
    finally:
        if pending.exists():os.replace(pending,final)
    print(name,result['status'],len(result['files']),flush=True)
wait('f16_split_cont')
# Launch the final one-update resume as comparison reads run on CPU.
from concurrent.futures import ThreadPoolExecutor
with ThreadPoolExecutor(max_workers=1) as pool:
    future=pool.submit(run,root/'f16_split_resume3_spec.json')
    check(root/'f16_split_resume/checkpoints/update_000002',
          root/'f16_split_cont/checkpoints/update_000002','f16_within_layout_repeat_comparison')
    future.result()
check(root/'f16_split_cont/checkpoints/update_000003',
      root/'f16_split_resume3/checkpoints/update_000003','f16_within_layout_resume_comparison')
