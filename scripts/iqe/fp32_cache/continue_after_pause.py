"""Manual continuation only: never sends a signal or starts from generic weights."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from iqe.contracts import require
from iqe.io import atomic_json, read_json
from iqe.training.resume import inspect_checkpoint


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--updates',type=int,default=1000)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--output',default='/mnt/project/iqe-fp32-cache-production-runtime-3931fed/continuation')
    a=p.parse_args();require(11<=a.updates<=100000,'explicit bounded continuation window')
    repo=Path(__file__).resolve().parents[3];os.chdir(repo)
    script="""from pathlib import Path
import json
active=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:cmd=(p/'cmdline').read_bytes().replace(b'\\0',b' ').decode(errors='replace');cwd=str((p/'cwd').resolve())
 except (FileNotFoundError,ProcessLookupError,PermissionError):continue
 if cwd=='/mnt/project/DriveVLA-M0-iqe-20261009' and 'iqe.cli train-base' in cmd:active.append({'pid':int(p.name),'command':cmd})
print(json.dumps({'active':active}))
"""
    found=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','training-vla-zt2','python -'],
                          input=script,text=True,capture_output=True,check=True)
    live=json.loads(found.stdout);require(not live['active'],'ACTIVE_BASE_STILL_RUNNING: this script never stops or duplicates it')
    root=Path('/mnt/project/DriveVLA-M0-iqe-20261009/outputs/iqe/main/query_base')
    stopped=read_json(root/'result.json')
    require(stopped['status']=='STOPPED_AT_BOUNDARY','original Base lacks a confirmed cooperative optimizer-boundary pause')
    paths=sorted(root.glob('step_*.pt'))
    require(paths,'no actual trained Base checkpoint')
    # A newer file lacking its receipt fails, rather than rolling back silently.
    chosen=inspect_checkpoint(paths[-1]);require(chosen['optimizer_step']==stopped['optimizer_steps'],
                                               'pause result does not bind newest complete state')
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    atomic_json(out/'SOURCE_PAUSE_INSPECTION.json',{'live_scan':live,'pause_result':stopped,'checkpoint':chosen})
    argv=[sys.executable,'-u','-m','torch.distributed.run','--standalone','--nproc_per_node=8','-m','iqe.vision_cache.cli',
          'resume-base-cached' if a.resume else 'train-base-cached','--config','configs/iqe/fp32_cache/source.yaml',
          '--production-config','configs/iqe/fp32_cache/continuation.yaml','--runtime','configs/iqe/performance_v2/combined_fp32.yaml',
          '--qualification','/mnt/project/iqe-fp32-cache-production-runtime-3931fed/CACHE_QUALIFICATION.json',
          '--output',str(out),'--mode','continuation','--stop-after',str(a.updates),'--warmup','10',
          '--checkpoint-every','1000','--hold-file','/mnt/project/iqe-fp32-cache-production-task-20261011/CACHE_BUILD_HOLD.json']
    if not a.resume:argv+=['--parent-checkpoint',chosen['path']]
    os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2'
    os.execv(sys.executable,argv)

if __name__=='__main__':main()
