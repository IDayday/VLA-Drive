"""Start exactly one registered line after complete assets and capacity admission."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import psutil
from starVLA.rl.constrained_flow_noise.contracts import require_dispatch,file_sha
from starVLA.rl.flow_grpo.transactions import atomic_json


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--variant',choices=['C_SDE','N_FLOW_NOISE'],required=True)
    p.add_argument('--run-root',required=True);p.add_argument('--population',required=True)
    p.add_argument('--output',required=True);p.add_argument('--source-lock',required=True)
    p.add_argument('--resume',help='Exact complete checkpoint; never reset optimizer, lambda or the raw cursor')
    a=p.parse_args();host=require_dispatch();root=Path(a.run_root);out=Path(a.output)
    guard=out.with_suffix('.launcher.lock').open('a');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
    lock=json.loads(Path(a.source_lock).read_text())
    def verify_sources():
        for name,digest in lock['sources'].items():
            if file_sha(name)!=digest:raise ValueError('Frozen formal source changed: '+name)
    verify_sources()
    if out.exists() and not a.resume:raise FileExistsError('Existing formal run requires explicit exact resume')
    progress=out.with_suffix('.launcher.json')
    required=[Path(x) for x in lock['required_acceptance']]
    while not (root/'full_TRAIN_assets.json').exists() or any(not p.exists() for p in required):
        require_dispatch();verify_sources()
        atomic_json(progress,dict(status='WAITING_FOR_COMPLETE_ASSETS_AND_ACCEPTANCE',host=host,variant=a.variant,
                    missing_acceptance=[str(p) for p in required if not p.exists()],full_assets_ready=(root/'full_TRAIN_assets.json').exists(),utc_unix=time.time()))
        time.sleep(30)
    for path in required:
        if json.loads(path.read_text())['status']!='PASS':raise ValueError('Required acceptance failed: '+str(path))
    assets=json.loads((root/'full_TRAIN_assets.json').read_text())
    if assets['status']!='COMPLETE':raise ValueError('Cache preparation did not finish correctly')
    budget=json.loads((root/'formal_budget.json').read_text())
    while True:
        require_dispatch();verify_sources()
        inventory=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free,utilization.gpu','--format=csv,noheader'],text=True)
        rows=[line.split(',') for line in inventory.strip().splitlines()]
        if len(rows)==8 and len({r[1].strip() for r in rows})==8 and min(int(r[2].split()[0]) for r in rows)>=35000:break
        atomic_json(progress,dict(status='WAITING_FOR_CAPACITY_WITHOUT_PREEMPTION',host=host,inventory=inventory,utc_unix=time.time()))
        time.sleep(30)
    args=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=8','-m','scripts.analysis.constrained_flow_noise.train',
          '--variant',a.variant,'--population',a.population,'--condition-cache',str(root/'condition_cache'),
          '--calibration',str(root/'calibration/calibration_lock.json'),'--registration',str(root/'precalibration_registration.json'),
          '--metric-cache','/mnt/project/flow-safe-improvement-20261013/metric_cache_v1','--metric-cache',str(root/'metric_cache_v1'),
          '--metric-manifest',str(root/'full_TRAIN_assets.json'),'--budget-registration',str(root/'formal_budget.json'),
          '--kernel-qualification',str(root/'validation/gpu_kernel_v2.json'),'--save-every',str(budget['save_every_optimizer_updates']),
          '--workers','2','--output',str(out)]
    if a.resume:args.extend(['--resume',a.resume])
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
                                    CUBLAS_WORKSPACE_CONFIG=':4096:8',NO_ALBUMENTATIONS_UPDATE='1',PYTHONUNBUFFERED='1')
    log=out.with_suffix('.train.log')
    with log.open('a' if a.resume else 'x') as handle:
        process=subprocess.Popen(args,env=env,stdout=handle,stderr=subprocess.STDOUT)
        receipt=dict(status='RUNNING',host=host,variant=a.variant,pid=process.pid,args=args,inventory=inventory,
                     started_unix=time.time(),source_lock_sha256=file_sha(a.source_lock),resume=a.resume,log=str(log))
        atomic_json(progress,receipt)
        signalled=False
        while process.poll() is None:
            try:require_dispatch()
            except PermissionError:
                if not signalled:
                    for child in psutil.Process(process.pid).children(recursive=True):
                        if 'scripts.analysis.constrained_flow_noise.train' in child.cmdline():child.send_signal(signal.SIGUSR1)
                    signalled=True;receipt['cooperative_withdrawal_stop_requested']=True
            time.sleep(30)
        receipt.update(status='TRAIN_PROCESS_EXITED',returncode=process.returncode,finished_unix=time.time())
        if (out/'result_rank0.json').exists():receipt['result']=json.loads((out/'result_rank0.json').read_text())
        atomic_json(progress,receipt)
    if process.returncode:raise SystemExit(process.returncode)


if __name__=='__main__':main()
