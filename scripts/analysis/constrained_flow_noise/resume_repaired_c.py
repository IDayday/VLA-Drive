"""Launch the verified C continuation from its recovered native 1850 boundary."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from starVLA.rl.constrained_flow_noise.contracts import require_dispatch, file_sha
from starVLA.rl.flow_grpo.transactions import atomic_json
from .train_stable_probability import source_identity


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--root',required=True);p.add_argument('--checkpoint',required=True)
    p.add_argument('--acceptance',required=True);p.add_argument('--output',required=True)
    p.add_argument('--receipt',required=True)
    a=p.parse_args();host=require_dispatch();root,cp,out,receipt=map(Path,(a.root,a.checkpoint,a.output,a.receipt))
    if out.exists() or receipt.exists():raise FileExistsError('Never overwrite a previous continuation')
    guard=out.with_suffix('.launcher.lock').open('a');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for source,digest in json.loads((root/'formal_sources.json').read_text())['sources'].items():
        if file_sha(source)!=digest:raise ValueError('Frozen scientific source changed: '+source)
    proof=json.loads(Path(a.acceptance).read_text());execution=source_identity()
    if proof['status']!='PASS' or proof['execution_sha256']!=execution['sha256']:
        raise ValueError('Missing exact numerical-repair qualification')
    if not (cp/'COMPLETE').exists():raise ValueError('Incomplete recovery')
    boundary=json.loads((cp/'boundary.json').read_text())
    if (boundary['phase'],boundary['update'],boundary['raw_exposure'])!=('ready',1850,23184):
        raise ValueError('This launcher is for the exact recovered C boundary')
    recovery=cp.parent.parent
    for rank in range(8):
        record=json.loads((recovery/f'recovery_rank{rank}.json').read_text())
        if record['status']!='RECOVERED_EXACT_OBSERVED_SEQUENCE' or record['last_verified_update']!=1850 or Path(record['result_checkpoint'])!=cp:
            raise ValueError('Incomplete eight-rank historical recovery')
    inventory=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free,utilization.gpu','--format=csv,noheader'],text=True)
    rows=[line.split(',') for line in inventory.strip().splitlines()]
    if len(rows)!=8 or min(int(row[2].split()[0]) for row in rows)<35000:
        raise RuntimeError('Insufficient capacity; no process was preempted')
    command=list(json.loads((root/'formal_C_SDE_seed42_v1.launcher.json').read_text())['args'])
    i=command.index('scripts.analysis.constrained_flow_noise.train')
    command[i:i+1]=['scripts.analysis.constrained_flow_noise.train_stable_probability','--layout-acceptance',str(Path(a.acceptance).resolve()),'--']
    command[0]=sys.executable;command[command.index('--output')+1]=str(out)
    command+=['--resume',str(cp)]
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',CUBLAS_WORKSPACE_CONFIG=':4096:8',
             OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NO_ALBUMENTATIONS_UPDATE='1',PYTHONUNBUFFERED='1',
             CFN_RUN_ID=str(out))
    receipt.parent.mkdir(parents=True,exist_ok=True)
    log=receipt.with_suffix('.train.log')
    with log.open('x') as stream:
        process=subprocess.Popen(command,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    record=dict(status='RUNNING',host=host,pid=process.pid,args=command,checkpoint=str(cp),output=str(out),
                started_unix=time.time(),execution_sha256=execution['sha256'],acceptance=a.acceptance,
                log=str(log),inventory=inventory,scientific_budget_unchanged=True,seed42_not_restarted=True)
    atomic_json(receipt,record)
    atomic_json(root/'formal_C_SDE_active_continuation.json',dict(output=str(out),receipt=str(receipt),parent=str(root/'formal_C_SDE_seed42_v1'),checkpoint=str(cp)))
    code=process.wait();record.update(status='TRAIN_PROCESS_EXITED',returncode=code,finished_unix=time.time())
    atomic_json(receipt,record)
    raise SystemExit(code)


if __name__=='__main__':main()
