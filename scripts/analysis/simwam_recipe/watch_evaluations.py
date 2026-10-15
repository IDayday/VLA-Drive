"""Finite milestone export/evaluation queue for the two registered formal runs.

Never restarts training, extends its budget, or opens Navtest. Each evaluation
host runs at most one registered stage at a time, using both ODE readouts.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import traceback
from starVLA.rl.simwam_recipe.contracts import ROOT,require_dispatch
from starVLA.rl.flow_grpo.transactions import atomic_json


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--snapshot',required=True);a=p.parse_args();require_dispatch()
    root=Path(ROOT);queue=root/'milestone_queue';queue.mkdir(exist_ok=True)
    lock=(queue/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    routes={'R_SOURCE':'training-vlawm-tsinghua-zt','R_CANONICAL':'training-vlawm-tsinghua-zt2'}
    stages=[5000,10000,15000,20000];snapshot=Path(a.snapshot)
    while True:
        require_dispatch();finished=[]
        try:
            for variant,host in routes.items():
                for step in stages:
                    label=f'{variant}_update{step:06}';done=queue/(label+'.complete.json')
                    if done.exists():finished.append(label);continue
                    ck=root/'runs'/(variant+'_seed42_v1')/'checkpoints'/f'update{step:06}_inner0'
                    if not (ck/'COMPLETE').exists():break
                    export=root/'exports'/label
                    if not (export/'COMPLETE').exists():
                        policy=json.loads(Path('/mnt/project/server_dispatch_policy.json').read_text())
                        if host not in policy['task_authorizations']['flow_safe_improvement_20261013']['allowed_hosts'] or host not in policy['allowed_hosts']:raise PermissionError(host)
                        command=[sys.executable,'-m','scripts.analysis.simwam_recipe.export_policy','--checkpoint',str(ck),'--output',str(export)]
                        env='CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 NO_ALBUMENTATIONS_UPDATE=1 '
                        with (queue/(label+'.export.log')).open('ab',buffering=0) as log:
                            subprocess.run(['ssh','-o','BatchMode=yes',host,'cd '+shlex.quote(str(snapshot))+' && '+env+shlex.join(command)],stdout=log,stderr=subprocess.STDOUT,check=True)
                    meta=json.loads((export/'COMPLETE').read_text())
                    dispatched=queue/(label+'.dispatched.json')
                    if not dispatched.exists():
                        subprocess.run([sys.executable,'-m','scripts.analysis.simwam_recipe.dispatch_evaluation','--host',host,
                                        '--repo',str(snapshot),'--export',str(export),'--export-sha256',meta['weight_sha256'],'--label',label],check=True)
                        atomic_json(dispatched,dict(host=host,export=meta,time=time.time()))
                    complete=True;results={}
                    for cohort in ('DEV','TRAIN_DIAGNOSTIC'):
                        evaluation=root/'evaluations'/(label+'_'+cohort)
                        rank_files=[evaluation/f'rank{rank}.json' for rank in range(8)]
                        if not all(f.exists() and json.loads(f.read_text())['status']=='ASSIGNED_COMPLETE' for f in rank_files):
                            complete=False;continue
                        summary=evaluation/'summary.json'
                        if not summary.exists():
                            subprocess.run([sys.executable,'-m','scripts.analysis.simwam_recipe.analyze_evaluation',
                                            '--evaluation',str(evaluation),'--reference',str(root/'evaluations'/('SFT_update000000_'+cohort)),
                                            '--output',str(summary)],check=True)
                        results[cohort]=str(summary)
                    if complete:
                        atomic_json(done,dict(status='COMPLETE',label=label,results=results,time=time.time()));finished.append(label)
                    else:break
            atomic_json(queue/'progress.json',dict(status='COMPLETE' if len(finished)==8 else 'WAITING_FOR_REGISTERED_STAGES',
                        finished=finished,pid=os.getpid(),time=time.time(),Navtest_opened=False))
            if len(finished)==8:return
        except Exception:
            atomic_json(queue/'failure.json',dict(error=traceback.format_exc(),time=time.time(),pid=os.getpid()))
            raise
        time.sleep(20)


if __name__=='__main__':main()
