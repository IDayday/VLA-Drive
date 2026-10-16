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
from starVLA.rl.simwam_recipe.contracts import ROOT
from starVLA.rl.flow_grpo.transactions import atomic_json
from scripts.analysis.simwam_recipe.controller_runtime import require_controller_host,require_worker_host,probe_routes,preflight_analysis_compatibility


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--snapshot',required=True)
    p.add_argument('--administrative-local',action='store_true');p.add_argument('--preflight-only',action='store_true')
    p.add_argument('--followup-x1',action='store_true')
    p.add_argument('--source-compatibility')
    p.add_argument('--x1-dev-host',default='training-vlawm-tsinghua-zt3')
    p.add_argument('--x1-train-host',default='training-vlawm-tsinghua-zt4')
    a=p.parse_args();require_controller_host(a.administrative_local)
    root=Path('/mnt/project/simwam-x-anchor-20261015') if a.followup_x1 else Path(ROOT)
    baseline_root=Path(ROOT)
    if a.followup_x1:
        authorization=json.loads((root/'AUTHORIZATION.json').read_text())
        if authorization['status']!='AUTHORIZED_BY_USER_GO_ON' or authorization['proposed_name']!='R_CANONICAL_X1':
            raise ValueError('X1 registration required')
    queue=root/'milestone_queue';queue.mkdir(exist_ok=True)
    lock=(queue/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    routes={'R_SOURCE':'training-vlawm-tsinghua-zt','R_CANONICAL':'training-vlawm-tsinghua-zt2'}
    if a.followup_x1:routes={'R_CANONICAL_X1':a.x1_dev_host}
    cohort_hosts={'DEV':a.x1_dev_host,'TRAIN_DIAGNOSTIC':a.x1_train_host} if a.followup_x1 else {}
    expected=len(routes)*4
    stages=[5000,10000,15000,20000];snapshot=Path(a.snapshot)
    transport=dict(time=time.time(),pid=os.getpid(),administrative_local=a.administrative_local,
                   routes=probe_routes(sorted(set(routes.values())|set(cohort_hosts.values())),sys.executable),
                   analysis_compatibility=preflight_analysis_compatibility(snapshot,a.source_compatibility))
    atomic_json(queue/'transport_preflight.json',transport)
    if a.preflight_only:print(json.dumps(transport));return
    while True:
        require_controller_host(a.administrative_local);finished=[]
        try:
            for variant,host in routes.items():
                require_worker_host(host)
                for step in stages:
                    label=f'{variant}_update{step:06}';done=queue/(label+'.complete.json')
                    if done.exists():finished.append(label);continue
                    ck=root/'runs'/(variant+'_seed42_v1')/'checkpoints'/f'update{step:06}_inner0'
                    if not (ck/'COMPLETE').exists():break
                    export=root/'exports'/label
                    if not (export/'COMPLETE').exists():
                        require_worker_host(host)
                        command=[sys.executable,'-m','scripts.analysis.simwam_recipe.export_policy','--checkpoint',str(ck),'--output',str(export)]
                        env='CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 NO_ALBUMENTATIONS_UPDATE=1 '
                        with (queue/(label+'.export.log')).open('ab',buffering=0) as log:
                            subprocess.run(['ssh','-o','BatchMode=yes',host,'cd '+shlex.quote(str(snapshot))+' && '+env+shlex.join(command)],stdout=log,stderr=subprocess.STDOUT,check=True)
                    meta=json.loads((export/'COMPLETE').read_text())
                    dispatched=queue/(label+'.dispatched.json')
                    if not dispatched.exists():
                        assignments=[(cohort_hosts[c],[c]) for c in ('DEV','TRAIN_DIAGNOSTIC')] if cohort_hosts else [(host,['DEV','TRAIN_DIAGNOSTIC'])]
                        for target,cohorts in assignments:
                            require_worker_host(target)
                            subprocess.run([sys.executable,'-m','scripts.analysis.simwam_recipe.dispatch_evaluation','--host',target,
                                            '--repo',str(snapshot),'--root',str(root),'--export',str(export),
                                            '--export-sha256',meta['weight_sha256'],'--label',label,'--cohorts',*cohorts],check=True)
                        atomic_json(dispatched,dict(host=host,export=meta,time=time.time()))
                    complete=True;results={}
                    for cohort in ('DEV','TRAIN_DIAGNOSTIC'):
                        evaluation=root/'evaluations'/(label+'_'+cohort)
                        rank_files=[evaluation/f'rank{rank}.json' for rank in range(8)]
                        if not all(f.exists() and json.loads(f.read_text())['status']=='ASSIGNED_COMPLETE' for f in rank_files):
                            complete=False;continue
                        summary=evaluation/'summary.json'
                        if not summary.exists():
                            command=[sys.executable,'-m','scripts.analysis.simwam_recipe.analyze_evaluation',
                                            '--evaluation',str(evaluation),'--reference',str(baseline_root/'evaluations'/('SFT_update000000_'+cohort)),
                                            '--output',str(summary)]
                            if a.source_compatibility:command+=['--source-compatibility',a.source_compatibility]
                            subprocess.run(command,check=True)
                        results[cohort]=str(summary)
                    if complete:
                        atomic_json(done,dict(status='COMPLETE',label=label,results=results,time=time.time()));finished.append(label)
                    else:break
            atomic_json(queue/'progress.json',dict(status='COMPLETE' if len(finished)==expected else 'WAITING_FOR_REGISTERED_STAGES',
                        finished=finished,pid=os.getpid(),time=time.time(),Navtest_opened=False))
            if len(finished)==expected:return
        except Exception:
            atomic_json(queue/'failure.json',dict(error=traceback.format_exc(),time=time.time(),pid=os.getpid()))
            raise
        time.sleep(20)


if __name__=='__main__':main()
