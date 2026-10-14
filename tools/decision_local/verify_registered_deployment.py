"""Finite formal online checks on currently free task-authorized devices.

Runs only the existing real-image verification/deployment entry points. It never
trains, resumes, selects a checkpoint or opens a Navtest evaluation. All slots
are reserved in this worker before CUDA initialization becomes visible.
"""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.decision_local.common import atomic_json,check_host,digest,read,source_sha
from tools.decision_local.run_registered_heads import free_gpus


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--registry',type=Path,required=True);p.add_argument('--gpus',nargs='+',type=int,required=True)
    p.add_argument('--max-seconds',type=float,required=True);a=p.parse_args()
    resource=check_host();registration=read(a.registry);sha=digest(a.registry)
    if registration['scope'] not in ('formal_current_only_deployment','execution_smoke') or a.max_seconds<=0:raise ValueError('Finite deployment list required')
    state=Path(registration['runtime']);state.mkdir(parents=True,exist_ok=True)
    for name in ('claims','results'):(state/name).mkdir(exist_ok=True)
    started=time.monotonic();active={};previous_free=set()
    atomic_json(state/'WORKER.json',{'pid':os.getpid(),'resources':resource,'source_sha':source_sha(ROOT),
        'registry_sha256':sha,'started_unix':time.time(),'max_seconds':a.max_seconds})
    while time.monotonic()-started<a.max_seconds or active:
        check_host()
        if digest(a.registry)!=sha:raise ValueError('Registered checks changed')
        for gpu,(child,job,stream,began) in list(active.items()):
            code=child.poll();timeout=False
            if code is None and time.monotonic()-began>job['max_seconds']+30:
                os.killpg(child.pid,signal.SIGTERM);code=child.wait(timeout=30);timeout=True
            if code is None:continue
            stream.close();result=Path(job['result']);success=code==0 and result.exists()
            if success:
                payload=read(result)
                success=payload.get('passed',False) if job['entry']=='verify_current_only.py' else payload.get('python_training_file_access_guard',False)
            atomic_json(state/'results'/(job['id']+'.json'),{'complete':success,'returncode':code,'timeout':timeout,
                'seconds':time.monotonic()-began,'gpu':gpu,'finished_unix':time.time()})
            del active[gpu]
        now_free=free_gpus(a.gpus)-set(active);admitted=now_free&previous_free;previous_free=now_free
        for job in registration['jobs']:
            claim=state/'claims'/job['id']
            if claim.exists() or not all(Path(x).exists() for x in job['requires']):continue
            if not admitted or a.max_seconds-(time.monotonic()-started)<job['max_seconds']+60:break
            if job['entry'] not in ('verify_current_only.py','predict_current.py') or job['command'][2]!=str(ROOT/'tools/decision_local'/job['entry']):
                raise ValueError('Only the actual current-image entry points are eligible')
            if '--split' in job['command'] and job['command'][job['command'].index('--split')+1]!='dev':raise ValueError('Deployment check uses development only')
            if registration['scope']=='formal_current_only_deployment':
                checkpoint=Path(job['command'][job['command'].index('--refiner')+1])
                complete=read(checkpoint.parent/'COMPLETE.json')
                if checkpoint.name!='epoch_24.pt' or complete['epoch']!=24 or not complete['full_training_population']:
                    raise ValueError('Formal verification requires the completed full-data common endpoint')
            try:claim.mkdir()
            except FileExistsError:continue
            gpu=min(admitted);admitted.remove(gpu)
            stream=(state/(job['id']+'.log')).open('x')
            child=subprocess.Popen(job['command'],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),
                OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1'),stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            atomic_json(claim/'LAUNCH.json',{'pid':child.pid,'gpu':gpu,'command':job['command'],'started_unix':time.time()})
            active[gpu]=(child,job,stream,time.monotonic())
        if not active and all((state/'claims'/j['id']).exists() for j in registration['jobs']):break
        if not active and time.monotonic()-started>=a.max_seconds:break
        time.sleep(10)
    atomic_json(state/'FINISHED.json',{'seconds':time.monotonic()-started,
        'pending':[j['id'] for j in registration['jobs'] if not (state/'claims'/j['id']).exists()],
        'no_training_or_Navtest_launched':True,'failures_retained_no_automatic_retry':True})


if __name__=='__main__':main()
