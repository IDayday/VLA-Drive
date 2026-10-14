"""Finite registered relation checks after completed predictor endpoints.

Reuses the existing evaluator and resource admission. No training, checkpoint
selection, Navtest, retries or additional scientific arms are introduced.
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.decision_local.common import atomic_json,check_host,digest,read,source_sha
from tools.decision_local.evaluate_milestones import eligible_gpus


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--registry',type=Path,required=True)
    p.add_argument('--gpus',type=int,nargs='+',required=True)
    p.add_argument('--max-seconds',type=float,required=True)
    a=p.parse_args();resource=check_host();registration=read(a.registry);sha=digest(a.registry)
    if registration['scope']!='fixed_predictor_endpoint_checks' or a.max_seconds<=0:
        raise ValueError('Finite preregistered endpoint checks required')
    state=Path(registration['runtime']);state.mkdir(parents=True,exist_ok=True)
    (state/'claims').mkdir(exist_ok=True);(state/'results').mkdir(exist_ok=True)
    started=time.monotonic()
    atomic_json(state/'WORKER.json',{'pid':os.getpid(),'started_unix':time.time(),'registry_sha256':sha,
        'source_sha':source_sha(ROOT),'resources':resource,'max_seconds':a.max_seconds})
    while time.monotonic()-started<a.max_seconds:
        check_host()
        if digest(a.registry)!=sha:raise ValueError('Evaluation registry changed')
        if all((state/'claims'/j['id']).exists() for j in registration['jobs']):break
        for job in registration['jobs']:
            claim=state/'claims'/job['id']
            if claim.exists() or not all(Path(x).exists() for x in job['requires']):continue
            if a.max_seconds-(time.monotonic()-started)<job['max_seconds']+60:continue
            devices=eligible_gpus(a.gpus)
            if not devices:break
            command=job['command']
            if command[2]!=str(ROOT/'tools/decision_local/evaluate_relations.py'):
                raise ValueError('Only the existing fixed-query evaluator is supported')
            if float(command[command.index('--max-seconds')+1])!=job['max_seconds']:
                raise ValueError('Child time bound differs from registration')
            try:claim.mkdir()
            except FileExistsError:continue
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(devices[0]),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
            atomic_json(claim/'COMMAND.json',{'command':command,'gpu':devices[0],
                'source_sha':source_sha(ROOT),'started_unix':time.time()})
            success=False;code=None;timed_out=False
            with (state/(job['id']+'.log')).open('x') as stream:
                try:
                    code=subprocess.run(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,
                        timeout=job['max_seconds']+30).returncode
                    success=code==0 and (Path(job['output'])/'COMPLETE.json').exists()
                except subprocess.TimeoutExpired:
                    # Only our stateless evaluator is killed by subprocess.run;
                    # it has no training state or external worker children.
                    timed_out=True
            atomic_json(state/'results'/(job['id']+'.json'),{'complete':success,'returncode':code,
                'timed_out':timed_out,'finished_unix':time.time()})
        time.sleep(10.)
    atomic_json(state/'FINISHED.json',{'seconds':time.monotonic()-started,
        'pending':[j['id'] for j in registration['jobs'] if not (state/'claims'/j['id']).exists()],
        'failed_results_preserved':True,'no_automatic_retry':True})


if __name__=='__main__':main()
