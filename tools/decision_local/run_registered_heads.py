"""Finite single-GPU job list; reuse the head trainer and its native checkpoints.

This runner never stops GPU owners, resumes failed tasks, selects checkpoints,
changes a recipe, or starts Navtest. Independent workers claim ready jobs through
atomic directories on the shared project filesystem. Dependencies are completed
artifacts, not elapsed waits. A formal acceptance receipt is required separately.
"""
import argparse
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.decision_local.common import atomic_json,check_host,digest,read,source_sha


def free_gpus(selected,shared=False):
    output=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu,memory.total',
        '--format=csv,noheader,nounits'],text=True)
    return {int(f[0]) for line in output.splitlines() for f in [line.split(',')]
        if int(f[0]) in selected and ((int(f[1])<512 and int(f[2])<10) or
            (shared and int(f[3])-int(f[1])>=8192 and int(f[2])<95))}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--registry',type=Path,required=True)
    p.add_argument('--gpus',type=int,nargs='+',required=True)
    p.add_argument('--max-seconds',type=float,required=True)
    a=p.parse_args();resource=check_host();registered=read(a.registry);registry_hash=digest(a.registry)
    host=socket.gethostname().removesuffix('-worker-0');started=time.monotonic()
    if host not in registered['allowed_hosts'] or not set(a.gpus)<=set(registered['allowed_hosts'][host]):
        raise ValueError('Unregistered worker resource')
    repair=registered.get('schema')=='minimal_repair_finite_heads_v1'
    cap=registered['GPU_hour_cap']
    if (cap!=64 if repair else cap!=100) or sum(registered['budget_allocations'].values())>cap:
        raise ValueError('Finite registered total budget required')
    state=Path(registered['runtime']);state.mkdir(parents=True,exist_ok=True)
    (state/'claims').mkdir(exist_ok=True);(state/'results').mkdir(exist_ok=True)
    worker=state/(host+'_'+str(os.getpid())+'.json')
    atomic_json(worker,{'registry_sha256':registry_hash,'host':host,'pid':os.getpid(),'gpus':a.gpus,
        'source_sha':source_sha(ROOT),'started_unix':time.time(),'max_seconds':a.max_seconds,'resource_check':resource})
    active={};previous_free=set();files={}
    while time.monotonic()-started<a.max_seconds or active:
        check_host()
        if digest(a.registry)!=registry_hash:raise ValueError('Immutable task registry changed')
        for gpu,(child,job,claim) in list(active.items()):
            code=child.poll()
            if code is None:continue
            files.pop(job['id']).close()
            complete=Path(job['output'])/'COMPLETE.json'
            result={'job':job['id'],'returncode':code,'complete':complete.exists(),
                'host':host,'gpu':gpu,'finished_unix':time.time(),
                'status':'complete' if code==0 and complete.exists() else 'paused_or_failed_no_automatic_retry'}
            atomic_json(state/'results'/(job['id']+'.json'),result);del active[gpu]
        free=free_gpus(a.gpus,shared=repair and registered.get('allow_shared_gpus',False))-set(active);admitted=free&previous_free;previous_free=free
        remaining=a.max_seconds-(time.monotonic()-started)
        for job in registered['jobs']:
            if not admitted or remaining<job['max_seconds']+60:break
            if (state/'claims'/job['id']).exists() or (Path(job['output'])/'identity.json').exists():continue
            if not all(Path(x).exists() for x in job['requires']):continue
            if job['scope'] in ('formal','repair_formal'):
                acceptance=read(Path(registered['formal_acceptance']))
                if not acceptance.get('accepted') or acceptance['training_source_sha']!=job['training_source_sha']:
                    raise ValueError('Formal common recipe not accepted for this source')
            source=Path(job['source'])
            if source_sha(source)!=job['training_source_sha'] or subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=source,text=True).strip():
                raise ValueError('Frozen training source changed')
            claim=state/'claims'/job['id']
            try:claim.mkdir()
            except FileExistsError:continue
            gpu=min(admitted);admitted.remove(gpu)
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false')
            log=state/(job['id']+'.log');stream=log.open('x');files[job['id']]=stream
            command=job['command']
            # Only the already-tested small-head trainer is eligible here.
            if command[2]!=str(source/'tools/decision_local/train_small_model.py') or '--resume' in command:
                raise ValueError('Only registered fresh small-head training commands are supported')
            if float(command[command.index('--max-seconds')+1])!=job['max_seconds']:
                raise ValueError('Child must implement its own bounded safe checkpoint')
            child=subprocess.Popen(command,cwd=source,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            atomic_json(claim/'LAUNCH.json',{'job':job['id'],'command':command,'pid':child.pid,'host':host,'gpu':gpu,
                'process_start_ticks':Path('/proc/'+str(child.pid)+'/stat').read_text().split()[21],
                'started_unix':time.time(),'registry_sha256':registry_hash,'log':str(log)})
            active[gpu]=(child,job,claim);print(job['id'],host,gpu,child.pid,flush=True)
        if not active and all((state/'claims'/j['id']).exists() or (Path(j['output'])/'COMPLETE.json').exists() for j in registered['jobs']):break
        if not active and remaining<=0:break
        time.sleep(10.)
    atomic_json(worker.with_suffix('.finished.json'),{'elapsed_seconds':time.monotonic()-started,'active_children':len(active),
        'pending_jobs':[j['id'] for j in registered['jobs'] if not (state/'claims'/j['id']).exists()]})


if __name__=='__main__':main()
