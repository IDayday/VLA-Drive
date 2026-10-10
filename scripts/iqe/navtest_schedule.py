"""Finite, checksum-locked Base checkpoint monitoring; never launches training."""
import argparse
import os
from pathlib import Path
import shlex
import subprocess
import time
from iqe.io import read_json,atomic_json,digest,lock
from iqe.navtest import code_hash
from iqe.resources import qualify
from iqe.training.resume import inspect_checkpoint


def idle_host(hosts):
    policy=read_json('/mnt/project/server_dispatch_policy.json')
    permitted=policy['task_authorizations']['iqe_v1']['allowed_hosts']
    for host in hosts:
        if host not in permitted:continue
        r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8',host,
            'nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits'],capture_output=True,text=True,timeout=15)
        if r.returncode:continue
        rows=[tuple(map(int,line.split(','))) for line in r.stdout.splitlines()]
        if len(rows)==8 and all(m<=1024 and u<=5 for m,u in rows):return host
    return None


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--plan',required=True);a=p.parse_args()
    plan=read_json(a.plan);root=Path(plan['root']);source=Path(plan['source']);start=time.time()
    qualify('cpu')
    with lock(root/'schedule.lock'):
        for step in plan['steps']:
            out=root/f'step_{step:06d}';cp=Path(plan['checkpoint_root'])/f'step_{step:06d}.pt'
            while True:
                if (root/'STOP_SCHEDULING').exists():
                    atomic_json(root/'SCHEDULE_STATUS.json',{'status':'PAUSED_BY_OPERATOR','checkpoint_step':step,'pid':os.getpid()})
                    return
                if code_hash()!=plan['code_hash']:raise RuntimeError('evaluation code changed; schedule stopped')
                if time.time()-start>plan['watch_seconds']:raise TimeoutError('finite checkpoint watch budget exhausted')
                if (out/'NAVTEST_RESULT.json').exists():
                    result=read_json(out/'NAVTEST_RESULT.json')
                    if result['status']!='COMPLETE':raise RuntimeError('invalid benchmark; manual diagnosis required')
                    break
                ready=cp.exists() and Path(str(cp)+'.COMPLETE.json').exists()
                host=idle_host(plan['hosts']) if ready and not (out/'EXPORT_COMPLETE.json').exists() else None
                if ready and (host or (out/'EXPORT_COMPLETE.json').exists()):break
                atomic_json(root/'SCHEDULE_STATUS.json',{'status':'WAITING_FOR_CHECKPOINT' if not ready else 'WAITING_FOR_IDLE_8GPU_HOST',
                            'checkpoint_step':step,'pid':os.getpid(),'updated_unix':time.time(),'plan_hash':digest(plan)})
                time.sleep(30)
            if (out/'NAVTEST_RESULT.json').exists():continue
            info=inspect_checkpoint(cp)
            if info['run_identity']!=plan['run_identity']:raise RuntimeError('checkpoint came from another training run')
            common=['--config',plan['config'],'--dataset',plan['dataset'],'--output',str(out)]
            subprocess.run([plan['python'],'-m','iqe.navtest','lock-checkpoint',*common,'--checkpoint',str(cp)],cwd=source,check=True)
            jobs=[]
            try:
                if not (out/'EXPORT_COMPLETE.json').exists():
                    # The output lock prevents competing coordinators. A previous
                    # recorded launch is conservatively inspected before retry.
                    prior=out/'EXPORT_LAUNCH.json'
                    if prior.exists():
                        old=read_json(prior);oldhost=old['resources']['canonical_host'];pid=old['coordinator_pid']
                        probe=subprocess.run(['ssh',oldhost,'cat /proc/'+str(pid)+'/cmdline'],capture_output=True)
                        if probe.returncode==0 and str(out).encode() in probe.stdout:
                            raise RuntimeError('existing export is live; refusing duplicate dispatch')
                    command=[plan['python'],'scripts/iqe/navtest_export.py',*common,'--gpus','0,1,2,3,4,5,6,7','--timeout',str(plan['job_seconds'])]
                    remote='cd '+shlex.quote(str(source))+' && PYTHONPATH=. '+shlex.join(command)
                    log=(out/'coordinator.log').open('a');jobs.append((subprocess.Popen(['ssh',host,remote],stdout=log,stderr=subprocess.STDOUT),log))
                env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONPATH=str(source))
                log=(out/'scoring.log').open('a')
                jobs.append((subprocess.Popen([plan['metric_python'],'-m','iqe.navtest','score',*common,'--workers',str(plan['cpu_workers']),'--timeout',str(plan['job_seconds'])],cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT),log))
                atomic_json(root/'SCHEDULE_STATUS.json',{'status':'EVALUATING','checkpoint_step':step,'host':host,'pid':os.getpid(),
                            'children':[j.pid for j,l in jobs],'plan_hash':digest(plan),'updated_unix':time.time()})
                for child,log in jobs:
                    if child.wait()!=0:raise RuntimeError('checkpoint evaluation failed; evidence preserved')
            finally:
                for child,log in jobs:log.close()
        atomic_json(root/'SCHEDULE_STATUS.json',{'status':'COMPLETE','steps':plan['steps'],'plan_hash':digest(plan),'pid':os.getpid()})


if __name__=='__main__':main()
