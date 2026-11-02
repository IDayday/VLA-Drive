"""Finite single-owner qualification queue; no model or scientific recipe edits.

Retries only actual pre-child resource refusals. Every attempt and failed
calculation remains on disk. GPU leases/UUIDs/ancestry and finite-host guards
are rechecked by the existing job worker before CUDA startup.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
from tools.d44_native_rl.launch import launch,DispatchPreflightFailed
from tools.d44_native_rl.campaign import wait_for_completion
from starVLA.rl.qwen_world_compare.contracts import allowed_hosts,fingerprint,file_sha
from starVLA.rl.flow_grpo.transactions import atomic_json


def read(p):return json.loads(Path(p).read_text())


def capacity(host,minimum,profiled=False):
    code='''import json,pathlib,subprocess
r=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True)
owners={}
for l in subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader,nounits'],text=True).splitlines():
 p,u=l.split(',');owners.setdefault(u.strip(),[]).append(int(p))
def frozen(pid):
 try:
  a=[x for x in(pathlib.Path('/proc')/str(pid)/'cmdline').read_bytes().decode().split('\\0')if x]
  return '-m'in a and a[a.index('-m')+1]in('tools.qwen_world_compare.evaluate','tools.qwen_world_safety_exploration.evaluate')
 except(OSError,IndexError):return False
print(json.dumps([dict(frozen_eval=all(frozen(p)for p in owners.get(l.split(',')[1].strip(),[])),index=int(l.split(',')[0]),uuid=l.split(',')[1].strip(),free=int(l.split(',')[2]),owners=owners.get(l.split(',')[1].strip(),[]),visible=all(pathlib.Path('/proc/'+str(p)+'/cmdline').exists()for p in owners.get(l.split(',')[1].strip(),[])))for l in r.splitlines()]))'''
    result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8',host,'/usr/bin/python3','-'],input=code,text=True,capture_output=True,timeout=20)
    if result.returncode:raise DispatchPreflightFailed(result.stderr)
    rows=json.loads(result.stdout)
    protected=host in ('training-vla-zt','training-vlawm-tsinghua-zt')
    # Unexplained memory with no NVML owner remains unavailable (e.g. WM2 GPU6).
    return [r['index']for r in rows if r['free']>=minimum and r['visible']and len(r['owners'])<(3 if profiled and r['frozen_eval']else 2)
        and (not protected or not r['owners'])and(r['owners']or r['free']>=79000)]


def qualification_done(task):
    """Exit0 alone is not mathematical PASS or a completed update budget."""
    command=task['command'];output=Path(command[command.index('--output')+1])
    if 'tools.d44_native_rl.train'not in command:
        proof=read(output)
        if proof.get('status')!='PASS':raise ValueError('Qualification proof did not PASS: '+str(output))
        return True
    world=int(next(a.split('=')[1]for a in command if a.startswith('--nproc_per_node=')))
    target=int(command[command.index('--stop-after-update')+1]);rows=[read(output/f'result_rank{r}.json')for r in range(world)]
    if any(r.get('status')!='QUALIFICATION_STOPPED'or not r.get('frozen_before')or r.get('frozen_before')!=r.get('frozen_after')for r in rows):
        raise ValueError('Native qualification/frozen parameter evidence failed')
    if len({r['updates']for r in rows})!=1:raise ValueError('Qualification rank updates disagree')
    if rows[0]['updates']>=target:return True
    latest=read(output/'latest.json');checkpoint=Path(latest['checkpoint']);seal=read(checkpoint/'COMPLETE')
    if seal['updates']!=rows[0]['updates']or seal['phase']not in('READY','ROLLOUT_COMMITTED','INNER_COMMITTED','COMPLETE')or seal['files_sha256']!=file_sha(checkpoint/'files.json'):
        raise ValueError('Short qualification lacks a legal native recovery point')
    command=list(command)
    if '--resume'in command:
        i=command.index('--resume');del command[i:i+2]
    task['command']=command+['--resume',str(checkpoint)];task.setdefault('native_resume_lineage',[]).append(dict(checkpoint=str(checkpoint),updates=seal['updates']))
    return False


def tick(plan,state):
    allowed=allowed_hosts(admit_long_job=True);pool={};reserved=set();worker_source=plan['worker_source']
    for task in state['tasks']:
        if task['status']in ('COMPLETE','FAILED_REQUIRES_REVIEW'):continue
        jobs=task['attempts']
        if jobs:
            path=Path(jobs[-1]['job'])/'RESULT.json'
            if not path.exists():task['status']='RUNNING_OR_ADMITTING';reserved.update((jobs[-1]['host'],g)for g in jobs[-1]['gpus']);continue
            result=read(path)
            if result.get('returncode')==0:
                try:done=qualification_done(task)
                except Exception as error:task.update(status='FAILED_REQUIRES_REVIEW',error=repr(error),result=str(path));continue
                if not done:
                    task['status']='PENDING_NATIVE_RESUME'
                elif task.get('postprocess_command'):
                    started=time.time();post=subprocess.run(task['postprocess_command'],cwd=plan['operator_source'],capture_output=True,text=True)
                    task.setdefault('postprocess_attempts',[]).append(dict(started=started,end=time.time(),returncode=post.returncode,stdout=post.stdout[-4000:],stderr=post.stderr[-4000:]))
                    if post.returncode:task.update(status='FAILED_REQUIRES_REVIEW',result=str(path));continue
                if done:task['status']='COMPLETE';task['result']=str(path);continue
            if result.get('returncode')not in(0,75):task.update(status='FAILED_REQUIRES_REVIEW',result=str(path));continue
            if result.get('returncode')==75 and time.time()<result['end_unix']+600:task['status']='RESOURCE_BACKOFF';continue
        for host in task['hosts']:
            if host not in allowed:continue
            key=(host,task['minimum_free_MiB'],task.get('profiled_native_qualification_third_slot',False))
            try:
                if key not in pool:pool[key]=capacity(*key)
                gpus=[g for g in pool[key]if(host,g)not in reserved]
            except (subprocess.SubprocessError,OSError,DispatchPreflightFailed,ValueError)as error:
                task['last_prechild_error']=repr(error);continue
            if len(gpus)<task['gpus']:continue
            gpus=gpus[:task['gpus']];job=Path(plan['root'])/'jobs/QUALIFICATION_QUEUE'/task['id']/f'attempt{len(jobs)+1:03}'
            try:
                receipt=launch(host,task['source'],job,task['command'],gpus,share_min_free_mib=None if host in ('training-vla-zt','training-vlawm-tsinghua-zt')else task['minimum_free_MiB'],worker_source=worker_source,profiled_native_qualification_third_slot=task.get('profiled_native_qualification_third_slot',False))
            except DispatchPreflightFailed as error:task['last_prechild_error']=repr(error);continue
            except Exception as error:
                task.update(status='DISPATCH_REQUIRES_RECONCILIATION',error=repr(error));raise
            jobs.append(receipt);task['status']='RUNNING_OR_ADMITTING';reserved.update((host,g)for g in gpus);break
        else:task['status']='WAITING_FOR_AUTHORIZED_CAPACITY'
    state['at']=time.time();state['complete']=all(t['status']=='COMPLETE'for t in state['tasks']);return state


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--plan',required=True);p.add_argument('--state',required=True);p.add_argument('--interval',type=int,default=600);a=p.parse_args()
    plan=read(a.plan);statepath=Path(a.state)
    with statepath.with_suffix('.owner.lock').open('a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state=read(statepath)if statepath.exists()else dict(schema='d44_qualification_queue_v1',plan_sha256=fingerprint(plan),tasks=[dict(t,status='PENDING',attempts=[])for t in plan['tasks']])
        if state['plan_sha256']!=fingerprint(plan):raise ValueError('Qualification plan changed')
        state['owner_pid']=os.getpid()
        while True:
            state=tick(plan,state);atomic_json(statepath,state)
            if state['complete']:break
            # Metadata heartbeat, not high-frequency training-log polling.
            watched=dict(runs={t['id']:dict(qualification=t['source'],output=plan['root'],jobs=t['attempts'])for t in state['tasks']},evaluations={})
            wait_for_completion(watched,max(60,a.interval))


if __name__=='__main__':main()
