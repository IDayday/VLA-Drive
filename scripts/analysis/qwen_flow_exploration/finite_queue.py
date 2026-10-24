"""Finite operator-only dependency queue for the registered three-route screen.

Never retries failed workers, reads rewards, selects checkpoints, changes budgets,
or terminates a GPU process. The durable existing worker owns each model process.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time
import traceback


def seal(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def atomic(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.writing-'+str(os.getpid()))
    with temp.open('x') as stream:json.dump(value,stream,indent=2)
    temp.replace(path)


def admitted(plan):
    value=read('/mnt/project/server_dispatch_policy.json');host=plan['host']
    if (host!=socket.gethostname().removesuffix('-worker-0') or host not in value['allowed_hosts'] or
        host not in value['task_authorizations']['qwen_flow_three_routes_20261024']['allowed_hosts'] or
        value.get('global_dispatch_prohibitions',{}).get(host,{}).get('new_task_dispatch_allowed') is False):
        raise PermissionError('Queue host is no longer authorized')


def dependency_state(job):
    for name in job.get('successful_workers',[]):
        path=Path(name)
        if not path.exists():return 'WAITING'
        if read(path).get('status')!='PROCESS_COMPLETE':return 'HELD_FAILED_DEPENDENCY'
    for name in job.get('required_complete',[]):
        path=Path(name)
        if not path.exists():return 'WAITING'
        if read(path).get('status')!='COMPLETE':return 'HELD_INVALID_DEPENDENCY'
    return 'READY'


def gpu_rows():
    output=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True)
    return {int(i):dict(uuid=u.strip(),free_MiB=int(m)) for i,u,m in (x.split(',') for x in output.splitlines())}


def busy_study_gpus(root, host=None):
    busy=set()
    for p in (root/'dispatch').glob('*/spec.json'):
        if not (p.parent/'result.json').exists():
            # A missing result remains held even if the process was lost.
            # Never silently recover or schedule over an unreviewed invocation.
            spec=read(p)
            if host is None or spec['host']==host:busy.update(spec.get('gpus',[]))
    return busy


def submit(plan, job, gpu, physical):
    root=Path(plan['root']);directory=root/'dispatch'/job['name']
    directory.mkdir(exist_ok=False)
    spec=dict(source=job['source'],source_sha=job['source_sha'],host=plan['host'],
              gpus=[gpu],gpu_uuids=[physical['uuid']],min_free_MiB=job['min_free_MiB'],
              module=job['module'],command=[plan['model_python'],'-u','-m',job['module'],*job['arguments']],
              receipt_dir=str(directory),queue_plan_sha256=plan['sha256'])
    atomic(directory/'spec.json',spec)
    with (directory/'operator.log').open('x') as stream:
        child=subprocess.Popen(['/usr/bin/python3',plan['worker'],'--spec',str(directory/'spec.json')],
            cwd='/',stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    atomic(directory/'operator.json',dict(operator_pid=child.pid,
        start_ticks=int((Path('/proc')/str(child.pid)/'stat').read_text().split()[21]),at=time.time()))


def tick(plan):
    admitted(plan);root=Path(plan['root']);busy=busy_study_gpus(root,plan['host']);physical=gpu_rows();states={}
    for job in plan['jobs']:
        receipt=root/'dispatch'/job['name'];result=receipt/'result.json'
        if result.exists():
            states[job['name']]='COMPLETE' if read(result)['status']=='PROCESS_COMPLETE' else 'FAILED_NO_AUTOMATIC_RETRY'
            continue
        if receipt.exists():states[job['name']]='RUNNING_OR_HELD_INVOCATION';continue
        status=dependency_state(job)
        if status!='READY':states[job['name']]=status;continue
        states[job['name']]='WAITING_FOR_MEASURED_GPU'
        if (Path(plan.get('control_dir',root/'finite_queue'))/'STOP_DISPATCH').exists():continue
        for gpu in job.get('preferred_gpus',plan['gpu_indices']):
            if gpu in busy or gpu not in plan['gpu_indices']:continue
            row=physical[gpu]
            if row['uuid']!=plan['gpu_uuids'][str(gpu)]:raise ValueError('Changed physical GPU mapping')
            if row['free_MiB']<job['min_free_MiB']:continue
            admitted(plan)
            # A second immediate check is also repeated inside the worker.
            row=gpu_rows()[gpu]
            if row['free_MiB']<job['min_free_MiB']:continue
            submit(plan,job,gpu,row);busy.add(gpu);states[job['name']]='SUBMITTED';break
    terminal=all(v=='COMPLETE' or v.startswith(('FAILED','HELD_')) for v in states.values())
    return dict(at=time.time(),states=states,terminal=terminal,
                status=('COMPLETE' if all(v=='COMPLETE' for v in states.values()) else 'COMPLETE_WITH_HELD_JOBS') if terminal else 'RUNNING')


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--plan',required=True);p.add_argument('--once',action='store_true');a=p.parse_args()
    plan=read(a.plan);body=dict(plan)
    if body.pop('sha256')!=seal(body):raise ValueError('Changed finite operator plan')
    if len({j['name'] for j in plan['jobs']})!=len(plan['jobs']):raise ValueError('Duplicate planned invocation')
    control=Path(plan.get('control_dir',Path(plan['root'])/'finite_queue'));control.mkdir(exist_ok=True)
    lock=(control/'owner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    atomic(control/'owner.json',dict(pid=os.getpid(),at=time.time(),plan_sha256=plan['sha256']))
    try:
        while True:
            result=tick(plan);atomic(control/'status.json',result)
            if result['terminal']:
                atomic(control/'COMPLETE.json',result)
                # Release only once all already-running study workers also exit.
                if plan.get('release_reservation_on_complete',True) and not busy_study_gpus(Path(plan['root'])):
                    reservation=Path(plan['root'])/'DEVICE_RESERVATION.json'
                    value=read(reservation);atomic(control/'reservation_before_release.json',value)
                    value.update(status='RELEASED',released_unix=time.time(),reason='finite registered screen queue reached terminal state')
                    value.pop('sha256',None);value['sha256']=seal(value);atomic(reservation,value)
                break
            if a.once:break
            time.sleep(60)
    except BaseException:
        atomic(control/'failure.json',dict(at=time.time(),error=traceback.format_exc(),automatic_retry=False));raise


if __name__=='__main__':main()
