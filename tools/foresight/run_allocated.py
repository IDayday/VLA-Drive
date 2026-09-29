"""Borrow only verified pressure-script GPUs and restore pressure after work."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
from tools.ddpolicy_vehicle.prepare_data import atomic_json


def pressure_parent(pid, script):
    for _ in range(16):
        proc=Path(f'/proc/{pid}')
        try:
            args=proc.joinpath('cmdline').read_bytes().decode().split('\0')
            cwd=proc.joinpath('cwd').resolve(strict=True)
            if any(x and Path(x).name==script.name and (cwd/x).resolve()==script for x in args):return pid
            pid=int(proc.joinpath('stat').read_text().split(') ',1)[1].split()[1])
        except (FileNotFoundError,ProcessLookupError):
            return None  # another allocation may have just released this worker
        if pid<=1:return None
    return None


def occupants(script):
    ids=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True)
    uuid={v.strip():int(k.strip()) for k,v in (line.split(',') for line in ids.splitlines())}
    rows=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
    result=[]
    for u,p in (line.split(',') for line in rows.splitlines()):
        pid=int(p.strip());parent=pressure_parent(pid,script)
        if parent is None and not Path(f'/proc/{pid}').exists():continue
        result.append((uuid[u.strip()],pid,parent))
    return result


def main():
    p=argparse.ArgumentParser(__doc__)
    for k in ('gpus','worktree','record','pressure-script','pressure-python'):p.add_argument('--'+k,required=True)
    p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args()
    command=a.command[1:] if a.command and a.command[0]=='--' else a.command
    if not command:raise ValueError('Missing workload command')
    if not (Path(a.worktree)/'.git').exists():raise ValueError('Worktree checkout is not ready')
    gpus={int(x) for x in a.gpus.split(',')};script=Path(a.pressure_script).resolve();out=Path(a.record)
    out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists():raise FileExistsError('New allocation record required')
    locks=[]
    for gpu in sorted(gpus):
        f=Path(f'/tmp/foresight_gpu_{gpu}.lock').open('a+');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
    rows=occupants(script);parents={parent for gpu,pid,parent in rows if gpu in gpus}
    if None in parents:raise RuntimeError('Allocated GPU has unrelated workload')
    if any(gpu not in gpus and parent in parents for gpu,pid,parent in rows):raise RuntimeError('Pressure parent also controls unallocated GPUs')
    record={'host':socket.gethostname(),'gpus':sorted(gpus),'command':command,'worktree':a.worktree,
            'pressure_parents_released':sorted(parents),'started_unix':time.time(),'status':'STARTING'}
    atomic_json(out,record);child=None
    def stop(*_):
        if child is not None and child.poll() is None:os.killpg(child.pid,signal.SIGTERM)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        for pid in parents:
            if pressure_parent(pid,script)!=pid:raise RuntimeError('Pressure process changed before signal')
            os.kill(pid,signal.SIGINT)
        deadline=time.time()+30
        while any(g in gpus for g,_,_ in occupants(script)):
            if time.time()>deadline:raise RuntimeError('Pressure did not release allocated cards')
            time.sleep(.5)
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=','.join(map(str,sorted(gpus))),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1')
        child=subprocess.Popen(command,cwd=a.worktree,env=env,start_new_session=True)
        record.update(pid=child.pid,status='RUNNING');atomic_json(out,record)
        code=child.wait();record.update(exit_code=code,status='COMPLETE' if code==0 else 'FAILED')
    finally:
        # A distributed supervisor may exit before its workers release CUDA.
        # Never restore pressure on top of a draining training process.
        if child is not None:
            deadline=time.time()+60
            while any(g in gpus for g,_,_ in occupants(script)) and time.time()<deadline:time.sleep(.5)
        restored=[]
        for gpu in sorted(gpus):
            if any(g==gpu for g,_,_ in occupants(script)):continue
            with out.with_name(out.stem+f'_pressure_gpu{gpu}.log').open('a') as log:
                proc=subprocess.Popen([a.pressure_python,'-u',str(script),'--gpus','0','--memory-gb','64','--status-interval','60'],
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu)),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            restored.append({'gpu':gpu,'parent_pid':proc.pid})
        deadline=time.time()+30
        while restored and time.time()<deadline:
            visible={g for g,_,parent in occupants(script) if parent in {r['parent_pid'] for r in restored}}
            if visible=={r['gpu'] for r in restored}:break
            time.sleep(.5)
        record.update(restored_pressure=restored,ended_unix=time.time());atomic_json(out,record)
    if record.get('exit_code',1):raise SystemExit(record.get('exit_code',1))

if __name__=='__main__':main()
