"""Independent GPU-hour campaign; no inherited step cap. Failed work stays charged."""
import fcntl
import json
import os
import socket
import time
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def locked(path):
    path=Path(path)
    with open(str(path)+'.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        data=json.loads(path.read_text())
        yield data
        temp=path.with_suffix('.tmp');temp.write_text(json.dumps(data,indent=2)+'\n');temp.replace(path)


def total_hours(data):
    now=time.time()
    return sum(r.get('gpu_hours',0)+(now-r['heartbeat'])*r['gpu_count']/3600
               if r['status']=='running' else r.get('gpu_hours',0) for r in data['runs'])


class Run:
    def __init__(self,path,run_id,metadata,gpu_count=1,resume=False,max_gpu_hours=None):
        self.path,self.id,self.metadata=path,run_id,metadata
        self.gpu_count,self.resume,self.max_hours=gpu_count,resume,max_gpu_hours
        self.step=0
        self.terminal_status=None

    def __enter__(self):
        with locked(self.path) as data:
            row=next((r for r in data['runs'] if r['id']==self.id),None)
            if row is not None and (not self.resume or row['status'] not in ('paused','failed')):
                raise ValueError('Existing run requires accounted explicit resume; cannot restart running/completed run')
            if total_hours(data)>=data['gpu_hour_cap']:raise RuntimeError('Independent campaign GPU-hour cap reached')
            if row is None:
                row={'id':self.id,'optimizer_steps':0,'gpu_hours':0};data['runs'].append(row)
            elif row['metadata']!=self.metadata or row['gpu_count']!=self.gpu_count:
                raise ValueError('Resume source/config/budget identity changed')
            self.step=row['optimizer_steps']
            row.update(status='running',metadata=self.metadata,gpu_count=self.gpu_count,pid=os.getpid(),
                       host=socket.gethostname(),heartbeat=time.time(),cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                       max_gpu_hours=self.max_hours)
        return self

    def update(self,step=None,status='running'):
        if step is not None:
            if step<self.step:raise ValueError('Optimizer accounting cannot decrease')
            self.step=step
        with locked(self.path) as data:
            row=next(r for r in data['runs'] if r['id']==self.id);now=time.time()
            if row['status']=='running':row['gpu_hours']+=(now-row['heartbeat'])*row['gpu_count']/3600
            row.update(heartbeat=now,optimizer_steps=self.step,status=status)
            data['gpu_hours']=total_hours(data);data['optimizer_steps']=sum(r['optimizer_steps'] for r in data['runs'])
            allowed=data['gpu_hours']<data['gpu_hour_cap'] and (self.max_hours is None or row['gpu_hours']<self.max_hours)
        return allowed

    def __exit__(self,kind,error,trace):
        self.update(status=(self.terminal_status or 'complete') if kind is None else 'failed')

