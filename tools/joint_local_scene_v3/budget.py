"""Locked run identities, full wall GPU accounting and separate update caps."""
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import socket
import threading
import time


def atomic_json(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def initialize_ledger(path,gpu_hours=2.,synthetic_updates=20,real_updates=0):
    path=Path(path)
    if path.exists():raise FileExistsError('Ledger already exists')
    path.parent.mkdir(parents=True,exist_ok=True)
    if gpu_hours<=0 or synthetic_updates<0 or real_updates<0:raise ValueError('Invalid budget')
    atomic_json(path,{'schema_version':1,'gpu_hour_cap':gpu_hours,'update_caps':{'synthetic':synthetic_updates,'real':real_updates},'runs':[]})


@contextmanager
def locked(path):
    path=Path(path)
    with open(str(path)+'.lock','a') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX)
        data=json.loads(path.read_text())
        if data['schema_version']!=1:raise ValueError('Budget schema mismatch')
        yield data
        atomic_json(path,data)


def charge(data):
    now=time.time()
    for row in data['runs']:
        if row['status']=='running':
            elapsed=now-row['heartbeat'];row['wall_seconds']+=elapsed;row['gpu_hours']+=elapsed*row['gpu_count']/3600;row['heartbeat']=now
    data['gpu_hours']=sum(r['gpu_hours'] for r in data['runs'])
    data['optimizer_updates']={kind:sum(r['completed_updates'][kind] for r in data['runs']) for kind in ('synthetic','real')}
    data['attempted_optimizer_updates']={kind:sum(r['attempted_updates'][kind] for r in data['runs']) for kind in ('synthetic','real')}


class BudgetRun:
    def __init__(self,ledger,run_id,identity,gpu_count=0,resume=False):
        self.ledger,self.run_id,self.identity,self.gpu_count,self.resume=ledger,run_id,identity,gpu_count,resume
        if not run_id or gpu_count not in (0,1):raise ValueError('Explicit single-process run identity required')
        self.stop=threading.Event();self.exceeded=False;self.thread=None;self.terminal='complete'
    def __enter__(self):
        with locked(self.ledger) as data:
            charge(data)
            if data['gpu_hours']>=data['gpu_hour_cap']:raise RuntimeError('GPU budget exhausted')
            row=next((r for r in data['runs'] if r['id']==self.run_id),None)
            if row is not None:
                if not self.resume or row['status'] not in ('paused','failed') or row['identity']!=self.identity or row['gpu_count']!=self.gpu_count:raise ValueError('Existing run cannot be duplicated or resumed with changed identity')
            else:
                if self.resume:raise ValueError('Cannot resume an unknown run ID')
                output=self.identity.get('output')
                if output is not None and any(r['identity'].get('output')==output for r in data['runs']):raise ValueError('Output is already owned by another run ID')
                row={'id':self.run_id,'identity':self.identity,'gpu_count':self.gpu_count,'gpu_hours':0.,'wall_seconds':0.,
                     'attempted_updates':{'synthetic':0,'real':0},'completed_updates':{'synthetic':0,'real':0},
                     'forward_calls':{'synthetic':0,'real':0},'backward_calls':0};data['runs'].append(row)
            row.update(status='running',pid=os.getpid(),host=socket.gethostname(),heartbeat=time.time())
        def heartbeat():
            while not self.stop.wait(.5):
                try:self.check()
                except RuntimeError:self.exceeded=True
        self.thread=threading.Thread(target=heartbeat,daemon=True);self.thread.start();return self
    def check(self):
        with locked(self.ledger) as data:
            charge(data);self.exceeded=data['gpu_hours']>=data['gpu_hour_cap']
        if self.exceeded:raise RuntimeError('GPU budget exhausted; save at the next boundary')
    def require_updates(self,kind):
        if kind not in ('real','synthetic'):raise ValueError('Unknown data provenance')
        with locked(self.ledger) as data:
            charge(data)
            if data['attempted_optimizer_updates'][kind]>=data['update_caps'][kind]:raise RuntimeError(f'{kind} optimizer updates forbidden/exhausted')
    def claim_update(self,kind):
        with locked(self.ledger) as data:
            charge(data)
            if data['gpu_hours']>=data['gpu_hour_cap'] or data['attempted_optimizer_updates'][kind]>=data['update_caps'][kind]:raise RuntimeError('Update budget exhausted')
            row=next(r for r in data['runs'] if r['id']==self.run_id);row['attempted_updates'][kind]+=1
    def completed_update(self,kind):
        with locked(self.ledger) as data:
            charge(data);row=next(r for r in data['runs'] if r['id']==self.run_id)
            if row['completed_updates'][kind]>=row['attempted_updates'][kind]:raise RuntimeError('Unaccounted optimizer update')
            row['completed_updates'][kind]+=1;charge(data)
    def note(self,kind,forwards=0,backwards=0):
        with locked(self.ledger) as data:
            charge(data);row=next(r for r in data['runs'] if r['id']==self.run_id)
            row['forward_calls'][kind]+=forwards;row['backward_calls']+=backwards
    def __exit__(self,kind,error,trace):
        self.stop.set()
        if self.thread:self.thread.join(timeout=2)
        with locked(self.ledger) as data:
            charge(data);row=next(r for r in data['runs'] if r['id']==self.run_id)
            row['status']='failed' if kind is not None else self.terminal;row['error']=str(error) if error else None;charge(data)
