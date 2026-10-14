"""Registered process-group deadline guard for finite, atomic-record workers."""
import argparse
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

from iqe.contracts import require
from iqe.io import atomic_json, file_hash, read_json
from .temporary_lease import authorization, can_admit, start_ticks


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--lease',required=True);p.add_argument('--config',required=True)
    p.add_argument('--name',required=True);p.add_argument('--kind',choices=('score-helper','rich-precompute'),required=True)
    p.add_argument('--worker',type=int,required=True);p.add_argument('--workers',type=int,required=True)
    p.add_argument('--gpu',type=int);p.add_argument('--read-mib-per-second',type=float,default=32.)
    a=p.parse_args();lease=read_json(a.lease);host=socket.gethostname()
    require(a.name.replace('_','').replace('-','').isalnum(),'safe temporary worker name')
    require(os.getpid()==os.getpgrp() and os.getsid(0)==os.getpid(),'wrapper requires its own start_new_session')
    kind='cooperative-official-scoring' if a.kind=='score-helper' else 'precompute-rich-candidates'
    os.environ.update(IQE_TEMPORARY_LEASE_ID=lease['authorization_id'],
                      IQE_TEMPORARY_LEASE_EXPIRES_UNIX=str(lease['expires_unix']))
    authorization(read_json('/mnt/project/server_dispatch_policy.json'),host,kind,require_guard=False)
    require(can_admit(lease),'temporary new-stage admission closed')
    root=Path(a.lease).parent/'specialist_study';folder=root/'launches'/host
    receipt=folder/(a.name+'.json');require(not receipt.exists(),'temporary identity already dispatched; inspect receipt')
    stop=root/'stop_requests'/host/(a.name+'.json')
    guard={'schema':'registered_atomic_worker_deadline_v1','lease_id':lease['authorization_id'],
           'lease_path':str(Path(a.lease).resolve()),'lease_hash':file_hash(a.lease),'host':host,
           'pid':os.getpid(),'start_ticks':start_ticks(os.getpid()),'pgid':os.getpgrp(),
           'kind':kind,'config':str(Path(a.config).resolve()),'config_hash':file_hash(a.config),
           'started_unix':time.time(),'expires_unix':lease['expires_unix'],
           'stop_admission_unix':lease['expires_unix']-1800,'graceful_stop_unix':lease['expires_unix']-1200,
           'hard_exit_unix':lease['expires_unix']-60,'stop_file':str(stop),'optimizer_training':False}
    atomic_json(receipt,guard,immutable=True)
    os.environ.update(IQE_TEMPORARY_GUARD_RECEIPT=str(receipt),IQE_TEMPORARY_STOP_FILE=str(stop))
    repo=Path(__file__).resolve().parents[2]
    env=dict(os.environ,PYTHONPATH=str(repo),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    if a.kind=='score-helper':
        command=['/root/miniconda3/envs/navsim/bin/python','-u','-m','iqe.specialist_study.score_assist',
                 '--config',a.config,'--worker',str(a.worker),'--workers',str(a.workers),'--deadline-hours','8']
        env['CUDA_VISIBLE_DEVICES']=''
    else:
        require(a.gpu is not None and 0<=a.gpu<8,'explicit temporary GPU required')
        from .capacity import probe
        capacity=probe(a.gpu);atomic_json(folder/(a.name+'.CAPACITY.json'),capacity)
        require(capacity['ready'],'temporary GPU not free; no unknown-owner signals')
        command=[sys.executable,'-u','-m','iqe.specialist_study.rich_precompute','--config',a.config,
                 '--shard',str(a.worker),'--shards',str(a.workers),'--read-mib-per-second',str(a.read_mib_per_second)]
        env['CUDA_VISIBLE_DEVICES']=str(a.gpu)
    requested=[False]
    def request_stop(*_):requested[0]=True
    signal.signal(signal.SIGTERM,request_stop);signal.signal(signal.SIGINT,request_stop)
    with open(folder/(a.name+'.log'),'ab') as log:
        child=subprocess.Popen(command,cwd=repo,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
    atomic_json(folder/(a.name+'.CHILD.json'),{'pid':child.pid,'start_ticks':start_ticks(child.pid),
                'pgid':os.getpgid(child.pid),'command':command,'lease_hash':guard['lease_hash']},immutable=True)
    while child.poll() is None:
        now=time.time()
        if requested[0] or now>=guard['graceful_stop_unix']:
            atomic_json(stop,{'lease_id':lease['authorization_id'],'requested_unix':now,'reason':'absolute_deadline_or_operator'})
        if now>=guard['hard_exit_unix']:
            atomic_json(folder/(a.name+'.EXIT.json'),{'status':'HARD_BOUNDARY_RECORDS_RETAINED',
                        'at':now,'child_pid':child.pid,'lease_hash':guard['lease_hash']})
            os.killpg(os.getpgrp(),signal.SIGKILL)
        time.sleep(2)
    atomic_json(folder/(a.name+'.EXIT.json'),{'status':'COMPLETE' if child.returncode==0 else 'FAILED',
                'returncode':child.returncode,'finished_unix':time.time(),'lease_hash':guard['lease_hash'],
                'committed_records_retained':True})
    require(child.returncode==0,'temporary worker failed; preserve evidence and resume on permanent resource')


if __name__=='__main__':main()
