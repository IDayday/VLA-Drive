#!/usr/bin/env python3
"""Finalize only the registered geometric trial after its trainer releases the GPU.

No optimizer updates, new candidates, navtest, ACTIVE changes or process signals.
The exact trainer handle is checked while waiting; a vanished incomplete job is
reported as stopped rather than being silently restarted or waited on forever.
"""
import argparse
from datetime import datetime,timezone
import fcntl
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from iqe.contracts import require
from iqe.io import atomic_json,file_hash,read_json
from iqe.geometric_readout.full import load_config
from scripts.iqe.finish_robust_scorer import available_gpu as basic_available_gpu


def available_gpu(gpu):
    available,row=basic_available_gpu(gpu)
    # An empty container-visible process list is insufficient: skip unexplained
    # physical occupancy as well, even if most of an 80GB GPU remains free.
    used=int(subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.used',
        '--format=csv,noheader,nounits'],text=True).strip())
    row['used_MiB']=used
    return available and used<=256,row


def ready(config):
    folder=config.root/'scorer';path=folder/'result.json'
    if not path.exists():return False
    result=read_json(path)
    if result.get('status')!='COMPLETE' or result.get('optimizer_steps')!=6000 or not result.get('termination',{}).get('terminal'):
        return False
    try:
        with open(folder/'RUN.lock','a') as guard:fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:return False
    checkpoint=Path(result['checkpoint']);receipt=read_json(str(checkpoint)+'.COMPLETE.json')
    require(receipt['optimizer_step']==6000 and file_hash(checkpoint)==receipt['checksum'],'terminal checkpoint corrupt')
    require(file_hash(folder/'best.pt')==result['best_checkpoint_hash'],'selected snapshot corrupt')
    return True


def training_handle(pid,start_ticks):
    p=Path('/proc')/str(pid)
    try:
        stat=p.joinpath('stat').read_text().rsplit(')',1)[1].split()
        command=p.joinpath('cmdline').read_bytes().replace(b'\0',b' ').decode()
    except FileNotFoundError:return False
    require(stat[19]==str(start_ticks),'training PID was reused; do not act on it')
    require('iqe.geometric_readout.full train' in command,'unexpected training owner')
    return stat[0]!='Z'


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--config',required=True)
    p.add_argument('--gpu',type=int,required=True);p.add_argument('--expect-host',required=True)
    p.add_argument('--training-pid',type=int,required=True);p.add_argument('--training-start-ticks',required=True)
    p.add_argument('--max-wait-seconds',type=int,default=7200);a=p.parse_args()
    require(0<a.max_wait_seconds<=10800 and a.training_pid>0,'bounded wait and real trainer PID required')
    host=socket.gethostname().removesuffix('-worker-0');config=load_config(a.config)
    require(host==a.expect_host and host!='training-vla-zt2','wrong or protected host')
    def allowed():require(host in read_json('/mnt/project/server_dispatch_policy.json')['allowed_hosts'],
                          'permanent host permission withdrawn')
    allowed();folder=config.root/'runtime/finalizer';folder.mkdir(parents=True,exist_ok=True)
    with open(folder/'RUN.lock','a') as guard:
        fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
        paths=('scripts/iqe/finish_geometric_scorer.py','scripts/iqe/finish_robust_scorer.py',
               'iqe/geometric_readout/full.py','iqe/geometric_readout/evaluation.py',
               'iqe/geometric_readout/bundle.py','iqe/geometric_readout/bundle_check.py',
               'iqe/geometric_readout/model.py','iqe/geometric_readout/geometry.py')
        sources={name:file_hash(ROOT/name) for name in paths}
        launch=dict(host=host,pid=os.getpid(),start_ticks=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],
            training_pid=a.training_pid,training_start_ticks=a.training_start_ticks,gpu=a.gpu,
            source_hashes=sources,config_hash=file_hash(a.config),max_wait_seconds=a.max_wait_seconds,
            commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            utc=datetime.now(timezone.utc).isoformat(),no_process_signals=True)
        atomic_json(folder/'LAUNCH.json',launch)
        expires=time.monotonic()+a.max_wait_seconds
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(a.gpu),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
            CUBLAS_WORKSPACE_CONFIG=':4096:8',IQE_SPATIAL_READER_MODE='ram_mmap_v1',IQE_SPATIAL_MMAP_RECORDS='64',
            PYTHONUNBUFFERED='1')
        commands=[('evaluation',['-m','iqe.geometric_readout.evaluation']),
                  ('bundle',['-m','iqe.geometric_readout.bundle']),
                  ('capture',['-m','iqe.geometric_readout.bundle_check','capture']),
                  ('verify',['-m','iqe.geometric_readout.bundle_check','verify'])]
        for stage,command in commands:
            while True:
                allowed();complete=ready(config)
                available,gpu=available_gpu(a.gpu)
                if complete and available:break
                if not complete:require(training_handle(a.training_pid,a.training_start_ticks),
                    'trainer handle missing before complete terminal result; inspect and resume explicitly')
                atomic_json(folder/'STATUS.json',dict(status='WAITING_INPUTS' if not complete else 'WAITING_RESOURCE',
                    stage=stage,gpu=gpu,utc=datetime.now(timezone.utc).isoformat()))
                require(time.monotonic()<expires,'bounded finalizer wait exhausted')
                time.sleep(20)
            require(all(file_hash(ROOT/name)==value for name,value in sources.items()),'finalizer dependency changed')
            cmd=[sys.executable,*command,'--config',str(Path(a.config).resolve())]
            if stage!='bundle':cmd+=['--device','cuda:0']
            atomic_json(folder/'STATUS.json',dict(status='RUNNING',stage=stage,command=cmd,gpu=gpu,
                utc=datetime.now(timezone.utc).isoformat()))
            with open(folder/(stage+'.log'),'a') as log:
                result=subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=1800)
            atomic_json(folder/(stage+'.json'),dict(returncode=result.returncode,command=cmd,
                utc=datetime.now(timezone.utc).isoformat()))
            require(result.returncode==0,'finalizer stage failed: '+stage)
        atomic_json(folder/'STATUS.json',dict(status='COMPLETE',utc=datetime.now(timezone.utc).isoformat(),
            new_training=False,new_navtest=False,ACTIVE_updated=False))


if __name__=='__main__':main()
