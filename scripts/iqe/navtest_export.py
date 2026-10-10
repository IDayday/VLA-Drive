"""Run eight independent Base inference shards on one qualified idle GPU host."""
import argparse
from pathlib import Path
import os
import subprocess
import sys
import time
from iqe.io import atomic_json,read_json,lock
from iqe.resources import qualify


def main():
    p=argparse.ArgumentParser(__doc__)
    for k in ('config','dataset','output'):p.add_argument('--'+k,required=True)
    p.add_argument('--gpus',default='0,1,2,3,4,5,6,7');p.add_argument('--timeout',type=int,default=14400)
    a=p.parse_args();gpus=a.gpus.split(',');out=Path(a.output)
    if len(set(gpus))!=len(gpus) or not 1<=len(gpus)<=8 or a.timeout<=0:raise ValueError('GPU/time budget')
    with lock(out/'export_coordinator.lock'):
        os.environ['CUDA_VISIBLE_DEVICES']=a.gpus
        resources=qualify('cuda');atomic_json(out/'EXPORT_RESOURCES.json',resources)
        jobs=[];start=time.time()
        try:
            for rank,gpu in enumerate(gpus):
                command=[sys.executable,'-u','-m','iqe.navtest','export','--config',a.config,'--dataset',a.dataset,
                         '--output',a.output,'--rank',str(rank),'--world-size',str(len(gpus)),'--device','cuda']
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false')
                log=(out/f'export_rank_{rank}.log').open('a')
                child=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT)
                jobs.append((child,log,command))
            atomic_json(out/'EXPORT_LAUNCH.json',{'resources':resources,'started_unix':start,'coordinator_pid':os.getpid(),
                        'jobs':[{'pid':j.pid,'argv':cmd} for j,log,cmd in jobs],'timeout_seconds':a.timeout})
            while any(j.poll() is None for j,log,cmd in jobs):
                if any(j.poll() not in (None,0) for j,log,cmd in jobs):raise RuntimeError('export child failed; see per-rank logs')
                if time.time()-start>a.timeout:raise TimeoutError('bounded Navtest export paused; artifacts retained')
                time.sleep(2)
            if any(j.returncode for j,log,cmd in jobs):raise RuntimeError('export child failure')
            atomic_json(out/'EXPORT_COMPLETE.json',{'status':'COMPLETE','seconds':time.time()-start,'shards':len(jobs)})
        finally:
            for child,log,cmd in jobs:
                if child.poll() is None:child.terminate()
            for child,log,cmd in jobs:
                child.wait();log.close()


if __name__=='__main__':main()
