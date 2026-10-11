"""Capacity-gated full cache launcher. No model training and no checkpoint cleanup."""
import argparse
import math
import os
from pathlib import Path
import subprocess
import sys
from iqe.contracts import require
from iqe.io import atomic_json, read_json
from iqe.resources import qualify
from iqe.vision_cache.plan import load_plan, capacity_check
from iqe.vision_cache.qualification import validate_cache_qualification


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--workers',type=int,default=8)
    p.add_argument('--rate-mib-s-per-worker',type=float,default=128.)
    p.add_argument('--execute',action='store_true',help='otherwise capacity/readiness only')
    a=p.parse_args();require(1<=a.workers<=8 and 0<a.rate_mib_s_per_worker<=256,'explicit bounded IO/workers')
    os.chdir(Path(__file__).resolve().parents[3])
    root=Path('/mnt/project/iqe-fp32-cache-production-runtime-3931fed');output=root/'full_fit_40k';output.mkdir(exist_ok=True)
    plan=load_plan(root/'FULL_PLAN_40K.json')
    cert=validate_cache_qualification(root/'CACHE_QUALIFICATION.json',fit_hash=plan['fit_sha256'],for_build=True)
    prior=read_json(root/'representative200/COMPLETE.json')
    retention=plan['checkpoint_retention_budget'];protected=retention['remaining_count']*retention['parent_bytes']+(180+200)*1024**3
    done=sum(f.stat().st_size for f in output.glob('shard_*.bin') if Path(str(f)+'.COMPLETE.json').exists())
    needed=max(0,int(cert['representative_record_max_bytes']*plan['requested_scenes']*1.05)-prior['bytes']-done)
    capacity=capacity_check(output,needed,reserve_bytes=protected,temporary_bytes=a.workers*2*1024**3)
    atomic_json(output/'LAUNCH_CAPACITY.json',capacity)
    require(capacity['sufficient'],'BLOCKED_SHARED_CAPACITY: preserve Base checkpoints/22k/safety reserve')
    if not a.execute:print({'status':'CAPACITY_READY','execution_started':False,'capacity':capacity});return
    resources=qualify('cuda');require(len(resources['GPUs'])>=a.workers,'idle GPU count below worker budget')
    shards=math.ceil(math.ceil(plan['requested_scenes']/plan['records_per_shard'])/a.workers)
    max_gib=2*shards+2;procs=[]
    for worker in range(a.workers):
        argv=[sys.executable,'-u','-m','iqe.vision_cache.cli','cache-build','--config','configs/iqe/fp32_cache/source.yaml',
            '--plan',str(root/'FULL_PLAN_40K.json'),'--root',str(output),'--device','cuda:0','--worker',str(worker),
            '--workers',str(a.workers),'--max-shards',str(shards),'--max-gib',str(max_gib),'--reserve-gib',str(protected/1024**3),
            '--rate-mib-s',str(a.rate_mib_s_per_worker),'--qualification',str(root/'CACHE_QUALIFICATION.json'),
            '--hold-file','/mnt/project/iqe-fp32-cache-production-task-20261011/CACHE_BUILD_HOLD.json',
            '--reuse-cache',str(root/'representative200')]
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=resources['GPUs'][worker]['index'],OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
        with (output/f'worker_{worker}.log').open('ab') as log:
            proc=subprocess.Popen(argv,env=env,stdout=log,stderr=subprocess.STDOUT);procs.append(proc)
    atomic_json(output/'LAUNCH.json',{'resources':resources,'pids':[x.pid for x in procs],
                                    'workers':a.workers,'rate_mib_s_per_worker':a.rate_mib_s_per_worker})
    codes=[x.wait() for x in procs]
    atomic_json(output/'EXIT.json',{'exit_codes':codes,'complete':(output/'COMPLETE.json').exists()})
    require(all(x==0 for x in codes) and (output/'COMPLETE.json').exists(),'full build incomplete; preserve shards and resume same plan')

if __name__=='__main__':
    import json
    try:main()
    except Exception as error:
        print(json.dumps({'status':'BLOCKED' if 'BLOCKED' in str(error) else 'FAILED','error':str(error)}))
        sys.exit(2)
