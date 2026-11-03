"""Bounded exclusive GPU lease for finite latency measurement, no evictions."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--wait-seconds', type=int, default=1800)
    args=p.parse_args();cfg=json.loads(Path(args.config).read_text())
    from starVLA.rl.qwen_world_compare.contracts import require_dispatch
    require_dispatch(admit_long_job=True)
    # One existing evaluation may finish naturally. Reserve the otherwise
    # vacant primary slot while waiting, never signal an existing process.
    uuid=cfg['gpu_uuid'];locks=[];deadline=time.monotonic()+args.wait_seconds
    folders=('runtime_gpu_leases','runtime_gpu_shared_workers','runtime_gpu_research_slots',
             'runtime_gpu_evaluation_slots','runtime_gpu_research_evaluation_slots','runtime_gpu_profiled_third_workers')
    try:
        primary=Path('/mnt/project')/folders[0]/(uuid+'.lock')
        primary.parent.mkdir(exist_ok=True);stream=primary.open('a')
        fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(stream)
        while True:
            require_dispatch(admit_long_job=True)
            acquired=[]
            try:
                for folder in folders[1:]:
                    path=Path('/mnt/project')/folder/(uuid+'.lock');path.parent.mkdir(exist_ok=True)
                    s=path.open('a');acquired.append(s);fcntl.flock(s,fcntl.LOCK_EX|fcntl.LOCK_NB)
                gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
                row=next([v.strip()for v in r.split(',')]for r in gpu.splitlines()if uuid in r)
                owners=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader,nounits'],text=True)
                if int(row[0])!=cfg['gpu_index'] or int(row[2])<79000 or int(row[3])>10 or uuid in owners:
                    raise BlockingIOError('Actual GPU is still occupied')
                locks.extend(acquired);break
            except BlockingIOError:
                for s in acquired:s.close()
                if time.monotonic()>=deadline:raise TimeoutError('No exclusive capacity before finite admission deadline')
                time.sleep(10)
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(cfg['gpu_index']),OMP_NUM_THREADS=str(cfg['cpu_threads']),
                 MKL_NUM_THREADS=str(cfg['cpu_threads']),TOKENIZERS_PARALLELISM='false',
                 CUBLAS_WORKSPACE_CONFIG=':4096:8',FLASH_ATTENTION_DETERMINISTIC='1')
        command=[cfg['python'],'-u',str(Path(__file__).with_name('benchmark.py')),'--config',args.config,'--output',args.output]
        print(json.dumps(dict(status='EXCLUSIVE',gpu_uuid=uuid,command=command)),flush=True)
        return subprocess.call(command,cwd=cfg['scientist_source'],env=env)
    finally:
        for s in locks:s.close()


if __name__=='__main__':raise SystemExit(main())
