"""Bounded model-only probe with exact cohost identities; not a training worker."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from starVLA.rl.qwen_world_compare.contracts import require_dispatch,authorization
from tools.qwen_world_compare.job_worker import shared_training_host
from tools.il_report.bd_nccl_probe import sample


def write(path, value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));os.replace(tmp,path)


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--plan',type=Path,required=True)
    a=p.parse_args();plan=json.loads(a.plan.read_text());out=Path(plan['receipt']);out.mkdir(exist_ok=False)
    if plan['command'][1:4]!=['-u','-m','tools.qwen_world_compare.benchmark_microbatch']:
        raise ValueError('This finite worker cannot launch a training entry point')
    host=require_dispatch();assert host==plan['host']
    source=Path(plan['source'])
    for name,digest in plan['source_files'].items():
        if hashlib.sha256((source/name).read_bytes()).hexdigest()!=digest:raise ValueError('Probe source changed')
    if plan['timeout_seconds']>1200:raise ValueError('Only finite model-only tests are authorized')
    slot=plan.get('lease_slot','primary')
    if slot not in ('primary','shared'):raise ValueError('Unknown lease namespace')
    if slot=='shared'and not shared_training_host(host,authorization()[0]):raise PermissionError('Sharing not authorized')
    gpu=plan['gpu'];lease=Path('/mnt/project/runtime_gpu_shared_workers'if slot=='shared'else'/mnt/project/runtime_gpu_leases')/(plan['uuid']+'.lock')
    lease.parent.mkdir(exist_ok=True)
    with lease.open('a')as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            for _ in range(2):
                device,owners=sample(gpu)
                if device['uuid']!=plan['uuid']or device['free_MiB']<plan['minimum_free_MiB']or owners!=plan['cohosts']:
                    raise RuntimeError('Pinned resources changed; no child started')
        except (BlockingIOError,RuntimeError,PermissionError)as error:
            write(out/'RESULT.json',dict(status='REFUSED_RESOURCE',child_not_started=True,returncode=75,
                reason=str(error),plan=str(a.plan),at_unix=time.time()))
            return 75
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONPATH=str(source),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false')
        start=time.time()
        with(out/'stdout.log').open('ab',buffering=0)as log:
            child=subprocess.Popen(plan['command'],cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            state=dict(status='RUNNING',host=host,gpu=gpu,uuid=plan['uuid'],plan=str(a.plan),start_unix=start,
                worker_pid=os.getpid(),child_pid=child.pid,child_start_ticks=int(Path(f'/proc/{child.pid}/stat').read_text().split()[21]),
                model_optimizer_updates=0)
            write(out/'RUNNING.json',state)
            try:code=child.wait(timeout=plan['timeout_seconds'])
            except subprocess.TimeoutExpired:
                # This subprocess owns no optimizer and cannot contain a learner.
                os.killpg(child.pid,signal.SIGTERM)
                try:code=child.wait(timeout=20)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);code=child.wait()
                state['timeout']=True
        state.update(status='COMPLETE'if code==0 else'FAILED',returncode=code,end_unix=time.time())
        write(out/'RESULT.json',state)
        return code


if __name__=='__main__':raise SystemExit(main())
