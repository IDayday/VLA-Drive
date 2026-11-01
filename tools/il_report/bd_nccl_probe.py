"""Finite two-host NCCL probe with pinned cohosts and one actual GPU lease."""
import argparse
from datetime import timedelta
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time


def sample(gpu):
    lines=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True)
    device=next(dict(index=int(i),uuid=u.strip(),free_MiB=int(f))for i,u,f in(x.split(',')for x in lines.splitlines())if int(i)==gpu)
    owners=[]
    for line in subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader'],text=True).splitlines():
        pid,uuid=line.split(',')
        if uuid.strip()!=device['uuid']:continue
        root=Path('/proc')/pid.strip();cmd=(root/'cmdline').read_bytes();args=cmd.decode().split(chr(0))
        if not any(a.startswith(('/mnt/project/qwen-world-','/mnt/project/VLA-Drive-qwen'))for a in args):
            raise PermissionError('Unidentified cohost; probe must not run')
        owners.append(dict(pid=int(pid),start_ticks=int((root/'stat').read_text().split()[21]),cmd_sha256=hashlib.sha256(cmd).hexdigest()))
    return device,sorted(owners,key=lambda r:r['pid'])


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--rank',type=int,choices=(0,1),required=True);a=p.parse_args();plan=json.loads(a.plan.read_text());row=plan['ranks'][a.rank]
    sys=__import__('sys');sys.path.insert(0,plan['source'])
    from starVLA.rl.qwen_world_compare.contracts import require_dispatch
    host=require_dispatch();assert host==row['host']
    lease=Path('/mnt/project/runtime_gpu_leases')/(row['uuid']+'.lock');lease.parent.mkdir(exist_ok=True)
    with lease.open('a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for _ in range(2):
            device,owners=sample(row['gpu'])
            if device['uuid']!=row['uuid']or device['free_MiB']<4096 or owners!=row['cohosts']:
                raise RuntimeError('Pinned GPU/cohost admission changed before probe')
        os.environ['CUDA_VISIBLE_DEVICES']=str(row['gpu']);os.environ['OMP_NUM_THREADS']='2'
        os.environ['NCCL_DEBUG']='INFO';os.environ['NCCL_DEBUG_SUBSYS']='INIT,NET'
        import torch
        import torch.distributed as dist
        torch.set_num_threads(2);torch.cuda.set_device(0)
        start=time.monotonic();dist.init_process_group('nccl',init_method=plan['init_method'],rank=a.rank,world_size=2,timeout=timedelta(seconds=75))
        results=[]
        for MiB in [1,16,64,256]:
            tensor=torch.ones(MiB*1024**2//4,device='cuda',dtype=torch.float32)
            for _ in range(2):dist.all_reduce(tensor);tensor.fill_(1)
            torch.cuda.synchronize();dist.barrier();tick=time.monotonic()
            for _ in range(5):dist.all_reduce(tensor);tensor.fill_(1)
            torch.cuda.synchronize();elapsed=(time.monotonic()-tick)/5
            results.append(dict(MiB=MiB,mean_seconds=elapsed,payload_GiB_s=(MiB/1024)/elapsed))
            del tensor
        dist.destroy_process_group()
        out=Path(plan['output'])/f'rank{a.rank}.json';out.write_text(json.dumps(dict(status='COMPLETE',host=host,rank=a.rank,rows=results,
            wall_seconds=time.monotonic()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            model_optimizer_updates=0,scope='Two-host one-GPU-per-host collective, not full16rank model speedup'),indent=2))


if __name__=='__main__':main()
