"""Bounded same-algorithm FP32 collective throughput probe, no model training."""
import json
import os
from pathlib import Path
import time
import torch
import torch.distributed as dist
from datetime import timedelta

def main():
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    dist.init_process_group('nccl',timeout=timedelta(seconds=120))
    rank,world=dist.get_rank(),dist.get_world_size()
    tensor=torch.empty(50_000_000,device='cuda',dtype=torch.float32)
    timings=[]
    for i in range(12):
        tensor.fill_(rank+1);torch.cuda.synchronize();dist.barrier()
        start=time.monotonic();dist.all_reduce(tensor);torch.cuda.synchronize()
        timings.append(time.monotonic()-start)
        assert bool((tensor==world*(world+1)//2).all())
    rows=[None]*world;dist.all_gather_object(rows,{'rank':rank,'seconds':timings[2:]})
    if rank==0:
        result={'status':'PASS','scope':'FP32 200 MB all-reduce, warmup2 measured10, no algorithm/precision changes',
                'world_size':world,'environment':{k:os.getenv(k) for k in ('NCCL_SOCKET_NTHREADS','NCCL_NSOCKS_PERTHREAD','NCCL_ALGO')},'ranks':rows}
        Path(os.environ['FLOW_TRANSPORT_OUTPUT']).write_text(json.dumps(result,indent=2))
        print(json.dumps(result),flush=True)
    dist.destroy_process_group()

if __name__=='__main__':main()
