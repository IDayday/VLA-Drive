"""Finite isolated NCCL measurements; never joins a training process group."""
import argparse,json,os,socket,time
from datetime import timedelta
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from starVLA.rl.qwen_sourceflow.contracts import require_dispatch,file_sha
from starVLA.rl.flow_grpo.transactions import atomic_json


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--output',required=True);p.add_argument('--sizes-MiB',nargs='+',type=float,default=[.004,4,64,256])
    p.add_argument('--repeats',type=int,default=5);a=p.parse_args()
    if a.repeats not in range(1,11) or any(s<=0 or s>256 for s in a.sizes_MiB):raise ValueError('Bounded probe required')
    host=require_dispatch();torch.set_num_threads(1);local=int(os.environ.get('LOCAL_RANK',0));torch.cuda.set_device(local)
    dist.init_process_group('nccl',timeout=timedelta(seconds=120));rank=dist.get_rank();world=dist.get_world_size()
    torch.cuda.reset_peak_memory_stats();rows=[]
    try:
        for MiB in a.sizes_MiB:
            n=max(1,int(MiB*1024*1024)//4);x=torch.ones(n,device='cuda',dtype=torch.float32)
            for _ in range(2):dist.all_reduce(x);torch.cuda.synchronize()
            values=[]
            for _ in range(a.repeats):
                dist.barrier();torch.cuda.synchronize();begin=time.perf_counter()
                dist.all_reduce(x);torch.cuda.synchronize();values.append(time.perf_counter()-begin)
            all_values=[None]*world;dist.all_gather_object(all_values,values)
            critical=np.max(np.array(all_values),axis=0)
            rows.append(dict(bytes=n*4,rank_seconds=all_values,mean_max_rank_seconds=float(critical.mean()),
                median_max_rank_seconds=float(np.median(critical)),algorithm_GB_per_second=n*4/float(np.median(critical))/1e9,
                ring_equivalent_bus_GB_per_second=2*(world-1)/world*n*4/float(np.median(critical))/1e9))
            del x
        record=dict(schema='native_finite_NCCL_collective_benchmark_v1',status='PASS',host=host,rank=rank,world=world,
            script_sha256=file_sha(__file__),rows=rows,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            environment={k:v for k,v in os.environ.items() if k.startswith(('NCCL_','MASTER_'))},
            limits='Shared GPUs; isolated communicator. Bandwidth is measured for these tensors/ranks, not a promised model scaling factor.')
        out=Path(a.output);out.mkdir(parents=True,exist_ok=True);path=out/f'rank{rank}.json'
        if path.exists():raise FileExistsError(path)
        atomic_json(path,record);dist.barrier()
        if rank==0:atomic_json(out/'COMPLETE.json',dict(status='PASS',world=world,rows=rows,script_sha256=file_sha(__file__)))
    finally:dist.destroy_process_group()


if __name__=='__main__':main()
