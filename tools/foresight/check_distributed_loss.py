"""Two REAL ranks: empty auxiliary rank and uneven valid counts, with accumulation."""
import argparse
import json
from pathlib import Path
import tempfile
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from starVLA.model.modules.foresight.losses import masked_regression
from tools.ddpolicy_vehicle.run_meter import metered_run


def worker(rank,backend,url,output):
    device=torch.device('cuda',rank) if backend=='nccl' else torch.device('cpu')
    if device.type=='cuda':torch.cuda.set_device(rank)
    dist.init_process_group(backend,init_method=url,rank=rank,world_size=2)
    model=torch.nn.Linear(1,1,bias=False).to(device)
    with torch.no_grad():model.weight.fill_(1.)
    model=DDP(model,device_ids=[rank] if device.type=='cuda' else None)
    x=torch.tensor([[1.],[2.]] if rank==0 else [[3.],[4.]],device=device)
    target=torch.full_like(x,float('nan')) if rank==0 else torch.zeros_like(x)
    valid=torch.full_like(x,rank==1,dtype=torch.bool)
    loss,count=masked_regression(model(x),target,valid);loss.backward()
    direct=float(model.module.weight.grad)
    model.zero_grad()
    # Denominator2 is the global UPDATE count across ranks and microbatches.
    for i in range(2):
        loss,_=masked_regression(model(x[i:i+1]),target[i:i+1],valid[i:i+1],global_count=2)
        loss.backward()
    accumulated=float(model.module.weight.grad)
    assert abs(direct-25.)<1e-6 and abs(accumulated-25.)<1e-6
    if rank==0:Path(output).write_text(json.dumps({'backend':backend,'ranks':2,'empty_auxiliary_rank':True,
        'expected_single_process_gradient':25.,'distributed_gradient':direct,'accumulated_gradient':accumulated,'passed':True},indent=2))
    dist.destroy_process_group()


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--backend',choices=('gloo','nccl'),required=True)
    p.add_argument('--output',required=True);p.add_argument('--campaign-root',required=True);p.add_argument('--run-id',required=True)
    a=p.parse_args()
    with metered_run(a.campaign_root,a.run_id,2 if a.backend=='nccl' else 0,{'kind':'normalization_test','real_optimizer_updates':0}):
        with tempfile.TemporaryDirectory() as temp:
            mp.spawn(worker,args=(a.backend,'file://'+str(Path(temp)/'rendezvous'),a.output),nprocs=2,join=True)


if __name__=='__main__':main()
