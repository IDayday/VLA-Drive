import os
from pathlib import Path
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from iqe.decision_selection.objectives import cost_ce,fixed_normalizer


def worker(rank, rendezvous, output):
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method='file://'+rendezvous,rank=rank,world_size=2)
    model=torch.nn.Linear(2,3,bias=False)
    with torch.no_grad():model.weight.copy_(torch.arange(6).reshape(3,2)/10)
    wrapper=torch.nn.parallel.DistributedDataParallel(model)
    inputs=torch.tensor([[1.,2.],[2.,1.],[.5,1.]])
    rewards=torch.tensor([[.5,.5,.5],[.9,0.,.2],[.4,.6,.1]])
    valid=torch.ones_like(rewards,dtype=torch.bool);z=fixed_normalizer(rewards,valid)
    ix=torch.tensor([0] if rank==0 else [1,2]);local=cost_ce(wrapper(inputs[ix]),rewards[ix],valid[ix],z)
    count=local.denominator.clone();dist.all_reduce(count)
    (local.numerator*2/count).backward()
    if rank==0:torch.save(model.weight.grad,output)
    dist.destroy_process_group()


def test_gloo_unequal_rank_denominators_and_all_tie_rank(tmp_path):
    rendezvous=str(tmp_path/'rendezvous');output=str(tmp_path/'gradient.pt')
    mp.spawn(worker,args=(rendezvous,output),nprocs=2,join=True)
    model=torch.nn.Linear(2,3,bias=False)
    with torch.no_grad():model.weight.copy_(torch.arange(6).reshape(3,2)/10)
    inputs=torch.tensor([[1.,2.],[2.,1.],[.5,1.]])
    rewards=torch.tensor([[.5,.5,.5],[.9,0.,.2],[.4,.6,.1]])
    valid=torch.ones_like(rewards,dtype=torch.bool);z=fixed_normalizer(rewards,valid)
    full=cost_ce(model(inputs),rewards,valid,z);(full.numerator/full.denominator).backward()
    assert torch.allclose(model.weight.grad,torch.load(output,weights_only=True),atol=1e-7)
