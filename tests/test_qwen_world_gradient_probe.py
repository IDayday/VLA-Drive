"""Real reduced-gradient arithmetic and DDP-backward noninterference."""
from pathlib import Path
import tempfile

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from starVLA.rl.qwen_world_research_algorithms.gradient_probe import measure_components
from starVLA.rl.simwam_recipe import checkpoint


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.action = torch.nn.Parameter(torch.tensor([1.,2.]))
        self.scene_provider = torch.nn.Linear(2,1,bias=False)
        with torch.no_grad():
            self.scene_provider.weight.copy_(torch.tensor([[3.,4.]]))
        self.report = None

    def forward(self, rank):
        pg = (rank+1)*self.action.sum()+self.scene_provider.weight.sum()
        anchor = -self.action.sum()+2*self.scene_provider.weight.sum()
        self.report = measure_components(self, {'pg':pg,'anchor':anchor})
        return pg+anchor


def test_component_arithmetic_and_no_grad_mutation():
    model=Toy()
    total=model(0)
    report=model.report['scopes']
    assert report['action']['cosine']['anchor__pg']==pytest.approx(-1)
    assert report['memory']['cosine']['anchor__pg']==pytest.approx(1)
    assert model.action.grad is None
    total.backward()
    assert torch.equal(model.action.grad,torch.zeros(2))
    assert torch.equal(model.scene_provider.weight.grad,torch.full((1,2),3.))


def _distributed_worker(rank, rendezvous):
    dist.init_process_group('gloo',init_method='file://'+rendezvous,rank=rank,world_size=2)
    checkpoint.init_control_group()
    model=Toy()
    ddp=DDP(model,find_unused_parameters=True,static_graph=False)
    opt=torch.optim.SGD(ddp.parameters(),lr=.1)
    try:
        for _ in range(2):
            opt.zero_grad(set_to_none=True)
            total=ddp(rank)
            report=model.report['scopes']
            assert report['action']['global_mean_gradient_norm']['pg']==pytest.approx(1.5*2**.5)
            assert report['action']['summed_gradient_norm']==pytest.approx(.5*2**.5)
            assert model.action.grad is None
            total.backward()
            assert torch.equal(model.action.grad,torch.full((2,),.5))
            assert torch.equal(model.scene_provider.weight.grad,torch.full((1,2),3.))
            opt.step()
        assert torch.allclose(model.action.detach(),torch.tensor([.9,1.9]))
    finally:
        dist.destroy_process_group()


def test_ddp_normal_backward_and_update_are_unchanged():
    with tempfile.TemporaryDirectory() as d:
        mp.spawn(_distributed_worker,args=(str(Path(d)/'rendezvous'),),nprocs=2,join=True)
