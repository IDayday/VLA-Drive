import torch
from torch import nn
import torch.distributed as dist
import torch.multiprocessing as mp
from navsim.agents.EpisodeDrive.gradient_diagnostics import isolated_same_batch_audit


class _DDPAgent(nn.Module):
    def __init__(self):
        super().__init__()
        self.world_model_enabled=True
        self.backbone=nn.Module()
        self.backbone.planning_register_adapter=nn.Module()
        self.backbone.planning_register_adapter.planning_registers=nn.Parameter(torch.ones(2,3))
    def forward(self,features):
        return {'planning_registers':self.backbone.planning_register_adapter.planning_registers*features['x']}
    def compute_loss(self,features,targets,predictions):
        value=predictions['planning_registers']
        return {'plan_loss':value.square().sum(),'wm_loss':value.sin().sum(),'wm_weight_current':value.new_tensor(.1)}


def _worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method=rendezvous,rank=rank,world_size=2)
    try:
        agent=_DDPAgent()
        wrapped=nn.parallel.DistributedDataParallel(agent)
        optimizer=torch.optim.AdamW(agent.parameters(),lr=.001)
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            features={'x':torch.tensor(float(rank+1))}
            # Match Lightning ordering: main graph first, isolated diagnostic,
            # then backward of that same pending DDP graph.
            pred=wrapped(features)
            rng=torch.random.get_rng_state()
            isolated_same_batch_audit(agent,features,{})
            assert torch.equal(rng,torch.random.get_rng_state())
            assert all(p.grad is None for p in agent.parameters())
            pred['planning_registers'].square().sum().backward()
            optimizer.step()
        parameter=next(agent.parameters()).detach()
        gathered=[torch.zeros_like(parameter) for _ in range(2)]
        dist.all_gather(gathered,parameter)
        assert torch.equal(gathered[0],gathered[1])
    finally:
        dist.destroy_process_group()


def test_two_rank_ddp_audit_preserves_pending_backward(tmp_path):
    mp.spawn(_worker,args=('file://'+str(tmp_path/'rendezvous'),),nprocs=2,join=True)
