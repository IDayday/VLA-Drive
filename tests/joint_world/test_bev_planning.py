import copy
import torch

from tools.joint_world.planner_runtime import CachedCurrentPlanner,graph_noise
from starVLA.model.modules.joint_world.flow import training_loss_sums


def test_bev_graph_gate_learns_before_outer_planning_gate_and_targets_remain_separate():
    torch.manual_seed(17)
    model=CachedCurrentPlanner(16,dict(dim=16,heads=4,layers=1,sampling_steps=2,
        trajectory_mode='current_residual',agent_scale_m=5.),bev_enabled=True)
    model.graph.requires_grad_(False)
    current=dict(actor_features=torch.randn(1,3,16),context=torch.randn(1,4,16),
                 current_xy=torch.randn(1,3,2),existence=torch.rand(1,3))
    features=torch.randn(1,1960,1024);coords=torch.randn(1,1960,3);support=torch.ones(1,1960,dtype=torch.bool)
    updated,pred=model.fuse_bev(current,features,coords,support)
    torch.testing.assert_close(updated['actor_features'],current['actor_features'],rtol=0,atol=0)
    xy=torch.randn(1,3,8,2);valid=torch.ones(1,3,8,dtype=torch.bool)
    sums,counts=training_loss_sums(model.graph,xy,valid,torch.ones(1,3,dtype=torch.bool),torch.randn_like(xy),torch.tensor([.5]),**updated)
    sum(sums[k]/counts[k] for k in sums).backward()
    assert model.bev_fusion.gate.grad.abs()>0
    assert all(p.grad is None for p in model.graph.parameters())
    model.zero_grad(set_to_none=True)
    with torch.no_grad():model.bev_fusion.gate.fill_(.1);model.adapter.gate.fill_(.1)
    updated,_=model.fuse_bev(current,features,coords,support)
    native=torch.randn(1,5,16);noise=graph_noise(1,3,8,'cpu')
    condition,_,_=model.rollout_condition(native,updated,noise)
    condition.square().mean().backward()
    assert model.bev_encoder.project[1].weight.grad.norm()>0
    assert model.bev_fusion.output.weight.grad.norm()>0
    saved=copy.deepcopy(model.state_dict())
    restore=CachedCurrentPlanner(16,model.graph_config,bev_enabled=True);restore.load_state_dict(saved,strict=True)
    restore.graph.requires_grad_(False)
    other,_=restore.fuse_bev(current,features,coords,support)
    actual=restore.rollout_condition(native,other,noise)[0]
    torch.testing.assert_close(condition,actual,rtol=0,atol=0)
