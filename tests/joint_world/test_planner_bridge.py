import torch

from tools.joint_world.planner_runtime import CachedCurrentPlanner, graph_noise


def test_frozen_graph_probe_matches_online_rollout_and_bridge_gradients():
    torch.manual_seed(8)
    model=CachedCurrentPlanner(16,dict(dim=16,heads=4,layers=1,scale_m=20.,sampling_steps=2,
        trajectory_mode='current_residual',agent_scale_m=5.))
    model.graph.requires_grad_(False)
    current=dict(actor_features=torch.randn(2,3,16),context=torch.randn(2,4,16),
                 current_xy=torch.randn(2,3,2),existence=torch.rand(2,3))
    native=torch.randn(2,5,16);noise=graph_noise(2,3,8,'cpu')
    a,joint=model.condition_from_frozen_graph(native,current,noise)
    b,other,_=model.rollout_condition(native,current,noise)
    torch.testing.assert_close(a,native,rtol=0,atol=0)
    torch.testing.assert_close(a,b,rtol=0,atol=0)
    torch.testing.assert_close(joint,other,rtol=0,atol=0)
    a.square().mean().backward()
    assert model.adapter.gate.grad.abs()>0
    assert all(p.grad is None for p in model.graph.parameters())
    model.zero_grad(set_to_none=True)
    with torch.no_grad():model.adapter.gate.fill_(.1)
    a,_=model.condition_from_frozen_graph(native,current,noise);a.square().mean().backward()
    assert model.graph_to_world.weight.grad.norm()>0
    assert model.adapter.attention.in_proj_weight.grad.norm()>0


def test_frozen_probe_rejects_silently_detaching_trainable_graph():
    import pytest
    model=CachedCurrentPlanner(16,dict(dim=16,heads=4,layers=1))
    with pytest.raises(ValueError,match='trainable graph'):
        model.condition_from_frozen_graph(None,None,None)


def test_per_scene_graph_noise_independent_of_training_batch_size():
    a=graph_noise(1,4,8,'cpu');b=graph_noise(3,4,8,'cpu')
    for sample in b:torch.testing.assert_close(sample,a[0],rtol=0,atol=0)
