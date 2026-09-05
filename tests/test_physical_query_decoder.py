import inspect
import pytest
import torch
from navsim.agents.EpisodeDrive.layers.world_model.physical_query_decoder import PhysicalQueryDecoder, trajectory_features


@pytest.mark.parametrize('k',[1,8,64])
def test_k_chunk_independence_gradient(k):
    torch.manual_seed(17)
    torch.set_num_threads(2)
    model=PhysicalQueryDecoder()
    registers=torch.randn(2,16,256,requires_grad=True)
    trajectories=torch.randn(2,k,8,3,requires_grad=True)
    status=torch.randn(2,8)
    out=model(registers,trajectories,status)
    chunk=model(registers,trajectories,status,chunk_size=3)
    assert out['gap_logits'].shape==(2,k,8,5)
    for name in out:
        torch.testing.assert_close(out[name],chunk[name],atol=1e-6,rtol=1e-5)
    sum(v.square().mean() for v in out.values()).backward()
    assert registers.grad.norm()>0 and trajectories.grad is None
    assert all(p.dtype==torch.float32 for p in model.parameters())
    assert sum(p.numel() for p in model.parameters())<1_000_000


def test_current_no_future_api_hindsight_frozen_vision_trainable_shared_decoder():
    model=PhysicalQueryDecoder()
    assert not any('future' in name or 'label' in name or 'teacher' in name for name in inspect.signature(model.forward).parameters)
    reg=torch.randn(2,16,256,requires_grad=True)
    future=torch.randn_like(reg,requires_grad=True)
    poses=torch.randn(2,3,requires_grad=True)
    out=model.forward_hindsight(reg,future,torch.randn(2,8,8,3),torch.randn(2,8),poses,1.5)
    sum(v.square().mean() for v in out.values()).backward()
    assert reg.grad is None and future.grad is None and poses.grad is None
    assert model.gap_head.weight.grad.norm()>0
    assert model.frame_pose_key[0].weight.grad.norm()>0


def test_measured_initial_speed_constant_velocity():
    traj=torch.zeros(2,8,8,3)
    traj[...,0]=torch.arange(1,9)*2.
    features=trajectory_features(traj,torch.full((2,),4.))
    assert features[...,5].abs().max()==0


def test_same_seed_bitwise_new_decoder_init():
    torch.manual_seed(13); a=PhysicalQueryDecoder()
    torch.manual_seed(13); b=PhysicalQueryDecoder()
    assert all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items())
