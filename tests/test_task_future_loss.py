import torch
from navsim.agents.EpisodeDrive.layers.world_model.physical_query_decoder import PhysicalQueryDecoder
from navsim.agents.EpisodeDrive.layers.world_model.task_future_loss import (
    global_task_mean, task_future_lite_loss, sample_training_candidates, distillation_element_losses,
)


def test_invalid_not_diluting_valid_fields_and_finite_empty():
    values=torch.ones(2,8,8,3,requires_grad=True)
    mask=torch.zeros_like(values,dtype=torch.bool);mask[...,1]=True
    loss,_,counts=global_task_mean(values,mask)
    assert loss==1 and counts.tolist()==[0,128,0]
    empty,_,_=global_task_mean(values,torch.zeros_like(mask))
    assert empty==0 and torch.isfinite(empty)
    empty.backward(); assert values.grad.abs().sum()==0


def test_lite_gradient_routes_kd_stopgrad_future_bins():
    torch.set_num_threads(2)
    torch.manual_seed(33)
    model=PhysicalQueryDecoder()
    reg=torch.randn(2,16,256,requires_grad=True)
    current=torch.randn_like(reg,requires_grad=True)
    future=torch.randn(2,3,16,256,requires_grad=True)
    trajectories=torch.randn(2,8,8,3,requires_grad=True)
    valid=torch.ones(2,8,8,3,dtype=torch.bool)
    future_mask=torch.tensor([[True,False,True],[False,False,False]])
    losses=task_future_lite_loss(model,reg,trajectories,torch.randn(2,8),current,future,
        future_mask,torch.randn(2,3,3),torch.randn(2,8,8,3),torch.zeros(2,8,8,dtype=torch.long),valid)
    losses['wm_loss'].backward()
    assert reg.grad.norm()>0 and trajectories.grad is None
    assert current.grad is None and future.grad is None
    assert losses['physical_future_gap_count']==16  # 8 candidates * only 2 observed horizons
    assert losses['legacy_future_register_loss']==0
    assert model.frame_pose_key[0].weight.grad.norm()>0
    a={k:torch.randn_like(v,requires_grad=True) for k,v in model(reg.detach(),trajectories.detach(),torch.randn(2,8)).items()}
    b={k:torch.randn_like(v,requires_grad=True) for k,v in a.items()}
    distillation_element_losses(a,b).sum().backward()
    assert all(v.grad is None for v in b.values())


def test_uniform_without_replacement_and_gt_no_flag():
    gt=torch.randn(3,8,3);proposals=torch.randn(3,64,8,3,requires_grad=True)
    chosen,indices=sample_training_candidates(gt,proposals)
    assert chosen.shape==(3,8,8,3) and not chosen.requires_grad
    torch.testing.assert_close(chosen[:,0],gt)
    assert all(len(row.unique())==7 for row in indices)
