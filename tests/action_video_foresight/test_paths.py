from types import SimpleNamespace
import torch
import pytest
from torch import nn
from starVLA.model.modules.foresight.future_spatiotemporal_head import FutureSpatiotemporalHead, normalized_clip_loss
from starVLA.model.framework.DDPForesight import DDPForesight
from starVLA.model.modules.foresight.world_intervention import permute_world_at_layer
from tools.full_foresight.evaluate_auxiliary import FixedPositionMoments, frozen_mae_decode


def test_gt_action_changes_auxiliary_without_grad_to_gt_or_execution_head():
    torch.manual_seed(7)
    head=FutureSpatiotemporalHead(16,12,[(.5,1.),(1.5,2.)],dim=32,action_condition='gt_ego')
    world=torch.randn(2,6,16,requires_grad=True)
    gt=torch.randn(2,8,4,requires_grad=True)
    original=world.detach().clone()
    output=head(world,(2,3),gt_action=gt)
    other=head(world,(2,3),gt_action=gt+1.)
    assert not torch.equal(output,other)
    assert torch.equal(world.detach(),original)
    output.square().mean().backward()
    assert gt.grad is None
    assert world.grad.abs().sum()>0
    assert head.action_encoder.point[0].weight.grad.abs().sum()>0
    no_action=FutureSpatiotemporalHead(16,12,[(.5,1.)],dim=32)
    with pytest.raises(ValueError):no_action(world,(2,3),gt_action=gt)
    with pytest.raises(ValueError):head(world,(2,3))
    only_action=FutureSpatiotemporalHead(16,12,[(.5,1.)],dim=32,action_condition='gt_ego',use_world=False)
    assert only_action(None,(2,3),gt_action=gt).shape==(2,3,1,2,3,12)
    with pytest.raises(ValueError,match='no W input'):
        only_action(world,(2,3),gt_action=gt)


def test_direct_world_condition_preserves_raw_hidden_and_gradient():
    model=DDPForesight.__new__(DDPForesight);nn.Module.__init__(model)
    a=torch.randn(2,8,16,requires_grad=True);w=torch.randn(2,144,16,requires_grad=True)
    model.foresight_config=SimpleNamespace(planner_condition_mode='action_plus_W')
    condition=model.build_planner_condition({'action_queries':a,'W':w})
    assert condition.shape==(2,152,16)
    assert torch.equal(condition[:,:8],a) and torch.equal(condition[:,8:],w)
    condition.square().sum().backward();assert w.grad.abs().sum()>0
    model.foresight_config=SimpleNamespace(planner_condition_mode='action_only')
    assert model.build_planner_condition({'action_queries':a,'W':w}) is a


def test_clip_mask_before_normalization_and_valid_scene_view_mean():
    torch.manual_seed(12)
    prediction=torch.randn(3,3,2,2,2,5,requires_grad=True)
    target=torch.randn_like(prediction)
    valid=torch.ones(prediction.shape[:-1],dtype=torch.bool)
    valid[1,1:]=False;valid[2]=False
    clean,_=normalized_clip_loss(prediction,target,valid)
    dirty=target.clone();dirty[~valid]=float('nan')
    result,n=normalized_clip_loss(prediction,dirty,valid)
    assert n==2 and torch.equal(clean,result)
    result.backward();assert torch.isfinite(prediction.grad).all()
    with pytest.raises(ValueError):normalized_clip_loss(prediction,dirty,torch.ones_like(valid))
    empty,_=normalized_clip_loss(prediction,dirty,torch.zeros_like(valid))
    assert empty.item()==0
    # Different local target populations are summed with global count, not means.
    reference,_=normalized_clip_loss(prediction,target,valid)
    parts=[normalized_clip_loss(prediction[i:i+1],target[i:i+1],valid[i:i+1],global_count=2)[0] for i in range(3)]
    assert torch.allclose(sum(parts),reference)


def test_fixed_spatial_template_has_zero_cross_scene_variance():
    x=torch.arange(24.).reshape(3,2,4);valid=torch.ones(3,1,4,dtype=torch.bool)
    moments=FixedPositionMoments()
    for _ in range(5):moments.add(x,valid)
    assert x.var()>0 and moments.result()==0
    moments.add(x+1,valid);assert moments.result()>0


def test_teacher_functional_decode_does_not_renormalize_or_change_anchor():
    teacher=SimpleNamespace(reconstruct=nn.Sequential(nn.LayerNorm(4),nn.Linear(4,2)),xy_scale=20.)
    z=torch.randn(2,8,4);anchor=torch.randn(2,2)
    assert torch.equal(frozen_mae_decode(teacher,z,anchor),teacher.reconstruct(z)*20+anchor[:,None])


@pytest.mark.parametrize('as_tuple',[False,True])
def test_midlayer_world_swap_only_changes_world_positions(as_tuple):
    class Layer(nn.Module):
        def forward(self,x):return (x.clone(),None) if as_tuple else x.clone()
    layer=Layer();model=nn.Module();model.eval()
    model.qwen_vl_interface=SimpleNamespace(model=SimpleNamespace(model=SimpleNamespace(language_model=SimpleNamespace(layers=[layer]))))
    model._diagnostic_world_positions=torch.tensor([[1,2],[2,3]])
    x=torch.arange(40.).reshape(2,5,4)
    with permute_world_at_layer(model,0,[1,0]):y=layer(x)
    if as_tuple:y=y[0]
    assert torch.equal(y[0,1:3],x[1,2:4])
    assert torch.equal(y[0,[0,3,4]],x[0,[0,3,4]])
    original=layer(x);assert torch.equal(original[0] if as_tuple else original,x)
