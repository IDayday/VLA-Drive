import pytest
import torch
from starVLA.rl.qwen_world_postcot.loss_designs import LossDesign
from starVLA.model.modules.qwen_world_postcot.flow import clean_x_l1


def test_control_and_time_ones_match_old_value_and_gradient():
    x=torch.randn(2,8,3,requires_grad=True);y=torch.randn_like(x);mask=torch.rand(2,8)>.2
    reference=clean_x_l1(x,y,mask);g=torch.autograd.grad(reference,x,retain_graph=True)[0]
    for kind in ['CONTROL','TIME_RESIDUAL']:
        result,_=LossDesign(kind)(x,y,mask)
        assert torch.equal(result,reference)
        assert torch.equal(torch.autograd.grad(result,x,retain_graph=True)[0],g)


def test_delta_matches_gt_velocity_not_zero_speed():
    target=torch.zeros(1,8,3);target[0,:,0]=torch.arange(1,9)*.01
    x=target.clone().requires_grad_(True)
    loss,_=LossDesign('DELTA',delta_coefficient=.1)(x,target)
    assert loss==0 and torch.equal(torch.autograd.grad(loss,x)[0],torch.zeros_like(x))
    stopped=torch.zeros_like(target)
    value,_=LossDesign('DELTA',delta_coefficient=.1)(stopped,target)
    assert value>clean_x_l1(stopped,target)


def test_delta_does_not_bridge_invalid_horizon():
    x=torch.zeros(1,8,3);y=torch.zeros_like(x);x[:,3,0]=100
    valid=torch.ones(1,8,dtype=torch.bool);valid[:,3]=False
    loss,_=LossDesign('DELTA',delta_coefficient=.1)(x,y,valid)
    assert loss==0


@pytest.mark.parametrize('kind,kwargs',[('TIME_RESIDUAL',dict(horizon_weights=[2]*4+[.5]*4)),
    ('DELTA',dict(delta_coefficient=.04)),('HUBER',{})])
def test_global_reduction_and_all_invalid(kind,kwargs):
    objective=LossDesign(kind,**kwargs);x=torch.randn(5,8,3,requires_grad=True);y=torch.randn_like(x);mask=torch.rand(5,8)>.2
    loss,all_stats=objective(x,y,mask)
    parts=[objective(x[:2],y[:2],mask[:2])[1],objective(x[2:],y[2:],mask[2:])[1]]
    assert torch.allclose(loss,sum(p['numerator'] for p in parts)/all_stats['denominator'])
    zero,_=objective(x,y,torch.zeros_like(mask))
    assert zero==0 and torch.equal(torch.autograd.grad(zero,x)[0],torch.zeros_like(x))


def test_huber_large_error_gradient_matches_l1_and_small_error_is_continuous():
    x=torch.full((1,8,3),2.,requires_grad=True);y=torch.zeros_like(x)
    h,_=LossDesign('HUBER')(x,y)
    assert torch.equal(torch.autograd.grad(h,x)[0],torch.autograd.grad(clean_x_l1(x,y),x)[0])
    x=torch.zeros(1,8,3,requires_grad=True);h,_=LossDesign('HUBER')(x,y)
    assert torch.equal(torch.autograd.grad(h,x)[0],torch.zeros_like(x))
