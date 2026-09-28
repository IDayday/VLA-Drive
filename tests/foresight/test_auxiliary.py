import pytest
import torch
from starVLA.model.modules.foresight.future_latent_head import FutureLatentHead
from starVLA.model.modules.foresight.interaction_latent_head import InteractionLatentHead
from starVLA.model.modules.foresight.losses import masked_regression,interaction_loss
from starVLA.model.modules.foresight.config import ForesightConfig


def test_both_readouts_backpropagate_shared_W_and_detach_targets():
    w=torch.randn(3,64,48,requires_grad=True)
    visual=FutureLatentHead(48,16,dim=32,heads=4)
    interaction=InteractionLatentHead(48,dim=32,heads=4)
    vp=visual(w,torch.tensor([0,1,2]),(4,6));vt=torch.randn_like(vp,requires_grad=True)
    loss,_=masked_regression(vp,vt,torch.ones(3,3,1,1,1,dtype=torch.bool))
    loss.backward();assert w.grad.abs().sum()>0 and vt.grad is None
    w.grad.zero_();ip=interaction(w);it=torch.randn_like(ip,requires_grad=True)
    loss,_=interaction_loss(ip,it,torch.ones(3,dtype=torch.bool));loss.backward()
    assert w.grad.abs().sum()>0 and it.grad is None


def test_masked_nans_empty_rank_and_valid_error():
    p=torch.randn(2,8,32,requires_grad=True);t=torch.randn_like(p);t[1]=float('nan')
    loss,count=interaction_loss(p,t,torch.tensor([True,False]));loss.backward()
    assert torch.isfinite(p.grad).all() and not p.grad[1].any() and count==256
    p.grad.zero_();t[:]=float('nan')
    loss,count=interaction_loss(p,t,torch.zeros(2,dtype=torch.bool));loss.backward()
    assert loss==0 and count==0 and torch.isfinite(p.grad).all() and not p.grad.any()
    with pytest.raises(ValueError):interaction_loss(p,t,torch.ones(2,dtype=torch.bool))


def test_global_element_normalization_differs_from_average_local_means():
    p=torch.tensor([1.,2.,3.],requires_grad=True);t=torch.zeros(3)
    loss,n=masked_regression(p,t,torch.tensor([True,False,False]));assert loss==1 and n==1
    p.grad=None
    loss,n=masked_regression(p,t,torch.tensor([True,False,False]),global_count=3);loss.backward()
    assert p.grad.tolist()==pytest.approx([2/3,0,0])


def test_visual_query_definition_and_config_contract():
    head=FutureLatentHead(32,16,dim=32,heads=4);w=torch.randn(1,64,32)
    assert head(w,torch.tensor([0]),(2,3)).shape==(1,3,16,2,3)
    with pytest.raises(ValueError):head(w,torch.tensor([3]),(2,3))
    with pytest.raises(ValueError):ForesightConfig(arm='R').validate()
    ForesightConfig(arm='R',num_queries=0).validate()
