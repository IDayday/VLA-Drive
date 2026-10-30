"""Numerical contracts for the source-control and objective-only CISPO arm."""
import copy
import pytest
import torch

from starVLA.rl.simwam_recipe.trainer_adapter import SourceLoss
from starVLA.rl.qwen_world_research_algorithms.policy import (
    ResearchLoss, policy_gradient_objective, ratio_diagnostics,
)


def cfg(algorithm='ppo'):
    return dict(grpo=dict(train=dict(ppo_clip_range=.005, adv_clip_max=5., drop_zero_adv=True,
        adv_eps=1e-8, adv_clip_lower_quantile=0., adv_clip_upper_quantile=1.,
        adv_global_std=True, use_bc_loss=False)),
        research=dict(policy_objective=algorithm, cispo_upper=1.01))


class ToyModel(torch.nn.Module):
    grpo_kl_beta = 50.
    grpo_denoising_discount = 1.
    grpo_use_bc_loss = False
    grpo_bc_coeff = .1
    grpo_kl_type = 'x'

    def __init__(self):
        super().__init__()
        self.p = torch.nn.Parameter(torch.linspace(-.02,.02,6*3*8*3,dtype=torch.float64).reshape(6,3,8,3))

    def action_chain_logprobs(self, *args, use_ref=False, return_terms=False, **kwargs):
        value = self.p*0 if use_ref else self.p
        logp = value.mean((-1,-2))
        terms = dict(mean=value,vel=value*2,std=torch.tensor([.05,.04,.03],dtype=torch.float64))
        return (logp,terms) if return_terms else logp

    def kl_weight_matrix(self, *, horizon, action_dim, device, dtype):
        return torch.tensor([.2,1.,1.],device=device,dtype=dtype)[None,:].expand(horizon,action_dim)


def rollout(mask=None):
    return dict(chain=torch.zeros(6,11,8,3,dtype=torch.float64), timesteps=torch.arange(10),
                deltas=torch.ones(10)*-.1,stoch_idx=torch.tensor([7,8,9]),cond_bg={},
                old_logp=torch.zeros(6,3,dtype=torch.float64),
                advantages=torch.tensor([1.,-1.,.2,-.3,0.,.1],dtype=torch.float64),
                adv_mask=torch.ones(6,dtype=torch.float64) if mask is None else mask)


@pytest.mark.parametrize('discount',[1.,.6])
@pytest.mark.parametrize('kl_type',['x','x_kl','v'])
def test_ppo_control_matches_source_value_and_parameter_gradient(discount,kl_type):
    a=ToyModel();a.grpo_denoising_discount=discount;a.grpo_kl_type=kl_type
    b=copy.deepcopy(a);r=rollout(torch.tensor([1,1,1,1,0,0],dtype=torch.float64))
    original,om=SourceLoss(cfg())._policy_loss(a,r)
    control,cm=ResearchLoss(cfg())._policy_loss(b,r)
    original.backward();control.backward()
    torch.testing.assert_close(control,original,rtol=0,atol=0)
    torch.testing.assert_close(b.p.grad,a.p.grad,rtol=0,atol=0)
    for name in ('policy_loss','kl_loss','kl_x','kl_v','kl_x_norm','ratio_mean','clipfrac'):
        torch.testing.assert_close(cm[name],om[name],rtol=0,atol=0)


def test_cispo_detaches_clipped_importance_weights():
    logp=torch.tensor([[.2,-.2],[.001,-.001]],dtype=torch.float64,requires_grad=True)
    advantages=torch.tensor([[1.,-1.],[2.,-2.]],dtype=torch.float64)
    mask=torch.ones(2,dtype=torch.float64)
    loss,d=policy_gradient_objective(logp,torch.zeros_like(logp),advantages,mask,algorithm='cispo',cispo_upper=1.01)
    loss.backward()
    expected=-logp.detach().exp().clamp(max=1.01)*advantages/4
    torch.testing.assert_close(logp.grad,expected,rtol=0,atol=0)
    assert not d['derivative_weight'].requires_grad


def test_positive_and_negative_clipped_ppo_samples_retain_cispo_gradients():
    p=torch.tensor([[.1,-.1]],dtype=torch.float64,requires_grad=True)
    adv=torch.tensor([[1.,-1.]],dtype=torch.float64);mask=torch.ones(1,dtype=torch.float64)
    source,_=policy_gradient_objective(p,torch.zeros_like(p),adv,mask)
    g=torch.autograd.grad(source,p)[0]
    assert torch.equal(g,torch.zeros_like(g))
    cispo,d=policy_gradient_objective(p,torch.zeros_like(p),adv,mask,algorithm='cispo')
    g2=torch.autograd.grad(cispo,p)[0]
    assert bool((g2.abs()>0).all())
    diag=ratio_diagnostics(d,adv,mask,[7,8],clip_range=.005,cispo_upper=1.005)
    assert diag['positive_ppo_blocked_fraction']==1
    assert diag['negative_ppo_blocked_fraction']==1
    assert diag['positive_active_fraction']==1 and diag['negative_active_fraction']==1


def test_tied_groups_leave_reference_anchor_value_and_gradient_unchanged():
    a=ToyModel();b=copy.deepcopy(a);r=rollout(torch.zeros(6,dtype=torch.float64))
    source,sm=SourceLoss(cfg())._policy_loss(a,r)
    cispo,cm=ResearchLoss(cfg('cispo'))._policy_loss(b,r)
    source.backward();cispo.backward()
    assert cm['policy_loss']==0 and cm['kl_loss']>0
    torch.testing.assert_close(cispo,source,rtol=0,atol=0)
    torch.testing.assert_close(a.p.grad,b.p.grad,rtol=0,atol=0)
    assert bool(torch.isfinite(b.p.grad).all())


def test_masked_group_cannot_change_either_pg_gradient_or_diagnostics():
    p=torch.tensor([[.1,-.1],[.3,-.3]],dtype=torch.float64,requires_grad=True)
    adv=torch.tensor([[1.,-1.],[2.,-2.]],dtype=torch.float64)
    mask=torch.tensor([1.,0.],dtype=torch.float64)
    loss,d=policy_gradient_objective(p,torch.zeros_like(p),adv,mask,algorithm='cispo')
    loss.backward()
    assert torch.equal(p.grad[1],torch.zeros_like(p.grad[1]))
    diag=ratio_diagnostics(d,adv,mask,[7,8],clip_range=.005,cispo_upper=1.005)
    assert diag['positive_ppo_blocked_fraction']==1 and diag['negative_ppo_blocked_fraction']==1


def test_nonfinite_ratio_is_rejected():
    p=torch.full((2,3),float('inf'))
    with pytest.raises(FloatingPointError):
        policy_gradient_objective(p,torch.zeros_like(p),torch.ones_like(p),torch.ones(2))


def test_cispo_changes_only_policy_component_for_nonzero_reference_drift():
    a=ToyModel();b=copy.deepcopy(a);r=rollout()
    source,sm=SourceLoss(cfg())._policy_loss(a,r)
    cispo,cm=ResearchLoss(cfg('cispo'))._policy_loss(b,r)
    for key in ('kl_loss','kl_div','kl_x','kl_x_norm','kl_v'):
        torch.testing.assert_close(cm[key],sm[key],rtol=0,atol=0)
    torch.testing.assert_close(cispo-cm['policy_loss'],source-sm['policy_loss'],rtol=1e-12,atol=1e-12)
