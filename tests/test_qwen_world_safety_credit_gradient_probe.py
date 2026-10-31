"""Finite CPU checks for shared-branch attribution and source-loss replay."""
import copy

import pytest
import torch

from starVLA.rl.qwen_world_research_algorithms.safety_credit_gradient_probe import (
    CreditReplayLoss, candidate_update_geometry, shared_branch_safe_objective,
)
from starVLA.rl.qwen_world_research_algorithms.gradient_probe import measure_components
from starVLA.rl.qwen_world_research_algorithms.policy import ResearchLoss, policy_gradient_objective
from starVLA.rl.qwen_world_research_algorithms.safety_efficiency import safety_efficiency_advantages
from starVLA.rl.simwam_recipe.source_advantages import advantages as source_advantages


def test_safe_component_uses_total_cap_and_total_clip_branch():
    ratio=torch.tensor([[1.02,.98],[.98,1.02],[1.001,.999],[1.,1.]],dtype=torch.float64)
    logp=ratio.log().requires_grad_()
    total=torch.tensor([[20.,-20.],[1.,-1.],[20.,-20.],[0.,0.]],dtype=torch.float64)
    safe=torch.tensor([[30.,30.],[-2.,2.],[30.,30.],[1.,-1.]],dtype=torch.float64)
    mask=torch.tensor([1.,1.,1.,0.],dtype=torch.float64)
    safe_loss,terms=shared_branch_safe_objective(logp,torch.zeros_like(logp),total,safe,mask,
                                               clip_range=.005,cap=5.)
    pg,_=policy_gradient_objective(logp,torch.zeros_like(logp),total.clamp(-5,5),mask)
    ep=pg-safe_loss
    g=torch.autograd.grad(safe_loss,logp,retain_graph=True)[0]
    assert torch.equal(g[0],torch.zeros(2,dtype=g.dtype))  # total clipped despite opposite component signs
    torch.testing.assert_close(terms['safe_effective'][2],torch.full((2,),7.5,dtype=g.dtype))
    torch.testing.assert_close(g[1],torch.tensor([2*.98,-2*1.02],dtype=g.dtype)/6)
    assert torch.equal(g[3],torch.zeros(2,dtype=g.dtype))
    together=torch.autograd.grad(safe_loss+ep,logp,retain_graph=True)[0]
    original=torch.autograd.grad(pg,logp)[0]
    torch.testing.assert_close(together,original,rtol=0,atol=1e-15)


class Toy(torch.nn.Module):
    grpo_kl_beta=50.
    grpo_denoising_discount=1.
    grpo_bc_coeff=0.
    grpo_kl_type='x'

    def __init__(self):
        super().__init__()
        self.action=torch.nn.Parameter(torch.linspace(-.02,.02,8*3*8*3,dtype=torch.float64).reshape(8,3,8,3))
        self.calls=[]

    def action_chain_logprobs(self,*args,use_ref=False,return_terms=False,**kwargs):
        self.calls.append(use_ref)
        mean=self.action*0 if use_ref else self.action
        logp=mean.mean((-1,-2))
        terms=dict(mean=mean,vel=mean*2,std=torch.tensor([.05,.04,.03],dtype=mean.dtype))
        return (logp,terms) if return_terms else logp

    def kl_weight_matrix(self,*,horizon,action_dim,device,dtype):
        return torch.tensor([.2,1,1],device=device,dtype=dtype)[None,:].expand(horizon,action_dim)


def config():
    return dict(grpo=dict(train=dict(ppo_clip_range=.005,adv_clip_max=5.,drop_zero_adv=True,
        adv_eps=1e-8,adv_clip_lower_quantile=0.,adv_clip_upper_quantile=1.,adv_global_std=True,
        use_bc_loss=False)),research=dict(policy_objective='ppo'))


def material():
    components=torch.ones(8,8,4,dtype=torch.float64)
    components[:,:,3]=torch.linspace(.1,.8,8)
    components[0,4:,2]=0  # mixed; a feasible EP deficit can oppose safety credit
    components[1,:,0]=0  # all-fail equal costs
    components[2,:,3]=1  # saturated tie retained in global populations
    components[3,:,3]=.5 # canonical reward constant, must remain masked in PPO
    rewards=(components[:,:,0]*components[:,:,1]*(5*components[:,:,3]+5*components[:,:,2]+2)/12).float()
    s=safety_efficiency_advantages(rewards[:1],components[:1],
        all_material=torch.cat((rewards.double().unsqueeze(-1),components),-1))
    p=source_advantages(rewards[:1],all_rewards=rewards)
    return dict(chain=torch.zeros(8,11,8,3),timesteps=torch.arange(10),deltas=torch.full((10,),-.1),
        stoch_idx=torch.tensor([7,8,9]),cond_bg={},old_logp=torch.zeros(8,3,dtype=torch.float64),
        advantages=s['advantages'],adv_mask=s['adv_mask'],normalization_std=s['normalization_std'],
        structured_advantage={k:s[k] for k in ('A_safe','A_EP','A_raw','feasible','cost')},
        credit_replay_ppo=p),rewards


def test_replay_uses_one_source_forward_and_mean_anchor_without_changing_native_loss():
    actor=Toy();control=copy.deepcopy(actor);r,rewards=material()
    observer=CreditReplayLoss(config())
    actual,_=observer._policy_loss(actor,r)
    original,_=ResearchLoss(config())._policy_loss(control,r)
    assert actor.calls == [False,True]  # no duplicate model forward for alternate credit
    torch.testing.assert_close(actual,original,rtol=0,atol=0)
    ga=torch.autograd.grad(actual,actor.action,retain_graph=True)[0]
    gc=torch.autograd.grad(original,control.action)[0]
    torch.testing.assert_close(ga,gc,rtol=0,atol=0)
    report=candidate_update_geometry(measure_components(actor,observer.parts))
    assert all(p.grad is None for p in actor.parameters())
    scope=report['scopes']['all']
    assert 'summed_gradient_norm' not in scope
    assert scope['candidate_update_norms']['safety_plus_beta50_anchor']==pytest.approx(float(ga.float().norm()),rel=1e-6)
    alternate=dict(r,advantages=r['credit_replay_ppo']['advantages'],adv_mask=r['credit_replay_ppo']['adv_mask'])
    ppo_total,_=ResearchLoss(config())._policy_loss(actor,alternate)
    gp=torch.autograd.grad(ppo_total,actor.action)[0]
    assert scope['candidate_update_norms']['ppo_plus_beta50_anchor']==pytest.approx(float(gp.float().norm()),rel=1e-6)
    norm=float(gp.norm()*ga.norm())
    assert scope['candidate_update_cosine']==pytest.approx(float((gp*ga).sum())/norm,abs=1e-6)
    # The alternative credit is source raw-reward global std, not centered-A or group std.
    assert float(r['credit_replay_ppo']['global_std'])==float(rewards.double().std(unbiased=False))
    tied=source_advantages(rewards[3:4],all_rewards=rewards)
    assert not bool(tied['adv_mask'].any())
