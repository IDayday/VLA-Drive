"""Numerical contracts for the source-control and objective-only CISPO arm."""
import copy
import pytest
import torch

from starVLA.rl.simwam_recipe.trainer_adapter import SourceLoss
from starVLA.rl.qwen_world_research_algorithms.policy import (
    ResearchLoss, policy_gradient_objective, ratio_diagnostics,
)
from starVLA.rl.qwen_world_research_algorithms.batchstd import (
    NORMALIZATION_SCHEMA, centered_batch_advantages,
    finalize_centered_batch_advantages, validate_centered_batch_ranks,
    validate_centered_batch_boundary,
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


def batch_rewards():
    return torch.tensor([[1.]*8, [0.]*8, [.89,.90,.91,.92,.93,.94,.95,.96],
        [0.,1.,0.,1.,0.,1.,0.,1.], [.5]*8, [.2,.3,.4,.5,.6,.7,.8,.9],
        [.8,.82,.84,.86,.88,.90,.92,.94], [.01,.02,.03,.04,.05,.06,.07,.08]], dtype=torch.float32)


def test_centered_batchstd_uses_all64_once_and_reuses_saved_advantages_four_inner(monkeypatch):
    """One global collection, tied groups retained, four gradients on same A."""
    from starVLA.rl.qwen_world_research_algorithms import batchstd
    rewards=batch_rewards(); gathered_calls=[]
    monkeypatch.setattr(batchstd.dist,'is_available',lambda:True)
    monkeypatch.setattr(batchstd.dist,'is_initialized',lambda:True)
    monkeypatch.setattr(batchstd.dist,'get_world_size',lambda:8)
    def gather(out, value):
        gathered_calls.append(1)
        for rank in range(8):out[rank].copy_(rewards[rank:rank+1].double())
    monkeypatch.setattr(batchstd.dist,'all_gather',gather)
    r=rollout()
    r.update(rewards=rewards[3:4], chain=torch.zeros(8,11,8,3,dtype=torch.float64),
             old_logp=torch.zeros(8,3,dtype=torch.float64))
    del r['advantages'];del r['adv_mask']
    finalize_centered_batch_advantages(r)
    assert len(gathered_calls)==1
    assert r['advantage_statistics']['normalization_population_candidates']==64
    values=rewards.double();off=values-values[:,:1];c=off-off.mean(1,keepdim=True)
    expected=(c/(c.std(unbiased=False)+1e-8)).float()
    torch.testing.assert_close(r['advantages'],expected[3],rtol=0,atol=0)
    assert r['global_std']>r['normalization_std']
    config=cfg();config['research']['advantage_normalization']=NORMALIZATION_SCHEMA
    model=ToyModel()
    model.p=torch.nn.Parameter(torch.linspace(-.02,.02,8*3*8*3,dtype=torch.float64).reshape(8,3,8,3))
    adv_id=id(r['advantages']);adv_bits=r['advantages'].clone()
    for _ in range(4):
        reference=copy.deepcopy(model)
        got,gm=ResearchLoss(config)._policy_loss(model,r)
        want,wm=SourceLoss(cfg())._policy_loss(reference,r)
        got.backward();want.backward()
        torch.testing.assert_close(got,want,rtol=0,atol=0)
        torch.testing.assert_close(model.p.grad,reference.p.grad,rtol=0,atol=0)
        torch.testing.assert_close(gm['kl_loss'],wm['kl_loss'],rtol=0,atol=0)
        assert torch.isfinite(model.p.grad).all() and bool(model.p.grad.abs().sum()>0)
        assert id(r['advantages'])==adv_id and torch.equal(r['advantages'],adv_bits)
        assert len(gathered_calls)==1
        model.p.grad=None
    with pytest.raises(ValueError,match='must not be renormalized'):
        finalize_centered_batch_advantages(r)


def test_centered_batchstd_is_global_not_pergroup_and_rank_validator_rejects_rawstd():
    rewards=batch_rewards();values=rewards.double();centered=values-values.mean(1,keepdim=True)
    expected=centered/(centered.std(unbiased=False)+1e-8)
    summaries=[]
    for rank in range(8):
        result=centered_batch_advantages(rewards[rank:rank+1],all_rewards=rewards)
        torch.testing.assert_close(result['advantages'],expected[rank].float(),rtol=0,atol=0)
        # Adding arbitrary scene offsets cannot change this denominator.
        shifted=rewards.double()+torch.arange(8,dtype=torch.float64)[:,None]*3
        again=centered_batch_advantages(shifted[rank:rank+1],all_rewards=shifted)
        torch.testing.assert_close(result['advantages'],again['advantages'],atol=1e-7,rtol=1e-6)
        summaries.append(dict(phase='ROLLOUT_COMMITTED',updates=0,collections=1,
            scene_cursor=8,next_inner=0,run='new-arm',pending=True,behavior={'update':0},
            advantage_normalization_schema=NORMALIZATION_SCHEMA,
            rewards=values[rank:rank+1].tolist(), advantages=result['advantages'].tolist(),
            mask=result['adv_mask'].tolist(),global_std=float(result['global_std']),
            normalization_std=float(result['normalization_std'])))
    validate_centered_batch_ranks(summaries)
    # Constant all-1/all-0 groups have zero PG and remain in the 64-way scale.
    assert summaries[0]['mask']==[0.]*8 and summaries[1]['advantages']==[0.]*8
    bad=copy.deepcopy(summaries)
    bad[3]['advantages']=(centered[3]/(values.std(unbiased=False)+1e-8)).float().tolist()
    with pytest.raises(ValueError,match='not globally finalized'):
        validate_centered_batch_ranks(bad)


def test_centered_batchstd_all_tied_is_finite_zero_and_requires_explicit_protocol():
    rewards=torch.arange(8,dtype=torch.float64)[:,None].expand(8,8)
    got=centered_batch_advantages(rewards,all_rewards=rewards)
    assert torch.equal(got['advantages'],torch.zeros(64))
    assert torch.equal(got['adv_mask'],torch.zeros(64)) and got['normalization_std']==0
    c=cfg('cispo');c['research']['advantage_normalization']=NORMALIZATION_SCHEMA
    with pytest.raises(ValueError,match='PPO-only'):
        ResearchLoss(c)
    c=cfg();c['research']['advantage_normalization']=NORMALIZATION_SCHEMA
    with pytest.raises(ValueError,match='once-finalized'):
        ResearchLoss(c)._policy_loss(ToyModel(),rollout())


def test_centered_batchstd_preserves_native_boundary_contract_and_separate_identity():
    identity=dict(world=8,sha256='new-arm',schema='world_postcot_cleanx_scope_RL_v1',
        projected_condition_identity='fixed-condition',recipe=dict(max_steps=50000,
            action_shape=[8,3],model_family='QwenWorldPostCoT',prediction_type='x0',
            adaptation='ATTN_LORA_MEMORY',variant='R_CANONICAL',
            research=dict(advantage_normalization=NORMALIZATION_SCHEMA)))
    state=dict(updates=0,collections=1,next_inner=0,phase='ROLLOUT_COMMITTED',
               scene_cursor=8,raw_scene_exposures=8,new_candidates=64)
    rewards=batch_rewards();summaries=[]
    for rank in range(8):
        r=rollout();r.update(rewards=rewards[rank:rank+1],
            chain=torch.zeros(8,11,8,3),old_logp=torch.zeros(8,3),
            old_probability={'source_clamped_logprob':torch.zeros(8,3,8,3)},
            records=[],token=f'scene{rank}',log=f'log{rank}',
            behavior_version=dict(run='new-arm',update=0,adapter_sha256='fixed'),
            cond_bg=dict(schema='world_postcot_trainable_memory_observation_v1',
                         token=f'scene{rank}',identity='fixed-condition',repeats=8))
        material=centered_batch_advantages(rewards[rank:rank+1],all_rewards=rewards)
        for key in ['advantages','adv_mask','global_std','raw_group_std','normalization_std']:
            r[key]=material[key]
        r['advantage_statistics']={k:v for k,v in material.items() if not torch.is_tensor(v)}
        summaries.append(validate_centered_batch_boundary(state,r,identity))
    validate_centered_batch_ranks(summaries)
    old=copy.deepcopy(identity);old['recipe']['research']={}
    with pytest.raises(ValueError,match='separately registered'):
        validate_centered_batch_boundary(state,r,old)
    from starVLA.rl.simwam_recipe.recovery_state import validate_ranks
    with pytest.raises(ValueError,match='not globally finalized'):
        validate_ranks(summaries)
