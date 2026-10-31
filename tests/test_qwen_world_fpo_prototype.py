"""CPU FPO ratio/units/behavior cache tests, no registered runtime or GPU."""
import copy

import pytest
import torch

from test_qwen_world_rescue_auxiliary import make_bank, collect, TinyActor, recipe
from starVLA.rl.qwen_world_research_algorithms import rescue_auxiliary as rescue
from starVLA.rl.qwen_world_research_algorithms import fpo_prototype as fpo
from starVLA.rl.qwen_world_research_algorithms.safety_efficiency import finalize_safety_efficiency_advantages


def test_energy_units_and_per_MC_PPO_sign_clipping_old_detach():
    assert torch.equal(fpo.cfm_energy(torch.ones(8,8,3),torch.zeros(8,8,3)),torch.full((8,),8.))
    old=torch.ones(2,8,requires_grad=True)
    current=torch.ones(2,8,requires_grad=True)
    a=torch.tensor([1.,-1.,0.,0.,0.,0.,0.,0.]); mask=torch.ones(8)
    loss,metrics=fpo.per_mc_ppo_objective(current,old,a,mask,clip_range=.01)
    loss.backward()
    assert bool((current.grad[:,0]>0).all()) and bool((current.grad[:,1]<0).all())
    assert old.grad is None and metrics['fpo_proxy_ratio_mean']==1
    # For positive A, rho2 is clipped but rho1 remains differentiable. Clipping
    # exp(mean(logratio)) instead would block BOTH MC gradients for this action.
    new=torch.ones(2,8);new[1,0]-=.1;new.requires_grad_()
    loss,_=fpo.per_mc_ppo_objective(new,old,a.abs(),mask,clip_range=.01);loss.backward()
    assert new.grad[0,0]>0 and new.grad[1,0]==0
    rho=torch.exp(old.detach()-new.detach())
    advantage=a.abs().clamp(-5,5)[None].expand_as(rho)
    expected=torch.maximum(-advantage*rho,-advantage*rho.clamp(.99,1.01)).mean()
    torch.testing.assert_close(loss,expected,atol=0,rtol=0)


def test_behavior_current_cache_private_RNG_four_inner_fresh_Memory_and_no_IL_reference(tmp_path,monkeypatch):
    monkeypatch.setattr(torch.cuda,'synchronize',lambda:None)
    path=make_bank(tmp_path/'bank');bank=rescue.RescueTeacherBank(path,rescue.file_sha(path),['scene0','scene1'])
    actor,out=collect(bank);finalize_safety_efficiency_advantages(out)
    spec=fpo.PrototypeSpec(mc_samples=2,clip_range=.01,sigma_min=.2,sigma_max=.8)
    before_A=out['advantages'].clone();before_rng=torch.random.get_rng_state().clone()
    sde=torch.Generator().manual_seed(42);before_sde=sde.get_state().clone();actor.velocity_steps=[]
    fpo.cache_behavior_material(actor,out,spec,rank=0);p=out['fpo_prototype'];digest=p['material_sha256']
    assert torch.equal(before_rng,torch.random.get_rng_state()) and torch.equal(before_sde,sde.get_state())
    assert p['old_energy'].shape==(2,8) and not p['old_energy'].requires_grad
    assert all(not use_ref for use_ref,_ in actor.velocity_steps) and len(actor.velocity_steps)==2
    assert actor.scope_calls==1
    fpo.validate_material(out,spec,rank=0)
    for inner in range(4):
        loss,metrics=fpo.prototype_policy_loss(actor,out,spec)
        if inner==0: assert metrics['fpo_proxy_ratio_mean']==1
        loss.backward();assert torch.isfinite(actor.p.grad) and actor.p.grad.abs()>0
        with torch.no_grad():actor.p.sub_(actor.p.grad*1e-5)
        actor.p.grad=None
        assert fpo.material_digest(p)==digest and torch.equal(out['advantages'],before_A)
        assert all(not ref for ref,_ in actor.velocity_steps)
    assert actor.scope_calls==5 and len(actor.velocity_steps)==10
    fpo.validate_material(out,spec,rank=0)  # old behavior is fixed, current actor now changed
    with pytest.raises(ValueError,match='once'):fpo.cache_behavior_material(actor,out,spec,rank=0)


def test_all_A_zero_stays_zero_PG_no_fake_direction(tmp_path,monkeypatch):
    monkeypatch.setattr(torch.cuda,'synchronize',lambda:None)
    path=make_bank(tmp_path/'bank');bank=rescue.RescueTeacherBank(path,rescue.file_sha(path),['scene0','scene1'])
    actor,out=collect(bank,rank=2);finalize_safety_efficiency_advantages(out)
    assert not bool(out['advantages'].any()) and not bool(out['adv_mask'].any())
    spec=fpo.PrototypeSpec(mc_samples=2,clip_range=.01,sigma_min=.2,sigma_max=.8)
    fpo.cache_behavior_material(actor,out,spec,rank=2)
    loss,_=fpo.prototype_policy_loss(actor,out,spec);loss.backward()
    assert loss==0 and actor.p.grad==0


def test_detached_torch_roundtrip_rejects_source_MC_and_old_energy_corruption(tmp_path,monkeypatch):
    monkeypatch.setattr(torch.cuda,'synchronize',lambda:None)
    path=make_bank(tmp_path/'bank');bank=rescue.RescueTeacherBank(path,rescue.file_sha(path),['scene0','scene1'])
    actor,out=collect(bank);finalize_safety_efficiency_advantages(out)
    spec=fpo.PrototypeSpec(mc_samples=2,clip_range=.01,sigma_min=.2,sigma_max=.8)
    fpo.cache_behavior_material(actor,out,spec,rank=0)
    path=tmp_path/'native_material.pt';torch.save(out,path);saved=torch.load(path,weights_only=False)
    fpo.validate_material(saved,spec,rank=0)
    for key in ('clean','sigma','noise','old_energy','behavior_velocity'):
        bad=copy.deepcopy(saved);bad['fpo_prototype'][key].reshape(-1)[0]+=.1
        with pytest.raises(ValueError,match='digest'):fpo.validate_material(bad,spec,rank=0)
        bad['fpo_prototype']['material_sha256']=fpo.material_digest(bad['fpo_prototype'])
        with pytest.raises(ValueError,match='material changed|audit differs'):fpo.validate_material(bad,spec,rank=0)
    bad=copy.deepcopy(saved);bad['advantages'][0]+=.1
    with pytest.raises(ValueError,match='identity'):fpo.validate_material(bad,spec,rank=0)
    with pytest.raises(ValueError,match='identity'):fpo.validate_material(saved,spec,rank=1)
    # The adapter must never silently extrapolate to sigma0 / unspecified clip.
    with pytest.raises(ValueError):fpo.PrototypeSpec(mc_samples=2,clip_range=.01,sigma_min=0,sigma_max=1)
