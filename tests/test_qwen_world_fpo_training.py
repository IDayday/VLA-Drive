"""Targeted actual-source FPO collection/four-inner/native-boundary checks, CPU."""
import copy
from types import SimpleNamespace

import pytest
import torch

from test_qwen_world_lastdet_contract import recipe as source_recipe, descriptor
from test_qwen_world_rescue_auxiliary import TinyActor, Feedback
from starVLA.rl.qwen_sourceflow.collection import collect
from starVLA.rl.qwen_world_research_algorithms import fpo_training as fpo
from starVLA.rl.qwen_world_research_algorithms.fpo_entry import FPORuntimeMixin
from starVLA.rl.qwen_world_research_algorithms.policy import ResearchLoss
from starVLA.rl.qwen_world_research_algorithms import safety_efficiency as se
from starVLA.rl.simwam_recipe.recovery_state import rollout_digest


def recipe():
    cfg=source_recipe(False)
    cfg['research'].update(arm=fpo.ARM,structured_advantage=se.STRUCTURED_SCHEMA,fpo=copy.deepcopy(fpo.CONFIG))
    return cfg


def collected(rank=0):
    cfg=recipe();actor=TinyActor(cfg)
    out=collect(actor,descriptor(rank),f'scene{rank}',f'log{rank}',Feedback(rank),
        torch.Generator().manual_seed(42+rank),variant='R_CANONICAL',
        version=dict(run='run',update=0,adapter_sha256='same'))
    return actor,out


def identity():
    return dict(world=8,sha256='run',schema='world_postcot_cleanx_scope_RL_v1',
        projected_condition_identity='condition',recipe=recipe(),fpo_protocol=copy.deepcopy(fpo.PROTOCOL))


def state(inner=0):
    return dict(updates=inner,collections=1,next_inner=inner,
        phase='ROLLOUT_COMMITTED' if inner==0 else 'INNER_COMMITTED',
        scene_cursor=8,raw_scene_exposures=8,new_candidates=64)


def finalized_one(monkeypatch,rank=0):
    monkeypatch.setattr(torch.cuda,'synchronize',lambda:None)
    actor,out=collected(rank);se.finalize_safety_efficiency_advantages(out)
    fpo.attach_material(actor,out,rank=rank)
    return actor,out


def test_recipe_fixed_fresh_W3_and_perMC_gradient_math():
    c=recipe();assert fpo.validate_recipe(c) is c
    for key,value in [('warmstart',{}),('native_fork',{}),('sampler_ablation','w2'),('fpo',dict(fpo.CONFIG,mc_samples=4))]:
        bad=copy.deepcopy(c);bad['research'][key]=value
        with pytest.raises(ValueError):fpo.validate_recipe(bad)
    for where,key,value in [('sample','train_sde_steps',2),('sample','noise_level',.03),('train','kl_beta',40)]:
        bad=copy.deepcopy(c);bad['grpo'][where][key]=value
        with pytest.raises(ValueError):fpo.validate_recipe(bad)
    old=torch.ones(2,8,requires_grad=True);new=old.detach().clone().requires_grad_()
    a=torch.tensor([1.,-1.,0,0,0,0,0,0]);mask=torch.ones(8)
    loss,_=fpo.per_mc_ppo_objective(new,old,a,mask,clip_range=.01);loss.backward()
    assert new.grad[:,0].gt(0).all() and new.grad[:,1].lt(0).all() and old.grad is None
    assert torch.equal(fpo.cfm_energy(torch.ones(8,8,3),torch.zeros(8,8,3)),torch.full((8,),8.))


def test_actual_W3_source_four_inner_fresh_Memory_source_anchor_parity_and_audit(monkeypatch):
    actor,out=finalized_one(monkeypatch)
    assert out['stoch_idx'].tolist()==[7,8,9] and out['old_logp'].shape==(8,3)
    before=rollout_digest(out);material=fpo.material_digest(out['fpo']);rng=torch.random.get_rng_state().clone()
    actor._loss=fpo.FPOTrainingLoss(recipe());actor.velocity_steps=[]
    for inner in range(4):
        # Pin physical anchor value and gradient to the historical ResearchLoss,
        # with all advantages zero so it performs exactly the anchor objective.
        anchor_rollout=copy.deepcopy(out);anchor_rollout['advantages'].zero_();anchor_rollout['adv_mask'].zero_()
        anchor,_=ResearchLoss(recipe())._policy_loss(actor,anchor_rollout)
        want_grad=torch.autograd.grad(anchor,actor.p)[0]
        actual,extra=fpo._physical_anchor(actor,out,50.)
        torch.testing.assert_close(actual,anchor,atol=0,rtol=0)
        torch.testing.assert_close(torch.autograd.grad(actual,actor.p)[0],want_grad,atol=0,rtol=0)
        actor.p.grad=None
        loss,metrics=actor._loss._policy_loss(actor,out)
        assert metrics['fpo_fresh_checked']==float(inner==0)
        if inner==0:assert metrics['fpo_proxy_ratio_max_abs_from_one']<1e-6
        current=actor._probability_audit['current']['source_clamped_logprob'].mean((-1,-2))
        assert current.shape==(8,3)  # ordinary trainer fresh source-ratio audit stays available
        loss.backward();assert torch.isfinite(actor.p.grad) and actor.p.grad.abs()>0
        with torch.no_grad():actor.p.sub_(1e-5*actor.p.grad)
        actor.p.grad=None
        assert rollout_digest(out)==before and fpo.material_digest(out['fpo'])==material
    assert torch.equal(rng,torch.random.get_rng_state())
    fpo.validate_material(out,rank=0)  # behaviorold remains fixed after four realupdates
    assert {round(t) for ref,t in actor.velocity_steps if ref}=={round(float(out['timesteps'][i])) for i in (7,8,9)}


def test_exact_zero_A_keeps_physical_anchor_and_fresh_mismatch_rejected(monkeypatch):
    actor,out=finalized_one(monkeypatch,rank=2)
    assert not out['advantages'].any() and not out['adv_mask'].any()
    actor._loss=fpo.FPOTrainingLoss(recipe());loss,metrics=actor._loss._policy_loss(actor,out)
    assert metrics['policy_loss']==0 and metrics['fpo_all_A_zero']==1
    loss.backward();assert actor.p.grad.abs()>0  # no invented PG; existing anchorstill updates
    actor.p.grad=None
    object.__setattr__(actor,'_fpo_expect_fresh_material',out['fpo']['material_sha256'])
    with torch.no_grad():actor.p.add_(1.)
    with pytest.raises(FloatingPointError,match='Fresh behavior CFM ratio'):actor._loss._policy_loss(actor,out)


def test_eight_rank_native_roundtrip_allinner_restore_markers_and_pollution_rejection(monkeypatch,tmp_path):
    monkeypatch.setattr(torch.cuda,'synchronize',lambda:None)
    pairs=[collected(rank) for rank in range(8)];rollouts=[p[1] for p in pairs]
    rewards=torch.cat([r['rewards'].double() for r in rollouts]);com=torch.cat([se.canonical_components(r['records'])['components'] for r in rollouts])
    full=torch.cat((rewards[:,:,None],com),-1)
    monkeypatch.setattr(se.dist,'is_available',lambda:True);monkeypatch.setattr(se.dist,'is_initialized',lambda:True)
    monkeypatch.setattr(se.dist,'get_world_size',lambda:8)
    def gather(out,local):
        for rank in range(8):out[rank].copy_(full[rank:rank+1])
    monkeypatch.setattr(se.dist,'all_gather',gather)
    summaries=[]
    for rank,(actor,r) in enumerate(pairs):
        se.finalize_safety_efficiency_advantages(r);rng=torch.random.get_rng_state().clone()
        fpo.attach_material(actor,r,rank=rank);assert torch.equal(rng,torch.random.get_rng_state())
        path=tmp_path/f'rank{rank}.pt';torch.save(r,path);saved=torch.load(path,weights_only=False)
        for inner in range(4):
            sm=fpo.validate_boundary(state(inner),saved,identity(),
                optimizer=dict(state={0:dict(step=inner)},param_groups=[dict(params=[0])]),
                scheduler=dict(last_epoch=inner),rank=rank)
            if inner==0:summaries.append(sm)
        for k in ['clean','sigma','noise','behavior_velocity','old_energy']:
            bad=copy.deepcopy(saved);bad['fpo'][k].reshape(-1)[0]+=.1
            with pytest.raises(ValueError,match='digest'):fpo.validate_material(bad,rank=rank)
            bad['fpo']['material_sha256']=fpo.material_digest(bad['fpo'])
            with pytest.raises(ValueError,match='draw changed|audit differs'):fpo.validate_material(bad,rank=rank)
        # A restore atinner0 must verifyfresh; inner1..3 explicitly mustnot.
        class Runtime(FPORuntimeMixin):pass
        rt=Runtime();rt.actor=actor
        monkeypatch.setenv('RANK',str(rank))
        for inner in range(4):
            rt.validate_boundary(state(inner),saved,identity())
            assert actor._fpo_expect_fresh_material==(saved['fpo']['material_sha256'] if inner==0 else None)
    fpo.validate_ranks(summaries)
    bad=copy.deepcopy(summaries);bad[2]['advantages'][0]+=.1
    with pytest.raises(ValueError,match='globally finalized'):fpo.validate_ranks(bad)
    bad=copy.deepcopy(summaries);bad[2]['fpo_material_rank']=0
    with pytest.raises(ValueError,match='rank/MC/behavior'):fpo.validate_ranks(bad)


def test_runtime_finalizer_writes_only_initial_native_scale_and_calls_probe(monkeypatch,tmp_path):
    monkeypatch.setattr(torch.cuda,'synchronize',lambda:None)
    from starVLA.rl.qwen_world_research_algorithms import gradient_probe
    from starVLA.rl.simwam_recipe import checkpoint
    calls=[]
    def check(error):
        if error:raise AssertionError(error)
    monkeypatch.setattr(checkpoint,'check_errors',check)
    monkeypatch.setenv('RANK','0')
    class Parent:
        def load_recipe(self,args):return self.cfg
        def make_actor(self,cfg,args,device):
            self.actor=TinyActor(cfg);return self.actor,None,None
        def recipe_identity(self,cfg,args,condition_sha):return dict(recipe=cfg)
    class Runtime(FPORuntimeMixin,Parent):pass
    rt=Runtime();rt.cfg=recipe();rt.args=SimpleNamespace(output=str(tmp_path))
    assert rt.load_recipe(rt.args) is rt.cfg
    actor,_,_=rt.make_actor(rt.cfg,rt.args,torch.device('cpu'))
    assert isinstance(actor._loss,fpo.FPOTrainingLoss)
    _,out=collected();before=torch.random.get_rng_state().clone()
    rt.finalize_advantages(out)
    assert torch.equal(before,torch.random.get_rng_state())
    import json
    scale=json.loads((tmp_path/'initial_FPO_scale_rank0.json').read_text())
    assert scale['old_energy_min']>=0 and scale['protocol']==fpo.PROTOCOL
    assert scale['diagnostic_only'] and not scale['World_gradient_scale_known_from_this']
    assert rt.recipe_identity(rt.cfg,rt.args,'condition')['fpo_protocol']==fpo.PROTOCOL
    def probe(model,rollout,cfg,parts):
        calls.append(parts)
        assert set(parts)=={'pg','anchor'} and all(v.requires_grad for v in parts.values())
        assert model.p.grad is None
        return dict(gradient_probe_seconds=torch.tensor(0.))
    monkeypatch.setattr(gradient_probe,'maybe_probe_components',probe)
    loss,metrics=actor._loss._policy_loss(actor,out)
    assert len(calls)==1 and 'gradient_probe_seconds' in metrics
    assert 'fpo_positive_PPO_blocked_fraction' in metrics
    loss.backward();assert torch.isfinite(actor.p.grad)
    with pytest.raises(ValueError,match='once'):rt.finalize_advantages(out)
