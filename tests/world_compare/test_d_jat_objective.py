"""Small CPU tests for D joint-ratio/nativeKL and native-Qwen gradient closure."""
import copy
import json
from types import SimpleNamespace
import pytest
import torch
from torch.distributions import Normal

from test_runtime import CurrentModel,Data,processor
from starVLA.rl.qwen_world_compare.runtime import expected_recipe,FrozenCurrentConditions
from starVLA.rl.qwen_world_compare.policy_adapter import FourFrameworkRecipeActor
from starVLA.rl.qwen_world_compare import recovery_state as source_recovery
from starVLA.rl.qwen_world_compare.contracts import fingerprint
from starVLA.rl.qwen_sourceflow.collection import collect
from starVLA.rl.qwen_world_abcd_research import d_objective as d
from starVLA.rl.qwen_world_abcd_research.d_entry import DRuntimeMixin


@pytest.fixture(autouse=True)
def cpu():
    threads=torch.get_num_threads();torch.set_num_threads(2);torch.manual_seed(42)
    yield
    torch.set_num_threads(threads)


def cfg():
    c=expected_recipe('D');c.update(learning_rate=1e-6,research=dict(d_objective=copy.deepcopy(d.CONFIG)))
    c['grpo']['sample']['randn_clip']=None
    c['grpo']['train'].update(ppo_clip_range=.2,kl_beta=.001,kl_type='x_kl',kl_dim_weights=None,kl_horizon_weights=None)
    return c


def test_joint24_gaussian_identity_native_KL_gradient_and_PPO_clip():
    old=torch.randn(8,3,8,3,requires_grad=True);cur=(old.detach()+.01*torch.randn_like(old)).requires_grad_()
    ref=torch.randn_like(old,requires_grad=True);std=torch.tensor([.1,.15,.2],requires_grad=True)
    nxt=old.detach()+std.detach()[None,:,None,None]*torch.randn_like(old)
    logr,kl=d.gaussian_logratio_kl(cur,old,ref,std,nxt)
    density=(Normal(cur.double(),std.detach().double()[None,:,None,None]).log_prob(nxt.double())-
             Normal(old.detach().double(),std.detach().double()[None,:,None,None]).log_prob(nxt.double())).sum((-1,-2))
    torch.testing.assert_close(logr.double(),density,atol=3e-6,rtol=3e-5)
    exactKL=.5*((cur.double()-ref.detach().double())/std.detach().double()[None,:,None,None]).square().sum((-1,-2))
    torch.testing.assert_close(kl.double(),exactKL,atol=2e-3,rtol=3e-7)
    assert torch.equal(d.gaussian_logratio_kl(old.detach(),old,ref,std,nxt)[0],torch.zeros(8,3))
    loss,_=d.clipped_objective(logr,kl,torch.linspace(-1,1,8),torch.ones(8));loss.backward()
    assert torch.isfinite(cur.grad).all() and cur.grad.abs().sum()>0
    assert old.grad is None and ref.grad is None and std.grad is None
    leaf=torch.zeros(8,3,requires_grad=True)
    with torch.no_grad():leaf[0]=torch.log(torch.tensor(1.3));leaf[1]=torch.log(torch.tensor(.7))
    adv=torch.tensor([1.,-1.,0,0,0,0,0,0]);loss,_=d.clipped_objective(leaf,leaf*0,adv,torch.ones(8));loss.backward()
    assert torch.equal(leaf.grad[:2],torch.zeros(2,3))


class Service:
    protocol={'sha256':'official'}
    def score(self,args):
        token,log,trajectories=args;rows=[]
        for i,request in enumerate(trajectories):
            ep=(i+1)/8;score=(5*ep+7)/12
            metrics=dict(score=score,no_at_fault_collisions=1.,drivable_area_compliance=1.,
                time_to_collision_within_bound=1.,ego_progress=ep,comfort=1.,driving_direction_compliance=1.)
            rows.append(dict(status='OK',token=token,log=log,protocol_sha256='official',metrics=metrics,
                requested_trajectory=request.tolist(),readouts=dict(canonical_PDMS=score),scoring_seconds=0.))
        return rows


def native(monkeypatch):
    monkeypatch.setattr(torch.cuda,'synchronize',lambda:None)
    actual=cfg();construction=copy.deepcopy(actual);construction['grpo']['sample']['randn_clip']=float('inf')
    # The source actor needs a runtime-only infinity sentinel. No inf enters
    # the sealed cfg/material: formal root ResearchActor owns this narrow shim.
    model=CurrentModel('D');actor=FourFrameworkRecipeActor(model,construction);actor._loss=d.DActionJaTLoss(actual)
    provider=FrozenCurrentConditions(model,Data(),dict(model_sha256='ownIL',identity={'architecture_id':'D'}),processor=processor)
    object.__setattr__(actor,'conditions',provider)
    descriptor=provider.get('s0')
    identity=dict(schema='qwen_world_compare_RL_v1',architecture_id='D',world=8,recipe=actual,
        projected_condition_identity=provider.sha,d_objective_protocol=copy.deepcopy(d.PROTOCOL))
    identity['sha256']=fingerprint(identity)
    out=collect(actor,descriptor,'s0','log0',Service(),torch.Generator().manual_seed(42),
        variant='R_CANONICAL',version=dict(run=identity['sha256'],update=0,adapter_sha256='ownIL'))
    return actor,provider,out,identity


def test_actual_tiny_native_D_four_updates_rebuild_prefix_and_only_text_lora_changes(monkeypatch):
    actor,provider,out,identity=native(monkeypatch)
    d.finalize_advantages(out);object.__setattr__(actor,'_d_expect_fresh_material',out['d_gaussian']['sha256'])
    before=d.material_digest(out['d_gaussian']);old=out['old_logp'].clone()
    frozen={n:p.clone() for n,p in actor.named_parameters() if not p.requires_grad}
    params=[p for p in actor.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(params,lr=1e-6,betas=(.9,.95),eps=1e-8,weight_decay=.01)
    builds=actor.prepared_context_builds['current']
    for inner in range(4):
        optimizer.zero_grad(set_to_none=True);loss,metrics=actor(out)
        assert metrics['d_fresh_joint_ratio_checked']==float(inner==0)
        if inner==0:assert metrics['d_joint_ratio_max_abs_from_one']<1e-4 and metrics['d_reference_KL_joint']==0
        loss.backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
        assert sum(float(p.grad.abs().sum()) for p in params)>0
        torch.nn.utils.clip_grad_norm_(params,1.);optimizer.step();actor.set_policy_version(f'inner{inner+1}')
        assert d.material_digest(out['d_gaussian'])==before
        assert actor._prepared_scope is None
    assert actor.prepared_context_builds['current']==builds+4 and actor.prepared_context_builds['reference']==4
    assert provider.prefill_calls==1
    assert all('language_model.layers.' in n and ('lora_A' in n or 'lora_B' in n)
               for n,p in actor.named_parameters() if p.requires_grad)
    for n,p in actor.named_parameters():
        if n in frozen:torch.testing.assert_close(p,frozen[n],atol=0,rtol=0)
    torch.testing.assert_close(old,out['old_logp'],atol=0,rtol=0)
    assert actor._probability_audit['current']['source_clamped_logprob'].shape==(8,3,8,3)
    tied=copy.deepcopy(out);tied['advantages'].zero_();tied['adv_mask'].zero_()
    optimizer.zero_grad(set_to_none=True);loss,metrics=actor(tied);loss.backward()
    assert metrics['policy_loss']==0 and metrics['d_reference_KL_joint']>0
    assert sum(float(p.grad.abs().sum()) for p in params)>0  # NativeIL KL stays, no fabricated PG.


def test_eight_rank_centered_scale_native_roundtrip_and_pollution_rejection(monkeypatch,tmp_path):
    actor,provider,raw,identity=native(monkeypatch)
    rolls=[copy.deepcopy(raw) for _ in range(8)]
    rewards=[]
    for rank,out in enumerate(rolls):
        out['log']=f'log{rank}'
        # Same valid input has different canonical outcomes in the synthetic
        # rank dataset; actor/transition physics remains the actual native call.
        for row in out['records']:row['log']=out['log']
        if rank==1:
            for row in out['records']:row['metrics']['ego_progress']=row['metrics']['score']=row['readouts']['canonical_PDMS']=1.
            out['rewards'].fill_(1.)
        rewards.append(out['rewards'].double())
    global_rewards=torch.cat(rewards)
    monkeypatch.setattr(d.dist,'is_available',lambda:True);monkeypatch.setattr(d.dist,'is_initialized',lambda:True)
    monkeypatch.setattr(d.dist,'get_world_size',lambda:8)
    def gather(out,value):
        for i in range(8):out[i].copy_(global_rewards[i:i+1])
    monkeypatch.setattr(d.dist,'all_gather',gather)
    class Parent:
        def validate_boundary(self,*args,**kwargs):return source_recovery.validate_boundary(*args,**kwargs)
    class Runtime(DRuntimeMixin,Parent):pass
    runtime=Runtime();runtime.actor=actor;summaries=[]
    for rank,out in enumerate(rolls):
        runtime.finalize_advantages(out);path=tmp_path/f'rank{rank}.pt';torch.save(out,path);saved=torch.load(path,weights_only=False)
        assert saved['d_gaussian']['protocol']==d.PROTOCOL
        for inner in range(4):
            state=dict(updates=inner,collections=1,next_inner=inner,scene_cursor=8,raw_scene_exposures=8,new_candidates=64,
                phase='ROLLOUT_COMMITTED' if inner==0 else 'INNER_COMMITTED')
            sm=runtime.validate_boundary(state,saved,identity,
                optimizer=dict(state={0:dict(step=inner)},param_groups=[dict(params=[0])]),scheduler=dict(last_epoch=inner))
            if inner==0:summaries.append(sm)
            assert actor._d_expect_fresh_material==(saved['d_gaussian']['sha256'] if inner==0 else None)
        for key in ('old_mean','std','recorded_next','innovation'):
            bad=copy.deepcopy(saved);bad['d_gaussian'][key].reshape(-1)[0]+=.1
            with pytest.raises(ValueError,match='digest'):d.validate_material(bad)
            bad['d_gaussian']['sha256']=d.material_digest(bad['d_gaussian'])
            with pytest.raises(ValueError,match='differs'):d.validate_material(bad)
    d.validate_ranks(summaries)
    bad=copy.deepcopy(summaries);bad[2]['advantages'][0]+=.1
    with pytest.raises(ValueError,match='globally finalized'):d.validate_ranks(bad)
    assert not rolls[1]['advantages'].any() and not rolls[1]['adv_mask'].any()
    for change in ('clip','lr','physical','noise'):
        bad=cfg()
        if change=='clip':bad['grpo']['train']['ppo_clip_range']=.005
        if change=='lr':bad['learning_rate']=5e-5
        if change=='physical':bad['grpo']['train']['kl_beta']=50
        if change=='noise':bad['grpo']['sample']['randn_clip']=5.
        with pytest.raises(ValueError):d.validate_recipe(bad)
    assert d.validate_recipe(cfg())
    json.dumps(dict(protocol=d.PROTOCOL,cfg=cfg()),allow_nan=False)  # No runtime-only inf in native metadata.
