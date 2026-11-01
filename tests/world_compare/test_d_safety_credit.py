"""Three bounded CPU checks for the unregistered D credit-only ablation."""
import copy
import json
import pytest
import torch

from test_d_jat_objective import cfg as baseline_cfg, native, cpu
from starVLA.rl.qwen_world_abcd_research import abc, d_objective as gaussian, d_safety as candidate
from starVLA.rl.qwen_world_compare.contracts import fingerprint


def cfg():
    value=baseline_cfg()
    value['research']['d_objective']=copy.deepcopy(candidate.CONFIG)
    value['research']['structured_advantage']=copy.deepcopy(abc.STRUCTURED_PROTOCOL)
    return value


def fresh(monkeypatch):
    actor,provider,rollout,identity=native(monkeypatch)
    # Collection physics is identical; only its fresh run's credit identity is
    # different. No candidate was learned or modified by the original loss.
    identity['recipe']=cfg();identity['d_objective_protocol']=copy.deepcopy(candidate.PROTOCOL)
    identity.pop('sha256');identity['sha256']=fingerprint(identity)
    rollout['behavior_version']['run']=identity['sha256']
    actor._loss=candidate.DSafetyJaTLoss(cfg())
    return actor,provider,rollout,identity


def test_credit_is_original_ABC_math_and_only_D_recipe_delta():
    original=baseline_cfg();changed=cfg();assert candidate.validate_recipe(changed) is changed
    check=copy.deepcopy(changed)
    check['research']['d_objective']=copy.deepcopy(gaussian.CONFIG)
    check['research'].pop('structured_advantage')
    assert check==original and gaussian.validate_recipe(original)
    with pytest.raises(ValueError):candidate.validate_recipe(original)
    with pytest.raises(ValueError):gaussian.validate_recipe(changed)
    for key,value in [('learning_rate',5e-5),('architecture_id','C')]:
        bad=cfg();bad[key]=value
        with pytest.raises(ValueError):candidate.validate_recipe(bad)
    x=torch.ones(8,8,4,dtype=torch.float64)
    x[:,:,3]=torch.linspace(0,1,8)
    x[0,2:,2]=0  # Real feasible alternatives: unsafe must have negative credit.
    x[1,:,2]=0   # All-fail same deficit despite progress variation: no invented A.
    x[2,:,3]=1   # Safe EP1 saturation: no invented A.
    x[3,:,0]=.5;x[3,0,0]=0  # Least-deficit fallback is still unsafe, not a safe label.
    rewards=x[:,:,0]*x[:,:,1]*(5*x[:,:,3]+5*x[:,:,2]+2)/12
    material=torch.cat((rewards.unsqueeze(-1),x),-1)
    stats=abc.safety_efficiency_advantages(rewards,x,all_material=material)
    a=stats['advantages'].reshape(8,8);mask=stats['adv_mask'].reshape(8,8)
    assert (a[0,2:]<0).all() and a[0,:2].min()>a[0,2:].max()
    assert not a[1].any() and not mask[1].any() and not a[2].any()
    assert a[3,1:].min()>0 and a[3,0]<0
    json.dumps(dict(cfg=cfg(),protocol=candidate.PROTOCOL),allow_nan=False)


def test_actual_tiny_D_four_inner_fresh_Gaussian_nativeKL_and_frozen_scope(monkeypatch):
    actor,provider,rollout,identity=fresh(monkeypatch)
    runtime=candidate.DSafetyRuntimeMixin();runtime.actor=actor
    runtime.finalize_advantages(rollout)
    before=gaussian.material_digest(rollout['d_gaussian'])
    canonical=rollout['rewards'].clone();adv=rollout['advantages'].clone()
    frozen={n:p.clone() for n,p in actor.named_parameters() if not p.requires_grad}
    parameters=[p for p in actor.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(parameters,lr=1e-6,betas=(.9,.95),eps=1e-8,weight_decay=.01)
    builds=actor.prepared_context_builds['current']
    for inner in range(4):
        optimizer.zero_grad(set_to_none=True);loss,metrics=actor(rollout)
        assert metrics['d_fresh_joint_ratio_checked']==float(inner==0)
        if inner==0:assert metrics['d_joint_ratio_max_abs_from_one']<1e-4 and metrics['d_reference_KL_joint']==0
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters)
        assert sum(float(p.grad.abs().sum()) for p in parameters)>0
        optimizer.step();actor.set_policy_version(f'inner{inner+1}')
        assert gaussian.material_digest(rollout['d_gaussian'])==before
        assert torch.equal(adv,rollout['advantages']) and torch.equal(canonical,rollout['rewards'])
    assert actor.prepared_context_builds['current']==builds+4
    assert actor.prepared_context_builds['reference']==4 and provider.prefill_calls==1
    assert all('language_model.layers.' in n for n,p in actor.named_parameters() if p.requires_grad)
    for n,p in actor.named_parameters():
        if n in frozen:assert torch.equal(p,frozen[n])
    assert actor._probability_audit['current']['source_clamped_logprob'].shape==(8,3,8,3)
    tied=copy.deepcopy(rollout);tied['advantages'].zero_();tied['adv_mask'].zero_()
    optimizer.zero_grad(set_to_none=True);loss,metrics=actor(tied);loss.backward()
    assert metrics['policy_loss']==0 and metrics['d_reference_KL_joint']>0
    assert sum(float(p.grad.abs().sum()) for p in parameters)>0


def test_eight_rank_native_roundtrip_all_inner_boundaries_and_poison_rejection(monkeypatch,tmp_path):
    actor,_,raw,identity=fresh(monkeypatch)
    rolls=[copy.deepcopy(raw) for _ in range(8)]
    for rank,out in enumerate(rolls):
        out['log']=f'log{rank}'
        for index,row in enumerate(out['records']):
            row['log']=out['log'];m=row['metrics']
            if rank==0 and index>=2:m['time_to_collision_within_bound']=0.
            if rank==1:m['ego_progress']=1.;m['comfort']=float(index%2)
            if rank==2:m['time_to_collision_within_bound']=0.
            m['score']=m['no_at_fault_collisions']*m['drivable_area_compliance']*(5*m['ego_progress']+5*m['time_to_collision_within_bound']+2*m['comfort'])/12
            row['readouts']['canonical_PDMS']=m['score']
        out['rewards']=torch.tensor([[r['metrics']['score'] for r in out['records']]],dtype=torch.float32)
    global_material=torch.cat([torch.cat((r['rewards'].double().unsqueeze(-1),abc.canonical_components(r['records'])['components']),-1) for r in rolls])
    monkeypatch.setattr(abc.dist,'is_available',lambda:True)
    monkeypatch.setattr(abc.dist,'is_initialized',lambda:True)
    monkeypatch.setattr(abc.dist,'get_world_size',lambda:8)
    def gather(outputs,value):
        for rank in range(8):outputs[rank].copy_(global_material[rank:rank+1])
    monkeypatch.setattr(abc.dist,'all_gather',gather)
    runtime=candidate.DSafetyRuntimeMixin();runtime.actor=actor;summaries=[]
    for rank,out in enumerate(rolls):
        runtime.finalize_advantages(out)
        path=tmp_path/f'rank{rank}.pt';torch.save(out,path);saved=torch.load(path,weights_only=False)
        assert saved['d_gaussian']['protocol']==candidate.PROTOCOL!=gaussian.PROTOCOL
        for inner in range(4):
            state=dict(updates=inner,collections=1,next_inner=inner,scene_cursor=8,raw_scene_exposures=8,new_candidates=64,
                phase='ROLLOUT_COMMITTED' if inner==0 else 'INNER_COMMITTED')
            summary=runtime.validate_boundary(state,saved,identity,
                optimizer=dict(state={0:dict(step=inner)},param_groups=[dict(params=[0])]),scheduler=dict(last_epoch=inner))
            assert actor._d_expect_fresh_material==(saved['d_gaussian']['sha256'] if inner==0 else None)
            if inner==0:summaries.append(summary)
        bad=copy.deepcopy(saved);bad['d_gaussian']['old_mean'].reshape(-1)[0]+=.1
        bad['d_gaussian']['sha256']=gaussian.material_digest(bad['d_gaussian'])
        with pytest.raises(ValueError,match='source differs'):candidate.validate_material(bad)
        bad=copy.deepcopy(saved);bad['d_gaussian']['protocol']=copy.deepcopy(gaussian.PROTOCOL)
        bad['d_gaussian']['sha256']=gaussian.material_digest(bad['d_gaussian'])
        with pytest.raises(ValueError,match='protocol'):candidate.validate_material(bad)
    candidate.validate_ranks(summaries)
    assert not rolls[1]['adv_mask'].any() and not rolls[2]['adv_mask'].any()
    assert rolls[1]['rewards'].std()>0 and not rolls[1]['advantages'].any()  # Comfort-only, not a failure.
    bad=copy.deepcopy(summaries);bad[0]['advantages'][0]+=.1
    with pytest.raises(ValueError,match='globally finalized'):candidate.validate_ranks(bad)
    bad=copy.deepcopy(summaries);bad[3]['d_objective_schema']=gaussian.SCHEMA
    with pytest.raises(ValueError,match='objective'):candidate.validate_ranks(bad)
