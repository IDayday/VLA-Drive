"""Directed actual-TinyScope CPU checks, not production driving-score evidence."""
import copy
import importlib.util
import io
from pathlib import Path

import pytest
import torch
import yaml

from test_world_performance_nominal import actual,primary,Service,Capture
from starVLA.rl.qwen_world_performance import nominal_improvement as N,ode_target as O
from starVLA.rl.qwen_world_verified_quality.recipe import build_recipe,validate_recipe,PROTOCOL,QUALITY_PROTOCOL_SHA256
from starVLA.rl.qwen_world_verified_quality.runtime import VerifiedQualityLoss,VerifiedQualityRuntime
from starVLA.rl.qwen_world_verified_quality import recovery
from starVLA.rl.qwen_world_safety_exploration.retention_material import digest,sealed


def material(*,high=True,ep=.8,primary_ep=(.9,1.)):
    torch.set_num_threads(1);a,c,cfg=actual();service=Service(nominal_good=high,nominal_ep=ep,primary_ep=primary_ep)
    leaf=primary(a,c,service);rng=torch.get_rng_state().clone()
    N.attach_material(a,leaf,c,service,exposure_epoch=0.,rank=0,config=N.QUALITY_CONFIG)
    assert torch.equal(rng,torch.get_rng_state())
    return a,c,cfg,service,leaf


def test_safe_high_now_has_verified_teacher_and_original_budget_split():
    a,c,cfg,service,r=material();p=N.validate_material(r,rank=0,expected_epoch=0.,config=N.QUALITY_CONFIG)
    assert p['enabled'] and p['active_alpha']==.02 and p['teacher']['gain']>=.02 and p['extra'] is None
    assert service.counts==[16,1]
    _,_,old_skip=N._select_teacher(r,p['nominal'],None)
    assert old_skip=='nominal_already_strict_high'
    base,_=a(r);pg=a._loss.pg_component;anchor=a._loss.weighted_anchor_component
    cfm,ode,_=N.compute_nominal_losses(a,r,N.QUALITY_CONFIG)
    normalized,_=N.calibrate_components(a,pg,anchor/50,cfm,ode,N.new_calibration_state())
    expected=base+.02*(50*normalized['rescue']-anchor)
    a._loss=VerifiedQualityLoss(cfg);a._loss.auxiliary_state=N.new_calibration_state();value,metrics=a(r)
    torch.testing.assert_close(value,expected,rtol=1e-6,atol=1e-6)
    assert float(metrics['nominal_aux_IL_share'])==pytest.approx(.98)
    assert float(metrics['nominal_aux_rescue_share'])==pytest.approx(.02)
    assert a.grpo_kl_beta==50.
    value.backward()
    for prefix in ('action_expert.','scene_provider.','predictive_provider.'):
        assert sum(float(p.grad.norm()) for n,p in a.named_parameters() if n.startswith(prefix) and p.requires_grad)>0


def test_not_strict_B1_uses_five_percent_and_small_gain_or_unsafe_teachers_refused():
    a,c,cfg,s,r=material(high=False)
    assert r['nominal_improvement']['enabled'] and r['nominal_improvement']['active_alpha']==.05
    a._loss=VerifiedQualityLoss(cfg);a._loss.auxiliary_state=N.new_calibration_state();_,m=a(r)
    assert float(m['nominal_aux_IL_share'])==pytest.approx(.95) and float(m['nominal_aux_rescue_share'])==pytest.approx(.05)
    _,_,_,s,small=material(ep=.995)
    assert not small['nominal_improvement']['enabled'] and small['nominal_improvement']['skip_reason']=='teacher_gain_below_margin'
    assert small['nominal_improvement']['active_alpha']==0 and s.counts==[16,1]
    _,_,_,_,small_verified=material(ep=.98)
    assert small_verified['nominal_improvement']['enabled'] and .005<=small_verified['nominal_improvement']['teacher']['gain']<.02
    _,_,_,_,small_repair=material(ep=.75,primary_ep=(.773,.784))
    assert not small_repair['nominal_improvement']['enabled'] and small_repair['nominal_improvement']['skip_reason']=='teacher_gain_below_margin'
    # Selector-only counterexample: high scalar score with TTC<1 is not strict.
    bad=copy.deepcopy(r)
    for record in bad['records']:
        metric=record['metrics'];metric['time_to_collision_within_bound']=.99
        metric['score']=metric['no_at_fault_collisions']*metric['drivable_area_compliance']*(5*metric['ego_progress']+5*.99+2*metric['comfort'])/12
        record['readouts']['canonical_PDMS']=metric['score']
    assert N._select_teacher(bad,bad['nominal_improvement']['nominal'],None,N.QUALITY_CONFIG)[2]=='no_strict_high_teacher'


def test_pending_native_material_and_calibration_resume_exact_and_gate_tampering_rejected():
    a,c,cfg,s,r=material();a._loss=VerifiedQualityLoss(cfg);a._loss.auxiliary_state=N.new_calibration_state()
    params=[p for p in a.parameters() if p.requires_grad];opt=torch.optim.AdamW(params,lr=1e-5)
    unchanged=digest({k:r[k] for k in ('old_logp','advantages','adv_mask','rewards','chain')})
    for _ in range(2):
        opt.zero_grad(set_to_none=True);loss,_=a(r);loss.backward();opt.step()
    payload=io.BytesIO();torch.save(dict(actor=a.state_dict(),optimizer=opt.state_dict(),rollout=r,calibration=a._loss.auxiliary_state),payload)
    payload.seek(0);saved=torch.load(payload,weights_only=False);right,rc,_=actual();right.load_state_dict(saved['actor'])
    right._loss=VerifiedQualityLoss(cfg);right._loss.auxiliary_state=saved['calibration'];ropt=torch.optim.AdamW([p for p in right.parameters()if p.requires_grad],lr=1e-5);ropt.load_state_dict(saved['optimizer'])
    N.validate_material(saved['rollout'],rank=0,expected_epoch=0.,config=N.QUALITY_CONFIG)
    assert saved['calibration']['effective_updates']==2
    for _ in range(2):
        opt.zero_grad(set_to_none=True);ropt.zero_grad(set_to_none=True)
        left,_=a(r);other,_=right(saved['rollout']);assert torch.equal(left,other)
        left.backward();other.backward()
        for p,q in zip(a.parameters(),right.parameters()):
            if p.requires_grad:assert torch.equal(p.grad,q.grad)
        opt.step();ropt.step();assert digest(opt.state_dict())==digest(ropt.state_dict())
    assert digest({k:r[k] for k in ('old_logp','advantages','adv_mask','rewards','chain')})==unchanged
    for field,value in [('active_alpha',.05),('exposure_epoch',-.1)]:
        bad=copy.deepcopy(r);bad['nominal_improvement'][field]=value
        bad['nominal_improvement']=sealed({k:v for k,v in bad['nominal_improvement'].items()if k!='material_sha256'},'material_sha256')
        with pytest.raises(ValueError):N.validate_material(bad,rank=0,expected_epoch=0.,config=N.QUALITY_CONFIG)


def test_V1_material_and_loss_default_are_bitexact_to_frozen_old_S1():
    path=Path('/mnt/project/qwen-world-performance-rl-20261103/runtime/reference_cache_probe_v2/starVLA/rl/qwen_world_performance/nominal_improvement.py')
    spec=importlib.util.spec_from_file_location('starVLA.rl.qwen_world_performance.frozen_nominal_fixture',path);old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    a,c,cfg=actual();service=Service();r=primary(a,c,service)
    nominal=O.probe_nominal(a,c,'t0','TRAIN_log',service,r['behavior_version']);left,right=copy.deepcopy(r),copy.deepcopy(r)
    old.attach_material(a,left,c,service,exposure_epoch=.5,rank=0,nominal=nominal)
    N.attach_material(a,right,c,service,exposure_epoch=.5,rank=0,nominal=nominal)
    assert digest(left['nominal_improvement'])==digest(right['nominal_improvement'])
    OldLoss=type('OldLoss',(old.NominalLossMixin,Capture),{});NewLoss=type('NewLoss',(N.NominalLossMixin,Capture),{})
    a._loss=OldLoss(cfg);a._loss.auxiliary_state=old.new_calibration_state();x,mx=a(left)
    gx=torch.autograd.grad(x,[p for p in a.parameters()if p.requires_grad])
    a._loss=NewLoss(cfg);a._loss.auxiliary_state=N.new_calibration_state();y,my=a(right)
    gy=torch.autograd.grad(y,[p for p in a.parameters()if p.requires_grad])
    assert torch.equal(x,y) and digest(mx)==digest(my)
    assert all(torch.equal(g,h)for g,h in zip(gx,gy))


def test_recipe_exact_B64_fresh_mature_clock_early_gates_and_default_reference_cache():
    base=yaml.safe_load(Path('/mnt/project/qwen-world-performance-rl-20261103/configs/S1_NOMINAL_RESCUE.yaml').read_text());cfg=build_recipe(base)
    assert cfg['max_steps']==129120 and cfg['gradient_accumulation_steps']==8
    assert cfg['research']['performance']['evaluation_epochs'][:2]==[.0625,.125]
    assert cfg['research']['verified_quality']==PROTOCOL and len(QUALITY_PROTOCOL_SHA256)==64
    import argparse
    parser=argparse.ArgumentParser();VerifiedQualityRuntime.add_arguments(parser)
    args=parser.parse_args(['--il-checkpoint','p','--parent-selection','s','--data','d']);assert args.frozen_reference_cache is True
    for change in [('learning_rate',3e-5),('max_steps',64560)]:
        bad=copy.deepcopy(cfg);bad[change[0]]=change[1]
        with pytest.raises(ValueError):validate_recipe(bad)
    bad=copy.deepcopy(cfg);bad['research']['performance']['nominal_rescue']['strict_high_nominal_anchor_budget_share']=.05
    with pytest.raises(ValueError):validate_recipe(bad)


def test_strict_S6_recovery_passes_explicit_validator_and_rejects_protocol_swap(monkeypatch):
    cfg=build_recipe(yaml.safe_load(Path('/mnt/project/qwen-world-performance-rl-20261103/configs/S1_NOMINAL_RESCUE.yaml').read_text()))
    identity=dict(recipe=cfg,verified_quality_protocol=PROTOCOL,quality_protocol_sha256=QUALITY_PROTOCOL_SHA256,
        performance_implementation_protocols=dict(nominal=N.QUALITY_CONFIG))
    called=[]
    def base(*args,**kwargs):called.append(kwargs);return {}
    monkeypatch.setattr(recovery.base,'validate_boundary',base)
    result=recovery.validate_boundary({},None,identity,expected_rank=3)
    assert result['quality_protocol_sha256']==QUALITY_PROTOCOL_SHA256
    assert called[0]['expected_rank']==3 and called[0]['recipe_validator'] is validate_recipe and called[0]['nominal_config']==N.QUALITY_CONFIG
    bad=copy.deepcopy(identity);bad['performance_implementation_protocols']['nominal']=N.CONFIG
    with pytest.raises(ValueError,match='quality/alpha/start'):recovery.validate_boundary({},None,bad,expected_rank=3)
    _,_,_,_,leaf=material();window=dict(valid_global_scenes=64,rank=0,samples=[leaf])
    recovery.validate_boundary(dict(valid_scene_exposures=64),window,identity,expected_rank=0)
    with pytest.raises(ValueError,match='identity changed'):
        recovery.validate_boundary(dict(valid_scene_exposures=128),window,identity,expected_rank=0)


def test_native_qualification_plan_uses_original_Trainer_without_artificial_gate(tmp_path):
    from tools.qwen_world_verified_quality.qualify_native import prepare
    cfg=Path(__file__).resolve().parents[1]/'configs/qwen_world_verified_quality/S6_VERIFIED_QUALITY.yaml'
    plan=prepare(tmp_path/'qualification',cfg.parents[2],cfg)
    assert plan['status']=='PREPARED_NOT_DISPATCHED' and plan['actual_start_epoch']==0.
    assert len(plan['stages'])==3 and plan['stages'][2]['depends_on']=='split2'
    for stage in plan['stages']:
        assert 'tools.qwen_world_verified_quality.train' in stage['command'] and '--qualification' in stage['command']
        assert '--qualification-force-auxiliary' not in stage['command']
    assert '--resume' in plan['stages'][2]['command'] and plan['cache_default'] is True


def test_formal_CPU_export_cannot_accept_pending_native(tmp_path):
    import json
    from tools.qwen_world_verified_quality.native_review import load_validated
    for name,value in [('COMPLETE',dict(schema=2,world=8,next_inner=2,phase='INNER_COMMITTED')),('identity.json',{}),('files.json',{})]:
        (tmp_path/name).write_text(json.dumps(value))
    with pytest.raises(ValueError,match='complete-window'):load_validated('unused',tmp_path)
    with pytest.raises(ValueError,match='qualification-only'):load_validated('unused',tmp_path,allow_pending=True)
