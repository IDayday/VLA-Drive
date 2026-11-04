"""Teacher safety/ancestry, native target geometry, and real tiny-FPO gradients."""
import copy
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
import yaml
from starVLA.rl.d46_fpo_verified import history
from starVLA.rl.d46_fpo_verified.runtime import VerifiedActor,VerifiedExtension,VerifiedBoundary
from starVLA.rl.d44_native_rl.recovery import material_digest
from starVLA.rl.d44_native_rl.safety_credit import raw_credit
from starVLA.rl.d44_native_rl.adaptive_kl import AdaptiveKL
from starVLA.model.modules.qwen_world_postcot.flow import clean_x_step
from tools.d46_world16.qualify_native import make_actor,condition,SyntheticService,TinyActor


def record(score=.95,**metrics):
    return dict(status='OK',metrics=dict(score=score,no_at_fault_collisions=1.,drivable_area_compliance=1.,
        time_to_collision_within_bound=1.,comfort=1.,ego_progress=.9,driving_direction_compliance=1.,**metrics))


def target(update=0,score=.95,run='own'):
    return dict(action=torch.zeros(8,3).tolist(),record=record(score),version=dict(run=run,update=update,collection=0),source='own_initial_IL46_B1')


def test_strict_history_rejects_unsafe_and_foreign_and_future():
    t=target();assert history.strict(t['record'])
    for key in('no_at_fault_collisions','drivable_area_compliance','time_to_collision_within_bound','comfort'):
        bad=copy.deepcopy(t);bad['record']['metrics'][key]=0
        with pytest.raises(ValueError):history.validate_teacher(bad,'own',1)
    with pytest.raises(ValueError):history.validate_teacher(target(2),'own',1)
    with pytest.raises(ValueError):history.validate_teacher(target(run='other'),'own',1)
    assert history.retention_share(t,.8)==.25 and history.retention_share(t,.95)==.05


def test_committed_history_retains_best_and_enforces_TRAIN_membership():
    bank=history.initial('own');d=dict(token='TRAIN_A',version=dict(run='own',update=1),nominal_score=.7,teacher=target())
    history.commit(bank,[d],{'TRAIN_A'},capacity=103288);old=bank['digest']
    history.commit(bank,[dict(d,nominal_score=.99,teacher=target(1,.99))],{'TRAIN_A'},capacity=103288)
    assert bank['teachers']['TRAIN_A']['record']['metrics']['score']==.99 and bank['commits']==2 and old!=bank['digest']
    with pytest.raises(ValueError):history.commit(bank,[dict(d,token='NAVTEST_X')],{'TRAIN_A'},capacity=103288)
    with pytest.raises(ValueError):history.commit(bank,[d,d],{'TRAIN_A'},capacity=103288)


def test_real_ODE_final_state_equals_clean_prediction_and_teacher_digest_changes():
    z=torch.randn(1,8,3);x=torch.randn_like(z)
    assert torch.equal(clean_x_step(z,x,torch.tensor(.3571428657),torch.tensor(0.)),x)
    before=dict(scenes=[dict(retention_target=x,retention_version=dict(run='own',update=0))])
    after=copy.deepcopy(before);after['scenes'][0]['retention_target'].add_(.01)
    assert material_digest(before)!=material_digest(after)


def test_native_FPO_and_verified_CFM_keep_group_and_frozen_reference(monkeypatch):
    monkeypatch.setattr(torch.cuda,'synchronize',lambda *a,**kw:None)
    torch.set_num_threads(1);torch.manual_seed(42)
    cfg=yaml.safe_load(Path('configs/d46_fpo_verified/verified_b1.yaml').read_text())
    actor=make_actor(cfg)
    class TinyVerified(VerifiedActor):scope_audit=TinyActor.scope_audit
    actor.__class__=TinyVerified
    reference={n:p.detach().clone()for n,p in actor.ref_action_expert.named_parameters()}
    ext=VerifiedExtension();ext.cfg=cfg;ext.bank=history.initial('own');ext.rows=[]
    ext.stream=SimpleNamespace(tokens=['TRAIN_CPU_0'])
    g=torch.Generator().manual_seed(420000)
    scene=ext.collect_scene(actor,condition(actor,0),'TRAIN_CPU_0','TRAIN',SyntheticService(),g,cfg,0.,dict(run='own',update=0,collection=1))
    assert scene['chain'].shape==(16,11,8,3)and len(scene['records'])==16 and scene['PG_candidates']==16
    assert scene['fm_state'].shape==(4,16,8,3)and scene['rescue_state'].shape==(2,8,3)
    assert not scene['rescue_teacher'].requires_grad and scene['extra_B1']==1
    raw,_=raw_credit(scene['factors'][None]);scene.update(advantages=(raw/(raw.std(unbiased=False)+1e-8)).clamp(-5,5).float()[0],beta=.001,expect_fresh=True,padding=False)
    loss,metrics=actor(scene)
    assert metrics['fresh_ratio_error']==0 and metrics['fpo_joint_logratio_min']==0 and metrics['fpo_joint_logratio_max']==0
    assert metrics['rescue_active']==1 and torch.isfinite(loss)
    loss.backward();grads=[p.grad for p in actor.parameters()if p.requires_grad and p.grad is not None]
    assert grads and all(torch.isfinite(g).all()for g in grads)and sum(float(g.square().sum())for g in grads)>0
    assert all(torch.equal(reference[n],p)for n,p in actor.ref_action_expert.named_parameters())
    state=dict(verified_history=ext.bank);ext.after_collection(state,g,0,1)
    batch=dict(scenes=[scene],behavior_update=0,valid_scenes=1,active_global_scenes=1,epoch=0.);batch['digest']=material_digest(batch)
    state.update(updates=0,collections=1,next_inner=0,phase='ROLLOUT_COMMITTED',scene_cursor=1,valid_scene_exposures=1,new_candidates=16,adaptive_kl=AdaptiveKL().state_dict())
    identity=dict(sha256='own',world=1,recipe=dict(inner_updates=2,global_scenes=1,arm='D3_FPO'))
    VerifiedBoundary.validate_boundary(state,batch,identity)
    bad=copy.deepcopy(batch);bad['scenes'][0]['rescue_record']['metrics']['time_to_collision_within_bound']=0;bad['digest']=material_digest(bad)
    with pytest.raises(ValueError,match='Rescue'):VerifiedBoundary.validate_boundary(state,bad,identity)
