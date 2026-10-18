"""Bounded CPU fixtures for objective decomposition, not driving evidence."""
from copy import deepcopy
from dataclasses import replace
import json

import pytest
import torch

from tools.st_world_policy import audit_flow_gradient_balance as audit
from starVLA.model.modules.flow_posttraining import (
    FlowRandomStreams, module_hash, group_centered_advantages, fpo_surrogate)
from tools.st_world_policy.train_flow_policy import save_update_feedback
from test_audit_flow_gradient import fixture


def test_rank_ownership_matches_actual_strided_sampler():
    m={'entries':[{'scene_id':str(i)} for i in range(70)]}
    e={'world_size':4,'global_batch':32,'micro_batch':4,'seed':42}
    ranks,cursor=audit.next_rank_entries(m,{'epoch':0,'offset':1},e)
    assert [[int(x['scene_id']) for x in rows] for rows in ranks] == [cursor['manifest_indices'][r::4] for r in range(4)]
    assert len({x['scene_id'] for rows in ranks for x in rows})==32
    with pytest.raises(ValueError,match='complete 32'):
        audit.next_rank_entries(m,{'epoch':0,'offset':2},e)
    with pytest.raises(ValueError,match='world4'):
        audit.next_rank_entries(m,{'epoch':0,'offset':0},{**e,'world_size':1})


def test_rank_payload_is_bound_to_rank_and_before_update(tmp_path):
    cfg,old,c,rollout,comparison,feedback,_=fixture(batch=1)
    save_update_feedback(tmp_path,rollout,comparison,feedback,numeric_hash='2'*64,rank=3)
    kw=dict(numeric_hash='2'*64,step=0,binding_hash='a'*64,
        entries=[{'scene_id':'scene_0','observation_hash':'b'*64}],fpo_config=cfg['fpo'])
    actual=audit.load_rank_payload(tmp_path/'rank_3_COMPLETE.json',3,**kw)
    assert actual[0].rollout_id==rollout.rollout_id
    with pytest.raises(ValueError,match='rank boundary'):
        audit.load_rank_payload(tmp_path/'rank_3_COMPLETE.json',0,**kw)
    with pytest.raises(ValueError,match='boundary'):
        audit.load_rank_payload(tmp_path/'rank_3_COMPLETE.json',3,**{**kw,'step':1})


def test_cosine_weighted_contributions_and_zero_norm_are_explicit():
    s={'bc_squared_l2':4.,'policy_squared_l2':9.,'dot':-6.,'elements':1}
    r=audit.derived_statistics(s,.5)
    assert r['cosine']==-1 and r['weighted_policy_over_bc']==.75
    assert r['combined_gradient_l2']==.5
    assert r['first_order_BC_change_under_combined_descent_per_unit_lr']==-1
    zero=audit.derived_statistics({**s,'bc_squared_l2':0.,'dot':0},1.)
    assert zero['cosine'] is None and zero['weighted_policy_over_bc'] is None


@pytest.mark.parametrize('checkpoint_velocity',[False,True])
def test_original_dropout_BC_and_eval_FPO_decompose_without_optimizer(checkpoint_velocity,monkeypatch):
    cfg,old,condition,rollout,comparison,feedback,_=fixture(batch=2)
    policy=deepcopy(old.policy).requires_grad_(True)
    before=module_hash(old.policy)
    targets=torch.randn(2,8,4,generator=torch.Generator().manual_seed(9))
    streams=FlowRandomStreams(55);rng=torch.get_rng_state().clone();stream_state=streams.state_dict()
    def no_optimizer(*a,**k):raise AssertionError('diagnostic must never construct an optimizer')
    monkeypatch.setattr(torch.optim,'AdamW',no_optimizer)
    sums=audit.GradientSum(policy)
    report=audit.audit_rank(policy,condition,targets,rollout,comparison,feedback,streams,sums,
        global_scenes=8,micro_batch=1,mc_chunk=5,checkpoint_velocity=checkpoint_velocity)
    stats=sums.report()
    assert stats['microbatch_counts']=={'bc':2,'policy':2}
    assert stats['groups']['DiT']['bc_l2']>0 and stats['groups']['qwen_proj']['policy_l2']>0
    assert module_hash(policy)==module_hash(old.policy)==before
    assert not torch.equal(torch.get_rng_state(),rng)  # Production dropout stays on.
    assert all(p.grad is None for p in old.parameters())
    assert all(x['bit_exact'] for x in report['current_CFM'])
    # Independent original joint graph with the SAME saved BC/dropout RNG.
    torch.set_rng_state(rng);streams.load_state_dict(stream_state)
    policy.train();policy.zero_grad(set_to_none=True)
    adv=group_centered_advantages(feedback.rewards,feedback.valid).flatten()
    for at in range(2):
        bc=policy.bc_anchor(condition[at:at+1],targets[at:at+1],noise_generator=streams['bc_noise'],
            time_generator=streams['bc_time'],global_scene_count=8,world_size=1)
        sl=slice(at*4,(at+1)*4)
        current=policy(condition[at:at+1].repeat_interleave(4,0),rollout.raw_actions.flatten(0,1)[sl],
            comparison.samples.slice(sl),mc_chunk=5,checkpoint_velocity=checkpoint_velocity)
        pl,_=fpo_surrogate(comparison.old_cfm[sl],current,adv[sl],feedback.valid.flatten()[sl],
            comparison.config,global_count=32,world_size=1)
        (bc+pl).backward()
    # FP32 CPU arithmetic diagnostic; no claim of BF16/ZeRO bit equality.
    for name,p in policy.named_parameters():
        torch.testing.assert_close(sums.sums['bc'][name]+sums.sums['policy'][name],p.grad,rtol=1e-5,atol=2e-7)


def test_four_unequal_rank_contributions_use_global_denominator():
    # Analytic scalar model avoids a hidden average of local means.
    parameter=torch.nn.Linear(1,1,bias=False);parameter.weight.data.fill_(.3)
    acc=audit.GradientSum(parameter)
    batches=[torch.tensor([1.,2.]),torch.tensor([3.]),torch.tensor([4.,5.,6.]),torch.tensor([7.,8.])]
    for rows in batches:
        for kind,sign in [('bc',1.),('policy',-.5)]:
            parameter.zero_grad(set_to_none=True)
            loss=sign*(parameter.weight.flatten()[0]*rows).sum()/8
            loss.backward();acc.add(kind,parameter)
    r=acc.report()['groups']['all']
    assert r['bc_l2']==4.5 and r['policy_l2']==2.25 and r['cosine']==-1
    assert r['combined_gradient_l2']==2.25


def test_zero_advantage_keeps_BC_and_zero_policy():
    _,old,c,rollout,comparison,feedback,_=fixture(tied=True,batch=1)
    p=deepcopy(old.policy).requires_grad_(True);acc=audit.GradientSum(p)
    result=audit.audit_rank(p,c,torch.ones(1,8,4),rollout,comparison,feedback,FlowRandomStreams(12),acc,
        global_scenes=4,micro_batch=1,mc_chunk=5,checkpoint_velocity=False)
    s=acc.report()['groups']['all']
    assert result['nonzero_advantages']==0 and s['policy_l2']==0 and s['bc_l2']>0 and s['cosine'] is None


@pytest.mark.parametrize('wrong',['condition','policy','comparison'])
def test_wrong_replay_refused_before_attributing_gradients(wrong):
    _,old,c,rollout,comparison,feedback,_=fixture(batch=1)
    p=deepcopy(old.policy).requires_grad_(True)
    if wrong=='condition':c=c+.01
    elif wrong=='policy':
        with torch.no_grad():next(p.parameters()).add_(.01)
    else:comparison.old_cfm[0,0]+=.1
    with pytest.raises(ValueError):
        audit.audit_rank(p,c,torch.ones(1,8,4),rollout,comparison,feedback,FlowRandomStreams(12),audit.GradientSum(p),
            global_scenes=4,micro_batch=1,mc_chunk=5,checkpoint_velocity=False)


def test_rng_scope_restores_on_failure(monkeypatch):
    # CPU emulation exercises context manager control flow only, not CUDA claims.
    values={'state':1}
    monkeypatch.setattr(audit,'capture_rng',lambda generators:dict(values))
    monkeypatch.setattr(audit,'restore_rng',lambda state,generators:values.update(state))
    streams=type('Streams',(),{'generators':{}})()
    with pytest.raises(RuntimeError):
        with audit.isolated_native_rng({'state':7},streams):
            assert values['state']==7
            values['state']=8;raise RuntimeError('injected')
    assert values['state']==1


def test_cli_rejects_unknown_fields_before_real_load(tmp_path,monkeypatch):
    registration=tmp_path/'REGISTRATION.json';registration.write_text(json.dumps({'schema':audit.SCHEMA,'bogus':1}))
    def forbidden(*a,**k):raise AssertionError('no model/data execution during invalid registration')
    monkeypatch.setattr(audit,'prepare_inputs',forbidden)
    assert audit.main(['--registration',str(registration),'--execute'])==1


def registration_fixture(tmp_path,monkeypatch):
    def artifact(p,value):
        p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value))
        return {'path':str(p),'sha256':audit.file_sha256(p)}
    root=tmp_path/'runs';run=root/'C1';tag='step_000001_fixture'
    cfg={'cell':'C1','runtime':{'output_root':str(root)},'loss':{'bc_weight':1.,'policy_weight':1.}}
    ca=artifact(tmp_path/'CONFIG.json',cfg)
    tr=artifact(tmp_path/'TRAINING.json',{'run_id':'C1','artifacts':{'config':ca}})
    cp=artifact(run/'checkpoints'/tag/'COMPLETE.json',{'world_size':4,'state':{
        'completed':1,'policy_version':1,'pending_feedback':None,'pending_old_policy':None}})
    rr=[artifact(run/'rollouts/boundary_000001'/f'rank_{r}_COMPLETE.json',
        {'identity':{'old_policy':{'optimizer_step':1,'version':1}}}) for r in range(4)]
    reg={'schema':audit.SCHEMA,'source_sha':'a'*40,'training_registration':tr,'checkpoint_tag':tag,
        'checkpoint_receipt':cp,'rollout_receipts':rr,'budget':dict(audit.LIMITS),
        'output':str(tmp_path/'audit'),'resource':{'host':'fixture-host','gpu_uuid':'GPU-fixture','cpu_threads':2},
        'diagnostic':{'cell':'C1','global_scenes':32,'original_world_size':4,'optimizer_steps':0,
            'new_score_calls':0,'analysis':'separate_micro_gradients_fp32_sum'}}
    monkeypatch.setattr(audit.subprocess,'check_output',lambda *a,**k:('a'*40).encode())
    monkeypatch.setattr(audit.subprocess,'run',lambda *a,**k:type('Result',(),{'returncode':0})())
    p=tmp_path/'AUDIT_REGISTRATION.json';p.write_text(json.dumps(reg))
    return p,reg


def test_receipt_preflight_no_model_manifest_or_GPU_load(tmp_path,monkeypatch):
    p,reg=registration_fixture(tmp_path,monkeypatch)
    def forbidden(*a,**k):raise AssertionError('small preflight must not load tensors/models')
    monkeypatch.setattr(torch,'load',forbidden)
    assert audit.main(['--registration',str(p)])==0
    assert not (tmp_path/'audit').exists()


@pytest.mark.parametrize('change',['source','budget','later_checkpoint','output','missing_rank','weight'])
def test_registration_rejects_identity_and_budget_changes(tmp_path,monkeypatch,change):
    p,r=registration_fixture(tmp_path,monkeypatch)
    if change=='source':r['source_sha']='b'*40
    elif change=='budget':r['budget']['max_seconds']=1801
    elif change=='later_checkpoint':
        path=__import__('pathlib').Path(r['checkpoint_receipt']['path'])
        c=json.loads(path.read_text());c['state']['completed']=c['state']['policy_version']=2
        path.write_text(json.dumps(c));r['checkpoint_receipt']['sha256']=audit.file_sha256(path)
    elif change=='output':r['output']=str(tmp_path/'runs/C1/forbidden')
    elif change=='missing_rank':r['rollout_receipts'].pop()
    else:r['diagnostic']['optimizer_steps']=1
    p.write_text(json.dumps(r))
    with pytest.raises(ValueError):audit.validate_registration(p)
