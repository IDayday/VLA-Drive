"""Finite CPU logic/source-FM qualification. Synthetic scores are NOT research results."""
import argparse
import copy
import json
from pathlib import Path
import tempfile

import numpy as np
import torch

from starVLA.model.modules.qwen_sourceflow.source_flow_policy import SourceFlowPolicy
from starVLA.model.modules.qwen_sourceflow.vendor.action_dit import ActionDiT
from starVLA.rl.qwen_flow_exploration.feedback.absorption import draw_teacher, native_source_fm_loss, teacher_probabilities
from starVLA.rl.qwen_flow_exploration.feedback.process import NoveltyScale, process_summary
from starVLA.rl.qwen_flow_exploration.feedback.proposal import structured_proposal
from starVLA.rl.qwen_flow_exploration.feedback.search import METHODS, decorate, discovery_budget_curve, physical_rank, run_search, scalar_rank, select_teacher_candidates
from starVLA.rl.qwen_flow_exploration.checkpoint import save_state,load_state
from starVLA.rl.qwen_sourceflow.actor import NativeRecipeActor
from starVLA.rl.qwen_sourceflow.recipe import expected_recipe
from starVLA.rl.simwam_recipe.lora_adapter import adapter_state,restore_adapters
from starVLA.rl.simwam_recipe.recovery_state import tensor_digest


def fixture(request, score=.9, nc=1., dac=1., ttc=1.):
    t=np.arange(41)*.1
    states=np.zeros((41,11));states[:,0]=2*t;states[:,3]=2.
    feedback=dict(status='OK', metrics=dict(score=score,no_at_fault_collisions=nc,
        drivable_area_compliance=dac,time_to_collision_within_bound=ttc,ego_progress=.8,comfort=1.),
        requested_trajectory=np.asarray(request,dtype=np.float32),simulated_states=states,
        non_drivable_body_by_time=np.zeros(41,dtype=bool),raw_progress_meters=float(request[-1,0]),
        official_first_at_fault_collision_seconds=None,official_first_ttc_infraction_seconds=None)
    return dict(status='OK',planned_trajectory=np.asarray(request,dtype=np.float32),raw_reward=score,feedback=feedback)


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--output',required=True);a=p.parse_args()
    torch.set_num_threads(2);torch.manual_seed(42);checks=[]
    base=np.column_stack([np.arange(1,9),np.zeros(8),np.zeros(8)]).astype(np.float32)
    identity,diagnostic=structured_proposal(base,[2.,0.],[0.,1.,0.,0.])
    assert np.array_equal(identity,base) and diagnostic['status']=='PHYSICAL_PROPOSAL_VALID'
    checks.append('Zero control exactly preserves native8x3 request, including heading')
    altered,diagnostic=structured_proposal(base,[2.,0.],[-.2,1.,.3,.1])
    assert np.isfinite(altered).all() and altered.shape==(8,3) and diagnostic['status']=='PHYSICAL_PROPOSAL_VALID'
    checks.append('Shared smooth four-parameter proposal, bounded physical driving-time validation')
    irregular=base.copy();irregular[0,0]=.5
    _,native_request=structured_proposal(irregular,[2.,0.],[0.,1.,0.,0.])
    assert native_request['status']=='PHYSICAL_PROPOSAL_VALID' and native_request['requested_curve_limits_are_diagnostics_only']
    checks.append('Finite native pose request is not rejected by unexecuted Hermite acceleration; real simulated dynamics required')
    good=decorate(fixture(base,.7), 'good',source='unit')
    bad=decorate(fixture(base,.9,nc=.5), 'bad',source='unit')
    assert scalar_rank(bad,42)<scalar_rank(good,42) and physical_rank(good,42)<physical_rank(bad,42)
    late=copy.deepcopy(bad);late['feedback']['official_first_at_fault_collision_seconds']=3.9
    early=copy.deepcopy(bad);early['feedback']['official_first_at_fault_collision_seconds']=.1
    assert physical_rank(early,42)==physical_rank(late,42)
    checks.append('Scalar-only PDMS versus feasible-first full-window feedback; delaying first accident alone is not rewarded')
    missing=copy.deepcopy(good['feedback']);missing.pop('non_drivable_body_by_time')
    try:process_summary(missing)
    except ValueError:pass
    else:raise AssertionError('Missing physical mask was silently filled as safe')
    summary=process_summary(good['feedback'])
    assert summary['available_mask']['full_window_collision_mask'] is False
    assert summary['available_mask']['full_window_ttc_mask'] is False
    checks.append('Physical missingness is explicit; no invented per-time collision/TTC or minimum TTC')
    base_rows=[fixture(base.copy(),.8+i*.001) for i in range(8)]
    scale=NoveltyScale.fit_train([process_summary(row['feedback'])['novelty_features'] for row in base_rows],split='TRAIN')
    try:NoveltyScale.fit_train([[0,0,0,0]],split='DEV')
    except ValueError:pass
    else:raise AssertionError('DEV used to fit novelty')
    def score_requests(requests):
        return [fixture(x,float(np.clip(.85+.08*(x[-1,0]-8),0,1)),dac=float(abs(x[-1,1])<.5)) for x in requests]
    def iid(round_index,count):return [fixture(base+np.array([[(round_index+1)*.01,0,0]],dtype=np.float32),.9) for _ in range(count)]
    details=[]
    for method in METHODS:
        result=run_search(method,base_rows,initial_velocity_xy=[2.,0.],sample_iid=iid,score_requests=score_requests,
                          seed=42,novelty_scale=scale if method=='F4_NOVELTY' else None)
        assert len(result['candidates'])==32 and result['costs']['candidate_attempts']==32
        assert all(row.get('on_policy_logprob') is None and not row['can_enter_GRPO_ratio'] for row in result['candidates'])
        curve=discovery_budget_curve(result,reference_reward=.8)
        assert [x['K'] for x in curve]==[8,16,24,32]
        details.append(dict(method=method,costs=result['costs'],budget_curve=curve))
    checks.append('All F0-F4 preserve32 attempts, common first8, independent official-score callbacks, no external on-policy probabilities')
    class PoisonMetrics(dict):
        def __getitem__(self,key):
            if key!='score':raise AssertionError('F2 read non-scalar metric '+key)
            return super().__getitem__(key)
    poison=[]
    for row in base_rows:
        row=copy.deepcopy(row);row['feedback']['metrics']=PoisonMetrics(row['feedback']['metrics']);poison.append(row)
    def poisoned_score(requests):
        rows=score_requests(requests)
        for row in rows:row['feedback']['metrics']=PoisonMetrics(row['feedback']['metrics'])
        return rows
    run_search('F2_PDMS',poison,initial_velocity_xy=[2.,0.],sample_iid=iid,score_requests=poisoned_score,seed=42)
    checks.append('Instrumented F2 cannot read NC/DAC/TTC or any extra metric during search')
    teacher=select_teacher_candidates([decorate(fixture(base,.95),'teacher',source='external')],reference_reward=.8,split='TRAIN')
    assert len(teacher)==1
    try:select_teacher_candidates(teacher,reference_reward=.8,split='DEV')
    except ValueError:pass
    else:raise AssertionError('DEV teacher entered update')
    requests=torch.tensor(np.stack([np.stack([base,altered]),np.stack([altered,base])]))
    weights=teacher_probabilities(torch.tensor([[.95,.96],[.97,.95]]),torch.ones(2,2,dtype=torch.bool))
    target,indices=draw_teacher(requests,weights,generator=torch.Generator().manual_seed(401))
    assert not weights.requires_grad and not target.requires_grad and indices.shape==(2,)
    try:draw_teacher(requests,-weights,generator=torch.Generator().manual_seed(1))
    except ValueError:pass
    else:raise AssertionError('Negative MSE weights accepted')
    checks.append('Only verified TRAIN targets, positive scene-normalized target sampling, no manufactured success labels')
    architecture=dict(action_dim=3,hidden_dim=32,ffn_dim=64,text_dim=40,freq_dim=16,num_heads=2,
                      attn_head_dim=12,num_layers=2,eps=1e-6,use_gradient_checkpointing=False)
    policy=SourceFlowPolicy(ActionDiT(**architecture)).eval()
    condition=dict(context=torch.randn(2,5,40),context_mask=torch.ones(2,5,dtype=torch.bool),visual_mask=torch.ones(2,7,dtype=torch.bool),
                   visual_kv=[dict(k=torch.randn(2,7,24),v=torch.randn(2,7,24)) for _ in range(2)])
    noise=torch.randn(target.shape,generator=torch.Generator().manual_seed(402));time=policy.scheduler.sample_training_t(2,'cpu',torch.float32)
    reference,_=policy(condition,target,noise=noise,timestep=time)
    actual,_=native_source_fm_loss(policy.velocity,condition,target,noise=noise,timestep=time)
    torch.testing.assert_close(actual,reference,atol=0,rtol=0)
    gradient_ref=torch.autograd.grad(reference,tuple(policy.parameters()),allow_unused=True,retain_graph=True)
    gradient_actual=torch.autograd.grad(actual,tuple(policy.parameters()),allow_unused=True)
    for x,y in zip(gradient_ref,gradient_actual):
        if x is None or y is None:assert x is None and y is None
        else:torch.testing.assert_close(x,y,atol=0,rtol=0)
    opt=torch.optim.AdamW(policy.parameters(),lr=1e-4)
    before={k:v.clone() for k,v in policy.state_dict().items()}
    actual,_=native_source_fm_loss(policy.velocity,condition,target,noise=noise,timestep=time)
    actual.backward();opt.step()
    assert any(not torch.equal(v,before[k]) for k,v in policy.state_dict().items())
    checks.append('Native source shift5 weighted FM loss and every parameter gradient match original policy exactly; real optimizer step')
    base_head=copy.deepcopy(policy.action_expert).eval().requires_grad_(False)
    def learner():
        actor=NativeRecipeActor(copy.deepcopy(base_head),expected_recipe('R_CANONICAL'))
        optimizer=torch.optim.AdamW([p for p in actor.parameters() if p.requires_grad],lr=5e-5,betas=(.9,.95),weight_decay=.01)
        return actor,optimizer
    def update(actor,optimizer,index):
        optimizer.zero_grad()
        local_noise=torch.randn(target.shape,generator=torch.Generator().manual_seed(800+index))
        loss,_=native_source_fm_loss(actor.action_velocity,condition,target,noise=local_noise,timestep=time)
        loss.backward();optimizer.step()
    continuous,continuous_opt=learner();update(continuous,continuous_opt,1)
    with tempfile.TemporaryDirectory() as temporary:
        state=dict(phase='READY',round=1,updates=1,next_inner=0,arm='feedback_CPU_resume_qualification',seed=42,scene_cursor=16)
        saved=save_state(temporary,state,dict(batch=None,adapters=adapter_state(continuous),optimizer=continuous_opt.state_dict()),'CPU_fixture_protocol_only')
        update(continuous,continuous_opt,2)
        expected_state=dict(adapters=adapter_state(continuous),optimizer=continuous_opt.state_dict())
        resumed,resumed_opt=learner();restored,payload=load_state(saved,'CPU_fixture_protocol_only')
        restore_adapters(resumed,payload['adapters']);resumed_opt.load_state_dict(payload['optimizer']);update(resumed,resumed_opt,2)
        assert tensor_digest(expected_state)==tensor_digest(dict(adapters=adapter_state(resumed),optimizer=resumed_opt.state_dict()))
        assert restored==state
    checks.append('Actual tiny native-LoRA source-FM READY checkpoint resumes to bitwise identical adapters and Adam moments')
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    report=dict(status='PASS',qualification_only=True,synthetic_score_fixture_is_not_driving_evidence=True,checks=checks,
                methods=details,source_FM_loss=float(reference.detach()),source_FM_gradient_exact=True)
    out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='methods'},indent=2))


if __name__=='__main__':main()
