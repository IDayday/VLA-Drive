"""Two CPU checks: actual source random coupling and whole-population closure."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tools.qwen_world_safety_exploration import probe_support as P
from starVLA.rl.simwam_recipe.vendor.action_math import ActionGRPOMixin
from starVLA.rl.simwam_recipe.vendor.scheduler_continuous import WanContinuousFlowMatchScheduler


class Toy(ActionGRPOMixin):
    def __init__(self):
        self.device=torch.device('cpu');self.torch_dtype=self.grpo_action_dtype=torch.float32
        self.action_expert=SimpleNamespace(action_dim=3)
        self.infer_action_scheduler=WanContinuousFlowMatchScheduler(1000,shift=5.)
        self.grpo_num_inference_steps=10;self.grpo_infer_shift=None
        self.grpo_sde_step_indices=None;self.grpo_train_sde_steps=3
        self.grpo_sde_exclude_last_step=False;self.grpo_merge_sde_window=False
        self.grpo_noise_level=.1;self.grpo_sde_mode='rigorous';self.grpo_sigma_max_guard=.99
        self.grpo_min_std=.001;self.grpo_randn_clip=5.;self.grpo_final_action_clip=None
        self.record_sampling_terms=True;self._noise_audit=[]
    def action_velocity(self,x,t,condition,use_ref=False):return .15*x+.01*t[:,None,None]/1000


def test_actual_source_four_strategy_rng_and_instance_restore():
    actor=Toy();before=copy.deepcopy(actor.__dict__);rng=torch.get_rng_state().clone()
    condition=dict(context=torch.ones(16,1,2))
    chains,draws,audit=P.matched_chains(actor,condition,'token')
    assert torch.equal(rng,torch.get_rng_state())
    assert '_randn' not in actor.__dict__
    for key in ('grpo_sde_step_indices','grpo_train_sde_steps','grpo_sde_exclude_last_step','record_sampling_terms'):
        assert getattr(actor,key)==before[key]
    assert audit['raw_roots_equal_all4'] and audit['generator_poststate_equal_ABC']
    assert audit['ordered_raw_innovations_equal_ABC'] and audit['SDE_vs_ODE_pre7_prefix_equal']
    assert chains['C_shared_root'][:,0].unique(dim=0).shape[0]==1
    assert chains['A_late'][:,0].unique(dim=0).shape[0]==16
    assert all(torch.equal(draws['A_late'][j],draws['C_shared_root'][j]) for j in range(4))
    assert not torch.equal(chains['B_mixed'][:,-1],chains['A_late'][:,-1])
    original=actor.sample_action_chain
    def bad(*a,**kw):
        if actor.grpo_sde_step_indices==[3,6,8]:raise RuntimeError('test-only interruption')
        return original(*a,**kw)
    actor.sample_action_chain=bad
    with pytest.raises(RuntimeError):P.matched_chains(actor,condition,'token')
    assert '_randn' not in actor.__dict__ and actor.grpo_sde_step_indices is None and actor.record_sampling_terms


def metric():
    return dict(no_at_fault_collisions=1.,drivable_area_compliance=1.,time_to_collision_within_bound=1.,
                comfort=1.,ego_progress=.88,score=.95,driving_direction_compliance=1.)


def test_full_union_over32_same_log_controls_complete_shards_and_no_partial_mean(tmp_path):
    ledger=[];logs={};train=[]
    for j in range(40):
        f=f'failure{j}';c=f'control{j}';log='physical_log_00000_00010';train.extend([f,c]);logs[f]=logs[c]=log
        source=dict(rank=0,collection=j+1,behavior_update=4*j,n_feasible=0,
            canonical_allzero=True,allfail_cost_tie=j%2==0,canonical_scores=[0.]*8,token=f,log=log,
            behavior_path='not-read-for-failure',behavior_sha256='provenance')
        ledger.append(source)
        path=tmp_path/f'{c}.pt';torch.save(dict(token=c,log=log,records=[dict(metrics=metric()) for _ in range(8)]),path)
        ledger.append(dict(source,token=c,n_feasible=8,canonical_allzero=False,allfail_cost_tie=False,
            canonical_scores=[.95]*8,behavior_path=str(path),behavior_sha256=P.sha(path)))
    lp=tmp_path/'groups.jsonl';lp.write_text(''.join(json.dumps(r)+'\n' for r in ledger))
    pop=dict(train_tokens=train,dev_tokens=[],navtest_tokens=['nav'],token_logs=dict(logs,nav='other_log_00000_00010'))
    cases,selection=P.select_cases(lp,pop)
    assert len(cases)==80 and selection['failure_union']==selection['controls']==40
    assert all(P.physical(logs[r['token']])==P.physical(logs[r['pair_token']]) for r in cases)
    # Tiny independent two-case mock summary: every shard must explicitly close.
    cases=cases[:2];out=tmp_path/'results';(out/'scenes').mkdir(parents=True)
    identity=P.sealed(dict(schema=P.SCHEMA,world=8,cases=cases,count=2,protocol=dict(sha256='protocol'),checkpoint='fixed10k'))
    P.atomic(out/'identity.json',identity)
    chains,draws,pairing=P.matched_chains(Toy(),dict(context=torch.ones(16,1,2)),'summary')
    for case in cases:
        token=case['token'];array=out/'scenes'/f'{token}.npz';requested=np.zeros((64,8,3),dtype=np.float32)
        np.savez(array,**{k+'_chain':v.numpy() for k,v in chains.items()},roots=draws['A_late'][0].numpy(),
                 innovations=np.stack([v.numpy() for v in draws['A_late'][1:]]),requested=requested)
        records=[dict(status='OK',token=token,log=case['log'],protocol_sha256='protocol',
            cache=dict(sha256='cache'),metrics=metric(),requested_trajectory=requested[0].tolist()) for _ in range(16)]
        row=P.sealed(dict(schema=P.SCHEMA,identity=identity['sha256'],status='OK',**case,pairing=pairing,
            strategies={k:dict(records=records) for k in P.STRATEGIES},diagnostic_B1_from_ODE16_first=records[0],
            arrays=dict(path='scenes/'+array.name,sha256=P.sha(array)),
            costs=dict(new_generated_candidates=64,new_scored_candidates=64,extra_B1_candidates=0,Qwen_prefills=1,current_camera_frames=3,actor_velocity_batch_calls=40)))
        P.atomic(out/'scenes'/f'{token}.json',row)
    for rank in range(8):
        ts=[c['token'] for c in cases[rank::8]]
        P.atomic(out/f'rank{rank}.json',dict(status='COMPLETE',identity=identity['sha256'],expected_tokens=ts,completed_tokens=ts))
    result=P.summarize(out)
    assert result['status']=='COMPLETE' and result['count']==2 and result['costs']['new_scored_candidates']==128
    assert result['cohorts']['historical_failure']['A_late']['strict_quality_count']==16
    assert result['cohorts']['historical_failure']['A_late']['quality_when_B1_not_quality']==0
    assert not result['scenes'][0]['strategies']['A_late']['Safety_credit16']['Safety_raw_PG_active']
    assert P.describe([dict(metrics=dict(metric(),comfort=0.,score=5.0/6))])['strict_quality_count']==0
    assert P.describe([dict(metrics=dict(metric(),driving_direction_compliance=0.))])['strict_quality_count']==1
    (out/'SUMMARY.json').unlink()
    state=json.loads((out/'rank0.json').read_text());state['status']='RUNNING';P.atomic(out/'rank0.json',state)
    with pytest.raises(ValueError):P.summarize(out)
    assert not (out/'SUMMARY.json').exists()
