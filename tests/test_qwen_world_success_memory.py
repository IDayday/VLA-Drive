"""Directed CPU tests for genuinely independent historical success supervision."""
import copy
import json
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from starVLA.model.modules.qwen_sourceflow.action_codec import ActionCodec
from starVLA.rl.qwen_world_postcot.scope_contracts import MEMORY,expected_scope_recipe
from starVLA.rl.qwen_world_research_algorithms import success_memory as sm
from starVLA.rl.qwen_world_research_algorithms.policy import ResearchLoss
from starVLA.rl.qwen_world_research_algorithms.success_memory_entry import SuccessMemoryRuntimeMixin


RUN='fresh-run';COND=dict(schema='world_postcot_trainable_memory_observation_v1',identity='current-only',repeats=8)


class Conditions:
    def get(self,token,device):return dict(COND,token=token)


def config():
    cfg=expected_scope_recipe(MEMORY);cfg.update(max_steps=50000,lr_warmup_steps=100,lr_scheduler_type='constant')
    cfg['research']=dict(arm=sm.ARM,structured_advantage='candidate_metric_safety_feasible_EP_global_std_v1',policy_objective='ppo',data_policy='uniform_full',success_memory=copy.deepcopy(sm.CONFIG))
    return cfg


def rollout(c,*,success=True,EP=.8):
    token=f'TRAIN{c}';chain=torch.linspace(-.1,.1,8*11*8*3).reshape(8,11,8,3);records=[]
    for i in range(8):
        clean=chain[i,-1].clone();requested=ActionCodec.denorm_odo(clean).double().tolist();ep=EP if success else 0.;dac=1 if success else 0
        score=dac*(5*ep+5+2)/12
        metrics=dict(score=score,no_at_fault_collisions=1.,drivable_area_compliance=dac,time_to_collision_within_bound=1.,ego_progress=ep,comfort=1.,driving_direction_compliance=1.)
        records.append(dict(token=token,log='log',status='OK',metrics=metrics,readouts=dict(canonical_PDMS=score),requested_trajectory=requested,raw_progress_meters=10+i,protocol_sha256='p',cache=dict(sha256='cache')))
    return dict(token=token,log='log',chain=chain,records=records,cond_bg=dict(COND,token=token),protocol_sha256='p',behavior_version=dict(run=RUN,update=(c-1)*4,adapter_sha256=f'theta{c}'))


def allowed():return {f'TRAIN{i}':'log' for i in range(1,100)}


def populated():
    m=sm.new_memory(RUN,0);p=None
    for c in range(1,5):
        p=rollout(c);sm.attach_history(m,p,Conditions(),torch.device('cpu'),allowed())
    return m,p


def test_empty_bank_becomes_real_independent_historical_aux_without_main_rng_or_PG_changes():
    cfg=config();sm.validate_recipe(cfg);m=sm.new_memory(RUN,0);torch.manual_seed(47);global_rng=torch.get_rng_state().clone()
    main=torch.Generator().manual_seed(42);main_rng=main.get_state().clone();selected=[]
    for c in range(1,9):
        p=rollout(c,success=True,EP=1);before=copy.deepcopy(p)
        sm.attach_history(m,p,Conditions(),torch.device('cpu'),allowed())
        assert torch.equal(p['chain'],before['chain']) and p['records']==before['records']
        if p['success_memory_aux']['enabled']:
            aux=p['success_memory_aux'];selected.append(c)
            assert aux['teacher']['token']!=p['token'] and aux['teacher']['source_update']<p['behavior_version']['update']
            # Fresh saturation does not disable historical retention.
            assert aux['teacher']['metric_verified'] and aux['mc_samples']==1
    assert selected==[4,8] and m['aux_selected_collections']==2
    assert len(m['entries'])==8 and len(set(m['aux_selection_counts']))>=1
    assert torch.equal(global_rng,torch.get_rng_state()) and torch.equal(main_rng,main.get_state())
    bad=config();bad['research']['warmstart']={'checkpoint':'oldSafety10k'}
    with pytest.raises(ValueError):sm.validate_recipe(bad)


def test_partial_inner_native_state_serialization_preserves_bank_rng_pending_and_next_selections(tmp_path):
    m,p=populated();state=dict(updates=14,collections=4,next_inner=2,phase='INNER_COMMITTED',success_memory=m)
    before=sm.memory_summary(state,p,RUN,0,allowed(),full=True);original_memory=copy.deepcopy(m)
    assert before['completed_aux_optimizer_updates']==2 and before['completed_distinct_aux_scenes']==1
    path=tmp_path/'rank0.pt';torch.save(dict(state=state,rollout=p),path);own=torch.load(path,map_location='cpu',weights_only=False)
    assert sm.memory_summary(own['state'],own['rollout'],RUN,0,allowed(),full=True)==before
    for c in range(5,9):
        a,b=rollout(c),rollout(c);sm.attach_history(m,a,Conditions(),torch.device('cpu'),allowed());sm.attach_history(own['state']['success_memory'],b,Conditions(),torch.device('cpu'),allowed())
        assert a['success_memory_aux']['material_sha256']==b['success_memory_aux']['material_sha256']
    assert torch.equal(m['rng_state'],own['state']['success_memory']['rng_state'])
    corrupt=copy.deepcopy(p);corrupt['success_memory_aux']['noise'][0,0,0]+=1
    with pytest.raises(ValueError):sm.validate_payload(corrupt['success_memory_aux'],original_memory,allowed())
    with pytest.raises(ValueError):sm.memory_summary({**state,'success_memory':{}},p,RUN,0,allowed())


def test_verified_admission_keeps_low_EP_stop_unlabelled_as_failure_and_refuses_bad_scored_endpoint():
    m=sm.new_memory(RUN,0);p=rollout(1,EP=.1);sm.attach_history(m,p,Conditions(),torch.device('cpu'),allowed())
    assert not m['entries'] and m['below_EP_threshold_groups']==1 and m['no_strict_success_groups']==0
    p=rollout(2);p['records'][0]['requested_trajectory'][0][0]+=100
    # Force candidate0 to win the otherwise equal successful group.
    p['records'][0]['raw_progress_meters']=1000
    with pytest.raises(ValueError):sm.verified_winner(p,RUN,allowed())
    p=rollout(2);p['records'][0]['metrics']['no_at_fault_collisions']=.5
    # The actual winner is another strict candidate,never the unsafe one.
    p['records'][0]['metrics']['score']*=.5;p['records'][0]['readouts']['canonical_PDMS']*=.5
    entry,_=sm.verified_winner(p,RUN,allowed());assert entry['candidate_index']!=0



def test_four_inner_CFM_rebuilds_current_memory_and_saved_material_without_retained_graph(monkeypatch):
    m,p=populated();cfg=config();cfg['research']['gradient_probe']={'schema':'root_probe_no_enabled_flag'};loss=sm.SuccessMemoryLoss(cfg)
    from starVLA.rl.qwen_world_research_algorithms import gradient_probe
    calls=[]
    def probe(model,rollout,cfg,components):
        assert set(components)=={'pg','anchor','aux'} and all(v.requires_grad for v in components.values())
        calls.append(1);return {'probe_witness':next(model.parameters()).detach()*0+1}
    monkeypatch.setattr(gradient_probe,'maybe_probe_components',probe)
    def base(self,model,rollout):
        pg=(model.action-model.memory).square();anchor=model.memory.square()*.2
        self.record_policy_components(pg,anchor);return pg+anchor,{}
    monkeypatch.setattr(ResearchLoss,'_policy_loss',base)
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__();self.action=torch.nn.Parameter(torch.tensor(.1));self.memory=torch.nn.Parameter(torch.tensor(.2));self.ref=torch.tensor(.4);self.memory_builds=0
        @contextmanager
        def _context_call(self,descriptor):yield dict(context=torch.zeros(8,1,1),valid=torch.ones(8,1,dtype=torch.bool),ego=torch.zeros(8,1))
        def action_velocity(self,state,timestep,c):
            assert c['context'].shape[0]==1;self.memory_builds+=1
            return self.action*state+self.memory
    model=Model();opt=torch.optim.SGD(model.parameters(),lr=.01);digest=p['success_memory_aux']['material_sha256'];rng=m['rng_state'].clone()
    for _ in range(4):
        opt.zero_grad();total,metrics=loss._policy_loss(model,p);total.backward()
        assert model.action.grad is not None and model.memory.grad is not None and loss._live_components is None
        opt.step();assert p['success_memory_aux']['material_sha256']==digest
    assert model.memory_builds==4 and torch.equal(m['rng_state'],rng) and len(calls)==4 and 'probe_witness' in metrics


def test_frozen_two_scene_LRU_avoids_Qwen_reencoding_and_public_JSON_never_contains_bank_tensors():
    class Base:
        def __init__(self):self.token=None;self.cached=None;self.calls=0
        def materialize(self,d,device):
            if self.token!=d['token']:self.calls+=1;self.cached={'frozen_features':torch.tensor([self.calls])};self.token=d['token']
            return self.cached
    base=Base();cache=sm.FrozenFeatureLRU(base)
    for _ in range(4):
        for token in ('fresh','historical'):assert not cache.materialize({'token':token},'cpu')['frozen_features'].requires_grad
    assert base.calls==2
    m,p=populated();state=dict(updates=14,collections=4,next_inner=2,phase='INNER_COMMITTED',success_memory=m)
    class Runtime(SuccessMemoryRuntimeMixin):pass
    r=Runtime();r.pending_success_rollout=p;r.success_memory=m;r.allowed=allowed()
    public=r.public_training_state(state);json.dumps(public,allow_nan=False)
    assert 'success_memory' not in public and public['success_memory_summary']['completed_aux_optimizer_updates']==2
    assert state['success_memory'] is m and torch.is_tensor(m['rng_state'])
