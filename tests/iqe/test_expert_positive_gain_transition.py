from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from iqe.io import atomic_json, atomic_torch, digest, file_hash, read_json
from iqe.internal_baseline.bank import COMPONENTS, failure_and_target_audit
from iqe.internal_baseline.expert_data_revision import ARTIFACTS, artifact, training_root
from iqe.internal_baseline.expert_data_transition import (PRESERVED_FIELDS, converted_state,
    positive_gain_artifacts, revise_positive_gain, state_hash, replay_expert_history)
from iqe.internal_baseline.sampling import ExpertSampler


def label(v):
    return {'score_valid': True, 'total_score_01': v, 'named_components': {c:1. for c in COMPONENTS},
            'component_valid_masks': {c:True for c in COMPONENTS}}


@pytest.mark.parametrize('gain,expected', [(0.,False),(-.0001,False),(1e-12,True),(.02,True),(.05,True)])
def test_only_strict_positive_gain(gain, expected):
    assert failure_and_target_audit(label(.81),label(.81+gain),failure_threshold=.85,minimum_gt_gain=0.)['eligible'] is expected


@pytest.mark.parametrize('base,gt', [(.86,.95),(.70,.79)])
def test_other_gates_retained(base,gt):
    assert not failure_and_target_audit(label(base),label(gt),failure_threshold=.85,minimum_gt_gain=0.)['eligible']


def fixture_manifest():
    pairs = {'h':(.70,.90),'small':(.81,.82),'tie':(.81,.81),'bad':(.70,.79),
             'hv':(.71,.95),'smallv':(.81,.83),'n':(1.,1.)}
    audits = {sid:{'Base':label(a),'GT':label(b),'input_consistency':'verified','target_hash':'gt-'+sid,
                   'reasons':[]} for sid,(a,b) in pairs.items()}
    m = {'hard':['h'],'ordinary':['n'],'hard_val':['hv'],'failure_val':['hv','smallv'],
         'gt_audit':audits,'base_failure_threshold':.85,'gt_minimum':.8,'target_gain_minimum':.05,
         'sampling':{'hard':.8,'ordinary':.2},'seed':31}
    m['hash']=digest(m)
    s = {'F_train':['h','small','tie','bad'],'H_train':['h'],'F_val':['hv','smallv'],'H_val':['hv'],
         'rules':{'high_quality':.8,'base_failure_threshold':.85,'target_gain':.05},'defined_before_expert_training':True}
    return m,s


def test_selection_reuses_labels_and_ordinary_without_relaxing_quality():
    m,s=fixture_manifest(); before=deepcopy(m)
    a=positive_gain_artifacts(m,s,2)
    assert a['EXPERT_MANIFEST.json']['hard']==['h','small']
    assert a['EXPERT_MANIFEST.json']['ordinary']==['n']
    assert a['EXPERT_MANIFEST.json']['hard_val']==['hv','smallv']
    assert a['HARD_SETS_BEFORE_EXPERT.json']['defined_before_expert_training'] is False
    assert m==before


def fixture_run(tmp_path, monkeypatch):
    from iqe.internal_baseline import training, budget
    from iqe.training.checkpoint import capture_rng, environment
    m,s=fixture_manifest()
    cfg=SimpleNamespace(root=tmp_path,experiment_id='test_transition')
    tc={'lr':1e-4,'weight_decay':.01,'global_batch_size':128,'micro_batch_size':32,
        'max_optimizer_steps':10,'warmup_steps':1,'lr_min_ratio':.1,'schedule':'warmup_constant',
        'early_stop_patience':1000000000,'validation_every_steps':2,'checkpoint_every_steps':2,'grad_clip_norm':1.}
    monkeypatch.setattr(training,'training_config',lambda c,k:tc)
    monkeypatch.setattr(budget,'training_dependencies',lambda c:{})
    atomic_json(tmp_path/'BASE_BINDING.json',{'Base':'fixed'})
    atomic_json(tmp_path/'DATA_PARTITION.json',{'D_E':['h','small','tie','bad','n'],'D_S':['unused']})
    atomic_json(tmp_path/'runtime/HOLD_HEAD_TRAINING.json',{'active':True})
    folder=tmp_path/'expert_data_revisions'/'parent'
    arts={'EXPERT_MANIFEST.json':m,'EXPERT_TRAIN_MANIFEST.json':m,
          'HARD_SETS_BEFORE_EXPERT.json':s,'TARGET_AUDIT_QUEUE.json':{}}
    for name,value in arts.items():atomic_json(folder/name,value)
    selected={'directory':str(folder),'base_binding_hash':file_hash(tmp_path/'BASE_BINDING.json'),
              'partition_hash':file_hash(tmp_path/'DATA_PARTITION.json'),
              'artifacts':{n:file_hash(folder/n) for n in ARTIFACTS}}
    atomic_json(tmp_path/'SELECTED_EXPERT_DATA.json',selected)
    sampler=ExpertSampler(m['hard'],m['ordinary'],m['seed'],m['hash'],.8)
    module=torch.nn.Linear(1,1); optimizer=torch.optim.AdamW(module.parameters(),lr=1e-4)
    scheduler=training.train.__globals__['scheduler_factory'](optimizer,tc)
    consumed=[]; rows=[]
    for step in range(1,3):
        batch=sampler.peek(128);ids=[e['scene_id'] for e in batch]; consumed+=ids;sampler.consume(128)
        optimizer.zero_grad();module(torch.ones(1,1)).sum().backward();optimizer.step();scheduler.step()
        rows.append({'optimizer_step':step,'sample_ids':ids})
    root=tmp_path/'expert'; root.mkdir()
    (root/'steps.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    deps={'experiment_id':cfg.experiment_id,'base':selected['base_binding_hash'],
          'manifest':file_hash(folder/'EXPERT_MANIFEST.json'),'kind':'own_expert_1'}
    source={'schema_version':1,'module':module.state_dict(),'optimizer':optimizer.state_dict(),
            'scheduler':scheduler.state_dict(),'scaler':None,'optimizer_step':2,'sampler':sampler.state_dict(),
            'consumed_sample_ids':consumed,'rng_by_rank':[capture_rng()],'world_size':1,'global_batch':128,
            'accumulation':4,'dependencies':deps,'resolved_config':tc,'environment':environment(),
            'resume_boundary':'optimizer_step','trainer_state':{'best':(.1,.2),'bad_cycles':1,'best_snapshot':None,
               'termination':{'termination_reason':'OPERATOR_PAUSED','terminal':False,'stopped_at_step':2,
                              'early_stop_patience':1000000000,'best_snapshot':None}}}
    p=root/'step_000002.pt'; atomic_torch(p,source)
    atomic_json(str(p)+'.COMPLETE.json',{'checksum':file_hash(p),'optimizer_step':2,'world_size':1,'dependencies':deps})
    atomic_json(root/'result.json',{'checkpoint':str(p),'status':'STOPPED_AT_BOUNDARY'})
    return cfg,p,source,rows


def test_transaction_preserves_full_state_and_is_idempotent(tmp_path,monkeypatch):
    cfg,parent,source,rows=fixture_run(tmp_path,monkeypatch); old_hash=file_hash(parent)
    selected=revise_positive_gain(cfg,'positive',parent)
    destination=Path(selected['training_root']); saved=torch.load(destination/parent.name,weights_only=False)
    assert file_hash(parent)==old_hash
    for name in PRESERVED_FIELDS: assert state_hash(saved[name])==state_hash(source[name])
    assert saved['sampler']['identity']!=source['sampler']['identity']
    assert saved['sampler']['rng']==source['sampler']['rng'] and saved['sampler']['consumed']==256
    assert saved['trainer_state']['best'] is None
    assert training_root(cfg)==destination and artifact(cfg,'EXPERT_MANIFEST.json')==Path(selected['directory'])/'EXPERT_MANIFEST.json'
    assert revise_positive_gain(cfg,'positive',parent)==selected
    from iqe.training.checkpoint import load_checkpoint
    from iqe.training.trainer import scheduler_factory
    m=torch.nn.Linear(1,1);opt=torch.optim.AdamW(m.parameters(),lr=1e-4);sched=scheduler_factory(opt,source['resolved_config'])
    manifest=read_json(artifact(cfg,'EXPERT_MANIFEST.json'))
    sampler=ExpertSampler(manifest['hard'],manifest['ordinary'],manifest['seed'],manifest['hash'],.8)
    restored=load_checkpoint(destination/parent.name,m,opt,sched,None,sampler,saved['dependencies'],
                             source['resolved_config'],global_batch=128,accumulation=4)
    assert restored['step']==2 and restored['data_distribution_changed'] and restored['resume_mode']=='STATE_PRESERVING_DATA_TRANSITION'
    assert state_hash(opt.state_dict())==state_hash(source['optimizer'])
    entries=sampler.peek(128);rows.append({'optimizer_step':3,'sample_ids':[x['scene_id'] for x in entries]})
    from iqe.internal_baseline.audit_training import replay_consumption
    counts=replay_expert_history(cfg,manifest,rows,replay_consumption)
    assert sum(sum(x.values()) for x in counts.values())==384


def test_recover_commit_before_pointer_window(tmp_path,monkeypatch):
    cfg,parent,source,rows=fixture_run(tmp_path,monkeypatch)
    from iqe.internal_baseline import expert_data_transition as mod
    real=mod.atomic_json
    def failing(path,value,**kw):
        if Path(path)==tmp_path/'SELECTED_EXPERT_DATA.json':raise RuntimeError('injected pointer interruption')
        return real(path,value,**kw)
    monkeypatch.setattr(mod,'atomic_json',failing)
    with pytest.raises(RuntimeError,match='pointer interruption'):revise_positive_gain(cfg,'positive',parent)
    assert 'training_root' not in read_json(tmp_path/'SELECTED_EXPERT_DATA.json')
    monkeypatch.setattr(mod,'atomic_json',real)
    selected=revise_positive_gain(cfg,'positive',parent)
    assert training_root(cfg)==Path(selected['training_root'])


def test_transition_rejects_hold_off_and_corrupt_parent(tmp_path,monkeypatch):
    cfg,parent,source,rows=fixture_run(tmp_path,monkeypatch)
    atomic_json(tmp_path/'runtime/HOLD_HEAD_TRAINING.json',{'active':False})
    with pytest.raises(ValueError,match='hold required'):revise_positive_gain(cfg,'positive',parent)
    atomic_json(tmp_path/'runtime/HOLD_HEAD_TRAINING.json',{'active':True})
    parent.write_bytes(b'incomplete')
    with pytest.raises(ValueError,match='checksum'):revise_positive_gain(cfg,'positive',parent)
