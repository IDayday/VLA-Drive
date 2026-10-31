"""Directed CPU LR clock and real native restore tests; no policy/model load."""
import copy
import json
import random
from pathlib import Path
from types import SimpleNamespace
import warnings

import numpy as np
import pytest
import torch

from starVLA.rl.qwen_world_research_algorithms import safety_lr as lr
from starVLA.rl.qwen_world_research_algorithms.policy import ResearchLoss
from starVLA.rl.qwen_world_research_algorithms.safety_efficiency import (
    STRUCTURED_SCHEMA,validate_safety_efficiency_boundary,
)
from starVLA.rl.qwen_world_postcot.scope_contracts import expected_scope_recipe,MEMORY
from starVLA.rl.simwam_recipe import checkpoint
from starVLA.rl.simwam_recipe.lora_adapter import adapter_state
from starVLA.rl.simwam_recipe.vendor.trainer_math import SimWAMGRPOTrainer


class Actor(torch.nn.Module):
    def __init__(self):
        super().__init__();self.a=torch.nn.Parameter(torch.tensor([.25,-.4]));self.b=torch.nn.Parameter(torch.tensor([.8]));self.device=torch.device('cpu')


def optimizer(actor):
    return torch.optim.AdamW([dict(params=[actor.a],name=lr.GROUP_NAMES[0],lr=5e-5),dict(params=[actor.b],name=lr.GROUP_NAMES[1],lr=5e-6)],lr=5e-5,betas=(.9,.95))


def parent_recipe():
    cfg=expected_scope_recipe(MEMORY);cfg.update(max_steps=50000,lr_warmup_steps=100,learning_rate=5e-5,memory_learning_rate=5e-6,lr_scheduler_type='constant')
    cfg['research']=dict(arm='PPO_SAFETY_EP',structured_advantage=STRUCTURED_SCHEMA,policy_objective='ppo',data_policy='uniform_full')
    return cfg


def fork_recipe(parent,arm='PPO_SAFETY_LRLOW',cp='checkpoint'):
    cfg=copy.deepcopy(parent);cfg['research'].update(arm=arm,lr_schedule=lr.expected_schedule(arm),native_fork=dict(
        schema=lr.FORK_SCHEMA,checkpoint=str(cp),expected_updates=10000,checkpoint_files_sha256='f'*64,
        checkpoint_identity_sha256='e'*64,optimizer_file_sha256='d'*64,expected_parent_model_sha256='c'*64,
        optimizer='inherited_AdamW',rng='inherited_all_native_per_rank',reference='original_frozen_IL70'))
    cfg['lr_scheduler_type']=cfg['research']['lr_schedule']['kind']
    if arm in lr.MIX_ARMS:cfg['research'].update(data_policy='static_rescue_retention',manifest_sha256=lr.MIX_MANIFEST_SHA,sampling_distribution='full80_low10_high10_static_v1')
    return cfg


@pytest.mark.parametrize('arm',lr.ARMS)
def test_relative_clock_group_ratio_floor_and_exact_resume(arm):
    actor=Actor();opt=optimizer(actor);s=lr.RelativeSchedule(opt,lr.expected_schedule(arm));s.seek(10000)
    for u in (10000,12000,15000,18000,20000):
        s.seek(u);g=s.get_last_lr();assert g[0]/g[1]==pytest.approx(10)
    assert s.get_last_lr()==[v*lr.expected_schedule(arm)['end_factor'] for v in lr.BASE_LRS]
    s.seek(13999)
    for p in actor.parameters():p.grad=torch.full_like(p,.3)
    opt.step();s.step();oldopt=copy.deepcopy(opt.state_dict());oldsched=copy.deepcopy(s.state_dict())
    restored_actor=Actor();restored_opt=optimizer(restored_actor);rs=lr.RelativeSchedule(restored_opt,lr.expected_schedule(arm))
    restored_opt.load_state_dict(oldopt);rs.load_state_dict(oldsched);assert rs.last_epoch==14000 and rs.get_last_lr()==s.get_last_lr()
    for p,q in zip(actor.parameters(),restored_actor.parameters()):q.data.copy_(p);p.grad=torch.full_like(p,.2);q.grad=torch.full_like(q,.2)
    opt.step();s.step();restored_opt.step();rs.step()
    assert all(torch.equal(p,q) for p,q in zip(actor.parameters(),restored_actor.parameters()))
    assert rs.state_dict()==s.state_dict()
    corrupt=copy.deepcopy(oldsched);corrupt['spec']['end_factor']=.3
    with pytest.raises(ValueError):rs.load_state_dict(corrupt)


def test_recipe_rejects_extra_changes_and_loss_override_is_scheduler_only():
    parent=dict(recipe=parent_recipe());cfg=fork_recipe(parent['recipe']);lr.validate_fork_recipe(cfg,parent)
    for field,value in [('seed',43),('memory_learning_rate',2e-6),('max_steps',20000)]:
        bad=copy.deepcopy(cfg);bad[field]=value
        with pytest.raises(ValueError):lr.validate_fork_recipe(bad,parent)
    bad=copy.deepcopy(cfg);bad['grpo']['sample']['noise_level']=.03
    with pytest.raises(ValueError):lr.validate_fork_recipe(bad,parent)
    assert lr.ScheduledSafetyLoss._policy_loss is ResearchLoss._policy_loss
    loss=lr.ScheduledSafetyLoss(cfg);loss.optimizer=optimizer(Actor());assert isinstance(loss._build_scheduler('constant',50000,100),lr.RelativeSchedule)
    with pytest.raises(ValueError):loss._build_scheduler('constant',5000,100)


def test_real_native_parent_restore_preserves_moments_rng_and_global_cursor(tmp_path,monkeypatch):
    """Native file IO/validator/Adam+RNG restore is real;8-rank control gather is
    represented by eight equal committed summaries,not eight GPU processes.
    """
    random.seed(19);np.random.seed(19);torch.manual_seed(19)
    actor=Actor();opt=optimizer(actor);original=SimpleNamespace(optimizer=opt,learning_rate=5e-5)
    sched=SimWAMGRPOTrainer._build_scheduler(original,'constant',50000,100)
    for _ in range(3):
        for p in actor.parameters():p.grad=torch.full_like(p,.3)
        opt.step();sched.step()
    # Synthetic committed10k fixture: moments contain real nonzero Adam state.
    for state in opt.state.values():state['step'].fill_(10000)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for _ in range(9997):sched.step()
    oldopt=copy.deepcopy(opt.state_dict());oldparams=copy.deepcopy(adapter_state(actor));generator=torch.Generator().manual_seed(991)
    state=dict(updates=10000,collections=2500,next_inner=0,phase='READY',scene_cursor=20000,raw_scene_exposures=20000,new_candidates=160000)
    parent=dict(sha256='parent-run',world=8,recipe=parent_recipe(),IL_parent=dict(model_sha256='c'*64),parameter_contract={'trainable_names':['a','b']})
    summary=validate_safety_efficiency_boundary(state,None,parent,opt,sched)
    cp=tmp_path/'checkpoint';cp.mkdir()
    torch.save(dict(adapters=oldparams,optimizer=oldopt,scheduler=sched.state_dict()),cp/'optimizer.pt')
    rng=dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.random.get_rng_state(),cuda=None,rollout_generator=generator.get_state())
    torch.save(dict(state=state,rollout=None,identity_sha256=parent['sha256'],rng=rng),cp/'rank0.pt')
    (cp/'identity.json').write_text(json.dumps(parent));(cp/'transaction.json').write_text(json.dumps(dict(ranks=[summary]*8)))
    files={p.name:lr.file_sha(p) for p in cp.iterdir()};(cp/'files.json').write_text(json.dumps(files))
    (cp/'COMPLETE').write_text(json.dumps(dict(schema=2,world=8,updates=10000,next_inner=0,phase='READY',files_sha256=lr.file_sha(cp/'files.json'))))
    cfg=fork_recipe(parent['recipe'],cp=cp);identity=dict(parent,sha256='new-scientific-run',recipe=cfg)
    fresh=Actor();newopt=optimizer(fresh);newsched=lr.RelativeSchedule(newopt,cfg['research']['lr_schedule']);newgen=torch.Generator().manual_seed(42)
    class Runtime(lr.SafetyLRRuntimeMixin):pass
    runtime=Runtime();runtime.cfg=cfg;runtime.native_parent_identity=parent;runtime.native_fork_checkpoint=cp
    monkeypatch.setattr(checkpoint,'dist',SimpleNamespace(is_initialized=lambda:True,get_rank=lambda:0,get_world_size=lambda:8))
    monkeypatch.setattr(checkpoint,'gather',lambda s:[copy.deepcopy(s) for _ in range(8)])
    random.random();np.random.random();torch.rand(4);torch.rand(4,generator=newgen)
    got,rollout=runtime.initialize_training_state(fresh,newopt,newsched,newgen,identity)
    assert got==state and rollout is None and newsched.last_epoch==10000
    assert newsched.get_last_lr()==[v*.2 for v in lr.BASE_LRS]
    assert random.getstate()==rng['python']
    current=np.random.get_state();assert current[0]==rng['numpy'][0] and np.array_equal(current[1],rng['numpy'][1]) and current[2:]==rng['numpy'][2:]
    assert torch.equal(torch.random.get_rng_state(),rng['torch']) and torch.equal(newgen.get_state(),rng['rollout_generator'])
    assert all(torch.equal(p,oldparams[n]) for n,p in fresh.named_parameters())
    for i,s in newopt.state_dict()['state'].items():
        for k,v in s.items():assert torch.equal(v,oldopt['state'][i][k])
    # Native own-resume state validates exact group LRs and global10k; no reset.
    runtime.validate_boundary(got,None,identity,newopt,newsched)
    altered=copy.deepcopy(newopt.state_dict());altered['param_groups'][1]['lr']=5e-6
    with pytest.raises(ValueError):runtime.validate_boundary(got,None,identity,altered,newsched)
