"""Production V2 execution-only regression; synthetic labels test plumbing, not PDM physics."""
import copy
from pathlib import Path
from types import SimpleNamespace, MethodType
import numpy as np
import pytest
import torch
from torch import nn
from navsim.agents.EpisodeDrive.planreg_v2.agent import PlanRegV2Agent
from navsim.agents.EpisodeDrive.planreg_v2.action import V2ActionDecoder
from navsim.agents.EpisodeDrive.planreg_v2.predictor import ActionCausalPredictor
from navsim.agents.EpisodeDrive.planreg_v2.normalizers import measured_statistics, MotionConditionNormalizer
from navsim.agents.EpisodeDrive.planreg_v2.timing import StepTiming
from navsim.agents.EpisodeDrive.planreg_v2.runtime import load_config, execution_settings
from navsim.agents.EpisodeDrive.layers.losses.episode_drive_loss import EpisodeDriveLoss


class Deferred:
    def __init__(self,fn,args,events):
        self.fn,self.args,self.events,self.cancelled=fn,args,events,False
    def result(self):
        self.events.append('resolve')
        return self.fn(*self.args)
    def cancel(self): self.cancelled=True


class Pool:
    def __init__(self,events): self.events,self.futures=events,[]
    def submit(self,fn,*args):
        self.events.append('submit')
        future=Deferred(fn,args,self.events);self.futures.append(future);return future


def labels(path,poses,test):
    assert test is False
    scores=np.ones((len(poses),7),dtype=np.float32)
    scores[:,4]=1/(1+np.exp(-poses[:,0,0]))
    return scores,None,None,None


def agent_fixture(monkeypatch,overlap):
    from navsim.agents.EpisodeDrive.score_module import compute_navsim_score
    monkeypatch.setattr(compute_navsim_score,'get_sub_score',labels)
    torch.manual_seed(77)
    agent=PlanRegV2Agent.__new__(PlanRegV2Agent);nn.Module.__init__(agent)
    agent.config=dict(scorer_processes=2,total_steps=21789,overlap_metric_target_with_ema=overlap)
    agent.backbone=SimpleNamespace(step_timing=StepTiming())
    agent.world_model_enabled=True
    agent.register_buffer('optimizer_updates',torch.tensor(1))
    normalizer=measured_statistics([('test',torch.randn(8,3),torch.ones(8,dtype=torch.bool))],'train','unit')
    agent.action_head=V2ActionDecoder(normalizer)
    agent.wm_predictor=ActionCausalPredictor()
    agent.motion_normalizer=MotionConditionNormalizer()
    agent.exact_loss=EpisodeDriveLoss()
    agent.events=[];agent._score_pool=Pool(agent.events)
    def teacher(self,features,predictions,include_current=False):
        self.events.append('teacher')
        return features['teacher'].detach()
    agent.encode_teacher=MethodType(teacher,agent)
    return agent


def batch(agent):
    torch.manual_seed(78)
    visual=torch.randn(2,16,256,requires_grad=True)
    semantic=torch.randn(2,16,256,requires_grad=True)
    valid=torch.ones(2,16,dtype=torch.bool)
    predictions=agent.action_head(visual,valid,torch.randn(2,8))
    predictions.update(visual_content=visual,semantic_queries=semantic,
        tile_geometry=torch.rand(2,16,5),scene_valid_mask=valid)
    features={'teacher':torch.randn(2,3,16,256,requires_grad=True)}
    targets=dict(metric_cache_path=['a','b'],trajectory=torch.randn(2,8,3),trajectory_valid=torch.ones(2,8,dtype=torch.bool),
        trajectory_long=torch.randn(2,8,3),trajectory_long_valid=torch.ones(2,dtype=torch.bool),
        motion_sequence=torch.randn(2,8,8),motion_timestamps=torch.arange(1,9)[None].expand(2,-1)*.5,
        motion_valid=torch.ones(2,8,dtype=torch.bool),future_valid_mask=torch.ones(2,3,dtype=torch.bool))
    return features,targets,predictions


def test_production_serial_overlap_full_loss_and_gradients_exact(monkeypatch):
    serial=agent_fixture(monkeypatch,False);overlap=agent_fixture(monkeypatch,True)
    sf,st,sp=batch(serial);af,at,ap=batch(overlap)
    sl=serial.compute_loss(sf,st,sp);al=overlap.compute_loss(af,at,ap)
    assert serial.events==['submit','submit','resolve','resolve','teacher']
    assert overlap.events==['submit','submit','teacher','resolve','resolve']
    assert sl.keys()==al.keys()
    for key in sl: torch.testing.assert_close(sl[key],al[key],rtol=0,atol=0)
    sl['loss'].backward();al['loss'].backward()
    for (sn,s),(an,a) in zip(serial.named_parameters(),overlap.named_parameters()):
        assert sn==an
        if s.grad is None: assert a.grad is None
        else: torch.testing.assert_close(s.grad,a.grad,rtol=0,atol=0)
    assert sf['teacher'].grad is None and af['teacher'].grad is None
    torch.testing.assert_close(sp['visual_content'].grad,ap['visual_content'].grad,rtol=0,atol=0)


def test_detached_coordinate_snapshot_and_candidate_order(monkeypatch):
    agent=agent_fixture(monkeypatch,True)
    proposals=torch.randn(2,64,8,3,requires_grad=True)
    expected=torch.tensor(np.stack([labels('a',p.detach().numpy(),False)[0] for p in proposals]))
    request=agent.submit_metric_targets({'metric_cache_path':['a','b']},proposals)
    with torch.no_grad(): proposals.add_(100)
    torch.testing.assert_close(agent.resolve_metric_targets(request),expected,rtol=0,atol=0)
    assert proposals.grad is None


def test_teacher_failure_cancels_all_pending_jobs(monkeypatch):
    agent=agent_fixture(monkeypatch,True)
    def fail(*args,**kwargs): raise RuntimeError('teacher failed')
    agent.encode_teacher=fail
    with pytest.raises(RuntimeError,match='teacher failed'):
        agent.metric_targets_with_teacher({}, {'metric_cache_path':['a','b']},
                                          {'proposals':torch.randn(2,64,8,3)},True)
    assert all(f.cancelled for f in agent._score_pool.futures)


def test_partial_submit_failure_cancels_accepted_job(monkeypatch):
    agent=agent_fixture(monkeypatch,True);original=agent._score_pool.submit
    def submit(*args):
        if agent._score_pool.futures:raise RuntimeError('queue failed')
        return original(*args)
    agent._score_pool.submit=submit
    with pytest.raises(RuntimeError,match='queue failed'):
        agent.submit_metric_targets({'metric_cache_path':['a','b']},torch.randn(2,64,8,3))
    assert agent._score_pool.futures[0].cancelled


def test_no_wm_does_not_call_teacher(monkeypatch):
    agent=agent_fixture(monkeypatch,True)
    scores,teacher=agent.metric_targets_with_teacher({}, {'metric_cache_path':['a','b']},
        {'proposals':torch.randn(2,64,8,3)},False)
    assert teacher is None and 'teacher' not in agent.events and scores.shape==(2,64,7)


def test_acceleration_configs_preserve_algorithm_lr_and_shared_identity(monkeypatch):
    from navsim.agents.EpisodeDrive.planreg_v2.initialization import initialization_identity
    for key in ('PLANREG_BASE_VLM_PATH','PLANREG_VQA_VLM_PATH','PLANREG_V2_NORMALIZER','PLANREG_V2_SHARED_INIT'):
        monkeypatch.setenv(key,'unit')
    root=Path(__file__).resolve().parents[1]/'navsim/planning/script/config/common/agent'
    legacy=load_config(root/'planreg_wm_v2_base.yaml')
    fast=load_config(root/'planreg_wm_v2_base_fast.yaml')
    ckpt=load_config(root/'planreg_wm_v2_base_fast_checkpointed.yaml')
    vqa=load_config(root/'planreg_wm_v2_vqa_fast.yaml')
    assert initialization_identity(legacy)==initialization_identity(fast)==initialization_identity(vqa)
    allowed=set(execution_settings({}))
    for key in set(legacy)|set(fast):
        if key not in allowed: assert legacy.get(key)==fast.get(key)
    assert execution_settings(legacy)==dict(read_only_attention_backend='eager',language_attention_backend='eager',gradient_checkpointing=True,overlap_metric_target_with_ema=False)
    assert execution_settings(fast)==dict(read_only_attention_backend='split_sdpa',language_attention_backend='eager',gradient_checkpointing=False,overlap_metric_target_with_ema=True)
    assert execution_settings(ckpt)['gradient_checkpointing'] is True
    assert {k:v for k,v in fast.items() if k!='variant'}=={k:v for k,v in vqa.items() if k!='variant'}
