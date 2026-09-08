"""Semantic regressions: first run unchanged at 6e1d9f8, never mock the function under test."""
import copy
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from scipy.interpolate import CubicSpline

from navsim.agents.EpisodeDrive.planreg_v2.action import V2ActionDecoder, scorer_config
from navsim.agents.EpisodeDrive.planreg_v2.targets import long_target
from navsim.agents.EpisodeDrive.planreg_v2.normalizers import measured_statistics
from navsim.agents.EpisodeDrive.planreg_v2.optimizer import build_optimizer
from navsim.agents.EpisodeDrive.planreg_v2.runtime import load_config
from navsim.agents.EpisodeDrive.planreg_v2.checkpoint import warm_start_v1
from scripts.audit_drivor_scorer_parity import (load_and_verify_upstream_sources,
    load_upstream_classes, _forward_scorer_path, _upstream_pdm_score)


def normalizer():
    return measured_statistics([(str(i), torch.randn(8,3), torch.ones(8,dtype=torch.bool))
                                for i in range(3)], 'train', 'synthetic-review-unit')


@pytest.mark.parametrize('seed,n,padded', [(0,16,False),(13,64,False),(3,16,True),(9,64,True)])
def test_r1_actual_v2_score_against_pinned_upstream(seed,n,padded):
    torch.manual_seed(seed)
    action = V2ActionDecoder(normalizer()).eval()
    decoder_cls, scorer_cls = load_upstream_classes(load_and_verify_upstream_sources(Path('/mnt/project/external/DrivoR')))
    reference = decoder_cls(4,256,.1,.2,action.config).eval()
    heads = scorer_cls(action.config).eval()
    reference.load_state_dict(action.scorer_attention.state_dict(),strict=True)
    heads.load_state_dict(action.scorer.state_dict(),strict=True)
    proposals = torch.randn(2,64,8,3,requires_grad=True)
    memory = torch.randn(2,n,256,requires_grad=True)
    ego = torch.randn(2,256,requires_grad=True)
    valid = torch.ones(2,n,dtype=torch.bool)
    if padded:
        valid[0,n//2:] = False
        valid[1,-3:] = False
    logits,scores = action.score(proposals,memory,valid,ego)
    # Upstream is executed from the verified fixed git objects, including its
    # decoder/heads and verbatim model aggregation. Per-sample trim for padding.
    if padded:
        refs = [_forward_scorer_path(action.pos_embed,reference,heads,proposals[i:i+1],
            memory[i:i+1,valid[i]],ego[i:i+1,None])[0] for i in range(2)]
        expected = {k:torch.cat([r[k] for r in refs]) for k in refs[0]}
    else:
        expected = _forward_scorer_path(action.pos_embed,reference,heads,proposals,memory,ego[:,None])[0]
    # Fixed a-priori CPU FP32 budget for masked-vs-trimmed MHA, not adjusted at runtime.
    for name in expected:
        torch.testing.assert_close(logits[name],expected[name],atol=2e-6,rtol=1e-5)
    reference_scores = _upstream_pdm_score(expected,action.config)
    torch.testing.assert_close(scores,reference_scores,atol=2e-6,rtol=1e-5)
    assert torch.equal(scores.argmax(-1),reference_scores.argmax(-1))
    sum(x.square().mean() for x in logits.values()).backward()
    assert memory.grad.norm()>0 and ego.grad.norm()>0 and proposals.grad is None
    assert all(p.grad is None for p in action.attention.parameters())


@pytest.mark.parametrize('kind',['straight','acceleration','curve','jitter','wrap'])
def test_r2_progressive_long_cubic_reference(kind):
    t=np.arange(11,dtype=np.float64)*.5
    if kind=='jitter': t[1:]+=np.sin(np.arange(1,11))*.009
    p=np.stack([10*t,np.zeros(11),np.zeros(11)],-1)
    if kind in ('acceleration','curve','jitter'):
        p[:,0]=2*t+t**2+.02*t**3
        p[:,1]=np.sin(t*.7)+t**3*.03
        p[:,2]=t*.12
    if kind=='wrap': p[:,2]=np.arctan2(np.sin(3+t*.3),np.cos(3+t*.3))
    # Independent V1 index-domain reference, future nodes only. The explicit
    # query array does not call a production mapping/interpolation helper.
    q=np.array([1/18,7/6,7/3,32/9,29/6,37/6,68/9,9.])
    times=np.interp(q,np.arange(10),t[1:])
    unwrapped=p[1:].copy();unwrapped[:,2]=np.unwrap(unwrapped[:,2])
    expected=CubicSpline(t[1:],unwrapped,bc_type='not-a-knot',extrapolate=False)(times)
    expected[:,2]=np.arctan2(np.sin(expected[:,2]),np.cos(expected[:,2]))
    actual,valid=long_target(p,t,np.ones(11,bool))
    assert valid
    np.testing.assert_allclose(actual.numpy(),expected,atol=4e-6,rtol=1e-6)


class GroupAgent(torch.nn.Module):
    def __init__(self,config):
        super().__init__(); self.config=config
        self.scene_memory=torch.nn.Linear(3,4)
        self.wm_predictor=torch.nn.Linear(3,4)
        self.backbone=torch.nn.Module()
        self.backbone.task_queries=torch.nn.Linear(3,4)


@pytest.mark.parametrize('gb',[1,32,128])
def test_r3_resolved_yaml_optimizer_peak(monkeypatch,gb):
    monkeypatch.setenv('PLANREG_BASE_VLM_PATH','/unit/base')
    monkeypatch.setenv('PLANREG_V2_NORMALIZER','/unit/stats')
    cfg=load_config('navsim/planning/script/config/common/agent/planreg_wm_v2_base.yaml')
    cfg['global_batch']=gb
    opt,scheduler,summary=build_optimizer(GroupAgent(cfg),100,cfg['learning_rates'])
    for g in summary:
        semantic=g['name'].startswith(('semantic_qformer/','semantic_queries/'))
        expected=min((1e-4 if semantic else 2e-4)*np.sqrt(gb/32),1.5e-4 if semantic else 3e-4)
        assert g['lr']==pytest.approx(expected,rel=1e-12)
    assert all(g['lr']==pytest.approx(peak*.01) for g,peak in zip(opt.param_groups,scheduler.base_lrs))


def test_r4_actual_v2_backbone_register_initialization():
    # Full actual pretrained class construction; CPU avoids occupying a GPU.
    from navsim.agents.EpisodeDrive.planreg_v2.backbone import V2Backbone
    path=os.environ.get('PLANREG_BASE_VLM_PATH','/mnt/project/DriveVLA-M0-models/planreg-formal/InternVL3-2B-base-aligned')
    if not Path(path).is_dir(): pytest.skip('Real InternVL checkpoint unavailable')
    backbone=V2Backbone(path,device='cpu',gradient_checkpointing=False)
    registers=backbone.planning_register_adapter.planning_registers
    assert registers.dtype==torch.float32
    assert .018 < float(registers.std()) < .022


def test_r5_production_warm_start_decoder_mapping_and_nonzero_ego():
    from omegaconf import OmegaConf
    from navsim.agents.EpisodeDrive.action_decoder import ActionDecoder
    cfg=vars(scorer_config()).copy()
    cfg.update(full_history_status=False,num_scene_tokens=16,lidar_pc=[])
    cfg.update({k:([3] if k=='cam_f0' else []) for k in ['cam_f0','cam_l0','cam_l1','cam_l2','cam_r0','cam_r1','cam_r2','cam_b0']})
    old=ActionDecoder(OmegaConf.create(cfg)).eval()
    target=torch.nn.Module(); target.action_head=V2ActionDecoder(normalizer()).eval()
    target.world_model_enabled=False
    report=warm_start_v1(target,{'action_head.'+k:v for k,v in old.state_dict().items()})
    for name,value in old.trajectory_decoder.state_dict().items():
        torch.testing.assert_close(target.action_head.attention.state_dict()[name],value,atol=0,rtol=0)
    status=torch.tensor([[1.,0.,0.,0.,8.,-2.,1.,-.5]])
    raw=torch.cat((torch.zeros(1,3),status),-1)
    normalized=torch.cat((torch.zeros(1,3),target.action_head.ego_normalizer(status)),-1)
    torch.testing.assert_close(target.action_head.hist_encoding(normalized),old.hist_encoding(raw),atol=1e-6,rtol=1e-5)
