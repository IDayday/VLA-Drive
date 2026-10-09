"""E1: independently fine-tune a complete copied policy on the fixed IQE manifest.

The locked S0 files never change. The visual tower retains the original freeze;
all other original policy parameters (including VLM language layers) may update.
This baseline uses online images and inherits the Query IL objective, without a
candidate pool, Scorer, auxiliary teacher objectives or distillation.
"""
from __future__ import annotations
from dataclasses import replace
from pathlib import Path
import os
import torch
from torch import nn
from ..contracts import require, strict_record, SceneRecord, CandidateRecord
from ..query_base import build_query_framework
from ..losses import expert_il_terms
from ..io import read_json, atomic_json, atomic_torch, digest, file_hash
from ..data.sources import load_observation, load_target
from ..data.candidate_bank import save_trajectory
from ..data.sampler import ConsumedSampler
from ..evaluation.selection import selection_report
from .trainer import train
from .checkpoint import configure_language_checkpointing


class PolicyCopy(nn.Module):
    def __init__(self, framework, contract):
        super().__init__()
        self.framework, self.contract = framework, contract

    def forward(self, observations):
        encoded = self.framework.encode_current(observations)
        bridge = self.framework.action_model
        features = bridge.encode_features(self.framework.build_planner_condition(encoded), bridge.ego_state,
                                          [o['token'] for o in observations], digest(self.contract))
        return bridge.expert(features)


def load_copy(pipeline, checkpoint=None):
    from omegaconf import OmegaConf
    c = pipeline.contract
    framework = build_query_framework(OmegaConf.create(c['source_config']), c['query_architecture'], c['source_root'], c['source_commit']).float()
    require(file_hash(c['query_checkpoint']) == c['query_checkpoint_hash'], 'locked S0 changed')
    framework.load_state_dict(torch.load(c['query_checkpoint'], map_location='cpu', weights_only=False, mmap=True)['model'], strict=True)
    framework.strip_auxiliary_heads()
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        configure_language_checkpointing(framework)
    copy = PolicyCopy(framework, c).to(pipeline.device)
    if checkpoint:
        copy.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True), strict=True)
    return copy


def evaluate_copy(pipeline, number, role, network=None):
    require(role in {'stage_val','dev_report'}, 'E1 evaluation only stage_val/dev_report')
    folder = pipeline.round_root(number) / 'policy_copy'
    network = network or load_copy(pipeline, folder / 'best.pt')
    network.eval()
    base = pipeline.bank_arrays(0, role)
    scores, components = [], {c:[] for c in base['components']}
    masks = {c:[] for c in components}
    cases = []
    losses = []
    from ..evaluation.retention import module_hash
    state_hash = module_hash(network)
    with torch.no_grad():
        for row,scene in enumerate(base['scenes']):
            pred = network([load_observation(scene, pipeline.contract)])
            ref = folder / 'evaluation' / role / state_hash / (scene.scene_id+'.npz')
            th = save_trajectory(ref,pred.physical[0].cpu().numpy())
            candidate = CandidateRecord(1,scene.scene_id,'policy_copy',state_hash,digest(pipeline.contract),th,str(ref),
                pipeline.trajectory_contract.raw_representation,'ego_relative_rear_axle',pipeline.trajectory_contract.horizon,
                pipeline.trajectory_contract.dt,bool(torch.isfinite(pred.physical).all()),pipeline.contract['iqe_code_hash'],pipeline.config['config_hash'],'E1')
            score = pipeline.backend().score(candidate,scene)
            require(score.score_valid, 'E1 official scoring failed')
            scores.append([float(base['scores'][row,0]), score.total_score_01])
            for c in components:
                components[c].append([float(base['components'][c][row,0]), score.named_components[c]])
                masks[c].append([bool(base['component_valid'][c][row,0]), score.component_valid_masks[c]])
            terms = expert_il_terms(pred,load_target(scene,pipeline.trajectory_contract)[None].to(pipeline.device),prev_weight=network.framework.action_model.prev_weight)
            losses.append(float(terms['il'].numerator))
            cases.append({'scene_id':scene.scene_id,'copy_score_01':score.total_score_01,'trajectory_hash':th})
    import numpy as np
    count=len(scores)
    report=selection_report(scores,np.ones((count,2),bool),np.ones(count,int),np.zeros(count,int),components,masks,[s.source_group_id for s in base['scenes']])
    report.update(role=role,baseline='E1_single_policy_copy',cases=cases,IL_loss=float(np.mean(losses)),
                  selection_key=[report['selected_mean_01'],-float(np.mean(losses))],scope='single_policy_deployment',science='UNTESTED' if pipeline.mode=='smoke' else 'REPORTED')
    return report


def train_copy(pipeline, number, *, steps=None, resume=None, stop_after=None):
    manifest=read_json(pipeline.round_root(number)/'expert_manifest.json')
    require(manifest['manifest_hash']==digest({k:v for k,v in manifest.items() if k!='manifest_hash'}),'E1 manifest checksum')
    records={r['scene_id']:strict_record(SceneRecord,r) for r in manifest['records']}
    module=load_copy(pipeline)
    cfg=dict(pipeline.config['expert_train'])
    cfg['precision']=pipeline.config['model']['precision']
    cfg['find_unused_parameters']=True  # Original VLM's unused LM output projection remains intact.
    if steps:
        cfg['max_optimizer_steps']=steps;cfg['warmup_steps']=min(cfg['warmup_steps'],steps-1)
    require(pipeline.mode=='full' or cfg['max_optimizer_steps']<=32,'bounded E1 smoke/profile')
    require(len(manifest['sampling_plan']['entries'])==cfg['max_optimizer_steps']*cfg['global_batch_size'],'E1 budget must match the same incremental manifest')
    def fetch(entries,device):
        scenes=[records[e['scene_id']] for e in entries]
        return {'observations':[load_observation(s,pipeline.contract) for s in scenes],
            'target':torch.stack([load_target(s,pipeline.trajectory_contract) for s in scenes]).to(device),
            'mask':torch.tensor([not e.get('_padding',False) for e in entries],device=device)[:,None].expand(-1,pipeline.trajectory_contract.horizon)}
    def loss(net,data):
        return expert_il_terms(net(data['observations']),data['target'],prev_weight=module.framework.action_model.prev_weight,masks=data['mask'])
    before=file_hash(pipeline.contract['query_checkpoint'])
    output=pipeline.round_root(number)/'policy_copy'
    train(module,loss,fetch,ConsumedSampler(manifest['sampling_plan']),cfg,
          {'S0':before,'manifest':manifest['manifest_hash'],'baseline':'E1_whole_policy_copy'},output,
          device=pipeline.device,resume=resume,stop_after=stop_after,seed=pipeline.config['seed'],
          denominator_function=lambda data:{'il':data['mask'].any(-1).sum().float()},
          validate=lambda model,step:evaluate_copy(pipeline,number,'stage_val',model))
    require(file_hash(pipeline.contract['query_checkpoint'])==before,'E1 changed locked S0')
    result=read_json(output/'result.json')
    result.update(locked_S0_unchanged=True,whole_policy_copy=True,trainable_parameters=sum(p.numel() for p in module.parameters() if p.requires_grad))
    atomic_json(output/'result.json',result)
    return result
