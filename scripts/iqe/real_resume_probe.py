"""Exact resume of a real private Query head with cached real-image features/GT.

This is an isolated optimizer diagnostic, never a candidate or deployment update.
"""
import argparse
from copy import deepcopy
from pathlib import Path
import torch
from omegaconf import OmegaConf
from iqe.config import load_config
from iqe.pipeline import Pipeline,stack_features
from iqe.query_base import bind_source,QueryActionBridge
from iqe.contracts import SceneRecord,strict_record,require
from iqe.data.sources import load_target
from iqe.data.sampler import ConsumedSampler
from iqe.losses import expert_il_terms
from iqe.training.trainer import train
from iqe.training.checkpoint import capture_rng
from iqe.io import read_json,atomic_json,file_hash
from iqe.resources import qualify


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--config',required=True);p.add_argument('--round',type=int,default=1)
    p.add_argument('--output',required=True);a=p.parse_args()
    qualify('cpu');torch.set_num_threads(2)
    config=load_config(a.config);pipeline=Pipeline(config,mode='smoke',max_samples=32,device='cpu');pipeline.use_locked_base()
    c=pipeline.contract;bind_source(c['source_root'],c['source_commit'])
    require(file_hash(c['query_checkpoint'])==c['query_checkpoint_hash'],'real S0 changed')
    state=torch.load(c['query_checkpoint'],map_location='cpu',weights_only=False,mmap=True)
    bridge=QueryActionBridge(OmegaConf.create(c['source_config']),c['query_architecture'])
    bridge.expert.load_state_dict({k.removeprefix('action_model.expert.'):v for k,v in state['model'].items()
                                  if k.startswith('action_model.expert.')},strict=True)
    full=deepcopy(bridge.expert);partial=deepcopy(full);del state,bridge
    manifest=read_json(pipeline.round_root(a.round)/'expert_manifest.json')
    records={r['scene_id']:strict_record(SceneRecord,r) for r in manifest['records']}
    cfg=dict(pipeline.config['expert_train']);cfg.update(max_optimizer_steps=2,warmup_steps=1,validation_every_steps=2)
    plan=manifest['sampling_plan'];require(len(plan['entries'])==2*cfg['global_batch_size'],'diagnostic requires actual 2-step manifest')
    def fetch(entries,device):
        scenes=[records[e['scene_id']] for e in entries]
        return {'features':stack_features([pipeline.cached(s) for s in scenes]).to(device),
                'target':torch.stack([load_target(s,pipeline.trajectory_contract) for s in scenes]).to(device)}
    def loss(module,batch):return expert_il_terms(module(batch['features']),batch['target'],prev_weight=c['query_architecture']['prev_weight'])
    denominator=lambda b:{'il':b['target'].new_tensor(len(b['target']))}
    deps={'S0':c['query_checkpoint_hash'],'manifest':manifest['manifest_hash'],'diagnostic_only':True}
    root=Path(a.output)
    whole=train(full,loss,fetch,ConsumedSampler(plan),cfg,deps,root/'whole',seed=97,denominator_function=denominator)
    rng=capture_rng()
    train(partial,loss,fetch,ConsumedSampler(plan),cfg,deps,root/'interrupted',seed=97,denominator_function=denominator,stop_after=1)
    resumed=deepcopy(partial)
    tail=train(resumed,loss,fetch,ConsumedSampler(plan),cfg,deps,root/'interrupted',seed=97,denominator_function=denominator,
               resume=root/'interrupted/step_000001.pt')
    require(whole[1]['sample_ids']==tail[0]['sample_ids'],'real resume next IDs')
    require(whole[1]['loss']==tail[0]['loss'] and whole[1]['gradient_norm']==tail[0]['gradient_norm'],'real resume loss/gradient norm')
    # Detailed log dictionaries are intentionally sampled at each job's first
    # update, so compare actual final gradient tensors rather than log presence.
    require(all((a.grad is None and b.grad is None) or
                (a.grad is not None and b.grad is not None and torch.equal(a.grad,b.grad))
                for a,b in zip(full.parameters(),resumed.parameters())),'real resume gradient tensors')
    require(all(torch.equal(a,b) for a,b in zip(full.parameters(),resumed.parameters())),'real resume parameters')
    require(torch.equal(rng['cpu'],capture_rng()['cpu']) and rng['python']==capture_rng()['python'],'real resume RNG')
    atomic_json(root/'RESULT.json',{'status':'PASS','real_image_feature_cache':True,'real_audited_targets':True,
        'optimizer_boundary':True,'next_ids_loss_gradients_parameters_RNG_match':True,'steps':2,'diagnostic_only':True,
        'interrupted_checkpoint':str(root/'interrupted/step_000001.pt'),'science':'UNTESTED'})


if __name__=='__main__':main()
