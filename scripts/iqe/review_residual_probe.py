"""Two-step real raw-residual diagnostic on audited fit features; never admission."""
import argparse
from pathlib import Path
import torch
from omegaconf import OmegaConf
from iqe.config import load_config
from iqe.pipeline import Pipeline,stack_features
from iqe.query_base import bind_source,QueryActionBridge
from iqe.ablations.residual_v2 import ResidualExpertV2
from iqe.evaluation.retention import module_hash
from iqe.data.sources import load_target
from iqe.contracts import SceneRecord,strict_record
from iqe.io import read_json,atomic_json
from iqe.resources import qualify


def main():
    a=argparse.ArgumentParser(__doc__);a.add_argument('--config',required=True);args=a.parse_args()
    qualify('cpu');torch.set_num_threads(2)
    p=Pipeline(load_config(args.config),mode='smoke',max_samples=32,device='cpu');p.use_locked_base();c=p.contract
    bind_source(c['source_root'],c['source_commit'])
    state=torch.load(c['query_checkpoint'],map_location='cpu',mmap=True,weights_only=False)
    bridge=QueryActionBridge(OmegaConf.create(c['source_config']),c['query_architecture'])
    bridge.expert.load_state_dict({k.removeprefix('action_model.expert.'):v for k,v in state['model'].items() if k.startswith('action_model.expert.')},strict=True)
    base=bridge.expert.eval().requires_grad_(False);expert=ResidualExpertV2(base).eval()
    manifest=read_json(p.round_root(1)/'expert_manifest.json');lookup={r['scene_id']:strict_record(SceneRecord,r) for r in manifest['records']}
    scenes=[lookup[i] for i in manifest['buckets']['hard_original'][:4]]
    assert scenes and all(s.split_role=='incremental_fit' for s in scenes)
    f=stack_features([p.cached(s) for s in scenes]);target=torch.stack([load_target(s,p.trajectory_contract) for s in scenes])
    before=module_hash(expert.initial);initial_output=expert(f).raw.detach();assert torch.equal(initial_output,base(f).raw)
    assert torch.count_nonzero(expert.delta(f).raw)==0
    optimizer=torch.optim.AdamW(expert.delta.parameters(),lr=3e-4);losses=[]
    for _ in range(2):
        optimizer.zero_grad();out=expert(f);loss=(out.raw-target).abs().sum(-1).mean();loss.backward();optimizer.step();losses.append(float(loss))
    assert module_hash(expert.initial)==before and torch.equal(base(f).raw,initial_output)
    delta=expert.delta(f).raw.detach();assert delta.abs().max()>0
    evidence={'status':'PASS','science':'UNTESTED','diagnostic_only':True,'fit_scene_ids':[s.scene_id for s in scenes],
        'updates':2,'initial_raw_max_abs':0.,'delta_max_abs_after':float(delta.abs().max()),'frozen_base_hash_unchanged':True,
        'frozen_raw_max_abs':0.,'losses':losses,'parameterization':expert.ablation_identity()}
    atomic_json(p.root/'RESIDUAL_V2_PROBE.json',evidence,immutable=True)


if __name__=='__main__':main()
