"""Merge unmerged source LoRA into the original standalone DDP deployment."""
import argparse
import json
import os
from pathlib import Path
import shutil
import torch
from starVLA.rl.simwam_recipe.contracts import SFT_EXPORT,SFT_SHA256,file_sha,require_dispatch,load_recipe
from starVLA.rl.simwam_recipe.trainer_adapter import make_actor
from starVLA.rl.simwam_recipe.lora_adapter import restore_adapters,merged_state_dict
from starVLA.rl.simwam_recipe.checkpoint import atomic_torch
from starVLA.rl.flow_grpo.transactions import atomic_json
from starVLA.rl.flow_grpo.reproducibility import configure_numerics


def load_adapters(path,actor):
    path=Path(path);complete=json.loads((path/'COMPLETE').read_text())
    files=json.loads((path/'files.json').read_text())
    if complete['files_sha256']!=file_sha(path/'files.json') or files['optimizer.pt']!=file_sha(path/'optimizer.pt'):
        raise ValueError('Unsealed checkpoint')
    identity=json.loads((path/'identity.json').read_text())
    if identity['SFT_sha256']!=SFT_SHA256:raise ValueError('Wrong base')
    saved=torch.load(path/'optimizer.pt',map_location='cpu',weights_only=False)
    restore_adapters(actor,saved['adapters']);return identity,complete


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();require_dispatch();configure_numerics()
    ckpt=Path(a.checkpoint);identity=json.loads((ckpt/'identity.json').read_text());cfg=identity['recipe']
    actor,_,_=make_actor(cfg);identity,complete=load_adapters(ckpt,actor)
    merged=merged_state_dict(actor.action_expert)
    reference=actor.ref_action_expert.state_dict()
    if set(merged)!=set(reference) or any('lora_' in k or '.base.' in k for k in merged):raise ValueError('Invalid merged original key closure')
    original=Path(SFT_EXPORT)
    if file_sha(original/'pytorch_model.pt')!=SFT_SHA256:raise ValueError('SFT changed')
    with torch.serialization.safe_globals([set]):state=torch.load(original/'pytorch_model.pt',mmap=True,map_location='cpu',weights_only=True)
    state=state.get('module',state);output=dict(state)
    attention_weights={item['path']+'.weight' for item in actor.closure['adapters']}
    changed=[]
    for key,value in merged.items():
        name='action_model.'+key;value=value.detach().cpu().to(state[name].dtype)
        if not torch.equal(value,state[name]):
            if name not in attention_weights:raise ValueError('Unexpected frozen deployment weight change: '+name)
            changed.append(name)
        output[name]=value
    root=Path(a.output)
    if root.exists():raise FileExistsError(root)
    attempt=root.with_name(root.name+'.incomplete');attempt.mkdir(parents=True,exist_ok=False)
    atomic_torch(attempt/'pytorch_model.pt',output);shutil.copyfile(original/'config.yaml',attempt/'config.yaml')
    report=dict(schema='simwam_merged_original_DDP_export_v1',checkpoint=str(ckpt),run_identity=identity['sha256'],
                updates=complete['updates'],SFT_sha256=SFT_SHA256,weight_sha256=file_sha(attempt/'pytorch_model.pt'),
                changed_attention_weights=changed,LoRA_compute='FP32 B@A; sum then original storage dtype',
                frozen_upstream_identical=True,no_teacher_cache_search_or_noise_head_dependency=True,
                not_exact_resume_use_unmerged_checkpoint=True)
    atomic_json(attempt/'export.json',report);atomic_json(attempt/'COMPLETE',report);os.rename(attempt,root)
    print(json.dumps(dict(status='EXPORTED',output=str(root),sha256=report['weight_sha256'],changed=len(changed))))


if __name__=='__main__':main()
