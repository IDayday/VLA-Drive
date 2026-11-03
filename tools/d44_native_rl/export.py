"""Atomic, unmerged deployment adapters bound to the complete immutable IL44 asset."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import uuid
import torch
from starVLA.rl.d44_native_rl.deployment import inspect
from starVLA.rl.d44_native_rl.parent import load
from starVLA.rl.d44_native_rl.expanded_scope import install
from starVLA.rl.qwen_world_compare.rl_scope import scope_targets
from starVLA.rl.qwen_world_compare.contracts import file_sha,fingerprint
from starVLA.rl.qwen_world_abcd_research.checkpoint import atomic_torch
from starVLA.rl.flow_grpo.transactions import atomic_json


def content_sha(tensors):
    value=hashlib.sha256()
    for name,p in sorted(tensors.items()):
        p=p.detach().cpu().contiguous();value.update(name.encode());value.update(str((tuple(p.shape),p.dtype)).encode());value.update(p.numpy().tobytes())
    return value.hexdigest()


def export(checkpoint,output):
    checkpoint=Path(checkpoint);out=Path(output);identity,seal,files=inspect(checkpoint)
    if out.exists():raise FileExistsError('Existing deployment asset is immutable')
    payload=torch.load(checkpoint/'optimizer.pt',map_location='cpu',weights_only=False)['adapters']
    expected=set(identity['scope']['trainable_names'])
    if set(payload)!=expected:raise ValueError('Incomplete trainable closure')
    weights={k.removeprefix('action_expert.'):v for k,v in payload.items()if not k.startswith('action_expert.readouts.')}
    if any(not torch.isfinite(v).all()or v.dtype!=torch.float32 for v in weights.values()):raise ValueError('Nonfinite/non-FP32 deployment weights')
    temp=out.with_name(out.name+'.incomplete.'+uuid.uuid4().hex);temp.mkdir(parents=True)
    atomic_torch(temp/'adapters.pt',weights)
    manifest=dict(schema='d44_native_compact_deployment_v1',IL_parent=identity['recipe']['parent_checkpoint'],
        IL_parent_model_sha256=identity['parent_model_sha256'],training_identity=identity['sha256'],
        source_checkpoint=str(checkpoint.resolve()),source_inventory_sha256=seal['files_sha256'],updates=seal['updates'],
        recipe=identity['recipe'],scope=sorted(weights),adapter_content_sha256=content_sha(weights),
        adapters_file_sha256=file_sha(temp/'adapters.pt'),deployment='single_B1_cleanx_K10_shift5_seed2101',
        unmerged_LoRA=True,R16=False,CoT=False,scorer=False,qualification=identity['qualification'],
        requires_complete_immutable_IL_parent=True,contains_Adam=False)
    manifest['model_state_sha256']=fingerprint(dict(IL=manifest['IL_parent_model_sha256'],weights=manifest['adapter_content_sha256']))
    atomic_json(temp/'export.json',manifest);atomic_json(temp/'COMPLETE',dict(export_sha256=file_sha(temp/'export.json')))
    os.rename(temp,out);return manifest


def load_export(path,device='cpu'):
    path=Path(path)
    if '.incomplete.'in path.name:raise ValueError('Unpublished deployment')
    if read(path/'COMPLETE')['export_sha256']!=file_sha(path/'export.json'):raise ValueError('Changed export metadata')
    meta=read(path/'export.json')
    if meta['schema']!='d44_native_compact_deployment_v1'or meta['adapters_file_sha256']!=file_sha(path/'adapters.pt'):
        raise ValueError('Foreign/changed deployment asset')
    cfg=meta['recipe'];model,info=load(meta['IL_parent'],'cpu',readouts=False,binding=cfg.get('parent_registration'))
    if info['model_sha256']!=meta['IL_parent_model_sha256']:raise ValueError('Base IL44 changed')
    install(model,scope_targets(model),cfg['text_rank'],cfg['text_rank']*2)
    if cfg['film_last_layers']:
        install(model,[f'decision.time_films.{i}.{j}'for i in range(28-cfg['film_last_layers'],28)for j in(0,1)],8,16)
    values=torch.load(path/'adapters.pt',map_location='cpu',weights_only=True);params=dict(model.named_parameters())
    if set(values)!=set(meta['scope'])or not set(values)<=params.keys()or content_sha(values)!=meta['adapter_content_sha256']:
        raise ValueError('Deployment trainable tensor closure changed')
    with torch.no_grad():
        for name,value in values.items():
            p=params[name]
            if p.shape!=value.shape or p.dtype!=value.dtype:raise ValueError('Deployment parameter shape/dtype changed')
            p.copy_(value)
    if hasattr(model,'readouts')or model.decision.r_embeddings is not None:raise ValueError('Training-only modules leaked')
    return model.eval().requires_grad_(False).to(device),meta


def read(path):return json.loads(Path(path).read_text())


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    print(json.dumps(export(a.checkpoint,a.output),indent=2))


if __name__=='__main__':main()
