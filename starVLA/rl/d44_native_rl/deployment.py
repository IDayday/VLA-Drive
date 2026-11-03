"""Strict native-parent+trainables loading; deployment has no R/readouts/ref."""
import json
from pathlib import Path
import torch
from torch import nn
from .parent import load
from .expanded_scope import install
from starVLA.rl.qwen_world_compare.rl_scope import scope_targets
from starVLA.rl.qwen_world_compare.deployment import prune_deployment
from starVLA.rl.qwen_world_compare.contracts import file_sha,fingerprint


def inspect(path):
    path=Path(path);seal=json.loads((path/'COMPLETE').read_text());files=json.loads((path/'files.json').read_text());identity=json.loads((path/'identity.json').read_text())
    body=dict(identity);sha=body.pop('sha256')
    if identity['schema']!='d44_native_rl_v1'or identity['architecture_id']!='D'or fingerprint(body)!=sha:
        raise ValueError('Foreign D44 run identity')
    if identity['recipe'].get('parent')=='D_IL46K':
        from .config import validate
        from .parent import audit_export,D46_MODEL_SHA
        cfg=validate(identity['recipe']);parent=audit_export(cfg['parent_checkpoint'],verify=False,binding=cfg['parent_registration'])
        if identity['parent_model_sha256']!=D46_MODEL_SHA or identity['parent']!=parent:
            raise ValueError('D46 checkpoint policy-parent provenance changed')
    if seal['files_sha256']!=file_sha(path/'files.json'):raise ValueError('Changed native checkpoint inventory')
    for name in ('identity.json','optimizer.pt','transaction.json'):
        if file_sha(path/name)!=files[name]:raise ValueError('Changed native payload '+name)
    return identity,seal,files


def load_policy(path,device='cpu'):
    identity,seal,files=inspect(path);cfg=identity['recipe']
    model,info=load(cfg['parent_checkpoint'],'cpu',readouts=cfg['post_feedback'])
    if info['model_sha256']!=identity['parent_model_sha256']:raise ValueError('IL44 parent identity changed')
    install(model,scope_targets(model),cfg['text_rank'],cfg['text_rank']*2)
    if cfg['film_last_layers']:
        paths=[f'decision.time_films.{i}.{j}'for i in range(28-cfg['film_last_layers'],28)for j in(0,1)]
        install(model,paths,8,16)
    payload=torch.load(Path(path)/'optimizer.pt',map_location='cpu',weights_only=False)
    current=dict(model.named_parameters());expected={n.removeprefix('action_expert.')for n in identity['scope']['trainable_names']}
    actual={n.removeprefix('action_expert.')for n in payload['adapters']}
    if expected!=actual or not expected.issubset(current):raise ValueError('D44 trained parameter closure differs')
    with torch.no_grad():
        for name,value in payload['adapters'].items():
            p=current[name.removeprefix('action_expert.')]
            if value.dtype!=p.dtype or value.shape!=p.shape or not torch.isfinite(value).all():raise ValueError('Invalid deployment trainable '+name)
            p.copy_(value)
    model=prune_deployment(model,copy_model=False).to(device)
    if hasattr(model,'readouts')or model.decision.r_embeddings is not None:raise ValueError('Post-action module leaked into deployment')
    return model,identity,seal,fingerprint(dict(parent=info['model_sha256'],trainable_payload=files['optimizer.pt'],scope=identity['scope']))
