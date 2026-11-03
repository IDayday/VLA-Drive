"""Restore the trained native D trunk/interfaces into the original B1 actor."""
import json
from pathlib import Path
import torch
from starVLA.rl.qwen_world_compare.contracts import fingerprint,file_sha
from starVLA.rl.qwen_world_compare.deployment import prune_deployment
from starVLA.rl.qwen_world_compare.rl_scope import scope_targets
from starVLA.rl.d44_native_rl.expanded_scope import install
from starVLA.rl.d44_native_rl.parent import load,D46_MODEL_SHA
from .config import SCHEMA,validate


def inspect(path):
    path=Path(path);identity=json.loads((path/'identity.json').read_text())
    body=dict(identity);digest=body.pop('sha256',None)
    seal=json.loads((path/'COMPLETE').read_text());files=json.loads((path/'files.json').read_text())
    if(identity.get('schema')!='d44_native_rl_v1'or identity.get('algorithm_schema')!=SCHEMA
        or fingerprint(body)!=digest or identity.get('parent_model_sha256')!=D46_MODEL_SHA
        or identity.get('augustus_protocol_sha256')!=fingerprint(identity['recipe']['augustus'])):
        raise ValueError('Not an independently initialized native D46 AUGUstus checkpoint')
    validate(identity['recipe'])
    if seal['files_sha256']!=file_sha(path/'files.json'):raise ValueError('Native inventory changed')
    for name in('identity.json','optimizer.pt','transaction.json'):
        if file_sha(path/name)!=files[name]:raise ValueError('Native payload changed: '+name)
    return identity,seal,files


def load_policy(path,device='cpu'):
    identity,seal,files=inspect(path);cfg=identity['recipe']
    model,info=load(cfg['parent_checkpoint'],'cpu',readouts=False,binding=cfg['parent_registration'])
    if info['model_sha256']!=D46_MODEL_SHA or info!=identity['parent']:raise ValueError('IL46 parent changed')
    if cfg['augustus']['scope']=='lora_text_action':
        install(model,scope_targets(model),cfg['text_rank'],2*cfg['text_rank'])
        paths=[f'decision.time_films.{i}.{j}'for i in range(model.decision.num_layers-cfg['film_last_layers'],model.decision.num_layers)for j in(0,1)]
        install(model,paths,8,16)
    common=torch.load(Path(path)/'optimizer.pt',map_location='cpu',weights_only=False)
    parameters=dict(model.named_parameters());saved=common['adapters']
    if set(saved)!=set(identity['scope']['trainable_names']):raise ValueError('Trained native closure changed')
    with torch.no_grad():
        for name,value in saved.items():
            key=name.removeprefix('action_expert.')
            if key not in parameters:raise ValueError('Unknown deployed native parameter: '+name)
            target=parameters[key]
            if value.dtype!=torch.float32 or value.shape!=target.shape or not torch.isfinite(value).all():
                raise ValueError('Invalid trained native tensor: '+name)
            target.copy_(value)
    del common
    model=prune_deployment(model,copy_model=False).to(device).eval().requires_grad_(False)
    return model,identity,seal,fingerprint(dict(parent=info['model_sha256'],trainable_payload=files['optimizer.pt'],scope=identity['scope']))
