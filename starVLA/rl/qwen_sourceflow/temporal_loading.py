"""Strict temporal model loading; no teacher/labels/cache in the actor."""
import json
from pathlib import Path
import torch
from starVLA.model.modules.qwen_sourceflow.temporal_dino import TemporalDINOFlowVLA
from starVLA.model.modules.qwen_sourceflow.vendor.action_dit import ActionDiT
from .contracts import QWEN_PATH,ACTION_CONFIG,file_sha,runtime_identity


def load_temporal(path,device='cpu',*,qualification=False):
    path=Path(path);seal=json.loads((path/'COMPLETE').read_text());files=json.loads((path/'files.json').read_text())
    identity=json.loads((path/'identity.json').read_text())
    if seal['schema']!='temporal_dino_IL_READY_v1' or seal['phase']!='READY' or '.incomplete' in path.name:
        raise ValueError('Unpublished temporal policy')
    if identity['schema']!='native_temporal_DINO_IL_v1' or (identity['qualification'] and not qualification):
        raise ValueError('Wrong policy or qualification used as formal model')
    if file_sha(path/'files.json')!=seal['files_sha256'] or file_sha(path/'identity.json')!=files['identity.json'] or file_sha(path/'model.pt')!=files['model.pt']:
        raise ValueError('Changed sealed temporal model')
    if identity['runtime']!=runtime_identity():raise ValueError('Unqualified numerical/runtime environment')
    cfg=json.loads((QWEN_PATH/'config.json').read_text())
    model=TemporalDINOFlowVLA(QWEN_PATH,ActionDiT(**ACTION_CONFIG),checkpoint_config=cfg)
    model.load_state_dict(torch.load(path/'model.pt',map_location='cpu',weights_only=True,mmap=True),strict=True)
    return model.to(device).eval().requires_grad_(False),dict(identity=identity,seal=seal,files=files)
