"""Strict generic Wan loading for the otherwise unchanged official converter.

Only initialization instantiates WanVideoDiT, on CPU. VAE/T5/tokenizer are unused
by backbone interpolation and are not loaded/downloaded. Online VLA has no Wan
video module. No driving checkpoint is accepted at this entry.
"""
import json
from pathlib import Path
from types import SimpleNamespace
import torch
from safetensors import safe_open
from .vendor.wan_video_dit import WanVideoDiT
from .vendor.action_dit import ActionDiT
from starVLA.rl.qwen_sourceflow.contracts import ROOT, ACTION_CONFIG, file_sha


def load_wan22_ti2v_5b_components(device, torch_dtype, model_id, tokenizer_model_id,
                                redirect_common_files, dit_config):
    if model_id != 'Wan-AI/Wan2.2-TI2V-5B' or str(device) != 'cpu':
        raise ValueError('Generic official Wan CPU-only initialization required')
    identity=json.loads((ROOT/'assets/wan_identity.json').read_text())
    if identity['revision']!='921dbaf3f1674a56f47e83fb80a34bac8a8f203e' or not identity['official_weight_sha_verified']:
        raise ValueError('Unverified generic Wan asset')
    root=Path(identity['path'])
    with torch.device('meta'):
        model=WanVideoDiT(**dit_config)
    expected=model.state_dict();state={}
    for name, sha in identity['files'].items():
        if not name.endswith('.safetensors'):continue
        path=root/name
        if file_sha(path)!=sha:raise ValueError('Wan content hash changed: '+name)
        with safe_open(path,framework='pt',device='cpu') as f:
            for key in f.keys():
                if key in state:raise ValueError('Duplicate Wan tensor '+key)
                state[key]=f.get_tensor(key).to(torch_dtype)
    missing=set(expected)-set(state);extra=set(state)-set(expected)
    if missing or extra:
        raise ValueError(f'Strict generic Wan state mismatch: missing={sorted(missing)} extra={sorted(extra)}')
    if any(expected[k].shape!=state[k].shape for k in state):
        raise ValueError('Generic Wan tensor shape mismatch')
    model.load_state_dict(state,strict=True,assign=True)
    return SimpleNamespace(dit=model)


def load_action_initialization(path, *, device='cpu', dtype=torch.float32, verify=True):
    path=Path(path)
    if not path.is_file():raise FileNotFoundError(path)
    identity=json.loads(path.with_suffix('.identity.json').read_text())
    if (identity['source']!='generic_Wan2.2-TI2V-5B_official_conversion' or
            identity['wan_revision']!='921dbaf3f1674a56f47e83fb80a34bac8a8f203e' or
            identity['action_config']!=ACTION_CONFIG):
        raise ValueError('Action initialization is not the locked generic conversion')
    if verify and file_sha(path)!=identity['payload_sha256']:
        raise ValueError('Converted payload content changed')
    # This source method validates every backbone key, metadata and tensor shape,
    # merges ONLY its declared random-kept prefixes, then uses strict=True.
    return ActionDiT.from_pretrained(ACTION_CONFIG, str(path), device=device, torch_dtype=dtype)
