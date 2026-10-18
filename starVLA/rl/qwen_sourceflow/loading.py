"""Load a sealed, complete native IL model; never accept a DDP/B policy."""
import json
from pathlib import Path
import torch
from starVLA.model.modules.qwen_sourceflow.model import QwenSourceFlowVLA
from starVLA.model.modules.qwen_sourceflow.vendor.action_dit import ActionDiT
from .contracts import QWEN_PATH, ACTION_CONFIG, file_sha, runtime_identity


def inspect_il(path, *, qualification=False):
    path=Path(path)
    seal=json.loads((path/'COMPLETE').read_text());files=json.loads((path/'files.json').read_text())
    identity=json.loads((path/'identity.json').read_text())
    if ('.incomplete' in path.name or seal['schema']!='native_qwen_flow_IL_READY_v1' or
        seal['phase']!='READY' or identity['schema']!='native_qwen_sourceflow_IL_v1' or
        file_sha(path/'files.json')!=seal['files_sha256'] or
        file_sha(path/'identity.json')!=files['identity.json']):
        raise ValueError('Unsealed/incompatible native IL checkpoint')
    if not qualification and (identity['qualification'] or seal['updates'] not in identity['config']['milestones']):
        raise ValueError('Formal RL requires a registered, fully trained IL candidate')
    return dict(path=str(path),seal=seal,files=files,identity=identity)


def load_il(path, *, device='cpu', qualification=False, verify=True):
    info=inspect_il(path,qualification=qualification);path=Path(path)
    if info['identity'].get('runtime')!=runtime_identity():
        raise ValueError('IL deployment/runtime differs from its qualified execution')
    if verify and file_sha(path/'model.pt')!=info['files']['model.pt']:
        raise ValueError('Native IL model content changed')
    # Every tensor is overwritten by the strictly loaded full trained model;
    # this constructor is never a substitute for generic initialization.
    model=QwenSourceFlowVLA(QWEN_PATH,ActionDiT(**ACTION_CONFIG))
    state=torch.load(path/'model.pt',map_location='cpu',mmap=True,weights_only=True)
    model.load_state_dict(state,strict=True)
    return model.to(device).eval(),info
