"""Load a sealed, complete native IL model; never accept a DDP/B policy."""
import json
from pathlib import Path
import torch
from starVLA.model.modules.qwen_sourceflow.model import QwenSourceFlowVLA
from starVLA.model.modules.qwen_sourceflow.vendor.action_dit import ActionDiT
from .contracts import QWEN_PATH, ACTION_CONFIG, file_sha, runtime_identity


def inspect_il_checkpoint(path):
    """Inspect native sealing only; this does not grant training eligibility."""
    path=Path(path)
    seal=json.loads((path/'COMPLETE').read_text());files=json.loads((path/'files.json').read_text())
    identity=json.loads((path/'identity.json').read_text())
    if ('.incomplete' in path.name or seal['schema']!='native_qwen_flow_IL_READY_v1' or
        seal['phase']!='READY' or identity['schema']!='native_qwen_sourceflow_IL_v1' or
        file_sha(path/'files.json')!=seal['files_sha256'] or
        file_sha(path/'identity.json')!=files['identity.json']):
        raise ValueError('Unsealed/incompatible native IL checkpoint')
    return dict(path=str(path),seal=seal,files=files,identity=identity)


def inspect_il(path, *, qualification=False, interim_navtest_registration=None, parent_selection=None):
    if parent_selection is not None and (qualification or interim_navtest_registration is not None):
        raise ValueError('Formal parent selection cannot use qualification/interim eligibility')
    info=inspect_il_checkpoint(path);seal=info['seal'];identity=info['identity']
    if interim_navtest_registration is not None:
        if qualification:raise ValueError('Interim evaluation cannot use qualification eligibility')
        from .interim_evaluation import validate_interim_checkpoint
        validate_interim_checkpoint(interim_navtest_registration,info)
    elif parent_selection is not None:
        from .selection import validate_selection
        selected=validate_selection(parent_selection,info)
        if selected['schema']=='native_IL_DEV_selection_v1' and (
                identity['qualification'] or seal['updates'] not in identity['config']['milestones']):
            raise ValueError('The original DEV selection still requires a registered IL milestone')
    elif not qualification and (identity['qualification'] or seal['updates'] not in identity['config']['milestones']):
        raise ValueError('Formal RL requires a registered, fully trained IL candidate')
    return info


def load_il(path, *, device='cpu', qualification=False, verify=True, interim_navtest_registration=None, parent_selection=None):
    info=inspect_il(path,qualification=qualification,interim_navtest_registration=interim_navtest_registration,
        parent_selection=parent_selection);path=Path(path)
    if info['identity'].get('runtime')!=runtime_identity():
        raise ValueError('IL deployment/runtime differs from its qualified execution')
    if verify and file_sha(path/'model.pt')!=info['files']['model.pt']:
        raise ValueError('Native IL model content changed')
    # Every tensor is overwritten by the strictly loaded full trained model;
    # this constructor is never a substitute for generic initialization.
    model=QwenSourceFlowVLA(QWEN_PATH,ActionDiT(**ACTION_CONFIG))
    # Buffered storage reads avoid thousands of remote mmap page faults on
    # Lustre. This changes file I/O only; dtype and strict tensor loading stay.
    state=torch.load(path/'model.pt',map_location='cpu',mmap=False,weights_only=True)
    model.load_state_dict(state,strict=True)
    return model.to(device).eval(),info
