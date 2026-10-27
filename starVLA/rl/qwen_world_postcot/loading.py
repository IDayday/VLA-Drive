"""Strict complete-checkpoint/deployment loading; no historical conversion."""
import json
import math
import copy
from pathlib import Path
import torch
from .contracts import file_sha,fingerprint
from starVLA.model.modules.qwen_world_postcot.model import QwenWorldPostCoT,TRAIN_ONLY_PREFIXES

def inspect_checkpoint(path,*,verify_files='all'):
    root=Path(path);seal=json.loads((root/'COMPLETE').read_text());identity=json.loads((root/'identity.json').read_text())
    files=json.loads((root/'files.json').read_text())
    if (seal.get('status')!='COMPLETE' or seal['files_sha256']!=file_sha(root/'files.json')
            or seal['identity_sha256']!=fingerprint(identity)):raise ValueError('Checkpoint receipt mismatch')
    if verify_files not in ('all','model'):raise ValueError('Unknown checkpoint verification scope')
    name=seal['model_file']
    if name!='native/mp_rank_00_model_states.pt' or name not in files:raise ValueError('Missing native full model state identity')
    world=identity['world']
    if (world<1 or len([p for p in files if p.endswith('_optim_states.pt')])!=world
            or any(f'rank{i:02d}_state.pt' not in files for i in range(world))):raise ValueError('Missing optimizer/rank recovery coverage')
    for name,digest in files.items():
        target=root/name
        if not target.resolve().is_relative_to(root.resolve()) or not target.is_file():raise ValueError('Unsafe/missing checkpoint file: '+name)
        if (verify_files=='all' or name==seal['model_file']) and file_sha(target)!=digest:raise ValueError('Incomplete/changed checkpoint file: '+name)
    name=seal['model_file']
    return dict(path=str(root.resolve()),seal=seal,identity=identity,files=files,model_sha256=files[name])

def load_deployment_checkpoint(path,device='cpu'):
    # Deployment uses model tensors only. Check the sealed complete shard
    # inventory, and rehash that model once; optimizer bytes are verified by
    # full-state resume, not reread for every distributed evaluator worker.
    info=inspect_checkpoint(path,verify_files='model');root=Path(path)
    raw=torch.load(root/info['seal']['model_file'],map_location='cpu',weights_only=False)
    state=raw['module'];expected=info['identity']
    if any(value.dtype!=torch.float32 for value in state.values() if isinstance(value,torch.Tensor) and value.is_floating_point()):raise ValueError('Stored full-framework parameters/buffers must preserve FP32 values')
    model=QwenWorldPostCoT(None,config=copy.deepcopy(expected['model_config']),
        checkpoint_config=copy.deepcopy(expected['qwen_config']),train_heads=False)
    filtered={k:v for k,v in state.items() if not k.startswith(TRAIN_ONLY_PREFIXES)}
    removed=set(state)-set(filtered)
    if not removed and expected.get('train_heads',True):raise ValueError('Full training checkpoint lacks declared auxiliary heads')
    model.load_state_dict(filtered,strict=True)
    model.eval().requires_grad_(False).to(device)
    return model,info

def validate_rl_parent_selection(path,info):
    data=json.loads(Path(path).read_text());seal=data.pop('sha256',None)
    if fingerprint(data)!=seal:raise ValueError('Changed DEV parent selection')
    if data.get('schema')=='world_postcot_explicit_IL40000_parent_v1':
        return _validate_explicit_40k_parent(data,seal,info)
    if data['schema']!='world_postcot_DEV_selection_v1' or data['completed_IL_updates']!=100000 or data['Navtest_used'] is not False:raise ValueError('Formal RL requires completed100k IL and DEV-only selection')
    if info['identity'].get('qualification') is not False:raise ValueError('Qualification checkpoints cannot initialize formal RL')
    selected=data['selected']
    if (selected['model_sha256']!=info['model_sha256'] or selected['updates']!=info['seal']['updates']
            or Path(selected['checkpoint']).resolve()!=Path(info['path']).resolve()):raise ValueError('Selected parent weight/update identity changed')
    if sorted(r['updates'] for r in data['evaluated_candidates'])!=list(range(5000,100001,5000)):raise ValueError('DEV milestone window incomplete')
    if any(not math.isfinite(row['PDMS']) or not row.get('model_sha256') for row in data['evaluated_candidates']):raise ValueError('DEV candidate score/hash is invalid')
    best=max(data['evaluated_candidates'],key=lambda v:(v['PDMS'],v['updates']))
    if any(best[k]!=selected[k] for k in ('checkpoint','updates','model_sha256')):raise ValueError('Parent violates predeclared DEV PDMS selection')
    return dict(data,sha256=seal)


def _validate_explicit_40k_parent(data,seal,info):
    """A separately registered user choice; never a completed100k DEV selection.

    The IL40k Navtest was observed before this instruction. Preserve that fact
    rather than forging the original campaign's completed-IL/DEV-only receipt.
    The default selection schema above remains unchanged.
    """
    if (data.get('user_instruction')!='基于 40K的权重，启动Qwen-world 的RL训练'
            or data.get('selection_method')!='explicit_user_IL40000'
            or data.get('completed_IL_updates')!=40000
            or data.get('Navtest_used') is not True
            or data.get('training_seed')!=42 or data.get('evaluation_seed')!=2101):
        raise ValueError('Invalid explicit40k parent authorization')
    if info['identity'].get('qualification') is not False:
        raise ValueError('Qualification checkpoints cannot initialize formal RL')
    selected=data['selected']
    if (selected['updates']!=40000 or info['seal']['updates']!=40000
            or selected['model_sha256']!=info['model_sha256']
            or Path(selected['checkpoint']).resolve()!=Path(info['path']).resolve()
            or selected['files_sha256']!=info['seal']['files_sha256']
            or selected['identity_sha256']!=fingerprint(info['identity'])):
        raise ValueError('Explicit40k selected parent identity changed')
    for key in ('training_recipe_sha256','hard_manifest_sha256'):
        value=data.get(key)
        if not isinstance(value,str) or len(value)!=64 or any(c not in '0123456789abcdef' for c in value):
            raise ValueError('Missing explicit40k training binding: '+key)
    return dict(data,sha256=seal)
