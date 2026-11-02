"""One immutable native D44 parent, including IL-trained R16/readouts."""
import copy
import json
import os
from pathlib import Path
import torch
from starVLA.rl.qwen_world_compare.loading import inspect_IL_checkpoint, model_from_identity, strict_tensor_load
from starVLA.rl.qwen_world_compare.contracts import file_sha, fingerprint

PARENT=Path('/mnt/project/qwen-world-four-frameworks-20261029/runs/ARCH_D/IL_seed42/checkpoints/update044000')
ROOT=Path('/mnt/project/d44-native-rl-20261103')


def audit(path=PARENT, *, verify=True):
    info=inspect_IL_checkpoint(path,verify_files='model'if verify else'inventory')
    if info['identity']['architecture_id']!='D' or info['seal']['updates']!=44000 or info['identity']['qualification']:
        raise ValueError('Only the actual formal D IL44000 parent is allowed')
    cfg=info['identity']['config']['deployment']
    if cfg['action_shape']!=[8,3] or cfg['steps']!=10 or cfg['shift']!=5 or cfg['seed']!=2101:
        raise ValueError('D44 deployment identity changed')
    return info


def pin():
    info=audit();dest=ROOT/'IL44_PARENT';dest.mkdir(parents=True,exist_ok=True)
    for name in list(info['files'])+['COMPLETE','identity.json','files.json']:
        src=PARENT/name;out=dest/name;out.parent.mkdir(parents=True,exist_ok=True)
        if not out.exists():os.link(src,out)
        if not os.path.samefile(src,out):raise ValueError('Parent pin differs: '+name)
    return audit(dest,verify=False)


def load(path,device,*,readouts=False):
    info=audit(path,verify=False)
    raw=torch.load(Path(path)/info['seal']['model_file'],map_location='cpu',weights_only=False,mmap=True)
    model=model_from_identity(info['identity'],train_heads=True)
    strict_tensor_load(model,raw['module'])
    if not readouts:
        from starVLA.rl.qwen_world_compare.deployment import prune_deployment
        model=prune_deployment(model,copy_model=False)
    return model.eval().requires_grad_(False).to(device),info
