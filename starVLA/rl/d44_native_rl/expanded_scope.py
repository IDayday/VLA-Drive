"""Exact native module paths; same vendor LoRA arithmetic for each rank."""
import copy
import torch
from torch import nn
from starVLA.rl.simwam_recipe.vendor.lora import LoRALinear


def install(model,paths,rank,alpha):
    rows=[];seen=set()
    for name in paths:
        base=model.get_submodule(name)
        if not isinstance(base,nn.Linear)or id(base)in seen:raise ValueError('Missing/aliased native Linear '+name)
        seen.add(id(base));parent,_,child=name.rpartition('.')
        wrapped=LoRALinear(base,r=rank,alpha=alpha,dropout=0.)
        setattr(model.get_submodule(parent),child,wrapped)
        rows.append(dict(path=name,rank=rank,alpha=alpha,numel=wrapped.lora_A.numel()+wrapped.lora_B.numel()))
    return rows


def independent_trainable_module(model,path):
    # Reference shares only frozen base tensors. Clone before unfreezing any I/O/head.
    parent,_,child=path.rpartition('.');module=copy.deepcopy(model.get_submodule(path))
    module.requires_grad_(True);setattr(model.get_submodule(parent),child,module)


def groups(actor,cfg):
    roles={}
    for name,p in actor.named_parameters():
        if not p.requires_grad:continue
        if name.startswith('action_expert.readouts.'):role,lr='post_readout',2e-5
        elif '.time_films.'in name:role,lr='film_lora',1e-5
        elif name.endswith('action_position'):role,lr='action_position',1e-6
        elif '.action_input.'in name or '.action_output.'in name:role,lr='action_io',3e-6
        else:role,lr='text_lora',cfg['learning_rate']
        row=roles.setdefault(role,dict(name=role,lr=lr,initial_lr=lr,params=[]));row['params'].append(p)
    return list(roles.values())
