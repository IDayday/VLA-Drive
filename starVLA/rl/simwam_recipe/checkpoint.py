"""Atomic unmerged LoRA + native Adam/RNG/inner-rollout recovery boundaries."""
import json
import os
from pathlib import Path
import random
import time
import uuid
import numpy as np
import torch
import torch.distributed as dist
from starVLA.rl.flow_grpo.transactions import atomic_json
from .contracts import file_sha
from .lora_adapter import adapter_state,restore_adapters


def cpu_tree(value):
    if torch.is_tensor(value):return value.detach().cpu()
    if isinstance(value,dict):return {k:cpu_tree(v) for k,v in value.items()}
    if isinstance(value,list):return [cpu_tree(v) for v in value]
    if isinstance(value,tuple):return tuple(cpu_tree(v) for v in value)
    return value


def device_tree(value,device):
    if torch.is_tensor(value):return value.to(device)
    if isinstance(value,dict):return {k:device_tree(v,device) for k,v in value.items()}
    if isinstance(value,list):return [device_tree(v,device) for v in value]
    if isinstance(value,tuple):return tuple(device_tree(v,device) for v in value)
    return value


def atomic_torch(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.writing-'+uuid.uuid4().hex)
    with tmp.open('xb') as f:
        torch.save(cpu_tree(value),f);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)


def save(root,actor,optimizer,scheduler,generator,state,rollout,identity,*,tag=None):
    rank=dist.get_rank() if dist.is_initialized() else 0
    world=dist.get_world_size() if dist.is_initialized() else 1
    name=tag or f"update{state['updates']:06}_inner{state['next_inner']}"
    final=Path(root)/'checkpoints'/name
    attempt=final.with_name(name+'.incomplete')
    if rank==0:
        if final.exists():raise FileExistsError('Checkpoint identity must not be overwritten: '+str(final))
        attempt.mkdir(parents=True,exist_ok=False)
    if world>1:dist.barrier()
    rng=dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
             cuda=torch.cuda.get_rng_state(),rollout_generator=generator.get_state())
    atomic_torch(attempt/f'rank{rank}.pt',dict(state=state,rollout=rollout,rng=rng,identity_sha256=identity['sha256']))
    if rank==0:
        atomic_torch(attempt/'optimizer.pt',dict(adapters=adapter_state(actor),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict()))
        atomic_json(attempt/'identity.json',identity)
    if world>1:dist.barrier()
    if rank==0:
        files={p.name:file_sha(p) for p in sorted(attempt.iterdir()) if p.is_file()}
        atomic_json(attempt/'files.json',files)
        atomic_json(attempt/'COMPLETE',dict(world=world,updates=state['updates'],next_inner=state['next_inner'],files_sha256=file_sha(attempt/'files.json')))
        os.rename(attempt,final)
        atomic_json(Path(root)/'latest.json',dict(checkpoint=str(final),updates=state['updates']))
    if world>1:dist.barrier()
    return str(final)


def restore(path,actor,optimizer,scheduler,generator,identity):
    rank=dist.get_rank() if dist.is_initialized() else 0
    world=dist.get_world_size() if dist.is_initialized() else 1
    path=Path(path);complete=json.loads((path/'COMPLETE').read_text())
    if complete['world']!=world or file_sha(path/'files.json')!=complete['files_sha256']:raise ValueError('Checkpoint publication changed')
    if json.loads((path/'identity.json').read_text())!=identity:raise ValueError('Exact resume identity mismatch')
    files=json.loads((path/'files.json').read_text())
    for n in ['optimizer.pt',f'rank{rank}.pt','identity.json']:
        if file_sha(path/n)!=files[n]:raise ValueError('Checkpoint content changed: '+n)
    # These are our sealed local native optimizer/RNG states, never downloads.
    common=torch.load(path/'optimizer.pt',map_location='cpu',weights_only=False)
    own=torch.load(path/f'rank{rank}.pt',map_location='cpu',weights_only=False)
    restore_adapters(actor,common['adapters']);optimizer.load_state_dict(common['optimizer']);scheduler.load_state_dict(common['scheduler'])
    rng=own['rng'];random.setstate(rng['python']);np.random.set_state(rng['numpy']);torch.set_rng_state(rng['torch'])
    torch.cuda.set_rng_state(rng['cuda']);generator.set_state(rng['rollout_generator'])
    return own['state'],device_tree(own['rollout'],actor.device)
