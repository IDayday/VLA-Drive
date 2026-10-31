"""New-family adapter of the audited atomic SimWAM checkpoint transaction.

Publication order, control plane, file seals and rank/optimizer/RNG semantics
are retained.  New schema validation and dataclass CPU/device trees are local
to this namespace; old trainers/validators are unchanged.
"""
from dataclasses import fields,is_dataclass,replace
import json
import os
from pathlib import Path
import random
import time
import uuid

import numpy as np
import torch
import torch.distributed as dist

from starVLA.rl.simwam_recipe import checkpoint as source
from starVLA.rl.simwam_recipe.lora_adapter import adapter_state,restore_adapters
from starVLA.rl.flow_grpo.transactions import atomic_json
from starVLA.rl.qwen_world_compare.contracts import file_sha
from starVLA.rl.qwen_world_compare.recovery_state import validate_boundary as source_boundary,validate_ranks as source_ranks

init_control_group = source.init_control_group
gather = source.gather
check_errors = source.check_errors
fatal_device_error = source.fatal_device_error
EXECUTION_IDENTITY = None
MIGRATION_RECEIPT = None
CHECKPOINT_CONTRACT = 'qwen_world_abcd_research_checkpoint_v1'


def _tree(value, transform):
    if torch.is_tensor(value):
        return transform(value)
    if is_dataclass(value) and not isinstance(value,type):
        return replace(value,**{f.name:_tree(getattr(value,f.name),transform) for f in fields(value) if f.init})
    if isinstance(value,dict):
        return {k:_tree(v,transform) for k,v in value.items()}
    if isinstance(value,list):
        return [_tree(v,transform) for v in value]
    if isinstance(value,tuple):
        return tuple(_tree(v,transform) for v in value)
    return value


def cpu_tree(value):
    return _tree(value,lambda tensor:tensor.detach().cpu())


def device_tree(value,device):
    return _tree(value,lambda tensor:tensor.to(device))


def atomic_torch(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.writing-'+uuid.uuid4().hex)
    with tmp.open('xb')as stream:
        torch.save(cpu_tree(value),stream);stream.flush();os.fsync(stream.fileno())
    os.replace(tmp,path)


def failure_evidence(root,state,error,invocation,*,rollout=None):
    # Local evidence only: no Gloo/NCCL call and no publication/latest update.
    rank=dist.get_rank() if dist.is_initialized()else 0
    target=Path(root)/'failures'/invocation/(f'rank{rank}-'+uuid.uuid4().hex)
    target.mkdir(parents=True,exist_ok=False)
    atomic_json(target/'evidence.json',dict(schema='qwen_world_compare_failure_v1',recoverable=False,
        state=state,error=str(error),invocation=invocation,latest_unchanged=True,time_unix=time.time()))
    if rollout is not None and not fatal_device_error(error):
        try:
            atomic_torch(target/'partial_rollout.pt',rollout)
        except Exception as secondary:
            atomic_json(target/'evidence_write_error.json',dict(error=repr(secondary)))
    return str(target)


def save(root,actor,optimizer,scheduler,generator,state,rollout,identity,*,tag=None,fault=None,runtime=None):
    validate_boundary=runtime.validate_boundary if runtime is not None else source_boundary
    validate_ranks=runtime.validate_ranks if runtime is not None else source_ranks
    rank=dist.get_rank() if dist.is_initialized()else 0
    world=dist.get_world_size() if dist.is_initialized()else 1
    name=tag or f"update{state['updates']:06}_inner{state['next_inner']}"
    final=Path(root)/'checkpoints'/name
    error=None;summary=None
    try:
        if identity['world'] != world:
            raise ValueError('Actual checkpoint rank count differs from the run identity')
        summary=validate_boundary(state,rollout,identity,optimizer,scheduler)
    except Exception as exc:
        error=repr(exc)
    check_errors(error)
    summaries=validate_ranks(gather(summary))
    attempt_id=gather(uuid.uuid4().hex if rank==0 else None)[0]
    attempt=final.with_name(name+'.incomplete.'+attempt_id)
    error=None
    try:
        if rank==0:
            if final.exists():
                raise FileExistsError('Checkpoint identity must not be overwritten: '+str(final))
            attempt.mkdir(parents=True,exist_ok=False)
    except Exception as exc:
        error=repr(exc)
    check_errors(error)
    rng=dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
        cuda=torch.cuda.get_rng_state() if torch.cuda.is_available()else None,
        rollout_generator=generator.get_state())
    error=None
    try:
        if fault:
            fault('rank_write',rank)
        atomic_torch(attempt/f'rank{rank}.pt',dict(state=state,rollout=rollout,rng=rng,identity_sha256=identity['sha256']))
        if rank==0:
            atomic_torch(attempt/'optimizer.pt',dict(adapters=adapter_state(actor),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict()))
            atomic_json(attempt/'identity.json',identity)
            atomic_json(attempt/'transaction.json',dict(schema=CHECKPOINT_CONTRACT,attempt=attempt_id,ranks=summaries,
                execution_identity=EXECUTION_IDENTITY,migration_receipt_sha256=MIGRATION_RECEIPT['sha256']if MIGRATION_RECEIPT else None))
    except Exception as exc:
        if fatal_device_error(exc):
            raise
        error=repr(exc)
    check_errors(error)
    error=None
    try:
        if rank==0:
            files={p.name:file_sha(p)for p in sorted(attempt.iterdir())if p.is_file()}
            required={'optimizer.pt','identity.json','transaction.json'}|{f'rank{r}.pt'for r in range(world)}
            if not required.issubset(files):
                raise ValueError('Incomplete checkpoint files')
            atomic_json(attempt/'files.json',files)
            if fault:
                fault('before_complete',rank)
            atomic_json(attempt/'COMPLETE',dict(schema=2,contract=CHECKPOINT_CONTRACT,phase=state['phase'],world=world,
                updates=state['updates'],next_inner=state['next_inner'],files_sha256=file_sha(attempt/'files.json')))
            os.rename(attempt,final)
            atomic_json(Path(root)/'latest.json',dict(checkpoint=str(final),updates=state['updates'],files_sha256=file_sha(final/'files.json')))
    except Exception as exc:
        error=repr(exc)
    check_errors(error)
    return str(final)


def restore(path,actor,optimizer,scheduler,generator,identity,*,migration=None,runtime=None):
    validate_boundary=runtime.validate_boundary if runtime is not None else source_boundary
    validate_ranks=runtime.validate_ranks if runtime is not None else source_ranks
    rank=dist.get_rank() if dist.is_initialized()else 0
    world=dist.get_world_size() if dist.is_initialized()else 1
    error=None;summary=None
    try:
        path=Path(path);complete=json.loads((path/'COMPLETE').read_text())
        if ('.incomplete' in path.name or complete.get('schema')!=2
                or complete.get('contract')!=CHECKPOINT_CONTRACT or migration is not None):
            raise ValueError('Unpublished/foreign checkpoint; an explicit new-family audited migration is required')
        if (complete['world']!=world or identity['world']!=world
                or file_sha(path/'files.json')!=complete['files_sha256']):
            raise ValueError('Checkpoint publication/rank count changed')
        if json.loads((path/'identity.json').read_text())!=identity:
            raise ValueError('Exact resume identity mismatch')
        files=json.loads((path/'files.json').read_text())
        for name in ['optimizer.pt',f'rank{rank}.pt','identity.json','transaction.json']:
            if file_sha(path/name)!=files[name]:
                raise ValueError('Checkpoint content changed: '+name)
        common=torch.load(path/'optimizer.pt',map_location='cpu',weights_only=False)
        own=torch.load(path/f'rank{rank}.pt',map_location='cpu',weights_only=False)
        if own['identity_sha256']!=identity['sha256']:
            raise ValueError('Rank identity changed')
        summary=validate_boundary(own['state'],own['rollout'],identity,common['optimizer'],common['scheduler'])
        transaction=json.loads((path/'transaction.json').read_text())
        if (transaction.get('schema')!=CHECKPOINT_CONTRACT or summary!=transaction['ranks'][rank]
                or complete['phase']!=summary['phase']):
            raise ValueError('Committed state/digests changed')
    except Exception as exc:
        error=repr(exc)
    check_errors(error)
    validate_ranks(gather(summary))
    own['state']['phase']=summary['phase']
    restore_adapters(actor,common['adapters']);optimizer.load_state_dict(common['optimizer']);scheduler.load_state_dict(common['scheduler'])
    rng=own['rng'];random.setstate(rng['python']);np.random.set_state(rng['numpy']);torch.set_rng_state(rng['torch'])
    if rng['cuda']is not None:
        torch.cuda.set_rng_state(rng['cuda'])
    generator.set_state(rng['rollout_generator'])
    return own['state'],device_tree(own['rollout'],actor.device)
