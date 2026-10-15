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
from .recovery_state import validate_boundary, validate_ranks

CONTROL_GROUP = None
EXECUTION_IDENTITY = None
MIGRATION_RECEIPT = None


def init_control_group():
    """CPU control plane, initialized while all workers are healthy."""
    global CONTROL_GROUP
    if dist.is_initialized():
        from datetime import timedelta
        CONTROL_GROUP = dist.new_group(backend='gloo', timeout=timedelta(seconds=90))


def gather(value):
    if not dist.is_initialized(): return [value]
    if CONTROL_GROUP is None and dist.get_backend() == 'nccl':
        raise RuntimeError('Recovery control group must be initialized before training')
    result = [None] * dist.get_world_size()
    dist.all_gather_object(result, value, group=CONTROL_GROUP)
    return result


def check_errors(error):
    errors = gather(error)
    if any(e is not None for e in errors):
        raise RuntimeError('Uncommitted distributed transaction: ' + repr(errors))


def fatal_device_error(error):
    text = str(error).lower()
    return any(x in text for x in ('cuda error', 'device-side assert', 'illegal memory access',
               'nccl', 'collective', 'processgroup', 'connection closed by peer'))


def failure_evidence(root, state, error, invocation, *, rollout=None):
    """Local only. Never COMPLETE/latest, never a collective after CUDA damage."""
    rank = dist.get_rank() if dist.is_initialized() else 0
    target = Path(root)/'failures'/invocation/(f'rank{rank}-' + uuid.uuid4().hex)
    target.mkdir(parents=True, exist_ok=False)
    atomic_json(target/'evidence.json', dict(schema='simwam_failure_v2', recoverable=False,
                state=state, error=str(error), invocation=invocation,
                latest_unchanged=True, time_unix=time.time()))
    if rollout is not None and not fatal_device_error(error):
        try: atomic_torch(target/'partial_rollout.pt', rollout)
        except Exception as secondary:
            atomic_json(target/'evidence_write_error.json', dict(error=repr(secondary)))
    return str(target)


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


def save(root,actor,optimizer,scheduler,generator,state,rollout,identity,*,tag=None, fault=None):
    rank=dist.get_rank() if dist.is_initialized() else 0
    world=dist.get_world_size() if dist.is_initialized() else 1
    name=tag or f"update{state['updates']:06}_inner{state['next_inner']}"
    final=Path(root)/'checkpoints'/name
    error=None; summary=None
    try: summary=validate_boundary(state,rollout,identity,optimizer,scheduler)
    except Exception as exc: error=repr(exc)
    check_errors(error)
    summaries=validate_ranks(gather(summary))
    attempt_id=gather(uuid.uuid4().hex if rank==0 else None)[0]
    attempt=final.with_name(name+'.incomplete.'+attempt_id)
    error=None
    try:
        if rank==0:
            if final.exists():raise FileExistsError('Checkpoint identity must not be overwritten: '+str(final))
            attempt.mkdir(parents=True,exist_ok=False)
    except Exception as exc:error=repr(exc)
    check_errors(error)
    rng=dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
             cuda=torch.cuda.get_rng_state() if torch.cuda.is_available() else None,rollout_generator=generator.get_state())
    error=None
    try:
        if fault: fault('rank_write',rank)
        atomic_torch(attempt/f'rank{rank}.pt',dict(state=state,rollout=rollout,rng=rng,identity_sha256=identity['sha256']))
        if rank==0:
            atomic_torch(attempt/'optimizer.pt',dict(adapters=adapter_state(actor),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict()))
            atomic_json(attempt/'identity.json',identity)
            atomic_json(attempt/'transaction.json',dict(schema='simwam_checkpoint_transaction_v2',attempt=attempt_id,ranks=summaries,
                execution_identity=EXECUTION_IDENTITY,migration_receipt_sha256=MIGRATION_RECEIPT['sha256'] if MIGRATION_RECEIPT else None))
    except Exception as exc:
        if fatal_device_error(exc):raise
        error=repr(exc)
    check_errors(error)
    error=None
    try:
        if rank==0:
            files={p.name:file_sha(p) for p in sorted(attempt.iterdir()) if p.is_file()}
            required={'optimizer.pt','identity.json','transaction.json'}|{f'rank{r}.pt' for r in range(world)}
            if not required.issubset(files):raise ValueError('Incomplete checkpoint files')
            atomic_json(attempt/'files.json',files)
            if fault:fault('before_complete',rank)
            atomic_json(attempt/'COMPLETE',dict(schema=2,phase=state['phase'],world=world,updates=state['updates'],next_inner=state['next_inner'],files_sha256=file_sha(attempt/'files.json')))
            os.rename(attempt,final)
            atomic_json(Path(root)/'latest.json',dict(checkpoint=str(final),updates=state['updates'],files_sha256=file_sha(final/'files.json')))
    except Exception as exc:error=repr(exc)
    check_errors(error)
    return str(final)


def restore(path,actor,optimizer,scheduler,generator,identity,*,migration=None):
    rank=dist.get_rank() if dist.is_initialized() else 0
    world=dist.get_world_size() if dist.is_initialized() else 1
    error=None;summary=None
    try:
        path=Path(path);complete=json.loads((path/'COMPLETE').read_text())
        legacy=complete.get('schema')!=2
        if '.incomplete' in path.name or (legacy and migration is None):
            raise ValueError('Unpublished or legacy checkpoint: explicit audited migration required')
        if complete['world']!=world or file_sha(path/'files.json')!=complete['files_sha256']:raise ValueError('Checkpoint publication changed')
        if json.loads((path/'identity.json').read_text())!=identity:raise ValueError('Exact resume identity mismatch')
        files=json.loads((path/'files.json').read_text())
        for n in ['optimizer.pt',f'rank{rank}.pt','identity.json']+([] if legacy else ['transaction.json']):
            if file_sha(path/n)!=files[n]:raise ValueError('Checkpoint content changed: '+n)
        # These are our sealed local native optimizer/RNG states, never downloads.
        common=torch.load(path/'optimizer.pt',map_location='cpu',weights_only=False)
        own=torch.load(path/f'rank{rank}.pt',map_location='cpu',weights_only=False)
        if own['identity_sha256']!=identity['sha256']:raise ValueError('Rank identity changed')
        summary=validate_boundary(own['state'],own['rollout'],identity,common['optimizer'],common['scheduler'],legacy=legacy)
        if not legacy:
            transaction=json.loads((path/'transaction.json').read_text())
            if summary!=transaction['ranks'][rank] or complete['phase']!=summary['phase']:
                raise ValueError('Committed state/digests changed')
    except Exception as exc:error=repr(exc)
    check_errors(error)
    validate_ranks(gather(summary))
    own['state']['phase']=summary['phase']
    restore_adapters(actor,common['adapters']);optimizer.load_state_dict(common['optimizer']);scheduler.load_state_dict(common['scheduler'])
    rng=own['rng'];random.setstate(rng['python']);np.random.set_state(rng['numpy']);torch.set_rng_state(rng['torch'])
    if rng['cuda'] is not None:torch.cuda.set_rng_state(rng['cuda'])
    generator.set_state(rng['rollout_generator'])
    return own['state'],device_tree(own['rollout'],actor.device)
