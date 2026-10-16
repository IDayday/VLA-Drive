"""Full SFT state at optimizer boundaries, using the audited atomic primitives."""
import json
import os
from pathlib import Path
import random
import uuid
import numpy as np
import torch
import torch.distributed as dist
from starVLA.rl.simwam_recipe import checkpoint as common
from starVLA.rl.simwam_recipe.contracts import file_sha
from starVLA.rl.flow_grpo.transactions import atomic_json


def validate(state, identity, optimizer, scheduler):
    updates=state['updates']
    if state['phase']!='READY' or state['cursor']!=updates*identity['config']['global_batch']:
        raise ValueError('Uncommitted SFT cursor/update boundary')
    if scheduler.state_dict()['last_epoch']!=updates:
        raise ValueError('Scheduler progress differs from optimizer boundary')
    if updates:
        if not optimizer.state or any(int(v['step'])!=updates for v in optimizer.state.values()):
            raise ValueError('Adam moments do not match the committed update')
    return dict(updates=updates,cursor=state['cursor'],phase=state['phase'],identity=identity['sha256'])


def save(root, model, optimizer, scheduler, state, identity):
    rank=dist.get_rank() if dist.is_initialized() else 0
    world=dist.get_world_size() if dist.is_initialized() else 1
    summary=validate(state,identity,optimizer,scheduler)
    summaries=common.gather(summary)
    if any(x!=summary for x in summaries):raise ValueError('Ranks disagree on the SFT boundary')
    nonce=common.gather(uuid.uuid4().hex if rank==0 else None)[0]
    final=Path(root)/'checkpoints'/f"update{state['updates']:06d}"
    attempt=final.with_name(final.name+'.incomplete.'+nonce)
    error=None
    try:
        if rank==0:
            if final.exists():raise FileExistsError(final)
            attempt.mkdir(parents=True,exist_ok=False)
    except Exception as exc:error=repr(exc)
    common.check_errors(error)
    rng=dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
        cuda=torch.cuda.get_rng_state() if torch.cuda.is_available() else None)
    error=None
    try:
        common.atomic_torch(attempt/f'rank{rank}.pt',dict(state=state,rng=rng,identity=identity['sha256']))
        if rank==0:
            common.atomic_torch(attempt/'model.pt',model.state_dict())
            common.atomic_torch(attempt/'optimizer.pt',dict(optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict()))
            atomic_json(attempt/'identity.json',identity)
    except Exception as exc:
        if common.fatal_device_error(exc):raise
        error=repr(exc)
    common.check_errors(error)
    error=None
    try:
        if rank==0:
            files={p.name:file_sha(p) for p in attempt.iterdir() if p.is_file()}
            required={'model.pt','optimizer.pt','identity.json'}|{f'rank{r}.pt' for r in range(world)}
            if set(files)!=required:raise ValueError('Incomplete SFT checkpoint')
            atomic_json(attempt/'files.json',files)
            atomic_json(attempt/'COMPLETE',dict(schema='fm_transfer_SFT_READY_v1',phase='READY',
                world=world,updates=state['updates'],cursor=state['cursor'],files_sha256=file_sha(attempt/'files.json')))
            os.rename(attempt,final)
            atomic_json(Path(root)/'latest.json',dict(path=str(final),updates=state['updates'],files_sha256=file_sha(final/'files.json')))
    except Exception as exc:error=repr(exc)
    common.check_errors(error)
    return str(final)


def restore(path, model, optimizer, scheduler, identity):
    path=Path(path);rank=dist.get_rank() if dist.is_initialized() else 0
    world=dist.get_world_size() if dist.is_initialized() else 1
    seal=json.loads((path/'COMPLETE').read_text())
    if '.incomplete' in path.name or seal['world']!=world or seal['phase']!='READY':
        raise ValueError('Not a compatible committed SFT boundary')
    if file_sha(path/'files.json')!=seal['files_sha256'] or json.loads((path/'identity.json').read_text())!=identity:
        raise ValueError('SFT recovery identity changed')
    files=json.loads((path/'files.json').read_text())
    for name in ('model.pt','optimizer.pt',f'rank{rank}.pt'):
        if file_sha(path/name)!=files[name]:raise ValueError('Corrupt recovery file: '+name)
    model.load_state_dict(torch.load(path/'model.pt',map_location='cpu',weights_only=True,mmap=True),strict=True)
    value=torch.load(path/'optimizer.pt',map_location='cpu',weights_only=False,mmap=True)
    optimizer.load_state_dict(value['optimizer']);scheduler.load_state_dict(value['scheduler'])
    own=torch.load(path/f'rank{rank}.pt',map_location='cpu',weights_only=False)
    if own['identity']!=identity['sha256']:raise ValueError('Rank ancestry changed')
    summary=validate(own['state'],identity,optimizer,scheduler)
    if any(x!=summary for x in common.gather(summary)):raise ValueError('Recovered rank boundary differs')
    rng=own['rng'];random.setstate(rng['python']);np.random.set_state(rng['numpy']);torch.set_rng_state(rng['torch'])
    if rng['cuda'] is not None:torch.cuda.set_rng_state(rng['cuda'])
    return own['state']
