"""Native IL state, reusing audited atomic publication and control collectives."""
import json
import os
from pathlib import Path
import random
import uuid
import numpy as np
import torch
import torch.distributed as dist
from starVLA.rl.simwam_recipe import checkpoint as common
from starVLA.rl.flow_grpo.transactions import atomic_json
from .contracts import file_sha
from .il_optimization import validate_optimizer


def save(root,model,optimizer,scheduler,state,identity):
    rank=dist.get_rank() if dist.is_initialized() else 0
    world=dist.get_world_size() if dist.is_initialized() else 1
    validate_optimizer(model,optimizer,scheduler,state,identity['config'])
    summary=dict(updates=state['updates'],cursor=state['cursor'],phase=state['phase'],identity=identity['sha256'])
    if any(v!=summary for v in common.gather(summary)):
        raise ValueError('Ranks disagree on committed IL boundary')
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
    rng=dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state())
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
            if set(files)!={'model.pt','optimizer.pt','identity.json'}|{f'rank{i}.pt' for i in range(world)}:
                raise ValueError('Incomplete IL publication')
            atomic_json(attempt/'files.json',files)
            atomic_json(attempt/'COMPLETE',dict(schema='native_qwen_flow_IL_READY_v1',phase='READY',world=world,
                updates=state['updates'],cursor=state['cursor'],files_sha256=file_sha(attempt/'files.json')))
            os.rename(attempt,final)
            atomic_json(Path(root)/'latest.json',dict(path=str(final),updates=state['updates'],files_sha256=file_sha(final/'files.json')))
    except Exception as exc:error=repr(exc)
    common.check_errors(error)
    return str(final)


def retain_rolling(root, identity):
    """Only NEW native IL rolling payloads expire; all milestone/failure data stay.

    Called after latest is atomically published. Small manifests, hashes, rank
    RNG/cursors and an explicit retirement receipt survive payload retirement.
    Qualification states and every pre-existing experiment are excluded.
    """
    if identity['qualification']:return
    root=Path(root).resolve();cfg=identity['config']
    latest=Path(json.loads((root/'latest.json').read_text())['path']).resolve()
    eligible=[]
    for path in (root/'checkpoints').glob('update*'):
        if path.is_symlink() or '.incomplete' in path.name or not (path/'COMPLETE').is_file():continue
        seal=json.loads((path/'COMPLETE').read_text())
        if seal['schema']!='native_qwen_flow_IL_READY_v1' or seal['updates'] in cfg['milestones']:continue
        if json.loads((path/'identity.json').read_text())!=identity:raise ValueError('Foreign checkpoint in native run')
        eligible.append((seal['updates'],path))
    eligible.sort()
    for _,path in eligible[:-cfg['rolling_recovery_keep']]:
        if path.resolve()==latest:raise ValueError('Cannot retire current latest')
        sizes={n:(path/n).stat().st_size for n in ['model.pt','optimizer.pt']}
        atomic_json(path/'RETIRED.json',dict(reason='pre-registered native IL rolling retention',
            retain_complete_rolling=cfg['rolling_recovery_keep'],replacement=str(latest),
            identity=identity['sha256'],payloads=sizes,files_sha256=file_sha(path/'files.json')))
        os.rename(path/'COMPLETE',path/'RETIRED_COMPLETE')
        for name in sizes:(path/name).unlink()


def restore(path,model,optimizer,scheduler,identity):
    path=Path(path);rank=dist.get_rank() if dist.is_initialized() else 0
    world=dist.get_world_size() if dist.is_initialized() else 1
    seal=json.loads((path/'COMPLETE').read_text())
    if '.incomplete' in path.name or seal['phase']!='READY' or seal['world']!=world:
        raise ValueError('Unpublished/incompatible native IL boundary')
    if json.loads((path/'identity.json').read_text())!=identity or file_sha(path/'files.json')!=seal['files_sha256']:
        raise ValueError('IL scientific or source identity changed')
    files=json.loads((path/'files.json').read_text())
    for name in ('model.pt','optimizer.pt',f'rank{rank}.pt'):
        if file_sha(path/name)!=files[name]:raise ValueError('Corrupt IL checkpoint '+name)
    model.load_state_dict(torch.load(path/'model.pt',map_location='cpu',mmap=True,weights_only=True),strict=True)
    payload=torch.load(path/'optimizer.pt',map_location='cpu',mmap=True,weights_only=False)
    optimizer.load_state_dict(payload['optimizer']);scheduler.load_state_dict(payload['scheduler'])
    own=torch.load(path/f'rank{rank}.pt',map_location='cpu',weights_only=False)
    if own['identity']!=identity['sha256']:raise ValueError('Changed rank ancestry')
    if (own['state']['updates']!=seal['updates'] or own['state']['cursor']!=seal['cursor'] or
            own['state']['phase']!=seal['phase']):
        raise ValueError('Rank state differs from the published IL boundary')
    if any(v!=own['state'] for v in common.gather(own['state'])):
        raise ValueError('IL recovery ranks disagree')
    validate_optimizer(model,optimizer,scheduler,own['state'],identity['config'])
    rng=own['rng'];random.setstate(rng['python']);np.random.set_state(rng['numpy']);torch.set_rng_state(rng['torch']);torch.cuda.set_rng_state(rng['cuda'])
    return own['state']
