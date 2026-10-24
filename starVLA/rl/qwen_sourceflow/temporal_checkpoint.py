"""READY-only atomic full model + per-rank FP32 Adam shards; audited primitives."""
import json,os,random,uuid
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from starVLA.rl.simwam_recipe import checkpoint as common
from starVLA.rl.flow_grpo.transactions import atomic_json
from .contracts import file_sha
from .temporal_optimizer import local_parameter_names,validate_local_adam


def save(root,model,optimizer,scheduler,state,identity):
    rank=dist.get_rank();world=dist.get_world_size()
    error=None
    try:
        if state['phase']!='READY' or scheduler.last_epoch!=state['updates']:raise ValueError('Uncommitted local phase/scheduler')
        validate_local_adam(model,optimizer,state['updates'],identity['parent_steps'],identity['config'].get('head_only_updates',0))
    except Exception as e:error=repr(e)
    common.check_errors(error)
    summary=dict(state,identity=identity['sha256'])
    if any(x!=summary for x in common.gather(summary)):
        raise ValueError('Only a globally committed READY update can publish')
    nonce=common.gather(uuid.uuid4().hex if rank==0 else None)[0]
    final=Path(root)/'checkpoints'/f"update{state['updates']:06d}";attempt=final.with_name(final.name+'.incomplete.'+nonce)
    error=None
    try:
        if rank==0:
            if final.exists():raise FileExistsError(final)
            attempt.mkdir(parents=True,exist_ok=False)
    except Exception as e:error=repr(e)
    common.check_errors(error);error=None
    device=next(model.parameters()).device
    rng=dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
        cuda=torch.cuda.get_rng_state() if device.type=='cuda' else None)
    try:
        common.atomic_torch(attempt/f'rank{rank}.pt',dict(state=state,rng=rng,optimizer=optimizer.optim.state_dict(),
            local_parameter_names=local_parameter_names(model,optimizer),scheduler=scheduler.state_dict(),identity=identity['sha256']))
        if rank==0:
            common.atomic_torch(attempt/'model.pt',model.state_dict());atomic_json(attempt/'identity.json',identity)
    except Exception as e:
        if common.fatal_device_error(e):raise
        error=repr(e)
    common.check_errors(error);error=None
    try:
        if rank==0:
            files={p.name:file_sha(p) for p in attempt.iterdir() if p.is_file()}
            if set(files)!={'model.pt','identity.json'}|{f'rank{i}.pt' for i in range(world)}:raise ValueError('Missing optimizer/RNG shard')
            atomic_json(attempt/'files.json',files)
            atomic_json(attempt/'COMPLETE',dict(schema='temporal_dino_IL_READY_v1',world=world,
                **state,files_sha256=file_sha(attempt/'files.json')))
            os.rename(attempt,final);atomic_json(Path(root)/'latest.json',dict(path=str(final),updates=state['updates'],files_sha256=file_sha(final/'files.json')))
    except Exception as e:error=repr(e)
    common.check_errors(error);return str(final)


def restore(path,model,optimizer,scheduler,identity):
    path=Path(path);rank=dist.get_rank();seal=json.loads((path/'COMPLETE').read_text())
    if '.incomplete' in path.name or seal['schema']!='temporal_dino_IL_READY_v1' or seal['phase']!='READY' or seal['world']!=dist.get_world_size():
        raise ValueError('Unpublished/wrong-world temporal checkpoint')
    if file_sha(path/'files.json')!=seal['files_sha256'] or json.loads((path/'identity.json').read_text())!=identity:
        raise ValueError('Changed scientific/execution identity')
    files=json.loads((path/'files.json').read_text())
    for n in ['model.pt',f'rank{rank}.pt','identity.json']:
        if file_sha(path/n)!=files[n]:raise ValueError('Corrupt temporal checkpoint: '+n)
    model.load_state_dict(torch.load(path/'model.pt',map_location='cpu',weights_only=True,mmap=False),strict=True)
    own=torch.load(path/f'rank{rank}.pt',map_location='cpu',weights_only=False,mmap=False)
    if own['identity']!=identity['sha256'] or own['state']['phase']!='READY' or own['state']['updates']!=seal['updates']:
        raise ValueError('Uncommitted rank state')
    if own['local_parameter_names']!=local_parameter_names(model,optimizer):raise ValueError('Optimizer partition changed')
    optimizer.optim.load_state_dict(own['optimizer'])
    # Same LR phase on both local and exposed ZeRO groups. Tensor membership
    # differs across ranks; only optimizer hyperparameters are copied here.
    for local,outer in zip(optimizer.optim.param_groups,optimizer.param_groups):
        for k,v in local.items():
            if k!='params':outer[k]=v
    scheduler.load_state_dict(own['scheduler'])
    state=own['state']
    if any(s!=state for s in common.gather(state)):raise ValueError('Rank data cursors disagree')
    validate_local_adam(model,optimizer,state['updates'],identity['parent_steps'],identity['config'].get('head_only_updates',0))
    if scheduler.last_epoch!=state['updates']:raise ValueError('Scheduler progress changed')
    rng=own['rng'];random.setstate(rng['python']);np.random.set_state(rng['numpy']);torch.set_rng_state(rng['torch'])
    if rng['cuda'] is not None:torch.cuda.set_rng_state(rng['cuda'])
    return state


def retain_rolling(root,identity):
    if identity['qualification']:return
    root=Path(root).resolve()
    if not str(root).startswith('/mnt/project/qwen-sourceflow-temporal-dino-20261024/runs/'):
        raise ValueError('Retention applies only to this newly registered task')
    latest=Path(json.loads((root/'latest.json').read_text())['path']).resolve();eligible=[]
    for path in (root/'checkpoints').glob('update*'):
        if path.is_symlink() or not (path/'COMPLETE').is_file() or '.incomplete' in path.name:continue
        seal=json.loads((path/'COMPLETE').read_text())
        if seal['schema']!='temporal_dino_IL_READY_v1':raise ValueError('Foreign checkpoint in new run')
        if seal['updates']==0 or seal['updates'] in identity['config']['milestones']:continue
        if json.loads((path/'identity.json').read_text())!=identity:raise ValueError('Foreign checkpoint identity')
        eligible.append((seal['updates'],path))
    for _,path in sorted(eligible)[:-identity['config']['rolling_recovery_keep']]:
        if path.resolve()==latest:raise ValueError('Cannot retire latest')
        payloads=['model.pt']+[f'rank{r}.pt' for r in range(identity['world'])]
        for rank in range(identity['world']):
            old=torch.load(path/f'rank{rank}.pt',map_location='cpu',mmap=True,weights_only=False)
            common.atomic_torch(path/f'retained_rank{rank}.pt',{k:v for k,v in old.items() if k!='optimizer'})
        atomic_json(path/'RETIRED.json',dict(reason='registered two rolling plus milestones policy for new task only',
            replacement=str(latest),payload_bytes={n:(path/n).stat().st_size for n in payloads},identity=identity['sha256']))
        os.rename(path/'COMPLETE',path/'RETIRED_COMPLETE')
        for n in payloads:(path/n).unlink()
