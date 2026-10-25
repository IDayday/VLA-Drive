"""ZeRO2 native full-state boundary save, atomic seal, and same-run resume."""
import json
import os
from pathlib import Path
import random
import shutil
import uuid
import numpy as np
import torch
import torch.distributed as dist
from .contracts import file_sha,fingerprint
from .loading import inspect_checkpoint

def rank():return dist.get_rank() if dist.is_initialized() else 0
def barrier():
    if dist.is_initialized():dist.barrier()
def canonical_identity(value):
    # Qwen config contains integer id2label keys; JSON receipts stringify them.
    # Normalize both live and persisted identities without removing any field.
    return json.loads(json.dumps(value,sort_keys=True,allow_nan=False))
def identity_diff(expected,actual,prefix=''):
    if isinstance(expected,dict) and isinstance(actual,dict):
        result=[]
        for key in sorted(set(expected)|set(actual)):
            name=prefix+'.'+str(key) if prefix else str(key)
            if key not in expected or key not in actual:result.append(dict(field=name,expected=expected.get(key),actual=actual.get(key)))
            else:result.extend(identity_diff(expected[key],actual[key],name))
        return result
    if expected!=actual:return [dict(field=prefix,expected=expected,actual=actual)]
    return []
def atomic_json(path,data):
    path=Path(path);tmp=path.with_name(path.name+'.tmp.'+str(os.getpid()))
    with tmp.open('w') as stream:
        json.dump(data,stream,indent=2,allow_nan=False);stream.flush();os.fsync(stream.fileno())
    os.replace(tmp,path)
def rng_state():return dict(python=random.getstate(),numpy=np.random.get_state(),cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state() if torch.cuda.is_available() else None)
def restore_rng(state):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['cpu'])
    if state['cuda'] is not None:torch.cuda.set_rng_state(state['cuda'])

def _rank_zero_call(function):
    """Propagate metadata/setup failures instead of stranding peers at a barrier."""
    result=[None]
    if rank()==0:
        try:result[0]=dict(value=function())
        except Exception as error:result[0]=dict(error=f'{type(error).__name__}: {error}')
    if dist.is_initialized():dist.broadcast_object_list(result,src=0)
    if 'error' in result[0]:raise RuntimeError(result[0]['error'])
    return result[0]['value']

def save(out,engine,state,identity,builder):
    identity=canonical_identity(identity)
    out=Path(out);tag=f"update{state['updates']:06d}";final=out/'checkpoints'/tag
    def begin():
        if final.exists():raise ValueError('Refusing to overwrite a sealed checkpoint')
        # A previously interrupted partial save is retained as evidence, but
        # cannot prevent a fresh save of the same valid optimizer boundary.
        staging=out/'.saving'/(tag+'.'+uuid.uuid4().hex)
        staging.mkdir(parents=True,exist_ok=False)
        return str(staging)
    staging=Path(_rank_zero_call(begin))
    client=dict(state=dict(state),builder=builder.state_dict(),rng=rng_state(),identity_sha256=fingerprint(identity))
    success=engine.save_checkpoint(str(staging),tag='native',client_state={},save_latest=False,exclude_frozen_parameters=False)
    if success is False:raise RuntimeError('Native DeepSpeed checkpoint save failed')
    with (staging/f'rank{rank():02d}_state.pt').open('wb') as stream:
        torch.save(client,stream);stream.flush();os.fsync(stream.fileno())
    barrier()
    def commit():
        world=dist.get_world_size() if dist.is_initialized() else 1
        if identity['world']!=world:raise ValueError('Checkpoint world identity differs from native save topology')
        for i in range(world):
            if not (staging/f'rank{i:02d}_state.pt').is_file():raise ValueError('Missing rank recovery state')
        if len(list((staging/'native').glob('*_optim_states.pt')))!=world:raise ValueError('Missing native ZeRO2 optimizer shard')
        if not (staging/'native/mp_rank_00_model_states.pt').is_file():raise ValueError('Missing native full model state')
        atomic_json(staging/'identity.json',identity)
        files={str(p.relative_to(staging)):file_sha(p) for p in staging.rglob('*') if p.is_file() and p.name!='identity.json'}
        atomic_json(staging/'files.json',files)
        atomic_json(staging/'COMPLETE',dict(status='COMPLETE',updates=state['updates'],cursor=state['cursor'],
            phase=state['phase'],model_file='native/mp_rank_00_model_states.pt',files_sha256=file_sha(staging/'files.json'),identity_sha256=fingerprint(identity)))
        final.parent.mkdir(exist_ok=True);os.rename(staging,final)
        atomic_json(out/'LATEST.json',dict(checkpoint=str(final.resolve()),updates=state['updates']))
        return str(final.resolve())
    _rank_zero_call(commit)
    barrier();return str(final.resolve())

def restore(path,engine,identity,builder):
    identity=canonical_identity(identity)
    # Verify all large immutable shards once on rank0, not world times from
    # shared storage. Native engine then reads its own complete optimizer state.
    info=_rank_zero_call(lambda:inspect_checkpoint(path))
    if info['identity']!=identity:raise ValueError('Resume identity differs from this training run')
    loaded,_=engine.load_checkpoint(str(Path(path)),tag='native',load_optimizer_states=True,load_lr_scheduler_states=True,
        load_module_strict=True,load_module_only=False)
    if loaded is None:raise RuntimeError('Native DeepSpeed did not load the requested checkpoint')
    client=torch.load(Path(path)/f'rank{rank():02d}_state.pt',map_location='cpu',weights_only=False)
    if client['identity_sha256']!=fingerprint(identity):raise ValueError('Rank recovery identity changed')
    if any(client['state'][key]!=info['seal'][key] for key in ('updates','cursor','phase')):raise ValueError('Rank cursor/stage differs from the complete checkpoint receipt')
    if hasattr(engine,'global_steps') and engine.global_steps!=client['state']['updates']:raise ValueError('Native optimizer update cursor differs from recovery cursor')
    builder.load_state_dict(client['builder']);restore_rng(client['rng']);return client['state']

def retain_rolling(out,keep=2):
    """Only this task's sealed nonmilestone checkpoints; never user/parent assets."""
    out=Path(out);values=[]
    for p in (out/'checkpoints').glob('update*'):
        if not (p/'COMPLETE').exists():continue
        seal=json.loads((p/'COMPLETE').read_text());step=seal['updates']
        if step%5000:values.append((step,p))
    for _,p in sorted(values)[:-keep]:shutil.rmtree(p)
