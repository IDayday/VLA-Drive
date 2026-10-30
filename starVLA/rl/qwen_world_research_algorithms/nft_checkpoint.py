"""Explicit NFT recovery transactions; compatible inference tensor inventory.

The surrounding native files match the World deployment loader. Recovery is
deliberately independent of simwam_recipe.recovery_state: NFT has no SDE
chain, old logprob, or PPO advantage. EMA, pending ODE samples, all RNG and
optimizer/scheduler progress are part of every complete transaction.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import random
import uuid
import numpy as np
import torch

from starVLA.rl.simwam_recipe import checkpoint as shared
from starVLA.rl.simwam_recipe.lora_adapter import adapter_state,restore_adapters
from starVLA.rl.simwam_recipe.recovery_state import tensor_digest
from starVLA.rl.qwen_world_postcot.contracts import fingerprint,file_sha
from starVLA.rl.flow_grpo.transactions import atomic_json


def rank_world():
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank(),torch.distributed.get_world_size()
    return 0,1


def validate(state,rollout,ema,identity,optimizer=None,scheduler=None):
    u,c,i=(state[k] for k in ('updates','collections','next_inner'))
    world=identity['world']; maximum=identity['recipe']['max_steps']
    if not (0<=u<=maximum and c>=0 and 0<=i<4 and i==u%4):
        raise ValueError('Invalid NFT update/inner boundary')
    pending=i!=0
    if c!=u//4+int(pending) or pending!=(rollout is not None):
        raise ValueError('NFT collection/pending boundary mismatch')
    if ema.updates!=u//4 or state['ema_updates']!=ema.updates:
        raise ValueError('Behavior EMA must update once per complete four-inner collection')
    if state['scene_cursor']!=c*world or state['raw_scene_exposures']!=c*world or state['new_candidates']!=c*world*8:
        raise ValueError('NFT fresh scene/candidate accounting changed')
    expected_phase='COMPLETE' if u==maximum else 'INNER_COMMITTED' if pending else 'READY'
    if state['phase']!=expected_phase:
        raise ValueError('NFT recoverable phase mismatch')
    digest=None
    if pending:
        if rollout.get('schema')!='world_nft_ODE_collection_v1' or rollout['ema_version']!=ema.updates:
            raise ValueError('NFT pending samples do not belong to the fixed behavior EMA')
        for name,shape in [('clean_actions',(8,8,3)),('rewards',(1,8)),('optimality',(8,))]:
            v=rollout[name]
            if tuple(v.shape)!=shape or v.requires_grad or not bool(torch.isfinite(v).all()):
                raise ValueError('Invalid NFT pending payload: '+name)
        if bool(((rollout['optimality']<0)|(rollout['optimality']>1)).any()):
            raise ValueError('Pending NFT optimality outside [0,1]')
        digest=tensor_digest({k:rollout[k] for k in ('clean_actions','rewards','optimality','cond_bg','ema_version')})
    if optimizer is not None:
        opt=optimizer.state_dict() if hasattr(optimizer,'state_dict') else optimizer
        steps=[float(v['step']) for v in opt['state'].values()]
        if (u and len(steps)!=sum(len(g['params']) for g in opt['param_groups'])) or any(s!=u for s in steps):
            raise ValueError('NFT optimizer state/update progress mismatch')
    if scheduler is not None:
        sched=scheduler.state_dict() if hasattr(scheduler,'state_dict') else scheduler
        if sched['last_epoch']!=u:
            raise ValueError('NFT scheduler progress mismatch')
    return dict(run=identity['sha256'],phase=state['phase'],updates=u,collections=c,
                next_inner=i,ema_updates=ema.updates,scene_cursor=state['scene_cursor'],pending=pending,
                pending_payload_sha256=digest)


def _rank_consistency(summaries):
    fields=('run','phase','updates','collections','next_inner','ema_updates','scene_cursor','pending')
    common={k:summaries[0][k] for k in fields}
    if any({k:s[k] for k in fields}!=common for s in summaries):
        raise ValueError('NFT ranks disagree on the committed boundary')


def save(root,actor,optimizer,scheduler,generator,state,rollout,ema,identity):
    rank,world=rank_world(); root=Path(root)
    error=None; summary=None
    try:summary=validate(state,rollout,ema,identity,optimizer,scheduler)
    except Exception as exc:error=repr(exc)
    shared.check_errors(error)
    summaries=shared.gather(summary);_rank_consistency(summaries)
    name=f"update{state['updates']:06}_inner{state['next_inner']}"
    final=root/'checkpoints'/name
    attempt_id=shared.gather(uuid.uuid4().hex if rank==0 else None)[0]
    staging=final.with_name(name+'.incomplete.'+attempt_id)
    error=None
    try:
        if rank==0:
            if final.exists():raise FileExistsError('NFT checkpoint cannot overwrite a published boundary')
            staging.mkdir(parents=True,exist_ok=False)
    except Exception as exc:error=repr(exc)
    shared.check_errors(error)
    rng=dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
             cuda=torch.cuda.get_rng_state(actor.device) if actor.device.type=='cuda' else None,
             rollout_generator=generator.get_state())
    error=None
    try:
        shared.atomic_torch(staging/f'rank{rank}.pt',dict(state=state,rollout=rollout,rng=rng,
                                                       identity_sha256=identity['sha256']))
        if rank==0:
            shared.atomic_torch(staging/'optimizer.pt',dict(adapters=adapter_state(actor),
                optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),behavior_ema=ema.state_dict()))
            atomic_json(staging/'identity.json',identity)
            atomic_json(staging/'transaction.json',dict(schema='world_nft_checkpoint_transaction_v1',
                protocol='world_nft_forward_v1',attempt=attempt_id,ranks=summaries,
                no_SDE_chain_or_old_logprob=True))
    except Exception as exc:error=repr(exc)
    shared.check_errors(error)
    error=None
    try:
        if rank==0:
            files={p.name:file_sha(p) for p in sorted(staging.iterdir()) if p.is_file()}
            required={'optimizer.pt','identity.json','transaction.json'}|{f'rank{r}.pt' for r in range(world)}
            if not required.issubset(files):raise ValueError('Missing native NFT checkpoint state')
            atomic_json(staging/'files.json',files)
            atomic_json(staging/'COMPLETE',dict(schema=2,training_protocol='world_nft_forward_v1',
                phase=state['phase'],world=world,updates=state['updates'],next_inner=state['next_inner'],
                files_sha256=file_sha(staging/'files.json')))
            os.rename(staging,final)
            atomic_json(root/'latest.json',dict(checkpoint=str(final),updates=state['updates'],
                                               files_sha256=file_sha(final/'files.json')))
    except Exception as exc:error=repr(exc)
    shared.check_errors(error)
    return str(final)


def restore(path,actor,optimizer,scheduler,generator,ema,identity):
    path=Path(path);rank,world=rank_world();error=None;own=common=None
    try:
        seal=json.loads((path/'COMPLETE').read_text())
        if ('.incomplete.' in path.name or seal.get('training_protocol')!='world_nft_forward_v1'
                or seal['world']!=world or seal['files_sha256']!=file_sha(path/'files.json')):
            raise ValueError('Not a complete matching NFT native checkpoint')
        stored=json.loads((path/'identity.json').read_text());body=dict(stored)
        if body.pop('sha256',None)!=fingerprint(body) or stored!=identity:
            raise ValueError('NFT exact-resume scientific/source identity mismatch')
        files=json.loads((path/'files.json').read_text())
        for name in ('optimizer.pt',f'rank{rank}.pt','identity.json','transaction.json'):
            p=path/name
            if not p.resolve().is_relative_to(path.resolve()) or file_sha(p)!=files[name]:
                raise ValueError('Changed NFT native recovery payload: '+name)
        transaction=json.loads((path/'transaction.json').read_text())
        if transaction['schema']!='world_nft_checkpoint_transaction_v1':
            raise ValueError('Cannot restore an old SDE transaction as NFT')
        common=torch.load(path/'optimizer.pt',map_location='cpu',weights_only=False)
        own=torch.load(path/f'rank{rank}.pt',map_location='cpu',weights_only=False)
        if own['identity_sha256']!=identity['sha256']:
            raise ValueError('NFT rank identity changed')
        restore_adapters(actor,common['adapters']);optimizer.load_state_dict(common['optimizer'])
        scheduler.load_state_dict(common['scheduler']);ema.load_state_dict(common['behavior_ema'],actor)
        state,pending=own['state'],own['rollout']
        summary=validate(state,pending,ema,identity,optimizer,scheduler)
        if summary!=transaction['ranks'][rank] or seal['phase']!=summary['phase']:
            raise ValueError('NFT rank transaction boundary differs')
    except Exception as exc:error=repr(exc)
    shared.check_errors(error)
    rng=own['rng'];random.setstate(rng['python']);np.random.set_state(rng['numpy']);torch.set_rng_state(rng['torch'])
    if rng['cuda'] is not None:torch.cuda.set_rng_state(rng['cuda'],actor.device)
    generator.set_state(rng['rollout_generator'])
    return own['state'],shared.device_tree(own['rollout'],actor.device)
