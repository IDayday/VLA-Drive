"""Single-worker study boundaries using the project's existing atomic writers."""
import fcntl
import json
import os
from pathlib import Path
import random
import uuid
import numpy as np
import torch
from starVLA.rl.simwam_recipe.checkpoint import atomic_torch
from starVLA.rl.flow_grpo.transactions import atomic_json
from starVLA.rl.qwen_sourceflow.contracts import file_sha,fingerprint


def validate(state,payload):
    phase=state['phase'];inner=state['next_inner']
    if phase not in ('READY','ROLLOUT_COMMITTED','INNER_COMMITTED','COMPLETE') or not 0<=inner<=4:
        raise ValueError('Uncommitted collection/failure cannot become latest')
    for key in ('round','updates'):
        if not isinstance(state[key],int) or state[key]<0:raise ValueError('Invalid committed progress')
    if not state.get('arm') or not isinstance(state.get('seed'),int):raise ValueError('Missing scientific arm/seed')
    if phase in ('ROLLOUT_COMMITTED','INNER_COMMITTED'):
        if payload.get('batch') is None:raise ValueError('Pending inner requires fixed behavior batch')
    elif inner!=0 or payload.get('batch') is not None:
        raise ValueError('Ready boundary must not carry unresolved inner state')


def rng_state():
    return dict(torch=torch.get_rng_state(),numpy=np.random.get_state(),python=random.getstate(),
                cuda=torch.cuda.get_rng_state() if torch.cuda.is_available() else None)


def save_state(run,state,payload,protocol_sha):
    validate(state,payload);run=Path(run);root=run/'checkpoints';root.mkdir(parents=True,exist_ok=True)
    with (run/'checkpoint.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        name=f"round{state['round']:04d}_update{state['updates']:06d}_inner{state['next_inner']}_{uuid.uuid4().hex[:8]}"
        attempt=root/(name+'.incomplete');attempt.mkdir(exist_ok=False)
        scientific=dict(protocol_sha256=protocol_sha,arm=state['arm'],seed=state['seed'])
        atomic_torch(attempt/'payload.pt',dict(state=state,payload=payload,rng=rng_state(),scientific=scientific))
        seal=dict(schema='three_route_single_worker_boundary_v1',state=state,
                  scientific=scientific,payload_sha256=file_sha(attempt/'payload.pt'))
        atomic_json(attempt/'COMPLETE',seal)
        dest=root/name;os.rename(attempt,dest)
        atomic_json(run/'latest.json',dict(checkpoint=str(dest.resolve()),seal_sha256=file_sha(dest/'COMPLETE')))
        return dest


def load_state(path_or_run,protocol_sha,restore_rng=True):
    path=Path(path_or_run)
    if (path/'latest.json').exists():
        latest=json.loads((path/'latest.json').read_text());path=Path(latest['checkpoint'])
        if file_sha(path/'COMPLETE')!=latest['seal_sha256']:raise ValueError('Changed latest publication')
    if path.name.endswith('.incomplete'):raise ValueError('Never resume incomplete publication')
    seal=json.loads((path/'COMPLETE').read_text())
    if seal['scientific']['protocol_sha256']!=protocol_sha or file_sha(path/'payload.pt')!=seal['payload_sha256']:
        raise ValueError('Changed common protocol or payload')
    value=torch.load(path/'payload.pt',map_location='cpu',weights_only=False)
    if value['state']!=seal['state'] or value['scientific']!=seal['scientific']:raise ValueError('Boundary mismatch')
    validate(value['state'],value['payload'])
    if restore_rng:
        rng=value['rng'];torch.set_rng_state(rng['torch']);np.random.set_state(rng['numpy']);random.setstate(rng['python'])
        if rng['cuda'] is not None:torch.cuda.set_rng_state(rng['cuda'])
    return value['state'],value['payload']
