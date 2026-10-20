"""Small immutable receipts and dependency checks shared by both research routes."""
from __future__ import annotations
from contextlib import contextmanager
from pathlib import Path
import hashlib
import json
import os
import tempfile
import fcntl
import math


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),
                                    allow_nan=False).encode()).hexdigest()


def json_bytes(value):
    return (json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()


def atomic_bytes(path, data, *, immutable=True):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    if immutable and p.exists():
        if p.read_bytes()!=data:raise ValueError('Immutable content conflict: '+str(p))
        return
    fd,name=tempfile.mkstemp(prefix=p.name+'.partial.',dir=p.parent)
    try:
        with os.fdopen(fd,'wb') as f:f.write(data);f.flush();os.fsync(f.fileno())
        if immutable:
            try:os.link(name,p)
            except FileExistsError:
                if p.read_bytes()!=data:raise ValueError('Concurrent immutable conflict: '+str(p))
        else:os.replace(name,p)
    finally:
        Path(name).unlink(missing_ok=True)


def write_json(path,value,*,immutable=True):
    atomic_bytes(path,json_bytes(value),immutable=immutable)


def read_json(path):
    return json.loads(Path(path).read_text())


@contextmanager
def lock(path):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        yield


def artifact(path):
    p=Path(path).resolve()
    return {'path':str(p),'sha256':sha256(p),'bytes':p.stat().st_size}


def verify_artifact(row):
    p=Path(row['path'])
    if not p.is_absolute() or not p.is_file() or sha256(p)!=row['sha256']:
        raise ValueError('Missing/stale dependency: '+str(p))
    return p


def load_config(path):
    import yaml
    cfg=yaml.safe_load(Path(path).read_text())
    fields={'schema','campaign_id','base_binding','train_manifest','artifact_root','seed',
            'cohort_size','pilot_scenes','feature_shard_scenes','g0','resources','training'}
    if not isinstance(cfg,dict) or set(cfg)!=fields or cfg['schema']!='action_effect_campaign_v1':
        raise ValueError('Unresolved/unknown campaign configuration fields')
    if any(type(cfg[k]) is not int or cfg[k]<1 for k in ('cohort_size','pilot_scenes','feature_shard_scenes')):
        raise ValueError('Positive explicit scene/shard budgets required')
    if cfg['pilot_scenes']>cfg['cohort_size']:raise ValueError('Pilot exceeds locked population')
    for k in ('base_binding','train_manifest','artifact_root'):
        if not Path(cfg[k]).is_absolute():raise ValueError('Actual absolute preflight path required: '+k)
    if type(cfg['seed']) is not int or cfg['seed']<0:raise ValueError('Invalid seed')
    if not isinstance(cfg['campaign_id'],str) or not cfg['campaign_id'] or any(x in cfg['campaign_id'] for x in ('/','\\','..')):
        raise ValueError('Invalid campaign identity')
    groups={
        'g0':{'workers','max_simulation_budget','retry_budget','candidate_count','fixed_grid',
              'safe_score','minimum_progress_fraction','navsim_root','maps_root'},
        'resources':{'artifact_cap_bytes','minimum_free_bytes'},
        'training':{'hidden','layers','heads','batch_scenes','microbatch_scenes','max_updates',
            'warmup_updates','learning_rate','weight_decay','gradient_clip','validation_every',
            'training_seeds','finalist_seeds'}}
    for k,keys in groups.items():
        if not isinstance(cfg[k],dict) or set(cfg[k])!=keys:raise ValueError('Unknown/missing '+k+' configuration')
    def integer(value,low,high):
        if type(value) is not int or not low<=value<=high:raise ValueError('Integer budget out of range')
    def number(value,low,high):
        if type(value) not in (float,int) or not math.isfinite(value) or not low<=value<=high:
            raise ValueError('Numeric recipe value out of range')
    g=cfg['g0'];t=cfg['training'];r=cfg['resources']
    integer(g['workers'],1,32);integer(g['max_simulation_budget'],1,1000000)
    integer(g['retry_budget'],0,g['max_simulation_budget']);integer(g['candidate_count'],26,26)
    if g['fixed_grid']!='action_effect_quintic_intervention_v1':raise ValueError('Unsupported intervention schema')
    number(g['safe_score'],0,1);number(g['minimum_progress_fraction'],0,1)
    for k in ('navsim_root','maps_root'):
        if not isinstance(g[k],str) or not Path(g[k]).is_absolute():raise ValueError('Actual NAVSIM/maps path required')
    for k in r:integer(r[k],1,2**63-1)
    for k in ('hidden','layers','heads','batch_scenes','microbatch_scenes','max_updates','warmup_updates','validation_every'):
        integer(t[k],1,1000000)
    if t['hidden']%t['heads'] or t['microbatch_scenes']>t['batch_scenes'] or t['warmup_updates']>t['max_updates']:
        raise ValueError('Inconsistent training dimensions/budget')
    number(t['learning_rate'],1e-12,1);number(t['weight_decay'],0,1);number(t['gradient_clip'],1e-12,1e6)
    for k in ('training_seeds','finalist_seeds'):
        if not isinstance(t[k],list) or not t[k] or len(set(t[k]))!=len(t[k]):raise ValueError('Invalid seed population')
        for value in t[k]:integer(value,0,2**63-1)
    return cfg


def verify_identity(value):
    if not isinstance(value,dict) or value.get('identity')!=digest({k:v for k,v in value.items() if k!='identity'}):
        raise ValueError('Artifact content identity mismatch')
    return value


def check_space(cfg):
    p=Path(cfg['artifact_root']);p.mkdir(parents=True,exist_ok=True)
    s=os.statvfs(p);free=s.f_bavail*s.f_frsize
    if free<cfg['resources']['minimum_free_bytes']:
        raise RuntimeError('BLOCKED_STORAGE: shared free space below declared reservation')
    used=0
    for f in p.rglob('*'):
        try:
            if f.is_file() and not f.is_symlink():used+=f.stat().st_size
        except FileNotFoundError:continue
    if used>cfg['resources']['artifact_cap_bytes']:raise RuntimeError('BLOCKED_STORAGE: own artifact cap')
    return {'shared_free_bytes':free,'own_output_bytes':used}
