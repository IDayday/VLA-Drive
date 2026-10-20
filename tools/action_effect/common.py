"""Small immutable receipts and dependency checks shared by both research routes."""
from __future__ import annotations
from contextlib import contextmanager
from pathlib import Path
import hashlib
import json
import os
import tempfile
import fcntl


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
    return cfg


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
