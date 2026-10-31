"""Independent Safety exploration task grant; existing host withdrawals win."""
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import socket
import time

ROOT=Path(os.environ.get('WORLD_COMPARE_ROOT','/mnt/project/qwen-world-safety-exploration-20261031'))
TASK=os.environ.get('WORLD_COMPARE_TASK','qwen_world_safety_exploration_20261031')
POLICY=Path('/mnt/project/server_dispatch_policy.json')


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as f:
        for data in iter(lambda:f.read(16*1024**2),b''):h.update(data)
    return h.hexdigest()


def authorization():
    if TASK!='qwen_world_safety_exploration_20261031':raise PermissionError('Wrong explicit task environment')
    policy=json.loads(POLICY.read_text());grant=policy['task_authorizations'][TASK]
    original=json.loads(Path(grant['record']).read_text())
    for key in ('permanent_hosts','temporary_hosts','unbounded_temporary_hosts','shared_training_hosts',
                'starts_unix','expires_unix','host_authorization_origin_unix'):
        if grant.get(key)!=original.get(key):raise PermissionError('Task grant differs from sealed record: '+key)
    return grant,policy


def allowed_hosts(*,at=None,admit_long_job=False):
    grant,policy=authorization();now=time.time()if at is None else at
    cutoff=grant['expires_unix']-(grant['checkpoint_guard_seconds']if admit_long_job else 0)
    hosts=list(grant['permanent_hosts'])+list(grant.get('unbounded_temporary_hosts',[]))
    if grant['starts_unix']<=now<cutoff:hosts+=grant['temporary_hosts']
    origin=grant.get('host_authorization_origin_unix',grant['starts_unix']);result=[]
    for host in hosts:
        ban=policy.get('global_dispatch_prohibitions',{}).get(host,{})
        if ban.get('new_task_dispatch_allowed')is False and ban.get('utc'):
            if datetime.fromisoformat(ban['utc'].replace('Z','+00:00')).timestamp()>origin:continue
        result.append(host)
    return result


def require_dispatch(*,host=None,admit_long_job=False):
    host=host or socket.gethostname().removesuffix('-worker-0')
    if host not in allowed_hosts(admit_long_job=admit_long_job):raise PermissionError('Host not authorized: '+host)
    return host


def boundary_stop_due(host=None,*,now=None):
    stop=os.environ.get('WORLD_POSTCOT_STOP_FILE')
    if stop and Path(stop).exists():return True
    host=host or socket.gethostname().removesuffix('-worker-0');grant,_=authorization()
    if host not in allowed_hosts(at=now):return True
    if host in grant['permanent_hosts']or host in grant.get('unbounded_temporary_hosts',[]):return False
    return (time.time()if now is None else now)>=grant['expires_unix']-grant['checkpoint_guard_seconds']
