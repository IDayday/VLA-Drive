"""Finite, frozen NAVTEST capability; never grants TRAIN or model updates."""
import json
import os
from pathlib import Path
import socket
import sys
import time
from starVLA.rl.qwen_sourceflow.contracts import fingerprint,file_sha


ROOT=Path('/mnt/project/qwen-flow-three-routes-20261024')
TASK='qwen_flow_three_routes_20261024'
GRANT='qwen_sourceflow_navtest_resources_20261024'


def authorization_fingerprint(policy,host):
    return fingerprint(dict(standing_grant=policy['task_authorizations'].get(GRANT),
        study_grant=policy['task_authorizations'].get(TASK),
        global_host=policy.get('global_dispatch_prohibitions',{}).get(host),
        permanent_host=policy.get('allowed_hosts',{}).get(host)))


def validate_capability(policy,lease,locked,host,physical_indices,entry_sha256,now):
    body=dict(lease)
    if body.pop('sha256')!=fingerprint(body) or lease['status']!='ACTIVE' or now>=lease['expires_unix']:
        raise PermissionError('No active finite NAVTEST evaluation lease')
    if host not in lease['hosts'] or lease['hosts'][host]['authorization_sha256']!=authorization_fingerprint(policy,host):
        raise PermissionError('NAVTEST host authorization changed or was withdrawn')
    ordinary=(host in policy['allowed_hosts'] and host in policy['task_authorizations'][TASK]['allowed_hosts'] and
              policy.get('global_dispatch_prohibitions',{}).get(host,{}).get('new_task_dispatch_allowed') is not False)
    special=policy['task_authorizations'].get(GRANT,{})
    if not ordinary and host not in special.get('allowed_hosts',[]):
        raise PermissionError('NAVTEST does not authorize another host')
    allowed=set(lease['hosts'][host]['gpu_indices'])
    if host in special.get('allowed_hosts',[]):allowed &= set(special['authorized_gpu_indices'][host])
    if len(physical_indices)!=1 or not set(physical_indices)<=allowed:
        raise PermissionError('GPU is outside the exact NAVTEST grant')
    if entry_sha256!=lease['evaluation_entry_sha256']:
        raise PermissionError('Lease belongs to a different qualified evaluation entry')
    body=dict(locked)
    if body.pop('sha256')!=fingerprint(body) or locked['sha256']!=lease['configuration_lock_sha256']:
        raise PermissionError('Final weights and selection have not been locked')
    return host


def require_navtest_dispatch(physical_indices=None,entry_sha256=None):
    host=socket.gethostname().removesuffix('-worker-0')
    policy=json.loads(Path('/mnt/project/server_dispatch_policy.json').read_text())
    lease=json.loads((ROOT/'NAVTEST_RESOURCE_LEASE.json').read_text())
    locked=json.loads((ROOT/'FINAL_CONFIGURATION_LOCK.json').read_text())
    if physical_indices is None:
        values=os.environ.get('CUDA_VISIBLE_DEVICES','').split(',')
        if not values or any(not value.isdigit() for value in values):
            raise PermissionError('A finite evaluation must declare physical GPU indices')
        physical_indices=list(map(int,values))
    if entry_sha256 is None:entry_sha256=file_sha(sys.modules['__main__'].__file__)
    return validate_capability(policy,lease,locked,host,physical_indices,entry_sha256,time.time())
