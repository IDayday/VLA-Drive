"""Independent full-framework identities and the explicit finite host lease."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import socket
import time
import os

ROOT = Path(os.environ.get('WORLD_COMPARE_ROOT', '/mnt/project/qwen-world-four-frameworks-20261029'))
TASK = os.environ.get('WORLD_COMPARE_TASK', 'qwen_world_compare_20261029')
POLICY = Path('/mnt/project/server_dispatch_policy.json')


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def file_sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(16*1024**2), b''):
            value.update(block)
    return value.hexdigest()


def authorization():
    policy = json.loads(POLICY.read_text())
    grant = policy['task_authorizations'].get(TASK)
    if not grant:
        raise PermissionError('Full-framework task has no current resource grant')
    original = json.loads(Path(grant['record']).read_text())
    for key in ['permanent_hosts', 'temporary_hosts', 'starts_unix', 'expires_unix']:
        if grant[key] != original[key]:
            raise PermissionError('Task host lease and immutable grant disagree: '+key)
    if grant.get('unbounded_temporary_hosts', []) != original.get('unbounded_temporary_hosts', []):
        raise PermissionError('Unbounded temporary task authorization differs from its immutable record')
    return grant, policy


def allowed_hosts(*, at=None, admit_long_job=False):
    grant, policy = authorization()
    now = time.time() if at is None else at
    # A new exact-host withdrawal after this grant always wins. Existing older
    # prohibitions are narrowly superseded for this new task and time window.
    result = []
    cutoff = grant['expires_unix'] - (grant['checkpoint_guard_seconds'] if admit_long_job else 0)
    candidates = list(grant['permanent_hosts'])
    candidates += grant.get('unbounded_temporary_hosts', [])
    if grant['starts_unix'] <= now < cutoff:
        candidates += grant['temporary_hosts']
    for host in candidates:
        ban = policy.get('global_dispatch_prohibitions', {}).get(host, {})
        recorded = ban.get('utc')
        if ban.get('new_task_dispatch_allowed') is False and recorded:
            try:
                if datetime.fromisoformat(recorded.replace('Z', '+00:00')).timestamp() > grant['starts_unix']:
                    continue
            except ValueError:
                raise PermissionError('Malformed host withdrawal time: '+host)
        result.append(host)
    return result


def require_dispatch(*, host=None, admit_long_job=False):
    host = host or socket.gethostname().removesuffix('-worker-0')
    if host not in allowed_hosts(admit_long_job=admit_long_job):
        raise PermissionError('Host is not currently authorized for this task: '+host)
    return host


def boundary_stop_due(host=None, *, now=None):
    requested=os.environ.get('WORLD_POSTCOT_STOP_FILE')
    if requested and Path(requested).exists():
        return True
    host = host or socket.gethostname().removesuffix('-worker-0')
    grant, _ = authorization()
    if host in grant['permanent_hosts'] or host in grant.get('unbounded_temporary_hosts', []):
        return host not in allowed_hosts(at=now)
    now = time.time() if now is None else now
    return now >= grant['expires_unix'] - grant['checkpoint_guard_seconds'] or host not in allowed_hosts(at=now)
