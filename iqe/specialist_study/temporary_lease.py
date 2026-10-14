"""Absolute, task-scoped leases for atomic-record work; never optimizer jobs."""
import os
from pathlib import Path
import socket
import time

from iqe.contracts import require
from iqe.io import file_hash, read_json

COMMANDS = {'cooperative-official-scoring', 'precompute-rich-candidates'}


def process_environment(lease, inherited):
    """Set markers before exec so /proc/PID/environ also identifies the guard."""
    env = dict(inherited)
    for key, value in {'IQE_TEMPORARY_LEASE_ID': lease['authorization_id'],
                       'IQE_TEMPORARY_LEASE_EXPIRES_UNIX': str(lease['expires_unix'])}.items():
        require(key not in env or env[key] == value, 'cannot inherit a different temporary lease')
        env[key] = value
    # A new guard must never inherit another process group's guard receipt.
    env.pop('IQE_TEMPORARY_GUARD_RECEIPT', None)
    env.pop('IQE_TEMPORARY_STOP_FILE', None)
    return env


def start_ticks(pid):
    return int(Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19])


def authorization(policy, host, command, env=None, now=None, require_guard=True):
    env = os.environ if env is None else env
    now = time.time() if now is None else now
    require(command in COMMANDS, 'temporary study lease permits atomic records only, no training')
    key = env.get('IQE_TEMPORARY_LEASE_ID')
    entry = policy.get('temporary_task_authorizations', {}).get(key)
    require(entry and entry['new_task_dispatch_allowed'], 'temporary lease not currently admitted')
    path = Path(entry['authorization_path']); lease = read_json(path)
    require(file_hash(path) == entry['authorization_sha256'], 'temporary authorization checksum changed')
    require(key == lease['authorization_id'] and 'existing_iqe_specialist_scorer_study' in lease['scope'],
            'wrong temporary task scope')
    require(host in {h+s for h in lease['hosts'] for s in ('', '-worker-0')}
            and host in {h+s for h in entry['hosts'] for s in ('', '-worker-0')}, 'temporary host outside lease')
    require(lease['starts_unix'] == entry['starts_unix'] and lease['expires_unix'] == entry['expires_unix']
            and lease['starts_unix'] <= now < lease['expires_unix'], 'temporary lease expired/not started')
    require(float(env.get('IQE_TEMPORARY_LEASE_EXPIRES_UNIX', -1)) == lease['expires_unix'],
            'temporary expiry marker mismatch')
    if require_guard:
        guard = read_json(env.get('IQE_TEMPORARY_GUARD_RECEIPT', ''))
        require(guard['lease_id'] == key and guard['lease_hash'] == entry['authorization_sha256']
                and guard['host'] == host and start_ticks(guard['pid']) == guard['start_ticks']
                and os.getpgid(os.getpid()) == guard['pgid'], 'temporary process lacks its live registered guard')
    return lease


def active_lease(command, *, guard=True):
    if not os.environ.get('IQE_TEMPORARY_LEASE_ID'): return None
    return authorization(read_json('/mnt/project/server_dispatch_policy.json'), socket.gethostname(),
                         command, require_guard=guard)


def can_admit(lease, now=None):
    return (time.time() if now is None else now) < lease['expires_unix']-lease['stop_admission_before_expiry_seconds']


def record_boundary(lease, now=None):
    if lease is None:return True
    stop = os.environ.get('IQE_TEMPORARY_STOP_FILE')
    return not (stop and Path(stop).exists()) and (time.time() if now is None else now) < (
        lease['expires_unix']-lease['graceful_stop_before_expiry_seconds'])
