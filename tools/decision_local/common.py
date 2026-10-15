"""Small artifact helpers, with no model or distributed initialization."""
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+f'.{os.getpid()}.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def source_sha(root):
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()


def check_host():
    """Recheck the current allowed hosts and withdrawals at every dispatch.

    The 2026-10-15 instruction permits shared GPUs on the allowed hosts. It
    does not revive temporary leases, exclusive allocations or paused jobs.
    Actual free memory/load admission remains the dispatcher's responsibility.
    """
    path = Path('/mnt/project/server_dispatch_policy.json')
    policy = read(path)
    host = socket.gethostname().removesuffix('-worker-0')
    if host not in policy['allowed_hosts'] or host in policy['global_dispatch_prohibitions']:
        raise RuntimeError('Host is not currently authorized, or has been withdrawn')
    if policy.get('host_reservations', {}).get(host, {}).get('exclude_new_other_workloads', False):
        raise RuntimeError('A current exclusive host reservation prevents new work')
    return {'host': host, 'policy_sha256': digest(path), 'checked_unix': time.time()}
