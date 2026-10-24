"""Operator-safe, task-scoped admission; does not import model dependencies."""
import json
from pathlib import Path
import socket

TASK = 'qwen_sourceflow_safe_rl_20261024'


def require_dispatch():
    policy = json.loads(Path('/mnt/project/server_dispatch_policy.json').read_text())
    host = socket.gethostname().removesuffix('-worker-0')
    grant = policy['task_authorizations'][TASK]
    if (host not in policy['allowed_hosts'] or host not in grant['allowed_hosts'] or
            policy.get('global_dispatch_prohibitions', {}).get(host, {}).get('new_task_dispatch_allowed') is False):
        raise PermissionError('Native safety research host is not currently authorized: ' + host)
    return host
