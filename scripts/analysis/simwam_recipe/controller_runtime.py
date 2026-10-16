"""Administrative queue admission, separate from experiment-worker admission.

SSH aliases and credentials belong to the local control host. No credentials
are copied to GPU workers. This module does not authorize local GPU work.
"""
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess

from starVLA.rl.simwam_recipe.contracts import require_dispatch

POLICY = Path('/mnt/project/server_dispatch_policy.json')


def preflight_analysis_compatibility(snapshot, receipt=None):
    """Check the evaluator/reference source pair before launching any workers."""
    from starVLA.rl.simwam_recipe.contracts import ROOT, file_sha
    from scripts.analysis.simwam_recipe.evaluation_compatibility import match_protocol
    reference = json.loads((Path(ROOT)/'evaluations/SFT_update000000_DEV/manifest.json').read_text())
    candidate = dict(reference)
    candidate['source_sha256'] = file_sha(Path(snapshot)/'scripts/analysis/simwam_recipe/evaluate_policy.py')
    return match_protocol(candidate, reference, receipt)


def require_worker_host(host):
    policy = json.loads(POLICY.read_text())
    if (host not in policy['allowed_hosts']
        or host not in policy['task_authorizations']['flow_safe_improvement_20261013']['allowed_hosts']
        or policy.get('global_dispatch_prohibitions', {}).get(host, {}).get('new_task_dispatch_allowed') is False):
        raise PermissionError('Experiment worker host not authorized: ' + host)
    return host


def require_controller_host(administrative_local=False):
    if not administrative_local:
        return require_dispatch()
    host = socket.gethostname().removesuffix('-worker-0')
    policy = json.loads(POLICY.read_text())
    if (host != 'training-vla-zt' or host not in policy['allowed_hosts']
        or policy.get('global_dispatch_prohibitions', {}).get(host, {}).get('new_task_dispatch_allowed') is False
        or not policy.get('administrative_control')):
        raise PermissionError('Local administrative control is not authorized: ' + host)
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('Administrative local controller requires CUDA_VISIBLE_DEVICES empty')
    return host


def probe_routes(hosts, python):
    """Fail before waiting for a milestone if a control-to-worker route is broken."""
    result = {}
    for host in sorted(set(hosts)):
        require_worker_host(host)
        command = [python, '-c', 'import socket; print(socket.gethostname())']
        actual = subprocess.check_output(
            ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', host, shlex.join(command)],
            text=True, timeout=20).strip()
        if actual.removesuffix('-worker-0') != host:
            raise RuntimeError('SSH destination identity mismatch: ' + host + ' -> ' + actual)
        result[host] = actual
    return result
