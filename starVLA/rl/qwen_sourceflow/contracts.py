"""Separate scientific identity: native Qwen/three-dimensional source policy."""
import hashlib
import json
from pathlib import Path
import socket

ROOT = Path('/mnt/project/qwen-sourceflow-native-20261018')
POPULATION = Path('/mnt/project/flow-safe-improvement-20261013/population/manifest.json')
UPSTREAM = Path('/mnt/project/simwam-recipe-20261015/upstream/SimWAM')
QWEN_PATH = Path('/mnt/project/DriveDreamer-Policy/models/Qwen3-VL-2B-Instruct')
ACTION_CONFIG = dict(action_dim=3, hidden_dim=1024, ffn_dim=4096, num_heads=24,
    attn_head_dim=128, num_layers=30, text_dim=4096, freq_dim=256, eps=1e-6,
    use_gradient_checkpointing=True)


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(16 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def require_dispatch():
    policy = json.loads(Path('/mnt/project/server_dispatch_policy.json').read_text())
    host = socket.gethostname().removesuffix('-worker-0')
    auth = policy['task_authorizations']['qwen_sourceflow_native_20261018']
    if host not in policy['allowed_hosts'] or host not in auth['allowed_hosts']:
        raise PermissionError('Unauthorized native Qwen-Flow host: ' + host)
    if policy.get('global_dispatch_prohibitions', {}).get(host, {}).get('new_task_dispatch_allowed') is False:
        raise PermissionError('Withdrawn host: ' + host)
    return host


def source_files():
    paths = []
    for root in ['starVLA/model/modules/qwen_sourceflow', 'starVLA/rl/qwen_sourceflow', 'scripts/analysis/qwen_sourceflow']:
        paths.extend(Path(root).rglob('*.py'))
    paths.append(Path('starVLA/model/framework/qwen_sourceflow.py'))
    return {str(p): file_sha(p) for p in sorted(paths)}
