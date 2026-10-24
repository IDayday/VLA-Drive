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
    from .resource_authorization import authorized_hosts
    policy = json.loads(Path('/mnt/project/server_dispatch_policy.json').read_text())
    host = socket.gethostname().removesuffix('-worker-0')
    if host not in authorized_hosts(policy,ROOT):
        raise PermissionError('Unauthorized native Qwen-Flow host: ' + host)
    return host


def source_files():
    paths = []
    for root in ['starVLA/model/modules/qwen_sourceflow', 'starVLA/rl/qwen_sourceflow', 'scripts/analysis/qwen_sourceflow',
                 'starVLA/rl/simwam_recipe', 'starVLA/rl/fm_transfer']:
        paths.extend(Path(root).rglob('*.py'))
    paths += [Path(p) for p in ['starVLA/model/framework/qwen_sourceflow.py',
        'scripts/analysis/simwam_recipe/train.py', 'starVLA/rl/flow_grpo/transactions.py',
        'starVLA/rl/flow_grpo/reproducibility.py','starVLA/rl/safe_flow_improvement/v1_reward.py',
        'scripts/analysis/constrained_flow_noise/batched_canonical_score.py']]
    return {str(p): file_sha(p) for p in sorted(set(paths))}


def runtime_identity():
    import importlib
    import torch
    import transformers
    record=json.loads(Path('configs/qwen_sourceflow/runtime_lock.json').read_text())
    if record['torch']!=torch.__version__ or record['transformers']!=transformers.__version__:
        raise ValueError('Unqualified native numerical/runtime package version')
    for name, expected in record['native_Qwen_runtime'].items():
        if file_sha(importlib.import_module(name).__file__)!=expected['sha256']:
            raise ValueError('Native Qwen/processor implementation changed: '+name)
    return record


def configure_native_numerics():
    """Same qualified FP32 accumulation/reproducibility switches as source RL."""
    import os
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    if os.environ['CUBLAS_WORKSPACE_CONFIG'] != ':4096:8':
        raise ValueError('Unregistered cuBLAS reproducibility workspace')
    from starVLA.rl.flow_grpo.reproducibility import configure_numerics
    configure_numerics()
