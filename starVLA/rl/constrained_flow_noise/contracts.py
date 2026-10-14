"""Experiment boundaries, execution identities, and exact trainable closure."""
import json
from pathlib import Path
import socket

from starVLA.rl.safe_flow_improvement.contracts import SFT_SHA256, file_sha, fingerprint, seed_for

VARIANTS = ("C_SDE", "N_FLOW_NOISE")
COSTS = ("NC", "DAC", "TTC", "road_margin", "object_margin")
CONDITION_BOUNDARY = "normalized_action_tokens_before_action_model.qwen_proj"
TSINGHUA_HOSTS = tuple("training-vlawm-tsinghua-zt" + suffix for suffix in ("", "2", "3", "4"))


def require_dispatch():
    policy = json.loads(Path('/mnt/project/server_dispatch_policy.json').read_text())
    host = socket.gethostname().removesuffix('-worker-0')
    if host not in TSINGHUA_HOSTS or host not in policy['allowed_hosts']:
        raise PermissionError(f'Current Flow host is not authorized: {host}')
    return host


def validate_parameters(policy, noise_head=None):
    original = dict(policy.action_model.named_parameters())
    if len(original) != 359 or sum(p.numel() for p in original.values()) != 819503620:
        raise ValueError('Original action-model parameter closure changed')
    ids = {id(p) for p in original.values()}
    if not all(p.requires_grad for p in original.values()):
        raise ValueError('Full action model must remain trainable')
    if any(p.requires_grad for p in policy.parameters() if id(p) not in ids):
        raise ValueError('Upstream parameter entered actor optimizer')
    noise = {} if noise_head is None else dict(noise_head.named_parameters())
    if any(not p.requires_grad for p in noise.values()) or ids & {id(p) for p in noise.values()}:
        raise ValueError('Noise parameter closure/freeze mismatch')
    return {'action_names': sorted(original), 'action_numel': 819503620,
            'noise_names': sorted(noise), 'noise_numel': sum(p.numel() for p in noise.values())}
