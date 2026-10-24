"""Shared scientific identity, split gates, and finite-budget bookkeeping."""
import json
from pathlib import Path
import socket
import time
import torch
from starVLA.rl.qwen_sourceflow.contracts import fingerprint, file_sha
from starVLA.rl.flow_grpo.transactions import atomic_json
from starVLA.rl.flow_grpo.math import centered_group_rewards

ROOT = Path('/mnt/project/qwen-flow-three-routes-20261024')
TASK = 'qwen_flow_three_routes_20261024'


def read_protocol(path=None):
    path = Path(path) if path else ROOT/'common_protocol.json'
    value = json.loads(path.read_text()); body = dict(value)
    if body.pop('sha256') != fingerprint(body):
        raise ValueError('Changed common scientific protocol')
    return value


def require_budget_lock(protocol,phase='SCREEN_LOCKED'):
    name='DIAGNOSTIC_BUDGET_LOCK.json' if phase=='DIAGNOSTIC_LOCKED' else 'BUDGET_LOCK.json'
    value=json.loads((ROOT/name).read_text());body=dict(value)
    if body.pop('sha256')!=fingerprint(body) or value['protocol_sha256']!=protocol['sha256']:
        raise ValueError('Missing/changed common finite experiment budget')
    if value['phase']!=phase:raise ValueError('Unregistered study stage')
    return value


def require_dispatch():
    policy = json.loads(Path('/mnt/project/server_dispatch_policy.json').read_text())
    host = socket.gethostname().removesuffix('-worker-0')
    if (host not in policy['allowed_hosts'] or host not in policy['task_authorizations'][TASK]['allowed_hosts'] or
        policy.get('global_dispatch_prohibitions', {}).get(host, {}).get('new_task_dispatch_allowed') is False):
        raise PermissionError('Unauthorized three-route worker: '+host)
    return host


def require_scene(protocol, token, split, *, update=False):
    if token not in protocol['splits'].get(split, []):
        raise ValueError('Scene outside fixed split: '+split)
    if update and split != 'TRAIN':
        raise ValueError('DEV/test feedback cannot enter updates or fitted statistics')
    if split == 'DEV_AUDIT' and not (ROOT/'FINAL_CONFIGURATION_LOCK.json').exists():
        raise PermissionError('This round audit logs remain sealed until configuration lock')


@torch.no_grad()
def group_advantages(rewards, floor=.01):
    if rewards.ndim != 2 or not torch.isfinite(rewards).all():
        raise ValueError('Finite [scene,candidate] original-scale rewards required')
    delta, std, _ = centered_group_rewards(rewards.double())
    scale = rewards.double().std(unbiased=False).clamp_min(floor)
    advantage = delta/scale
    mask = (std > 1e-8).expand_as(advantage).float()
    return dict(advantages=advantage.float(), mask=mask, global_std=float(rewards.double().std(unbiased=False)),
                scale=float(scale), raw_group_std=std[:, 0], floor=floor)


class Ledger:
    """Logical work, actual execution, and cache reuse remain separate."""
    def __init__(self, path=None):
        self.path = Path(path) if path else None
        self.start = time.monotonic()
        self.values = dict(candidate_attempts=0, scored_candidates=0, scorer_batch_calls=0,
                           logical_scoring_cost=0, cache_reused_candidates=0, model_forward_calls=0,
                           qwen_prefills=0, optimizer_updates=0, scoring_wait_seconds=0.,
                           condition_seconds=0., generation_seconds=0., update_seconds=0.)

    def add(self, **values):
        for key, value in values.items():
            self.values[key] = self.values.get(key, 0)+value

    def snapshot(self):
        return dict(self.values, invocation_wall_seconds=time.monotonic()-self.start,
                    GPU_semantics='Shared GPU worker wall occupancy, not exclusive GPU compute time')

    def save(self):
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            atomic_json(self.path, self.snapshot())
