"""Strict public-foundation head loading and labels-free cached planning runtime."""
import json
from pathlib import Path
import subprocess
import torch
from omegaconf import OmegaConf
from starVLA.model.modules.action_model.GR00T_ActionHeader import FlowmatchingActionHead
from starVLA.model.modules.joint_world.public_baseline import sha256
from starVLA.model.modules.joint_world.local_cache import load_payload, pack_payload
from starVLA.model.modules.joint_world.local_planner import LocalPlanningBridge, predict_local


def load_public_head(checkpoint, expected_sha):
    if sha256(checkpoint) != expected_sha:
        raise ValueError('Foundation checkpoint differs from current cache')
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
    identity = saved['identity']
    if identity.get('private_driving_weights_loaded', True):
        raise ValueError('Private foundation forbidden')
    cfg = identity.get('model_config')
    if cfg is None:
        cfg = subprocess.check_output(['git', 'show', identity['code_sha'] + ':' + identity['arguments']['config']], text=True)
    cfg = OmegaConf.create(cfg)
    head = FlowmatchingActionHead(cfg)
    head.load_state_dict(saved['modules']['DiT'], strict=True)
    return head.cuda().float().eval().requires_grad_(False), cfg


def load_bridge(checkpoint, current_identity, condition_dim, device='cuda'):
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    identity = saved['identity']
    if identity['current_identity'] != current_identity:
        raise ValueError('Planner foundation/graph/current contract mismatch')
    model = LocalPlanningBridge(condition_dim, identity['graph_config'], identity['arguments']['mode'], identity['arguments']['seed'])
    model.load_state_dict(saved['model'], strict=True)
    return model.to(device).eval().requires_grad_(False), identity


class CurrentOnlyCorpus:
    """Inference never opens WorldTargets, ego future labels, or metric caches."""
    def __init__(self, root):
        self.root = Path(root)
        self.manifest = json.loads((self.root / 'manifest.json').read_text())
        if not self.manifest['complete'] or self.manifest['failed']:
            raise ValueError('Incomplete current observations')
        self.records = self.manifest['records']
        if len({r['token'] for r in self.records}) != len(self.records):
            raise ValueError('Duplicate current scenes')

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        return load_payload(self.root, self.records[i], self.manifest)


@torch.no_grad()
def predict_payload(head, bridge, payload, identity, seed=20260926):
    native = payload['native_actions'].cuda()
    current = pack_payload(payload, identity, 'cuda')
    return predict_local(head, bridge, native, current, [payload['token']], seed)


def decode_actions(actions):
    """Original absolute ego(t0) x/y and sin/cos yaw contract, not velocity."""
    value = actions.detach().float().clone()
    xy = value[..., :2] * value.new_tensor([8.805105, 2.277741]) + value.new_tensor([10.172484, .360762])
    yaw = torch.atan2(value[..., 2], value[..., 3])
    return torch.cat([xy, yaw[..., None]], -1)
