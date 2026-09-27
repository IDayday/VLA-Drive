"""Equal-capacity current/ALL/MASK bridges into the unchanged public-origin DiT."""
import hashlib
import torch
from torch import nn
from .flow import JointTrajectoryFlow
from .local_masks import stable_noise
from .local_cache import build_payload, pack_payload
from starVLA.model.modules.structured_world.action_adapter import WorldToActionAdapter


class LocalPlanningBridge(nn.Module):
    def __init__(self, condition_dim, graph_config, mode, seed=42):
        super().__init__()
        if mode not in ('current', 'all', 'mask'):
            raise ValueError('Unknown planning representation')
        self.mode = mode
        self.graph_config = dict(graph_config)
        dim = graph_config['dim']
        # Exactly the same trainable parameter initialization in all three arms.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed + 5000)
            self.current_projection = nn.Linear(condition_dim, dim)
            self.graph_to_world = nn.Linear(dim, condition_dim)
            self.adapter = WorldToActionAdapter(condition_dim)
        self.graph = None
        if mode != 'current':
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed)
                self.graph = JointTrajectoryFlow(condition_dim, **{k: v for k, v in graph_config.items()
                    if k in ('dim', 'heads', 'layers', 'steps', 'scale_m', 'trajectory_mode', 'agent_scale_m', 'edge_feature_dim')})
            self.graph.requires_grad_(False).eval()

    def train(self, mode=True):
        super().train(mode)
        if self.graph is not None:
            self.graph.eval()
        return self

    def forward(self, native_actions, current, tokens):
        """All inputs are current observations; generated futures never use GT."""
        graph = current['local_graph']
        active = graph.active_actor_mask
        context_mask = current['context_mask']
        # Clear invalid values BEFORE projections and attention.
        actor = torch.where(active[..., None], current['actor_features'].float(), 0.)
        context = torch.where(context_mask[..., None], current['context'].float(), 0.)
        features = self.current_projection(actor)
        joint = None
        if self.graph is not None:
            noise = stable_noise(tokens, graph.source_slot_ids, self.graph.steps,
                self.graph_config.get('sampling_seed', 2037), device=native_actions.device)
            with torch.no_grad():
                joint, generated = self.graph.sample(noise, **current,
                    sampling_steps=self.graph_config.get('sampling_steps', 10))
            features = features + torch.where(active[..., None], generated.float(), 0.)
        # Same current scene/risk memory is available in every arm, including B.
        memory = torch.cat([features, self.current_projection(context)], 1)
        mask = torch.cat([active, context_mask], 1)
        memory = torch.where(mask[..., None], memory, 0.)
        condition = self.adapter(native_actions.float(), self.graph_to_world(memory), mask)
        return condition, joint


def ego_noise(tokens, head, native, seed=20260926):
    values = []
    for token in tokens:
        value = int.from_bytes(hashlib.sha256(f'{seed}:{token}'.encode()).digest()[:4], 'little')
        gen = torch.Generator(device=native.device).manual_seed(value)
        # Preserve native BF16 draws and convert only at the FP32 DiT boundary.
        values.append(torch.randn(1, head.config.action_horizon, head.config.action_dim,
            device=native.device, dtype=native.dtype, generator=gen))
    return torch.cat(values).float()


@torch.no_grad()
def predict_local(head, bridge, native, current, tokens, seed=20260926):
    with torch.autocast(native.device.type, enabled=False):
        condition, joint = (native.float(), None) if bridge is None else bridge(native, current, tokens)
        actions = head.predict_action(condition.float(), initial_noise=ego_noise(tokens, head, native, seed))
    return {'normalized_actions': actions, 'joint_trajectories_xy': joint, 'action_conditions': condition}


class PublicLocalPolicy(nn.Module):
    """Actual online path shares graph packing, sampling and bridge with cached runs."""
    def __init__(self, world, bridge, identity):
        super().__init__()
        self.world, self.bridge, self.identity = world, bridge, identity
        if any(p.requires_grad for p in world.parameters()):
            raise ValueError('Planning transfer requires a frozen trained foundation')
        if getattr(world, 'perception_refinement', None) != identity.get('perception_override'):
            raise ValueError('Online current head differs from the cached planning head')
        if getattr(world, 'foundation_sha256', None) != identity.get('foundation_sha256'):
            raise ValueError('Online foundation differs from the cached planning foundation')
        if getattr(world, 'language_numerics', None) != identity.get('language_numerics'):
            raise ValueError('Online language precision differs from cached conditions')

    @torch.no_grad()
    def predict_action(self, examples, observations, seed=20260926):
        if len(examples) != len(observations):
            raise ValueError('Current observation batch mismatch')
        results = []
        # Singleton Qwen path preserves padding/noise invariance.
        for example, observation in zip(examples, observations):
            allowed = {k: example[k] for k in ('image', 'lang', 'state', 'token') if k in example}
            with torch.autocast('cuda', dtype=torch.bfloat16):
                native, pred = self.world.encode_conditions([allowed])
            payload = build_payload(native, pred, observation, self.identity)
            current = pack_payload(payload, self.identity, native.device)
            results.append(predict_local(self.world.baseline.action_model, self.bridge,
                native, current, [observation.token], seed))
        return results
