"""A: train-only world supervision; C: one-way compact world/action streams.

The independent actor preserves the repository CleanXActor calculations from
367bffd27aff5785fce6352481b89e2a433a9ee1.  C's two independently normalized
cross-attention sources and grouped KV bridges are the registered four-way
comparison design, not a claim of exact SimWAM mixed-attention replication.
"""
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from ..qwen_world_postcot.actor import CleanXActor, PreparedContext, timestep_embedding
from .world_stream import WorldPrefix, WorldStream


@dataclass(frozen=True)
class IndependentPrefix:
    architecture_id: str
    actor: PreparedContext
    world: Optional[WorldPrefix]
    world_kv: Tuple[Tuple[torch.Tensor, torch.Tensor], ...] = ()

    def index_select(self, indices):
        return IndependentPrefix(self.architecture_id, self.actor.index_select(indices),
            None if self.world is None else self.world.index_select(indices),
            tuple((k.index_select(0, indices), v.index_select(0, indices))
                  for k, v in self.world_kv))

    def detached(self):
        return IndependentPrefix(self.architecture_id, self.actor.detached(),
            None if self.world is None else self.world.detached(),
            tuple((k.detach(), v.detach()) for k, v in self.world_kv))


class LayerwiseCleanXActor(CleanXActor):
    """Original actor plus an explicit layer-output interface, without hooks."""

    def _start(self, action_state, sigma, context):
        if action_state.ndim != 3 or action_state.shape[1:] != (8, 3):
            raise ValueError('Action states must use the native normalized [B,8,3] codec')
        if sigma.ndim == 0:
            sigma = sigma.expand(len(action_state))
        if sigma.shape != (len(action_state),):
            raise ValueError('One continuous flow sigma is required per candidate')
        if len(context.layer_kv) != self.num_layers or len(context.memory_valid) != len(action_state):
            raise ValueError('Actor prefix belongs to another layer or batch layout')
        hidden = self.action_input(action_state.float())
        hidden = hidden + self.action_position.to(hidden.dtype)[None]
        condition = self.time_embedding(timestep_embedding(sigma, self.hidden_dim)).to(hidden.dtype)
        return hidden, condition + context.ego_condition.to(hidden.dtype)

    def _finish(self, hidden, condition):
        shift, scale = self.output_modulation(condition).chunk(2, dim=-1)
        return self.output_norm(hidden) * (1 + scale[:, None]) + shift[:, None]

    def forward_layer_features(self, action_state, sigma, context, world_kv=()):
        if world_kv:
            raise ValueError('Architecture A must not read a world prefix')
        hidden, condition = self._start(action_state, sigma, context)
        layers = []
        for block, kv in zip(self.blocks, context.layer_kv):
            if self.activation_checkpointing and torch.is_grad_enabled():
                hidden = checkpoint(block, hidden, condition, kv, context.memory_valid,
                                    use_reentrant=False)
            else:
                hidden = block(hidden, condition, kv, context.memory_valid)
            layers.append(hidden)
        return self._finish(hidden, condition), tuple(layers)

    def forward_features(self, action_state, sigma, prepared_context):
        return self.forward_layer_features(action_state, sigma, prepared_context)[0]


class WorldKVBridge(nn.Module):
    """Only K/V are new; actor cross-attention Q/O remain shared."""

    def __init__(self, world_dim, actor_dim, heads):
        super().__init__()
        self.heads, self.head_dim = heads, actor_dim // heads
        self.norm = nn.LayerNorm(world_dim)
        self.k = nn.Linear(world_dim, actor_dim)
        self.v = nn.Linear(world_dim, actor_dim)

    def forward(self, world):
        world = self.norm(world)
        def split(value):
            return value.reshape(len(value), value.shape[1], self.heads, self.head_dim).transpose(1, 2)
        return split(self.k(world)), split(self.v(world))


class WorldConnectedActor(LayerwiseCleanXActor):
    def __init__(self, *, world_dim=384, world_bridge_groups=4,
                 world_gate_init=1e-3, **actor_config):
        super().__init__(**actor_config)
        if self.num_layers % world_bridge_groups:
            raise ValueError('Each world KV group must cover the same number of actor layers')
        self.world_bridge_groups = int(world_bridge_groups)
        heads = self.blocks[0].cross_attn.heads
        self.world_kv_bridges = nn.ModuleList(
            WorldKVBridge(world_dim, self.hidden_dim, heads)
            for _ in range(world_bridge_groups))
        self.world_gate = nn.Parameter(torch.full((self.num_layers,), float(world_gate_init)))

    def prepare_world(self, world_prefix):
        if len(world_prefix.layer_states) != self.num_layers:
            raise ValueError('C uses one-to-one world/action depth mapping')
        group_size = self.num_layers // self.world_bridge_groups
        return tuple(self.world_kv_bridges[layer // group_size](state)
                     for layer, state in enumerate(world_prefix.layer_states))

    @staticmethod
    def _block_with_world(block, hidden, condition, scene_kv, scene_valid,
                          world_kv, gate):
        shift, scale, gate_self, shift_ffn, scale_ffn, gate_ffn = block.modulation(condition).chunk(6, -1)
        value = block.self_norm(hidden) * (1 + scale[:, None]) + shift[:, None]
        hidden = hidden + gate_self[:, None] * block.self_attn(value)
        attention = block.cross_attn
        query = attention.split(attention.q(block.cross_norm(hidden)))
        scene = F.scaled_dot_product_attention(query, scene_kv[0], scene_kv[1],
            attn_mask=scene_valid[:, None, None, :], dropout_p=0.0)
        world = F.scaled_dot_product_attention(query, world_kv[0], world_kv[1],
                                              dropout_p=0.0)
        mixed = scene + gate.tanh() * world
        mixed = mixed.transpose(1, 2).contiguous().reshape(hidden.shape)
        # Apply the shared O exactly once, including its one bias term.
        hidden = hidden + attention.o(mixed)
        value = block.ffn_norm(hidden) * (1 + scale_ffn[:, None]) + shift_ffn[:, None]
        return hidden + gate_ffn[:, None] * block.ffn(value)

    def forward_layer_features(self, action_state, sigma, context, world_kv=()):
        if len(world_kv) != self.num_layers:
            raise ValueError('C requires a current world KV pair for every actor layer')
        hidden, condition = self._start(action_state, sigma, context)
        layers = []
        for layer, (block, scene, world) in enumerate(zip(self.blocks, context.layer_kv, world_kv)):
            arguments = (block, hidden, condition, scene, context.memory_valid,
                         world, self.world_gate[layer])
            if self.activation_checkpointing and torch.is_grad_enabled():
                hidden = checkpoint(self._block_with_world, *arguments, use_reentrant=False)
            else:
                hidden = self._block_with_world(*arguments)
            layers.append(hidden)
        return self._finish(hidden, condition), tuple(layers)


class _IndependentArchitecture(nn.Module):
    def _validate_scene(self, scene, scene_valid, ego):
        if (not isinstance(scene, torch.Tensor) or scene.ndim != 3
                or scene.shape[1:] != (160, self.actor.hidden_dim)):
            raise ValueError('A/C require explicit current C160 tensors; legacy predictive memories are forbidden')
        if (scene_valid.shape != scene.shape[:2] or scene_valid.dtype != torch.bool
                or ego.shape != (len(scene), self.actor.ego_dim)):
            raise ValueError('Current scene validity and legal ego must align')

    def _validate_prefix(self, prefix):
        if prefix.architecture_id != self.architecture_id:
            raise ValueError('Architecture prefix belongs to another model family')

    def denoise_x0(self, action_state_fp32, sigma_fp32, prefix):
        self._validate_prefix(prefix)
        hidden, _ = self.actor.forward_layer_features(action_state_fp32, sigma_fp32,
                                                     prefix.actor, prefix.world_kv)
        return self.actor.predict_x0(hidden).float()

    def pre_reasoning_states(self, prefix):
        self._validate_prefix(prefix)
        if prefix.world is None:
            raise RuntimeError('Training-only world prefix was not requested')
        return prefix.world.final

    def replay_candidates(self, normalized_candidate_trajectory, candidate_to_scene, prefix):
        self._validate_prefix(prefix)
        if self.world_stream is None or prefix.world is None:
            raise RuntimeError('Post-action branch is unavailable in this deployment prefix')
        if candidate_to_scene.dtype != torch.long or candidate_to_scene.shape != (len(normalized_candidate_trajectory),):
            raise ValueError('Explicit int64 candidate-to-scene indices are required')
        selected = prefix.index_select(candidate_to_scene)
        sigma = torch.zeros(len(normalized_candidate_trajectory),
                            device=normalized_candidate_trajectory.device, dtype=torch.float32)
        _, layers = self.actor.forward_layer_features(normalized_candidate_trajectory,
                                                      sigma, selected.actor, selected.world_kv)
        return self.world_stream.replay(selected.world, layers)


class ArchitectureA(_IndependentArchitecture):
    architecture_id = 'A'

    def __init__(self, actor_config=None, world_config=None):
        super().__init__()
        self.actor = LayerwiseCleanXActor(**(actor_config or {}))
        world = dict(world_config or {})
        world.setdefault('scene_dim', self.actor.hidden_dim)
        world.setdefault('actor_dim', self.actor.hidden_dim)
        world.setdefault('num_layers', self.actor.num_layers)
        self.world_stream = WorldStream(**world)

    def prepare_prefix(self, scene, scene_valid, ego, *, need_auxiliary=True):
        # Explicit scene-only tensors; never StudentMemory.actor_memory, which
        # includes the legacy predictive32 condition that A must not read.
        self._validate_scene(scene, scene_valid, ego)
        actor = self.actor.prepare_context(scene, scene_valid, ego)
        world = (self.world_stream.prepare_prefix(scene, scene_valid, ego)
                 if need_auxiliary and self.world_stream is not None else None)
        return IndependentPrefix('A', actor, world)

    def prune_training_components(self):
        self.world_stream = None


class ArchitectureC(_IndependentArchitecture):
    architecture_id = 'C'

    def __init__(self, actor_config=None, world_config=None, world_gate_init=1e-3):
        super().__init__()
        actor = dict(actor_config or {})
        world = dict(world_config or {})
        self.actor = WorldConnectedActor(world_dim=world.get('hidden_dim', 384),
            world_bridge_groups=world.get('bridge_groups', 4), world_gate_init=world_gate_init,
            **actor)
        world.setdefault('scene_dim', self.actor.hidden_dim)
        world.setdefault('actor_dim', self.actor.hidden_dim)
        world.setdefault('num_layers', self.actor.num_layers)
        self.world_stream = WorldStream(**world)

    def prepare_prefix(self, scene, scene_valid, ego, *, need_auxiliary=True):
        self._validate_scene(scene, scene_valid, ego)
        actor = self.actor.prepare_context(scene, scene_valid, ego)
        world = self.world_stream.prepare_prefix(scene, scene_valid, ego)
        return IndependentPrefix('C', actor, world, self.actor.prepare_world(world))

    def prune_training_components(self):
        self.world_stream.prune_post_components()

