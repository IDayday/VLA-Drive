"""Action-independent compact world prefix and shared post-action reasoning.

The reading block is reused from this repository's Qwen-World implementation
(367bffd27aff5785fce6352481b89e2a433a9ee1).  The layerwise, one-way cache
organization follows the separated-stream interface in SimWAM MoT, locked at
1433ee7e06a62bd8b89da74455df0d82bda7b726.  This is a new compact module, not a
verbatim copy of the Wan/video expert or its mixed-attention formula.
"""
from dataclasses import dataclass
from typing import Tuple

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from ..qwen_world_postcot.memory import ReadBlock


@dataclass(frozen=True)
class WorldPrefix:
    """All fields depend on current scene/ego only; none depends on flow time."""

    current: torch.Tensor
    current_valid: torch.Tensor
    layer_states: Tuple[torch.Tensor, ...]
    final: torch.Tensor

    def index_select(self, indices):
        return WorldPrefix(self.current.index_select(0, indices),
                           self.current_valid.index_select(0, indices),
                           tuple(value.index_select(0, indices) for value in self.layer_states),
                           self.final.index_select(0, indices))

    def detached(self):
        return WorldPrefix(self.current.detach(), self.current_valid.detach(),
                           tuple(value.detach() for value in self.layer_states), self.final.detach())


class WorldStream(nn.Module):
    """Shared A/C world module; W32 pre-reasoning and R16 post-reasoning.

    W reads only current C.  R reuses the exact W block objects, while its cross
    memory additionally contains W and the matching clean-candidate action
    layer.  Computing R cannot write back into W or the candidate action pass.
    The reverse action bridges serve training only and can be removed at export.
    """

    def __init__(self, scene_dim=1024, hidden_dim=384, num_layers=20,
                 num_heads=6, ffn_dim=1536, prefix_tokens=32, post_tokens=16,
                 actor_dim=1024, ego_dim=8, bridge_groups=4,
                 activation_checkpointing=False):
        super().__init__()
        if num_layers < 1 or bridge_groups < 1 or num_layers % bridge_groups:
            raise ValueError('World layers must divide exactly into bridge groups')
        if prefix_tokens != 32 or post_tokens != 16:
            raise ValueError('The four-architecture contract fixes W32 and R16')
        self.scene_dim, self.hidden_dim = int(scene_dim), int(hidden_dim)
        self.num_layers, self.actor_dim = int(num_layers), int(actor_dim)
        self.ego_dim, self.bridge_groups = int(ego_dim), int(bridge_groups)
        self.activation_checkpointing = bool(activation_checkpointing)
        self.current_norm = nn.LayerNorm(scene_dim)
        self.current_bridge = nn.Linear(scene_dim, hidden_dim)
        self.ego_bridge = nn.Sequential(nn.Linear(ego_dim, hidden_dim), nn.SiLU(),
                                        nn.Linear(hidden_dim, hidden_dim))
        self.prefix_queries = nn.Parameter(torch.empty(prefix_tokens, hidden_dim))
        self.post_queries = nn.Parameter(torch.empty(post_tokens, hidden_dim))
        self.role_embedding = nn.Parameter(torch.empty(2, hidden_dim))
        self.blocks = nn.ModuleList(ReadBlock(hidden_dim, num_heads, ffn_dim)
                                    for _ in range(num_layers))
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.post_action_bridges = nn.ModuleList(
            nn.Sequential(nn.LayerNorm(actor_dim), nn.Linear(actor_dim, hidden_dim))
            for _ in range(bridge_groups))
        for value in (self.prefix_queries, self.post_queries, self.role_embedding):
            nn.init.normal_(value, std=0.02)

    def _block(self, block, hidden, memory, query_valid, memory_valid):
        # Explicit non-reentrant checkpoint also works for eval()+autograd RL.
        if self.activation_checkpointing and torch.is_grad_enabled():
            return checkpoint(block, hidden, memory, query_valid, memory_valid,
                              use_reentrant=False)
        return block(hidden, memory, query_valid, memory_valid)

    def prepare_prefix(self, scene, scene_valid, ego):
        if scene.ndim != 3 or scene.shape[-1] != self.scene_dim:
            raise ValueError('World current condition must be [B,C,scene_dim]')
        if scene_valid.shape != scene.shape[:2] or scene_valid.dtype != torch.bool:
            raise ValueError('World current validity must be an aligned bool mask')
        if ego.shape != (len(scene), self.ego_dim):
            raise ValueError('World prefix requires legal current ego only')
        if not bool(scene_valid.any(dim=1).all()):
            raise ValueError('Every world prefix needs at least one current token')
        current = self.current_bridge(self.current_norm(scene))
        current = current.masked_fill(~scene_valid[:, :, None], 0)
        hidden = self.prefix_queries[None].expand(len(scene), -1, -1)
        hidden = hidden + self.role_embedding[0][None, None]
        hidden = hidden + self.ego_bridge(ego.float())[:, None]
        query_valid = torch.ones(hidden.shape[:2], device=hidden.device, dtype=torch.bool)
        layers = []
        for block in self.blocks:
            hidden = self._block(block, hidden, current, query_valid, scene_valid)
            layers.append(hidden)
        return WorldPrefix(current, scene_valid, tuple(layers), self.output_norm(hidden))

    def replay(self, prefix, candidate_layer_states):
        """R-only read; candidate hidden retains its actor/current-scene graph."""
        if self.post_queries is None or self.post_action_bridges is None:
            raise RuntimeError('Post-action reasoning was removed for deployment')
        if len(candidate_layer_states) != self.num_layers:
            raise ValueError('One candidate hidden is required per world layer')
        batch = len(prefix.current)
        hidden = self.post_queries[None].expand(batch, -1, -1)
        hidden = hidden + self.role_embedding[1][None, None]
        query_valid = torch.ones(hidden.shape[:2], device=hidden.device, dtype=torch.bool)
        group_size = self.num_layers // self.bridge_groups
        for layer, (block, world, action) in enumerate(
                zip(self.blocks, prefix.layer_states, candidate_layer_states)):
            if action.ndim != 3 or action.shape[0] != batch or action.shape[-1] != self.actor_dim:
                raise ValueError('Candidate replay layers do not match the world prefix')
            bridge = self.post_action_bridges[layer // group_size]
            candidate = bridge(action)
            memory = torch.cat((prefix.current, world, candidate), dim=1)
            extra_valid = torch.ones((batch, world.shape[1] + action.shape[1]),
                                     device=hidden.device, dtype=torch.bool)
            memory_valid = torch.cat((prefix.current_valid, extra_valid), dim=1)
            hidden = self._block(block, hidden, memory, query_valid, memory_valid)
        return self.output_norm(hidden)

    def prune_post_components(self):
        """C keeps shared world blocks/W while deleting R-only deployment state."""
        self.post_queries = None
        self.post_action_bridges = None

