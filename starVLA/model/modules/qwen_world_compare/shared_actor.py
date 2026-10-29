"""B: one shared clean-x Transformer with cacheable directed P/A/R roles.

The local CleanXBlock/SDPA implementation is reused.  UWM's shared token-type
backbone and AdaLN organization informed the organization, not its DDIM loss
or future-noise deployment.  P/A/R directionality and exact layer caches are
the explicit VLA-Drive adaptation (see B_IMPLEMENTATION.md).
"""
from dataclasses import dataclass
from typing import Tuple

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from ..qwen_world_postcot.actor import CleanXBlock, timestep_embedding
from .masks import reasoning_mask


KV = Tuple[torch.Tensor, torch.Tensor]


def _select_kvs(kvs, indices):
    return tuple(tuple(value.index_select(0, indices) for value in pair) for pair in kvs)


@dataclass(frozen=True)
class SharedPrefix:
    """A same-policy, same-forward P prefix; never persistent across RL inner updates."""
    p_kv: Tuple[KV, ...]
    scene_kv: Tuple[KV, ...]
    scene_valid: torch.Tensor
    ego_condition: torch.Tensor
    p_hidden: torch.Tensor

    def index_select(self, indices):
        return SharedPrefix(_select_kvs(self.p_kv, indices), _select_kvs(self.scene_kv, indices),
                            self.scene_valid.index_select(0, indices),
                            self.ego_condition.index_select(0, indices),
                            self.p_hidden.index_select(0, indices))

    def detached(self):
        return SharedPrefix(tuple(tuple(value.detach() for value in pair) for pair in self.p_kv),
                            tuple(tuple(value.detach() for value in pair) for pair in self.scene_kv),
                            self.scene_valid.detach(), self.ego_condition.detach(), self.p_hidden.detach())

    @property
    def logical_nbytes(self):
        return sum(t.numel() * t.element_size() for kvs in (self.p_kv, self.scene_kv)
                   for pair in kvs for t in pair)


class SharedReasoningBlock(CleanXBlock):
    """Exactly one self/cross QKV/O, FFN and modulation for all three roles."""

    def _modulations(self, condition):
        values = self.modulation(condition).chunk(6, dim=-1)
        return tuple(x[:, None] for x in values) if condition.ndim == 2 else values

    def forward_role(self, hidden, condition, prefix_kv, scene_kv, scene_valid):
        shift, scale, gate, fshift, fscale, fgate = self._modulations(condition)
        value = self.self_norm(hidden) * (1 + scale) + shift
        own_kv = self.self_attn.prepare(value)
        all_kv = tuple(torch.cat([*(pair[j] for pair in prefix_kv), own_kv[j]], dim=2)
                       for j in range(2)) if prefix_kv else own_kv
        hidden = hidden + gate * self.self_attn.attend(value, all_kv)
        hidden = hidden + self.cross_attn.attend(self.cross_norm(hidden), scene_kv, scene_valid)
        value = self.ffn_norm(hidden) * (1 + fscale) + fshift
        return hidden + fgate * self.ffn(value), own_kv

    def forward_masked(self, hidden, condition, scene_kv, scene_valid, allowed):
        """Uncached qualification oracle: all roles execute together with a mask."""
        shift, scale, gate, fshift, fscale, fgate = self._modulations(condition)
        value = self.self_norm(hidden) * (1 + scale) + shift
        mask = allowed[None].expand(hidden.shape[0], -1, -1)
        hidden = hidden + gate * self.self_attn(value, mask=mask)
        hidden = hidden + self.cross_attn.attend(self.cross_norm(hidden), scene_kv, scene_valid)
        value = self.ffn_norm(hidden) * (1 + fscale) + fshift
        return hidden + fgate * self.ffn(value)


class ArchitectureB(nn.Module):
    """Shared 20x1024 decision trunk; C is the current-only C160 scene stream."""
    architecture_id = 'B'
    prediction_type = 'x0'

    def __init__(self, hidden_dim=1024, num_layers=20, num_heads=16, ffn_dim=4096,
                 action_tokens=8, action_dim=3, ego_dim=8, p_tokens=32, r_tokens=16,
                 dropout=0.0, activation_checkpointing=False):
        super().__init__()
        if dropout != 0 or action_tokens != 8 or action_dim != 3:
            raise ValueError('B requires dropout0 and the native 8x3 codec')
        if min(num_layers, p_tokens, r_tokens) <= 0:
            raise ValueError('Empty trunk or reasoning slots')
        self.hidden_dim, self.num_layers = hidden_dim, num_layers
        self.action_tokens, self.action_dim, self.ego_dim = action_tokens, action_dim, ego_dim
        self.p_tokens, self.r_tokens = p_tokens, r_tokens
        self.activation_checkpointing = bool(activation_checkpointing)
        self.action_input = nn.Linear(action_dim, hidden_dim)
        self.action_position = nn.Parameter(torch.empty(action_tokens, hidden_dim))
        self.p_queries = nn.Parameter(torch.empty(p_tokens, hidden_dim))
        self.r_queries = nn.Parameter(torch.empty(r_tokens, hidden_dim))
        self.role_embedding = nn.Embedding(3, hidden_dim)
        self.time_embedding = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
                                            nn.Linear(hidden_dim, hidden_dim))
        self.ego_embedding = nn.Sequential(nn.Linear(ego_dim, hidden_dim), nn.SiLU(),
                                           nn.Linear(hidden_dim, hidden_dim))
        self.blocks = nn.ModuleList(SharedReasoningBlock(hidden_dim, num_heads, ffn_dim)
                                    for _ in range(num_layers))
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, 2 * hidden_dim))
        self.action_output = nn.Linear(hidden_dim, action_dim)
        for value in (self.action_position, self.p_queries, self.r_queries, self.role_embedding.weight):
            nn.init.normal_(value, std=0.02)
        nn.init.zeros_(self.output_modulation[-1].weight)
        nn.init.zeros_(self.output_modulation[-1].bias)
        nn.init.xavier_uniform_(self.action_output.weight)
        nn.init.zeros_(self.action_output.bias)

    def _condition(self, sigma, ego_condition, role):
        return (self.time_embedding(timestep_embedding(sigma, self.hidden_dim))
                + ego_condition + self.role_embedding.weight[role])

    def _finish(self, hidden, condition):
        shift, scale = self.output_modulation(condition).chunk(2, dim=-1)
        if condition.ndim == 2:
            shift, scale = shift[:, None], scale[:, None]
        return self.output_norm(hidden) * (1 + scale) + shift

    def _role_block(self, block, hidden, condition, prefix_kv, scene_kv, scene_valid):
        # eval()+grad is the RL mode: activation checkpointing must remain active.
        if self.activation_checkpointing and torch.is_grad_enabled():
            return checkpoint(block.forward_role, hidden, condition, prefix_kv, scene_kv,
                              scene_valid, use_reentrant=False)
        return block.forward_role(hidden, condition, prefix_kv, scene_kv, scene_valid)

    def prepare_prefix(self, scene, valid, ego):
        if scene.ndim != 3 or scene.shape[-1] != self.hidden_dim:
            raise ValueError('Current scene must be [B,C,hidden_dim]')
        if valid.shape != scene.shape[:2] or valid.dtype != torch.bool:
            raise ValueError('Current scene validity must be aligned boolean [B,C]')
        if ego.shape != (len(scene), self.ego_dim):
            raise ValueError('Missing legal current ego state')
        scene = scene.masked_fill(~valid[:, :, None], 0)
        ego_condition = self.ego_embedding(ego.float())
        zero = torch.zeros(len(scene), dtype=torch.float32, device=scene.device)
        condition = self._condition(zero, ego_condition, 0)
        hidden = (self.p_queries + self.role_embedding.weight[0])[None].expand(len(scene), -1, -1)
        pkvs, ckvs = [], []
        for block in self.blocks:
            scene_kv = block.cross_attn.prepare(block.memory_norm(scene))
            hidden, p_kv = self._role_block(block, hidden, condition, (), scene_kv, valid)
            pkvs.append(p_kv)
            ckvs.append(scene_kv)
        return SharedPrefix(tuple(pkvs), tuple(ckvs), valid, ego_condition,
                            self._finish(hidden, condition))

    def pre_reasoning_states(self, prefix):
        return prefix.p_hidden

    def _action_features(self, state, sigma, prefix):
        if state.ndim != 3 or state.shape[1:] != (8, 3):
            raise ValueError('Action state must be normalized [B,8,3]')
        if sigma.ndim == 0:
            sigma = sigma.expand(len(state))
        if sigma.shape != (len(state),) or prefix.ego_condition.shape[0] != len(state):
            raise ValueError('Sigma/prefix candidate layout mismatch')
        condition = self._condition(sigma.float(), prefix.ego_condition, 1)
        hidden = (self.action_input(state.float()) + self.action_position[None]
                  + self.role_embedding.weight[1])
        action_kvs = []
        for block, p_kv, scene_kv in zip(self.blocks, prefix.p_kv, prefix.scene_kv):
            hidden, a_kv = self._role_block(block, hidden, condition, (p_kv,),
                                           scene_kv, prefix.scene_valid)
            action_kvs.append(a_kv)
        return self._finish(hidden, condition), tuple(action_kvs)

    def denoise_x0(self, state, sigma, prefix):
        hidden, _ = self._action_features(state, sigma, prefix)
        return self.action_output(hidden).float()

    def replay_candidates(self, candidate, candidate_to_scene, prefix):
        if self.r_queries is None:
            raise RuntimeError('R was removed from this deployment-only model')
        if candidate_to_scene.shape != (len(candidate),) or candidate_to_scene.dtype != torch.long:
            raise ValueError('Candidate-to-scene must be an aligned int64 index')
        selected = prefix.index_select(candidate_to_scene)
        zero = torch.zeros(len(candidate), dtype=torch.float32, device=candidate.device)
        _, action_kvs = self._action_features(candidate, zero, selected)
        condition = self._condition(zero, selected.ego_condition, 2)
        hidden = (self.r_queries + self.role_embedding.weight[2])[None].expand(len(candidate), -1, -1)
        for block, p_kv, a_kv, scene_kv in zip(self.blocks, selected.p_kv, action_kvs, selected.scene_kv):
            hidden, _ = self._role_block(block, hidden, condition, (p_kv, a_kv),
                                         scene_kv, selected.scene_valid)
        return self._finish(hidden, condition)

    def forward_full(self, state, sigma, scene, valid, ego):
        """Qualification path, not a second training algorithm.

        All P/A/R tokens run through the same trunk together.  The directional
        SDPA mask is the oracle for prefix caching.  R sees this A state; use
        sigma=0 when comparing with exact candidate replay.
        """
        ego_condition = self.ego_embedding(ego.float())
        zero = torch.zeros(len(state), dtype=torch.float32, device=state.device)
        roles = [self._condition(zero, ego_condition, 0),
                 self._condition(sigma.float(), ego_condition, 1),
                 self._condition(zero, ego_condition, 2)]
        p = (self.p_queries + self.role_embedding.weight[0])[None].expand(len(state), -1, -1)
        a = self.action_input(state.float()) + self.action_position[None] + self.role_embedding.weight[1]
        r = (self.r_queries + self.role_embedding.weight[2])[None].expand(len(state), -1, -1)
        hidden = torch.cat((p, a, r), dim=1)
        condition = torch.cat(tuple(x[:, None].expand(-1, n, -1) for x, n in
                                    zip(roles, (self.p_tokens, self.action_tokens, self.r_tokens))), dim=1)
        allowed = reasoning_mask(self.p_tokens, self.action_tokens, self.r_tokens, device=state.device)
        scene = scene.masked_fill(~valid[:, :, None], 0)
        for block in self.blocks:
            scene_kv = block.cross_attn.prepare(block.memory_norm(scene))
            hidden = block.forward_masked(hidden, condition, scene_kv, valid, allowed)
        hidden = self._finish(hidden, condition)
        p, a, r = hidden.split((self.p_tokens, self.action_tokens, self.r_tokens), dim=1)
        return self.action_output(a).float(), p, r

    def strip_training_only(self):
        """Call on an export copy: remove R slots/role, retain shared weights."""
        self.register_parameter('r_queries', None)
        self.role_embedding = nn.Embedding.from_pretrained(
            self.role_embedding.weight[:2].detach().clone(), freeze=True)
        return self

    def forward(self, action_state, sigma, prefix):
        return self.denoise_x0(action_state, sigma, prefix)
