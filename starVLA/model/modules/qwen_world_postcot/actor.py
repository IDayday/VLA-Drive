"""Independent clean-trajectory Transformer decoder; FP32 state, AMP compute."""
from dataclasses import dataclass
import math
from typing import Tuple

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .memory import Attention, StudentMemory


@dataclass(frozen=True)
class PreparedContext:
    layer_kv: Tuple[Tuple[torch.Tensor, torch.Tensor], ...]
    memory_valid: torch.Tensor
    ego_condition: torch.Tensor

    def index_select(self, indices):
        return PreparedContext(tuple((key.index_select(0, indices), value.index_select(0, indices))
                                     for key, value in self.layer_kv),
                               self.memory_valid.index_select(0, indices),
                               self.ego_condition.index_select(0, indices))

    def detached(self):
        return PreparedContext(tuple((key.detach(), value.detach()) for key, value in self.layer_kv),
                               self.memory_valid.detach(), self.ego_condition.detach())

    @property
    def logical_nbytes(self):
        return sum(value.numel() * value.element_size() for pair in self.layer_kv for value in pair)

    @property
    def kv(self):
        return self.layer_kv


def timestep_embedding(sigma, width):
    """Real FP32 sin/cos; no float64 complex RoPE or integer timestep buckets."""
    half = width // 2
    frequency = torch.exp(-math.log(10000) * torch.arange(half, device=sigma.device, dtype=torch.float32) / max(half, 1))
    angle = sigma.float()[:, None] * 1000.0 * frequency[None]
    value = torch.cat((angle.cos(), angle.sin()), dim=-1)
    if width % 2:
        value = torch.cat((value, torch.zeros_like(value[:, :1])), dim=-1)
    return value


class CleanXBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, ffn_dim):
        super().__init__()
        self.self_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.self_attn = Attention(hidden_dim, num_heads)
        self.cross_norm = nn.LayerNorm(hidden_dim)
        self.memory_norm = nn.LayerNorm(hidden_dim)
        self.cross_attn = Attention(hidden_dim, num_heads)
        self.ffn_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.ffn = nn.Sequential(nn.Linear(hidden_dim, ffn_dim), nn.GELU(approximate='tanh'),
                                 nn.Linear(ffn_dim, hidden_dim))
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, 6 * hidden_dim))
        # AdaLN-Zero for self-attention/FFN only; ordinary cross residuals
        # remain active and carry post-loss gradients from the first update.
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    def forward(self, hidden, condition, kv, memory_valid):
        shift_self, scale_self, gate_self, shift_ffn, scale_ffn, gate_ffn = self.modulation(condition).chunk(6, dim=-1)
        value = self.self_norm(hidden) * (1 + scale_self[:, None]) + shift_self[:, None]
        hidden = hidden + gate_self[:, None] * self.self_attn(value)
        hidden = hidden + self.cross_attn.attend(self.cross_norm(hidden), kv, memory_valid)
        value = self.ffn_norm(hidden) * (1 + scale_ffn[:, None]) + shift_ffn[:, None]
        return hidden + gate_ffn[:, None] * self.ffn(value)


class CleanXActor(nn.Module):
    prediction_type = 'x0'

    def __init__(self, hidden_dim=1024, num_layers=20, num_heads=16, ffn_dim=4096,
                 action_tokens=8, action_dim=3, ego_dim=8, dropout=0.0,
                 activation_checkpointing=False):
        super().__init__()
        if dropout != 0:
            raise ValueError('The fixed clean-x actor contract has dropout=0')
        if action_tokens != 8 or action_dim != 3:
            raise ValueError('Keep the native 8x3 action codec')
        self.hidden_dim, self.action_tokens, self.action_dim, self.ego_dim = hidden_dim, action_tokens, action_dim, ego_dim
        self.num_layers = num_layers
        self.activation_checkpointing = bool(activation_checkpointing)
        self.action_input = nn.Linear(action_dim, hidden_dim)
        self.action_position = nn.Parameter(torch.empty(action_tokens, hidden_dim))
        self.time_embedding = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
                                            nn.Linear(hidden_dim, hidden_dim))
        self.ego_embedding = nn.Sequential(nn.Linear(ego_dim, hidden_dim), nn.SiLU(),
                                           nn.Linear(hidden_dim, hidden_dim))
        self.blocks = nn.ModuleList(CleanXBlock(hidden_dim, num_heads, ffn_dim) for _ in range(num_layers))
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, 2 * hidden_dim))
        self.action_output = nn.Linear(hidden_dim, action_dim)
        nn.init.normal_(self.action_position, std=0.02)
        nn.init.zeros_(self.output_modulation[-1].weight)
        nn.init.zeros_(self.output_modulation[-1].bias)
        # A finite nonzero output initialization permits the main trajectory
        # loss to train cross-attention and memory immediately. Self/FFN AdaLN
        # gates unlock during the first updates, as audited in the tests.
        nn.init.xavier_uniform_(self.action_output.weight)
        nn.init.zeros_(self.action_output.bias)

    @property
    def input_embedding(self):
        return self.action_input

    @property
    def output_projection(self):
        return self.action_output

    def prepare_context(self, memory, memory_valid=None, ego=None):
        if isinstance(memory, StudentMemory):
            memory_valid = memory.actor_valid if memory_valid is None else memory_valid
            ego = memory.ego if ego is None else ego
            memory = memory.actor_memory
        if memory.ndim != 3 or memory.shape[-1] != self.hidden_dim:
            raise ValueError('Actor memory must be [B,N,hidden_dim]')
        if memory_valid is None:
            memory_valid = torch.ones(memory.shape[:2], device=memory.device, dtype=torch.bool)
        if memory_valid.shape != memory.shape[:2] or memory_valid.dtype != torch.bool:
            raise ValueError('Actor memory validity must be an aligned boolean mask')
        if ego is None or ego.shape != (memory.shape[0], self.ego_dim):
            raise ValueError('An explicit legal current ego vector is required')
        memory = memory.masked_fill(~memory_valid[:, :, None], 0)
        kv = tuple(block.cross_attn.prepare(block.memory_norm(memory)) for block in self.blocks)
        return PreparedContext(kv, memory_valid, self.ego_embedding(ego.float()))

    def forward_features(self, action_state, sigma, prepared_context):
        if action_state.ndim != 3 or action_state.shape[1:] != (self.action_tokens, self.action_dim):
            raise ValueError('Action state must be normalized native [B,8,3]')
        if sigma.ndim == 0:
            sigma = sigma.expand(action_state.shape[0])
        if sigma.shape != (action_state.shape[0],):
            raise ValueError('Sigma must have one continuous value per candidate')
        if len(prepared_context.layer_kv) != self.num_layers or prepared_context.memory_valid.shape[0] != len(action_state):
            raise ValueError('Prepared context belongs to another layer/batch layout')
        hidden = self.action_input(action_state.float())
        hidden = hidden + self.action_position.to(hidden.dtype)[None]
        time = self.time_embedding(timestep_embedding(sigma, self.hidden_dim)).to(hidden.dtype)
        condition = time + prepared_context.ego_condition.to(hidden.dtype)
        for block, kv in zip(self.blocks, prepared_context.layer_kv):
            if self.activation_checkpointing and self.training and torch.is_grad_enabled():
                hidden = checkpoint(block, hidden, condition, kv, prepared_context.memory_valid, use_reentrant=False)
            else:
                hidden = block(hidden, condition, kv, prepared_context.memory_valid)
        # The same clean-candidate features feed the post head. The final
        # clean-action projection is deliberately not run by this method.
        shift, scale = self.output_modulation(condition).chunk(2, dim=-1)
        return self.output_norm(hidden) * (1 + scale[:, None]) + shift[:, None]

    def predict_x0(self, hidden):
        return self.action_output(hidden).float()

    def forward(self, action_state, sigma, prepared_context):
        return self.predict_x0(self.forward_features(action_state, sigma, prepared_context))
