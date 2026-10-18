"""Explicit Qwen -> source attention-space adaptation; no query compression."""
import copy
import torch
from torch import nn
from .vendor.wan_video_dit import precompute_freqs_cis_3d, rope_apply


def depth_mapping(qwen_layers, action_layers):
    if min(qwen_layers, action_layers) < 2:
        raise ValueError('Depth mapping requires at least two layers')
    return [1 + l * (qwen_layers - 1) // (action_layers - 1) for l in range(action_layers)]


class VisualProjection(nn.Module):
    def __init__(self, block):
        super().__init__()
        # Deep copies are necessary: RL action LoRA must not invalidate memory.
        self.norm = copy.deepcopy(block.norm1)
        self.k = copy.deepcopy(block.self_attn.k)
        self.v = copy.deepcopy(block.self_attn.v)
        self.norm_k = copy.deepcopy(block.self_attn.norm_k)

    def forward(self, hidden):
        z = self.norm(hidden)
        return self.norm_k(self.k(z)), self.v(z)


class QwenConditionProvider(nn.Module):
    def __init__(self, action_expert, qwen_hidden=2048, qwen_layers=28, proprio_dim=8):
        super().__init__()
        self.layer_mapping = depth_mapping(qwen_layers, len(action_expert.blocks))
        self.visual_bridge = nn.Linear(qwen_hidden, action_expert.hidden_dim)
        self.text_bridge = nn.Linear(qwen_hidden, action_expert.text_dim)
        # Source current-state linear encoder appended to text, with raw SI units.
        self.proprio_encoder = nn.Linear(proprio_dim, action_expert.text_dim)
        self.proprio_dim = proprio_dim
        self.visual_projections = nn.ModuleList([VisualProjection(b) for b in action_expert.blocks])
        self.num_heads = action_expert.num_heads
        self.head_dim = action_expert.attn_head_dim
        self._freqs = precompute_freqs_cis_3d(self.head_dim, end=1024)
        actor_ids = {p.data_ptr() for p in action_expert.parameters()}
        if any(p.data_ptr() in actor_ids for p in self.parameters()):
            raise ValueError('Observation projection aliases action parameters')

    def forward(self, encoded, proprio):
        hiddens, visual_indices = encoded['hidden_states'], encoded['visual_indices']
        vm = encoded['visual_mask']
        if proprio.shape != (vm.shape[0], self.proprio_dim):
            raise ValueError('Current proprio shape changed')
        spatial = encoded['visual_positions']  # [B,S,3] time/row/tiled-column
        if (spatial[..., 0] != 0).any() or spatial.max() >= 1024 or spatial.min() < 0:
            raise ValueError('Invalid current-image memory coordinates')
        freqs = [f.to(vm.device) for f in self._freqs]
        # Source rope_apply accepts a broadcast batch dimension for positions.
        freq = torch.cat([f[spatial[..., axis]] for axis, f in enumerate(freqs)], -1).unsqueeze(2)
        kv, bridged = [], {}
        for j, projection in zip(self.layer_mapping, self.visual_projections):
            if j not in bridged:
                h = hiddens[j]
                selected = h.gather(1, visual_indices[..., None].expand(-1, -1, h.shape[-1]))
                bridged[j] = self.visual_bridge(selected)
            k, v = projection(bridged[j])
            kv.append(dict(k=rope_apply(k, freq, self.num_heads), v=v))
        text = encoded['final_hidden'].gather(1,
            encoded['text_indices'][..., None].expand(-1, -1, encoded['final_hidden'].shape[-1]))
        context = self.text_bridge(text)
        # Exact source append semantics; current numeric state only.
        state = self.proprio_encoder(proprio.to(context.dtype).unsqueeze(1)).to(context.dtype)
        context = torch.cat([context, state], 1)
        mask = torch.cat([encoded['text_mask'], torch.ones((vm.shape[0], 1),
            dtype=torch.bool, device=vm.device)], 1)
        return dict(context=context, context_mask=mask, visual_kv=kv, visual_mask=vm)
