"""Current-observation memories; no teacher target is an input to this module."""
from dataclasses import dataclass, replace
import math
from typing import Any, Mapping, Optional

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class StudentMemory:
    spatial: torch.Tensor
    global_memory: torch.Tensor
    predictive: torch.Tensor
    spatial_valid: torch.Tensor
    global_valid: torch.Tensor
    predictive_valid: torch.Tensor
    ego: Optional[torch.Tensor] = None
    metadata: Optional[Mapping[str, Any]] = None

    @property
    def actor_memory(self):
        return torch.cat((self.spatial, self.global_memory, self.predictive), dim=1)

    @property
    def actor_valid(self):
        return torch.cat((self.spatial_valid, self.global_valid, self.predictive_valid), dim=1)

    @property
    def global_tokens(self):
        return self.global_memory

    def index_select(self, indices):
        """Do not detach: repeated auxiliary candidates accumulate memory gradients."""
        return StudentMemory(*(getattr(self, key).index_select(0, indices) for key in
            ('spatial', 'global_memory', 'predictive', 'spatial_valid', 'global_valid', 'predictive_valid')),
            ego=None if self.ego is None else self.ego.index_select(0, indices), metadata=self.metadata)

    def detached(self):
        return StudentMemory(*(getattr(self, key).detach() for key in
            ('spatial', 'global_memory', 'predictive', 'spatial_valid', 'global_valid', 'predictive_valid')),
            ego=None if self.ego is None else self.ego.detach(), metadata=self.metadata)


class Attention(nn.Module):
    """SDPA with separate named projections, also used by the actor LoRA adapter."""
    def __init__(self, width, heads, *, bias=True):
        super().__init__()
        if width % heads:
            raise ValueError('Attention width must divide exactly into heads')
        self.width, self.heads, self.head_dim = width, heads, width // heads
        self.q = nn.Linear(width, width, bias=bias)
        self.k = nn.Linear(width, width, bias=bias)
        self.v = nn.Linear(width, width, bias=bias)
        self.o = nn.Linear(width, width, bias=bias)

    def split(self, value):
        return value.reshape(value.shape[0], value.shape[1], self.heads, self.head_dim).transpose(1, 2)

    def prepare(self, memory):
        return self.split(self.k(memory)), self.split(self.v(memory))

    def attend(self, query, kv, mask=None):
        q = self.split(self.q(query))
        if mask is not None:
            if mask.ndim == 2:
                mask = mask[:, None, None, :]
            elif mask.ndim == 3:
                mask = mask[:, None, :, :]
            else:
                raise ValueError('Attention validity must be [B,K] or [B,Q,K]')
        value = F.scaled_dot_product_attention(q, kv[0], kv[1], attn_mask=mask, dropout_p=0.0)
        value = value.transpose(1, 2).contiguous().reshape(query.shape[0], query.shape[1], self.width)
        return self.o(value)

    def forward(self, query, memory=None, mask=None):
        return self.attend(query, self.prepare(query if memory is None else memory), mask)


class ReadBlock(nn.Module):
    def __init__(self, width, heads, ffn_dim=None):
        super().__init__()
        self.self_norm = nn.LayerNorm(width)
        self.self_attn = Attention(width, heads)
        self.cross_norm = nn.LayerNorm(width)
        self.memory_norm = nn.LayerNorm(width)
        self.cross_attn = Attention(width, heads)
        self.ffn_norm = nn.LayerNorm(width)
        self.ffn = nn.Sequential(nn.Linear(width, ffn_dim or 4 * width), nn.GELU(approximate='tanh'),
                                 nn.Linear(ffn_dim or 4 * width, width))

    def forward(self, query, memory, query_valid, memory_valid):
        query = query + self.self_attn(self.self_norm(query), mask=query_valid)
        query = query + self.cross_attn(self.cross_norm(query), self.memory_norm(memory), memory_valid)
        query = query + self.ffn(self.ffn_norm(query))
        return query.masked_fill(~query_valid[:, :, None], 0)


def _gather_tokens(sequence, indices):
    return sequence.gather(1, indices[:, :, None].expand(-1, -1, sequence.shape[-1]))


class SceneMemoryProvider(nn.Module):
    """Two-layer current-only resampler with camera/grid-preserving queries.

    Selected language outputs are block outputs (1-based), not embedding output0.
    A same-forward merged-visual side channel is accepted in the same packed order.
    No object, DINO, future or metric labels are accepted by the API.
    """
    def __init__(self, input_dim=2048, hidden_dim=1024, num_layers=2, num_heads=16,
                 selected_layers=(8, 16, 28), camera_count=3, spatial_grid=(6, 8),
                 global_tokens=16, ego_dim=8, use_merged_visual=True):
        super().__init__()
        if not selected_layers or len(set(selected_layers)) != len(selected_layers) or min(selected_layers) <= 0:
            raise ValueError('Language layers are unique positive 1-based block indices')
        self.input_dim, self.hidden_dim = input_dim, hidden_dim
        self.selected_layers = tuple(selected_layers)
        self.camera_count, self.spatial_grid = camera_count, tuple(spatial_grid)
        self.global_count, self.ego_dim = global_tokens, ego_dim
        self.use_merged_visual = bool(use_merged_visual)
        per_camera = math.prod(spatial_grid)
        self.spatial_count = camera_count * per_camera
        self.input_norm = nn.LayerNorm(input_dim)
        self.bridge = nn.Linear(input_dim, hidden_dim)
        self.camera_embedding = nn.Embedding(camera_count, hidden_dim)
        self.layer_embedding = nn.Embedding(len(selected_layers) + int(use_merged_visual) + 1, hidden_dim)
        self.position = nn.Sequential(nn.Linear(2, hidden_dim), nn.GELU(approximate='tanh'),
                                      nn.Linear(hidden_dim, hidden_dim))
        self.ego_projection = nn.Sequential(nn.Linear(ego_dim, hidden_dim), nn.GELU(approximate='tanh'),
                                            nn.Linear(hidden_dim, hidden_dim))
        self.spatial_queries = nn.Parameter(torch.empty(camera_count, per_camera, hidden_dim))
        self.global_queries = nn.Parameter(torch.empty(global_tokens, hidden_dim))
        self.blocks = nn.ModuleList(ReadBlock(hidden_dim, num_heads) for _ in range(num_layers))
        self.output_norm = nn.LayerNorm(hidden_dim)
        rows, cols = spatial_grid
        y, x = torch.meshgrid(torch.linspace(0, 1, rows), torch.linspace(0, 1, cols), indexing='ij')
        self.register_buffer('query_xy', torch.stack((x, y), -1).reshape(per_camera, 2), persistent=True)
        nn.init.normal_(self.spatial_queries, std=0.02)
        nn.init.normal_(self.global_queries, std=0.02)

    def _selected(self, features):
        mapping = features.get('selected_hidden_states')
        if mapping is not None:
            if set(int(k) for k in mapping) != set(self.selected_layers):
                raise ValueError('Selected language block identities differ from provider contract')
            return [mapping[key] if key in mapping else mapping[str(key)] for key in self.selected_layers]
        hidden = features.get('hidden_states')
        if hidden is None or len(hidden) <= max(self.selected_layers):
            raise ValueError('Missing selected 1-based Qwen block outputs')
        return [hidden[key] for key in self.selected_layers]

    def forward(self, features, ego, navigation=None):
        if navigation is not None:
            raise ValueError('Navigation comes from current Qwen text and ego command, not a second input path')
        visual_valid = features['visual_mask'].bool()
        indices = features['visual_indices']
        camera_ids = features['camera_ids']
        position = features['visual_positions']
        batch, visual_count = visual_valid.shape
        if ego.shape != (batch, self.ego_dim) or camera_ids.shape != visual_valid.shape:
            raise ValueError('Current ego/camera layout differs from registered memory input')
        # Per-view grid coordinates do not incorporate image padding or another
        # camera's horizontal tile into the local normalized x coordinate.
        xy = position[..., -2:].float().clone()
        for camera in range(self.camera_count):
            present = visual_valid & (camera_ids == camera)
            for axis in range(2):
                values = xy[..., axis]
                lo = values.masked_fill(~present, float('inf')).amin(1, keepdim=True)
                hi = values.masked_fill(~present, -float('inf')).amax(1, keepdim=True)
                lo = torch.where(present.any(1, keepdim=True), lo, torch.zeros_like(lo))
                hi = torch.where(present.any(1, keepdim=True), hi, torch.ones_like(hi))
                normalized = (values - lo) / (hi - lo).clamp_min(1)
                xy[..., axis] = torch.where(present, normalized, values)
        # Qwen stores row,column. Expose standard x,y to position readout.
        xy = xy.flip(-1)
        selected = [_gather_tokens(value, indices) for value in self._selected(features)]
        if self.use_merged_visual:
            merged = features.get('merged_visual_features')
            if merged is None or merged.shape != (batch, visual_count, self.input_dim):
                raise ValueError('Same-forward packed merged_visual_features are required')
            selected.append(merged)
        values = torch.stack(selected, dim=1)
        values = self.bridge(self.input_norm(values))
        dtype = values.dtype
        spatial_position = self.position(xy).to(dtype)
        cameras = self.camera_embedding(camera_ids.clamp(0, self.camera_count - 1)).to(dtype)
        layers = self.layer_embedding.weight[:len(selected)].to(dtype)[None, :, None, :]
        values = values + spatial_position[:, None] + cameras[:, None] + layers
        values = values.masked_fill(~visual_valid[:, None, :, None], 0)
        source = values.reshape(batch, -1, self.hidden_dim)
        source_valid = visual_valid[:, None].expand(-1, len(selected), -1).reshape(batch, -1)
        ego_token = self.ego_projection(ego.float()).to(dtype)
        text = features.get('text_features')
        text_valid = features.get('text_mask')
        if text is None:
            if 'final_hidden' not in features or 'text_indices' not in features or text_valid is None:
                raise ValueError('Current navigation/global text evidence is required')
            text = _gather_tokens(features['final_hidden'], features['text_indices'])
        if text_valid is None or text.shape[:2] != text_valid.shape or text.shape[-1] != self.input_dim:
            raise ValueError('Text evidence and current attention mask are misaligned')
        text = self.bridge(self.input_norm(text))
        text = text + self.layer_embedding.weight[-1].to(text.dtype)
        source = torch.cat((source, text, ego_token[:, None]), dim=1)
        source_valid = torch.cat((source_valid, text_valid.bool(),
                                  torch.ones((batch, 1), device=source.device, dtype=torch.bool)), dim=1)
        source = source.masked_fill(~source_valid[:, :, None], 0)
        per_camera = math.prod(self.spatial_grid)
        query_cameras = torch.arange(self.camera_count, device=source.device).repeat_interleave(per_camera)
        queries = self.spatial_queries.reshape(self.spatial_count, self.hidden_dim).to(dtype)
        queries = queries + self.camera_embedding(query_cameras).to(dtype)
        queries = queries + self.position(self.query_xy).repeat(self.camera_count, 1).to(dtype)
        global_queries = self.global_queries.to(dtype)[None].expand(batch, -1, -1) + ego_token[:, None]
        queries = torch.cat((queries[None].expand(batch, -1, -1), global_queries), dim=1)
        camera_present = torch.stack([(visual_valid & (camera_ids == camera)).any(1)
                                      for camera in range(self.camera_count)], dim=1)
        spatial_valid = camera_present.repeat_interleave(per_camera, dim=1)
        global_valid = torch.ones((batch, self.global_count), device=source.device, dtype=torch.bool)
        query_valid = torch.cat((spatial_valid, global_valid), dim=1)
        # The fixed view/grid queries carry spatial semantics; source positional
        # metadata and camera type are retained. Global tokens can aggregate all
        # current evidence. No calibrated BEV interpretation is asserted.
        for block in self.blocks:
            queries = block(queries, source, query_valid, source_valid)
        queries = self.output_norm(queries).masked_fill(~query_valid[:, :, None], 0)
        empty = queries.new_empty((batch, 0, self.hidden_dim))
        return StudentMemory(queries[:, :self.spatial_count], queries[:, self.spatial_count:], empty,
            spatial_valid, global_valid, global_valid[:, :0], ego=ego,
            metadata=dict(camera_count=self.camera_count, spatial_grid=self.spatial_grid,
                selected_language_layers=self.selected_layers, spatial_camera_ids=query_cameras,
                spatial_xy=self.query_xy.repeat(self.camera_count, 1), current_only=True))


class PredictiveMemory(nn.Module):
    """Four future intervals × eight queries, conditioned only on student memory."""
    def __init__(self, input_dim=1024, hidden_dim=512, num_layers=2, num_heads=8,
                 intervals=4, tokens_per_interval=8):
        super().__init__()
        self.intervals, self.tokens_per_interval = intervals, tokens_per_interval
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.interval_embedding = nn.Embedding(intervals, hidden_dim)
        self.queries = nn.Parameter(torch.empty(intervals, tokens_per_interval, hidden_dim))
        self.blocks = nn.ModuleList(ReadBlock(hidden_dim, num_heads) for _ in range(num_layers))
        self.output_projection = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, input_dim))
        nn.init.normal_(self.queries, std=0.02)

    def forward(self, memory):
        if memory.predictive.shape[1] != 0:
            raise ValueError('Do not recursively predict from an existing predictive-memory output')
        current = torch.cat((memory.spatial, memory.global_memory), dim=1)
        current_valid = torch.cat((memory.spatial_valid, memory.global_valid), dim=1)
        projected = self.input_projection(current)
        interval = self.interval_embedding.weight[:, None].to(projected.dtype)
        query = (self.queries.to(projected.dtype) + interval).flatten(0, 1)
        query = query[None].expand(current.shape[0], -1, -1)
        valid = torch.ones(query.shape[:2], device=query.device, dtype=torch.bool)
        for block in self.blocks:
            query = block(query, projected, valid, current_valid)
        metadata = dict(memory.metadata or {})
        metadata.update(predictive_intervals=self.intervals, predictive_tokens_per_interval=self.tokens_per_interval,
                        predictive_interval_ids=torch.arange(self.intervals, device=query.device).repeat_interleave(self.tokens_per_interval))
        return replace(memory, predictive=self.output_projection(query), predictive_valid=valid, metadata=metadata)
