"""Sixteen planning slots: eight thumbnail reads plus eight crop-content reads."""
import torch
from torch import nn


class GlobalLocalRegisterReadout(nn.Module):
    def __init__(self, dim=256, num_heads=8, device=None, dtype=torch.float32):
        super().__init__()
        factory = dict(device=device, dtype=dtype)
        self.global_queries = nn.Parameter(torch.empty(1, 8, dim, **factory))
        self.local_queries = nn.Parameter(torch.empty(1, 8, dim, **factory))
        nn.init.normal_(self.global_queries, std=.02)
        nn.init.normal_(self.local_queries, std=.02)
        self.global_attention = nn.MultiheadAttention(dim, num_heads, batch_first=True, dropout=0., **factory)
        self.local_attention = nn.MultiheadAttention(dim, num_heads, batch_first=True, dropout=0., **factory)
        self.tile_position_mlp = nn.Sequential(nn.Linear(5, dim, **factory), nn.GELU(), nn.Linear(dim, dim, **factory))
        self.local_condition = nn.Linear(dim, dim, **factory)
        self.global_norm = nn.LayerNorm(dim, **factory)
        self.local_norm = nn.LayerNorm(dim, **factory)

    def forward(self, thumbnail, crops, crop_metadata):
        if thumbnail.ndim != 2 or crops.ndim != 3 or not crops.shape[0]:
            raise ValueError('global_local_8_8 needs thumbnail [R,D] and nonempty crops [T,R,D]')
        thumbnail = thumbnail.to(self.global_queries.dtype).unsqueeze(0)
        crops = crops.to(self.local_queries.dtype)
        metadata = crop_metadata.to(device=crops.device, dtype=self.local_queries.dtype)
        global_tokens = self.global_attention(self.global_queries, thumbnail, thumbnail, need_weights=False)[0]
        visual_values = crops.flatten(0, 1).unsqueeze(0)
        position = self.tile_position_mlp(metadata)[:, None, :].expand_as(crops)
        keys = (crops + position).flatten(0, 1).unsqueeze(0)
        queries = self.local_queries + self.local_condition(global_tokens.mean(1, keepdim=True))
        # Position enters K only; V is pure crop visual content. Learned query
        # identity is not added to the output and cannot manufacture its rank.
        local_tokens = self.local_attention(queries, keys, visual_values, need_weights=False)[0]
        return torch.cat([self.global_norm(global_tokens), self.local_norm(local_tokens)], dim=1).squeeze(0)
