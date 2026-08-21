# SPDX-License-Identifier: Apache-2.0
# Adapted from valeoai/DrivoR commit
# f02665403df799c1b4ddd8b0d34e073f0555c13a, file
# navsim/agents/drivoR/transformer_decoder.py, and from
# William-Yao-2000/DriveSuprim commit
# 80fe792d7654a596d92e20d030d1650f6f605c02, decoder blocks in
# navsim/agents/drivesuprim/drivesuprim_model.py.
# Project adaptations: pre-LN residual blocks with 256-dimensional candidate
# queries reading independently projected 2048-dimensional scene memories.

"""Asymmetric candidate-to-scene transformer decoder blocks."""

from __future__ import annotations

from typing import List, Optional, Union

import torch
from torch import Tensor, nn


class AsymmetricDecoderLayer(nn.Module):
    """Candidate self-attention followed by asymmetric scene cross-attention."""

    def __init__(
        self,
        query_dim: int = 256,
        memory_dim: int = 2048,
        num_heads: int = 8,
        ffn_dim: int = 1024,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if query_dim <= 0 or memory_dim <= 0 or ffn_dim <= 0:
            raise ValueError("decoder dimensions must be positive")
        if num_heads <= 0 or query_dim % num_heads:
            raise ValueError("query_dim must be divisible by num_heads")
        self.query_dim = query_dim
        self.memory_dim = memory_dim
        self.self_norm = nn.LayerNorm(query_dim)
        self.self_attn = nn.MultiheadAttention(
            query_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.self_dropout = nn.Dropout(dropout)
        self.cross_norm = nn.LayerNorm(query_dim)
        self.cross_attn = nn.MultiheadAttention(
            query_dim,
            num_heads,
            dropout=dropout,
            kdim=memory_dim,
            vdim=memory_dim,
            batch_first=True,
        )
        self.cross_dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(query_dim)
        self.ffn = nn.Sequential(
            nn.Linear(query_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, query_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        query: Tensor,
        memory: Tensor,
        *,
        memory_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        if query.ndim != 3 or query.shape[-1] != self.query_dim:
            raise ValueError(
                f"query must have shape [B,N,{self.query_dim}], got {tuple(query.shape)}"
            )
        if memory.ndim != 3 or memory.shape[-1] != self.memory_dim:
            raise ValueError(
                f"memory must have shape [B,S,{self.memory_dim}], got {tuple(memory.shape)}"
            )
        if query.shape[0] != memory.shape[0] or query.device != memory.device:
            raise ValueError("query and memory must share batch size and device")
        if memory_key_padding_mask is not None:
            if tuple(memory_key_padding_mask.shape) != tuple(memory.shape[:2]):
                raise ValueError("memory_key_padding_mask must have shape [B,S]")
            if memory_key_padding_mask.dtype is not torch.bool:
                raise TypeError("memory_key_padding_mask must have dtype torch.bool")
            if memory_key_padding_mask.device != memory.device:
                raise ValueError("memory mask and memory must share a device")

        normalized = self.self_norm(query)
        delta, _ = self.self_attn(
            normalized, normalized, normalized, need_weights=False
        )
        query = query + self.self_dropout(delta)

        normalized = self.cross_norm(query)
        delta, _ = self.cross_attn(
            normalized,
            memory,
            memory,
            key_padding_mask=memory_key_padding_mask,
            need_weights=False,
        )
        query = query + self.cross_dropout(delta)
        return query + self.ffn(self.ffn_norm(query))


class AsymmetricDecoder(nn.Module):
    """Stack decoder layers with independent key/value projection parameters."""

    def __init__(
        self,
        num_layers: int,
        query_dim: int = 256,
        memory_dim: int = 2048,
        num_heads: int = 8,
        ffn_dim: int = 1024,
        dropout: float = 0.0,
        return_intermediate: bool = False,
    ) -> None:
        super().__init__()
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        self.layers = nn.ModuleList(
            AsymmetricDecoderLayer(
                query_dim=query_dim,
                memory_dim=memory_dim,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                dropout=dropout,
            )
            for _ in range(num_layers)
        )
        self.return_intermediate = return_intermediate

    def forward(
        self,
        query: Tensor,
        memory: Tensor,
        *,
        memory_key_padding_mask: Optional[Tensor] = None,
    ) -> Union[Tensor, List[Tensor]]:
        outputs: List[Tensor] = []
        for layer in self.layers:
            query = layer(
                query,
                memory,
                memory_key_padding_mask=memory_key_padding_mask,
            )
            if self.return_intermediate:
                outputs.append(query)
        return outputs if self.return_intermediate else query
