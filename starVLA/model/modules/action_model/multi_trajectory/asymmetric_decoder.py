"""Transformer decoder with 256-wide queries and 2048-wide memory."""

from __future__ import annotations

from typing import List, Optional, Union

from torch import Tensor, nn


class AsymmetricTransformerDecoderLayer(nn.Module):
    """Pre-LN decoder layer with layer-local asymmetric K/V projections."""

    def __init__(
        self,
        planning_dim: int = 256,
        memory_dim: int = 2048,
        num_heads: int = 8,
        ffn_dim: int = 1024,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if planning_dim <= 0 or memory_dim <= 0:
            raise ValueError("planning_dim and memory_dim must be positive")
        if num_heads <= 0 or planning_dim % num_heads:
            raise ValueError("planning_dim must be divisible by num_heads")
        if ffn_dim <= 0:
            raise ValueError("ffn_dim must be positive")
        self.planning_dim = planning_dim
        self.memory_dim = memory_dim
        self.self_norm = nn.LayerNorm(planning_dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=planning_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.self_dropout = nn.Dropout(dropout)
        self.cross_norm = nn.LayerNorm(planning_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=planning_dim,
            num_heads=num_heads,
            dropout=dropout,
            kdim=memory_dim,
            vdim=memory_dim,
            batch_first=True,
        )
        self.cross_dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(planning_dim)
        self.ffn = nn.Sequential(
            nn.Linear(planning_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, planning_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        *,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        tgt_attn_mask: Optional[Tensor] = None,
    ) -> Tensor:
        if tgt.ndim != 3 or tgt.shape[-1] != self.planning_dim:
            raise ValueError(
                f"tgt must have shape [B, N, {self.planning_dim}]"
            )
        if memory.ndim != 3 or memory.shape[-1] != self.memory_dim:
            raise ValueError(
                f"memory must have shape [B, S, {self.memory_dim}]"
            )
        if tgt.shape[0] != memory.shape[0]:
            raise ValueError("tgt and memory batch sizes differ")
        if tgt.device != memory.device:
            raise ValueError("tgt and memory must be on the same device")
        if memory_key_padding_mask is not None and tuple(
            memory_key_padding_mask.shape
        ) != tuple(memory.shape[:2]):
            raise ValueError("memory_key_padding_mask must have shape [B, S]")

        q = self.self_norm(tgt)
        self_delta, _ = self.self_attn(
            q,
            q,
            q,
            attn_mask=tgt_attn_mask,
            key_padding_mask=tgt_key_padding_mask,
            need_weights=False,
        )
        tgt = tgt + self.self_dropout(self_delta)
        q = self.cross_norm(tgt)
        cross_delta, _ = self.cross_attn(
            q,
            memory,
            memory,
            key_padding_mask=memory_key_padding_mask,
            need_weights=False,
        )
        tgt = tgt + self.cross_dropout(cross_delta)
        return tgt + self.ffn(self.ffn_norm(tgt))


class AsymmetricTransformerDecoder(nn.Module):
    """Stack independent asymmetric decoder layers."""

    def __init__(
        self,
        num_layers: int,
        planning_dim: int = 256,
        memory_dim: int = 2048,
        num_heads: int = 8,
        ffn_dim: int = 1024,
        dropout: float = 0.0,
        return_intermediate: bool = False,
    ) -> None:
        super().__init__()
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        self.return_intermediate = return_intermediate
        self.layers = nn.ModuleList(
            [
                AsymmetricTransformerDecoderLayer(
                    planning_dim=planning_dim,
                    memory_dim=memory_dim,
                    num_heads=num_heads,
                    ffn_dim=ffn_dim,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        *,
        memory_key_padding_mask: Optional[Tensor] = None,
    ) -> Union[Tensor, List[Tensor]]:
        outputs: List[Tensor] = []
        for layer in self.layers:
            tgt = layer(
                tgt,
                memory,
                memory_key_padding_mask=memory_key_padding_mask,
            )
            if self.return_intermediate:
                outputs.append(tgt)
        return outputs if self.return_intermediate else tgt
