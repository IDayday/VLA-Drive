"""Learnable scene queries that compress a masked Qwen hidden sequence."""

from __future__ import annotations

from typing import Dict

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class SceneQueryBlock(nn.Module):
    """One self-attention, cross-attention, and feed-forward query block."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0 or num_heads <= 0:
            raise ValueError("hidden_dim and num_heads must be positive")
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if mlp_ratio <= 0:
            raise ValueError("mlp_ratio must be positive")

        feedforward_dim = int(hidden_dim * mlp_ratio)
        self.ln_self = nn.LayerNorm(hidden_dim)
        self.self_attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ln_cross = nn.LayerNorm(hidden_dim)
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ln_ffn = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, feedforward_dim),
            nn.GELU(),
            nn.Linear(feedforward_dim, hidden_dim),
        )

    def forward(
        self,
        queries: Tensor,
        memory: Tensor,
        memory_key_padding_mask: Tensor,
    ) -> Tensor:
        normalized = self.ln_self(queries)
        attended, _ = self.self_attention(
            normalized,
            normalized,
            normalized,
            need_weights=False,
        )
        queries = queries + attended

        normalized = self.ln_cross(queries)
        attended, _ = self.cross_attention(
            normalized,
            memory,
            memory,
            key_padding_mask=memory_key_padding_mask,
            need_weights=False,
        )
        queries = queries + attended

        normalized = self.ln_ffn(queries)
        return queries + self.ffn(normalized)


class SceneQueryCompressor(nn.Module):
    """Compress valid Qwen tokens into a fixed set of external scene queries."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_queries: int = 16,
        num_layers: int = 4,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        query_init_std: float = 1e-6,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or num_queries <= 0 or num_layers <= 0:
            raise ValueError("input_dim, num_queries, and num_layers must be positive")
        if query_init_std < 0:
            raise ValueError("query_init_std must be non-negative")

        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_queries = int(num_queries)
        self.input_projection = nn.Linear(self.input_dim, self.hidden_dim)
        self.scene_queries = nn.Parameter(
            torch.randn(1, self.num_queries, self.hidden_dim) * query_init_std
        )
        self.blocks = nn.ModuleList(
            [
                SceneQueryBlock(
                    hidden_dim=self.hidden_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.output_norm = nn.LayerNorm(self.hidden_dim)
        self.output_projection = nn.Linear(self.hidden_dim, self.input_dim)

    @staticmethod
    def _diagnostics(scene_tokens: Tensor, valid_mask: Tensor) -> Dict[str, Tensor]:
        normalized = F.normalize(scene_tokens.detach().float(), dim=-1)
        pairwise = normalized @ normalized.transpose(1, 2)
        query_count = scene_tokens.shape[1]
        if query_count > 1:
            off_diagonal = ~torch.eye(
                query_count,
                dtype=torch.bool,
                device=scene_tokens.device,
            )
            pairwise_cosine = pairwise[:, off_diagonal].mean()
        else:
            pairwise_cosine = pairwise.mean()
        detached = scene_tokens.detach().float()
        return {
            "sq3dmix/scene_token_norm": detached.norm(dim=-1).mean(),
            "sq3dmix/scene_token_std": detached.std(unbiased=False),
            "sq3dmix/scene_token_pairwise_cosine": pairwise_cosine,
            "sq3dmix/scene_memory_valid_tokens": valid_mask.detach()
            .sum(dim=1)
            .float()
            .mean(),
        }

    def forward(
        self,
        hidden_states: Tensor,
        valid_mask: Tensor,
    ) -> tuple[Tensor, Dict[str, Tensor]]:
        if hidden_states.ndim != 3:
            raise ValueError("hidden_states must be [B,L,D]")
        if valid_mask.shape != hidden_states.shape[:2]:
            raise ValueError("valid_mask must match hidden_states [B,L]")
        if valid_mask.dtype != torch.bool:
            raise TypeError("valid_mask must have dtype bool")
        if hidden_states.shape[-1] != self.input_dim:
            raise ValueError(
                f"hidden dimension mismatch: expected {self.input_dim}, "
                f"found {hidden_states.shape[-1]}"
            )
        if hidden_states.device != valid_mask.device:
            raise ValueError("hidden_states and valid_mask must share a device")
        if not valid_mask.any(dim=1).all():
            raise ValueError("every sample needs at least one valid scene memory token")

        projected_input = hidden_states.to(dtype=self.input_projection.weight.dtype)
        memory = self.input_projection(projected_input)
        queries = self.scene_queries.to(dtype=memory.dtype).expand(
            hidden_states.shape[0], -1, -1
        )
        key_padding_mask = ~valid_mask
        for block in self.blocks:
            queries = block(queries, memory, key_padding_mask)
        scene_tokens = self.output_projection(self.output_norm(queries))
        return scene_tokens, self._diagnostics(scene_tokens, valid_mask)
