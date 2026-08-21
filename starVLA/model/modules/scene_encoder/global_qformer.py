# SPDX-License-Identifier: Apache-2.0
# Adapted from ZebinX/DriveVLA-M0 commit
# 7fabe160fc9bb41f9278845b36d457bf871f697a, file
# navsim/agents/EpisodeDrive/layers/q_former/q_former.py.
# Project adaptations: consume the complete frozen-Qwen sequence, preserve its
# dense memory, implement PyTorch padding-mask semantics, expose configurable
# dimensions, and optionally checkpoint the four trainable Q-Former blocks.

"""Global query scene encoder over one complete frozen-Qwen hidden sequence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint


@dataclass
class SceneContext:
    """Scene memories produced by one Q-Former invocation.

    Shapes are ``global_tokens=[B,Q,D]``, ``dense_memory=[B,L,D]`` and
    ``memory_key_padding_mask=[B,L]``.  The mask follows PyTorch attention
    semantics: ``True`` means that the memory position must be ignored.
    """

    global_tokens: Tensor
    dense_memory: Tensor
    memory_key_padding_mask: Optional[Tensor]

    def validate(
        self,
        *,
        expected_num_queries: Optional[int] = None,
        expected_scene_dim: Optional[int] = None,
        check_finite: bool = False,
    ) -> None:
        if self.global_tokens.ndim != 3:
            raise ValueError("global_tokens must have shape [B,Q,D]")
        if self.dense_memory.ndim != 3:
            raise ValueError("dense_memory must have shape [B,L,D]")
        if self.global_tokens.shape[0] != self.dense_memory.shape[0]:
            raise ValueError("global and dense scene-memory batch sizes differ")
        if self.global_tokens.shape[-1] != self.dense_memory.shape[-1]:
            raise ValueError("global and dense scene-memory widths differ")
        if expected_num_queries is not None and self.global_tokens.shape[1] != expected_num_queries:
            raise ValueError(
                f"global query count is {self.global_tokens.shape[1]}, "
                f"expected {expected_num_queries}"
            )
        if expected_scene_dim is not None and self.global_tokens.shape[-1] != expected_scene_dim:
            raise ValueError(
                f"scene width is {self.global_tokens.shape[-1]}, "
                f"expected {expected_scene_dim}"
            )
        mask = self.memory_key_padding_mask
        if mask is not None:
            if mask.ndim != 2 or tuple(mask.shape) != tuple(self.dense_memory.shape[:2]):
                raise ValueError("memory_key_padding_mask must have shape [B,L]")
            if mask.dtype is not torch.bool:
                raise TypeError("memory_key_padding_mask must have dtype torch.bool")
            if mask.device != self.dense_memory.device:
                raise ValueError("scene memories and padding mask must share a device")
            if mask.all(dim=1).any():
                raise ValueError("every sample must contain at least one valid Qwen token")
        if self.global_tokens.device != self.dense_memory.device:
            raise ValueError("global and dense scene memories must share a device")
        if check_finite and (
            not torch.isfinite(self.global_tokens).all()
            or not torch.isfinite(self.dense_memory).all()
        ):
            raise ValueError("SceneContext contains NaN or Inf")


class GlobalSceneQFormerBlock(nn.Module):
    """DriveVLA-M0-style pre-LN self-attention/cross-attention/FFN block."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0 or num_heads <= 0 or hidden_dim % num_heads:
            raise ValueError("hidden_dim must be positive and divisible by num_heads")
        if ffn_dim <= 0:
            raise ValueError("ffn_dim must be positive")
        self.hidden_dim = hidden_dim
        self.self_norm = nn.LayerNorm(hidden_dim)
        self.self_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.self_dropout = nn.Dropout(dropout)
        self.cross_norm = nn.LayerNorm(hidden_dim)
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.cross_dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        queries: Tensor,
        memory: Tensor,
        memory_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        q = self.self_norm(queries)
        delta, _ = self.self_attn(q, q, q, need_weights=False)
        queries = queries + self.self_dropout(delta)

        q = self.cross_norm(queries)
        delta, _ = self.cross_attn(
            q,
            memory,
            memory,
            key_padding_mask=memory_key_padding_mask,
            need_weights=False,
        )
        queries = queries + self.cross_dropout(delta)
        return queries + self.ffn(self.ffn_norm(queries))


class GlobalSceneQFormer(nn.Module):
    """Encode a detached full Qwen sequence into global and dense memories.

    The Qwen input is detached at this module boundary.  The input LayerNorm,
    projection, learned queries and Q-Former blocks remain fully trainable.
    """

    def __init__(
        self,
        input_dim: int = 2048,
        hidden_dim: int = 2048,
        output_dim: int = 2048,
        num_queries: int = 16,
        num_layers: int = 4,
        num_heads: int = 32,
        ffn_dim: int = 8192,
        dropout: float = 0.0,
        query_init_std: float = 0.02,
        use_gradient_checkpointing: bool = False,
        debug_validate_finite: bool = False,
    ) -> None:
        super().__init__()
        if min(input_dim, hidden_dim, output_dim, num_queries, num_layers) <= 0:
            raise ValueError("all Q-Former dimensions and counts must be positive")
        if hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if query_init_std < 0:
            raise ValueError("query_init_std cannot be negative")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_queries = num_queries
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.debug_validate_finite = debug_validate_finite

        self.input_norm = nn.LayerNorm(input_dim)
        # Kept learnable even when input_dim == hidden_dim.
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.scene_queries = nn.Parameter(
            torch.empty(1, num_queries, hidden_dim).normal_(std=query_init_std)
        )
        self.blocks = nn.ModuleList(
            GlobalSceneQFormerBlock(hidden_dim, num_heads, ffn_dim, dropout)
            for _ in range(num_layers)
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = (
            nn.Identity() if output_dim == hidden_dim else nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, last_hidden: Tensor, attention_mask: Tensor) -> SceneContext:
        """Return scene memories without allowing gradients into Qwen.

        Args:
            last_hidden: Full final hidden sequence ``[B,L,input_dim]``.
            attention_mask: Qwen mask ``[B,L]`` where truthy entries are valid.
        """

        if last_hidden.ndim != 3 or last_hidden.shape[-1] != self.input_dim:
            raise ValueError(
                f"last_hidden must have shape [B,L,{self.input_dim}], got "
                f"{tuple(last_hidden.shape)}"
            )
        if attention_mask.ndim != 2 or tuple(attention_mask.shape) != tuple(last_hidden.shape[:2]):
            raise ValueError(
                "attention_mask must have shape [B,L] matching the complete Qwen sequence"
            )
        if attention_mask.device != last_hidden.device:
            raise ValueError("last_hidden and attention_mask must share a device")

        detached_hidden = last_hidden.detach()
        hidden_memory = self.input_proj(self.input_norm(detached_hidden))
        memory_key_padding_mask = ~attention_mask.bool()
        queries = self.scene_queries.to(dtype=hidden_memory.dtype).expand(
            hidden_memory.shape[0], -1, -1
        )
        for block in self.blocks:
            if self.use_gradient_checkpointing and self.training and torch.is_grad_enabled():
                queries = checkpoint(
                    block,
                    queries,
                    hidden_memory,
                    memory_key_padding_mask,
                    use_reentrant=False,
                )
            else:
                queries = block(queries, hidden_memory, memory_key_padding_mask)

        # Some accelerator LayerNorm kernels promote their output to fp32
        # under BF16 autocast.  SceneContext follows the frozen-Qwen input
        # dtype so downstream Flow/scorer conditions have one stable AMP
        # contract; the differentiable cast keeps all Q-Former gradients.
        global_tokens = self.output_proj(self.output_norm(queries)).to(
            dtype=last_hidden.dtype
        )
        dense_memory = self.output_proj(hidden_memory).to(dtype=last_hidden.dtype)
        context = SceneContext(
            global_tokens=global_tokens,
            dense_memory=dense_memory,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        context.validate(
            expected_num_queries=self.num_queries,
            expected_scene_dim=self.output_dim,
            check_finite=self.debug_validate_finite,
        )
        return context
