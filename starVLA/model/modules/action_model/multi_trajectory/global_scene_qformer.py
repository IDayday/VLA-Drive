"""Global 2048-wide Q-Former over the full final Qwen hidden sequence.

The module follows the global-query/self-attention/cross-attention/FFN
pattern described by DriveVLA-M0's Q-Former, but contains none of its
trajectory generation, memory retrieval, scoring, or test-time training.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .scene_context import SceneContext


class GlobalSceneQFormerBlock(nn.Module):
    """One pre-normalized global-query transformer block."""

    def __init__(
        self,
        scene_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if scene_dim <= 0 or num_heads <= 0 or scene_dim % num_heads:
            raise ValueError("scene_dim must be positive and divisible by num_heads")
        if ffn_dim <= 0:
            raise ValueError("ffn_dim must be positive")
        self.self_norm = nn.LayerNorm(scene_dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=scene_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.self_dropout = nn.Dropout(dropout)
        self.cross_norm = nn.LayerNorm(scene_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=scene_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(scene_dim)
        self.ffn = nn.Sequential(
            nn.Linear(scene_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, scene_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        queries: Tensor,
        memory: Tensor,
        *,
        memory_key_padding_mask: Tensor,
    ) -> Tensor:
        q = self.self_norm(queries)
        self_delta, _ = self.self_attn(q, q, q, need_weights=False)
        queries = queries + self.self_dropout(self_delta)

        q = self.cross_norm(queries)
        cross_delta, _ = self.cross_attn(
            q,
            memory,
            memory,
            key_padding_mask=memory_key_padding_mask,
            need_weights=False,
        )
        queries = queries + self.cross_dropout(cross_delta)
        queries = queries + self.ffn(self.ffn_norm(queries))
        return queries


class GlobalSceneQFormer(nn.Module):
    """Compress the full Qwen sequence into global queries without losing it."""

    def __init__(
        self,
        input_dim: int,
        scene_dim: int = 2048,
        num_queries: int = 16,
        num_layers: int = 4,
        num_heads: int = 32,
        ffn_dim: int = 8192,
        dropout: float = 0.0,
        query_init_std: float = 1e-6,
        *,
        debug_validate_finite: bool = False,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or scene_dim <= 0:
            raise ValueError("input_dim and scene_dim must be positive")
        if num_queries <= 0 or num_layers <= 0:
            raise ValueError("num_queries and num_layers must be positive")
        if query_init_std < 0:
            raise ValueError("query_init_std cannot be negative")
        self.input_dim = input_dim
        self.scene_dim = scene_dim
        self.num_queries = num_queries
        self.debug_validate_finite = debug_validate_finite

        # Deliberately retained even when input_dim == scene_dim: this is the
        # learned Qwen-to-planning-scene representation adapter.
        self.input_proj = nn.Linear(input_dim, scene_dim)
        self.scene_queries = nn.Parameter(
            torch.randn(1, num_queries, scene_dim) * query_init_std
        )
        self.blocks = nn.ModuleList(
            [
                GlobalSceneQFormerBlock(
                    scene_dim=scene_dim,
                    num_heads=num_heads,
                    ffn_dim=ffn_dim,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.output_norm = nn.LayerNorm(scene_dim)

    def forward(
        self,
        full_hidden_state: Tensor,
        attention_mask: Tensor,
        *,
        detach_input: bool = True,
    ) -> SceneContext:
        if full_hidden_state.ndim != 3:
            raise ValueError("full_hidden_state must have shape [B, L, D_qwen]")
        if attention_mask.ndim != 2:
            raise ValueError("attention_mask must have shape [B, L]")
        if tuple(attention_mask.shape) != tuple(full_hidden_state.shape[:2]):
            raise ValueError("attention_mask shape must match the full Qwen sequence")
        if attention_mask.device != full_hidden_state.device:
            raise ValueError("attention_mask and full_hidden_state must share a device")
        if full_hidden_state.shape[-1] != self.input_dim:
            raise ValueError(
                f"Qwen hidden width is {full_hidden_state.shape[-1]}, expected "
                f"{self.input_dim}"
            )
        if detach_input:
            full_hidden_state = full_hidden_state.detach()

        dense_scene_memory = self.input_proj(full_hidden_state)
        memory_key_padding_mask = ~attention_mask.bool()
        # Under AMP the projection output is typically bf16 while parameters
        # remain fp32.  Cast the expanded query view to the actual scene-memory
        # dtype so residuals do not silently promote all global tokens to fp32.
        # The differentiable cast preserves gradients to ``scene_queries``.
        queries = self.scene_queries.to(
            device=dense_scene_memory.device,
            dtype=dense_scene_memory.dtype,
        ).expand(full_hidden_state.shape[0], -1, -1)
        for block in self.blocks:
            queries = block(
                queries,
                dense_scene_memory,
                memory_key_padding_mask=memory_key_padding_mask,
            )
        global_scene_tokens = self.output_norm(queries).to(
            dtype=dense_scene_memory.dtype
        )
        context = SceneContext(
            global_scene_tokens=global_scene_tokens,
            dense_scene_memory=dense_scene_memory,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        context.validate(
            expected_num_queries=self.num_queries,
            expected_scene_dim=self.scene_dim,
            check_finite=self.debug_validate_finite,
        )
        return context
