"""Donor-fidelity trajectory-register decoder used by Register64.

The residual ordering follows DrivoR/DriveVLA-M0: register self-attention,
register-to-scene cross-attention, then an FFN.  This module is deliberately
independent of the legacy trajectory-scorer attention implementation.
"""

from __future__ import annotations

from typing import List, Optional, Union

import torch
from torch import Tensor, nn


class DropPath(nn.Module):
    """Per-sample stochastic depth with no dependency on timm."""

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        if not 0.0 <= float(drop_prob) < 1.0:
            raise ValueError("drop_prob must lie in [0, 1)")
        self.drop_prob = float(drop_prob)

    def forward(self, value: Tensor) -> Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return value
        keep_prob = 1.0 - self.drop_prob
        shape = (value.shape[0],) + (1,) * (value.ndim - 1)
        mask = value.new_empty(shape).bernoulli_(keep_prob)
        return value * mask.div_(keep_prob)


class LayerScale(nn.Module):
    """Learned residual scale used when a positive initialization is requested."""

    def __init__(self, dim: int, init_value: float) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.full((dim,), float(init_value)))

    def forward(self, value: Tensor) -> Tensor:
        return value * self.gamma


class RegisterMLP(nn.Module):
    """Two-layer GELU FFN matching the donor transformer block."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, value: Tensor) -> Tensor:
        value = self.fc1(value)
        value = self.activation(value)
        value = self.dropout1(value)
        value = self.fc2(value)
        return self.dropout2(value)


class RegisterAttention(nn.Module):
    """Batch-first donor attention with projection dropout."""

    def __init__(
        self,
        dim: int = 256,
        num_heads: int = 1,
        proj_drop: float = 0.1,
    ) -> None:
        super().__init__()
        if dim <= 0 or num_heads <= 0 or dim % num_heads:
            raise ValueError("dim must be positive and divisible by num_heads")
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=0.0,
            bias=True,
            batch_first=True,
        )
        self.proj_drop = nn.Dropout(float(proj_drop))

    def forward(self, query: Tensor, key_value: Optional[Tensor] = None) -> Tensor:
        if key_value is None:
            key_value = query
        if query.ndim != 3 or key_value.ndim != 3:
            raise ValueError("register attention expects [B,N,D] tensors")
        if query.shape[0] != key_value.shape[0] or query.shape[-1] != key_value.shape[-1]:
            raise ValueError("query and key/value must share batch and feature dimensions")
        output = self.attn(
            query=query,
            key=key_value,
            value=key_value,
            need_weights=False,
        )[0]
        return self.proj_drop(output)


def _layer_scale(dim: int, init_value: float) -> nn.Module:
    return LayerScale(dim, init_value) if float(init_value) > 0.0 else nn.Identity()


def _drop_path(probability: float) -> nn.Module:
    return DropPath(probability) if float(probability) > 0.0 else nn.Identity()


class RegisterDecoderBlock(nn.Module):
    """Self-attention -> scene cross-attention -> FFN residual block."""

    def __init__(
        self,
        dim: int = 256,
        num_heads: int = 1,
        mlp_ratio: float = 4.0,
        proj_drop: float = 0.1,
        drop_path: float = 0.2,
        layer_scale_init: float = 0.0,
    ) -> None:
        super().__init__()
        if dim <= 0 or mlp_ratio <= 0:
            raise ValueError("dim and mlp_ratio must be positive")
        hidden_dim = int(round(dim * float(mlp_ratio)))

        self.self_attn_norm = nn.LayerNorm(dim)
        self.self_attn = RegisterAttention(dim, num_heads, proj_drop)
        self.self_attn_layer_scale = _layer_scale(dim, layer_scale_init)
        self.self_attn_drop_path = _drop_path(drop_path)

        self.cross_attn_norm_query = nn.LayerNorm(dim)
        self.cross_attn_norm_memory = nn.LayerNorm(dim)
        self.cross_attn = RegisterAttention(dim, num_heads, proj_drop)
        self.cross_attn_layer_scale = _layer_scale(dim, layer_scale_init)
        self.cross_attn_drop_path = _drop_path(drop_path)

        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = RegisterMLP(dim, hidden_dim, dim, dropout=proj_drop)
        self.ffn_layer_scale = _layer_scale(dim, layer_scale_init)
        self.ffn_drop_path = _drop_path(drop_path)

    def forward(self, trajectory_tokens: Tensor, scene_tokens: Tensor) -> Tensor:
        if trajectory_tokens.ndim != 3 or scene_tokens.ndim != 3:
            raise ValueError("decoder inputs must have shape [B,N,D]")
        if trajectory_tokens.shape[0] != scene_tokens.shape[0]:
            raise ValueError("trajectory and scene tokens must share batch size")
        if trajectory_tokens.shape[-1] != scene_tokens.shape[-1]:
            raise ValueError("trajectory and scene tokens must share model dimension")

        trajectory_tokens = trajectory_tokens + self.self_attn_drop_path(
            self.self_attn_layer_scale(
                self.self_attn(self.self_attn_norm(trajectory_tokens))
            )
        )
        trajectory_tokens = trajectory_tokens + self.cross_attn_drop_path(
            self.cross_attn_layer_scale(
                self.cross_attn(
                    self.cross_attn_norm_query(trajectory_tokens),
                    self.cross_attn_norm_memory(scene_tokens),
                )
            )
        )
        trajectory_tokens = trajectory_tokens + self.ffn_drop_path(
            self.ffn_layer_scale(self.ffn(self.ffn_norm(trajectory_tokens)))
        )
        return trajectory_tokens


class RegisterTrajectoryDecoder(nn.Module):
    """Four-layer trajectory decoder returning every register-token stage."""

    def __init__(
        self,
        num_layers: int = 4,
        model_dim: int = 256,
        num_heads: int = 1,
        ffn_dim: int = 1024,
        proj_drop: float = 0.1,
        drop_path: float = 0.2,
        layer_scale_init: float = 0.0,
        return_intermediate: bool = True,
    ) -> None:
        super().__init__()
        if num_layers <= 0 or model_dim <= 0 or ffn_dim <= 0:
            raise ValueError("decoder dimensions and num_layers must be positive")
        if ffn_dim % model_dim:
            raise ValueError("ffn_dim must be an integer multiple of model_dim")
        self.num_layers = int(num_layers)
        self.model_dim = int(model_dim)
        self.num_heads = int(num_heads)
        self.return_intermediate = bool(return_intermediate)
        mlp_ratio = float(ffn_dim) / float(model_dim)
        self.layers = nn.ModuleList(
            [
                RegisterDecoderBlock(
                    dim=model_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    proj_drop=proj_drop,
                    drop_path=drop_path,
                    layer_scale_init=layer_scale_init,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self, trajectory_tokens: Tensor, scene_tokens: Tensor
    ) -> Union[List[Tensor], Tensor]:
        outputs: List[Tensor] = []
        for layer in self.layers:
            trajectory_tokens = layer(trajectory_tokens, scene_tokens)
            outputs.append(trajectory_tokens)
        return outputs if self.return_intermediate else trajectory_tokens
