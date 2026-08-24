"""Shared feature/metadata adapter for real and reference geometry memory."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class GeometryMemoryAdapter(nn.Module):
    def __init__(
        self,
        feature_dim: int = 2048,
        geometry_dim: int = 512,
        view_count: int = 3,
    ) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.geometry_dim = int(geometry_dim)
        self.view_count = int(view_count)
        self.feature_norm = nn.LayerNorm(
            self.feature_dim, elementwise_affine=False
        )
        self.feature_projection = nn.Linear(
            self.feature_dim, self.geometry_dim, bias=False
        )
        self.metadata_projection = nn.Linear(
            self.view_count + 2 + 6, self.geometry_dim, bias=False
        )
        self.output_norm = nn.LayerNorm(
            self.geometry_dim, elementwise_affine=False
        )

    def forward(
        self,
        features: Tensor,
        view_ids: Tensor,
        uv_coords: Tensor,
        ray_features: Tensor,
    ) -> Tensor:
        if features.ndim != 3 or features.shape[-1] != self.feature_dim:
            raise ValueError(f"features must be [B,N,{self.feature_dim}]")
        batch, tokens, _ = features.shape
        if view_ids.shape != (batch, tokens):
            raise ValueError("view_ids must be [B,N]")
        if uv_coords.shape != (batch, tokens, 2):
            raise ValueError("uv_coords must be [B,N,2]")
        if ray_features.shape != (batch, tokens, 6):
            raise ValueError("ray_features must be [B,N,6]")
        if (view_ids < 0).any() or (view_ids >= self.view_count).any():
            raise ValueError("view_ids are outside the configured camera range")
        compute_dtype = self.feature_projection.weight.dtype
        feature_input = self.feature_norm(features.to(dtype=compute_dtype))
        metadata = torch.cat(
            (
                F.one_hot(view_ids.long(), num_classes=self.view_count).to(
                    dtype=compute_dtype
                ),
                uv_coords.to(dtype=compute_dtype),
                ray_features.to(dtype=compute_dtype),
            ),
            dim=-1,
        )
        geometry = self.feature_projection(feature_input) + self.metadata_projection(
            metadata
        )
        return self.output_norm(geometry)
