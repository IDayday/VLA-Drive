"""Action-query reader centered against a matched geometry reference."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class CenteredActionGeometryReader(nn.Module):
    def __init__(
        self,
        action_dim: int = 2048,
        geometry_dim: int = 512,
        num_heads: int = 8,
        alpha_min: float = 0.05,
        alpha_max: float = 0.30,
        alpha_initial: float = 0.10,
    ) -> None:
        super().__init__()
        if not alpha_min < alpha_initial < alpha_max:
            raise ValueError("alpha_initial must be strictly inside its bounds")
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)
        self.action_norm = nn.LayerNorm(action_dim)
        self.action_projection = nn.Linear(action_dim, geometry_dim, bias=False)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=geometry_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.readout_norm = nn.LayerNorm(geometry_dim)
        self.up_projection = nn.Linear(geometry_dim, action_dim)
        nn.init.zeros_(self.up_projection.weight)
        nn.init.zeros_(self.up_projection.bias)
        probability = (alpha_initial - alpha_min) / (alpha_max - alpha_min)
        self.scale_logit = nn.Parameter(
            torch.tensor(math.log(probability / (1.0 - probability)))
        )

    @property
    def alpha(self) -> Tensor:
        return self.alpha_min + (self.alpha_max - self.alpha_min) * torch.sigmoid(
            self.scale_logit
        )

    def forward(
        self,
        action_queries: Tensor,
        real_geometry: Tensor,
        reference_geometry: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        if action_queries.ndim != 3 or action_queries.shape[1] != 8:
            raise ValueError("action_queries must be [B,8,2048]")
        if real_geometry.shape != reference_geometry.shape:
            raise ValueError("real and reference geometry must have identical shapes")
        if real_geometry.ndim != 3 or real_geometry.shape[:2] != (
            action_queries.shape[0],
            180,
        ):
            raise ValueError("geometry memories must be [B,180,512]")
        q = self.action_projection(self.action_norm(action_queries))
        real_readout, real_attention = self.cross_attention(
            q,
            real_geometry,
            real_geometry,
            need_weights=True,
            average_attn_weights=False,
        )
        reference_readout, _ = self.cross_attention(
            q,
            reference_geometry,
            reference_geometry,
            need_weights=True,
            average_attn_weights=False,
        )
        centered = real_readout - reference_readout
        # LayerNorm and Linear both own affine biases by contract. Center the
        # affine readout as well, so equal memories remain an exact identity
        # even after those biases have been trained.
        normalized_centered = self.readout_norm(centered)
        normalized_origin = self.readout_norm(torch.zeros_like(centered))
        delta = self.up_projection(normalized_centered) - self.up_projection(
            normalized_origin
        )
        residual = self.alpha * delta
        enhanced = action_queries + residual
        action_norm = action_queries.detach().float().norm(dim=-1)
        residual_norm = residual.detach().float().norm(dim=-1)
        probabilities = real_attention.detach().float().clamp_min(1e-12)
        entropy = -(probabilities * probabilities.log()).sum(dim=-1).mean()
        diagnostics = {
            "gp_sq3dmix/alpha": self.alpha.detach(),
            "gp_sq3dmix/centered_readout_norm": centered.detach().float().norm(dim=-1).mean(),
            "gp_sq3dmix/residual_norm": residual_norm.mean(),
            "gp_sq3dmix/action_query_norm": action_norm.mean(),
            "gp_sq3dmix/residual_action_ratio": (
                residual_norm / action_norm.clamp_min(1e-12)
            ).mean(),
            "gp_sq3dmix/real_reference_readout_cosine": F.cosine_similarity(
                real_readout.detach().float(),
                reference_readout.detach().float(),
                dim=-1,
            ).mean(),
            "gp_sq3dmix/reader_attention_entropy": entropy,
            # Private framework payloads; they are consumed immediately and
            # never attached to the public metric dictionary.
            "_centered_readout": centered.detach(),
            "_residual_action_ratio_per_horizon": (
                residual_norm / action_norm.clamp_min(1e-12)
            ),
        }
        for horizon in range(action_queries.shape[1]):
            diagnostics[f"gp_sq3dmix/residual_norm_h{horizon}"] = residual_norm[
                :, horizon
            ].mean()
        return enhanced, diagnostics
