"""Small action-conditioned probes for frozen driving-scene representations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn


ProbeInputMode = Literal["scene_action", "scene_only", "trajectory_only", "zero_action"]


@dataclass(frozen=True)
class ProbeForwardComponents:
    """Intermediate tensors exposed only for wiring and representation audits."""

    scene_embedding: torch.Tensor
    action_embedding: torch.Tensor
    fusion_input: torch.Tensor
    effect_latent: torch.Tensor


class _TokenPool(nn.Module):
    """Normalize, project, and attention-pool a short token sequence."""

    def __init__(self, input_dim: int, output_dim: int, dropout: float) -> None:
        super().__init__()
        self.project = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.score = nn.Linear(output_dim, 1)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        hidden = self.project(tokens)
        weights = torch.softmax(self.score(hidden).squeeze(-1), dim=-1)
        return torch.sum(hidden * weights.unsqueeze(-1), dim=1)


class _StructuredFutureDecoder(nn.Module):
    """Decode one latent into a compact horizon/channel BEV tensor."""

    def __init__(self, latent_dim: int, shape: tuple[int, ...]) -> None:
        super().__init__()
        if len(shape) != 4 or shape[-1] != shape[-2] or shape[-1] % 4:
            raise ValueError("structured future shape must be [H,C,R,R] with R divisible by four")
        self.shape = shape
        self.base_resolution = shape[-1] // 4
        output_channels = shape[0] * shape[1]
        self.project = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, 128 * self.base_resolution * self.base_resolution),
            nn.GELU(),
        )
        self.decode = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(64, output_channels, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        feature = self.project(latent).reshape(
            latent.shape[0], 128, self.base_resolution, self.base_resolution
        )
        return self.decode(feature).reshape(latent.shape[0], *self.shape)


class ActionEffectWorldProbe(nn.Module):
    """Candidate-conditioned lightweight world probe.

    The probe is deliberately independent of the Qwen+DiT implementation. It
    consumes offline frozen tokens, so Phase 5 cannot modify the baseline. The
    optional structured head is enabled only after the consequence target has
    passed its learnability check.
    """

    def __init__(
        self,
        *,
        scene_input_dim: int,
        consequence_dim: int,
        latent_dim: int = 192,
        trajectory_input_dim: int = 4,
        trajectory_token_dim: int = 96,
        action_hidden_dim: int | None = None,
        dropout: float = 0.1,
        structured_future_shape: tuple[int, ...] | None = None,
        input_mode: ProbeInputMode = "scene_action",
    ) -> None:
        super().__init__()
        if input_mode not in {"scene_action", "scene_only", "trajectory_only", "zero_action"}:
            raise ValueError(f"unsupported probe input mode: {input_mode}")
        self.input_mode = input_mode
        self.scene_encoder = _TokenPool(scene_input_dim, latent_dim, dropout)
        self.action_hidden_encoder = (
            _TokenPool(action_hidden_dim, latent_dim, dropout)
            if action_hidden_dim is not None
            else None
        )
        self.trajectory_project = nn.Sequential(
            nn.LayerNorm(trajectory_input_dim),
            nn.Linear(trajectory_input_dim, trajectory_token_dim),
            nn.GELU(),
        )
        trajectory_layer = nn.TransformerEncoderLayer(
            d_model=trajectory_token_dim,
            nhead=4,
            dim_feedforward=trajectory_token_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.trajectory_encoder = nn.TransformerEncoder(trajectory_layer, num_layers=1)
        self.trajectory_pool = nn.Sequential(
            nn.LayerNorm(trajectory_token_dim),
            nn.Linear(trajectory_token_dim, latent_dim),
            nn.GELU(),
        )
        self.fusion = nn.Sequential(
            nn.LayerNorm(latent_dim * 4),
            nn.Linear(latent_dim * 4, latent_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim * 2, latent_dim),
            nn.GELU(),
        )
        self.consequence_head = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, consequence_dim),
        )
        self.structured_future_shape = structured_future_shape
        self.structured_future_head = (
            _StructuredFutureDecoder(latent_dim, structured_future_shape)
            if structured_future_shape is not None
            else None
        )

    def encode_scene(
        self,
        scene_tokens: torch.Tensor,
        action_hidden: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode current-observation tokens without accessing future targets."""

        if scene_tokens.ndim != 3:
            raise ValueError("scene tokens must be a rank-3 tensor")
        scene = self.scene_encoder(scene_tokens)
        if action_hidden is not None:
            if self.action_hidden_encoder is None:
                raise ValueError("action_hidden was passed but no action-hidden encoder was configured")
            scene = scene + self.action_hidden_encoder(action_hidden)
        return scene

    def encode_trajectory(self, candidate_trajectory: torch.Tensor) -> torch.Tensor:
        """Encode a normalized candidate trajectory with the production probe path."""

        if candidate_trajectory.ndim != 3:
            raise ValueError("candidate trajectory must be a rank-3 tensor")
        trajectory_tokens = self.trajectory_project(candidate_trajectory)
        encoded = self.trajectory_encoder(trajectory_tokens).mean(dim=1)
        return self.trajectory_pool(encoded)

    def fuse_embeddings(
        self,
        scene_embedding: torch.Tensor,
        action_embedding: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fuse scene/action embeddings and return both fusion input and latent."""

        if scene_embedding.shape != action_embedding.shape:
            raise ValueError("scene and action embeddings must have identical shapes")
        fusion_input = torch.cat(
            (
                scene_embedding,
                action_embedding,
                scene_embedding * action_embedding,
                torch.abs(scene_embedding - action_embedding),
            ),
            dim=-1,
        )
        return fusion_input, self.fusion(fusion_input)

    def forward_components(
        self,
        scene_tokens: torch.Tensor,
        candidate_trajectory: torch.Tensor,
        action_hidden: torch.Tensor | None = None,
    ) -> ProbeForwardComponents:
        """Return auditable intermediate values while respecting ``input_mode``."""

        scene = self.encode_scene(scene_tokens, action_hidden)
        trajectory = self.encode_trajectory(candidate_trajectory)
        if self.input_mode == "scene_only":
            trajectory = torch.zeros_like(trajectory)
        elif self.input_mode == "trajectory_only":
            scene = torch.zeros_like(scene)
        elif self.input_mode == "zero_action":
            trajectory = trajectory * 0.0
        fusion_input, effect_latent = self.fuse_embeddings(scene, trajectory)
        return ProbeForwardComponents(
            scene_embedding=scene,
            action_embedding=trajectory,
            fusion_input=fusion_input,
            effect_latent=effect_latent,
        )

    def forward(
        self,
        scene_tokens: torch.Tensor,
        candidate_trajectory: torch.Tensor,
        action_hidden: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | None]:
        """Predict candidate effects from current-scene tokens and trajectory."""

        components = self.forward_components(scene_tokens, candidate_trajectory, action_hidden)
        effect_latent = components.effect_latent
        consequence = self.consequence_head(effect_latent)
        structured: torch.Tensor | None = None
        if self.structured_future_head is not None:
            assert self.structured_future_shape is not None
            structured = self.structured_future_head(effect_latent)
        return {
            "effect_latent": effect_latent,
            "consequence_prediction": consequence,
            "structured_future_prediction": structured,
        }


def count_parameters(module: nn.Module, *, trainable_only: bool = False) -> int:
    """Count parameters for experiment accounting."""

    return sum(
        parameter.numel()
        for parameter in module.parameters()
        if not trainable_only or parameter.requires_grad
    )
