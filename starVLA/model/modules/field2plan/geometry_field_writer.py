"""Teacher-free ego-centric geometry field writer for the Phase 1 MVP."""

from __future__ import annotations

from typing import Sequence, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from .camera_geometry import make_ego_bev_anchors, project_ego_points
from .types import CameraBatch, GeometryFieldOutput


class GeometryFieldWriter(nn.Module):
    """Project ego BEV anchors into multi-view features.

    Inputs are visual features ``[B,V,Cin,Hf,Wf]`` and calibrated cameras.
    The output is ``[B,Cout,Ny,Nx]``. No trajectory, action or future ground
    truth appears in this interface.
    """

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        field_size: Sequence[int],
        x_range_m: Sequence[float],
        y_range_m: Sequence[float],
        height_anchors_m: Sequence[float],
    ) -> None:
        super().__init__()
        if len(field_size) != 2 or min(field_size) <= 0:
            raise ValueError("field_size must be [Ny,Nx] with positive values")
        if len(x_range_m) != 2 or x_range_m[0] >= x_range_m[1]:
            raise ValueError("x_range_m must be increasing")
        if len(y_range_m) != 2 or y_range_m[0] >= y_range_m[1]:
            raise ValueError("y_range_m must be increasing")
        if not height_anchors_m:
            raise ValueError("height_anchors_m cannot be empty")
        self.input_channels = int(input_channels)
        self.output_channels = int(output_channels)
        self.field_size = (int(field_size[0]), int(field_size[1]))
        self.x_range_m = (float(x_range_m[0]), float(x_range_m[1]))
        self.y_range_m = (float(y_range_m[0]), float(y_range_m[1]))
        self.register_buffer(
            "height_anchors_m",
            torch.tensor(tuple(height_anchors_m), dtype=torch.float32),
            persistent=True,
        )
        self.view_score = nn.Linear(self.input_channels, 1)
        # Apply a pointwise channel MLP in NHWC layout.  This has the same
        # per-anchor role as a pair of 1x1 convolutions and deliberately avoids
        # a shape/data-dependent BF16 Conv2d dgrad fault observed on the PPU
        # development runtime for the real 24x24 field.
        self.output_projection = nn.Sequential(
            nn.Linear(self.input_channels, self.output_channels),
            nn.GELU(),
            nn.Linear(self.output_channels, self.output_channels),
        )

    def _anchors(self, device: torch.device) -> torch.Tensor:
        return make_ego_bev_anchors(
            self.field_size,
            self.x_range_m,
            self.y_range_m,
            self.height_anchors_m,
            device=device,
        )

    def forward(
        self,
        visual_features: torch.Tensor,
        camera: CameraBatch,
    ) -> GeometryFieldOutput:
        """Write ``[B,V,C,H,W]`` visual features into an ego field."""

        if visual_features.ndim != 5:
            raise ValueError("visual_features must have shape [B,V,C,H,W]")
        batch, views, channels, _, _ = visual_features.shape
        if channels != self.input_channels:
            raise ValueError(
                f"expected {self.input_channels} visual channels, got {channels}"
            )
        camera.validate()
        if camera.intrinsics.shape[:2] != (batch, views):
            raise ValueError("camera B,V dimensions must match visual_features")

        anchors = self._anchors(visual_features.device)
        ny, nx, nz = anchors.shape[:3]
        points = anchors.reshape(1, ny, nx, nz, 3).expand(batch, -1, -1, -1, -1)
        pixels, valid, _ = project_ego_points(
            points,
            camera.intrinsics,
            camera.ego_to_camera,
            camera.image_hw,
        )
        image_hw = camera.image_hw.to(device=visual_features.device, dtype=torch.float32)
        width = image_hw[..., 1, None, None, None]
        height = image_hw[..., 0, None, None, None]
        grid_x = 2.0 * (pixels[..., 0] + 0.5) / width - 1.0
        grid_y = 2.0 * (pixels[..., 1] + 0.5) / height - 1.0
        grid = torch.stack((grid_x, grid_y), dim=-1).reshape(
            batch * views, ny * nx * nz, 1, 2
        )

        # Projection and sampling coordinates stay float32.  Sampling in
        # float32 is also CPU-safe; autocast can lower the learned writer ops.
        sampled = F.grid_sample(
            visual_features.reshape(batch * views, channels, *visual_features.shape[-2:]).float(),
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        sampled = sampled[..., 0].reshape(batch, views, channels, ny, nx, nz)
        sampled = sampled.permute(0, 1, 5, 2, 3, 4)  # [B,V,Z,C,Ny,Nx]
        valid_sources = valid.permute(0, 1, 4, 2, 3)  # [B,V,Z,Ny,Nx]

        score_input = sampled.permute(0, 1, 2, 4, 5, 3)
        logits = self.view_score(score_input).squeeze(-1)
        flat_logits = logits.reshape(batch, views * nz, ny, nx)
        flat_valid = valid_sources.reshape(batch, views * nz, ny, nx)
        weights = torch.softmax(flat_logits.masked_fill(~flat_valid, -1e4), dim=1)
        weights = weights * flat_valid.to(weights.dtype)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        flat_sampled = sampled.reshape(batch, views * nz, channels, ny, nx)
        aggregated = (flat_sampled * weights[:, :, None]).sum(dim=1)
        field = self.output_projection(aggregated.permute(0, 2, 3, 1))
        field = field.permute(0, 3, 1, 2).contiguous()
        valid_ratio = flat_valid.float().mean(dim=(1, 2, 3))
        return GeometryFieldOutput(field, valid_ratio, valid_sources)
