"""Small typed containers for Field2Plan tensor boundaries."""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch


@dataclass(frozen=True)
class CameraBatch:
    """Calibrated cameras.

    Shapes:
        intrinsics: ``[B,V,3,3]`` in resized-image pixel coordinates.
        ego_to_camera: ``[B,V,4,4]`` with camera z pointing forward.
        image_hw: ``[B,V,2]`` ordered as height, width.
    """

    intrinsics: torch.Tensor
    ego_to_camera: torch.Tensor
    image_hw: torch.Tensor
    view_names: Tuple[str, ...]
    frame_index: int

    def validate(self) -> "CameraBatch":
        if self.intrinsics.ndim != 4 or self.intrinsics.shape[-2:] != (3, 3):
            raise ValueError("intrinsics must have shape [B,V,3,3]")
        if self.ego_to_camera.shape != (*self.intrinsics.shape[:2], 4, 4):
            raise ValueError("ego_to_camera must have shape [B,V,4,4]")
        if self.image_hw.shape != (*self.intrinsics.shape[:2], 2):
            raise ValueError("image_hw must have shape [B,V,2]")
        if len(self.view_names) != self.intrinsics.shape[1]:
            raise ValueError("view_names length must equal V")
        return self


@dataclass(frozen=True)
class VisualFeatureOutput:
    """Visual feature maps ``[B,V,C,H,W]`` and their Qwen grid metadata."""

    features: torch.Tensor
    grid_thw: torch.Tensor
    view_names: Tuple[str, ...]


@dataclass(frozen=True)
class GeometryFieldOutput:
    """Ego field ``[B,C,Ny,Nx]`` and projection validity diagnostics."""

    field: torch.Tensor
    valid_ratio: torch.Tensor
    projection_valid: torch.Tensor


@dataclass(frozen=True)
class TemporalAlignment:
    """Ego-motion transforms for one sample or a batch.

    Shapes:
        global_from_ego: ``[...,T,4,4]`` maps each ego frame into global.
        current_from_ego: ``[...,T,4,4]`` maps each ego frame into current ego.
        ego_from_current: ``[...,T,4,4]`` maps current ego into each ego frame.
        frame_times_s: ``[T]`` relative to dataset frame zero.
        valid_mask: ``[...,T]``.
    """

    global_from_ego: torch.Tensor
    current_from_ego: torch.Tensor
    ego_from_current: torch.Tensor
    frame_times_s: torch.Tensor
    valid_mask: torch.Tensor
    current_index: int
    history_indices: Tuple[int, ...]
    future_indices: Tuple[int, ...]

    def validate(self) -> "TemporalAlignment":
        if self.global_from_ego.ndim not in (3, 4):
            raise ValueError("global_from_ego must have shape [T,4,4] or [B,T,4,4]")
        if self.global_from_ego.shape[-2:] != (4, 4):
            raise ValueError("global_from_ego matrices must have shape [4,4]")
        if self.current_from_ego.shape != self.global_from_ego.shape:
            raise ValueError("current_from_ego must match global_from_ego shape")
        if self.ego_from_current.shape != self.global_from_ego.shape:
            raise ValueError("ego_from_current must match global_from_ego shape")
        time_count = self.global_from_ego.shape[-3]
        if self.frame_times_s.shape != (time_count,):
            raise ValueError("frame_times_s must have shape [T]")
        if self.valid_mask.shape != self.global_from_ego.shape[:-2]:
            raise ValueError("valid_mask must have shape [...] matching temporal matrices")
        if not 0 <= self.current_index < time_count:
            raise ValueError("current_index is outside the temporal sequence")
        if any(index > self.current_index for index in self.history_indices):
            raise ValueError("history indices cannot occur after current_index")
        if any(index <= self.current_index for index in self.future_indices):
            raise ValueError("future indices must occur after current_index")
        return self


@dataclass(frozen=True)
class DynamicsFieldOutput:
    """Action-free future field and uncertainty.

    Shapes are ``field=[B,H,C,Ny,Nx]`` and
    ``log_variance=[B,H,1,Ny,Nx]``.
    """

    field: torch.Tensor
    log_variance: torch.Tensor

    def validate(self) -> "DynamicsFieldOutput":
        if self.field.ndim != 5:
            raise ValueError("dynamics field must have shape [B,H,C,Ny,Nx]")
        expected = (*self.field.shape[:2], 1, *self.field.shape[-2:])
        if self.log_variance.shape != expected:
            raise ValueError(
                "dynamics log_variance must have shape [B,H,1,Ny,Nx]"
            )
        return self


@dataclass(frozen=True)
class TemporalCameraBatch:
    """Future calibrated cameras used only to align offline supervision.

    Shapes:
        intrinsics: ``[B,H,V,3,3]`` in teacher-input pixels.
        ego_to_camera: ``[B,H,V,4,4]`` from each future ego frame.
        image_hw: ``[B,H,V,2]``.
        current_to_ego: ``[B,H,4,4]`` from current ego to future ego.
        valid_mask: ``[B,H,V]``.
    """

    intrinsics: torch.Tensor
    ego_to_camera: torch.Tensor
    image_hw: torch.Tensor
    current_to_ego: torch.Tensor
    valid_mask: torch.Tensor
    view_names: Tuple[str, ...]
    frame_indices: Tuple[int, ...]

    def validate(self) -> "TemporalCameraBatch":
        if self.intrinsics.ndim != 5 or self.intrinsics.shape[-2:] != (3, 3):
            raise ValueError("temporal intrinsics must have shape [B,H,V,3,3]")
        batch, horizon, views = self.intrinsics.shape[:3]
        if self.ego_to_camera.shape != (batch, horizon, views, 4, 4):
            raise ValueError("temporal ego_to_camera must be [B,H,V,4,4]")
        if self.image_hw.shape != (batch, horizon, views, 2):
            raise ValueError("temporal image_hw must be [B,H,V,2]")
        if self.current_to_ego.shape != (batch, horizon, 4, 4):
            raise ValueError("current_to_ego must be [B,H,4,4]")
        if self.valid_mask.shape != (batch, horizon, views):
            raise ValueError("temporal camera valid_mask must be [B,H,V]")
        if len(self.view_names) != views:
            raise ValueError("temporal camera view_names length must equal V")
        if len(self.frame_indices) != horizon:
            raise ValueError("temporal camera frame_indices length must equal H")
        return self


@dataclass(frozen=True)
class TubeReadoutOutput:
    """Trajectory readout tensors.

    Shapes are waypoint_context ``[B,M,H,C]``, valid_mask ``[B,M,H,P]``
    and source_gates ``[B,M,H,S]``.
    """

    waypoint_context: torch.Tensor
    valid_mask: torch.Tensor
    source_gates: torch.Tensor
    tube_points: torch.Tensor


@dataclass(frozen=True)
class RefinerOutput:
    """Refinement result with actions ``[B,M,H,4]`` and deltas ``[...,3]``."""

    final_action: torch.Tensor
    delta_physical: torch.Tensor
    delta_norm: torch.Tensor
    gate: Optional[torch.Tensor] = None
