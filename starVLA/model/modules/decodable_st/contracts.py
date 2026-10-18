"""Closed input types; audit identities and targets never enter the student."""
from dataclasses import dataclass, fields
from typing import Any, Mapping, Optional, Tuple
import torch


@dataclass(frozen=True)
class AllowedObservation:
    images: Tuple[Tuple[Any, ...], ...]
    navigation: Tuple[str, ...]
    ego_state: torch.Tensor
    camera_ids: Tuple[str, ...]
    calibration: Optional[torch.Tensor] = None

    def validate(self, cameras, *, calibration_allowed=False):
        b = len(self.images)
        if not b or tuple(cameras) != self.camera_ids:
            raise ValueError("Observation camera contract mismatch")
        if len(self.navigation) != b or any(len(x) != len(cameras) for x in self.images):
            raise ValueError("Observation batch/view mismatch")
        if self.ego_state.shape != (b, 1, 4) or not torch.isfinite(self.ego_state).all():
            raise ValueError("Expected the audited baseline's legal [B,1,4] ego state")
        if self.calibration is not None and not calibration_allowed:
            raise ValueError("Additional calibration input was not registered")
        return self

    @classmethod
    def from_mapping(cls, values: Mapping):
        extra = set(values) - {f.name for f in fields(cls)}
        if extra:
            raise ValueError(f"Non-observation fields rejected: {sorted(extra)}")
        return cls(**values)


@dataclass(frozen=True)
class SampleIdentity:
    scene_token: str
    log_id: str
    sample_hash: str


@dataclass
class RepresentationTargets:
    """Trainer-only tensors. Each task has its own validity, never input masks."""
    values: Mapping[str, torch.Tensor]
    valid: Mapping[str, torch.Tensor]
    metadata: Mapping[str, Any]

    def validate(self):
        for k, v in self.values.items():
            if v.requires_grad:
                raise ValueError(f"Target {k} requires gradients")
        if not self.metadata.get("schema_version"):
            raise ValueError("Versioned target contract required")
        return self


@dataclass
class StudentRepresentation:
    original_action_hidden: torch.Tensor
    dino_hidden: torch.Tensor
    spatial_query_hidden: Optional[torch.Tensor] = None
    temporal_query_hidden: Optional[torch.Tensor] = None
    predicted_spatial_context: Optional[torch.Tensor] = None
    predicted_future_latent: Optional[torch.Tensor] = None


@dataclass(frozen=True)
class SpatialContract:
    channels: int
    height: int
    width: int
    roi_min: Tuple[float, float, float]
    roi_max: Tuple[float, float, float]
    # ViGT tensor height corresponds to physical x, width to physical y.
    axis_order: str = "CXY"

    def validate(self):
        if self.axis_order != "CXY" or min(self.channels, self.height, self.width) <= 0:
            raise ValueError("Unsupported ViGT context contract")
        if any(b <= a for a, b in zip(self.roi_min, self.roi_max)):
            raise ValueError("Invalid physical ROI")
        return self

    def in_roi(self, xyz):
        lo, hi = xyz.new_tensor(self.roi_min), xyz.new_tensor(self.roi_max)
        return torch.isfinite(xyz).all(-1) & (xyz >= lo).all(-1) & (xyz < hi).all(-1)


@dataclass(frozen=True)
class TemporalContract:
    hidden_size: int
    num_queries: int
    grid_xy: Tuple[int, int]
    height_bins: int
    horizon_seconds: Optional[float]
    coordinate_system: str
    semantics: str
    task_version: str

    def require_driving(self):
        if self.horizon_seconds is None or self.horizon_seconds <= 0:
            raise ValueError("BLOCKED_TEMPORAL_HORIZON: no verified physical horizon")
        if self.coordinate_system != "ego_t0_x_forward_y_left_z_up":
            raise ValueError("BLOCKED_TEMPORAL_COORDINATE: driving task requires ego(t0)")
