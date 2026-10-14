"""Versioned, explicit boundaries; no Scene or target object enters a provider."""
from dataclasses import dataclass
from typing import Mapping, Optional, Tuple
import torch

SCHEMA_VERSION = 1

@dataclass(frozen=True)
class ModelInputs:
    current_images: torch.Tensor  # B,V,3,H,W
    camera_intrinsics: torch.Tensor  # B,V,3,3, after image transform
    camera_extrinsics: torch.Tensor  # B,V,4,4, camera -> ego(t0)
    image_transforms: torch.Tensor  # B,V,3,3, raw pixels -> input pixels
    camera_names: Tuple[str, ...]
    timestamps: torch.Tensor  # B,V
    decision_time: torch.Tensor  # B
    ego_state: Optional[torch.Tensor] = None
    ego_history: Optional[torch.Tensor] = None
    navigation: Optional[torch.Tensor] = None
    distortion: Optional[torch.Tensor] = None  # B,V,5 OpenCV coefficients
    optional_current_feature_cache: Optional[Mapping] = None
    scene_tokens: Optional[Tuple[str, ...]] = None

    def validate(self, allowed_cameras):
        b, v, c, h, w = self.current_images.shape
        if tuple(self.camera_names) != tuple(allowed_cameras):
            raise ValueError('Sensor contract mismatch')
        if c != 3 or v != len(self.camera_names) or min(h, w) < 1:
            raise ValueError('Invalid image shape')
        for value, shape in [(self.camera_intrinsics,(b,v,3,3)),
                             (self.camera_extrinsics,(b,v,4,4)),
                             (self.image_transforms,(b,v,3,3)),
                             (self.timestamps,(b,v)), (self.decision_time,(b,))]:
            if tuple(value.shape) != shape or not torch.isfinite(value).all():
                raise ValueError('Invalid calibration or timestamp')
        if (self.timestamps > self.decision_time[:, None]).any():
            raise ValueError('Future observations forbidden')

@dataclass
class WorldTargets:
    current_boxes: torch.Tensor  # N,8: xyz,length,width,height,sin(yaw),cos(yaw)
    current_classes: torch.Tensor
    track_ids: Tuple[str, ...]
    future_xy_in_ego_t0: torch.Tensor  # N,T,2, metres
    future_valid_mask: torch.Tensor  # N,T
    current_supervision_mask: torch.Tensor  # N
    annotation_valid_mask: torch.Tensor  # scalar: missing annotation != empty scene
    box_valid_mask: torch.Tensor  # N,8
    # No-object supervision is valid only for predicted centres inside this region.
    supervision_bounds: torch.Tensor  # xmin,ymin,xmax,ymax
    overflow: int = 0
    supervision_grid: Optional[torch.Tensor] = None  # X,Y calibrated FOV support
    supervision_resolution: float = 1.

@dataclass
class WorldMemory:
    scene_memory: torch.Tensor
    agent_memory: torch.Tensor
    spatial_coordinates: torch.Tensor
    observation_support: torch.Tensor  # geometric FOV, NOT visibility/confidence
    provider_metadata: Mapping
