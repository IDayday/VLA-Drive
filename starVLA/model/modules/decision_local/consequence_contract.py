"""Separate current inputs, model-generated queries and loss-only targets."""
from dataclasses import dataclass
import torch

CHANNELS = ('road_margin_m', 'current_clearance_m', 'current_overlap',
            'future_clearance_m', 'future_overlap')
CONTINUOUS = (0, 1, 3)
DISCRETE = (2, 4)


@dataclass(frozen=True)
class ModelInputs:
    H_A: torch.Tensor
    W: torch.Tensor
    q0_encoded: torch.Tensor
    q0: torch.Tensor
    state: torch.Tensor
    body: torch.Tensor

    @classmethod
    def from_mapping(cls, values):
        allowed = set(cls.__dataclass_fields__)
        if set(values) != allowed:
            raise ValueError('camera_only ModelInputs rejects missing/extra fields, including GT labels')
        return cls(**values)


@dataclass(frozen=True)
class PredictionQueries:
    poses: torch.Tensor                 # B,7,T,3; actual physical ego(t0)
    times: torch.Tensor                 # T real physical times, NEVER phi(t)
    in_grid: torch.Tensor               # B,7,T; deterministic, no label validity
    kinematic_valid: torch.Tensor       # B,7; deterministic train-calibrated rule


@dataclass(frozen=True)
class TrainingTargets:
    relations: torch.Tensor             # B,7,T,5; no inputs are stored here
    valid: torch.Tensor
    ego: torch.Tensor
    future_valid: torch.Tensor


@dataclass(frozen=True)
class OracleRelations:
    """Accepted only by the explicitly privileged information-probe entry."""
    values: torch.Tensor
    common_valid: torch.Tensor
    source: str

    def __post_init__(self):
        if self.source not in ('CURRENT', 'CV', 'FUTURE', 'MASK_ONLY'):
            raise ValueError('Unknown privileged information source')


def query_features(inputs, queries):
    """No learned candidate ID; equal physical queries give equal features."""
    poses = queries.poses.float()
    B, K, T, _ = poses.shape
    if T not in (6, 8) or queries.times.shape != (T,) or inputs.state.shape != (B, 4):
        raise ValueError('Registered horizon/state shape required')
    time = queries.times.float()[None, None, :, None].expand(B, K, T, 1)/4.
    body = inputs.body.float()[:, None, None, :].expand(B, K, T, 3)/5.
    state = inputs.state.float()[:, None, None, :].expand(B, K, T, 4)
    return torch.cat((poses[..., :2]/40., poses[..., 2:3].sin(), poses[..., 2:3].cos(),
                      time, torch.sin(torch.pi*time), torch.cos(torch.pi*time), body, state), -1)


def predicted_relations(raw):
    return torch.stack((raw[..., 0], raw[..., 1], raw[..., 2].sigmoid(),
                        raw[..., 3], raw[..., 4].sigmoid()), -1)
