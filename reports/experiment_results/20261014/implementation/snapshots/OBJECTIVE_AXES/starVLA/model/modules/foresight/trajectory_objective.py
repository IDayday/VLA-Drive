"""Opt-in ego objectives in physical units, at the eight original timestamps.

These are supervised fitting losses, not differentiable official PDM scores.
Future labels enter here only, after current-observation encoding.
"""
from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F

from starVLA.dataloader.foresight_dataset import X_MEAN, X_STD, Y_MEAN, Y_STD


@dataclass(frozen=True)
class TrajectoryObjectiveConfig:
    fm_channel_weights: tuple = (1., 1., 1., 1.)
    position_weight: float = 0.
    heading_weight: float = 0.
    velocity_weight: float = 0.
    acceleration_weight: float = 0.
    risk_overshoot_weight: float = 0.
    longitudinal_tolerance_m: float = .5
    lateral_tolerance_m: float = .25
    heading_tolerance_rad: float = math.pi / 60  # 3 degrees, a scale not a guarantee
    velocity_tolerance_mps: float = 1.
    acceleration_tolerance_mps2: float = 1.
    risk_margin_m: float = .25
    timestep_s: float = .5
    horizon_weights: tuple = (1.,) * 8
    warmup_updates: int = 1000
    rollout_every: int = 1
    checkpoint_rollout: bool = True

    @classmethod
    def from_mapping(cls, options=None):
        cfg = cls(**dict(options or {}))
        for name, length in (('fm_channel_weights', 4), ('horizon_weights', 8)):
            values = getattr(cfg, name)
            if len(values) != length or any(not math.isfinite(float(v)) or float(v) <= 0 for v in values):
                raise ValueError(f'{name} requires {length} finite positive values')
        for name in ('position_weight', 'heading_weight', 'velocity_weight', 'acceleration_weight',
                     'risk_overshoot_weight', 'risk_margin_m'):
            value = getattr(cfg, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f'Invalid trajectory objective {name}')
        for name in ('longitudinal_tolerance_m', 'lateral_tolerance_m', 'heading_tolerance_rad',
                     'velocity_tolerance_mps', 'acceleration_tolerance_mps2'):
            value = getattr(cfg, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f'Invalid trajectory objective {name}')
        if cfg.timestep_s != .5:
            raise ValueError('This label contract has fixed 0.5-second timestamps')
        for name in ('warmup_updates', 'rollout_every'):
            value = getattr(cfg, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        if not isinstance(cfg.checkpoint_rollout, bool):
            raise ValueError('checkpoint_rollout must be boolean')
        return cfg

    @property
    def has_rollout_loss(self):
        return any(getattr(self, task + '_weight') > 0 for task in
                   ('position', 'heading', 'velocity', 'acceleration', 'risk_overshoot'))

    def active(self, completed_updates):
        # All microbatches/ranks use the same optimizer-update counter, including resume.
        return self.has_rollout_loss and (completed_updates + 1) % self.rollout_every == 0

    def effective_weights(self, completed_updates):
        ramp = min(1., (completed_updates + 1) / self.warmup_updates) if self.active(completed_updates) else 0.
        return {task: getattr(self, task + '_weight') * ramp for task in
                ('position', 'heading', 'velocity', 'acceleration', 'risk_overshoot')}


def weighted_flow_mse(prediction, target, channel_weights=None):
    """Preserve the exact legacy operation for equal/default weights.

    Weights refer to encoded [x, y, sin(yaw), cos(yaw)], NOT Frenet axes.
    Dividing by their mean preserves expected scale for isotropic encoded errors.
    """
    if channel_weights is None or all(float(v) == 1. for v in channel_weights):
        return ((prediction - target) ** 2).mean()
    weights = prediction.new_tensor(channel_weights)
    if weights.shape != (prediction.shape[-1],) or not torch.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError('Invalid FM channel weights')
    error = prediction.float() - target.float()
    weights = weights.float() / weights.float().mean()
    return (error.square() * weights).mean()


def physical_xy(encoded):
    return encoded[..., :2] * encoded.new_tensor([X_STD, Y_STD]) + encoded.new_tensor([X_MEAN, Y_MEAN])


def trajectory_loss_terms(prediction, target, cfg, *, risk_mask=None):
    """Return one value per scene per task, plus physical diagnostics.

    Every correspondence retains physical time. GT yaw defines tangent/normal;
    there is no nearest-point matching that could hide longitudinal lateness.
    Derivatives are interval-average vehicle motion, not FM vector-field velocity.
    """
    if prediction.shape != target.shape or target.ndim != 3 or target.shape[1:] != (8, 4):
        raise ValueError('Trajectory objective expects B x 8 x 4 encoded poses')
    if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
        raise ValueError('Nonfinite trajectory objective input')
    prediction, target = prediction.float(), target.detach().float()
    p, g = physical_xy(prediction), physical_xy(target)
    # [sin, cos] -> unit [cos, sin]. atan2 is unnecessary and singular at (0, 0).
    gt_heading = F.normalize(target[..., [3, 2]], dim=-1, eps=1e-6)
    if (target[..., 2:].norm(dim=-1) < .5).any():
        raise ValueError('GT heading must be a valid encoded direction')
    gt_normal = torch.stack((-gt_heading[..., 1], gt_heading[..., 0]), -1)
    error = p - g
    along = (error * gt_heading).sum(-1)
    lateral = (error * gt_normal).sum(-1)
    weights = prediction.new_tensor(cfg.horizon_weights)
    weights = weights / weights.mean()
    reduce = lambda value: (value * weights).mean(-1)
    position = reduce((along / cfg.longitudinal_tolerance_m).square() +
                      (lateral / cfg.lateral_tolerance_m).square())
    pred_heading = F.normalize(prediction[..., [3, 2]], dim=-1, eps=1e-6)
    # Unit-vector distance is 2*(1-cos(delta_yaw)) away from the zero-vector corner.
    heading = reduce((pred_heading - gt_heading).square().sum(-1) / cfg.heading_tolerance_rad ** 2)
    # Both current poses are exactly (0, 0) in the common current-ego frame.
    origin = p.new_zeros((len(p), 1, 2))
    vp = torch.diff(torch.cat((origin, p), 1), dim=1) / cfg.timestep_s
    vg = torch.diff(torch.cat((origin, g), 1), dim=1) / cfg.timestep_s
    dv = vp - vg
    velocity = reduce(dv.square().sum(-1) / cfg.velocity_tolerance_mps ** 2)
    # Seven interval-to-interval changes; no invented current velocity/acceleration.
    da = torch.diff(dv, dim=1) / cfg.timestep_s
    acceleration = (da.square().sum(-1) * weights[1:]).sum(-1) / weights[1:].sum()
    acceleration = acceleration / cfg.acceleration_tolerance_mps2 ** 2
    terms = dict(position=position, heading=heading, velocity=velocity, acceleration=acceleration)
    if cfg.risk_overshoot_weight > 0:
        if risk_mask is None:
            raise ValueError('risk_overshoot_weight requires explicit ego_risk_mask labels; no inferred blanket risk')
        if risk_mask.shape != along.shape or risk_mask.dtype != torch.bool:
            raise ValueError('ego_risk_mask must be a B x 8 boolean time-aligned mask')
        mask = risk_mask.to(device=p.device)
        # Average across ALL eight timestamps/scenes, so rare risk windows do not
        # get implicitly oversampled and microbatch/rank normalization is exact.
        terms['risk_overshoot'] = reduce(
            ((along - cfg.risk_margin_m).clamp_min(0) / cfg.longitudinal_tolerance_m).square() * mask)
    diagnostics = {
        'along_squared_m': reduce(along.square()),
        'lateral_squared_m': reduce(lateral.square()),
        'interval_velocity_squared_mps': reduce(dv.square().sum(-1)),
    }
    return terms, diagnostics


def scene_mean(values, *, global_scene_count=None, world_size=1):
    """Match existing DDP/ZeRO mean gradients over all optimizer microbatches."""
    if global_scene_count is None:
        return values.mean()
    if global_scene_count <= 0 or world_size < 1:
        raise ValueError('Invalid global ego scene normalization')
    return values.sum() * world_size / global_scene_count
