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
    fm_total_weight: float = 1.
    fm_endpoint_velocity_weight: float = 0.
    fm_endpoint_acceleration_weight: float = 0.
    fm_braking_scene_weight: float = 0.
    fm_braking_speed_drop_mps: float = 2.
    fm_warmup_updates: int = 0
    fm_normalize_scene_weights: bool = False
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
        for name in ('fm_endpoint_velocity_weight', 'fm_endpoint_acceleration_weight', 'fm_braking_scene_weight',
                     'position_weight', 'heading_weight', 'velocity_weight', 'acceleration_weight',
                     'risk_overshoot_weight', 'risk_margin_m'):
            value = getattr(cfg, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f'Invalid trajectory objective {name}')
        for name in ('fm_braking_speed_drop_mps', 'fm_total_weight', 'longitudinal_tolerance_m', 'lateral_tolerance_m', 'heading_tolerance_rad',
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
        if isinstance(cfg.fm_warmup_updates, bool) or not isinstance(cfg.fm_warmup_updates, int) or cfg.fm_warmup_updates < 0:
            raise ValueError('fm_warmup_updates must be a nonnegative integer')
        if not isinstance(cfg.fm_normalize_scene_weights, bool):
            raise ValueError('fm_normalize_scene_weights must be boolean')
        return cfg

    def effective_fm(self, completed_updates):
        """Ramp from the original FM objective using this run's update counter."""
        ramp = min(1., (completed_updates + 1) / self.fm_warmup_updates) if self.fm_warmup_updates else 1.
        absolute = tuple(1. + ramp * (w - 1.) for w in self.absolute_fm_channel_weights)
        # Keep the historical computation when warmup is disabled.
        channels = absolute if self.fm_warmup_updates else self.fm_channel_weights
        total = sum(absolute) / 4 if self.fm_warmup_updates else self.fm_total_weight
        return dict(channel_weights=channels, total_weight=total, ramp=ramp,
                    velocity_weight=ramp*self.fm_endpoint_velocity_weight,
                    acceleration_weight=ramp*self.fm_endpoint_acceleration_weight,
                    braking_scene_weight=ramp*self.fm_braking_scene_weight)

    @property
    def has_fm_endpoint_objective(self):
        return any(value > 0 for value in (self.fm_endpoint_velocity_weight,
                   self.fm_endpoint_acceleration_weight, self.fm_braking_scene_weight))

    @property
    def absolute_fm_channel_weights(self):
        mean = sum(self.fm_channel_weights) / len(self.fm_channel_weights)
        return tuple(self.fm_total_weight * value / mean for value in self.fm_channel_weights)

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


def weighted_flow_mse(prediction, target, channel_weights=None, scene_weights=None):
    """Preserve the exact legacy operation for equal/default weights.

    Weights refer to encoded [x, y, sin(yaw), cos(yaw)], NOT Frenet axes.
    Dividing by their mean preserves expected scale for isotropic encoded errors.
    """
    if scene_weights is None and (channel_weights is None or all(float(v) == 1. for v in channel_weights)):
        return ((prediction - target) ** 2).mean()
    weights = prediction.new_tensor(channel_weights if channel_weights is not None else [1.] * prediction.shape[-1])
    if weights.shape != (prediction.shape[-1],) or not torch.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError('Invalid FM channel weights')
    error = prediction.float() - target.float()
    weights = weights.float() / weights.float().mean()
    values = error.square() * weights
    if scene_weights is None:
        return values.mean()
    if scene_weights.shape != (len(prediction),) or not torch.isfinite(scene_weights).all() or (scene_weights <= 0).any():
        raise ValueError('Invalid FM scene weights')
    # Do not divide by the weight mean: non-braking scenes keep their full loss.
    return (values.mean(dim=(1, 2)) * scene_weights).mean()


def flow_endpoint_terms(prediction, target_flow, actions, times, *, channel_weights=None,
                        braking_scene_weight=0., braking_speed_drop_mps=2., timestep_s=.5,
                        scene_weight_normalizer=1.):
    """Physical derivatives of fitting error in a linear-bridge FM endpoint estimate.

    Flow time t and vehicle time tau are different. For the original bridge,
    x_t=(1-t)*noise+t*GT and v*=GT-noise. The estimated endpoint error is
    (1-t)*(v_pred-v*). Vehicle velocity error is its difference over 0.5 s;
    this is not the FM vector field itself or a final ten-step rollout score.
    """
    if prediction.shape != actions.shape or target_flow.shape != actions.shape or actions.shape[1:] != (8,4) or times.shape != (len(actions),):
        raise ValueError('FM endpoint objective expects repeated N x 8 x 4 and N flow times')
    if timestep_s != .5:
        raise ValueError('Vehicle timestamps must retain their original 0.5-second spacing')
    error = (prediction.float() - target_flow.detach().float()) * (1 - times.detach().float())[:,None,None]
    xy_error = error[...,:2] * error.new_tensor([X_STD, Y_STD])
    origin = xy_error.new_zeros((len(xy_error),1,2))
    velocity_error = torch.diff(torch.cat((origin,xy_error),1),dim=1) / timestep_s
    acceleration_error = torch.diff(velocity_error,dim=1) / timestep_s
    scene_weights = None
    fraction = error.new_zeros(())
    if braking_scene_weight > 0:
        braking = braking_scenes(actions, braking_speed_drop_mps)
        scene_weights = 1 + braking_scene_weight * braking.float()
        if not math.isfinite(scene_weight_normalizer) or scene_weight_normalizer < 1:
            raise ValueError('Invalid global scene weight normalizer')
        scene_weights = scene_weights / scene_weight_normalizer
        fraction = braking.float().mean()
    reduce = lambda value: value.mean() if scene_weights is None else (value*scene_weights).mean()
    return dict(fm=weighted_flow_mse(prediction,target_flow,channel_weights,scene_weights),
                channel_mse=(prediction.detach().float()-target_flow.detach().float()).square().mean(dim=(0,1)),
                endpoint_velocity=reduce(velocity_error.square().sum(-1).mean(-1)),
                endpoint_acceleration=reduce(acceleration_error.square().sum(-1).mean(-1)),
                braking_fraction=fraction)


def braking_scenes(actions, speed_drop_mps=2.):
    gt = physical_xy(actions.detach().float())
    origin = gt.new_zeros((len(gt), 1, 2))
    speed = torch.linalg.vector_norm(torch.diff(torch.cat((origin, gt), 1), dim=1) / .5, dim=-1)
    return speed[:, :2].mean(-1) - speed[:, -2:].mean(-1) > speed_drop_mps


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
