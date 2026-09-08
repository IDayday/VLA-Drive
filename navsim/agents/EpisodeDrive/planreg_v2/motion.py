"""Independent logged motion and strictly causal candidate interval kinematics."""
import torch
from torch import nn
from .normalizers import MotionConditionNormalizer

HORIZONS = (.5, 1.5, 4.)
OFFSETS = (1, 3, 8)


class CandidateKinematicsCodec(nn.Module):
    def __init__(self):
        super().__init__()
        self.normalizer = MotionConditionNormalizer()

    def forward(self, physical_trajectory, current_ego_state, timestamps, valid_mask):
        p = physical_trajectory
        if p.ndim != 4 or p.shape[-1] != 3:
            raise ValueError("Physical trajectory must be [B,K,T,3]")
        b, k, t, _ = p.shape
        times = torch.as_tensor(timestamps, device=p.device, dtype=p.dtype)
        times = torch.broadcast_to(times, (b, k, t))
        valid = torch.broadcast_to(valid_mask.to(device=p.device, dtype=torch.bool), (b, k, t))
        dt = times - torch.cat((times.new_zeros(b, k, 1), times[..., :-1]), -1)
        valid = valid & torch.isfinite(p).all(-1) & torch.isfinite(times) & (dt > 0)
        valid = valid.long().cumprod(-1).bool()
        clean = torch.where(valid[..., None], p, 0.)
        prev_p = torch.cat((clean.new_zeros(b, k, 1, 2), clean[..., :-1, :2]), -2)
        velocity = (clean[..., :2] - prev_p) / dt.clamp_min(1e-6)[..., None]
        tv = times - dt / 2
        tv_previous = torch.cat((tv.new_zeros(b, k, 1), tv[..., :-1]), -1)
        # All candidates share one current logged velocity, expressed in anchor axes.
        if current_ego_state.shape != (b, 4):
            raise ValueError("current_ego_state must be [B,4]: vx,vy,ax,ay")
        v0 = current_ego_state[:, None, None, :2].expand(b, k, 1, 2)
        vprev = torch.cat((v0, velocity[..., :-1, :]), -2)
        acceleration = (velocity - vprev) / (tv - tv_previous).clamp_min(1e-6)[..., None]
        motion = torch.cat((clean[..., :2], clean[..., 2:3].sin(), clean[..., 2:3].cos(),
                            velocity, acceleration), -1)
        motion = torch.where(valid[..., None], motion, 0.)
        return dict(motion_sequence=self.normalizer(motion), physical_motion=motion,
                    timestamps=times, valid_mask=valid,
                    velocity_semantics="interval average; effective time is interval midpoint")


class GTLogMotionBuilder:
    """Do not infer vector frames from EgoStatus.in_global_frame (pose flag)."""
    def __init__(self, source_vector_frame):
        if source_vector_frame not in ("ego", "global"):
            raise ValueError("Explicit source_vector_frame=ego|global required")
        self.source_vector_frame = source_vector_frame

    def build(self, poses_global, logged_velocity, logged_acceleration, timestamps, valid_mask):
        p, v, a = [torch.as_tensor(x, dtype=torch.float64) for x in
                   (poses_global, logged_velocity, logged_acceleration)]
        times = torch.as_tensor(timestamps, dtype=torch.float64)
        valid = torch.as_tensor(valid_mask).bool()
        if p.ndim != 2 or p.shape[-1] != 3 or v.shape != a.shape or v.shape != (len(p), 2):
            raise ValueError("Logged poses and vx/vy/ax/ay are required, never substituted by differences")
        anchor = p[0]
        angle = -anchor[2]
        def rotate(x, theta):
            return torch.stack((x[..., 0]*theta.cos()-x[..., 1]*theta.sin(),
                                x[..., 0]*theta.sin()+x[..., 1]*theta.cos()), -1)
        xy = rotate(p[:, :2] - anchor[:2], angle)
        heading = p[:, 2] - anchor[2]
        vector_angle = heading if self.source_vector_frame == "ego" else angle
        v, a = rotate(v, vector_angle), rotate(a, vector_angle)
        motion = torch.cat((xy, heading.sin()[:, None], heading.cos()[:, None], v, a), -1)
        valid = valid & torch.isfinite(motion).all(-1) & torch.isfinite(times)
        if not valid[0] or (times[1:] <= times[:-1]).any():
            raise ValueError("Invalid anchor or non-increasing logged timestamps")
        return dict(motion_sequence=torch.where(valid[:, None], motion, 0.).float(),
                    timestamps=(times-times[0]).float(), valid_mask=valid,
                    source=dict(kind="logged ego_dynamic_state, not finite differences",
                                vector_frame=self.source_vector_frame,
                                output_frame="current rear axle axes", units="m,s,rad"))


class IntervalMotionEncoder(nn.Module):
    def __init__(self, dim=256):
        super().__init__()
        self.point = nn.Sequential(nn.Linear(11, dim), nn.GELU(), nn.Linear(dim, dim))

    def forward(self, motion_sequence, timestamps, valid_mask, query_horizons):
        """Each interval pools *all* its valid points, with absolute/relative time and duration."""
        outputs, coverage = [], []
        left = 0.
        for right in query_horizons:
            right = float(right)
            in_interval = (timestamps > left + 1e-6) & (timestamps <= right + 1e-6)
            selected = in_interval & valid_mask
            endpoint = ((timestamps-right).abs() < 1e-5) & valid_mask
            ok = endpoint.any(-1) & (~in_interval | valid_mask).all(-1)
            extra = torch.stack((timestamps, timestamps-left,
                                 torch.full_like(timestamps, right-left)), -1)
            encoded = self.point(torch.cat((motion_sequence, extra), -1))
            pooled = (encoded * selected[..., None]).sum(-2) / selected.sum(-1).clamp_min(1)[..., None]
            outputs.append(pooled)
            coverage.append(ok)
            left = right
        return torch.stack(outputs, -2), torch.stack(coverage, -1)
