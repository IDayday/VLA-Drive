"""Independent logged motion and strictly causal candidate interval kinematics."""
import torch
from torch import nn
from .normalizers import MotionConditionNormalizer

HORIZONS = (.5, 1.5, 4.)
OFFSETS = (1, 3, 8)


def validate_motion_inputs(timestamps,valid,motion=None):
    """Declared valid values are data, never silently converted to padding."""
    bad=valid & (~torch.isfinite(timestamps) | (timestamps<=0))
    if motion is not None: bad=bad | (valid & ~torch.isfinite(motion).all(-1))
    clean=torch.where(valid,timestamps,0.)
    previous=torch.cat((clean.new_zeros(*clean.shape[:-1],1),clean.cummax(-1).values[...,:-1]),-1)
    bad=bad | (valid & (clean<=previous))
    if bad.any():
        raise ValueError('Invalid valid motion value/time at batch/candidate/time indices '+str(bad.nonzero().tolist()))


def interval_selection(timestamps,valid_mask,query_horizons=HORIZONS,tolerance=.02):
    validate_motion_inputs(timestamps,valid_mask)
    times=torch.where(valid_mask,timestamps,0.)
    selected,coverage=[],[];left=0.
    for right in query_horizons:
        right=float(right)
        interval=(times>(left+tolerance if left else 0.))&(times<=right+tolerance)
        chosen=interval&valid_mask
        endpoint=((times-right).abs()<=tolerance)&valid_mask
        # V2 logged 2Hz contract has complete 1/2/5-point transition intervals.
        count=round((right-left)/.5)
        coverage.append(endpoint.any(-1)&(chosen.sum(-1)==count))
        selected.append(chosen);left=right
    return torch.stack(selected,-2),torch.stack(coverage,-1)


class CandidateKinematicsCodec(nn.Module):
    def __init__(self,scales=(30.,10.,1.,1.,15.,15.,8.,8.)):
        super().__init__()
        self.normalizer = MotionConditionNormalizer(scales)

    def forward(self, physical_trajectory, current_ego_state, timestamps, valid_mask):
        p = physical_trajectory
        if p.ndim != 4 or p.shape[-1] != 3:
            raise ValueError("Physical trajectory must be [B,K,T,3]")
        b, k, t, _ = p.shape
        times = torch.as_tensor(timestamps, device=p.device, dtype=p.dtype)
        if times.ndim == 2: times = times[:,None]
        times = torch.broadcast_to(times, (b, k, t))
        valid_mask = valid_mask.to(device=p.device,dtype=torch.bool)
        if valid_mask.ndim == 2: valid_mask = valid_mask[:,None]
        valid = torch.broadcast_to(valid_mask, (b, k, t))
        validate_motion_inputs(times,valid,p)
        valid = valid.long().cumprod(-1).bool()
        times=torch.where(valid,times,0.)
        dt = times - torch.cat((times.new_zeros(b, k, 1), times[..., :-1]), -1)
        dt=torch.where(valid,dt,1.)
        clean = torch.where(valid[..., None], p, 0.)
        prev_p = torch.cat((clean.new_zeros(b, k, 1, 2), clean[..., :-1, :2]), -2)
        velocity = (clean[..., :2] - prev_p) / dt.clamp_min(1e-6)[..., None]
        tv = times - dt / 2
        tv_previous = torch.cat((tv.new_zeros(b, k, 1), tv[..., :-1]), -1)
        # All candidates share one current logged velocity, expressed in anchor axes.
        if current_ego_state.shape != (b, 4) or not torch.isfinite(current_ego_state).all():
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
        if times.shape!=(len(p),) or valid.shape!=(len(p),) or not valid[0] or not torch.isfinite(p[0]).all() or not torch.isfinite(times[0]):
            raise ValueError('Invalid logged current anchor')
        validate_motion_inputs(times[1:]-times[0],valid[1:],torch.cat((p[1:],v[1:],a[1:]),-1))
        p=torch.where(valid[:,None],p,p[0]);v=torch.where(valid[:,None],v,0.);a=torch.where(valid[:,None],a,0.)
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
        if not torch.isfinite(motion[valid]).all():raise ValueError('Nonfinite valid logged motion')
        return dict(motion_sequence=torch.where(valid[:, None], motion, 0.).float(),
                    timestamps=torch.where(valid,times-times[0],0.).float(), valid_mask=valid,
                    source=dict(kind="logged ego_dynamic_state, not finite differences",
                                vector_frame=self.source_vector_frame,
                                output_frame="current rear axle axes", units="m,s,rad"))


class IntervalMotionEncoder(nn.Module):
    def __init__(self, dim=256, timestamp_tolerance=.02):
        super().__init__()
        self.timestamp_tolerance = timestamp_tolerance
        self.point = nn.Sequential(nn.Linear(11, dim), nn.GELU(), nn.Linear(dim, dim))

    def forward(self, motion_sequence, timestamps, valid_mask, query_horizons):
        """Each interval pools *all* its valid points, with absolute/relative time and duration."""
        validate_motion_inputs(timestamps,valid_mask,motion_sequence)
        selections,coverage=interval_selection(timestamps,valid_mask,query_horizons,self.timestamp_tolerance)
        timestamps=torch.where(valid_mask,timestamps,0.)
        motion_sequence=torch.where(valid_mask[...,None],motion_sequence,0.)
        outputs = []
        left = 0.
        for index,right in enumerate(query_horizons):
            right = float(right)
            selected = selections[...,index,:]
            extra = torch.stack((timestamps, timestamps-left,
                                 torch.full_like(timestamps, right-left)), -1)
            inputs=torch.cat((motion_sequence, extra), -1)
            inputs=torch.where(selected[...,None],inputs,0.)
            encoded = self.point(inputs)
            pooled = (encoded * selected[..., None]).sum(-2) / selected.sum(-1).clamp_min(1)[..., None]
            outputs.append(pooled)
            left = right
        return torch.stack(outputs, -2), coverage
