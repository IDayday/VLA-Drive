"""V2 targets: actual 0.5/1.5/4.0 s images; optional real 5 s long target."""
import numpy as np
import torch
from scipy.interpolate import CubicSpline
from . import LONG_TARGET_VERSION, CACHE_SCHEMA
from .motion import GTLogMotionBuilder, HORIZONS, OFFSETS
from .normalizers import wrap_angle, unwrap_heading
from ..layers.world_model.future_image_io import encode_path_tensor


def long_target(poses, timestamps, valid, strict=False, timestamp_tolerance=.02, log_ids=None):
    poses, timestamps, valid = torch.as_tensor(poses), torch.as_tensor(timestamps), torch.as_tensor(valid).bool()
    if poses.ndim!=2 or poses.shape[-1]!=3 or timestamps.shape!=(len(poses),) or valid.shape!=(len(poses),):
        raise ValueError('Long target requires matching [T,3] poses, [T] timestamps and validity')
    ok = len(poses) >= 11 and bool(valid[:11].all())
    ok = ok and bool(torch.isfinite(poses[:11]).all() and torch.isfinite(timestamps[:11]).all())
    ok = ok and bool((timestamps[1:11]>timestamps[:10]).all())
    ok = ok and bool(torch.allclose(timestamps[:11].double(), torch.arange(11).double()*.5, atol=timestamp_tolerance, rtol=0))
    if log_ids is not None:
        ok = ok and len(log_ids)>=11 and len(set(log_ids[:11]))==1
    if not ok:
        if strict:
            raise ValueError("Long-2 requires actual same-log poses through 5 seconds")
        return torch.zeros(8, 3), torch.tensor(False)
    # V1 d9ca73 progressive long-2 mapping. Ten FUTURE knots, never an added t0.
    j = np.arange(8,dtype=np.float64)
    q_index = j + np.cumsum((j+1)*(2*2/(8*9)))
    actual = timestamps[1:11].double().cpu().numpy()
    query = np.interp(q_index,np.arange(10),actual)
    values = poses[1:11].double().cpu().numpy().copy()
    values[:,2] = np.unwrap(values[:,2])
    output = CubicSpline(actual,values,bc_type='not-a-knot',extrapolate=False)(query)
    if not np.isfinite(output).all():
        if strict: raise ValueError('Non-finite long-2 spline; extrapolation is prohibited')
        return torch.zeros(8,3),torch.tensor(False)
    result = torch.tensor(output, dtype=torch.float32)
    result[:, 2] = wrap_angle(result[:, 2])
    return result, torch.tensor(True)


class V2TrajectoryTargetBuilder:
    def __init__(self, source_vector_frame, strict_long=False):
        self.motion_builder = GTLogMotionBuilder(source_vector_frame)
        self.strict_long = strict_long

    def get_unique_name(self):
        return CACHE_SCHEMA

    def compute_targets(self, scene):
        current = scene.scene_metadata.num_history_frames - 1
        frames = scene.frames[current:current+11]
        # Reading numeric ego records does not request any extra images.
        poses = np.stack([f.ego_status.ego_pose for f in frames])
        velocity = np.stack([f.ego_status.ego_velocity for f in frames])
        acceleration = np.stack([f.ego_status.ego_acceleration for f in frames])
        timestamps = np.array([f.timestamp for f in frames], dtype=np.float64) / 1e6
        motion = self.motion_builder.build(poses, velocity, acceleration, timestamps, np.ones(len(frames), bool))
        t = motion["timestamps"]
        physical = motion["motion_sequence"]
        local_poses = torch.cat((physical[:, :2], torch.atan2(physical[:, 2], physical[:, 3])[:, None]), -1)
        trajectory = torch.zeros(8, 3)
        trajectory_valid = torch.zeros(8, dtype=torch.bool)
        for i in range(min(8, len(frames)-1)):
            trajectory[i] = local_poses[i+1]
            trajectory_valid[i] = motion["valid_mask"][i+1] & (abs(float(t[i+1]) - (i+1)*.5) <= .02)
        paths, lengths, future_valid = torch.zeros(3, 1024, dtype=torch.uint8), torch.zeros(3, dtype=torch.long), torch.zeros(3, dtype=torch.bool)
        for h, (offset, seconds) in enumerate(zip(OFFSETS, HORIZONS)):
            if offset < len(frames) and abs(float(t[offset])-seconds) <= .02:
                path = frames[offset].cameras.cam_f0.image
                if path is not None:
                    paths[h], lengths[h] = encode_path_tensor(str(path))
                    future_valid[h] = True
        logged = torch.zeros(8, 8)
        count = min(8, len(frames)-1)
        logged[:count] = physical[1:count+1]
        actual_times = torch.zeros(8)
        actual_times[:count] = t[1:count+1]
        long, long_valid = long_target(local_poses, t, motion["valid_mask"], self.strict_long)
        return dict(long_target_version=LONG_TARGET_VERSION,trajectory=trajectory, trajectory_valid=trajectory_valid,
                    trajectory_long=long, trajectory_long_valid=long_valid,
                    future_image_paths=paths, future_image_path_lengths=lengths,
                    future_valid_mask=future_valid, motion_sequence=logged,
                    motion_timestamps=actual_times, motion_valid=trajectory_valid.clone(),
                    motion_source=motion['source'],
                    token=scene.scene_metadata.initial_token)
