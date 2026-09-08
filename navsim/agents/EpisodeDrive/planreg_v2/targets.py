"""V2 targets: actual 0.5/1.5/4.0 s images; optional real 5 s long target."""
import numpy as np
import torch
from .motion import GTLogMotionBuilder, HORIZONS, OFFSETS
from .normalizers import wrap_angle, unwrap_heading
from ..layers.world_model.future_image_io import encode_path_tensor


def long_target(poses, timestamps, valid, strict=False):
    poses, timestamps, valid = torch.as_tensor(poses), torch.as_tensor(timestamps), torch.as_tensor(valid).bool()
    ok = len(poses) >= 11 and bool(valid[:11].all())
    ok = ok and bool(torch.allclose(timestamps[:11].float(), torch.arange(11)*.5, atol=.02, rtol=0))
    if not ok:
        if strict:
            raise ValueError("Long-2 requires actual same-log poses through 5 seconds")
        return torch.zeros(8, 3), torch.tensor(False)
    # Explicit retiming: sample the true [0,5s] path into eight output slots.
    query = np.arange(1, 9) * 5. / 8.
    unwrapped = unwrap_heading(poses[:11, 2]).numpy()
    output = np.stack([np.interp(query, timestamps[:11], poses[:11, i]) for i in (0, 1)] +
                      [np.interp(query, timestamps[:11], unwrapped)], -1)
    result = torch.tensor(output, dtype=torch.float32)
    result[:, 2] = wrap_angle(result[:, 2])
    return result, torch.tensor(True)


class V2TrajectoryTargetBuilder:
    def __init__(self, source_vector_frame, strict_long=False):
        self.motion_builder = GTLogMotionBuilder(source_vector_frame)
        self.strict_long = strict_long

    def get_unique_name(self):
        return "trajectory_target_planreg_v2_0138_logmotion_v1"

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
        return dict(trajectory=trajectory, trajectory_valid=trajectory_valid,
                    trajectory_long=long, trajectory_long_valid=long_valid,
                    future_image_paths=paths, future_image_path_lengths=lengths,
                    future_valid_mask=future_valid, motion_sequence=logged,
                    motion_timestamps=actual_times, motion_valid=trajectory_valid.clone(),
                    token=scene.scene_metadata.initial_token)
