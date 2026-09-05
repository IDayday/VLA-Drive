"""Training-side logged rear-axle frame alignment; never candidate coordinates."""
import numpy as np
import torch
from .task_future_training import INPUT_SCHEMA


def build_logged_future_pose_metadata(scene):
    current=int(scene.scene_metadata.num_history_frames)-1
    origin=np.asarray(scene.frames[current].ego_status.ego_pose,dtype=np.float64)
    pose=np.zeros((3,3),dtype=np.float32)
    valid=np.zeros(3,dtype=bool)
    c,s=np.cos(origin[2]),np.sin(origin[2])
    for h,offset in enumerate((1,3,6)):
        if current+offset >= len(scene.frames):
            continue
        future=np.asarray(scene.frames[current+offset].ego_status.ego_pose,dtype=np.float64)
        delta=future[:2]-origin[:2]
        pose[h,:2]=delta @ np.array([[c,-s],[s,c]])
        pose[h,2]=np.arctan2(np.sin(future[2]-origin[2]),np.cos(future[2]-origin[2]))
        valid[h]=np.isfinite(pose[h]).all()
    return dict(logged_future_poses=torch.from_numpy(pose),
                logged_future_pose_valid=torch.from_numpy(valid),
                physical_map_name=scene.scene_metadata.map_name,
                task_future_input_schema=INPUT_SCHEMA)
