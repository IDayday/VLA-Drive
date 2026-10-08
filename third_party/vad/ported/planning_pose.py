# Copyright (c) hustvl/VAD. Licensed under Apache-2.0.
# Extracted unchanged from commit 1688c4b1c3a9e2e7873ca9700ff8058170c0e3c8.
import numpy as np
from pyquaternion import Quaternion
from nuscenes.utils.geometry_utils import transform_matrix

def get_global_sensor_pose(rec, nusc, inverse=False):
    lidar_sample_data = nusc.get('sample_data', rec['data']['LIDAR_TOP'])

    sd_ep = nusc.get("ego_pose", lidar_sample_data["ego_pose_token"])
    sd_cs = nusc.get("calibrated_sensor", lidar_sample_data["calibrated_sensor_token"])
    if inverse is False:
        global_from_ego = transform_matrix(sd_ep["translation"], Quaternion(sd_ep["rotation"]), inverse=False)
        ego_from_sensor = transform_matrix(sd_cs["translation"], Quaternion(sd_cs["rotation"]), inverse=False)
        pose = global_from_ego.dot(ego_from_sensor)
        # translation equivalent writing
        # pose_translation = np.array(sd_cs["translation"])
        # rot_mat = Quaternion(sd_ep['rotation']).rotation_matrix
        # pose_translation = np.dot(rot_mat, pose_translation)
        # # pose_translation = pose[:3, 3]
        # pose_translation = pose_translation + np.array(sd_ep["translation"])
    else:
        sensor_from_ego = transform_matrix(sd_cs["translation"], Quaternion(sd_cs["rotation"]), inverse=True)
        ego_from_global = transform_matrix(sd_ep["translation"], Quaternion(sd_ep["rotation"]), inverse=True)
        pose = sensor_from_ego.dot(ego_from_global)
    return pose
