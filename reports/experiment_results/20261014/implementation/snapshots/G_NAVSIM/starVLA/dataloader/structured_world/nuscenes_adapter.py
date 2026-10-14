"""VAD keyframe protocol in an explicitly rotated current-LiDAR reference.

The internal axes are forward/left/up = source LiDAR y/-x/z. The origin is
the current LiDAR origin, not NAVSIM's rear axle. Sensor tilt is retained in
the rigid calibration, and the map uses the disclosed planar yaw projection.
The inverse rotation is exact; no default VAD axes enter the VLA/FGTR.
"""
import numpy as np
import torch
from third_party.vad.ported.planning_pose import get_global_sensor_pose
from starVLA.dataloader.foresight_dataset import encode_ego
from .adapters import nuscenes_scene, navigation_text, pose_matrix


INTERNAL_TO_LIDAR = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])


def to_internal(xy):
    value = np.asarray(xy)
    return np.stack((value[..., 1], -value[..., 0]), -1)


def to_lidar(xy):
    value = np.asarray(xy)
    return np.stack((-value[..., 1], value[..., 0]), -1)


def planning_metadata(nusc, sample):
    """Use the original mature VAD global-sensor pose function for targets."""
    initial = get_global_sensor_pose(sample, nusc)
    inverse = np.linalg.inv(initial)
    previous = nusc.get('sample', sample['prev']) if sample['prev'] else None
    if previous is None:
        raise ValueError('No legal past pose')
    past = inverse@get_global_sensor_pose(previous, nusc)
    past_heading = np.arctan2(past[1, 0], past[0, 0])
    past_sd = nusc.get('sample_data', previous['data']['LIDAR_TOP'])
    current_sd = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    dt = (current_sd['timestamp']-past_sd['timestamp'])/1e6
    if not 0.1 < dt < 1.:
        raise ValueError('Invalid legal past sensor interval')
    state_xy = to_internal(-past[:2, 3])*(.5/dt)
    state = encode_ego([[*state_xy, -past_heading*(.5/dt)]])
    positions, times, yaw, cursor = [], [], [], sample
    for index in range(6):
        if not cursor['next']:
            raise ValueError('Incomplete VAD six-keyframe planning horizon')
        cursor = nusc.get('sample', cursor['next'])
        relative = inverse@get_global_sensor_pose(cursor, nusc)
        positions.append(relative[:2, 3].copy())
        yaw.append(np.arctan2(relative[1, 0], relative[0, 0]))
        sd = nusc.get('sample_data', cursor['data']['LIDAR_TOP'])
        times.append((sd['timestamp']-current_sd['timestamp'])/1e6)
    positions = np.asarray(positions)
    command = 2 if positions[-1, 0] >= 2. else 0 if positions[-1, 0] <= -2. else 1
    physical = np.column_stack((to_internal(positions), yaw)).astype(np.float32)
    return {'ego_physical': physical, 'original_VAD_lidar_xy': positions.astype(np.float32),
            'future_times_s': times, 'state': state, 'navigation': command,
            'internal2global': initial@np.block([[INTERNAL_TO_LIDAR, np.zeros((3, 1))], [np.zeros((1, 3)), np.ones((1, 1))]])}


def adapted_nuscenes_scene(nusc, sample, *, image_size=(256, 448)):
    scene = nuscenes_scene(nusc, sample, image_size=image_size)
    metadata = planning_metadata(nusc, sample)
    # Each mature annotation/point operator receives a real rigid transformation.
    # The old adapter remains untouched while the NAVSIM label build is running.
    current_basis = None
    for frame in scene.frames:
        if frame is None:
            raise ValueError('Missing formal future annotations')
        internal2ego = frame.lidar2ego.copy()
        internal2ego[:3, :3] = internal2ego[:3, :3]@INTERNAL_TO_LIDAR
        ego2internal = np.linalg.inv(internal2ego)
        frame.boxes = frame.boxes.clone()
        # Original mature transform order: inverse translation then rotation.
        frame.boxes.translate(-internal2ego[:3, 3].astype(np.float32))
        frame.boxes.rotate(ego2internal[:3, :3].T.astype(np.float32))
        from .centered_boxes import correct_center_after_rotation
        correct_center_after_rotation(frame.boxes, ego2internal[:3, :3])
        frame.ego2global = frame.ego2global@internal2ego
        frame.lidar2ego = ego2internal@frame.lidar2ego
        if current_basis is None:
            current_basis = ego2internal
    calibration = scene.observations['calibration']
    calibration['sensor2ego'] = torch.from_numpy(current_basis).float()[None]@calibration['sensor2ego']
    scene.observations.update(state=metadata['state'], lang=navigation_text(metadata['navigation']))
    scene.ego_physical = metadata['ego_physical']
    scene.future_valid = np.ones(6, dtype=bool)
    scene.inverse_transform = metadata['internal2global']
    scene.ego_body = dict(length=4.084, width=1.85, rear_axle_to_center=.5)
    scene.protocol.update(origin='current LiDAR origin with forward/left axes',
        inverse_axes='lidar_x=-internal_y; lidar_y=internal_x; lidar_z=internal_z',
        body_geometry='original UniAD fixed 4.084 x 1.85 m rectangle, +0.5 m forward offset',
        ego_state='previous/current LiDAR poses only, displacement at nominal 0.5 s from actual past dt',
        navigation_source='original VAD global-sensor future endpoint +/-2 m, fixed high-level text only',
        nominal_future_times_s=[.5, 1., 1.5, 2., 2.5, 3.], actual_annotation_times_s=metadata['future_times_s'],
        timing_protocol='six successive annotated 2 Hz keyframes, original VAD eligibility; jitter retained and reported',
        map_height_approximation='map planar yaw projection; label height evidence masks unsupported road levels')
    return scene
