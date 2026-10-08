"""Both datasets use ego(t0): x forward, y left, z up, positive yaw CCW."""
from dataclasses import dataclass
from pathlib import Path
import json
import pickle
import numpy as np
import torch
from pyquaternion import Quaternion
from nuscenes.utils.geometry_utils import transform_matrix
from third_party.mmdetection3d.ported.bbox.lidar_box3d import LiDARInstance3DBoxes
from starVLA.dataloader.foresight_dataset import encode_ego
from .cameras import NAVSIM_CAMERAS, NUSCENES_CAMERAS, camera_inputs


def pose_matrix(translation, rotation, *, planar=False):
    q = Quaternion(rotation)
    if planar:
        # Canonical NAVSIM Scene uses pyquaternion's yaw_pitch_roll[0].
        # Projected matrix yaw differs slightly when pitch/roll are nonzero.
        yaw = q.yaw_pitch_roll[0]
        q = Quaternion(axis=[0, 0, 1], angle=yaw)
    return transform_matrix(np.asarray(translation), q, inverse=False)


@dataclass
class FrameAnnotation:
    boxes: object
    labels: np.ndarray
    track_tokens: list
    visibility: np.ndarray
    ego2global: np.ndarray
    timestamp_us: int
    annotations_complete: bool
    lidar_path: str
    lidar2ego: np.ndarray


@dataclass
class SceneRecord:
    token: str
    log: str
    dataset: str
    observations: dict
    ego_physical: np.ndarray
    future_valid: np.ndarray
    frames: list
    map_name: str
    inverse_transform: np.ndarray
    ego_body: dict
    protocol: dict


CLASS_NAMES = ('car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
               'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone')
NAV_CLASSES = {'vehicle': 'car', 'bicycle': 'bicycle', 'pedestrian': 'pedestrian',
               'traffic_cone': 'traffic_cone', 'barrier': 'barrier',
               'generic_object': 'barrier'}


def box_container(center_boxes):
    # mmdet3d v0.17 LiDAR box yaw is clockwise, whereas the internal contract is
    # CCW. Its mature corners/rotation operators stay unchanged.
    boxes = np.asarray(center_boxes, dtype=np.float32).reshape(-1, 7).copy()
    boxes[:, 6] *= -1.
    return LiDARInstance3DBoxes(boxes, origin=(.5, .5, .5))


def transform_boxes_to_ego(boxes, lidar2ego):
    boxes = boxes.clone()
    boxes.rotate(np.asarray(lidar2ego[:3, :3].T, dtype=np.float32))
    boxes.translate(np.asarray(lidar2ego[:3, 3], dtype=np.float32))
    from .centered_boxes import correct_center_after_rotation
    correct_center_after_rotation(boxes, lidar2ego[:3, :3])
    return boxes


def navsim_scene(current_record, *, log_root='/mnt/navsim/trainval_navsim_logs/trainval',
                 sensor_root='/mnt/navsim/trainval_all/trainval_sensor_blobs/trainval', image_size=(256, 448), frames=None):
    if frames is None:
        with (Path(log_root)/(current_record['log']+'.pkl')).open('rb') as handle:
            frames = pickle.load(handle)
    index = next(i for i, value in enumerate(frames) if value['token'] == current_record['token'])
    current = frames[index]
    e2g0 = pose_matrix(current['ego2global_translation'], current['ego2global_rotation'], planar=True)
    current_attitude = np.linalg.inv(e2g0)@pose_matrix(current['ego2global_translation'], current['ego2global_rotation'])
    l2e0 = pose_matrix(current['lidar2ego_translation'], current['lidar2ego_rotation'])
    paths, sensors, intrinsics, distortion = [], [], [], []
    def sensor_path(relative):
        # The full LiDAR export and original image export complement each other.
        # Both contain original bytes; no resized teacher cache substitutes raw RGB.
        for base in (Path(sensor_root), Path('/mnt/navsim/trainval_sensor_blobs/trainval')):
            path = base/relative
            if path.is_file():
                return str(path)
        raise FileNotFoundError(f'Original NAVSIM sensor unavailable in either export: {relative}')
    for camera in NAVSIM_CAMERAS:
        value = current['cams'][camera]
        camera2lidar = np.eye(4)
        camera2lidar[:3, :3], camera2lidar[:3, 3] = value['sensor2lidar_rotation'], value['sensor2lidar_translation']
        paths.append(sensor_path(value['data_path']))
        sensors.append(current_attitude@l2e0@camera2lidar)
        intrinsics.append(value['cam_intrinsic'])
        distortion.append(value.get('distortion'))
    observation = camera_inputs(paths, sensors, intrinsics, distortion, image_size=image_size)
    history = np.asarray(current_record['global_pose_history'])
    previous, now = history[-2:]
    R = e2g0[:2, :2]
    delta = (now[:2]-previous[:2]) @ R
    yaw_delta = (now[2]-previous[2]+np.pi)%(2*np.pi)-np.pi
    observation.update(state=encode_ego([[*delta, yaw_delta]]), token=current['token'],
                       lang=navigation_text(current_record['navigation']))
    annotations, ego, valid = [], [], []
    inverse = np.linalg.inv(e2g0)
    for offset in range(9):
        frame = frames[index+offset] if index+offset < len(frames) else None
        eligible = frame is not None and abs(frame['timestamp']-current['timestamp']-offset*500000) <= 60000
        if not eligible:
            annotations.append(None)
            if offset:
                ego.append([0., 0., 0.]); valid.append(False)
            continue
        e2g = pose_matrix(frame['ego2global_translation'], frame['ego2global_rotation'], planar=True)
        attitude = np.linalg.inv(e2g)@pose_matrix(frame['ego2global_translation'], frame['ego2global_rotation'])
        l2e = attitude@pose_matrix(frame['lidar2ego_translation'], frame['lidar2ego_rotation'])
        names = frame['anns']['gt_names']
        keep = np.array([name in NAV_CLASSES for name in names], dtype=bool)
        mapped = np.array([CLASS_NAMES.index(NAV_CLASSES[name]) for name in names if name in NAV_CLASSES], dtype=np.int64)
        boxes = transform_boxes_to_ego(box_container(frame['anns']['gt_boxes'][keep]), l2e)
        tracks = np.asarray(frame['anns']['track_tokens'])[keep].tolist()
        annotations.append(FrameAnnotation(boxes, mapped, tracks, np.full(len(mapped), -1), e2g,
                          int(frame['timestamp']), True, sensor_path(frame['lidar_path']), l2e))
        if offset:
            relative = inverse@e2g
            ego.append([relative[0, 3], relative[1, 3], np.arctan2(relative[1, 0], relative[0, 0])])
            valid.append(True)
    # Pacifica rear axle convention from the canonical nuPlan vehicle parameters.
    from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
    vehicle = get_pacifica_parameters()
    body = dict(length=vehicle.length, width=vehicle.width, rear_axle_to_center=vehicle.rear_axle_to_center)
    return SceneRecord(current['token'], current_record['log'], 'navsim', observation,
                        np.asarray(ego, dtype=np.float32), np.asarray(valid), annotations,
                        current['map_location'], e2g0, body,
                        {'origin': 'rear_axle', 'camera_order': NAVSIM_CAMERAS, 'horizon_s': 4.,
                         'navigation_source': 'provided NAVSIM high-level command',
                         'boxes': 'center xyz, length width height, source CCW yaw; planar NAVSIM rear-axle reference',
                         'visibility': 'not supplied by raw NAVSIM annotations; never interpreted as invisible',
                         'sensor_timing': 'source sensor2lidar calibration; no additional timestamp compensation available'})


def navigation_text(command):
    if command not in (0, 1, 2, 3):
        raise ValueError('Invalid high-level navigation')
    direction = ('turn left', 'keep straight', 'turn right', 'unknown')[command]
    return f'You are an autonomous driving agent. The navigation command for the current timestep is {direction}. Your task is to plan future actions based on the understanding of the driving scene.'


def nuscenes_scene(nusc, sample, *, image_size=(256, 448)):
    root = Path(nusc.dataroot)
    lidar = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    pose = nusc.get('ego_pose', lidar['ego_pose_token'])
    e2g0 = pose_matrix(pose['translation'], pose['rotation'])
    lidar_calibrated = nusc.get('calibrated_sensor', lidar['calibrated_sensor_token'])
    l2e0 = pose_matrix(lidar_calibrated['translation'], lidar_calibrated['rotation'])
    inverse = np.linalg.inv(e2g0)
    paths, sensors, intrinsics = [], [], []
    for name in NUSCENES_CAMERAS:
        image = nusc.get('sample_data', sample['data'][name])
        calibrated = nusc.get('calibrated_sensor', image['calibrated_sensor_token'])
        paths.append(str(root/image['filename']))
        from third_party.vad.ported.converter import obtain_sensor2top
        converted = obtain_sensor2top(nusc, image['token'], l2e0[:3, 3], l2e0[:3, :3],
                                      e2g0[:3, 3], e2g0[:3, :3], sensor_type=name)
        camera2lidar = np.eye(4)
        camera2lidar[:3, :3], camera2lidar[:3, 3] = converted['sensor2lidar_rotation'], converted['sensor2lidar_translation']
        sensors.append(l2e0@camera2lidar)
        intrinsics.append(calibrated['camera_intrinsic'])
    observation = camera_inputs(paths, sensors, intrinsics, image_size=image_size)
    # Previous/current sensor poses only; no first-future displacement fallback.
    if sample['prev']:
        previous = nusc.get('sample', sample['prev'])
        prev_lidar = nusc.get('sample_data', previous['data']['LIDAR_TOP'])
        prev_pose = nusc.get('ego_pose', prev_lidar['ego_pose_token'])
        relative = inverse@pose_matrix(prev_pose['translation'], prev_pose['rotation'])
        state = encode_ego([[-relative[0, 3], -relative[1, 3], -np.arctan2(relative[1, 0], relative[0, 0])]])
    else:
        raise ValueError('No legal previous ego pose; excluded identically from all formal groups')
    annotations, ego, valid, future_positions = [], [], [], []
    current = sample
    for offset in range(7):
        if current is None:
            annotations.append(None)
            if offset:
                ego.append([0., 0., 0.]); valid.append(False); future_positions.append(np.zeros(3))
            continue
        sd = nusc.get('sample_data', current['data']['LIDAR_TOP'])
        ep = nusc.get('ego_pose', sd['ego_pose_token'])
        e2g = pose_matrix(ep['translation'], ep['rotation'])
        cs = nusc.get('calibrated_sensor', sd['calibrated_sensor_token'])
        l2e = pose_matrix(cs['translation'], cs['rotation'])
        values, classes, tracks, visibility = [], [], [], []
        for token in current['anns']:
            annotation = nusc.get('sample_annotation', token)
            from nuscenes.eval.detection.utils import category_to_detection_name
            name = category_to_detection_name(annotation['category_name'])
            if name not in CLASS_NAMES:
                continue
            # Reuse devkit Box transformations, including calibrated ego pose.
            box = nusc.get_box(token)
            box.translate(-np.asarray(ep['translation']))
            box.rotate(Quaternion(ep['rotation']).inverse)
            values.append([*box.center, box.wlh[1], box.wlh[0], box.wlh[2], box.orientation.yaw_pitch_roll[0]])
            classes.append(CLASS_NAMES.index(name)); tracks.append(annotation['instance_token'])
            visibility.append(int(annotation['visibility_token']))
        annotations.append(FrameAnnotation(box_container(values), np.asarray(classes, dtype=np.int64), tracks,
                          np.asarray(visibility), e2g, sd['timestamp'], True, str(root/sd['filename']), l2e))
        if offset:
            relative = inverse@e2g
            ego.append([relative[0, 3], relative[1, 3], np.arctan2(relative[1, 0], relative[0, 0])])
            future_positions.append(relative[:3, 3].copy())
            valid.append(abs(sd['timestamp']-lidar['timestamp']-offset*500000) <= 60000)
        current = nusc.get('sample', current['next']) if current['next'] else None
    # VAD navigation protocol: final ego position in its lidar frame with
    # right/left x threshold +/-2 m. The intention is an explicitly supplied
    # log-future condition, not a coordinate/endpoint input to the policy.
    last = np.asarray(future_positions)[np.asarray(valid)][-1] if any(valid) else np.zeros(3)
    last_position = np.linalg.inv(l2e0)@np.array([*last, 1.])
    command = 2 if last_position[0] >= 2. else 0 if last_position[0] <= -2. else 1
    observation.update(state=state, token=sample['token'], lang=navigation_text(command))
    scene = nusc.get('scene', sample['scene_token'])
    log = nusc.get('log', scene['log_token'])
    return SceneRecord(sample['token'], scene['name'], 'nuscenes', observation,
                        np.asarray(ego, dtype=np.float32), np.asarray(valid), annotations,
                        log['location'], e2g0,
                        dict(length=4.084, width=1.85, rear_axle_to_center=0.),
                        {'origin': 'nuScenes ego sensor origin', 'camera_order': NUSCENES_CAMERAS,
                         'horizon_s': 3., 'navigation_source': 'VAD-style +/-2 m log-future intention',
                         'navigation_privileged_condition': True,
                         'ego_state': 'previous/current ego pose only',
                         'body_geometry': 'locked UniAD planning rectangle, ego-origin centered',
                         'sensor_timing': 'each current camera mapped via its timestamp ego pose to lidar-time ego(t0)'})
