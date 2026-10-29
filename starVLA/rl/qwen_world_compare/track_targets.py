"""Train-only current-visible tracks, linked into timestamped future states.

Coordinates follow locked NAVSIM v1.1 planar ego SE(2), not full 3-D ego roll
and pitch.  The existing current-object builder applies lidar2ego explicitly;
preparation refuses to mix it with the official path unless its actual input
transform is identity within the registered tolerance.
"""
from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np
from scipy.spatial.transform import Rotation

from tools.qwen_world_postcot.prepare_training_data import current_objects, CAMERAS, CLASSES


SCHEMA = 'qwen_world_compare_current_linked_future_tracks_v1'
FUTURE_TIMES_S = (1., 2., 3., 4.)
TIMESTAMP_TOLERANCE_US = 60000
LIDAR_IDENTITY_TOLERANCE = 1e-8
OFFICIAL_GT_SOURCE_SHA256 = '28f0ee850c35157ab5a9600a3aa68f0f5b0a848b278e7b322b10a000916723f9'


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 ** 2), b''):
            digest.update(chunk)
    return digest.hexdigest()


def official_coordinate_source():
    path = Path(__file__).parents[3] / 'navsim_v1.1/navsim/navsim/planning/scenario_builder/navsim_scenario_utils.py'
    actual = file_sha(path)
    if actual != OFFICIAL_GT_SOURCE_SHA256:
        raise ValueError('NAVSIM current object oracle differs from locked 3e8291 source')
    return dict(commit='3e8291bfa89ff247231e0227778840cd0a036896',
                path=str(path), sha256=actual,
                function='gt_boxes_oriented_box', downloaded_fixed_commit_byte_equal=True)


def ego_planar_pose(frame):
    quaternion = np.asarray(frame['ego2global_rotation'], dtype=np.float64)
    if quaternion.shape != (4,) or not np.isfinite(quaternion).all():
        raise ValueError('Missing/non-finite ego quaternion')
    rotation = Rotation.from_quat(quaternion[[1, 2, 3, 0]]).as_matrix()
    yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
    translation = np.asarray(frame['ego2global_translation'], dtype=np.float64)
    if translation.shape != (3,) or not np.isfinite(translation).all():
        raise ValueError('Missing/non-finite ego translation')
    return np.array([translation[0], translation[1], yaw], dtype=np.float64)


def planar_rotation(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def audit_annotation_frame(frame):
    """Return actual lidar/official-ego discrepancy; fail rather than guess."""
    quaternion = np.asarray(frame['lidar2ego_rotation'], dtype=np.float64)
    rotation = Rotation.from_quat(quaternion[[1, 2, 3, 0]]).as_matrix()
    translation = np.asarray(frame['lidar2ego_translation'], dtype=np.float64)
    rotation_error = float(np.max(np.abs(rotation - np.eye(3))))
    translation_error = float(np.max(np.abs(translation)))
    if max(rotation_error, translation_error) > LIDAR_IDENTITY_TOLERANCE:
        raise ValueError('Current object lidar2ego is not equivalent to the official ego annotation frame: '
                         f'token={frame.get("token")} rotation={rotation_error} translation={translation_error}')
    boxes = np.asarray(frame['anns']['gt_boxes'], dtype=np.float64)
    finite = boxes[np.isfinite(boxes).all(axis=1)] if len(boxes) else boxes
    ego = ego_planar_pose(frame)
    if len(finite):
        old_local = finite[:, :3] @ rotation.T + translation
        local_error = float(np.max(np.abs(old_local[:, :2] - finite[:, :2])))
        official_global = finite[:, :2] @ planar_rotation(ego[2]).T + ego[:2]
        old_global = old_local[:, :2] @ planar_rotation(ego[2]).T + ego[:2]
        global_error = float(np.max(np.abs(old_global - official_global)))
        reconstructed = (official_global - ego[:2]) @ planar_rotation(ego[2])
        roundtrip_error = float(np.max(np.abs(reconstructed - finite[:, :2])))
    else:
        local_error = global_error = roundtrip_error = 0.0
    if max(local_error, global_error, roundtrip_error) > 1e-6:
        raise ValueError('Official/current-builder planar coordinate audit failed')
    return dict(frames=1, boxes=len(finite), lidar_rotation_max_abs=rotation_error,
                lidar_translation_max_abs_m=translation_error, current_xy_max_abs_m=local_error,
                global_xy_max_abs_m=global_error, planar_roundtrip_max_abs_m=roundtrip_error)


def current_selected_indices(frame, image_sizes):
    """Exact existing center-frustum selection, retaining original track ids."""
    candidates = []
    for index, (box, name) in enumerate(zip(frame['anns']['gt_boxes'], frame['anns']['gt_names'])):
        if name not in CLASSES or not np.isfinite(box).all() or np.any(box[3:6] <= 0):
            continue
        visible = False
        for camera in CAMERAS:
            cam = frame['cams'][camera.upper()]
            width, height = image_sizes[camera]
            point = (np.asarray(box[:3]) - np.asarray(cam['sensor2lidar_translation'])) @ np.asarray(cam['sensor2lidar_rotation'])
            if point[2] <= .1:
                continue
            pixel = np.asarray(cam['cam_intrinsic']) @ point
            pixel = pixel[:2] / pixel[2]
            if 4 <= pixel[0] < width - 4 and 4 <= pixel[1] < height - 4:
                visible = True
        if visible:
            candidates.append(index)
    # audit_annotation_frame has checked lidar2ego=identity: same distance and
    # ordering as the original builder, including its stable annotation tie.
    return sorted(candidates, key=lambda i: (float(np.linalg.norm(frame['anns']['gt_boxes'][i][:2])), i))[:32]


def nearest_future_frames(frames, current_index):
    timestamps = np.asarray([int(frame['timestamp']) for frame in frames], dtype=np.int64)
    if len(timestamps) > 1 and np.any(np.diff(timestamps) <= 0):
        raise ValueError('Frame timestamps must be strictly increasing for future association')
    now = int(timestamps[current_index])
    selected = []
    for seconds in FUTURE_TIMES_S:
        target = now + int(seconds * 1e6)
        insertion = int(np.searchsorted(timestamps, target))
        options = [index for index in (insertion - 1, insertion)
                   if current_index < index < len(frames)]
        nearest = min(options, key=lambda index: (abs(int(timestamps[index]) - target), index)) if options else None
        if nearest is None or abs(int(timestamps[nearest]) - target) > TIMESTAMP_TOLERANCE_US:
            selected.append(None)
        else:
            selected.append(nearest)
    return tuple(selected)


def future_box_in_current_ego(box, future_frame, current_frame):
    """Official ego-local box -> global -> fixed current rear-axle frame."""
    future = ego_planar_pose(future_frame)
    current = ego_planar_pose(current_frame)
    global_xy = planar_rotation(future[2]) @ np.asarray(box[:2], dtype=np.float64) + future[:2]
    current_xy = planar_rotation(current[2]).T @ (global_xy - current[:2])
    yaw = float(box[6]) + future[2] - current[2]
    return np.array([current_xy[0], current_xy[1], box[3], box[4], np.sin(yaw), np.cos(yaw)], dtype=np.float32)


def _sealed_current_indices(frame, objects):
    """Attach tracks to the sealed old current selection, without future data."""
    lookup = {}
    boxes = frame['anns']['gt_boxes']
    names = frame['anns']['gt_names']
    for index, (box, name) in enumerate(zip(boxes, names)):
        if name in CLASSES and np.isfinite(box).all() and np.all(box[3:6] > 0):
            lookup.setdefault((*(float(v) for v in box[:6]), CLASSES.index(name)), []).append(index)
    selected, used = [], set()
    for target, category in zip(objects['boxes'], objects['classes']):
        choices = lookup.get((*(float(v) for v in target[:6]), int(category)), [])
        matching = [index for index in choices if index not in used and
                    np.allclose([np.sin(boxes[index][6]), np.cos(boxes[index][6])],
                                target[6:8], rtol=0, atol=1e-7)]
        if not matching:
            raise ValueError('Sealed current object cannot be linked to its raw current annotation')
        # Original current_objects preserves annotation order for exact geometry
        # ties. No future appearance/visibility enters this tie breaking.
        selected.append(matching[0])
        used.add(matching[0])
    return selected


def build_scene_tracks(frames, current_index, image_sizes, expected_current_objects=None,
                       *, coordinates_audited=False, verify_current_builder=True):
    current = frames[current_index]
    if not coordinates_audited:
        audit_annotation_frame(current)
    objects = (current_objects(current, image_sizes)
               if expected_current_objects is None or verify_current_builder else expected_current_objects)
    if expected_current_objects is not None and verify_current_builder:
        for field in ('boxes', 'classes', 'valid', 'coverage_valid'):
            a, b = objects[field], expected_current_objects[field]
            if field == 'boxes':
                if np.asarray(a).shape != np.asarray(b).shape or not np.allclose(a, b, rtol=0, atol=1e-7):
                    raise ValueError('Current object geometry differs from the sealed training record')
            elif a != b:
                raise ValueError('Current object selection differs from the sealed training record: ' + field)
    indices = (_sealed_current_indices(current, objects) if expected_current_objects is not None
               else current_selected_indices(current, image_sizes))
    if len(indices) != len(objects['boxes']):
        raise ValueError('Current visible-object selection lost correspondence')
    def track_token(value):
        return None if value is None or str(value) == '' else str(value)
    tokens = [track_token(current['anns']['track_tokens'][index]) for index in indices]
    known_tokens = [token for token in tokens if token is not None]
    if len(known_tokens) != len(set(known_tokens)):
        raise ValueError('Ambiguous current track identity')
    for index, box in zip(indices, objects['boxes']):
        raw = current['anns']['gt_boxes'][index]
        expected = [*raw[:6], np.sin(raw[6]), np.cos(raw[6])]
        if not np.allclose(box, expected, rtol=0, atol=1e-7):
            raise ValueError('Current selected object indices do not align to existing object targets')
    states = np.zeros((32, 4, 6), dtype=np.float32)
    valid = np.zeros((32, 4), dtype=np.bool_)
    timestamps = np.full((4,), -1, dtype=np.int64)
    actual_dt = np.full((4,), -1., dtype=np.float32)
    time_valid = np.zeros((4,), dtype=np.bool_)
    duplicate_future = 0
    for time_index, future_index in enumerate(nearest_future_frames(frames, current_index)):
        if future_index is None:
            continue
        future = frames[future_index]
        if not coordinates_audited:
            audit_annotation_frame(future)
        timestamps[time_index] = int(future['timestamp'])
        actual_dt[time_index] = (timestamps[time_index] - int(current['timestamp'])) / 1e6
        time_valid[time_index] = True
        lookup = {}
        for index, token in enumerate(future['anns']['track_tokens']):
            token = track_token(token)
            if token is not None:
                lookup.setdefault(token, []).append(index)
        selected_tracks, selected_boxes = [], []
        for track_index, token in enumerate(tokens):
            if token is None:
                continue
            matches = lookup.get(token, [])
            if len(matches) != 1:
                duplicate_future += int(len(matches) > 1)
                continue
            box = np.asarray(future['anns']['gt_boxes'][matches[0]])
            if not np.isfinite(box).all() or np.any(box[3:6] <= 0):
                continue
            selected_tracks.append(track_index)
            selected_boxes.append(box)
        if selected_boxes:
            box_array = np.asarray(selected_boxes, dtype=np.float64)
            future_pose, current_pose = ego_planar_pose(future), ego_planar_pose(current)
            global_xy = box_array[:, :2] @ planar_rotation(future_pose[2]).T + future_pose[:2]
            current_xy = (global_xy - current_pose[:2]) @ planar_rotation(current_pose[2])
            yaw = box_array[:, 6] + future_pose[2] - current_pose[2]
            result = np.column_stack((current_xy, box_array[:, 3:5], np.sin(yaw), np.cos(yaw)))
            states[selected_tracks, time_index] = result.astype(np.float32)
            valid[selected_tracks, time_index] = True
    return dict(schema=SCHEMA, token=str(current['token']), timestamp_us=int(current['timestamp']),
        current_objects=objects, track_tokens=tokens, current_count=len(tokens),
        future_states=states, future_valid=valid, future_times_s=FUTURE_TIMES_S,
        future_timestamps_us=timestamps, future_time_valid=time_valid, actual_dt_s=actual_dt,
        duplicate_future_track_matches=duplicate_future,
        future_coordinate='fixed_current_ego_rear_axle_xy_lw_sinyaw_cosyaw',
        coverage_valid=False, selection='existing_current_visible_center_frustum_nearest32')


class FutureTrackStore:
    """Hash-checked, finite-LRU per-log shard reader; no teacher/actor inputs."""

    def __init__(self, root, max_cached_logs=4):
        self.root = Path(root)
        self.manifest = json.loads((self.root / 'manifest.json').read_text())
        complete = json.loads((self.root / 'COMPLETE').read_text())
        if (self.manifest['schema'] != SCHEMA or complete.get('status') != 'COMPLETE'
                or complete['manifest_sha256'] != file_sha(self.root / 'manifest.json')):
            raise ValueError('Future track targets are incomplete or changed')
        if ('coordinate_audit_sha256' in self.manifest and
                file_sha(self.root / 'coordinate_audit.json') != self.manifest['coordinate_audit_sha256']):
            raise ValueError('Future track coordinate audit has changed')
        self.index = self.manifest['token_index']
        if len(self.index) != self.manifest['count']:
            raise ValueError('Future track token coverage is incomplete')
        if int(max_cached_logs) < 1:
            raise ValueError('Target reader needs a positive finite LRU size')
        self.max_cached_logs = int(max_cached_logs)
        self.cache = OrderedDict()

    def get(self, token):
        shard = self.index[token]
        if shard not in self.cache:
            entry = self.manifest['shards'][shard]
            path = self.root / shard
            if file_sha(path) != entry['sha256']:
                raise ValueError('Future track shard has changed: ' + shard)
            with path.open('rb') as stream:
                rows = pickle.load(stream)
            if set(rows) != set(entry['tokens']):
                raise ValueError('Future track shard token coverage differs')
            for row in rows.values():
                for value in row.values():
                    if isinstance(value, np.ndarray):
                        value.flags.writeable = False
            self.cache[shard] = rows
            while len(self.cache) > self.max_cached_logs:
                self.cache.popitem(last=False)
        self.cache.move_to_end(shard)
        return self.cache[shard][token]

    __getitem__ = get

