"""Causal, same-log NAVSIM history. No labels, future masks or teacher inputs.

Read geometry from real <=t0 frames. A missing frame stays missing; never copy
current RGB or substitute a future frame. Existing current_observation and its
1x4 ego encoding are intentionally unchanged.
"""
from __future__ import annotations

import json
from pathlib import Path
import hashlib
import math

import numpy as np
from PIL import Image
from pyquaternion import Quaternion

from .st_world_schema import identity_hash, require_hash

VIEWS = ('CAM_F0', 'CAM_L0', 'CAM_R0')
HISTORY_TIMES = (-1.0, -0.5)


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()


def _matrix(value, shape, label):
    a = np.asarray(value, dtype=np.float64)
    if a.shape != shape or not np.isfinite(a).all():
        raise ValueError('Invalid ' + label)
    return a


def ego_pose(frame, planar=False):
    q = Quaternion(frame['ego2global_rotation'])
    if planar:
        q = Quaternion(axis=[0, 0, 1], angle=q.yaw_pitch_roll[0])
    m = np.eye(4)
    m[:3, :3] = q.rotation_matrix
    m[:3, 3] = _matrix(frame['ego2global_translation'], (3,), 'ego translation')
    return m


def crop_resize_affine(original_wh, output_wh=(1024, 576)):
    """The existing current_observation center16:9 crop, then LANCZOS resize."""
    w, h = original_wh
    if min(w, h) <= 0:
        raise ValueError('Invalid image dimensions')
    cw, ch = (int(h * 16 / 9), h) if w / h > 16 / 9 else (w, int(w * 9 / 16))
    left, top = (w - cw) // 2, (h - ch) // 2
    ow, oh = output_wh
    affine = np.array([[ow / cw, 0, -left * ow / cw], [0, oh / ch, -top * oh / ch], [0, 0, 1.]])
    return [left, top, left + cw, top + ch], affine


def image_reference(path):
    path = Path(path).resolve()
    before = path.stat()
    digest = file_sha256(path)
    with Image.open(path) as im:
        wh = list(im.size)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError('Image changed during index construction')
    return {'path': str(path), 'sha256': digest, 'bytes': before.st_size,
            'mtime_ns': before.st_mtime_ns, 'original_wh': wh}


def resolve_camera_path(relative, sensor_roots):
    rel = Path(relative)
    if rel.is_absolute() or '..' in rel.parts:
        raise ValueError('Raw camera path must be relative to registered sensor roots')
    for root in sensor_roots:
        candidate = Path(root) / rel
        if candidate.is_file():
            return candidate
    return None


def causal_frames(frames, token, offsets_s=HISTORY_TIMES, tolerance_s=0.05):
    if (not offsets_s or tuple(sorted(set(offsets_s))) != tuple(offsets_s) or
            any(not math.isfinite(x) or x >= 0 for x in offsets_s) or not 0 <= tolerance_s <= .05):
        raise ValueError('History requires distinct increasing negative offsets and tolerance <=50ms')
    tokens = [f['token'] for f in frames]
    if len(set(tokens)) != len(tokens) or token not in tokens:
        raise ValueError('Ambiguous or missing current frame token')
    current = frames[tokens.index(token)]
    t0 = int(current['timestamp'])
    # Future dictionaries are not passed to geometry/path logic below.
    past = [f for f in frames if int(f['timestamp']) <= t0]
    times = np.array([int(f['timestamp']) for f in past], dtype=np.int64)
    if len(times) == 0 or not (np.diff(times) > 0).all():
        raise ValueError('Nonmonotonic/duplicate causal timestamps')
    chosen = []
    used = set()
    for offset in offsets_s:
        wanted = t0 + round(offset * 1e6)
        distances = np.abs(times - wanted)
        candidates = np.flatnonzero(distances <= round(tolerance_s * 1e6))
        candidates = [int(i) for i in candidates if times[i] < t0]
        if not candidates:
            chosen.append(None)
            continue
        best = min(distances[i] for i in candidates)
        nearest = [i for i in candidates if distances[i] == best]
        if len(nearest) != 1 or past[nearest[0]]['token'] in used:
            raise ValueError('Ambiguous or duplicated history frame')
        selected = past[nearest[0]]
        used.add(selected['token'])
        chosen.append(selected)
    return current, chosen


def build_history_record(frames, token, log, current_record, sensor_roots, *,
                         raw_log_sha256, offsets_s=HISTORY_TIMES, tolerance_s=.05):
    """The only raw fields read here are timestamp, pose, cams and lidar2ego."""
    require_hash(raw_log_sha256, 'raw_log_sha256')
    current, past = causal_frames(frames, token, offsets_s, tolerance_s)
    t0 = int(current['timestamp'])
    if current_record['token'] != token or current_record['log'] != log or current_record['timestamp'] != t0:
        raise ValueError('Current observation/raw-log binding changed')
    if len(current_record['image_paths']) != len(VIEWS):
        raise ValueError('Current view count changed')
    inv0 = np.linalg.inv(ego_pose(current, planar=True))
    rows = []
    for offset, frame in zip(offsets_s, past):
        entry = {'requested_s': float(offset), 'token': None, 'timestamp_us': None,
                 'actual_offset_s': None, 'relative_ego_planar': None, 'views': []}
        if frame is not None:
            entry.update(token=frame['token'], timestamp_us=int(frame['timestamp']),
                         actual_offset_s=(int(frame['timestamp']) - t0) / 1e6,
                         relative_ego_planar=(inv0 @ ego_pose(frame, planar=True)).tolist())
        for name in VIEWS:
            view = {'camera': name, 'valid': False, 'image': None, 'intrinsics': None,
                    'distortion': None, 'camera_to_current_planar_ego': None,
                    'crop_xyxy': None, 'resize_affine': None, 'sensor_timestamp_us': None}
            meta = frame.get('cams', {}).get(name) if frame is not None else None
            path = resolve_camera_path(meta['data_path'], sensor_roots) if meta else None
            if path is not None:
                ref = image_reference(path)
                crop, affine = crop_resize_affine(ref['original_wh'])
                c2l = np.eye(4)
                c2l[:3, :3] = _matrix(meta['sensor2lidar_rotation'], (3, 3), 'camera rotation')
                c2l[:3, 3] = _matrix(meta['sensor2lidar_translation'], (3,), 'camera translation')
                l2e = _matrix(frame['lidar2ego'], (4, 4), 'lidar2ego')
                c2e0 = inv0 @ ego_pose(frame) @ l2e @ c2l
                k = _matrix(meta['cam_intrinsic'], (3, 3), 'camera intrinsics')
                distortion = np.asarray(meta['distortion'], dtype=float)
                if not np.isfinite(distortion).all():
                    raise ValueError('Invalid camera distortion')
                view.update(valid=True, image=ref, intrinsics=(affine @ k).tolist(),
                            distortion=distortion.tolist(), camera_to_current_planar_ego=c2e0.tolist(),
                            crop_xyxy=crop, resize_affine=affine.tolist(),
                            sensor_timestamp_us=int(meta['timestamp']) if 'timestamp' in meta else None)
            entry['views'].append(view)
        rows.append(entry)
    record = {'schema': 'st_causal_history_v1', 'token': token, 'log': log,
              'current_timestamp_us': t0, 'view_names': list(VIEWS), 'offsets_s': list(offsets_s),
              'timestamp_tolerance_s': tolerance_s, 'source_raw_sha256': raw_log_sha256,
              'current_record_identity': identity_hash(current_record), 'frames': rows,
              'preprocess': 'original center16:9 then1024x576 LANCZOS',
              'missing_policy': 'masked_absent_no_current_or_future_substitution'}
    record['identity'] = identity_hash(record)
    validate_history_record(record)
    return record


def validate_history_record(record):
    keys = {'schema', 'token', 'log', 'current_timestamp_us', 'view_names', 'offsets_s',
            'timestamp_tolerance_s', 'source_raw_sha256', 'current_record_identity', 'frames',
            'preprocess', 'missing_policy', 'identity'}
    if set(record) != keys or record['schema'] != 'st_causal_history_v1':
        raise ValueError('History schema/whitelist changed')
    if identity_hash({k: v for k, v in record.items() if k != 'identity'}) != record['identity']:
        raise ValueError('History record content hash changed')
    if record['view_names'] != list(VIEWS) or len(record['frames']) != len(record['offsets_s']):
        raise ValueError('History layout mismatch')
    offsets = record['offsets_s']
    if tuple(sorted(set(offsets))) != tuple(offsets) or any(x >= 0 for x in offsets):
        raise ValueError('Noncausal history offsets')
    previous_tokens = set()
    for expected, frame in zip(offsets, record['frames']):
        if set(frame) != {'requested_s', 'token', 'timestamp_us', 'actual_offset_s', 'relative_ego_planar', 'views'}:
            raise ValueError('History frame contains undeclared input fields')
        if frame['requested_s'] != expected or len(frame['views']) != len(VIEWS):
            raise ValueError('History time/view order changed')
        if frame['timestamp_us'] is not None:
            delta = (frame['timestamp_us'] - record['current_timestamp_us']) / 1e6
            if (delta >= 0 or abs(delta - expected) > record['timestamp_tolerance_s'] + 1e-9 or
                    delta != frame['actual_offset_s'] or frame['token'] in previous_tokens or frame['token'] == record['token']):
                raise ValueError('Future/current/duplicate history substitution')
            _matrix(frame['relative_ego_planar'], (4, 4), 'history ego transform')
            previous_tokens.add(frame['token'])
        elif any(v['valid'] for v in frame['views']):
            raise ValueError('Missing timestamp cannot contain valid history')
        for name, view in zip(VIEWS, frame['views']):
            if set(view) != {'camera','valid','image','intrinsics','distortion','camera_to_current_planar_ego',
                             'crop_xyxy','resize_affine','sensor_timestamp_us'}:
                raise ValueError('History camera contains undeclared input fields')
            if view['camera'] != name or type(view['valid']) is not bool:
                raise ValueError('History camera order/mask mismatch')
            if view['valid']:
                if frame['token'] is None or view['image'] is None:
                    raise ValueError('Valid view lacks real history source')
                require_hash(view['image']['sha256'], 'history image hash')
                _matrix(view['intrinsics'], (3, 3), 'history intrinsics')
                _matrix(view['camera_to_current_planar_ego'], (4, 4), 'history camera transform')
            elif view['image'] is not None:
                raise ValueError('Invalid view must not contain substitute image')
    return record


def load_history_images(record, *, verify_images=True):
    """Returns [time][view] PIL or None, bool mask, and causal geometry record."""
    validate_history_record(record)
    images, valid = [], []
    for frame in record['frames']:
        per_view = []
        for view in frame['views']:
            if not view['valid']:
                per_view.append(None)
                continue
            ref = view['image']
            if verify_images and file_sha256(ref['path']) != ref['sha256']:
                raise ValueError('Changed history image')
            with Image.open(ref['path']) as source:
                if list(source.size) != ref['original_wh']:
                    raise ValueError('Changed history image geometry')
                image = source.convert('RGB').crop(tuple(view['crop_xyxy']))
                per_view.append(image.resize((1024, 576), Image.Resampling.LANCZOS))
        images.append(per_view)
        valid.append([v['valid'] for v in frame['views']])
    return images, np.asarray(valid, dtype=bool)


class HistoryReader:
    def __init__(self, root, expected_identity):
        self.root = Path(root)
        self.manifest = json.loads((self.root / 'HISTORY_MANIFEST.json').read_text())
        if (self.manifest['identity'] != expected_identity or
                identity_hash({k:v for k,v in self.manifest.items() if k != 'identity'}) != expected_identity):
            raise ValueError('History manifest identity mismatch')
        receipt = json.loads((self.root / 'COMPLETE.json').read_text())
        if receipt['history_manifest_sha256'] != file_sha256(self.root / 'HISTORY_MANIFEST.json'):
            raise ValueError('History manifest not completely committed')
        self.index = {r['token']: r for r in self.manifest['scenes']}
        if len(self.index) != len(self.manifest['scenes']):
            raise ValueError('Duplicate history scenes')

    def read(self, token, *, current_record=None):
        row = self.index[token]
        path = self.root / row['record']
        if file_sha256(path) != row['sha256']:
            raise ValueError('History record checksum mismatch')
        record = validate_history_record(json.loads(path.read_text()))
        if record['token'] != token or record['log'] != row['log']:
            raise ValueError('History source-group mismatch')
        if current_record is not None and record['current_record_identity'] != identity_hash(current_record):
            raise ValueError('History attached to changed current input')
        return record
