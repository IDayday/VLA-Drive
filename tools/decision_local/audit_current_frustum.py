"""Current-camera geometric support of fixed queries, separate from GT validity.

This is an analysis-only pinhole/distortion frustum proxy. It does not establish
visibility through occluders, the vehicle body, or an elevated road surface.
No image pixels, future annotations, or GT ego trajectory are used.
"""
import argparse
from collections import defaultdict
from pathlib import Path
import pickle
import sys
import time
import cv2
import numpy as np
from PIL import Image
import torch
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.decision_local.common import atomic_json, check_host, digest, identity, read, source_sha
from starVLA.dataloader.structured_world.cameras import NAVSIM_CAMERAS, crop_16_9
from starVLA.model.modules.structured_world.queries import body_points


def points_in_current_camera(points, sensor2ego, intrinsics, distortion, image_size):
    """Project ego(t0) points into the original, distorted, center-cropped ROI."""
    points = np.asarray(points, dtype=np.float64)
    sensor = np.asarray(sensor2ego, dtype=np.float64)
    camera = (points - sensor[:3, 3]) @ sensor[:3, :3]
    positive = np.isfinite(camera).all(-1) & (camera[:, 2] > 1e-6)
    result = np.zeros(len(points), dtype=bool)
    if not positive.any():
        return result
    pixels, _ = cv2.projectPoints(camera[positive], np.zeros(3), np.zeros(3),
                                 np.asarray(intrinsics, dtype=np.float64),
                                 None if distortion is None else np.asarray(distortion, dtype=np.float64))
    pixels = pixels[:, 0]
    left, top, right, bottom = crop_16_9(*image_size)
    # Pixel-center convention: the integer coordinates in the actual retained
    # crop are [left, right-1] x [top, bottom-1]. No clamping at image borders.
    result[positive] = (np.isfinite(pixels).all(-1) &
        (pixels[:, 0] >= left) & (pixels[:, 0] <= right-1) &
        (pixels[:, 1] >= top) & (pixels[:, 1] <= bottom-1))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('features', 'queries', 'output'):
        p.add_argument('--'+key, type=Path, required=True)
    p.add_argument('--max-seconds', type=float, required=True)
    a = p.parse_args(); resources = check_host(); started = time.monotonic()
    torch.set_num_threads(1); cv2.setNumThreads(1)
    from starVLA.dataloader.structured_world.adapters import pose_matrix
    fmeta, qmeta = read(a.features/'identity.json'), read(a.queries/'identity.json')
    if qmeta['features'] != fmeta['identity'] or qmeta['contains_GT']:
        raise ValueError('Only the current-input query bank is accepted')
    for root in (a.features, a.queries):
        if not (root/'COMPLETE.json').exists():
            raise ValueError('Complete population required')
    if a.output.exists():
        raise FileExistsError('Immutable analysis output')
    poses, bodies = {}, {}
    for receipt in sorted((a.queries/'records').glob('*.json')):
        meta = read(receipt); path = a.queries/'chunks'/(receipt.stem+'.safetensors')
        if digest(path) != meta['sha256']:
            raise ValueError('Changed query chunk')
        data = load_file(str(path))
        for i, token in enumerate(meta['tokens']):
            poses[token] = data['poses'][i, 0]
            bodies[token] = dict(zip(('length', 'width', 'rear_axle_to_center'), data['body'][i].tolist()))
    logs = defaultdict(list)
    for row in fmeta['index']:
        logs[row['log']].append(row['token'])
    fractions = {}; camera_fractions = {}; calibrations = {}
    for log, tokens in logs.items():
        if time.monotonic()-started > a.max_seconds:
            raise TimeoutError('Finite CPU analysis budget reached')
        with (Path('/mnt/navsim/trainval_navsim_logs/trainval')/(log+'.pkl')).open('rb') as stream:
            frames = {x['token']: x for x in pickle.load(stream)}
        for token in tokens:
            frame = frames[token]
            e2g = pose_matrix(frame['ego2global_translation'], frame['ego2global_rotation'], planar=True)
            attitude = np.linalg.inv(e2g) @ pose_matrix(frame['ego2global_translation'], frame['ego2global_rotation'])
            lidar2ego = attitude @ pose_matrix(frame['lidar2ego_translation'], frame['lidar2ego_rotation'])
            xy = body_points(poses[token], bodies[token]).numpy()
            xyz = np.concatenate((xy.reshape(-1, 2), np.zeros((xy.size//2, 1))), axis=-1)
            visible = []; calibration = []
            for name in NAVSIM_CAMERAS:
                value = frame['cams'][name]
                camera2lidar = np.eye(4)
                camera2lidar[:3, :3] = value['sensor2lidar_rotation']
                camera2lidar[:3, 3] = value['sensor2lidar_translation']
                sensor = lidar2ego @ camera2lidar
                path = None
                for root in ('/mnt/navsim/trainval_all/trainval_sensor_blobs/trainval',
                             '/mnt/navsim/trainval_sensor_blobs/trainval'):
                    candidate = Path(root)/value['data_path']
                    if candidate.is_file():
                        path = candidate; break
                if path is None:
                    raise FileNotFoundError('Current source image header unavailable')
                with Image.open(path) as image:
                    size = image.size  # header only, no RGB decode or export
                K = np.asarray(value['cam_intrinsic']); distortion = value.get('distortion')
                projected = points_in_current_camera(xyz, sensor, K, distortion, size)
                visible.append(projected.reshape(xy.shape[:-1]))
                calibration.append({'name': name, 'sensor2ego': sensor.tolist(), 'K': K.tolist(),
                                    'distortion': None if distortion is None else np.asarray(distortion).tolist(),
                                    'image_size': size, 'roi': crop_16_9(*size)})
            visible = np.stack(visible)
            fractions[token] = visible.any(0).mean(-1).astype(np.float32)
            camera_fractions[token] = visible.mean(-1).astype(np.float32)
            calibrations[token] = identity(calibration)
        print({'log': log, 'scenes': len(fractions)}, flush=True)
    tokens = [row['token'] for row in fmeta['index']]
    fraction = np.stack([fractions[t] for t in tokens])
    a.output.mkdir(parents=True)
    np.savez_compressed(a.output/'frustum.npz', tokens=np.array(tokens), fraction=fraction,
                        per_camera_fraction=np.stack([camera_fractions[t] for t in tokens]))
    atomic_json(a.output/'COMPLETE.json', {
        'scenes': len(tokens), 'features': fmeta['identity'], 'queries': qmeta['identity'],
        'source_sha': source_sha(ROOT), 'writer_sha256': digest(__file__),
        'calibration_hash': identity(calibrations), 'array_sha256': digest(a.output/'frustum.npz'),
        'CPU_wall_seconds': time.monotonic()-started, 'resources': resources,
        'sampling_seed': 42, 'camera_order': NAVSIM_CAMERAS,
        'full_body_frustum_fraction': float((fraction == 1).mean()),
        'any_body_frustum_fraction': float((fraction > 0).mean()),
        'mean_body_point_support': float(fraction.mean()),
        'support_by_time': fraction.mean((0, 1)).tolist(),
        'assumptions': ['ground plane z=0 in current planar ego frame',
                        'original distorted camera ROI, same source as S0',
                        'frustum membership only; no occlusion or road-height ground truth',
                        'analysis-only; never a GT validity substitute or policy input'],
        'no_future_or_annotation_access_in_computation': True})


if __name__ == '__main__':
    main()
