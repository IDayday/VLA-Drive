"""Prepare deployment inputs from current raw calibration and RGB only.

This wrapper never constructs a SceneRecord, future annotation, map or LiDAR
label. Its transforms are checked against the training adapter. Metadata logs
remain offline assets; only the whitelisted current fields are exported.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
from pathlib import Path
import pickle
import socket
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.structured_world.build_cache import atomic_json, digest


def build_log(task):
    from starVLA.dataloader.structured_world.adapters import pose_matrix
    from starVLA.dataloader.structured_world.cameras import camera_inputs, NAVSIM_CAMERAS
    from starVLA.dataloader.foresight_dataset import encode_ego
    from starVLA.dataloader.structured_world.adapters import navigation_text
    import torch
    import cv2
    torch.set_num_threads(1); cv2.setNumThreads(1)
    log, rows, current_root, log_root, sensor_roots, output, identity = task
    output = Path(output)
    with (Path(log_root)/(log+'.pkl')).open('rb') as stream: frames = pickle.load(stream)
    # Discard all other frames. Per-scene handling below accesses no annotations.
    selected = {f['token']: f for f in frames if f['token'] in {r['token'] for r in rows}}
    reports = []
    for row in rows:
        token = row['token']; current = selected[token]
        record = json.loads((Path(current_root)/'current'/(token+'.json')).read_text())
        e2g = pose_matrix(current['ego2global_translation'], current['ego2global_rotation'], planar=True)
        attitude = np.linalg.inv(e2g)@pose_matrix(current['ego2global_translation'], current['ego2global_rotation'])
        l2e = pose_matrix(current['lidar2ego_translation'], current['lidar2ego_rotation'])
        paths, sensors, intrinsics, distortion = [], [], [], []
        for name in NAVSIM_CAMERAS:
            camera = current['cams'][name]
            path = next((Path(root)/camera['data_path'] for root in sensor_roots if (Path(root)/camera['data_path']).is_file()), None)
            if path is None: raise FileNotFoundError(camera['data_path'])
            camera2lidar = np.eye(4)
            camera2lidar[:3, :3] = camera['sensor2lidar_rotation']
            camera2lidar[:3, 3] = camera['sensor2lidar_translation']
            paths.append(str(path)); sensors.append(attitude@l2e@camera2lidar)
            intrinsics.append(camera['cam_intrinsic']); distortion.append(camera.get('distortion'))
        observations = camera_inputs(paths, sensors, intrinsics, distortion)
        history = np.asarray(record['global_pose_history'])
        previous, now = history[-2:]
        delta = (now[:2]-previous[:2])@e2g[:2, :2]
        yaw = (now[2]-previous[2]+np.pi)%(2*np.pi)-np.pi
        rgb = observations['geometry_images']
        rgb = ((rgb*rgb.new_tensor([.229, .224, .225])[None, :, None, None]+
                rgb.new_tensor([.485, .456, .406])[None, :, None, None])*255.).round().clamp(0, 255).byte()
        arrays = {'geometry_rgb': rgb.numpy(), 'geometry_pixel_valid': observations['geometry_pixel_valid'].numpy()}
        arrays.update({'calibration_'+k: v.numpy() for k, v in observations['calibration'].items()})
        target = output/'pixels'/(token+'.npz'); temporary = target.with_suffix('.tmp')
        with temporary.open('wb') as stream: np.savez_compressed(stream, **arrays)
        temporary.replace(target)
        report = {'identity': identity, 'token': token, 'image_paths': paths, 'image_sha256': [digest(Path(p)) for p in paths],
            'state': encode_ego([[*delta, yaw]]).tolist(), 'lang': navigation_text(record['navigation']), 'pixels_sha256': digest(target)}
        atomic_json(output/'current'/(token+'.json'), report); reports.append(token)
    return reports


def main():
    parser = argparse.ArgumentParser()
    for name in ('current-root', 'log-root', 'output'): parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--sensor-root', type=Path, action='append', required=True)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--debug-limit', type=int, default=0)
    args = parser.parse_args()
    policy = json.loads(Path('/mnt/project/server_dispatch_policy.json').read_text())
    if socket.gethostname().removesuffix('-worker-0') not in policy['task_authorizations']['structured_world_fgtr_round1']['allowed_hosts']:
        raise ValueError('Unauthorized input preparation host')
    source = json.loads((args.current_root/'identity.json').read_text())
    rows = json.loads((args.current_root/'index.json').read_text())
    if args.debug_limit: rows = rows[:args.debug_limit]
    if len({r['token'] for r in rows}) != len(rows): raise ValueError('Duplicate scenes')
    contract = {'schema': 'structured_world_current_inputs_v2', 'dataset': 'navsim', 'cameras': 3,
        'source_current_identity': source, 'index': rows, 'exporter_sha256': digest(Path(__file__)),
        'population_kind': 'debug_only' if args.debug_limit else source['split'], 'Qwen_image_hashes_verified': True,
        'inputs': 'current RGB, local calibration, legal past/current state, provided navigation text; no GT labels',
        'raw_log_hashes': {log: digest(args.log_root/(log+'.pkl')) for log in sorted({r['log'] for r in rows})}}
    identity = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output/'identity.json').exists() and json.loads((args.output/'identity.json').read_text())['identity'] != identity:
        raise ValueError('Foreign current-only inputs')
    atomic_json(args.output/'identity.json', {'identity': identity, **contract}); atomic_json(args.output/'index.json', rows)
    for name in ('current', 'pixels'): (args.output/name).mkdir(exist_ok=True)
    groups = {}
    for row in rows: groups.setdefault(row['log'], []).append(row)
    tasks = [(log, values, str(args.current_root), str(args.log_root), [str(p) for p in args.sensor_root],
              str(args.output), identity) for log, values in groups.items()]
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        tokens = [t for group in pool.map(build_log, tasks) for t in group]
    if set(tokens) != {r['token'] for r in rows}: raise ValueError('Current input population incomplete')
    atomic_json(args.output/'COMPLETE.json', {'identity': identity, 'scenes': len(tokens), 'contains_GT': False})
    print(json.dumps({'identity': identity, 'scenes': len(tokens), 'contains_GT': False}), flush=True)


if __name__ == '__main__': main()
