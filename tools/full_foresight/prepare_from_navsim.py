"""Build current observations, ego labels and vehicle MAE data from raw NAVSIM.

No processed scene PKLs, old model outputs or teacher weights are required.
The existing ego normalization, vehicle selection and track geometry are reused.
"""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import fcntl
import json
from pathlib import Path
import subprocess
import time

import numpy as np
from PIL import Image
from pyquaternion import Quaternion
import torch

from tools.ddpolicy_vehicle.prepare_data import CAMERAS, atomic_json, camera_path, load_trusted
from tools.ddpolicy_vehicle.evaluate_ego import relative_ego_target
from tools.ddpolicy_vehicle.run_meter import metered_run
from tools.foresight.prepare_vehicle_trajectories import make_record, timed_frames, SCHEMA
from starVLA.dataloader.foresight_dataset import encode_ego
from starVLA.model.modules.structured_world.geometry import crop_resize_intrinsics
from starVLA.model.modules.vehicle_joint.graphs import VehicleGraphConfig
from starVLA.model.modules.vehicle_joint.initialization import file_sha256, identity_hash


def pose(frame):
    xy = np.asarray(frame['ego2global_translation'], dtype=np.float64)[:2]
    yaw = Quaternion(*frame['ego2global_rotation']).yaw_pitch_roll[0]
    result = [float(xy[0]), float(xy[1]), float(yaw)]
    if not np.isfinite(result).all():
        raise ValueError('Invalid ego pose')
    return result


def current_and_ego(frames, at, log, config, current_identity, ego_identity):
    if at < 3 or at + 8 >= len(frames):
        raise ValueError('Original DDP needs four observed and eight future ego poses')
    frame = frames[at]
    timestamp = int(frame['timestamp'])
    history = frames[at - 3:at + 1]
    if any(int(f['timestamp']) > timestamp for f in history):
        raise ValueError('Future frame in current ego history')
    paths, intrinsics, extrinsics, distortion = [], [], [], []
    for camera in CAMERAS:
        calibration = frame['cams'][camera]
        image_path = camera_path(calibration['data_path'], config)
        with Image.open(image_path) as image:
            k, _ = crop_resize_intrinsics(calibration['cam_intrinsic'], image.size, (1024, 576))
        e = np.eye(4)
        e[:3, :3] = calibration['sensor2lidar_rotation']
        e[:3, 3] = calibration['sensor2lidar_translation']
        paths.append(str(image_path.resolve()))
        intrinsics.append(k)
        extrinsics.append(e)
        distortion.append(calibration['distortion'])
    calibration = {'intrinsics': np.asarray(intrinsics),
                   'extrinsics': np.asarray(extrinsics), 'distortion': np.asarray(distortion)}
    current = {'identity': current_identity, 'token': frame['token'], 'log': log,
               'timestamp': timestamp, 'image_paths': paths,
               'global_pose_history': [pose(f) for f in history],
               'navigation': int(np.asarray(frame['driving_command']).argmax()),
               'ego_speed': float(np.linalg.norm(frame['ego_dynamic_state'][:2])),
               'current_calibration': {key: value.tolist() for key, value in calibration.items()}}
    if not np.isfinite(current['ego_speed']) or current['navigation'] not in (0, 1, 2, 3):
        raise ValueError('Invalid current ego state/navigation')
    # Future poses are read only after the complete current-input whitelist exists.
    global_poses = np.asarray([pose(f) for f in frames[at - 3:at + 9]], dtype=np.float64)
    ego = torch.from_numpy(encode_ego(relative_ego_target(global_poses)))
    target = {'identity': ego_identity, 'token': frame['token'], 'ego': ego}
    return current, target, calibration


def save_tensor(path, payload):
    temporary = path.with_suffix('.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)


def process_log(job):
    log, scenes, config = job
    torch.set_num_threads(1)
    output = Path(config['output'])
    raw_path = Path(config['raw_log_root']) / (log + '.pkl')
    digest = file_sha256(raw_path)
    report_path = output / 'logs' / (log + '.json')
    if report_path.exists():
        report = json.loads(report_path.read_text())
        if report['preparation_identity'] != config['identity'] or report['raw_log_sha256'] != digest:
            raise ValueError('Existing preparation/log content changed')
        for split, token in scenes:
            paths = [output / f'student_{split}_v1/current' / (token + '.json'),
                     output / f'student_{split}_v1/ego' / (token + '.pt'),
                     output / 'teacher_data_full_v1/records' / (token + '.pt')]
            if not all(path.is_file() for path in paths):
                raise ValueError('Completed preparation record is missing')
            if [file_sha256(path) for path in paths] != report['record_hashes'][token]:
                raise ValueError('Completed preparation record changed')
        return report
    frames = load_trusted(raw_path)
    positions = {f['token']: i for i, f in enumerate(frames)}
    if len(positions) != len(frames):
        raise ValueError('Duplicate raw frame token')
    rows, hashes = [], {}
    graph = VehicleGraphConfig(**config['graph'])
    for split, token in scenes:
        at = positions[token]
        current_id = config['student_identities'][split]['identity']
        ego_id = config['ego_identities'][split]['identity']
        current, ego, calibration = current_and_ego(frames, at, log, config, current_id, ego_id)
        future = timed_frames(frames, at, [.5 * (i + 1) for i in range(8)], .05)
        record, audit = make_record(frames[at], future, calibration, graph)
        student = output / f'student_{split}_v1'
        atomic_json(student / 'current' / (token + '.json'), current)
        save_tensor(student / 'ego' / (token + '.pt'), ego)
        save_tensor(output / 'teacher_data_full_v1/records' / (token + '.pt'),
                    {'schema': SCHEMA, 'identity': config['teacher_identity']['identity'],
                     'token': token, 'log': log, 'timestamp': current['timestamp'],
                     'record': record, 'audit': audit})
        hashes[token] = [file_sha256(path) for path in (
            student / 'current' / (token + '.json'), student / 'ego' / (token + '.pt'),
            output / 'teacher_data_full_v1/records' / (token + '.pt'))]
        rows.append({'token': token, 'log': log, 'split': split,
                     **{key: value for key, value in audit.items()
                        if key not in ('graph_audit', 'track_ids_for_audit_only')}})
    if file_sha256(raw_path) != digest:
        raise ValueError('Raw log changed during preparation')
    report = {'preparation_identity': config['identity'], 'log': log,
              'raw_log_sha256': digest, 'records': rows, 'record_hashes': hashes}
    atomic_json(report_path, report)
    return report


def validate_partition(partition):
    logs = partition['token_logs']
    indices = {}
    for split in ('train', 'dev'):
        tokens = partition[split + '_tokens']
        if not tokens or len(set(tokens)) != len(tokens):
            raise ValueError('Empty/duplicate scene population')
        indices[split] = [{'token': token, 'log': logs[token]} for token in tokens]
    if (set(partition['train_tokens']) & set(partition['dev_tokens']) or
            {r['log'] for r in indices['train']} & {r['log'] for r in indices['dev']}):
        raise ValueError('Training/development token or log leakage')
    return indices


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--split-manifest', default='reports/ddpolicy_vehicle_from_scratch/NAVTRAIN_PARTITION.json')
    for key in ('raw-log-root', 'sensor-root', 'output', 'campaign-root', 'run-id'):
        parser.add_argument('--' + key, required=True)
    parser.add_argument('--fallback-sensor-root', action='append', default=[])
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.workers <= 16:
        raise ValueError('Use 1–16 explicit CPU workers')
    if not Path(args.raw_log_root).is_dir() or not Path(args.sensor_root).is_dir():
        raise ValueError('Existing raw log and camera directories are required')
    if subprocess.check_output(['git', 'status', '--porcelain']).strip():
        raise ValueError('Lock a clean source before preparing training assets')
    partition = json.loads(Path(args.split_manifest).read_text())
    indices = validate_partition(partition)
    output = Path(args.output)
    definition = {'schema': 'ddp_full_raw_navsim_preparation_v1',
                  'source_sha': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                  'writer_sha256': file_sha256(__file__),
                  'partition_sha256': file_sha256(args.split_manifest),
                  'raw_log_root': str(Path(args.raw_log_root).resolve()),
                  'sensor_root': str(Path(args.sensor_root).resolve()),
                  'fallback_sensor_root': [str(Path(root).resolve()) for root in args.fallback_sensor_root],
                  'graph': asdict(VehicleGraphConfig(max_vehicles=16, max_context=0)),
                  'time_tolerance_s': .05, 'time_points_s': [.5 * (i + 1) for i in range(8)],
                  'helper_hashes': {name: file_sha256(name) for name in (
                      'tools/foresight/prepare_vehicle_trajectories.py',
                      'tools/ddpolicy_vehicle/evaluate_ego.py',
                      'starVLA/model/modules/vehicle_joint/graphs.py',
                      'starVLA/model/modules/structured_world/geometry.py')}}
    definition['identity'] = identity_hash(definition)
    if output.exists():
        if not args.resume or json.loads((output / 'identity.json').read_text()) != definition:
            raise ValueError('Existing output requires exact-identity resume')
    else:
        output.mkdir(parents=True)
        (output / 'logs').mkdir()
        atomic_json(output / 'identity.json', definition)
    lock = (output / 'RUN.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    observation_id = identity_hash({'preparation': definition['identity'], 'current_calibration_only': True})
    students, egos = {}, {}
    for split, index in indices.items():
        students[split] = {'schema': 'ddpolicy_current_cameras_v1', 'split': split,
                           'index_sha256': identity_hash(index), 'partition_sha256': definition['partition_sha256'],
                           'writer_sha256': definition['writer_sha256'],
                           'source_observation_identity': observation_id,
                           'cameras': list(CAMERAS), 'current_only': True, 'history_images': 0}
        egos[split] = {'schema': 'foresight_ego_labels_v1', 'split': split,
                       'index_sha256': identity_hash(index), 'raw_preparation': definition['identity'],
                       'normalization': 'original DDP1225 xy absolute ego(t0),sincos yaw'}
        for value in (students[split], egos[split]):
            value['identity'] = identity_hash(value)
        student = output / f'student_{split}_v1'
        (student / 'current').mkdir(parents=True, exist_ok=True)
        (student / 'ego').mkdir(exist_ok=True)
        atomic_json(student / 'identity.json', students[split])
        atomic_json(student / 'ego_identity.json', egos[split])
        atomic_json(student / 'index.json', index)
    teacher = {'schema': SCHEMA, 'raw_preparation': definition['identity'],
               'split_hash': definition['partition_sha256'], 'graph': definition['graph'],
               'time_tolerance_s': .05, 'time_points_s': definition['time_points_s'],
               'source_code_hash': definition['writer_sha256'], 'observation_identity': observation_id,
               'limit': 0}
    teacher['identity'] = identity_hash(teacher)
    teacher_root = output / 'teacher_data_full_v1'
    (teacher_root / 'records').mkdir(parents=True, exist_ok=True)
    atomic_json(teacher_root / 'identity.json', teacher)
    for split, index in indices.items():
        atomic_json(teacher_root / (split + '_index.json'), index)
    groups = defaultdict(list)
    for split, index in indices.items():
        for row in index:
            groups[row['log']].append((split, row['token']))
    config = {**definition, 'output': str(output), 'student_identities': students,
              'ego_identities': egos, 'teacher_identity': teacher}
    attempt = args.run_id + '_attempt_' + str(time.time_ns())
    rows = []
    with metered_run(args.campaign_root, attempt, 0,
                     {'kind': 'raw_navsim_training_asset_preparation', 'run_id_parent': args.run_id,
                      'source_sha': definition['source_sha']}) as (meter, _, save):
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for report in pool.map(process_log, [(log, scenes, config) for log, scenes in groups.items()]):
                rows.extend(report['records'])
                meter['prepared_scenes'] = len(rows)
                save()
                atomic_json(output / 'progress.json', {'completed': len(rows),
                            'requested': sum(map(len, indices.values()))})
        for split, index in indices.items():
            atomic_json(output / f'student_{split}_v1/COMPLETE.json',
                        {'scenes': len(index), 'logs': len({r['log'] for r in index}), 'failures': 0,
                         'current_identity': students[split]['identity'], 'ego_identity': egos[split]['identity']})
        numeric = ('source_current_objects', 'source_vehicles', 'selected_vehicles',
                   'selected_valid_future_vehicles', 'ego_only', 'vehicle_future_points')
        audit = {'identity': teacher['identity'], 'scenes': len(rows), 'failures': 0,
                 'population': {key: sum(row[key] for row in rows) for key in numeric}}
        atomic_json(teacher_root / 'audit.json', audit)
        atomic_json(teacher_root / 'COMPLETE.json', {'identity': teacher['identity'], 'scenes': len(rows), 'failed': 0})
        atomic_json(output / 'COMPLETE.json', {'identity': definition['identity'],
                    'scenes': {split: len(index) for split, index in indices.items()},
                    'train_dev_log_overlap': 0, 'failures': 0, 'processed_scene_pkls_required': False})
    print(json.dumps({'output': str(output), 'identity': definition['identity'], 'scenes': len(rows)}))


if __name__ == '__main__':
    main()
