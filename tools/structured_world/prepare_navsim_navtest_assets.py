"""Build and audit independent Navtest inputs with the frozen current-only builder.

Historical records, RGB, raw logs and reference metric caches remain read-only.
No annotations, future poses, LiDAR payload or map is exported to model inputs.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone

FROZEN = Path('/mnt/project/VLA-Drive-structured-world-fgtr-io-formal-frozen-20261009')
SOURCE_SHA = '0c54f1c49e7ed2a76b22826b654d6c72625f5b6d'
CURRENT = Path('/mnt/project/ddpolicy-vehicle-joint-artifacts/20260928/current_navtest_v1')
INDEX = Path('/mnt/project/local-interaction-mask-v2-artifacts/20260927/navtest_cache_index.json')
LOGS = Path('/mnt/navsim/test_navsim_logs/test')
SENSORS = Path('/mnt/navsim/test_sensor_blobs/test')
DEFAULT_OUTPUT = Path('/mnt/project/structured-world-fgtr-round1-artifacts/20261008/navsim_navtest_current_v1_0c54')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def atomic(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.' + str(os.getpid()) + '.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n'); temp.replace(path)


def authorize():
    policy = json.loads(Path('/mnt/project/server_dispatch_policy.json').read_text())
    allowed = policy['task_authorizations']['structured_world_fgtr_round1']['latest_explicit_resource_hosts']
    if socket.gethostname().removesuffix('-worker-0') not in allowed:
        raise ValueError('Host not in latest five-host authorization')
    if subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=FROZEN, text=True).strip() != SOURCE_SHA:
        raise ValueError('Evaluation source differs from frozen contract')
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=FROZEN, text=True).strip():
        raise ValueError('Frozen source is dirty')


def source_audit(output):
    rows = json.loads(INDEX.read_text())
    current = json.loads((CURRENT / 'index.json').read_text())
    audit = json.loads((CURRENT / 'audit.json').read_text())
    population = {(r['token'], r['log']) for r in rows}
    if len(rows) != 12146 or len(population) != 12146 or len({r['log'] for r in rows}) != 136:
        raise ValueError('Expected full 12146 / 136 Navtest population')
    if len(current) != 12146 or {(r['token'], r['log']) for r in current} != population:
        raise ValueError('Current and metric populations differ')
    if audit['completed'] != 12146 or audit['failures']:
        raise ValueError('Historical current population incomplete')
    for row in rows:
        for path in (Path(row['cache_path']), CURRENT / 'current' / (row['token'] + '.json'), LOGS / (row['log'] + '.pkl')):
            if not path.is_file(): raise FileNotFoundError(path)
    report = {'scenes': 12146, 'logs': 136, 'all_metric_paths_present': True,
        'population_matches_historical_current': True, 'metric_index': str(INDEX),
        'metric_index_sha256': digest(INDEX), 'historical_current': str(CURRENT),
        'historical_current_identity': json.loads((CURRENT / 'identity.json').read_text()),
        'raw_logs': str(LOGS), 'source_RGB': str(SENSORS), 'frozen_source': SOURCE_SHA,
        'historical_assets_mutated': False}
    atomic(output / 'SOURCE_ASSETS_VERIFIED.json', report)


def build(output, workers, debug=False):
    authorize()
    command = [sys.executable, '-B', '-u', str(FROZEN / 'tools/structured_world/build_navsim_current_inputs.py'),
        '--current-root', str(CURRENT), '--log-root', str(LOGS), '--sensor-root', str(SENSORS),
        '--output', str(output), '--workers', str(workers)]
    if debug: command += ['--debug-limit', '4']
    atomic(output.parent / (output.name + '_BUILD_COMMAND.json'), {'command': command})
    subprocess.run(command, cwd=FROZEN, check=True)


def validate(output):
    import numpy as np
    from pyquaternion import Quaternion
    sys.path.insert(0, str(FROZEN))
    from starVLA.dataloader.structured_world.adapters import pose_matrix, navigation_text
    from starVLA.dataloader.structured_world.cameras import NAVSIM_CAMERAS
    from starVLA.dataloader.foresight_dataset import encode_ego
    identity = json.loads((output / 'identity.json').read_text())
    done = json.loads((output / 'COMPLETE.json').read_text())
    rows = json.loads((output / 'index.json').read_text())
    metric = json.loads(INDEX.read_text())
    if identity['population_kind'] != 'navtest' or done['scenes'] != 12146 or done['contains_GT']:
        raise ValueError('Incomplete or non-Navtest inputs')
    if {(r['token'], r['log']) for r in rows} != {(r['token'], r['log']) for r in metric}:
        raise ValueError('Model inputs and canonical scoring population differ')
    if {p.stem for p in (output / 'current').glob('*.json')} != {r['token'] for r in rows}:
        raise ValueError('Current record population differs')
    if {p.stem for p in (output / 'pixels').glob('*.npz')} != {r['token'] for r in rows}:
        raise ValueError('Pixel population differs')
    by_log = {}
    for row in rows: by_log.setdefault(row['log'], []).append(row)
    checked = 0; image_bytes = 0
    for log, values in by_log.items():
        with (LOGS / (log + '.pkl')).open('rb') as stream: frames = pickle.load(stream)
        positions = {f['token']: i for i, f in enumerate(frames)}
        for row in values:
            token = row['token']; pos = positions[token]; frame = frames[pos]
            saved = json.loads((output / 'current' / (token + '.json')).read_text())
            old = json.loads((CURRENT / 'current' / (token + '.json')).read_text())
            if set(saved) != {'identity', 'token', 'image_paths', 'image_sha256', 'state', 'lang', 'pixels_sha256'}:
                raise ValueError('Forbidden input record fields')
            if saved['identity'] != identity['identity'] or saved['token'] != token: raise ValueError('Foreign current record')
            if pos < 3: raise ValueError('Insufficient legal pose history')
            history = np.asarray(old['global_pose_history'])
            raw_history = np.array([[*f['ego2global_translation'][:2],
                float(Quaternion(*f['ego2global_rotation']).yaw_pitch_roll[0])]
                for f in frames[pos - 3:pos + 1]])
            if not np.allclose(history[:, :2], raw_history[:, :2], atol=1e-7, rtol=0): raise ValueError('History position differs from <=t0 raw poses')
            if not np.allclose((history[:, 2] - raw_history[:, 2] + np.pi) % (2*np.pi) - np.pi, 0, atol=1e-7):
                raise ValueError('History heading differs from <=t0 raw poses')
            e2g = pose_matrix(frame['ego2global_translation'], frame['ego2global_rotation'], planar=True)
            attitude = np.linalg.inv(e2g) @ pose_matrix(frame['ego2global_translation'], frame['ego2global_rotation'])
            l2e = pose_matrix(frame['lidar2ego_translation'], frame['lidar2ego_rotation'])
            delta = (history[-1, :2] - history[-2, :2]) @ e2g[:2, :2]
            yaw = (history[-1, 2] - history[-2, 2] + np.pi) % (2*np.pi) - np.pi
            if not np.array_equal(np.asarray(saved['state'], dtype=np.float32), encode_ego([[*delta, yaw]])):
                raise ValueError('Encoded state differs from legal previous/current delta')
            if saved['lang'] != navigation_text(old['navigation']): raise ValueError('Navigation differs')
            pixel = output / 'pixels' / (token + '.npz')
            if digest(pixel) != saved['pixels_sha256']: raise ValueError('Pixel hash differs')
            with np.load(pixel, allow_pickle=False) as arrays:
                if set(arrays.files) != {'geometry_rgb', 'geometry_pixel_valid', 'calibration_sensor2ego',
                    'calibration_intrinsics', 'calibration_post_rots', 'calibration_post_trans'}: raise ValueError('Forbidden pixel fields')
                sensors = []; intrinsics = []
                for k, name in enumerate(NAVSIM_CAMERAS):
                    cam = frame['cams'][name]; c2l = np.eye(4)
                    c2l[:3, :3] = cam['sensor2lidar_rotation']; c2l[:3, 3] = cam['sensor2lidar_translation']
                    sensors.append(attitude @ l2e @ c2l); intrinsics.append(cam['cam_intrinsic'])
                    path = SENSORS / cam['data_path']
                    if str(path) != saved['image_paths'][k] or digest(path) != saved['image_sha256'][k]:
                        raise ValueError('Camera order/source bytes differ')
                    image_bytes += path.stat().st_size
                if not np.array_equal(arrays['calibration_sensor2ego'], np.asarray(sensors, dtype=np.float32)):
                    raise ValueError('Calibration differs from planar rear-axle current frame')
                if not np.array_equal(arrays['calibration_intrinsics'], np.asarray(intrinsics, dtype=np.float32)):
                    raise ValueError('Intrinsic calibration differs')
                if arrays['geometry_rgb'].shape != (3, 3, 256, 448) or arrays['geometry_rgb'].dtype != np.uint8:
                    raise ValueError('Geometry RGB shape/precision differs')
                if arrays['geometry_pixel_valid'].shape != (3, 256, 448): raise ValueError('Pixel mask shape differs')
            checked += 1
        atomic(output / 'VALIDATION_PROGRESS.json', {'checked': checked, 'total': 12146, 'logs_checked': list(by_log).index(log) + 1})
    atomic(output / 'ASSETS_VERIFIED.json', {'passed': True, 'observed_utc': datetime.now(timezone.utc).isoformat(),
        'input_identity': identity['identity'], 'scenes': checked, 'logs': 136, 'current_RGB_images': 3 * checked,
        'source_RGB_bytes': image_bytes, 'current_record_and_pixel_hashes_verified': True,
        'all_calibrations_and_legal_pose_history_verified': True, 'cameras': list(NAVSIM_CAMERAS),
        'axis': 'planar rear axle at t0: x forward, y left, yaw counterclockwise; matches frozen trainer and NAVSIM scorer',
        'state': 'previous/current global pose displacement in t0 planar ego, normalized encode_ego; no explicit future/acceleration/CANbus',
        'frozen_builder_sha256': digest(FROZEN / 'tools/structured_world/build_navsim_current_inputs.py'),
        'geometry_RGB': 'same distortion rectification, crop, 256x448 resize and pixel-valid mask as frozen trainer',
        'Qwen_RGB': 'original current image hashes bound; frozen ROI and 1024x576 resize',
        'metric_index': str(INDEX), 'metric_index_sha256': digest(INDEX), 'historical_assets_mutated': False})


def audit_headings(output):
    """Audit source heading precision without rebuilding or rewriting inputs."""
    import numpy as np
    from pyquaternion import Quaternion
    sys.path.insert(0, str(FROZEN))
    from starVLA.dataloader.structured_world.adapters import pose_matrix
    from starVLA.dataloader.foresight_dataset import encode_ego
    reports = {}
    roots = [('navtest_full', CURRENT),
        ('dev_full', CURRENT.parent / 'current_dev_v1'),
        ('train_smallfit64_sample', CURRENT.parent / 'current_smallfit64_v1')]
    def summary(values):
        v = np.asarray(values)
        return dict(count=len(v), max=float(v.max()), mean=float(v.mean()),
            p50=float(np.quantile(v, .5)), p99=float(np.quantile(v, .99)),
            above_1e_7=int((v > 1e-7).sum()))
    for name, root in roots:
        source = json.loads((root / 'identity.json').read_text())
        rows = json.loads((root / 'index.json').read_text())
        groups = {}
        for row in rows: groups.setdefault(row['log'], []).append(row)
        direct_errors = []; planar_errors = []; float32_errors = []; full_rotation_errors = []; states = 0
        for log, values in groups.items():
            with (Path(source['raw_log_root']) / (log + '.pkl')).open('rb') as stream: frames = pickle.load(stream)
            positions = {f['token']: i for i, f in enumerate(frames)}
            headings = {i: float(Quaternion(*f['ego2global_rotation']).yaw_pitch_roll[0]) for i, f in enumerate(frames)}
            for row in values:
                pos = positions[row['token']]
                if pos < 3: raise ValueError('Heading audit lacks legal history')
                old = json.loads((root / 'current' / (row['token'] + '.json')).read_text())
                history = np.asarray(old['global_pose_history'])
                raw = np.array([[*frames[i]['ego2global_translation'][:2], headings[i]] for i in range(pos-3, pos+1)])
                direct = np.abs((history[:, 2] - raw[:, 2] + np.pi) % (2*np.pi) - np.pi)
                direct_errors.extend(direct.tolist())
                for j, i in enumerate(range(pos-3, pos+1)):
                    f = frames[i]
                    for planar, bank in ((True, planar_errors), (False, full_rotation_errors)):
                        matrix = pose_matrix(f['ego2global_translation'], f['ego2global_rotation'], planar=planar)
                        yaw = np.arctan2(matrix[1, 0], matrix[0, 0])
                        bank.append(float(abs((history[j, 2] - yaw + np.pi) % (2*np.pi) - np.pi)))
                        if planar:
                            cast = matrix.astype(np.float32)
                            yaw32 = np.arctan2(cast[1, 0], cast[0, 0])
                            float32_errors.append(float(abs((history[j, 2] - yaw32 + np.pi) % (2*np.pi) - np.pi)))
                if not np.allclose(history[:, :2], raw[:, :2], atol=1e-7, rtol=0) or np.any(direct > 1e-7):
                    raise ValueError('Direct canonical raw history differs: ' + name + '/' + row['token'])
                if name == 'navtest_full':
                    saved = json.loads((output / 'current' / (row['token'] + '.json')).read_text())
                    e2g = pose_matrix(frames[pos]['ego2global_translation'], frames[pos]['ego2global_rotation'], planar=True)
                    delta = (raw[-1, :2] - raw[-2, :2]) @ e2g[:2, :2]
                    yaw = (raw[-1, 2] - raw[-2, 2] + np.pi) % (2*np.pi) - np.pi
                    if not np.array_equal(np.asarray(saved['state'], dtype=np.float32), encode_ego([[*delta, yaw]])):
                        raise ValueError('Raw-direct ego state differs from frozen train interface')
                    states += 1
        reports[name] = dict(scenes=len(rows), logs=len(groups), source_identity=source['identity'],
            direct_Quaternion_double=summary(direct_errors), planar_matrix_float32_roundtrip=summary(float32_errors),
            actual_planar_matrix_roundtrip=summary(planar_errors), actual_pose_matrix_dtype=str(matrix.dtype),
            full_rotation_projected_yaw=summary(full_rotation_errors), bitwise_raw_direct_state_verified=states)
    atomic(output / 'HEADING_VERIFIED.json', dict(passed=True, observed_utc=datetime.now(timezone.utc).isoformat(),
        input_identity=json.loads((output/'identity.json').read_text())['identity'], tolerance=1e-7,
        canonical_reference='float(Quaternion(*rotation).yaw_pitch_roll[0]), identical to historical preparer; no float32 matrix roundtrip',
        populations=reports, assets_rebuilt=False, tolerance_relaxed=False,
        training_adapter_sha256=digest(FROZEN/'starVLA/dataloader/structured_world/adapters.py'),
        frozen_builder_sha256=digest(FROZEN/'tools/structured_world/build_navsim_current_inputs.py'),
        historical_preparer_sha256=digest(Path('/mnt/project/VLA-Drive-action-video-source-1493ded/tools/ddpolicy_vehicle/prepare_current.py'))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--validate-only', action='store_true')
    parser.add_argument('--heading-only', action='store_true')
    parser.add_argument('--wait-for-built', action='store_true', help='Wait for an existing builder to finish, then audit only')
    args = parser.parse_args(); authorize(); args.output.mkdir(parents=True, exist_ok=True)
    if args.heading_only:
        audit_headings(args.output); return
    source_audit(args.output)
    if args.wait_for_built:
        while not (args.output / 'COMPLETE.json').exists(): time.sleep(10)
    elif not args.validate_only:
        debug = args.output.parent / (args.output.name + '_memory4')
        if not (debug / 'COMPLETE.json').exists(): build(debug, min(2, args.workers), True)
        if not (args.output / 'COMPLETE.json').exists(): build(args.output, args.workers)
    validate(args.output)
    audit_headings(args.output)


if __name__ == '__main__': main()
