"""CPU-only preparation of immutable, token-aligned planning label overlays."""
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def identity_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def released_labels(campaign):
    """Use the completed current campaign, never a latest/previous fallback."""
    campaign = Path(campaign).resolve()
    output = campaign / 'final'
    lineage = {'campaign': str(campaign)}
    requirement = campaign / 'required_geometry_release.json'
    if requirement.exists():
        release_path = Path(read(requirement)['release_file'])
        if not release_path.is_file() or read(release_path).get('complete') is not True:
            raise ValueError('Trajectory geometry refinement is not released yet: ' + str(release_path))
        release = read(release_path)
        if release.get('accepted_geometry_failures') != 0:
            raise ValueError('Released labels still contain accepted geometry failures')
        output = Path(release['final_output']).resolve()
        for path, expected in (
            (campaign / 'final/trajectories.npz', release['source_trajectories_sha256']),
            (output / 'trajectories.npz', release['output_trajectories_sha256']),
        ):
            if digest(path) != expected:
                raise ValueError('Stale or modified geometry release: ' + str(path))
        lineage.update(geometry_release=release, geometry_release_sha256=digest(release_path),
                       geometry_requirement_sha256=digest(requirement))
    summary = read(output / 'summary.json')
    if summary.get('complete') is not True or summary.get('missing_count', 0) or summary.get('error_count', 0):
        raise ValueError('Trajectory optimization output is incomplete')
    manifest = read(output / 'manifest.json')
    if not manifest.get('run_identity') or summary.get('run_identity') != manifest['run_identity']:
        raise ValueError('Trajectory summary/manifest identity mismatch')
    lineage.update(output=str(output), run_identity=manifest['run_identity'],
                   summary_sha256=digest(output / 'summary.json'),
                   manifest_sha256=digest(output / 'manifest.json'),
                   trajectories_sha256=digest(output / 'trajectories.npz'))
    hashes = output / 'artifact_hashes.json'
    if hashes.exists():
        expected = read(hashes)
        for name in ('manifest.json', 'trajectories.npz'):
            if expected.get(name) != digest(output / name):
                raise ValueError('Published artifact hash mismatch: ' + name)
    return output, manifest, summary, lineage


def training_population(data):
    data = Path(data)
    current, ego = read(data / 'identity.json'), read(data / 'ego_identity.json')
    index, done = read(data / 'index.json'), read(data / 'COMPLETE.json')
    tokens = [row['token'] for row in index]
    if (not tokens or len(set(tokens)) != len(tokens) or
        any(not isinstance(t, str) or not t or Path(t).name != t or t in ('.', '..') for t in tokens)):
        raise ValueError('Empty, duplicate or invalid training tokens')
    if (current.get('schema') != 'ddpolicy_current_cameras_v1' or current.get('split') != 'train'
        or ego.get('schema') != 'foresight_ego_labels_v1' or ego.get('split') != 'train'
        or current['index_sha256'] != identity_hash(index) or ego['index_sha256'] != current['index_sha256']
        or done.get('scenes') != len(tokens) or done.get('failures') != 0
        or done.get('current_identity') != current['identity'] or done.get('ego_identity') != ego['identity']):
        raise ValueError('Only a complete, unchanged training population is supported')
    for ident in (current, ego):
        if identity_hash({k: v for k, v in ident.items() if k != 'identity'}) != ident['identity']:
            raise ValueError('Source data identity hash mismatch')
    return current, ego, index


def prepare(campaign, data, destination, expected_scenes=103288):
    output, manifest, summary, lineage = released_labels(campaign)
    current, ego, index = training_population(data)
    with np.load(output / 'trajectories.npz', allow_pickle=False) as bank:
        tokens, poses, accepted = (bank[k].copy() for k in ('tokens', 'trajectories', 'accepted'))
    if digest(output / 'trajectories.npz') != lineage['trajectories_sha256']:
        raise ValueError('Trajectory source changed while preparing labels')
    if (tokens.ndim != 1 or tokens.dtype.kind != 'U' or len(set(tokens.tolist())) != len(tokens)
        or poses.shape != (len(tokens), 8, 3) or poses.dtype.kind != 'f' or not np.isfinite(poses).all()
        or accepted.shape != tokens.shape or accepted.dtype != np.bool_):
        raise ValueError('Invalid trajectory bank: require unique tokens, finite [N,8,3] and boolean accepted')
    if (len(tokens) != expected_scenes or len(manifest['tokens']) != len(tokens)
        or set(manifest['tokens']) != set(tokens.tolist()) or summary.get('exported_scenes') != len(tokens)
        or summary.get('expected_scenes') != len(tokens)):
        raise ValueError('Incomplete source token coverage')
    lookup = {token: i for i, token in enumerate(tokens.tolist())}
    missing = [row['token'] for row in index if row['token'] not in lookup]
    if missing:
        raise ValueError(f'{len(missing)} training tokens missing from trajectory bank: {missing[:5]}')
    at = np.asarray([lookup[row['token']] for row in index])
    # Coordinate contract of navsim_robust_labels: ego(t0) rear axle, meters,
    # radians, eight absolute poses at 0.5..4.0s. encode_ego happens at loading.
    selected = poses[at].astype(np.float32)
    if not np.isfinite(selected).all():
        raise ValueError('Trajectory values overflow float32')
    ident = {'schema': 's3_optimized_ego_overlay_v1', 'split': 'train',
             'current_identity': current['identity'], 'original_ego_identity': ego['identity'],
             'index_sha256': current['index_sha256'], 'partition_sha256': current['partition_sha256'],
             'source': lineage, 'source_scenes': len(tokens), 'scenes': len(index),
             'excluded_source_scenes': len(tokens) - len(index), 'accepted_scenes': int(accepted[at].sum()),
             'fallback_scenes': int((~accepted[at]).sum()),
             'policy': 'accepted optimized planning target; otherwise original ego target; preserve every train token',
             'video_condition': 'original_logged_ego', 'frame': 'local_rear_axle',
             'times_s': [.5 * i for i in range(1, 9)], 'fields': ['x_m', 'y_m', 'yaw_rad']}
    destination = Path(destination).resolve()
    if destination.exists():
        saved, _, _ = load_overlay(destination, current, ego, index)
        if {k: v for k, v in saved.items() if k not in ('identity', 'labels_sha256')} != ident:
            raise ValueError('Existing overlay has a different source; use a new output directory')
        return saved
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix='.' + destination.name + '-', dir=destination.parent))
    try:
        np.savez_compressed(temp / 'labels.npz', tokens=tokens[at], trajectories=selected, accepted=accepted[at])
        ident['labels_sha256'] = digest(temp / 'labels.npz')
        ident['identity'] = identity_hash(ident)
        write(temp / 'identity.json', ident)
        write(temp / 'COMPLETE.json', {'identity': ident['identity'], 'scenes': len(index)})
        temp.rename(destination)
    finally:
        if temp.exists():
            shutil.rmtree(temp)
    return ident


def load_overlay(root, current, ego, index):
    root = Path(root)
    ident, done = read(root / 'identity.json'), read(root / 'COMPLETE.json')
    if (ident.get('schema') != 's3_optimized_ego_overlay_v1' or ident.get('split') != 'train'
        or current.get('split') != 'train' or ident.get('video_condition') != 'original_logged_ego'
        or ident.get('frame') != 'local_rear_axle' or ident.get('times_s') != [.5 * i for i in range(1, 9)]
        or ident['current_identity'] != current['identity'] or ident['original_ego_identity'] != ego['identity']
        or ident['index_sha256'] != identity_hash(index) or ident['partition_sha256'] != current['partition_sha256']
        or done != {'identity': ident['identity'], 'scenes': len(index)} or ident['scenes'] != len(index)
        or identity_hash({k: v for k, v in ident.items() if k != 'identity'}) != ident['identity']
        or digest(root / 'labels.npz') != ident['labels_sha256']):
        raise ValueError('Optimized label identity, training population or content changed')
    with np.load(root / 'labels.npz', allow_pickle=False) as bank:
        tokens, poses, accepted = (bank[k].copy() for k in ('tokens', 'trajectories', 'accepted'))
    if (tokens.tolist() != [row['token'] for row in index] or poses.shape != (len(index), 8, 3)
        or poses.dtype != np.float32 or not np.isfinite(poses).all()
        or accepted.dtype != np.bool_ or accepted.shape != (len(index),)
        or int(accepted.sum()) != ident['accepted_scenes'] or int((~accepted).sum()) != ident['fallback_scenes']):
        raise ValueError('Invalid aligned optimized trajectory labels')
    return ident, poses, accepted
