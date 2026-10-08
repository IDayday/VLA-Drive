"""Official nuScenes annotations through the shared UniAD label generator."""
import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import asdict
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.structured_world.build_cache import digest, atomic_json


_NUSC = None
_MAP = None


def initialize(root):
    global _NUSC, _MAP
    import cv2
    import torch
    from nuscenes.nuscenes import NuScenes
    from starVLA.dataloader.structured_world.labels import RoadMap
    torch.set_num_threads(1); cv2.setNumThreads(1)
    _NUSC = NuScenes('v1.0-trainval', dataroot=root, verbose=False)
    _MAP = RoadMap('nuscenes', root)


def build_one(task):
    import torch
    from starVLA.dataloader.structured_world.nuscenes_adapter import adapted_nuscenes_scene, planning_metadata
    from starVLA.dataloader.structured_world.cameras import NUSCENES_CAMERAS
    from starVLA.dataloader.structured_world.labels import build_labels
    row, output, identity = task
    output = Path(output); token = row['token']
    metadata = output/'records'/(token+'.json'); target = output/'labels'/(token+'.npz')
    if metadata.exists() and target.exists():
        data = json.loads(metadata.read_text())
        if data['cache_identity'] != identity or digest(target) != data['label_sha256']:
            raise ValueError('Existing nuScenes cache identity/hash mismatch')
        return data
    sample = _NUSC.get('sample', token)
    scene = adapted_nuscenes_scene(_NUSC, sample)
    before = [f.boxes.tensor.clone() for f in scene.frames]
    labels = build_labels(scene, _MAP)
    if any(not torch.equal(a, f.boxes.tensor) for a, f in zip(before, scene.frames)):
        raise AssertionError('Shared UniAD generator polluted source boxes')
    arrays = {k: v for k, v in labels.items() if isinstance(v, np.ndarray)}
    images = scene.observations['geometry_images']
    arrays['geometry_rgb'] = ((images*images.new_tensor([.229, .224, .225])[None, :, None, None]+
        images.new_tensor([.485, .456, .406])[None, :, None, None])*255.).round().clamp(0, 255).byte().numpy()
    arrays['geometry_pixel_valid'] = scene.observations['geometry_pixel_valid'].numpy()
    arrays.update({f'calibration_{key}': value.numpy() for key, value in scene.observations['calibration'].items()})
    planning = planning_metadata(_NUSC, sample)
    from starVLA.dataloader.structured_world.temporal_validity import mask_auxiliary_time_mismatch
    timing_valid = mask_auxiliary_time_mismatch(arrays, planning['future_times_s'])
    labels['protocol'] = dict(labels['protocol'], auxiliary_time_contract={
        'nominal_times_s': [.5, 1., 1.5, 2., 2.5, 3.], 'maximum_annotation_jitter_s': .06,
        'valid_future_scene_label_times': timing_valid.tolist(),
        'larger_jitter': 'unknown255 for future scene losses only; original VAD ego protocol retained'})
    arrays['original_VAD_lidar_xy'] = planning['original_VAD_lidar_xy']
    arrays['actual_future_times_s'] = np.asarray(planning['future_times_s'], dtype=np.float64)
    temporary = target.with_suffix('.tmp')
    with temporary.open('wb') as stream: np.savez_compressed(stream, **arrays)
    temporary.replace(target)
    current = {'token': token, 'image_paths': [str(Path(_NUSC.dataroot)/_NUSC.get('sample_data', sample['data'][name])['filename'])
               for name in NUSCENES_CAMERAS], 'camera_order': list(NUSCENES_CAMERAS),
               'state': scene.observations['state'].tolist(), 'lang': scene.observations['lang'],
               'navigation': planning['navigation']}
    data = {'token': token, 'log': scene.log, 'dataset': 'nuscenes', 'cache_identity': identity,
        'label_sha256': digest(target), 'label_bytes': target.stat().st_size,
        'current_record': current, 'protocol': labels['protocol'], 'ego_body': labels['ego_body'],
        'height_quality': labels['height_quality'],
        'coverage': {'road': float(arrays['road_valid'].mean()),
                     'occupancy': arrays['occupancy_valid'].mean((1, 2)).tolist(),
                     'depth_pixels': (arrays['depth'] > 0).sum((1, 2)).tolist()}, 'class_counts': {}}
    occupancy, valid = arrays['occupancy'], arrays['occupancy_valid']
    pair = valid[1:] & valid[:1]
    populations = {'current': (occupancy[0] == 1, valid[0]), 'future': (occupancy[1:] == 1, pair),
                   'arrival': (occupancy[1:] == 1, pair & (occupancy[:1] == 0)),
                   'release': (occupancy[1:] == 0, pair & (occupancy[:1] == 1))}
    for name, (positive, mask) in populations.items():
        data['class_counts'][name] = {'positive': int((positive & mask).sum()), 'negative': int((~positive & mask).sum())}
    atomic_json(metadata, data)
    return data


def main():
    from starVLA.model.modules.structured_world.grid import GridSpec
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--population', type=Path, required=True)
    parser.add_argument('--split', choices=['train', 'val'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--debug-scenes', type=int, default=0)
    parser.add_argument('--debug-tokens', type=Path, help='Explicit real-sensor debug subset; never a formal population')
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    population = json.loads((args.population/'identity.json').read_text())
    path = args.population/(args.split+'_scenes.json')
    if digest(path) != population['file_sha256'][args.split]:
        raise ValueError('Foreign population index')
    rows = json.loads(path.read_text())
    if args.debug_tokens:
        if not args.debug_scenes: raise ValueError('Explicit debug tokens require a declared debug size')
        tokens = set(json.loads(args.debug_tokens.read_text()))
        rows = [row for row in rows if row['token'] in tokens]
        if len(rows) != len(tokens) or len(rows) < args.debug_scenes: raise ValueError('Invalid training-only debug subset')
    kind = 'full_train_population' if args.split == 'train' else 'full_validation_population'
    if args.debug_scenes:
        indices = np.random.default_rng(42).choice(len(rows), args.debug_scenes, replace=False)
        rows = [rows[index] for index in sorted(indices)]
        kind = 'fixed_train_debug_subset_not_formal' if args.split == 'train' else 'fixed_validation_debug_subset_not_formal'
    else:
        downloads = json.loads((args.root/'download_manifest.json').read_text())
        from tools.structured_world.download_nuscenes import NAMES
        if set(downloads['archives']) != set(NAMES):
            raise ValueError('Formal cache requires metadata, maps, CAN bus and all ten sensor archives')
        archives = downloads['archives'].values()
        if not all(a.get('extracted') and a.get('sha256') for a in archives):
            raise ValueError('Formal cache requires all official archives verified and extracted')
    code = ['starVLA/dataloader/structured_world/nuscenes_adapter.py',
            'starVLA/dataloader/structured_world/centered_boxes.py', 'tools/structured_world/build_nuscenes_cache.py',
            'starVLA/dataloader/structured_world/temporal_validity.py',
            'starVLA/dataloader/structured_world/adapters.py', 'starVLA/dataloader/structured_world/cameras.py',
            'starVLA/dataloader/structured_world/labels.py', 'third_party/uniad/ported/occflow_label.py',
            'third_party/vad/ported/planning_pose.py', 'third_party/vad/ported/converter.py']
    declaration = {'schema': 'structured_scene_cache_v1', 'dataset': 'nuscenes', 'population_kind': kind,
        'source_current_identity': {'identity': population['identity'], 'split': args.split, 'index_sha256': digest(path)},
        'index': rows, 'grid': asdict(GridSpec()), 'code_hashes': {p: digest(ROOT/p) for p in code},
        'external_sources_sha256': digest(ROOT/'third_party/LOCK.json'),
        'input_root': str(args.root), 'population_protocol': population,
        'label_semantics': 'same derived signed road distance, endpoint environment box occupancy and unknown255 as NAVSIM',
        'auxiliary_annotation_jitter_s': .06,
        'planning_GT_timing': 'unaltered native VAD six annotated keyframes; nominal 2Hz with stored actual times'}
    identity = hashlib.sha256(json.dumps(declaration, sort_keys=True).encode()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output/'identity.json').exists() and json.loads((args.output/'identity.json').read_text())['identity'] != identity:
        raise ValueError('Refusing to overwrite another label contract')
    atomic_json(args.output/'identity.json', {'identity': identity, **declaration})
    atomic_json(args.output/'index.json', rows)
    for name in ('records', 'labels'): (args.output/name).mkdir(exist_ok=True)
    start = time.monotonic(); records, errors = [], []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize, initargs=(str(args.root),),
                             mp_context=multiprocessing.get_context('spawn')) as pool:
        iterator = iter(rows); pending = {}
        def fill():
            while len(pending) < 2*args.workers:
                row = next(iterator, None)
                if row is None: break
                pending[pool.submit(build_one, (row, str(args.output), identity))] = row['token']
        fill()
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for item in done:
                token = pending.pop(item)
                try: records.append(item.result())
                except Exception as error: errors.append({'token': token, 'type': type(error).__name__, 'error': str(error)})
                if (len(records)+len(errors)) % 32 == 0:
                    print(json.dumps({'complete': len(records), 'errors': errors[-3:], 'seconds': time.monotonic()-start}), flush=True)
            fill()
    statistics = {k: {'positive': 0, 'negative': 0} for k in ('current', 'future', 'arrival', 'release')}
    for record in records:
        for key, counts in record['class_counts'].items():
            for name, value in counts.items(): statistics[key][name] += value
    summary = {'identity': identity, 'scenes': len(records), 'expected_scenes': len(rows), 'errors': errors,
        'population_kind': kind, 'class_counts': statistics, 'seconds': time.monotonic()-start,
        'bytes': sum(r['label_bytes'] for r in records),
        'road_coverage_mean': float(np.mean([r['coverage']['road'] for r in records])) if records else None}
    atomic_json(args.output/('COMPLETE.json' if not errors else 'INCOMPLETE.json'), summary)
    print(json.dumps(summary, indent=2), flush=True)
    if errors: raise SystemExit(1)


if __name__ == '__main__': main()
