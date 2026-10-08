"""Build independently hashed labels/current geometry pixels from real data."""
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


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True)+'\n')
    tmp.replace(path)


def build_one(task):
    # Cache generation has no model, planner, teacher forcing or GPU input.
    from starVLA.dataloader.structured_world.adapters import navsim_scene
    from starVLA.dataloader.structured_world.labels import RoadMap, build_labels
    from starVLA.model.modules.structured_world.grid import GridSpec
    import torch
    import cv2
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    record, root, identity = task
    root = Path(root)
    token = record['token']
    metadata = root/'records'/(token+'.json')
    target = root/'labels'/(token+'.npz')
    if metadata.exists() and target.exists():
        existing = json.loads(metadata.read_text())
        if existing['cache_identity'] != identity or digest(target) != existing['label_sha256']:
            raise ValueError('Existing label cache identity/hash mismatch')
        return existing
    scene = navsim_scene(record)
    before = [None if f is None else f.boxes.tensor.clone() for f in scene.frames]
    labels = build_labels(scene, RoadMap('navsim', '/mnt/navsim/maps'))
    if any(not torch.equal(a, b.boxes.tensor) for a, b in zip(before, scene.frames) if b is not None):
        raise AssertionError('In-place annotation pollution')
    arrays = {k: v for k, v in labels.items() if isinstance(v, np.ndarray)}
    images = scene.observations['geometry_images']
    # Lossless current-only geometry pixels are stored as uint8, avoiding large
    # normalized float32 image files. Dataset normalization is the exact inverse.
    rgb = ((images*images.new_tensor([.229, .224, .225])[None, :, None, None]+
             images.new_tensor([.485, .456, .406])[None, :, None, None])*255.).round().clamp(0, 255).byte()
    arrays['geometry_rgb'] = rgb.numpy()
    arrays['geometry_pixel_valid'] = scene.observations['geometry_pixel_valid'].numpy()
    arrays.update({f'calibration_{k}': v.numpy() for k, v in scene.observations['calibration'].items()})
    temporary = target.with_suffix('.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, target)
    data = {'token': token, 'log': scene.log, 'dataset': scene.dataset, 'cache_identity': identity,
            'label_sha256': digest(target), 'label_bytes': target.stat().st_size,
            'current_record': record, 'protocol': labels['protocol'], 'ego_body': labels['ego_body'],
            'height_quality': labels['height_quality'],
            'coverage': {'road': float(arrays['road_valid'].mean()),
                         'occupancy': arrays['occupancy_valid'].mean((1, 2)).tolist(),
                         'depth_pixels': (arrays['depth'] > 0).sum((1, 2)).tolist()},
            'class_counts': {}}
    O, valid = arrays['occupancy'], arrays['occupancy_valid']
    pair = valid[1:] & valid[:1]
    populations = {'current': (O[0] == 1, valid[0]), 'future': (O[1:] == 1, pair),
                   'arrival': (O[1:] == 1, pair & (O[:1] == 0)),
                   'release': (O[1:] == 0, pair & (O[:1] == 1))}
    for name, (positive, mask) in populations.items():
        data['class_counts'][name] = {'positive': int((positive & mask).sum()), 'negative': int((~positive & mask).sum())}
    atomic_json(metadata, data)
    return data


def main():
    from starVLA.model.modules.structured_world.grid import GridSpec
    parser = argparse.ArgumentParser()
    parser.add_argument('--current-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--debug-scenes', type=int, default=0)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    original = json.loads((args.current_root/'index.json').read_text())
    rows = original
    kind = 'full_train_population'
    if args.debug_scenes:
        chosen = np.random.default_rng(42).choice(len(rows), args.debug_scenes, replace=False)
        rows = [rows[i] for i in sorted(chosen)]
        kind = 'fixed_train_debug_subset_not_formal'
    code = ['starVLA/dataloader/structured_world/adapters.py', 'starVLA/dataloader/structured_world/cameras.py',
            'starVLA/dataloader/structured_world/labels.py', 'starVLA/dataloader/structured_world/centered_boxes.py',
            'third_party/uniad/ported/occflow_label.py', 'tools/structured_world/build_cache.py',
            'third_party/mmdetection3d/ported/bbox/lidar_box3d.py']
    declaration = {'schema': 'structured_scene_cache_v1', 'dataset': 'navsim', 'population_kind': kind,
        'source_current_identity': json.loads((args.current_root/'identity.json').read_text()),
        'index': rows, 'grid': asdict(GridSpec()), 'code_hashes': {p: digest(ROOT/p) for p in code},
        'external_sources_sha256': digest(ROOT/'third_party/LOCK.json'),
        'label_semantics': 'derived signed road distance and endpoint box occupancy in fixed ego(t0)',
        'height_support_rule': 'LiDAR ground-level evidence, maximum 2m fill; unknown is ignore255'}
    declaration['invalid_label_lidar_rule'] = 'mark affected auxiliary frame unknown255; retain original FM/ego trajectory scene and masks'
    declaration['box_center_rule'] = 'mature upright box transforms with explicit gravity-center pitch/roll correction'
    identity = hashlib.sha256(json.dumps(declaration, sort_keys=True).encode()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output/'identity.json').exists():
        if json.loads((args.output/'identity.json').read_text())['identity'] != identity:
            raise ValueError('Refusing to overwrite a cache built under another data/code contract')
    atomic_json(args.output/'identity.json', {'identity': identity, **declaration})
    atomic_json(args.output/'index.json', rows)
    for name in ('records', 'labels'):
        (args.output/name).mkdir(exist_ok=True)
    tasks = [(json.loads((args.current_root/'current'/(r['token']+'.json')).read_text()), str(args.output), identity) for r in rows]
    records, errors = [], []
    start = time.monotonic()
    # Forking after native Torch/OpenCV initialization can inherit locked thread
    # pools. Spawn gives each offline worker an independent CPU-only runtime.
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        # Bound the submit wakeup pipe as well as pending work. Submitting 100k
        # jobs before consuming results can deadlock CPython 3.10 ProcessPool.
        iterator = iter(tasks)
        pending = {}
        def fill():
            while len(pending) < args.workers*2:
                task = next(iterator, None)
                if task is None:
                    break
                pending[pool.submit(build_one, task)] = task[0]['token']
        fill()
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for item in done:
                token = pending.pop(item)
                try:
                    records.append(item.result())
                except Exception as error:
                    errors.append({'token': token, 'error': str(error), 'exception_type': type(error).__name__})
                if (len(records)+len(errors)) % 64 == 0:
                    print(json.dumps({'complete': len(records), 'errors': errors[-3:], 'seconds': time.monotonic()-start}), flush=True)
            fill()
    statistics = {k: {'positive': 0, 'negative': 0} for k in ('current', 'future', 'arrival', 'release')}
    for record in records:
        for key, counts in record['class_counts'].items():
            for name, number in counts.items():
                statistics[key][name] += number
    summary = {'identity': identity, 'scenes': len(records), 'expected_scenes': len(rows), 'errors': errors,
               'population_kind': kind, 'class_counts': statistics,
               'seconds': time.monotonic()-start, 'bytes': sum(r['label_bytes'] for r in records),
               'road_coverage_mean': float(np.mean([r['coverage']['road'] for r in records])) if records else None}
    atomic_json(args.output/('COMPLETE.json' if not errors else 'INCOMPLETE.json'), summary)
    print(json.dumps(summary, indent=2), flush=True)
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
