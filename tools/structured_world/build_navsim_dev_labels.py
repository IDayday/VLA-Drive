"""Build development labels with the unchanged, pinned training geometry core.

This population is explicitly ineligible for common perception preparation.
No teacher cache or planning evaluator is read by this offline label builder.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import asdict
import hashlib
import json
import multiprocessing
from pathlib import Path
import socket
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.structured_world.build_cache import build_one, atomic_json, digest
from starVLA.model.modules.structured_world.grid import GridSpec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--current-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    policy = json.loads(Path('/mnt/project/server_dispatch_policy.json').read_text())
    if socket.gethostname().removesuffix('-worker-0') not in policy['task_authorizations']['structured_world_fgtr_round1']['allowed_hosts']:
        raise ValueError('Unauthorized development label builder host')
    source = json.loads((args.current_root/'identity.json').read_text())
    if source['split'] != 'dev':
        raise ValueError('This builder is only for the frozen development population')
    rows = json.loads((args.current_root/'index.json').read_text())
    if len(rows) != 1696 or len({row['token'] for row in rows}) != len(rows):
        raise ValueError('Canonical NAVSIM v1 development population changed')
    paths = ['tools/structured_world/build_navsim_dev_labels.py', 'tools/structured_world/build_cache.py',
             'starVLA/dataloader/structured_world/adapters.py', 'starVLA/dataloader/structured_world/cameras.py',
             'starVLA/dataloader/structured_world/labels.py', 'starVLA/dataloader/structured_world/centered_boxes.py',
             'third_party/uniad/ported/occflow_label.py', 'third_party/mmdetection3d/ported/bbox/lidar_box3d.py']
    contract = {'schema': 'structured_scene_cache_v1', 'dataset': 'navsim',
        'population_kind': 'full_development_population_not_for_training',
        'source_current_identity': source, 'index': rows, 'grid': asdict(GridSpec()),
        'code_hashes': {name: digest(ROOT/name) for name in paths},
        'external_sources_sha256': digest(ROOT/'third_party/LOCK.json'),
        'geometry_core': 'unchanged build_cache.build_one and centered UniAD labels used by full train v5',
        'future_time_rule': 'nominal 0.5s annotation within 60ms, otherwise unknown; native planning population unchanged',
        'height_support_rule': 'local LiDAR ground-level evidence with at most 2m fill; unknown255'}
    identity = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output/'identity.json').exists() and json.loads((args.output/'identity.json').read_text())['identity'] != identity:
        raise ValueError('Refusing to mix development label versions')
    atomic_json(args.output/'identity.json', {'identity': identity, **contract})
    atomic_json(args.output/'index.json', rows)
    for name in ('records', 'labels'): (args.output/name).mkdir(exist_ok=True)
    tasks = [(json.loads((args.current_root/'current'/f"{row['token']}.json").read_text()), str(args.output), identity) for row in rows]
    start = time.monotonic()
    records, errors = [], []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        pending, iterator = {}, iter(tasks)
        def fill():
            while len(pending) < 2*args.workers:
                task = next(iterator, None)
                if task is None: break
                pending[pool.submit(build_one, task)] = task[0]['token']
        fill()
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                token = pending.pop(future)
                try: records.append(future.result())
                except Exception as error: errors.append({'token': token, 'error': repr(error)})
            fill()
            if (len(records)+len(errors)) % 64 == 0:
                print(json.dumps({'complete': len(records), 'errors': errors[-3:]}), flush=True)
    summary = {'identity': identity, 'scenes': len(records), 'expected_scenes': len(rows),
        'errors': errors, 'seconds': time.monotonic()-start, 'contains_training_population': False,
        'road_coverage_mean': float(np.mean([row['coverage']['road'] for row in records])) if records else None,
        'bytes': sum(row['label_bytes'] for row in records)}
    atomic_json(args.output/('INCOMPLETE.json' if errors else 'COMPLETE.json'), summary)
    print(json.dumps(summary), flush=True)
    if errors: raise SystemExit(1)


if __name__ == '__main__': main()
