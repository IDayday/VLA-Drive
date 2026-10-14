"""Bounded, topology-independent orchestration of the existing frozen context path.

The numeric capture module is imported unchanged: changing worker assignments
does not change its content-addressed feature identity. Only one topology may
own the cache at a time. Completed records survive worker interruption.
"""
import argparse
import os
import signal
import time
from pathlib import Path

import torch

from iqe.contracts import require
from iqe.io import atomic_json, digest, file_hash, lock, read_json
from iqe.data.sources import load_observation
from .binding import scenes_by_role
from .config import load_config
from .context import SharedCapture, SpatialCache
from .data import ReusedContext
from .resource_guard import check


def ordered_scenes(config):
    roles = scenes_by_role(config)
    rows = sum((roles[r] for r in ('scorer_fit', 'stage_val', 'selector_cal', 'dev_report')), [])
    return sorted(rows, key=lambda s: digest([config.selection_seed, 'spatial', s.scene_id]))


def assignments(ids, workers):
    require(1 <= workers <= 8 and len(set(ids)) == len(ids), 'bounded workers and unique scene list')
    return [ids[i::workers] for i in range(workers)]


def create_plan(config, topology, workers, write_mib_s):
    require(topology.replace('_', '').isalnum(), 'simple explicit topology identity')
    require(0 < write_mib_s <= 32, 'per-worker shared output bandwidth must be bounded')
    root = config.root / 'runtime/spatial_topologies' / topology
    rows = ordered_scenes(config)
    value = {'schema': 'shared_context_dispatch_v1', 'topology': topology, 'config_hash': config.identity,
             'Base_binding_hash': file_hash(config.root / 'BASE_BINDING.json'),
             'capture_code_hash': file_hash(Path(__file__).with_name('context.py')),
             'scene_ids': [s.scene_id for s in rows], 'assignments': assignments([s.scene_id for s in rows], workers),
             'workers': workers, 'write_mib_per_second_per_worker': write_mib_s,
             'original_sampler_and_training_data_unchanged': True}
    value['hash'] = digest(value)
    atomic_json(root / 'PLAN.json', value, immutable=True)
    atomic_json(config.root / 'runtime/ACTIVE_SPATIAL_TOPOLOGY.json',
                {'plan': str(root / 'PLAN.json'), 'hash': value['hash'], 'workers': workers})
    return value


def read_plan(config, topology):
    root = config.root / 'runtime/spatial_topologies' / topology
    plan = read_json(root / 'PLAN.json')
    require(digest({k: v for k, v in plan.items() if k != 'hash'}) == plan['hash'], 'topology manifest corrupt')
    require(plan['config_hash'] == config.identity and plan['Base_binding_hash'] == file_hash(config.root / 'BASE_BINDING.json'), 'cache/Base dependency changed')
    require(plan['capture_code_hash'] == file_hash(Path(__file__).with_name('context.py')), 'frozen capture implementation changed')
    require(plan['scene_ids'] == [s.scene_id for s in ordered_scenes(config)], 'fixed spatial scene list changed')
    require(read_json(config.root / 'runtime/ACTIVE_SPATIAL_TOPOLOGY.json')['hash'] == plan['hash'], 'worker topology superseded')
    return root, plan


def run_worker(config, topology, worker, device):
    check(config, 'cache-score-context')
    root, plan = read_plan(config, topology)
    require(0 <= worker < plan['workers'], 'worker outside plan')
    folder = root / f'worker_{worker}'
    with lock(folder / 'RUN.lock'):
        if (folder / 'COMPLETE.json').exists():
            receipt = read_json(folder / 'COMPLETE.json')
            require(receipt['plan_hash'] == plan['hash'], 'stale worker result')
            return receipt
        by_id = {s.scene_id: s for s in ordered_scenes(config)}
        rows = [by_id[s] for s in plan['assignments'][worker]]
        cache = SpatialCache(config)
        pending = []
        orphaned = []
        for scene in rows:
            if cache.valid(scene):
                continue
            # A crash between tensor commit and checksum receipt is incomplete,
            # not reusable. Retain its bytes as evidence before recomputing.
            p = cache.path(scene)
            if p.exists():
                target = root / 'orphaned' / f'{p.name}.{time.time_ns()}'
                target.parent.mkdir(parents=True, exist_ok=True)
                os.replace(p, target)
                orphaned.append(str(target))
            pending.append(scene)
        stop = [False]
        signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__(0, True))
        signal.signal(signal.SIGINT, lambda *_: stop.__setitem__(0, True))
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        start = time.perf_counter()
        errors, count, output_bytes = [], 0, 0
        atomic_json(folder / 'STATUS.json', {'status': 'LOADING' if pending else 'VERIFYING',
                    'pid': os.getpid(), 'host': os.uname().nodename, 'pending': len(pending), 'reused': len(rows)-len(pending)})
        if pending:
            from iqe.s0_adapter import S0Adapter
            adapter = S0Adapter.load_and_validate_s0(read_json(config.root / 'OFFLINE_BASE_CONTRACT.json'), device)
            require(not any(p.requires_grad for p in adapter.parameters()), 'shared VLA must remain frozen')
            capture = SharedCapture(adapter)
            old = ReusedContext(config)
            try:
                for at in range(0, len(pending), config.cache_batch):
                    if stop[0]:
                        break
                    batch_start = time.perf_counter()
                    sub = pending[at:at+config.cache_batch]
                    observations = [load_observation(s, adapter.contract) for s in sub]
                    planner, values = capture.encode(observations)
                    prediction = adapter.predict_s0(planner)
                    if count == 0:
                        dirty = [dict(o, GT='excluded', future_actors='excluded', metric_cache='excluded') for o in observations]
                        _, other = capture.encode(dirty)
                        require(all(torch.equal(values[k], other[k]) for k in ('world', 'action', 'planner_scene', 'visual')), 'labels changed shared context')
                    written = 0
                    for i, scene in enumerate(sub):
                        prior = old.read(scene)
                        row = {'scene_id': scene.scene_id,
                               'raw_max_abs': float((prediction.raw[i].cpu()-prior['base_raw']).abs().max()),
                               'world_max_abs': float((values['world'][i].cpu()-prior['world']).abs().max()),
                               'action_max_abs': float((values['action'][i].cpu()-prior['action']).abs().max())}
                        require(all(row[k] <= 1e-5 for k in ('raw_max_abs', 'world_max_abs', 'action_max_abs')), 'context drift exceeds preregistered tolerance')
                        errors.append(row)
                        cache.write(scene, {k: values[k][i] for k in ('visual', 'visual_valid', 'visual_xy', 'visual_grid')})
                        written += cache.path(scene).stat().st_size
                    count += len(sub)
                    output_bytes += written
                    # Bound only this job's writes, not unrelated filesystem IO.
                    delay = written/(plan['write_mib_per_second_per_worker']*2**20)-(time.perf_counter()-batch_start)
                    if delay > 0:
                        time.sleep(delay)
                    atomic_json(folder / 'STATUS.json', {'status': 'RUNNING', 'pid': os.getpid(), 'host': os.uname().nodename,
                        'new_records': count, 'pending': len(pending), 'reused': len(rows)-len(pending),
                        'output_bytes': output_bytes, 'wall_seconds': time.perf_counter()-start,
                        'VLM_calls_per_batch': 1, 'visual_calls_per_batch': 1})
            finally:
                capture.close()
        if stop[0]:
            result = {'status': 'PAUSED_RECORD_BOUNDARY', 'new_records': count, 'orphaned': orphaned,
                      'plan_hash': plan['hash'], 'numeric_comparison': errors}
            atomic_json(folder / f'PAUSED_{time.time_ns()}.json', result, immutable=True)
            atomic_json(folder / 'STATUS.json', {k: v for k, v in result.items() if k != 'numeric_comparison'})
            return result
        receipts = {}
        for scene in rows:
            require(cache.valid(scene), 'missing/corrupt context at worker commit')
            receipts[scene.scene_id] = file_hash(str(cache.path(scene))+'.COMPLETE.json')
        result = {'status': 'COMPLETE', 'plan_hash': plan['hash'], 'worker': worker,
                  'scene_receipt_hashes': receipts, 'new_records': count, 'reused': len(rows)-len(pending),
                  'numeric_comparison': errors, 'labels_do_not_affect_context': True,
                  'inherited_records': 'original unchanged capture code plus full tensor checksums; earlier per-record numeric logs were not persisted',
                  'orphaned': orphaned, 'output_bytes': output_bytes, 'wall_seconds': time.perf_counter()-start}
        atomic_json(folder / 'COMPLETE.json', result, immutable=True)
        atomic_json(folder / 'STATUS.json', {k: v for k, v in result.items() if k not in ('numeric_comparison', 'scene_receipt_hashes')})
        return result


def verify(config, topology):
    root, plan = read_plan(config, topology)
    by_id = {s.scene_id: s for s in ordered_scenes(config)}
    coverage = {}
    workers = {}
    for worker in range(plan['workers']):
        path = root / f'worker_{worker}/COMPLETE.json'
        receipt = read_json(path)
        require(receipt['status'] == 'COMPLETE' and receipt['plan_hash'] == plan['hash'], 'worker incomplete/stale')
        require(set(receipt['scene_receipt_hashes']) == set(plan['assignments'][worker]), 'partial worker coverage')
        require(not (set(coverage) & set(receipt['scene_receipt_hashes'])), 'duplicate worker scene ownership')
        for scene_id, expected in receipt['scene_receipt_hashes'].items():
            require(file_hash(str(SpatialCache(config).path(by_id[scene_id]))+'.COMPLETE.json') == expected, 'committed record changed')
        require(all(e[k] <= 1e-5 for e in receipt['numeric_comparison'] for k in ('raw_max_abs', 'world_max_abs', 'action_max_abs')), 'numeric audit failed')
        coverage.update(receipt['scene_receipt_hashes'])
        workers[str(path)] = file_hash(path)
    require(set(coverage) == set(plan['scene_ids']), 'spatial cache does not cover the locked full scene list')
    result = {'status': 'COMPLETE', 'plan_hash': plan['hash'], 'plan_path': str(root/'PLAN.json'),
              'requested': len(plan['scene_ids']), 'scene_receipt_hashes': coverage, 'worker_receipts': workers,
              'labels_do_not_affect_context': True, 'same_numeric_capture_code': plan['capture_code_hash']}
    atomic_json(config.root/'SPATIAL_CONTEXT_COMPLETE.json', result, immutable=True)
    return {'status': 'COMPLETE', 'requested': len(coverage), 'plan_hash': plan['hash']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('plan', 'worker', 'verify'))
    parser.add_argument('--config', required=True)
    parser.add_argument('--topology', required=True)
    parser.add_argument('--workers', type=int)
    parser.add_argument('--worker', type=int)
    parser.add_argument('--write-mib-s', type=float, default=16.)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    config = load_config(args.config)
    if args.command == 'plan':
        require(args.workers is not None and args.worker is None, 'plan requires --workers only')
        result = create_plan(config, args.topology, args.workers, args.write_mib_s)
        result = {k: v for k, v in result.items() if k not in ('scene_ids', 'assignments')}
    elif args.command == 'worker':
        require(args.worker is not None and args.workers is None, 'worker requires --worker only')
        result = run_worker(config, args.topology, args.worker, args.device)
        result = {k: v for k, v in result.items() if k not in ('scene_receipt_hashes', 'numeric_comparison')}
    else:
        require(args.worker is None and args.workers is None, 'verify does not accept worker arguments')
        result = verify(config, args.topology)
    import json
    print(json.dumps(result))


if __name__ == '__main__':
    main()
