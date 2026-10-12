"""Build a declared subset of a locked full cache without changing its population.

Only orchestration is new. The qualified frozen front, payload writer, reader,
numeric identity and full-population continuation checks remain unchanged.
COMPLETE inside the phase cache means that phase only; PHASE_RESULT explicitly
states whether the original full population is covered.
"""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from iqe.contracts import require
from iqe.io import atomic_json, digest, file_hash, lock, read_json
from iqe.vision_cache.plan import capacity_check, load_plan
from iqe.vision_cache.qualification import validate_cache_qualification


def partition(full, reusable, *, target_bytes, record_bytes, full_path, full_checksum):
    """A deterministic access subset, not a new training split or sampler."""
    require(full['scope'] == 'full' and 'phase' not in full, 'parent must be the original full plan')
    require(full['requested_scenes'] == full['fit_count'], 'parent does not cover its full fit population')
    require(target_bytes > 0 and record_bytes > 0, 'positive phase budget required')
    by_id = {r['scene_id']: r for r in full['rows']}
    require(set(reusable) <= set(by_id), 'reused observations outside parent population')
    for sid, row in reusable.items():
        require(row['observation_hash'] == by_id[sid]['observation_hash'] and
                row['role'] == by_id[sid]['role'] == 'incremental_fit', 'reuse observation/role mismatch')
    size = full['records_per_shard']
    limit = min(len(by_id), target_bytes // record_bytes)
    require(limit > len(reusable), 'phase budget contains no new observations')
    if limit < len(by_id):
        limit = len(reusable) + ((limit - len(reusable)) // size) * size
    require(limit > len(reusable), 'phase budget is smaller than one new shard')
    chosen = set(reusable)
    for row in full['rows']:
        if len(chosen) == limit:
            break
        chosen.add(row['scene_id'])
    value = {k: v for k, v in full.items() if k not in {'rows', 'plan_hash'}}
    value['rows'] = [r for r in full['rows'] if r['scene_id'] in chosen]
    value['requested_scenes'] = len(chosen)
    value['phase'] = {
        'schema': 'iqe_fp32_cache_partition_v1',
        'scope': 'STORAGE_PARTITION_OF_LOCKED_FULL_POPULATION',
        'parent_full_plan': str(Path(full_path).resolve()),
        'parent_full_plan_sha256': full_checksum,
        'parent_full_plan_hash': full['plan_hash'],
        'full_population_scenes': len(by_id),
        'requested_target_bytes': target_bytes,
        'measured_max_record_bytes': record_bytes,
        'estimated_logical_bytes': len(chosen) * record_bytes,
        'reused_scene_ids_hash': digest(sorted(reusable)),
        'reused_scenes': len(reusable),
        'missing_from_full': len(by_id) - len(chosen),
        'full_population_complete': len(chosen) == len(by_id),
        'sampler_changed': False,
        'partial_cache_cannot_authorize_training_continuation': True,
    }
    value['plan_hash'] = digest(value)
    return value


def validate_partition(plan):
    phase = plan['phase']
    require(phase['schema'] == 'iqe_fp32_cache_partition_v1', 'unknown storage partition')
    path = phase['parent_full_plan']
    require(file_hash(path) == phase['parent_full_plan_sha256'], 'original full plan changed')
    full = load_plan(path)
    require(full['plan_hash'] == phase['parent_full_plan_hash'], 'wrong original plan identity')
    require(len(plan['rows']) == plan['requested_scenes'] == len({r['scene_id'] for r in plan['rows']}),
            'duplicate or incomplete phase rows')
    original = {r['scene_id']: r for r in full['rows']}
    require(all(original.get(r['scene_id']) == r for r in plan['rows']), 'changed/foreign phase observations')
    for key in full.keys() - {'rows', 'requested_scenes', 'plan_hash'}:
        require(plan[key] == full[key], 'phase changed original training or numeric identity: ' + key)
    require(phase['full_population_scenes'] == len(original) and
            phase['missing_from_full'] == len(original) - plan['requested_scenes'] and
            phase['full_population_complete'] == (plan['requested_scenes'] == len(original)),
            'false full-population completion claim')
    return full


def split_partitions(plan, reusable, count):
    """Disjoint shard-aligned host assignments; reusable records belong to host 0."""
    require(1 <= count <= 8, 'bounded host count required')
    ordinary = [r for r in plan['rows'] if r['scene_id'] not in reusable]
    size = plan['records_per_shard']
    ids = [set(reusable)] + [set() for _ in range(count - 1)]
    for offset in range(0, len(ordinary), size):
        ids[(offset // size) % count].update(r['scene_id'] for r in ordinary[offset:offset + size])
    require(all(ids), 'empty host partition')
    parts = []
    for i, owned in enumerate(ids):
        value = {k: v for k, v in plan.items() if k not in {'rows', 'phase', 'plan_hash'}}
        value['rows'] = [r for r in plan['rows'] if r['scene_id'] in owned]
        value['requested_scenes'] = len(owned)
        value['phase'] = {**plan['phase'], 'partition_index': i, 'partition_count': count,
            'aggregate_plan_hash': plan['plan_hash'], 'aggregate_requested_scenes': plan['requested_scenes'],
            'estimated_logical_bytes': len(owned) * plan['phase']['measured_max_record_bytes'],
            'reused_scene_ids_hash': digest(sorted(reusable) if i == 0 else []),
            'reused_scenes': len(reusable) if i == 0 else 0,
            'missing_from_full': plan['phase']['full_population_scenes'] - len(owned),
            'full_population_complete': len(owned) == plan['phase']['full_population_scenes']}
        value['plan_hash'] = digest(value)
        parts.append(value)
    require(sum(len(x) for x in ids) == len(set.union(*ids)) == plan['requested_scenes'], 'partition overlap/loss')
    return parts


def capacity(plan, certificate, reusable_bytes, output, workers):
    retention = plan['checkpoint_retention_budget']
    protected = retention['remaining_count'] * retention['parent_bytes'] + (180 + 200) * 1024**3
    done = sum(p.stat().st_size for p in Path(output).glob('shard_*.bin')
               if Path(str(p) + '.COMPLETE.json').exists())
    estimate = math.ceil(certificate['representative_record_max_bytes'] * plan['requested_scenes'] * 1.05)
    needed = max(0, estimate - reusable_bytes - done)
    return capacity_check(output, needed, reserve_bytes=protected, temporary_bytes=workers * 2 * 1024**3)


def result_status(codes, complete, phase_count, full_count):
    # Earlier native workers return PARTIAL=3 while later shards are still being
    # built. Only the global receipt proves completed population coverage.
    if not complete or any(c not in (0, 3) for c in codes):
        return 'PARTIAL_OR_FAILED'
    return 'FULL_POPULATION_COMPLETE' if phase_count == full_count else 'PHASE_COMPLETE_FULL_POPULATION_PENDING'


def pause_owned_workers(root, jobs, reason):
    """Stop only children started by this controller, preserving atomic shards."""
    atomic_json(Path(root) / 'CONTROL.json', {'pause': True, 'reason': reason})
    for child, _, _ in jobs:
        if child.poll() is None:
            child.terminate()
    deadline = time.monotonic() + 180
    pending = []
    for child, _, _ in jobs:
        try:
            child.wait(timeout=max(.1, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            pending.append(child.pid)
    return pending


def main():
    parser = argparse.ArgumentParser(__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    plan_arg = sub.add_parser('plan', help='declare a bounded partition of the immutable full access list')
    plan_arg.add_argument('--full-plan', required=True)
    plan_arg.add_argument('--reuse-cache', required=True)
    plan_arg.add_argument('--qualification', required=True)
    plan_arg.add_argument('--target-tb', type=float, required=True, help='decimal TB, including reused payloads')
    plan_arg.add_argument('--output', required=True)
    run = sub.add_parser('run', help='resume qualified native frozen-only workers; never start training')
    for arg in ('config', 'plan', 'root', 'qualification', 'hold-file'):
        run.add_argument('--' + arg, required=True)
    run.add_argument('--reuse-cache')
    run.add_argument('--cohort-admission', help='aggregate storage admission for concurrent host partitions')
    run.add_argument('--workers', type=int, default=8)
    run.add_argument('--rate-mib-s-per-worker', type=float, default=128.)
    run.add_argument('--max-seconds', type=int, default=28800)
    args = parser.parse_args()
    from iqe.vision_cache.storage import CacheReader
    reader = CacheReader(args.reuse_cache) if args.reuse_cache else None
    if args.command == 'plan':
        full = load_plan(args.full_plan)
        cert = validate_cache_qualification(args.qualification, fit_hash=full['fit_sha256'], for_build=True)
        value = partition(full, reader.index['access'], target_bytes=int(args.target_tb * 10**12),
                          record_bytes=cert['representative_record_max_bytes'], full_path=args.full_plan,
                          full_checksum=file_hash(args.full_plan))
        atomic_json(args.output, value, immutable=True)
        print(__import__('json').dumps({'status': 'PLANNED', 'plan_hash': value['plan_hash'], **value['phase']}))
        reader.close()
        return 0

    require(1 <= args.workers <= 8 and 0 < args.rate_mib_s_per_worker <= 128 and
            0 < args.max_seconds <= 86400, 'explicit bounded GPU/IO/time budget required')
    plan = load_plan(args.plan)
    full = validate_partition(plan)
    cert = validate_cache_qualification(args.qualification, fit_hash=full['fit_sha256'], for_build=True)
    if reader:
        require(reader.complete['identity_hash'] == cert['visual_identity_hash'], 'foreign reuse numeric identity')
        require(digest(sorted(reader.index['access'])) == plan['phase']['reused_scene_ids_hash'], 'reuse list changed')
        reused_bytes = reader.complete['bytes']
        reader.close()
    else:
        require(plan['phase']['reused_scenes'] == 0 and plan['phase']['reused_scene_ids_hash'] == digest([]), 'missing reuse cache')
        reused_bytes = 0
    if plan['phase'].get('partition_count', 1) > 1:
        require(args.cohort_admission, 'concurrent partitions need aggregate storage admission')
        admission = read_json(args.cohort_admission)
        require(admission['status'] == 'PASS' and admission['capacity']['sufficient'] and
                plan['plan_hash'] in admission['partition_plan_hashes'] and
                plan['phase']['aggregate_plan_hash'] == admission['aggregate_plan_hash'], 'foreign/failed cohort admission')
        require(0 <= time.time() - admission['created_unix'] <= 600, 'stale cohort resource admission; inspect and renew')
    from iqe.vision_cache.cli import hold_check
    hold_check(args.hold_file)
    root = Path(args.root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with lock(root / 'PHASE_CONTROLLER.lock'):
        if (root / 'CONTROL.json').exists():
            require(not read_json(root / 'CONTROL.json').get('pause'),
                    'phase is paused; explicitly release its CONTROL.json before resuming')
        atomic_json(root / 'PHASE.json', plan['phase'], immutable=True)
        cap = capacity(plan, cert, reused_bytes, root, args.workers)
        atomic_json(root / 'PHASE_CAPACITY.json', cap)
        require(cap['sufficient'], 'BLOCKED_SHARED_CAPACITY: bounded phase plus unchanged reserves does not fit')
        from iqe.resources import qualify
        resources = qualify('cuda')
        require(len(resources['GPUs']) == args.workers, 'select exactly the intended idle GPUs')
        inventory = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'], text=True)
        mapping = {i.strip(): u.strip() for i, u in (line.split(',') for line in inventory.splitlines())}
        selected = {mapping[str(g['index'])] for g in resources['GPUs']}
        probe = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid', '--format=csv,noheader'], text=True)
        require(not any(line.split(',')[1].strip() in selected for line in probe.splitlines() if ',' in line),
                'GPU owner appeared on a selected GPU before phase launch')
        native = Path(__file__).resolve().parents[3]
        stamp = time.time_ns()
        attempt = root / 'attempts' / str(stamp)
        attempt.mkdir(parents=True)
        shards = math.ceil(math.ceil(plan['requested_scenes'] / plan['records_per_shard']) / args.workers)
        jobs = []
        started = time.monotonic()
        stopping = [False]
        previous = signal.signal(signal.SIGTERM, lambda *_: stopping.__setitem__(0, True))
        previous_int = signal.signal(signal.SIGINT, lambda *_: stopping.__setitem__(0, True))
        try:
            for worker in range(args.workers):
                argv = [sys.executable, '-u', '-m', 'iqe.vision_cache.cli', 'cache-build',
                        '--config', args.config, '--plan', args.plan, '--root', str(root),
                        '--device', 'cuda:0', '--worker', str(worker), '--workers', str(args.workers),
                        '--max-shards', str(shards), '--max-gib', str(2 * shards + 2),
                        '--reserve-gib', str(cap['reserve_bytes'] / 1024**3),
                        '--rate-mib-s', str(args.rate_mib_s_per_worker), '--qualification', args.qualification,
                        '--hold-file', args.hold_file]
                if args.reuse_cache:
                    argv.extend(['--reuse-cache', args.reuse_cache])
                env = dict(os.environ, PYTHONPATH=str(native), CUDA_VISIBLE_DEVICES=resources['GPUs'][worker]['index'],
                           OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', TOKENIZERS_PARALLELISM='false')
                log = (attempt / f'worker_{worker}.log').open('ab')
                child = subprocess.Popen(argv, cwd=native, env=env, stdout=log, stderr=subprocess.STDOUT)
                jobs.append((child, log, argv))
            launch = {'schema': 'iqe_fp32_cache_phase_launch_v1', 'started_unix': time.time(),
                      'controller_pid': os.getpid(), 'resources': resources, 'plan_hash': plan['plan_hash'],
                      'phase': plan['phase'], 'phase_capacity': cap, 'args': vars(args),
                      'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=native, text=True).strip(),
                      'script_sha256': file_hash(__file__), 'no_training_or_migration': True,
                      'jobs': [{'pid': p.pid, 'argv': cmd} for p, _, cmd in jobs]}
            atomic_json(attempt / 'LAUNCH.json', launch, immutable=True)
            atomic_json(root / 'PHASE_LAUNCH.json', {'path': str(attempt / 'LAUNCH.json'),
                        'sha256': file_hash(attempt / 'LAUNCH.json'), 'attempt': str(attempt)})
            while any(p.poll() is None for p, _, _ in jobs):
                codes = [p.poll() for p, _, _ in jobs]
                space = capacity_check(root, 0, reserve_bytes=cap['reserve_bytes'], temporary_bytes=args.workers * 2 * 1024**3)
                reason = ('OPERATOR_PAUSED' if stopping[0] else
                          'BUDGET_PAUSED' if time.monotonic() - started >= args.max_seconds else
                          'PARTIAL_STORAGE_RESERVE' if not space['sufficient'] else
                          'WORKER_FAILED' if any(c not in (None, 0, 3) for c in codes) else None)
                progress = [read_json(p) for p in root.glob('PROGRESS_*.json')]
                progress = [p for p in progress if p.get('updated_unix', 0) >= launch['started_unix']]
                status = {'status': reason or 'BUILDING', 'updated_unix': time.time(),
                          'elapsed_seconds': time.monotonic() - started, 'returncodes': codes,
                          'phase_scenes': plan['requested_scenes'], 'full_scenes': full['requested_scenes'],
                          'new_bytes_this_attempt': sum(x['bytes_written_this_attempt'] for x in progress),
                          'new_shards_this_attempt': sum(x['completed_shards_this_attempt'] for x in progress),
                          'reused_bytes': reused_bytes, 'free_bytes': space['free_bytes'], 'attempt': str(attempt)}
                atomic_json(root / 'PHASE_STATUS.json', status)
                if reason:
                    pending = pause_owned_workers(root, jobs, reason)
                    if pending:
                        atomic_json(root / 'PHASE_STATUS.json', {**status, 'status': 'WAITING_OWN_WORKER_SAFE_EXIT', 'pids': pending})
                    return 3
                time.sleep(30)
            codes = [p.returncode for p, _, _ in jobs]
            state = result_status(codes, (root / 'COMPLETE.json').exists(), plan['requested_scenes'], full['requested_scenes'])
            result = {'status': state, 'returncodes': codes, 'phase': plan['phase'],
                      'elapsed_seconds': time.monotonic() - started, 'updated_unix': time.time(),
                      'no_training_or_migration': True, 'full_cache_root_not_published': True}
            if state != 'PARTIAL_OR_FAILED':
                current = CacheReader(root)
                require(current.complete['plan_hash'] == plan['plan_hash'], 'wrong phase receipt')
                require(set(current.allowed) == {r['scene_id'] for r in plan['rows']}, 'phase coverage mismatch')
                ids = sorted(current.allowed)
                selected = [ids[round(i * (len(ids) - 1) / (min(128, len(ids)) - 1))]
                            for i in range(min(128, len(ids)))] if len(ids) > 1 else ids
                result['cross_population_record_verify'] = current.verify(scene_ids=selected)
                result['receipt'] = current.complete
                result['receipt_sha256'] = file_hash(root / 'COMPLETE.json')
                result['full_shard_hashes_verified_by_native_commit'] = True
                current.close()
            atomic_json(attempt / 'RESULT.json', result, immutable=True)
            atomic_json(root / 'PHASE_RESULT.json', result)
            atomic_json(root / 'PHASE_STATUS.json', result)
            print(__import__('json').dumps(result))
            return 0 if state != 'PARTIAL_OR_FAILED' else 2
        except BaseException as error:
            pending = pause_owned_workers(root, jobs, 'CONTROLLER_FAILED')
            atomic_json(root / 'PHASE_STATUS.json', {'status': 'FAILED', 'error': str(error),
                        'pending_own_workers': pending, 'attempt': str(attempt), 'updated_unix': time.time()})
            raise
        finally:
            signal.signal(signal.SIGTERM, previous)
            signal.signal(signal.SIGINT, previous_int)
            for p, log, _ in jobs:
                log.close()


if __name__ == '__main__':
    import json
    try:
        sys.exit(main())
    except Exception as error:
        print(json.dumps({'status': 'BLOCKED' if 'BLOCKED' in str(error) else 'FAILED', 'error': str(error)}), flush=True)
        sys.exit(2)
