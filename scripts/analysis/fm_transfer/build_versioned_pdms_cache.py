"""Build a separate official-release Navtest cache; never convert v1.1 caches."""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import multiprocessing
from pathlib import Path
import socket
import time
import uuid

from starVLA.rl.fm_transfer.versioned_pdms import upstream_identity, initialize, build_cache_log
from starVLA.rl.fm_transfer.contracts import require_dispatch
from starVLA.rl.simwam_recipe.contracts import POPULATION, file_sha, fingerprint
from starVLA.rl.flow_grpo.transactions import atomic_json, publication_lock


def make_spec(source, version, cache_root, population, raw_logs, maps_root):
    pop = json.loads(Path(population).read_text())
    tokens = pop['navtest_tokens']
    if len(tokens) != 12146 or len(set(tokens)) != 12146 or len({pop['token_logs'][t] for t in tokens}) != 136:
        raise ValueError('Expected complete locked Navtest population')
    protocol = upstream_identity(source, version)
    identity = dict(schema='official_release_navtest_cache_v1', protocol=protocol,
        population_identity=pop['identity_sha256'], tokens_sha256=fingerprint(tokens),
        raw_logs=str(Path(raw_logs).resolve()), maps_root=str(Path(maps_root).resolve()),
        map_version='nuplan-maps-v1.0', scenes=12146, logs=136,
        reference='unmodified version-specific MetricCacheProcessor/PDMClosedPlanner')
    identity['sha256'] = fingerprint(identity)
    spec = dict(source=str(Path(source).resolve()), version=version, protocol=protocol,
        cache_root=str(Path(cache_root).resolve()), cache_identity=identity['sha256'],
        maps_root=identity['maps_root'], raw_logs=identity['raw_logs'])
    return pop, identity, spec


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--source', required=True)
    p.add_argument('--version', choices=['v1.0', 'v1.1'], required=True)
    p.add_argument('--cache-root', required=True)
    p.add_argument('--population', default=POPULATION)
    p.add_argument('--raw-logs', default='/mnt/project/DriveDreamer-Policy/navsim_raw/navsim_logs/test')
    p.add_argument('--maps-root', default='/mnt/navsim/maps')
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--shards', type=int, default=1)
    p.add_argument('--limit', type=int, help='Profile only, never marks full population complete')
    a = p.parse_args()
    host = require_dispatch()
    if not 1 <= a.workers <= 32 or not 0 <= a.shard < a.shards <= 136 or (a.limit is not None and a.limit <= 0):
        raise ValueError('Invalid finite worker/shard/qualification budget')
    pop, identity, spec = make_spec(a.source, a.version, a.cache_root, a.population, a.raw_logs, a.maps_root)
    root = Path(a.cache_root); root.mkdir(parents=True, exist_ok=True)
    with publication_lock(root / '.manifest.lock'):
        path = root / 'manifest.json'
        if path.exists() and json.loads(path.read_text()) != identity:
            raise ValueError('Cache identity changed; use a separate root')
        if not path.exists(): atomic_json(path, identity)
    groups = defaultdict(list)
    for token in pop['navtest_tokens'][:a.limit]: groups[pop['token_logs'][token]].append(token)
    jobs = sorted(groups.items())[a.shard::a.shards]
    attempts = root / 'attempts'; attempts.mkdir(exist_ok=True)
    invocation = uuid.uuid4().hex
    status = dict(status='RUNNING', invocation=invocation, host=host, hostname=socket.gethostname(),
        started_utc=datetime.now(timezone.utc).isoformat(), identity=identity['sha256'],
        source_sha256=file_sha(__file__), worker_source_sha256=file_sha(Path(__file__).resolve().parents[3] / 'starVLA/rl/fm_transfer/versioned_pdms.py'),
        workers=a.workers, shard=a.shard, shards=a.shards, limit=a.limit,
        expected_scenes=sum(len(t) for _, t in jobs), expected_logs=len(jobs),
        completed_scenes=0, new=0, reused=0, worker_seconds=0., logs=[])
    output = attempts / (invocation + '.json'); atomic_json(output, status)
    start = time.monotonic()
    try:
        with ProcessPoolExecutor(a.workers, mp_context=multiprocessing.get_context('spawn'),
                                 initializer=initialize, initargs=(spec,)) as pool:
            futures = [pool.submit(build_cache_log, job) for job in jobs]
            for future in as_completed(futures):
                row = future.result()
                log_path = root / 'logs' / (fingerprint(row['log']) + '.json')
                log_path.parent.mkdir(exist_ok=True)
                # A qualification prefix is kept separately, never overwrites a
                # complete log population receipt.
                if a.limit: log_path = attempts / (invocation + '-' + fingerprint(row['log']) + '.log.json')
                atomic_json(log_path, row)
                status['logs'].append(dict(path=str(log_path), sha256=file_sha(log_path)))
                status['completed_scenes'] += len(row['entries'])
                for key in ('new', 'reused', 'worker_seconds'): status[key] += row[key]
                status['wall_seconds'] = time.monotonic() - start
                atomic_json(output, status)
                print(json.dumps({k: status[k] for k in ('completed_scenes', 'expected_scenes', 'new', 'reused', 'wall_seconds')}), flush=True)
        if status['completed_scenes'] != status['expected_scenes']: raise ValueError('Incomplete shard')
        status['status'] = 'QUALIFICATION_COMPLETE' if a.limit else 'ASSIGNED_COMPLETE'
    except BaseException as error:
        status.update(status='FAILED_PRESERVED', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        status['wall_seconds'] = time.monotonic() - start
        status['ended_utc'] = datetime.now(timezone.utc).isoformat()
        atomic_json(output, status)
    print(json.dumps(dict(receipt=str(output), status=status['status'])), flush=True)


if __name__ == '__main__': main()
