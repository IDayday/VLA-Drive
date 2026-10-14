"""Stream atomic-ready Navtest q0/q_final predictions to the pinned PDM pool.

Metric computation uses unmodified frozen score_async.initialize/score_chunk.
Only scheduling is streamed; a benchmark completes after both full populations.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import csv
import hashlib
import importlib.metadata
import json
import multiprocessing
import os
from pathlib import Path
import socket
import sys
import time

from run_navsim_navtest import authorized

for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
FROZEN = Path('/mnt/project/VLA-Drive-structured-world-fgtr-io-formal-frozen-20261009')
sys.path.insert(0, str(FROZEN))
from tools.local_interaction_mask_v2.score_async import initialize, score_chunk, digest, python_tree_digest
from tools.local_interaction_mask_v2.compare_pdms import METRICS
EXECUTED_TOOL_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def atomic(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.' + str(os.getpid()) + '.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n'); temp.replace(path)


def main():
    import numpy as np
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('predictions', 'index', 'devkit', 'output'): p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--workers', type=int, default=16)
    p.add_argument('--research-scope', choices=('structured_world', 'decision_local'), default='structured_world')
    p.add_argument('--population', choices=('navtest', 'dev', 'debug'), default='navtest')
    p.add_argument('--max-seconds', type=float, default=21600)
    a = p.parse_args()
    # Scoring uses the same current task, expiry and reservation checks as the
    # GPU dispatcher; the historical five-host field is not today's allowlist.
    if a.research_scope == 'decision_local':
        # Routing only: retain the same frozen canonical functions and hashes.
        # This does not resume the paused G-family evaluator or trainers.
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from tools.decision_local.common import check_host
        check_host()
    else:
        authorized()
        if a.population != 'navtest': raise ValueError('Historical G endpoint remains Navtest')
    if not 1 <= a.workers <= 32: raise ValueError('Bounded workers required')
    if a.max_seconds <= 0: raise ValueError('Positive bounded runtime required')
    index = json.loads(a.index.read_text())
    expected = {'navtest': 12146, 'dev': 1696}.get(a.population, len(index))
    expected_logs = {'navtest': 136, 'dev': 16}.get(a.population)
    if len(index) != expected or len({r['token'] for r in index}) != expected or (expected_logs is not None and len({r['log'] for r in index}) != expected_logs):
        raise ValueError('Streaming scorer requires the complete registered population')
    a.output.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    while not (a.predictions / 'identity.json').exists():
        if time.monotonic()-start > a.max_seconds: raise TimeoutError('Waiting for prediction identity; no full result')
        if (a.predictions / 'INFERENCE_FAILED.json').exists(): raise RuntimeError('Inference failed before predictions')
        atomic(a.output / 'PROGRESS.json', {'status': 'WAITING_FOR_FIRST_PREDICTION_IDENTITY', 'pid': os.getpid()})
        time.sleep(1)
    source = json.loads((a.predictions / 'identity.json').read_text())
    initialize(str(a.devkit)); import nuplan
    contract = {'prediction_identity': source['identity'], 'population': a.population,
        'index_sha256': digest(a.index), 'adapter_sha256': digest(FROZEN / 'tools/structured_world/score_navsim.py'),
        'canonical_adapter_sha256': digest(FROZEN / 'tools/local_interaction_mask_v2/score_async.py'),
        'navsim_python_tree_sha256': python_tree_digest(a.devkit / 'navsim'),
        'nuplan_python_tree_sha256': python_tree_digest(Path(nuplan.__file__).parent),
        'runtime_versions': {n: importlib.metadata.version(n) for n in ('numpy', 'scipy', 'shapely')},
        'protocol': 'locked NAVSIM v1 full-reference-cache PDM; original 8 x 0.5s trajectory, 40 x 0.1s simulation',
        'failure_policy': 'keep failed scenes at zero in all summaries and invalidate the benchmark'}
    if contract['navsim_python_tree_sha256'] != '1468974c9af4405597c9dd3e8db1b68e7d7ab337327a7595fb57c2b3dcd4a84f':
        raise ValueError('Wrong pinned NAVSIM Python tree')
    if contract['nuplan_python_tree_sha256'] != '0e83aab8d5b92bf5792d815a1f25686088d1f4b8db70ea604331cd98f1606849':
        raise ValueError('Wrong pinned nuPlan Python tree')
    if contract['runtime_versions'] != {'numpy': '1.26.4', 'scipy': '1.13.1', 'shapely': '2.0.7'}:
        raise ValueError('Wrong canonical runtime')
    identity = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    path = a.output / 'identity.json'
    if path.exists() and json.loads(path.read_text())['identity'] != identity: raise ValueError('Foreign scoring output')
    atomic(path, dict(contract, identity=identity))
    atomic(a.output / 'STREAM_ROUTING.json', {'routing_tool_sha256': EXECUTED_TOOL_SHA256, 'pid': os.getpid(),
        'workers': a.workers, 'streaming': True, 'scientific_contract_identical_to_frozen_score_navsim': True})
    submitted = set(); done = set(); ready = []; scanned = set(); pending = {}; failed = 0
    for variant in ('q0', 'q_final'):
        (a.output / variant / 'proposals').mkdir(parents=True, exist_ok=True)
        (a.output / variant / 'records').mkdir(exist_ok=True)
    # Resume only hash-bound prior score rows. No dropped/retried failing scene.
    by_token = {r['token']: r for r in index}
    def progress(status):
        atomic(a.output / 'PROGRESS.json', {'status': status, 'pid': os.getpid(), 'prediction_scenes_ready': len(scanned),
            'q0_scored': sum(v == 'q0' for v, _ in done), 'q_final_scored': sum(v == 'q_final' for v, _ in done),
            'failed_new_rows': failed, 'pending_batches': len(pending), 'seconds': time.monotonic()-start})
    def scan():
        last_progress = time.monotonic()
        if not scanned: progress('VALIDATING_RESUMED_INPUTS')
        for path in (a.predictions / 'records').glob('*.json'):
            token = path.stem
            if token in scanned: continue
            if token not in by_token: raise ValueError('Prediction token outside Navtest')
            record = json.loads(path.read_text()); prediction = a.predictions / 'predictions' / (token + '.npz')
            if record['identity'] != source['identity'] or digest(prediction) != record['sha256']:
                raise ValueError('Atomic-ready prediction changed')
            cache_sha = None
            with np.load(prediction, allow_pickle=False) as values:
                for variant in ('q0', 'q_final'):
                    trajectory = values[variant].copy()
                    if trajectory.shape != (8, 3) or not np.isfinite(trajectory).all(): raise ValueError('Invalid complete trajectory')
                    proposal = a.output / variant / 'proposals' / (token + '.npz')
                    temp = proposal.with_suffix('.tmp')
                    with temp.open('wb') as stream: np.savez_compressed(stream, trajectory=trajectory)
                    temp.replace(proposal)
                    task = dict(by_token[token], variant=variant, proposal_path=str(proposal), proposal_sha256=digest(proposal))
                    key = (variant, token); saved = a.output / variant / 'records' / (token + '.json')
                    if saved.exists():
                        previous = json.loads(saved.read_text())
                        # Both variants use the same immutable GT cache. Verify
                        # its bytes once for this token, retaining both guards.
                        if cache_sha is None: cache_sha = digest(task['cache_path'])
                        if previous['proposal_sha256'] != task['proposal_sha256'] or previous.get('metric_cache_sha256') != cache_sha:
                            raise ValueError('Resumed score input changed')
                        submitted.add(key); done.add(key)
                    else: ready.append(task)
            scanned.add(token)
            if time.monotonic()-last_progress >= 5:
                progress('VALIDATING_READY_INPUTS'); last_progress = time.monotonic()
    with ProcessPoolExecutor(max_workers=a.workers, mp_context=multiprocessing.get_context('spawn'),
            initializer=initialize, initargs=(str(a.devkit),)) as pool:
        while len(done) < 2 * expected:
            if time.monotonic()-start > a.max_seconds: raise TimeoutError('Bounded streaming window expired; partial scores preserved')
            if (a.predictions / 'INFERENCE_FAILED.json').exists(): raise RuntimeError('Inference failed; partial scores retained')
            scan()
            while ready and len(pending) < a.workers * 2:
                batch = ready[:8]; del ready[:8]
                for t in batch:
                    key = (t['variant'], t['token'])
                    if key in submitted: raise ValueError('Duplicate scoring task')
                    submitted.add(key)
                pending[pool.submit(score_chunk, batch)] = batch
            if pending:
                completed, _ = wait(pending, timeout=1, return_when=FIRST_COMPLETED)
                for future in completed:
                    batch = pending.pop(future)
                    try: rows = future.result()
                    except Exception as error:
                        rows = [dict({k: t[k] for k in ('token', 'log', 'variant', 'proposal_sha256')},
                            status='failed', score=0., error=repr(error)) for t in batch]
                    for row in rows:
                        atomic(a.output / row['variant'] / 'records' / (row['token'] + '.json'), row)
                        done.add((row['variant'], row['token'])); failed += row['status'] != 'ok'
            else: time.sleep(1)
            progress('STREAMING')
    while not (a.predictions / 'COMPLETE.json').exists():
        if time.monotonic()-start > a.max_seconds: raise TimeoutError('Incomplete export after scoring; no full result')
        if (a.predictions / 'INFERENCE_FAILED.json').exists(): raise RuntimeError('Inference failed after partial scores')
        time.sleep(1)
    complete = json.loads((a.predictions / 'COMPLETE.json').read_text())
    if complete['identity'] != source['identity'] or complete['scenes'] != expected: raise ValueError('Foreign/incomplete full predictions')
    summaries = {}
    for variant in ('q0', 'q_final'):
        bank = a.output / variant
        paths = list((bank / 'records').glob('*.json'))
        if len(paths) != expected or {p.stem for p in paths} != set(by_token): raise ValueError('Scored population differs')
        rows = [json.loads((bank / 'records' / (r['token'] + '.json')).read_text()) for r in index]
        keys = sorted(set().union(*(r.keys() for r in rows)))
        with (bank / 'scenes.csv').open('w') as stream:
            writer = csv.DictWriter(stream, keys); writer.writeheader(); writer.writerows(rows)
        bad = sum(r['status'] != 'ok' for r in rows)
        summary = {'identity': identity, 'scenes': expected, 'failed': bad, 'valid': bad == 0,
            'scene_csv_sha256': digest(bank / 'scenes.csv'),
            'metrics': {k: float(np.mean([r.get(k, 0.) if r['status'] == 'ok' else 0. for r in rows])) for k in METRICS},
            'zero_fraction': sum(r['score'] == 0. for r in rows)/expected, 'main_result': variant == 'q_final'}
        atomic(bank / 'summary.json', summary); summaries[variant] = summary
    result = dict(identity=identity, seconds=time.monotonic()-start, **summaries)
    if any(s['failed'] for s in summaries.values()):
        atomic(a.output / 'INVALID.json', result)
        raise RuntimeError('Scoring failures retained; benchmark invalid')
    atomic(a.output / 'COMPLETE.json', result); print(json.dumps(result), flush=True)


if __name__ == '__main__': main()
