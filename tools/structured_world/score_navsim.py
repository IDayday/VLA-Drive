"""Thin export adapter to the unchanged, pinned NAVSIM v1 canonical scorer."""
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
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.local_interaction_mask_v2.score_async import initialize, score_chunk, digest, python_tree_digest, atomic_json
from tools.local_interaction_mask_v2.compare_pdms import METRICS


def main():
    parser = argparse.ArgumentParser()
    for name in ('predictions', 'index', 'devkit', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--population', choices=('dev', 'navtest', 'debug'), required=True)
    parser.add_argument('--workers', type=int, default=12)
    args = parser.parse_args()
    policy = json.loads(Path('/mnt/project/server_dispatch_policy.json').read_text())
    if socket.gethostname().removesuffix('-worker-0') not in policy['task_authorizations']['structured_world_fgtr_round1']['allowed_hosts']:
        raise ValueError('Unauthorized scoring host')
    if not 1 <= args.workers <= 32: raise ValueError('Bounded CPU workers required')
    source = json.loads((args.predictions/'identity.json').read_text())
    done = json.loads((args.predictions/'COMPLETE.json').read_text())
    if done['identity'] != source['identity']: raise ValueError('Incomplete/foreign predictions')
    index = json.loads(args.index.read_text())
    expected = {'dev': 1696, 'navtest': 12146}.get(args.population, len(index))
    if len(index) != expected or done['scenes'] != expected or len({r['token'] for r in index}) != expected:
        raise ValueError('Canonical scoring requires the entire registered population')
    if {p.stem for p in (args.predictions/'records').glob('*.json')} != {r['token'] for r in index}:
        raise ValueError('Prediction and evaluation populations differ')
    initialize(str(args.devkit))
    import nuplan
    contract = {'prediction_identity': source['identity'], 'population': args.population,
        'index_sha256': digest(args.index), 'adapter_sha256': digest(Path(__file__)),
        'canonical_adapter_sha256': digest(ROOT/'tools/local_interaction_mask_v2/score_async.py'),
        'navsim_python_tree_sha256': python_tree_digest(args.devkit/'navsim'),
        'nuplan_python_tree_sha256': python_tree_digest(Path(nuplan.__file__).parent),
        'runtime_versions': {n: importlib.metadata.version(n) for n in ('numpy', 'scipy', 'shapely')},
        'protocol': 'locked NAVSIM v1 full-reference-cache PDM; original 8 x 0.5s trajectory, 40 x 0.1s simulation',
        'failure_policy': 'keep failed scenes at zero in all summaries and invalidate the benchmark'}
    expected_runtime = {'numpy': '1.26.4', 'scipy': '1.13.1', 'shapely': '2.0.7'}
    if contract['navsim_python_tree_sha256'] != '1468974c9af4405597c9dd3e8db1b68e7d7ab337327a7595fb57c2b3dcd4a84f':
        raise ValueError('NAVSIM canonical source differs from the historical frozen evaluator')
    if contract['nuplan_python_tree_sha256'] != 'ffbee2e824a071b5845fa3af657146e7b26ce9550e0c24b95ef624d81724cc36':
        raise ValueError('nuPlan canonical source differs from the historical frozen evaluator')
    if contract['runtime_versions'] != expected_runtime: raise ValueError('Use the locked canonical scoring environment')
    identity = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=True)
    identity_path = args.output/'identity.json'
    if identity_path.exists() and json.loads(identity_path.read_text())['identity'] != identity:
        raise ValueError('Foreign scorer output')
    atomic_json(identity_path, {'identity': identity, **contract})
    tasks = []
    for variant in ('q0', 'q_final'):
        bank = args.output/variant; (bank/'proposals').mkdir(parents=True, exist_ok=True)
        (bank/'records').mkdir(exist_ok=True)
        for row in index:
            token = row['token']; prediction = args.predictions/'predictions'/(token+'.npz')
            record = json.loads((args.predictions/'records'/(token+'.json')).read_text())
            if record['identity'] != source['identity'] or digest(prediction) != record['sha256']:
                raise ValueError('Prediction changed after inference')
            with np.load(prediction, allow_pickle=False) as values: trajectory = values[variant].copy()
            proposal = bank/'proposals'/(token+'.npz')
            temporary = proposal.with_suffix('.tmp')
            with temporary.open('wb') as stream: np.savez_compressed(stream, trajectory=trajectory)
            temporary.replace(proposal)
            tasks.append({**row, 'variant': variant, 'proposal_path': str(proposal), 'proposal_sha256': digest(proposal)})
    start = time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn'),
            initializer=initialize, initargs=(str(args.devkit),)) as pool:
        iterator = iter(tasks); pending = {}
        def fill():
            while len(pending) < args.workers*2:
                batch = []
                for _ in range(8):
                    item = next(iterator, None)
                    if item is None: break
                    saved = args.output/item['variant']/'records'/(item['token']+'.json')
                    if saved.exists():
                        previous = json.loads(saved.read_text())
                        if previous['proposal_sha256'] != item['proposal_sha256']:
                            raise ValueError('Scoring resume proposal changed')
                        if previous.get('metric_cache_sha256') and digest(item['cache_path']) != previous['metric_cache_sha256']:
                            raise ValueError('Metric cache changed after scoring')
                        continue
                    batch.append(item)
                if not batch:
                    if item is None: break
                    continue
                pending[pool.submit(score_chunk, batch)] = batch
        fill()
        while pending:
            ready, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in ready:
                batch = pending.pop(future)
                try: rows = future.result()
                except Exception as error:
                    rows = [{**{k: v[k] for k in ('token', 'log', 'variant', 'proposal_sha256')},
                             'status': 'failed', 'score': 0., 'error': repr(error)} for v in batch]
                for row in rows: atomic_json(args.output/row['variant']/'records'/(row['token']+'.json'), row)
            fill()
    summaries = {}
    for variant in ('q0', 'q_final'):
        bank = args.output/variant
        rows = [json.loads((bank/'records'/(r['token']+'.json')).read_text()) for r in index]
        keys = sorted(set().union(*(r.keys() for r in rows)))
        with (bank/'scenes.csv').open('w') as stream:
            writer = csv.DictWriter(stream, keys); writer.writeheader(); writer.writerows(rows)
        failed = sum(r['status'] != 'ok' for r in rows)
        summary = {'identity': identity, 'scenes': len(rows), 'failed': failed, 'valid': failed == 0,
            'metrics': {k: float(np.mean([r.get(k, 0.) if r['status'] == 'ok' else 0. for r in rows])) for k in METRICS},
            'zero_fraction': sum(r['score'] == 0. for r in rows)/len(rows), 'main_result': variant == 'q_final'}
        atomic_json(bank/'summary.json', summary); summaries[variant] = summary
    atomic_json(args.output/'COMPLETE.json', {'identity': identity, 'seconds': time.monotonic()-start,
                'q0': summaries['q0'], 'q_final': summaries['q_final']})
    print(json.dumps(summaries), flush=True)
    if any(s['failed'] for s in summaries.values()): raise RuntimeError('Scoring failures retained; benchmark invalid')


if __name__ == '__main__': main()
