"""Stream geometry-only diagnostics from sealed, officially scored references."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path
import time

import torch

from starVLA.rl.flow_grpo.transactions import atomic_json
from starVLA.rl.safe_flow_improvement.contracts import file_sha, fingerprint
from starVLA.rl.safe_flow_improvement.process_feedback import geometric_feedback
from starVLA.rl.safe_flow_improvement.v1_reward import initialize, protocol_identity


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--population', required=True)
    p.add_argument('--scores', required=True)
    p.add_argument('--cache-root', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--stop-after-for-qualification', type=int)
    a = p.parse_args()
    if not 1 <= a.workers <= 16:
        raise ValueError('Bounded CPU worker count required')
    source, out = Path(a.scores), Path(a.output)
    population = json.loads(Path(a.population).read_text())
    manifest = json.loads((source / 'manifest.json').read_text())
    if manifest['sample_manifest']['population_identity'] != population['identity_sha256']:
        raise ValueError('Population identity mismatch')
    from starVLA.rl.safe_flow_improvement import process_feedback
    identity = {'score_manifest': manifest['sha256'], 'population_identity': population['identity_sha256'],
                'source_sha256': file_sha(__file__), 'geometry_source_sha256': file_sha(process_feedback.__file__)}
    identity['sha256'] = fingerprint(identity)
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'manifest.json').exists():
        if not a.resume or json.loads((out / 'manifest.json').read_text()) != identity:
            raise ValueError('Explicit unchanged geometry resume required')
    else:
        atomic_json(out / 'manifest.json', identity)
    tokens = population[manifest['sample_manifest']['cohort']]
    pending, reused = [], 0
    for token in tokens:
        dst = out / 'scenes' / token[:2] / (token + '.json')
        if dst.exists():
            existing = json.loads(dst.read_text())
            seal = json.loads((source / 'scenes' / token[:2] / (token + '.json')).read_text())
            if existing['identity_sha256'] != identity['sha256'] or existing['score_payload_sha256'] != seal['payload_sha256']:
                raise ValueError('Geometry resume source changed')
            reused += 1
        else:
            pending.append(token)
    repo = str(Path(__file__).resolve().parents[3])
    spec = {'repo': repo, 'cache_roots': [a.cache_root], 'maps_root': '/mnt/navsim/maps',
            'protocol_sha256': protocol_identity(repo)['sha256']}
    start = last_progress = time.monotonic()
    status = dict(status='RUNNING', identity_sha256=identity['sha256'], requested_scenes=len(tokens), reused_scenes=reused, new_scenes=0)
    try:
        with ProcessPoolExecutor(max_workers=a.workers, mp_context=multiprocessing.get_context('spawn'), initializer=initialize, initargs=(spec,)) as pool:
            while pending:
                jobs, ready = [], []
                for token in pending:
                    path = source / 'scenes' / token[:2] / (token + '.pt')
                    if path.with_suffix('.json').exists():
                        seal = json.loads(path.with_suffix('.json').read_text())
                        if seal['identity_sha256'] != manifest['sha256'] or seal['payload_sha256'] != file_sha(path):
                            raise ValueError('Score integrity failure')
                        row = torch.load(path, map_location='cpu', weights_only=True)
                        jobs.append((token, population['token_logs'][token], [f['simulated_body_coordinates'].numpy() for f in row['feedback']]))
                        ready.append((token, seal['payload_sha256']))
                        if len(jobs) >= a.workers * 2:
                            break
                if not jobs:
                    if time.monotonic() - last_progress > 7200:
                        raise TimeoutError('Reference scoring did not progress for two hours')
                    time.sleep(2)
                    continue
                for (token, seal), feedback in zip(ready, pool.map(geometric_feedback, jobs)):
                    dst = out / 'scenes' / token[:2] / (token + '.json')
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    atomic_json(dst, {'identity_sha256': identity['sha256'], 'score_payload_sha256': seal, 'feedback': feedback})
                    pending.remove(token)
                    status['new_scenes'] += 1
                last_progress = time.monotonic()
                status.update(pending_scenes=len(pending), wall_seconds=last_progress-start)
                atomic_json(out / 'status.json', status)
                if a.stop_after_for_qualification and status['new_scenes'] >= a.stop_after_for_qualification:
                    status['status'] = 'QUALIFICATION_COMPLETE_FULL_POPULATION_PENDING'
                    break
            else:
                status['status'] = 'COMPLETE'
    except BaseException as e:
        status.update(status='FAILED_SCENES_REMAIN_IN_DENOMINATOR', error=repr(e))
        atomic_json(out / 'status.json', status)
        raise
    status['wall_seconds'] = time.monotonic()-start
    atomic_json(out / 'status.json', status)
    print(json.dumps(status))


if __name__ == '__main__':
    main()
