"""Bounded CPU scoring transport; all metrics come from unchanged V1RewardService."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
from pathlib import Path
import time

import numpy as np

from starVLA.rl.flow_grpo.transactions import atomic_json
from starVLA.rl.safe_flow_improvement.contracts import file_sha, fingerprint
from starVLA.rl.safe_flow_improvement.v1_reward import V1RewardService, protocol_identity


def serializable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(v) for v in value]
    return value


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--queue', required=True)
    p.add_argument('--cache-root', action='append', required=True)
    p.add_argument('--workers', type=int, default=16)
    p.add_argument('--max-new-scores', type=int, required=True)
    p.add_argument('--max-hours', type=float, required=True)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--stop-after-jobs', type=int)
    p.add_argument('--stop-file', help='Owned window completion marker; finish committed jobs and close workers')
    a = p.parse_args()
    if not 1 <= a.workers <= 32 or not 0 < a.max_hours <= 48 or a.max_new_scores <= 0:
        raise ValueError('Explicit bounded scoring resources required')
    out = Path(a.queue)
    out.mkdir(parents=True, exist_ok=True)
    lease = (out / 'server.lock').open('a')
    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    repo = str(Path(__file__).resolve().parents[3])
    from starVLA.rl.safe_flow_improvement import v1_reward
    identity = {'protocol_sha256': protocol_identity(repo)['sha256'], 'source_sha256': file_sha(__file__),
                'adapter_sha256': file_sha(v1_reward.__file__), 'cache_roots': a.cache_root,
                'max_new_scores_per_invocation': a.max_new_scores, 'max_hours_per_invocation': a.max_hours}
    identity['sha256'] = fingerprint(identity)
    if (out / 'manifest.json').exists():
        if not a.resume or json.loads((out / 'manifest.json').read_text()) != identity:
            raise ValueError('Explicit same-identity server resume required')
    else:
        atomic_json(out / 'manifest.json', identity)
    for directory in ('requests', 'results', 'attempts'):
        (out / directory).mkdir(exist_ok=True)
    started = time.monotonic()
    status = dict(status='RUNNING', identity_sha256=identity['sha256'], new_jobs=0, new_scores=0,
                  CPU_scoring_seconds_sum=0., workers=a.workers)
    atomic_json(out / 'status.json', status)
    seen = {p.stem for p in (out / 'results').glob('*.json')}
    spec = {'repo': repo, 'cache_roots': a.cache_root, 'maps_root': '/mnt/navsim/maps'}
    try:
        with V1RewardService(spec, a.workers) as service:
            while time.monotonic()-started < 3600*a.max_hours:
                if a.stop_file and Path(a.stop_file).exists():
                    status['status'] = 'STOPPED_COMPLETED_REGISTERED_WINDOW'
                    break
                paths = sorted((out / 'requests').glob('*.json'))
                todo = [path for path in paths if path.stem not in seen][:a.workers]
                if not todo:
                    time.sleep(.2)
                    continue
                jobs = [json.loads(path.read_text()) for path in todo]
                for job in jobs:
                    sha = job['sha256']
                    if fingerprint({k: v for k, v in job.items() if k != 'sha256'}) != sha or job['server_identity'] != identity['sha256'] or job['protocol_sha256'] != identity['protocol_sha256']:
                        raise ValueError('Invalid request identity')
                count = sum(len(job['requests']) for job in jobs)
                if status['new_scores']+count > a.max_new_scores:
                    status['status'] = 'STOPPED_REGISTERED_SCORE_BUDGET'
                    break
                # Each started attempt is retained. A crash before a committed
                # result is visible as possible duplicate work on recovery.
                attempt = out / 'attempts' / (str(time.time_ns())+'.json')
                atomic_json(attempt, {'status': 'STARTED', 'jobs': [j['sha256'] for j in jobs], 'candidate_count': count})
                values = service.score([(j['token'], j['log'], j['requests']) for j in jobs])
                status['new_scores'] += count
                for job, value in zip(jobs, values):
                    atomic_json(out / 'results' / (job['sha256']+'.json'),
                                {'status': 'OK', 'job_sha256': job['sha256'], 'server_identity': identity['sha256'],
                                 'feedback': serializable(value)})
                    seen.add(job['sha256'])
                    status['new_jobs'] += 1
                    status['CPU_scoring_seconds_sum'] += sum(x['scoring_seconds'] for x in value)
                atomic_json(attempt, {'status': 'COMMITTED', 'jobs': [j['sha256'] for j in jobs], 'candidate_count': count})
                status['wall_seconds'] = time.monotonic()-started
                atomic_json(out / 'status.json', status)
                if a.stop_after_jobs and status['new_jobs'] >= a.stop_after_jobs:
                    status['status'] = 'STOPPED_QUALIFICATION_JOB_LIMIT'
                    break
            else:
                status['status'] = 'STOPPED_REGISTERED_TIME_BUDGET'
    except BaseException as e:
        status.update(status='FAILED_NO_ZERO_REWARD_SUBSTITUTION', error=repr(e))
        atomic_json(out / 'status.json', status)
        raise
    status['wall_seconds'] = time.monotonic()-started
    atomic_json(out / 'status.json', status)
    print(json.dumps(status))


if __name__ == '__main__':
    main()
