"""Reuse sealed v1 assets and build only missing full-RL_TRAIN official caches."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import time

from starVLA.rl.constrained_flow_noise.contracts import require_dispatch, file_sha, fingerprint
from starVLA.rl.safe_flow_improvement.v1_reward import V1RewardService
from starVLA.rl.flow_grpo.transactions import atomic_json


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--population', required=True)
    p.add_argument('--cache-root', required=True)
    p.add_argument('--reuse-root', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--workers', type=int, default=16)
    p.add_argument('--resume', action='store_true')
    a = p.parse_args(); host = require_dispatch()
    pop = json.loads(Path(a.population).read_text()); out = Path(a.output)
    if out.exists() and not a.resume:
        raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True)
    spec = {'repo': str(Path(__file__).resolve().parents[3]), 'build_cache_root': a.cache_root,
            'cache_roots': [a.reuse_root, a.cache_root],
            'raw_logs': '/mnt/project/DriveDreamer-Policy/navsim_raw/navsim_logs/trainval',
            'maps_root': '/mnt/navsim/maps'}
    if not 1 <= a.workers <= 16:
        raise ValueError('Bounded sixteen-CPU cache producer')
    jobs = defaultdict(list); reused = []
    for token in pop['train_tokens']:
        log = pop['token_logs'][token]
        old = Path(a.reuse_root) / log / 'unknown' / token / 'metric_cache.pkl'
        if old.exists():
            reused.append({'token': token, 'log': log, 'sha256': file_sha(old), 'path': str(old)})
        else:
            jobs[log].append(token)
    started = time.monotonic()
    report = {'status': 'RUNNING', 'host': host, 'population_identity': pop['identity_sha256'],
              'requested_scenes': len(pop['train_tokens']), 'reused_scenes': len(reused),
              'reused_manifest_sha256': fingerprint(reused), 'workers': a.workers, 'logs': [], 'spec': spec}
    atomic_json(out/'reused.json', reused)
    with V1RewardService(spec, a.workers) as service:
        report['protocol_sha256'] = service.spec['protocol_sha256']
        for record in service.build(sorted(jobs.items())):
            report['logs'].append(record)
            report['wall_seconds'] = time.monotonic()-started
            atomic_json(out/'report.json', report)
            print(json.dumps({'completed_logs': len(report['logs']), 'total_logs': len(jobs),
                              'scenes_built_or_resumed': sum(len(x['entries']) for x in report['logs'])}), flush=True)
    completed = len(reused) + sum(len(x['entries']) for x in report['logs'])
    if completed != len(pop['train_tokens']):
        raise ValueError('Incomplete real v1 metric cache population')
    report.update(status='COMPLETE', wall_seconds=time.monotonic()-started, completed_scenes=completed)
    atomic_json(out/'report.json', report)
    print(json.dumps({'status': 'COMPLETE', 'scenes': completed, 'wall_seconds': report['wall_seconds']}))


if __name__ == '__main__':
    main()
