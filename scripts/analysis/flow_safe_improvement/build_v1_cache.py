"""Build real v1 metric caches for registered Navtrain populations."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import time

from starVLA.rl.safe_flow_improvement.contracts import file_sha
from starVLA.rl.safe_flow_improvement.v1_reward import V1RewardService
from starVLA.rl.flow_grpo.transactions import atomic_json


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--population', required=True)
    p.add_argument('--cohort', choices=['calibration_tokens', 'reference_screen_tokens', 'diagnostic_random_tokens', 'dev_tokens'], required=True)
    p.add_argument('--cache-root', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--limit', type=int, help='Qualification/profile only; recorded explicitly')
    p.add_argument('--raw-logs', default='/mnt/project/DriveDreamer-Policy/navsim_raw/navsim_logs/trainval')
    p.add_argument('--maps-root', default='/mnt/navsim/maps')
    a = p.parse_args()
    assert 1 <= a.workers <= 32
    manifest = json.loads(Path(a.population).read_text())
    tokens = manifest[a.cohort][:a.limit]
    assert tokens and set(tokens) <= set(manifest['train_tokens'] + manifest['dev_tokens'])
    jobs = defaultdict(list)
    for token in tokens:
        jobs[manifest['token_logs'][token]].append(token)
    output = Path(a.output)
    output.mkdir(parents=True, exist_ok=False)
    spec = {'repo': str(Path(__file__).resolve().parents[3]), 'build_cache_root': a.cache_root,
            'cache_roots': [a.cache_root], 'raw_logs': a.raw_logs, 'maps_root': a.maps_root}
    receipt = {'status': 'RUNNING', 'population_sha256': file_sha(a.population), 'cohort': a.cohort,
               'limit': a.limit, 'requested_scenes': len(tokens), 'workers': a.workers, 'logs': [],
               'script_sha256': file_sha(__file__), 'spec': spec}
    atomic_json(output / 'receipt.json', receipt)
    started = time.monotonic()
    try:
        with V1RewardService(spec, a.workers) as service:
            receipt['protocol_sha256'] = service.spec['protocol_sha256']
            for row in service.build(sorted(jobs.items())):
                receipt['logs'].append(row)
                atomic_json(output / 'receipt.json', receipt)
                print(json.dumps({'completed_logs': len(receipt['logs']), 'total_logs': len(jobs),
                                  'scenes_in_log': len(row['entries']), 'seconds_in_log': row['seconds']}), flush=True)
    except Exception as error:
        receipt.update(status='FAILED_INFRASTRUCTURE_NOT_REWARD_ZERO', error_type=type(error).__name__, error=str(error))
        atomic_json(output / 'receipt.json', receipt)
        raise
    receipt.update(status='COMPLETE', wall_seconds=time.monotonic() - started,
                   completed_scenes=sum(len(row['entries']) for row in receipt['logs']))
    atomic_json(output / 'receipt.json', receipt)


if __name__ == '__main__':
    main()
