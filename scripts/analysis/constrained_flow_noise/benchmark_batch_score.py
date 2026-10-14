"""Warm paired ABBA scoring benchmark, with geometry held identical."""
import argparse
import json
from pathlib import Path
import statistics
import time

import torch

from starVLA.rl.constrained_flow_noise import safety_costs
from starVLA.rl.constrained_flow_noise.contracts import file_sha, require_dispatch
from starVLA.rl.safe_flow_improvement import v1_reward
from starVLA.rl.flow_grpo.transactions import atomic_json
from .batched_canonical_score import score_scene as batch
from .qualify_collection_overlap import exact
from .vector_geometry import body_margins


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--run-root', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root, out = Path(args.run_root), Path(args.output)
    if out.exists():
        raise FileExistsError(out)
    host = require_dispatch()
    spec = dict(repo=str(Path.cwd()), maps_root='/mnt/navsim/maps',
                cache_roots=['/mnt/project/flow-safe-improvement-20261013/metric_cache_v1', str(root / 'metric_cache_v1')],
                metric_manifest=str(root / 'full_TRAIN_assets.json'),
                metric_manifest_sha256=file_sha(root / 'full_TRAIN_assets.json'),
                protocol_sha256=v1_reward.protocol_identity(Path.cwd())['sha256'])
    v1_reward.initialize(spec)
    serial = v1_reward.score_scene
    safety_costs.body_margins = body_margins
    path = root / 'formal_N_FLOW_NOISE_seed42_v1/behavior_collections/001100.pt'
    if file_sha(path) != json.loads(path.with_suffix('.json').read_text())['sha256']:
        raise ValueError('Changed source candidates')
    rows = torch.load(path, map_location='cpu', weights_only=False)['rows'][:8]
    result = dict(status='RUNNING', host=host, source_sha256=file_sha(__file__),
                  scenes=[], actual_candidate_score_calls=0,
                  scope='Fixed first eight TRAIN scenes of existing collection1100; same vector geometry in both methods, warmup excluded')
    atomic_json(out, result)
    try:
        for row in rows:
            job = (row['token'], row['records'][0]['log'], row['requests'])
            records = []
            for iteration, name in enumerate(('serial', 'batch', 'serial', 'batch', 'batch', 'serial')):
                v1_reward.score_scene = serial if name == 'serial' else batch
                t0 = time.monotonic()
                actual = safety_costs.score_scene(job)
                seconds = time.monotonic() - t0
                exact(row['records'], actual)
                result['actual_candidate_score_calls'] += len(actual)
                records.append(dict(iteration=iteration, method=name, warmup=iteration < 2, seconds=seconds))
            result['scenes'].append(dict(position=row['position'], measurements=records,
                                        all_feedback_and_geometry_exact=True))
            atomic_json(out, result)
        totals = {name: sum(statistics.median(x['seconds'] for x in row['measurements']
                                             if not x['warmup'] and x['method'] == name)
                           for row in result['scenes']) for name in ('serial', 'batch')}
        result.update(status='PASS', paired_warm_seconds=totals,
                      speedup=totals['serial'] / totals['batch'],
                      limitation='Eight-scene CPU microbenchmark under shared host load; not whole training throughput')
    except BaseException as error:
        result.update(status='FAILED', error=repr(error))
        raise
    finally:
        v1_reward.score_scene = serial
        atomic_json(out, result)
    print(json.dumps({k: v for k, v in result.items() if k != 'scenes'}))


if __name__ == '__main__':
    main()
