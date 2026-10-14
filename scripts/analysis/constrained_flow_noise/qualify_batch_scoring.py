"""Canonical singleton-vs-batch, partition, order and distractor invariance."""
import argparse
import functools
import json
from pathlib import Path
import time

import numpy as np
import torch

from starVLA.rl.constrained_flow_noise.contracts import file_sha, require_dispatch
from starVLA.rl.constrained_flow_noise import safety_costs
from starVLA.rl.safe_flow_improvement import v1_reward
from starVLA.rl.flow_grpo.transactions import atomic_json
from .batched_canonical_score import score_scene as batched
from .qualify_collection_overlap import exact
from .vector_geometry import body_margins


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--run-root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--scenes', type=int, default=32)
    args = parser.parse_args()
    host = require_dispatch()
    root, out = Path(args.run_root), Path(args.output)
    if out.exists():
        raise FileExistsError(out)
    spec = dict(repo=str(Path.cwd()), maps_root='/mnt/navsim/maps',
                cache_roots=['/mnt/project/flow-safe-improvement-20261013/metric_cache_v1', str(root / 'metric_cache_v1')],
                metric_manifest=str(root / 'full_TRAIN_assets.json'),
                metric_manifest_sha256=file_sha(root / 'full_TRAIN_assets.json'),
                protocol_sha256=v1_reward.protocol_identity(Path.cwd())['sha256'])
    v1_reward.initialize(spec)
    canonical = v1_reward.score_scene
    safety_costs.body_margins = body_margins
    train = set(json.loads(Path('/mnt/project/flow-safe-improvement-20261013/population/manifest.json').read_text())['train_tokens'])
    result = dict(status='RUNNING', host=host, source_sha256=file_sha(__file__),
                  adapter_sha256=file_sha(Path(__file__).with_name('batched_canonical_score.py')),
                  protocol_sha256=spec['protocol_sha256'], scenes=[], score_calls=0,
                  scope='Fixed consecutive existing TRAIN groups; no actor update or new formal exposure',
                  checks='All metrics, raw progress, states, bodies, time-aligned margins, event times, requests, cost and metadata; timing fields only excluded')
    atomic_json(out, result)
    cal = json.loads((root / 'calibration/calibration_lock.json').read_text())
    def call(job, chunk):
        v1_reward.score_scene = canonical if chunk is None else functools.partial(batched, candidate_chunk=chunk)
        t0 = time.monotonic()
        values = safety_costs.score_scene(job)
        result['score_calls'] += len(values)
        return values, time.monotonic() - t0
    try:
        for collection in range(1100, 1120):
            path = root / 'formal_N_FLOW_NOISE_seed42_v1/behavior_collections' / f'{collection:06d}.pt'
            if file_sha(path) != json.loads(path.with_suffix('.json').read_text())['sha256']:
                raise ValueError('Changed saved behavior archive')
            archive = torch.load(path, map_location='cpu', weights_only=False)
            for row in archive['rows']:
                if row['token'] not in train:
                    raise ValueError('Non-TRAIN scene')
                job = (row['token'], row['records'][0]['log'], row['requests'])
                baseline, serial_seconds = call(job, None)
                exact(row['records'], baseline, 'archived_canonical')
                item = dict(collection=collection, position=row['position'], canonical_seconds=serial_seconds,
                            candidates=len(baseline), chunk_seconds={}, all_scientific_fields_exact=True,
                            protocol_tests=[])
                for chunk in (1, 3, 8, 16):
                    actual, seconds = call(job, chunk)
                    exact(baseline, actual, f'chunk{chunk}')
                    for left, right in zip(baseline, actual):
                        np.testing.assert_array_equal(safety_costs.costs_from_record(left, cal['thresholds']),
                                                      safety_costs.costs_from_record(right, cal['thresholds']))
                    item['chunk_seconds'][str(chunk)] = seconds
                # Fixed first four scenes test reordered candidates, duplicates
                # and intentionally poor finite distractors. These are protocol
                # probes only, never inserted into the on-policy training data.
                if len(result['scenes']) < 4:
                    requests = np.asarray(row['requests'])
                    perm = np.arange(len(requests))[::-1]
                    reverse, _ = call((job[0], job[1], requests[perm]), 5)
                    exact([baseline[i] for i in perm], reverse, 'reverse_and_chunk5')
                    alternatives = requests[[0, 1, 0, 2]].copy()
                    alternatives[1, :, 0] *= 3
                    alternatives[3, :, 1] += 12
                    solo, _ = call((job[0], job[1], alternatives), None)
                    batch, _ = call((job[0], job[1], alternatives), 16)
                    exact(solo, batch, 'changed_other_candidates')
                    exact(baseline[0], batch[0], 'fixed_candidate_among_distractors')
                    exact(batch[0], batch[2], 'duplicate_does_not_gain_or_change_score')
                    item['protocol_tests'] = ['reverse_order_chunk5', 'changed_other_candidates', 'duplicates']
                result['scenes'].append(item)
                atomic_json(out, result)
                print(json.dumps(dict(scene=len(result['scenes']), **item)), flush=True)
                if len(result['scenes']) >= args.scenes:
                    break
            if len(result['scenes']) >= args.scenes:
                break
        if len(result['scenes']) != args.scenes:
            raise ValueError('Insufficient fixed scene population')
        serial = sum(x['canonical_seconds'] for x in result['scenes'])
        batch = sum(x['chunk_seconds']['16'] for x in result['scenes'])
        result.update(status='PASS', all_feedback_and_geometry_bitwise_equal=True,
                      canonical_seconds=serial, batch16_seconds=batch, measured_score_speedup=serial / batch,
                      limitation='CPU scoring microbenchmark; fixed order/cold cache can affect timings; full native learning acceptance remains required')
    except BaseException as error:
        result.update(status='FAILED', error=repr(error))
        raise
    finally:
        v1_reward.score_scene = canonical
        atomic_json(out, result)


if __name__ == '__main__':
    main()
