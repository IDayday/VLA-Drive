"""Rescore sealed full-DEV ODE outputs with the unchanged v2 one-stage finalizer.

Whole logs remain together for adjacent-frame comfort. This is auxiliary
evaluation only; these results never enter candidate search or learning targets.
"""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch

from starVLA.rl.flow_grpo.transactions import atomic_json
from starVLA.rl.flow_grpo.reward import build_cache_index, reward_metadata
from starVLA.rl.safe_flow_improvement.contracts import file_sha, fingerprint
from starVLA.rl.safe_flow_improvement.iteration import require_dispatch
from starVLA.rl.safe_flow_improvement.evaluation_override import evaluation_seeds


def initialize(devkit):
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['NUPLAN_MAPS_ROOT'] = '/mnt/navsim/maps'
    os.environ['NUPLAN_MAP_VERSION'] = 'nuplan-maps-v1.0'
    sys.path.insert(0, devkit)
    import navsim
    if not Path(navsim.__file__).resolve().is_relative_to(Path(devkit)):
        raise RuntimeError('Refusing mixed v1/v2 module imports')
    torch.set_num_threads(1)


def score_log(job):
    from starVLA.rl.flow_grpo.evaluation import original_protocol_scores
    seed, tokens, requests, view, devkit, destination = job
    start = time.monotonic()
    frame = original_protocol_scores(tokens, requests, devkit, view)
    if len(frame) != len(tokens) or set(frame.token) != set(tokens) or not frame.valid.all() or not np.isfinite(frame.score).all():
        raise ValueError('v2 result changed the scene denominator or validity')
    path = Path(destination)
    temporary = path.with_suffix('.csv.incomplete')
    frame.to_csv(temporary, index=False)
    temporary.replace(path)
    return dict(seed=seed, scenes=len(tokens), csv=str(path), sha256=file_sha(path),
                aggregation=frame.attrs['aggregation'], CPU_seconds=time.monotonic()-start)


def main():
    p = argparse.ArgumentParser(__doc__)
    for name in ('samples', 'population', 'cache-root', 'output'):
        p.add_argument('--'+name, required=True)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--evaluation-override')
    a = p.parse_args()
    require_dispatch()
    if not 1 <= a.workers <= 16:
        raise ValueError('Bounded explicit v2 CPU resources required')
    root, out = Path(a.samples), Path(a.output)
    sample, population = json.loads((root/'manifest.json').read_text()), json.loads(Path(a.population).read_text())
    seeds, override = evaluation_seeds(population, a.evaluation_override)
    if (sample['cohort'] != 'dev_tokens' or sample['mode'] != 'ode' or sample['role'] != 'evaluation'
            or sample['population_identity'] != population['identity_sha256'] or not set(seeds) <= set(sample['seeds']) <= set(population['evaluation_seeds'])):
        raise ValueError('Auxiliary v2 requires the same complete fresh DEV ODE population')
    tokens = population['dev_tokens']
    if sample['tokens_sha256'] != fingerprint(tokens):
        raise ValueError('Sample population changed')
    repo = Path(__file__).resolve().parents[3]; devkit = repo/'navsim'
    from starVLA.rl.flow_grpo import evaluation
    protocol = reward_metadata(devkit, split='validation')
    protocol.update(version='navsim_v2_official_one_stage', aggregation_source_sha256=file_sha(evaluation.__file__),
                    missing_two_frame='official_unavailable_value_and_removed_denominator_weight_not_zero_imputation')
    identity = dict(sample_manifest=sample, protocol=protocol, script_sha256=file_sha(__file__))
    if override:
        identity.update(evaluation_override=override, selected_seeds=seeds)
    identity['sha256'] = fingerprint(identity)
    out.mkdir(parents=True, exist_ok=a.resume)
    if (out/'manifest.json').exists() and json.loads((out/'manifest.json').read_text()) != identity:
        raise ValueError('v2 resume changed source or policy identity')
    atomic_json(out/'manifest.json', identity)
    cache_index = build_cache_index(a.cache_root)
    if set(tokens)-cache_index.keys():
        raise FileNotFoundError('Missing v2 DEV cache; never substitute a v1 cache or score')
    groups = defaultdict(list); requests = {}; sample_hashes = {}; cache_hashes = {}
    for token in tokens:
        path = root/'scenes'/token[:2]/(token+'.pt')
        seal = json.loads(path.with_suffix('.json').read_text())
        if seal['identity_sha256'] != sample['sha256'] or seal['payload_sha256'] != file_sha(path):
            raise ValueError('Incomplete or changed ODE samples')
        row = torch.load(path, map_location='cpu', weights_only=True)
        requests[token] = row['requested_trajectories'].numpy()
        if requests[token].shape != (len(sample['seeds']), 8, 3):
            raise ValueError('Requested outputs must match the declared inference seeds')
        requests[token] = requests[token][[sample['seeds'].index(seed) for seed in seeds]]
        sample_hashes[token] = seal['payload_sha256']
        cache_hashes[token] = file_sha(cache_index[token])
        groups[population['token_logs'][token]].append(token)
    atomic_json(out/'input_payload_hashes.json', dict(samples=sample_hashes, metric_cache=cache_hashes))
    views = {}
    for log, selected in groups.items():
        view = out/'cache_views'/fingerprint(log)
        for token in selected:
            source = Path(cache_index[token]).resolve()
            link = view/source.parents[2].name/source.parents[1].name/token
            link.parent.mkdir(parents=True, exist_ok=True)
            if link.exists():
                if link.resolve() != source.parent:
                    raise ValueError('Read-only v2 cache view changed')
            else:
                link.symlink_to(source.parent, target_is_directory=True)
        views[log] = view
    shard_root = out/'logs'; shard_root.mkdir(exist_ok=True)
    jobs, receipts = [], []
    for seed_index, seed in enumerate(seeds):
        for log, selected in groups.items():
            path = shard_root/f'{seed}_{fingerprint(log)}.csv'
            marker = path.with_suffix('.json')
            if marker.exists():
                record = json.loads(marker.read_text())
                if record['identity_sha256'] != identity['sha256'] or file_sha(path) != record['sha256']:
                    raise ValueError('Saved v2 shard changed')
                receipts.append(record)
            else:
                jobs.append((seed, selected, np.stack([requests[t][seed_index] for t in selected]), str(views[log]), str(devkit), str(path)))
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=a.workers, mp_context=multiprocessing.get_context('spawn'),
                             initializer=initialize, initargs=(str(devkit),)) as executor:
        for result in executor.map(score_log, jobs):
            result['identity_sha256'] = identity['sha256']
            atomic_json(Path(result['csv']).with_suffix('.json'), result)
            receipts.append(result)
    frames = []
    for seed in seeds:
        frame = pd.concat([pd.read_csv(x['csv'], dtype={'token': str, 'log_name': str}, float_precision='round_trip')
                           for x in receipts if x['seed'] == seed], ignore_index=True)
        if len(frame) != len(tokens) or frame.token.duplicated().any() or set(frame.token) != set(tokens):
            raise ValueError('v2 log shards did not cover every DEV scene exactly once')
        frame['inference_seed'] = seed
        frame.to_csv(out/f'seed{seed}.csv', index=False); frames.append(frame)
    all_rows = pd.concat(frames, ignore_index=True)
    numeric = ['score', 'pdm_score', 'no_at_fault_collisions', 'drivable_area_compliance', 'driving_direction_compliance',
               'traffic_light_compliance', 'ego_progress', 'time_to_collision_within_bound', 'lane_keeping', 'history_comfort']
    report = dict(status='COMPLETE_OFFICIAL_V2_ONE_STAGE_AUXILIARY', identity_sha256=identity['sha256'],
                  scenes=len(tokens), seeds=seeds, protocol='navsim_v2_official_one_stage',
                  means={k: float(all_rows[k].mean()) for k in numeric},
                  two_frame_available_rows=int(all_rows.two_frame_extended_comfort.notna().sum()),
                  total_scene_seed_rows=len(all_rows), v1_score_not_derived_from_v2_pdm_score=True,
                  missing_two_frame_kept_unavailable=True, full_logs_kept_together=True,
                  no_parameter_updates_or_fit_statistics=True, logical_scoring_calls=len(tokens)*len(seeds),
                  new_scoring_calls=sum(len(job[1]) for job in jobs), CPU_seconds_sum=sum(x['CPU_seconds'] for x in receipts),
                  wall_seconds_this_scoring_invocation=time.monotonic()-started,
                  payload_hashes=dict(samples=fingerprint(sample_hashes), metric_cache=fingerprint(cache_hashes)))
    atomic_json(out/'report.json', report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
