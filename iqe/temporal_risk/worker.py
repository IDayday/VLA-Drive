"""Read official traces after the unchanged single-candidate reference call."""
import argparse
import hashlib
import importlib
import lzma
import pickle
import random
import sys
import time
from pathlib import Path

import numpy as np

from iqe.contracts import require
from iqe.io import atomic_json, file_hash, read_json
from iqe.internal_baseline.score_worker import score_item
from .labels import event_labels


def official_events(request, response):
    r = read_json(request)
    scene, candidate = r['scene'], r['candidate']
    require(scene['split_role'] == 'incremental_fit', 'diagnostic labels are D_S training-only')
    sys.path.insert(0, r['nuplan_root'])
    sys.path.insert(0, r['navsim_root'])
    pdm = importlib.import_module('navsim.evaluate.pdm_score')
    require(Path(pdm.__file__).resolve().is_relative_to(Path(r['navsim_root']).resolve()), 'wrong NAVSIM')
    from navsim.common.dataclasses import Trajectory
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer, PDMScorerConfig
    from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import EgoAreaIndex
    from iqe.scoring.reference import COMPONENTS
    random.seed(r['seed']); np.random.seed(r['seed'])
    # Reload every candidate, including its mutable observation. Never score a
    # growing candidate pool or reuse another candidate's simulation response.
    with lzma.open(scene['metric_context_ref'], 'rb') as stream:
        cache = pickle.load(stream)
    with np.load(candidate['trajectory_ref']) as f:
        poses = f['trajectory'].astype(np.float64)
    require(poses.shape == (candidate['horizon'], 3) and np.isfinite(poses).all(), 'invalid physical trajectory')
    sampling = TrajectorySampling(**r['config']['proposal_sampling'])
    future = TrajectorySampling(num_poses=candidate['horizon'], interval_length=candidate['dt'])
    reference = pdm.get_trajectory_as_array(cache.trajectory, sampling, cache.ego_state.time_point)
    reference_hash = hashlib.sha256(reference.tobytes()).hexdigest()
    scorer = PDMScorer(sampling, PDMScorerConfig(**r['config']['scorer']))
    result = pdm.pdm_score(cache, Trajectory(poses, future), sampling, PDMSimulator(sampling), scorer)
    components = {k: float(getattr(result, k)) for k in COMPONENTS}
    old = r['expected_label']
    require(old['score_valid'] and old['official_reference_hash'] == reference_hash, 'reference identity changed')
    require(old['metric_context_hash'] == scene['metric_context_hash'] and
            old['score_protocol_hash'] == r['protocol_hash'], 'stale bank label')
    errors = {'total': abs(float(result.score)-old['total_score_01'])}
    errors.update({k: abs(components[k]-old['named_components'][k]) for k in COMPONENTS})
    require(max(errors.values()) <= 1e-10, 'official total/components differ from original bank')
    require(scorer._states.shape[0] == 2, 'must contain official reference + exactly one candidate')
    labels = event_labels(scorer._collision_time_idcs[1],
        scorer._ego_areas[1, :, EgoAreaIndex.NON_DRIVABLE_AREA], scorer._ttc_time_idcs[1],
        simulator_dt=sampling.interval_length, horizon=candidate['horizon'], dt=candidate['dt'],
        components=components)
    atomic_json(response, dict(schema='official_temporal_event_supervision_v1', score_valid=True,
        candidate_key=old['candidate_key'], scene_id=scene['scene_id'],
        trajectory_hash=candidate['trajectory_hash'], metric_context_hash=scene['metric_context_hash'],
        official_reference_hash=reference_hash, protocol_hash=r['protocol_hash'],
        original_bank_score_max_abs_error=max(errors.values()), total_score_01=float(result.score),
        components=components, labels=labels, supervision_only=True, inference_input=False,
        worker_hash=file_hash(__file__)))


def run(job_path, shard, shards):
    job = read_json(job_path)
    require(0 <= shard < shards <= job['worker_limit'] <= 8, 'worker budget')
    require(file_hash(__file__) == job['source_hashes']['worker.py'] and
            file_hash(Path(__file__).with_name('labels.py')) == job['source_hashes']['labels.py'], 'worker source changed')
    from iqe.specialist_study.resource_guard import check
    from types import SimpleNamespace
    check(SimpleNamespace(root=Path(job_path).parent), 'research-preflight')
    done, failed, start = 0, [], time.perf_counter()
    for item in job['items'][shard::shards]:
        if score_item(job, item, official_events):
            done += 1
        else:
            failed.append(item['request']['expected_label']['candidate_key'])
        atomic_json(Path(job_path).parent/f'worker_{shard}.json', dict(
            done=done, failed=failed, seconds=time.perf_counter()-start, status='RUNNING'))
    atomic_json(Path(job_path).parent/f'worker_{shard}.json', dict(
        done=done, failed=failed, seconds=time.perf_counter()-start,
        status='COMPLETE' if not failed else 'PARTIAL'))
    require(not failed, 'official temporal labeling incomplete; retain attempts and retry')


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('job'); p.add_argument('--shard', type=int, required=True)
    p.add_argument('--shards', type=int, required=True)
    a = p.parse_args(); run(a.job, a.shard, a.shards)
