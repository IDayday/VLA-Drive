# SPDX-License-Identifier: Apache-2.0
"""Reuse batch simulation while retaining canonical independent pair scoring.

The batching flow adapts DriveVLA-M0's existing
tools/navsim_candidate_relative_audit/score_candidates.py::score_pose_batch.
The full simulator, geometry and submetric implementations are imported from
this run's locked official NAVSIM v1.1. Only final proposal-pool aggregation is
partitioned into [official reference, one candidate] pairs. No custom
MetricCache.pdm_progress field, v2 scorer or global candidate maximum is used.
"""
import copy
import time

import numpy as np

from starVLA.rl.safe_flow_improvement import v1_reward


def independent_pair_aggregation(scorer, official_aggregate, progress_index):
    """Call the unchanged canonical aggregator on each exact two-row view.

This deliberately retains its zero-mask, >5m threshold, NC=0.5 treatment and
floating-point reduction order instead of approximating EP normalization.
The unused reference output is evaluated against itself; candidate outputs
always have their own reference-pair normalization.
"""
    multi = scorer._multi_metrics
    weighted = scorer._weighted_metrics.copy()
    progress = scorer._progress_raw
    scores = np.empty(multi.shape[1], dtype=np.float64)
    normalized = np.empty_like(scores)
    for index in range(len(scores)):
        pair = copy.copy(scorer)
        pair._num_proposals = 2
        pair._multi_metrics = multi[:, [0, index]].copy()
        pair._weighted_metrics = weighted[:, [0, index]].copy()
        pair._progress_raw = progress[[0, index]].copy()
        scores[index] = official_aggregate(pair)[1]
        normalized[index] = pair._weighted_metrics[progress_index, 1]
    scorer._weighted_metrics[progress_index] = normalized
    return scores


def canonical_batch_scorer(sampling):
    from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import WeightedMetricIndex

    class IndependentCandidateScorer(PDMScorer):
        def _aggregate_scores(self):
            return independent_pair_aggregation(self, PDMScorer._aggregate_scores, WeightedMetricIndex.PROGRESS)

    return IndependentCandidateScorer(sampling)


def score_scene(job, *, candidate_chunk=None):
    """Same canonical record interface; batch only within one immutable scene."""
    from navsim.common.dataclasses import Trajectory
    from navsim.evaluate.pdm_score import get_trajectory_as_array, transform_trajectory
    from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import EgoAreaIndex, MultiMetricIndex, WeightedMetricIndex
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

    token, log, requests = job
    requests = np.asarray(requests)
    if requests.ndim != 3 or requests.shape[1:] != (8, 3) or not len(requests) or not np.isfinite(requests).all():
        raise ValueError('Invalid request batch; never omit candidates or assign infrastructure reward zero')
    chunk = len(requests) if candidate_chunk is None else int(candidate_chunk)
    if chunk < 1:
        raise ValueError('candidate_chunk must be positive')
    cache, identity = v1_reward.load_cache(token, log)
    sampling = TrajectorySampling(num_poses=40, interval_length=.1)
    request_sampling = TrajectorySampling(num_poses=8, interval_length=.5)
    simulator, scorer = PDMSimulator(sampling), canonical_batch_scorer(sampling)
    initial = cache.ego_state
    reference_states = get_trajectory_as_array(cache.trajectory, sampling, initial.time_point)
    records = []
    for start in range(0, len(requests), chunk):
        started = time.monotonic()
        selected = requests[start:start + chunk]
        states = [reference_states]
        for requested in selected:
            transformed = transform_trajectory(Trajectory(requested, request_sampling), initial)
            states.append(get_trajectory_as_array(transformed, sampling, initial.time_point))
        simulated = simulator.simulate_proposals(np.stack(states), initial)
        scores = scorer.score_proposals(simulated, cache.observation, cache.centerline,
                                        cache.route_lane_ids, cache.drivable_area_map)
        elapsed = time.monotonic() - started
        for index, requested in enumerate(selected, 1):
            feedback = dict(
                no_at_fault_collisions=scorer._multi_metrics[MultiMetricIndex.NO_COLLISION, index],
                drivable_area_compliance=scorer._multi_metrics[MultiMetricIndex.DRIVABLE_AREA, index],
                ego_progress=scorer._weighted_metrics[WeightedMetricIndex.PROGRESS, index],
                time_to_collision_within_bound=scorer._weighted_metrics[WeightedMetricIndex.TTC, index],
                comfort=scorer._weighted_metrics[WeightedMetricIndex.COMFORTABLE, index],
                driving_direction_compliance=scorer._weighted_metrics[WeightedMetricIndex.DRIVING_DIRECTION, index],
                score=scores[index])
            if not all(np.isfinite(x) and 0 <= x <= 1 for x in feedback.values()):
                raise ValueError('Invalid canonical batch feedback')
            def event(value):
                return float(value) if np.isfinite(value) else None
            records.append(dict(token=token, log=log, status='OK', protocol=v1_reward.V1_PROTOCOL,
                                protocol_sha256=v1_reward.SPEC['protocol_sha256'], cache=identity, metrics=feedback,
                                requested_trajectory=requested.copy(), simulated_states=scorer._states[index].copy(),
                                simulated_body_coordinates=scorer._ego_coords[index].copy(),
                                non_drivable_body_by_time=scorer._ego_areas[index, :, EgoAreaIndex.NON_DRIVABLE_AREA].copy(),
                                official_first_at_fault_collision_seconds=event(scorer.time_to_at_fault_collision(index)),
                                official_first_ttc_infraction_seconds=event(scorer.time_to_ttc_infraction(index)),
                                raw_progress_meters=float(scorer._progress_raw[index]),
                                geometric_margins_status='NOT_YET_EXTRACTED_NOT_OFFICIAL_RISK_PROBABILITY',
                                # Amortized batch time; scene CPU/wall totals are
                                # measured by the unchanged safety worker wrapper.
                                scoring_seconds=elapsed / len(selected)))
    return records


def initialize_worker(spec):
    from .vector_geometry import initialize_worker as initialize_geometry
    initialize_geometry(spec)
    # This mutation is confined to the explicit scoring worker process. Its
    # identity includes this adapter; original source/protocol stays sealed.
    v1_reward.score_scene = score_scene
