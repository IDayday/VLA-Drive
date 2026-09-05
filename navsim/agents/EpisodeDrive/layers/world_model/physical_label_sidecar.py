"""Reuse scorer's exact rollout, then read-only geometry extraction outside scorer.

The old training cache serialized only polygon exteriors. Road labels therefore
read the original map, NEVER that lossy cached map. The exact scorer still reads
its original cache unaltered. No labels are attached to another candidate tuple.
"""
import hashlib
import inspect
import lzma
import os
from pathlib import Path
import pickle
import time
from functools import lru_cache

import numpy as np
from shapely.geometry import Point
from navsim.agents.EpisodeDrive.score_module import compute_navsim_score as scoring
from navsim.planning.simulation.planner.pdm_planner.observation.pdm_occupancy_map import PDMDrivableMap
from .minimal_physical_targets import (
    extract_minimal_physical_targets, logged_occupancy_frames, road_union, array_hash,
)


class RolloutTap:
    """Instance-local observer, not a monkey-patch; preserves first scorer rollout."""
    def __init__(self):
        self.simulator=scoring.PDMSimulator(scoring.proposal_sampling)
        self.proposal_sampling=self.simulator.proposal_sampling
        self.states=None
        self.calls=0

    def simulate_proposals(self, states, ego):
        result=self.simulator.simulate_proposals(states,ego)
        self.calls+=1
        if self.states is None:
            self.states=result
        return result


@lru_cache(maxsize=256)
def load_metric(path):
    source=Path(path)
    with lzma.open(source,'rb') as stream:
        metric=pickle.load(stream)
    return metric,hashlib.sha256(source.read_bytes()).hexdigest()


@lru_cache(maxsize=256)
def original_road_geometry(metric_path, map_name, map_root):
    from nuplan.common.maps.nuplan_map.map_factory import get_maps_api
    metric,_=load_metric(metric_path)
    center=metric.ego_state.center
    # Radius agrees with the queried region, NOT a hull of observed road polygons.
    radius=100.
    api=get_maps_api(map_root,'nuplan-maps-v1.0',map_name)
    fresh=PDMDrivableMap.from_simulation(api,metric.ego_state,map_radius=radius)
    union=road_union(fresh)
    coverage=Point(center.x,center.y).buffer(radius)
    return union,coverage,dict(map_name=map_name,map_version='nuplan-maps-v1.0',
        map_source='original_map_api_union_preserving_holes',
        road_union_sha256=None if union is None else hashlib.sha256(union.wkb).hexdigest(),
        map_coverage_sha256=hashlib.sha256(coverage.wkb).hexdigest())


def score_with_physical_sidecar(metric_path, proposals, gt, indices, map_name):
    """Return original four score arrays + labels for GT and seven sampled proposals.

    No proposal is simulated twice. GT is one extra rollout, never appended to the
    scorer's 64-candidate group. Large rollout arrays remain transient in worker.
    """
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_array_representation import ego_state_to_state_array
    from navsim.planning.metric_caching import train_cache_processor
    start=time.perf_counter()
    cache,cache_hash=load_metric(str(metric_path))
    if type(cache).__module__ != 'navsim.planning.metric_caching.train_metric_chache':
        raise ValueError('Lite training sidecar accepts audited train metric caches only, not Navtest')
    indices=np.asarray(indices)
    if proposals.shape != (64,8,3) or indices.shape != (7,) or len(set(indices.tolist())) != 7:
        raise ValueError('Sidecar requires the complete 64 group and seven distinct sample indices')
    tap=RolloutTap()
    scores=scoring.get_sub_score_from_metric_cache(cache,proposals,False,simulator_instance=tap)
    # Reuse identical GT if generated; otherwise simulate it once separately.
    matches=np.nonzero(np.all(proposals==gt[None],axis=(1,2)))[0]
    if len(matches):
        gt_states=tap.states[matches[:1]]
        gt_rollouts=0
    else:
        reference=scoring.transform_trajectory(scoring.Trajectory(gt),cache.ego_state)
        array=scoring.get_trajectory_as_array(reference,scoring.proposal_sampling,cache.ego_state.time_point)
        gt_states=tap.simulator.simulate_proposals(array[None],cache.ego_state)
        gt_rollouts=1
    candidates=np.concatenate((gt[None],proposals[indices]),0)
    states=np.concatenate((gt_states,tap.states[indices]),0)
    map_error=None
    try:
        union,coverage,map_meta=original_road_geometry(str(metric_path),map_name,os.environ['NUPLAN_MAPS_ROOT'])
    except (FileNotFoundError,KeyError,RuntimeError) as error:
        union,coverage=None,None
        map_error=f'{type(error).__name__}: {error}'
        map_meta=dict(map_name=map_name,map_source='unknown',map_error=map_error)
    source_conditions=dict(actor_source='logged_interpolated_10hz',
        cache_builder_sha256=hashlib.sha256(Path(inspect.getfile(train_cache_processor)).read_bytes()).hexdigest(),
        vehicle_parameters=dict(vars(cache.ego_state.car_footprint.vehicle_parameters)),
        simulator_sha256=hashlib.sha256(Path(inspect.getfile(scoring.PDMSimulator)).read_bytes()).hexdigest(),
        **map_meta)
    label_start=time.perf_counter()
    labels=extract_minimal_physical_targets(candidates,states,
        vehicle_parameters=cache.ego_state.car_footprint.vehicle_parameters,
        actor_frames=logged_occupancy_frames(cache.observation),drivable_union=union,map_coverage=coverage,
        centerline=cache.centerline,initial_ego_hash=array_hash(ego_state_to_state_array(cache.ego_state)),
        metric_cache_hash=cache_hash,source_conditions=source_conditions,
        red_light_token=cache.observation.red_light_token)
    labels['timing']=dict(label_extract_sec=time.perf_counter()-label_start,
        scorer_and_labels_sec=time.perf_counter()-start,proposal_rollouts=64,extra_gt_rollouts=gt_rollouts)
    return scores,labels
