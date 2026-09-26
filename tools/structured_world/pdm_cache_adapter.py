"""Single-trajectory adapter for compact training caches with fixed PDM progress.

Uses the existing training PDMScorer and official transform/simulation helpers.
No metric formula is changed. Full caches retain the original NAVSIM entrypoint.
"""
from navsim.evaluate.pdm_score import pdm_score as full_cache_pdm_score,transform_trajectory,get_trajectory_as_array
from navsim.agents.EpisodeDrive.score_module.train_pdm_scorer import PDMScorer
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import MultiMetricIndex,WeightedMetricIndex
from navsim.common.dataclasses import PDMResults

def fixed_cache_pdm_score(metric_cache,model_trajectory,future_sampling,simulator,scorer):
    if not hasattr(metric_cache,'pdm_progress'):
        raise RuntimeError('Compact cache lacks the fixed PDM reference normalizer')
    trajectory=transform_trajectory(model_trajectory,metric_cache.ego_state)
    states=get_trajectory_as_array(trajectory,future_sampling,metric_cache.ego_state.time_point)
    simulated=simulator.simulate_proposals(states[None],metric_cache.ego_state)
    fixed_scorer=PDMScorer(future_sampling)
    scores=fixed_scorer.score_proposals(simulated,metric_cache.observation,metric_cache.centerline,metric_cache.route_lane_ids,metric_cache.drivable_area_map,metric_cache.pdm_progress)
    multi=fixed_scorer._multi_metrics;weighted=fixed_scorer._weighted_metrics
    return PDMResults(float(multi[MultiMetricIndex.NO_COLLISION,0]),float(multi[MultiMetricIndex.DRIVABLE_AREA,0]),float(weighted[WeightedMetricIndex.PROGRESS,0]),float(weighted[WeightedMetricIndex.TTC,0]),float(weighted[WeightedMetricIndex.COMFORTABLE,0]),float(weighted[WeightedMetricIndex.DRIVING_DIRECTION,0]),float(scores[0]))

def compatible_pdm_score(**kwargs):
    if hasattr(kwargs['metric_cache'],'trajectory'):return full_cache_pdm_score(**kwargs)
    return fixed_cache_pdm_score(**kwargs)
