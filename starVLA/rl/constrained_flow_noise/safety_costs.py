"""Official event costs and separately named, bounded body-geometry proxies."""
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import multiprocessing
import numpy as np

from starVLA.rl.safe_flow_improvement import v1_reward
from .contracts import COSTS


def bounded_margin_cost(margins, critical, *, no_objects=None):
    """Input covers t=0..4.0 at 0.1s. Ignore immutable t0; unknown != empty."""
    values = np.asarray(margins, dtype=np.float64)
    if values.shape != (41,) or not np.isfinite(critical) or critical <= 0:
        raise ValueError('Expected 41 physical body margins and positive fixed threshold')
    empty = np.zeros(41, dtype=bool) if no_objects is None else np.asarray(no_objects, dtype=bool)
    if empty.shape != values.shape or not np.isfinite(values[~empty]).all():
        raise ValueError('Unknown geometry is not an empty scene')
    costs = np.zeros(41, dtype=np.float64)
    costs[~empty] = np.clip((critical-values[~empty])/critical, 0, 1)
    return float(costs[1:].mean())


def costs_from_record(record, thresholds):
    if record['status'] != 'OK' or record['protocol'] != v1_reward.V1_PROTOCOL:
        raise ValueError('Only valid canonical v1 feedback can enter updates')
    m = record['metrics']; geometry = record['geometry']
    binary = [float(m[name] < 1) for name in ('no_at_fault_collisions', 'drivable_area_compliance', 'time_to_collision_within_bound')]
    road = bounded_margin_cost(geometry['road_clearance_m'], thresholds['road_margin'])
    objects = bounded_margin_cost(geometry['object_distance_m'], thresholds['object_margin'], no_objects=geometry['no_objects'])
    return np.array(binary + [road, objects], dtype=np.float64)


@lru_cache(maxsize=8)
def _geometry_context(token, log):
    import shapely
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer
    cache, identity = v1_reward.load_cache(token, log)
    indices = cache.drivable_area_map.get_indices_of_map_type([
        SemanticMapLayer.ROADBLOCK, SemanticMapLayer.INTERSECTION,
        SemanticMapLayer.DRIVABLE_AREA, SemanticMapLayer.CARPARK_AREA])
    road = shapely.union_all([cache.drivable_area_map._geometries[i] for i in indices])
    if road.is_empty or not road.is_valid:
        raise ValueError('Missing/invalid union of official drivable polygons')
    observations = []
    for t in range(41):
        occupancy = cache.observation[t]
        # Every physical object in the official observation, irrespective of
        # semantic type. Virtual traffic-light stop-line polygons are not bodies.
        observations.append([occupancy._geometries[i] for i,key in enumerate(occupancy.tokens)
                             if key in cache.observation.unique_objects])
    return road, observations, identity


def body_margins(body_coordinates, road, observations):
    import shapely
    coordinates = np.asarray(body_coordinates, dtype=np.float64)
    if coordinates.shape != (41, 5, 2) or not np.isfinite(coordinates).all() or len(observations) != 41:
        raise ValueError('Expected aligned official 0.1s global vehicle bodies and objects')
    road_margin, object_margin, empty, outside = [], [], [], []
    for t, body in enumerate(coordinates):
        polygon = shapely.Polygon(body[:4])
        if not polygon.is_valid or polygon.area <= 0:
            raise ValueError('Invalid simulated vehicle body')
        inside = road.covers(polygon)
        # Outside/touching returns zero clearance and unit proxy cost; a
        # negative geometric penetration proxy is deliberately not invented.
        road_margin.append(float(polygon.distance(road.boundary)) if inside else 0.)
        outside.append(float(polygon.difference(road).area))
        objects = observations[t]
        if any(g is None or g.is_empty or not g.is_valid for g in objects):
            raise ValueError('Unknown/invalid object geometry cannot become empty')
        empty.append(not bool(objects))
        object_margin.append(float(min(polygon.distance(g) for g in objects)) if objects else None)
    return {'physical_time_seconds': (np.arange(41)*.1).tolist(),
            'road_clearance_m': road_margin, 'road_margin_semantics': 'nonnegative body clearance; zero outside or touching, not a signed penetration',
            'body_outside_union_area_m2': outside, 'object_distance_m': object_margin, 'no_objects': empty,
            'semantics': 'geometric proxies, not responsibility collision or official TTC probabilities'}


def score_scene(job):
    """Reuse canonical isolated candidate scoring without changing the proposal pool."""
    records = v1_reward.score_scene(job)
    road, observations, identity = _geometry_context(job[0], job[1])
    for record in records:
        if record['cache'] != identity:
            raise ValueError('Score and geometric consequences used different cache assets')
        record['geometry'] = body_margins(record['simulated_body_coordinates'], road, observations)
    return records


class ConstrainedRewardService:
    def __init__(self, spec, workers=2):
        self.spec = dict(spec)
        self.spec['protocol_sha256'] = v1_reward.protocol_identity(spec['repo'])['sha256']
        self.pool = ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'),
                                        initializer=v1_reward.initialize, initargs=(self.spec,))

    def score(self, jobs):
        return list(self.pool.map(score_scene, jobs, chunksize=1))

    def close(self):
        self.pool.shutdown(wait=True, cancel_futures=True)
