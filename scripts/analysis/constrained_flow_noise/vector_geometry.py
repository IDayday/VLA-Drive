"""Same GEOS body distances in vector calls; official scoring stays untouched."""
from collections import OrderedDict

import numpy as np

from starVLA.rl.constrained_flow_noise import safety_costs


_CONTEXTS=OrderedDict()


def static_context(road,observations):
    import shapely
    key=(id(road),id(observations))
    if key in _CONTEXTS:
        _CONTEXTS.move_to_end(key)
        return _CONTEXTS[key][2]
    if len(observations)!=41:raise ValueError('Expected 41 official object time slices')
    counts=np.array([len(objects) for objects in observations],dtype=np.int64)
    objects=np.array([g for group in observations for g in group],dtype=object)
    if len(objects) and (shapely.is_missing(objects).any() or shapely.is_empty(objects).any() or not shapely.is_valid(objects).all()):
        raise ValueError('Unknown/invalid object geometry cannot become empty')
    # Hold original objects strongly: IDs cannot be reused while cached. The
    # source context is immutable, scene-local official map/object geometry.
    value=(road.boundary,objects,counts,np.repeat(np.arange(41),counts))
    _CONTEXTS[key]=(road,observations,value)
    while len(_CONTEXTS)>8:_CONTEXTS.popitem(last=False)
    return value


def body_margins(body_coordinates,road,observations):
    import shapely
    coordinates=np.asarray(body_coordinates,dtype=np.float64)
    if coordinates.shape!=(41,5,2) or not np.isfinite(coordinates).all() or len(observations)!=41:
        raise ValueError('Expected aligned official 0.1s global vehicle bodies and objects')
    polygons=shapely.polygons(coordinates[:,:4])
    if not shapely.is_valid(polygons).all() or (shapely.area(polygons)<=0).any():
        raise ValueError('Invalid simulated vehicle body')
    boundary,objects,counts,body_indices=static_context(road,observations)
    inside=shapely.covers(road,polygons)
    road_margin=np.where(inside,shapely.distance(polygons,boundary),0.)
    outside=shapely.area(shapely.difference(polygons,road))
    object_margin=[None]*41
    if len(objects):
        distances=shapely.distance(polygons[body_indices],objects)
        # Original code uses Python min of the same ordered GEOS distances.
        offset=0
        for t,count in enumerate(counts):
            if count:object_margin[t]=float(min(distances[offset:offset+count]))
            offset+=count
    return {'physical_time_seconds':(np.arange(41)*.1).tolist(),
            'road_clearance_m':road_margin.tolist(),
            'road_margin_semantics':'nonnegative body clearance; zero outside or touching, not a signed penetration',
            'body_outside_union_area_m2':outside.tolist(),
            'object_distance_m':object_margin,'no_objects':(counts==0).tolist(),
            'semantics':'geometric proxies, not responsibility collision or official TTC probabilities'}


def initialize_worker(spec):
    """Apply only in this explicit worker process; no original file is edited."""
    from starVLA.rl.safe_flow_improvement import v1_reward
    v1_reward.initialize(spec)
    safety_costs.body_margins=body_margins
