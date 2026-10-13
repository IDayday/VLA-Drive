"""Explicit geometric diagnostics on official simulated vehicle bodies.

These measurements are NOT collision responsibility, official TTC, or a risk
probability. Their time axis is physical driving time, never Flow steps.
"""
import numpy as np

from . import v1_reward


def geometric_feedback(job):
    token, log, bodies = job
    cache, cache_identity = v1_reward.load_cache(token, log)
    import shapely
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer

    indices = cache.drivable_area_map.get_indices_of_map_type([
        SemanticMapLayer.ROADBLOCK, SemanticMapLayer.INTERSECTION,
        SemanticMapLayer.DRIVABLE_AREA, SemanticMapLayer.CARPARK_AREA])
    road = shapely.union_all([cache.drivable_area_map._geometries[i] for i in indices])
    if road.is_empty or not road.is_valid:
        raise ValueError('Invalid drivable polygon union; do not fabricate margins')
    result = []
    for coordinates in bodies:
        coordinates = np.asarray(coordinates, dtype=np.float64)
        if coordinates.shape != (41, 5, 2) or not np.isfinite(coordinates).all():
            raise ValueError('Expected official 0.1-second global body corners and center')
        road_clearance, outside_area, object_distance, overlap_area = [], [], [], []
        for time_index, body in enumerate(coordinates):
            polygon = shapely.Polygon(body[:4])
            if not polygon.is_valid:
                raise ValueError('Invalid official simulated body polygon')
            road_clearance.append(float(polygon.distance(road.boundary)) if road.covers(polygon) else None)
            outside_area.append(float(polygon.difference(road).area))
            occupancy = cache.observation[time_index]
            # Ignore virtual traffic-light stop polygons. Only actual recorded
            # tracked objects are geometric distance targets.
            geometries = [occupancy._geometries[i] for i, key in enumerate(occupancy.tokens)
                          if key in cache.observation.unique_objects]
            if geometries:
                distances = shapely.distance(polygon, geometries)
                object_distance.append(float(np.min(distances)))
                intersections = shapely.intersection(polygon, np.asarray(geometries, dtype=object)[distances == 0])
                overlap_area.append(float(np.max(shapely.area(intersections))) if len(intersections) else 0.)
            else:
                object_distance.append(None)
                overlap_area.append(0.)
        result.append({'driving_time_seconds': (np.arange(41) * .1).tolist(),
                       'body_inside_road_boundary_clearance_m': road_clearance,
                       'body_outside_drivable_union_area_m2': outside_area,
                       'body_to_recorded_object_distance_m': object_distance,
                       'maximum_body_object_overlap_area_m2': overlap_area})
    return {'token': token, 'log': log, 'cache': cache_identity, 'candidates': result,
            'semantics': 'Geometry only; 0.1s official simulated global bodies and matching cached observation index. Clearance null outside union; object distance null when no object. Overlap is not at-fault collision; distance is not TTC.'}
