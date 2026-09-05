import copy
import pickle
import numpy as np
import pytest
from shapely.geometry import box, Polygon, LineString, Point
from shapely.ops import unary_union
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from navsim.planning.simulation.planner.pdm_planner.observation.pdm_occupancy_map import PDMOccupancyMap
from navsim.agents.EpisodeDrive.layers.world_model.minimal_physical_targets import (
    TIME_BINS, gap_class_index, extract_minimal_physical_targets, label_identity,
    logged_occupancy_frames,
)


def fixture(k=1):
    poses = np.zeros((k, 8, 3), np.float32)
    states = np.zeros((k, 41, 11), np.float64)
    empty = PDMOccupancyMap([], [])
    kwargs = dict(vehicle_parameters=get_pacifica_parameters(), actor_frames=[empty]*50,
                  drivable_union=box(-100, -100, 100, 100), map_coverage=box(-200, -200, 200, 200),
                  centerline=LineString([(-100, 0), (100, 0)]), initial_ego_hash="ego",
                  metric_cache_hash="cache", source_conditions={"actor_source": "logged_interpolated_10hz"})
    return poses, states, kwargs


def test_bins_partition_all_41_source_times():
    assert [t for indices in TIME_BINS for t in indices] == list(range(41))
    assert TIME_BINS[0] == tuple(range(6))
    assert TIME_BINS[-1] == tuple(range(36, 41))


@pytest.mark.parametrize("distance,contact,expected", [(0., True, 0), (.5, False, 1),
    (.50001, False, 2), (2., False, 2), (2.0001, False, 3), (5., False, 3), (5.0001, False, 4),
    (0., False, 1)])
def test_gap_boundaries_and_contact_not_distance(distance, contact, expected):
    assert gap_class_index(distance, contact) == expected
    assert gap_class_index(5., False, no_actor=True) == 4


def test_fast_crossing_between_endpoints_and_read_only():
    p, s, kw = fixture()
    kw['actor_frames'][2] = PDMOccupancyMap(['pedestrian'], [box(-1, -1, 1, 1)])
    before = pickle.dumps(kw), s.copy(), np.random.get_state()
    result = extract_minimal_physical_targets(p, s, **kw)
    assert result['gap_class'][0, 0] == 0
    assert result['geometric_contact'][0, 0]
    assert pickle.dumps(kw) == before[0]
    np.testing.assert_array_equal(s, before[1])
    np.testing.assert_array_equal(np.random.get_state()[1], before[2][1])


def test_49th_frame_required_and_red_light_excluded():
    p, s, kw = fixture()
    kw['actor_frames'][49] = PDMOccupancyMap(['vehicle', 'red_light_x'], [box(-1,-1,1,1)]*2)
    result = extract_minimal_physical_targets(p, s, **kw)
    assert result['valid'][0, 7, 0] and result['gap_class'][0, 7] == 0
    kw['actor_frames'] = kw['actor_frames'][:41]
    result = extract_minimal_physical_targets(p, s, **kw)
    assert not result['valid'][0, 7, 0] and np.isnan(result['physical_values'][0,7,0])
    kw['actor_frames'] = [PDMOccupancyMap(['red_light_x'], [box(-1,-1,1,1)])]*50
    result = extract_minimal_physical_targets(p, s, **kw)
    assert result['no_actor'].all() and (result['gap_class'] == 4).all()


def test_union_internal_boundary_holes_corner_boundary_and_unknown():
    p, s, kw = fixture()
    kw['drivable_union'] = unary_union([box(-10,-10,10,0), box(-10,0,10,10)])
    a = extract_minimal_physical_targets(p,s,**kw)
    assert a['physical_values'][0,0,1] > 1  # seam y=0 is not an outer boundary
    kw['drivable_union'] = Polygon([(-10,-10),(10,-10),(10,10),(-10,10)],
                                   holes=[[(-2,-2),(5,-2),(5,2),(-2,2)]])
    b = extract_minimal_physical_targets(p,s,**kw)
    assert b['physical_values'][0,0,1] < 0
    vp = kw['vehicle_parameters']
    kw['drivable_union'] = box(-10,-vp.half_width,10,vp.half_width)
    assert extract_minimal_physical_targets(p,s,**kw)['physical_values'][0,0,1] == 0
    kw['map_coverage'] = None
    assert not extract_minimal_physical_targets(p,s,**kw)['valid'][...,1].any()


def test_center_not_rear_axle_and_progress_not_path_length():
    p,s,kw=fixture(2)
    s[0,:,2]=np.linspace(0,np.pi,41)  # rotate stationary rear axle: center moves backwards
    s[1,:,1]=np.sin(np.arange(41)/4)*3  # lateral travel makes no route progress
    result=extract_minimal_physical_targets(p,s,**kw)
    assert (result['physical_values'][...,2] == 0).all()
    s[0,:,0]=np.linspace(0,10,41)
    result=extract_minimal_physical_targets(p,s,**kw)
    expected=10-2*kw['vehicle_parameters'].rear_axle_to_center
    assert result['physical_values'][0,-1,2] == pytest.approx(expected)
    reversed_result=extract_minimal_physical_targets(p[::-1],s[::-1],**kw)
    np.testing.assert_equal(result['physical_values'][::-1], reversed_result['physical_values'])


def test_cache_key_binds_exact_coordinates_conditions():
    p,_,_=fixture()
    a=label_identity(p,'ego','cache')
    p[0,0,0]=.001
    assert label_identity(p,'ego','cache')['cache_key'] != a['cache_key']
    assert label_identity(p,'ego2','cache')['cache_key'] != label_identity(p,'ego','cache')['cache_key']


def test_resampled_occupancy_rejected():
    from types import SimpleNamespace
    with pytest.raises(ValueError,match='observation_sample_res'):
        logged_occupancy_frames(SimpleNamespace(_sample_interval=.1,_observation_sample_res=2))
