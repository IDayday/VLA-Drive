"""Geometry fixtures only. No policy checkpoint, scene, metric cache or scoring."""
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from shapely.geometry import Polygon, box, MultiPolygon
from starVLA.model.modules.flow_posttraining.road_margin_proxy import (
    AnchoredDecode, CommandRoadCost, DiagnosticUndefined, Footprint, RoadDiagnosticSpec,
    RoadUnion, command_poses, direction_variants, executed_geometry,
    finite_difference_check, footprint_corners, periodic_difference, validate_raw)
from starVLA.dataloader.foresight_dataset import decode_ego, X_MEAN, X_STD, Y_MEAN, Y_STD


def raw_fixture():
    # Off-center, not parallel to a rectangular symmetry bisector: stable nearest
    # segment and projection regimes. This is NOT a real-scene efficacy test.
    x=torch.arange(1,9,dtype=torch.float64)*.6
    y=torch.arange(1,9,dtype=torch.float64)*.017+1.0
    h=torch.arange(1,9,dtype=torch.float64)*.003
    return torch.stack(((x-X_MEAN)/X_STD,(y-Y_MEAN)/Y_STD,h.sin(),h.cos()),-1)


V=Footprint(2.588,1.1485,1.461,3.089)


def objective_fixture(road=None):
    raw=raw_fixture(); decoder=AnchoredDecode(raw,decode_ego(raw))
    return raw,CommandRoadCost(decoder,road or RoadUnion([box(-10,-10,20,2.6)]),V,
                                torch.zeros(3,dtype=torch.float64),1000000)


def test_original_decoder_chain_rule():
    raw=raw_fixture().requires_grad_(True)
    decoded=decode_ego(raw)
    grad,=torch.autograd.grad(decoded.sum(),raw)
    radius=raw.detach()[:,2:].square().sum(-1)
    expected=torch.stack((torch.full_like(radius,X_STD),torch.full_like(radius,Y_STD),
                          raw.detach()[:,3]/radius,-raw.detach()[:,2]/radius),-1)
    torch.testing.assert_close(grad,expected,rtol=1e-14,atol=1e-14)


def test_anchor_exact_original_and_explicit_difference():
    raw=raw_fixture(); physical=decode_ego(raw); physical[3,0]+=2e-7;physical[4,2]+=1e-8
    decoder=AnchoredDecode(raw,physical)
    assert torch.equal(decoder(raw),physical)
    assert decoder.audit()['cpu_decode_max_xy_difference_m']>0
    grad_raw=raw.clone().requires_grad_(True)
    a,=torch.autograd.grad(decoder(grad_raw).sum(),grad_raw)
    b,=torch.autograd.grad(decode_ego(grad_raw).sum(),grad_raw)
    torch.testing.assert_close(a,b)


def official(physical,pose,time_us):
    root=Path(__file__).resolve().parents[2]/'navsim_v1.1/navsim'
    sys.path.insert(0,str(root))
    from nuplan.common.actor_state.ego_state import EgoState
    from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
    from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    from navsim.common.dataclasses import Trajectory
    from navsim.evaluate.pdm_score import transform_trajectory,get_trajectory_as_array
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_array_representation import state_array_to_coords_array
    vehicle=get_pacifica_parameters()
    state=EgoState.build_from_rear_axle(StateSE2(*pose.tolist()),StateVector2D(0,0),
        StateVector2D(0,0),0,TimePoint(time_us),vehicle)
    trajectory=transform_trajectory(Trajectory(physical.numpy(),TrajectorySampling(num_poses=8,interval_length=.5)),state)
    array=get_trajectory_as_array(trajectory,TrajectorySampling(num_poses=40,interval_length=.1),state.time_point)
    corners=state_array_to_coords_array(array[None],vehicle)[0,:,:4]
    return array[:,:3],corners


@pytest.mark.parametrize('time_us',[0,1000001,1630000000123456])
@pytest.mark.parametrize('origin_yaw',[0.,3.13,-3.12])
def test_official_interpolation_and_rear_axle_parity(time_us,origin_yaw):
    physical=decode_ego(raw_fixture());physical[:,2]=torch.tensor([3.12,-3.13,-3.,2.9,2.8,-2.9,-2.8,-2.7])
    origin=torch.tensor([1234.12,-430.3,origin_yaw],dtype=torch.float64)
    poses,_=command_poses(physical,origin,time_us)
    expected,corners=official(physical,origin,time_us)
    np.testing.assert_allclose(poses.detach().numpy(),expected,rtol=0,atol=1e-10)
    np.testing.assert_allclose(footprint_corners(poses,V).numpy(),corners,rtol=0,atol=1e-10)


def test_yaw_wrap_interpolation_uses_short_arc_and_t0_fixed():
    p=decode_ego(raw_fixture());p[:,2]=torch.tensor([3.12,-3.12,-3.1,-3.,-2.9,-2.8,-2.7,-2.6])
    initial=torch.tensor([3.,4.,3.13],dtype=torch.float64)
    poses,turns=command_poses(p,initial,1000001)
    assert poses.shape==(41,3); assert np.isfinite(turns).all()
    assert torch.equal(poses[0,:2],initial[:2])
    assert abs(float(periodic_difference(poses[0,2],initial[2])))<1e-14
    assert poses[:,2].abs().max()<=math.pi


def test_footprint_back_axle_and_corner_order():
    poses=torch.zeros(41,3,dtype=torch.float64)
    corners=footprint_corners(poses,V)
    expected=torch.tensor([[4.049,1.1485],[-1.127,1.1485],[-1.127,-1.1485],[4.049,-1.1485]],dtype=torch.float64)
    torch.testing.assert_close(corners[0],expected)


def test_holes_disconnected_union_and_overlap_seam():
    hole=Polygon([(0,0),(10,0),(10,10),(0,10)],holes=[[(4,4),(6,4),(6,6),(4,6)]])
    road=RoadUnion([hole,box(20,0,30,10)])
    points=torch.tensor([[2.,2.],[5.,5.],[15.,5.],[25.,5.]],dtype=torch.float64).repeat(41,1,1)
    distance,regime=road.distances(points)
    torch.testing.assert_close(distance[0],torch.tensor([2.,-1.,-5.,5.],dtype=torch.float64))
    assert len(road.geometry.geoms)==2
    # Adjacent polygons' shared boundary is inside union but strict-contains none.
    seam=RoadUnion([box(0,0,1,2),box(1,0,2,2)])
    p=torch.tensor([1.,1.],dtype=torch.float64).repeat(41,4,1)
    d,r=seam.distances(p)
    assert (d>0).all() and r['union_vs_any_polygon_contains'].all()


def test_fixed_denominator_and_t0_included():
    road=RoadUnion([box(-5,-5,5,5)])
    corners=torch.tensor([0.,4.9],dtype=torch.float64).repeat(41,4,1)
    cost,d,_=road.cost(corners)
    expected=(.1*torch.nn.functional.softplus((.1-d)/.1)).sum()/164
    assert torch.equal(cost,expected)


def test_full_raw_finite_difference_chain():
    raw,obj=objective_fixture();x=raw.clone().requires_grad_(True)
    loss,_=obj(x);g,=torch.autograd.grad(loss,x)
    checks=finite_difference_check(obj,raw,g)
    assert all(row['passed'] for row in checks),checks
    assert g[:,:2].abs().max()>0 and g[:,2:].abs().max()>0


def test_equal_norm_directions_deterministic_rng_isolation():
    raw,obj=objective_fixture(); state=torch.random.get_rng_state().clone()
    result=direction_variants(obj,raw,'fixture','f'*64)
    assert torch.equal(torch.random.get_rng_state(),state)
    again=direction_variants(obj,raw,'fixture','f'*64)
    for name in ['negative','positive','random']:
        assert result['variants'][name]['raw_l2_offset']==pytest.approx(.005,abs=1e-15)
        assert torch.equal(result['variants'][name]['raw'],again['variants'][name]['raw'])
    assert result['variants']['negative']['command_cost_m']<result['variants']['original']['command_cost_m']
    assert result['variants']['positive']['command_cost_m']>result['variants']['original']['command_cost_m']


def test_all_safe_saturated_cost_has_no_fabricated_direction():
    raw,obj=objective_fixture(RoadUnion([box(-1e6,-1e6,1e6,1e6)]))
    with pytest.raises(DiagnosticUndefined,match='ZERO_OR_NONFINITE'):
        direction_variants(obj,raw,'safe','a'*64)


def test_boundary_and_tie_are_explicit_not_smooth_claim():
    road=RoadUnion([box(-1,-1,1,1)])
    p=torch.tensor([0.,0.],dtype=torch.float64).repeat(41,4,1)
    _,regime=road.distances(p);assert regime['nearest_tie'].all()
    p=torch.tensor([1.,0.],dtype=torch.float64).repeat(41,4,1)
    _,regime=road.distances(p);assert regime['boundary'].all();assert not regime['inside'].any()


@pytest.mark.parametrize('kind',['nan','shape','singular','dtype'])
def test_invalid_raw_refused_without_repair(kind):
    raw=raw_fixture()
    if kind=='nan':raw[0,0]=float('nan')
    if kind=='shape':raw=raw[:7]
    if kind=='singular':raw[2,2:]=0
    if kind=='dtype':raw=raw.float()
    before=raw.clone()
    with pytest.raises(ValueError):validate_raw(raw)
    torch.testing.assert_close(raw,before,equal_nan=True)


def test_invalid_map_no_buffer_repair():
    with pytest.raises(ValueError):RoadUnion([Polygon([(0,0),(1,1),(1,0),(0,1)])])


def test_nondefault_constants_rejected():
    with pytest.raises(ValueError):RoadDiagnosticSpec(raw_l2_step=.001).validate()


def test_executed_cost_same_geometry_function():
    raw,obj=objective_fixture();cost,aux=obj(raw)
    states=np.zeros((41,11));states[:,:3]=aux['poses'].detach().numpy()
    result=executed_geometry(obj.road,obj.vehicle,states)
    assert result['custom_executed_cost_m']==float(cost)


def test_prepare_no_asset_reads_or_scoring(tmp_path,monkeypatch):
    from tools.st_world_policy import diagnose_road_direction as cli
    result=cli.prepare(cli.EVIDENCE)
    assert result['status']=='PREPARED_NOT_EXECUTED';assert result['scene_count']==32
    assert not result['dispatch_authorized'];assert result['new_official_calls']==0
    path=tmp_path/'prep.json'
    assert cli.main(['--mode','prepare','--output',str(path)])==0
    assert cli.main(['--mode','prepare','--output',str(path)])==2


def test_prepare_rejects_selection_change(tmp_path):
    import json
    from tools.st_world_policy import diagnose_road_direction as cli
    evidence=json.loads(cli.EVIDENCE.read_text());evidence['selected'][0]['scene_id']='foreign'
    path=tmp_path/'foreign.json';path.write_text(json.dumps(evidence))
    with pytest.raises(ValueError):cli.prepare(path)


def test_wrong_or_unadmitted_canonical_engine_is_rejected():
    from starVLA.model.modules.flow_posttraining.road_margin_proxy import score_diagnostic_variants
    engine=SimpleNamespace(spec=SimpleNamespace(allow_engineering_controls=False))
    with pytest.raises(ValueError,match='has not admitted'):
        score_diagnostic_variants(engine,None,None,None,None,None,'a'*64,None,None)


def test_scoring_adapter_charges_before_call_and_no_retry(monkeypatch):
    # Explicit fake backend verifies call-budget ordering ONLY, not native scores.
    from starVLA.model.modules.flow_posttraining.road_margin_proxy import score_diagnostic_variants
    from starVLA.model.modules.st_world import consequence as canonical
    raw,obj=objective_fixture(); directions=direction_variants(obj,raw,'fixture','f'*64)
    physical=directions['variants']['original']['physical'].numpy()
    request=canonical.ConsequenceRequest('campaign','scene','log','candidate','4095',
        'a'*64,'b'*64,canonical.array_sha256(physical),'c'*64,'d'*64,'/not-read',physical)
    states=np.zeros((41,11));states[:,:3]=directions['variants']['original']['command_poses'].numpy()
    record={'identity':request.identity(),'valid':True,'requested_trajectory':physical,'executed_states':states}
    monkeypatch.setattr(canonical,'validated_reference_input',lambda r,q:{'bound':'same-context'})
    events=[]
    class Fake:
        spec=SimpleNamespace(allow_engineering_controls=True)
        def score(self,q,expected_policy_version):
            events.append('score')
            return {'identity':q.identity(),'valid':False,'status':'TRANSIENT_SCORING_FAILURE'}
    def charge(identity):events.append('charge')
    saved=[]
    rows=score_diagnostic_variants(Fake(),request,record,directions,obj.road,V,'e'*64,charge,saved.append)
    assert events==['charge','score']*3
    assert len(saved)==4 and len(rows)==4
    assert rows['original']['original_reused']
    assert not rows['negative']['record']['valid']
    assert rows['negative']['geometry'] is None
    events.clear()
    def exhausted(identity):
        events.append('charge')
        raise RuntimeError('CALL_CAP')
    with pytest.raises(RuntimeError,match='CALL_CAP'):
        score_diagnostic_variants(Fake(),request,record,directions,obj.road,V,'e'*64,exhausted,lambda r:None)
    assert events==['charge']
