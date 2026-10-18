"""Additional independent geometry/gradient fixtures, no scientific scene IO."""
import math
import numpy as np
import pytest
import torch
from shapely.geometry import Point, Polygon, box
from starVLA.model.modules.flow_posttraining.road_margin_proxy import (
    AnchoredDecode, CommandRoadCost, DiagnosticUndefined, Footprint,
    RoadUnion, command_poses, direction_variants, footprint_corners,
    periodic_difference)
from starVLA.dataloader.foresight_dataset import decode_ego, X_MEAN, X_STD, Y_MEAN, Y_STD


def raw():
    x = torch.arange(1,9,dtype=torch.float64)*.63
    y = torch.arange(1,9,dtype=torch.float64)*.013+.91
    h = torch.arange(1,9,dtype=torch.float64)*.004
    return torch.stack(((x-X_MEAN)/X_STD,(y-Y_MEAN)/Y_STD,h.sin(),h.cos()),-1)


def test_original_point_has_zero_t0_gradient_with_nonzero_origin():
    value=raw().requires_grad_()
    decoder=AnchoredDecode(value.detach(),decode_ego(value.detach()))
    origin=torch.tensor([37.1,-4.2,2.8],dtype=torch.float64)
    poses,_=command_poses(decoder(value),origin,1630000000123456)
    gradient,=torch.autograd.grad(poses[0].sum(),value)
    assert torch.equal(gradient,torch.zeros_like(value))
    torch.testing.assert_close(poses[0],origin,rtol=0,atol=1e-15)


def test_ring_gradient_direction_in_hole_and_outside():
    polygon=Polygon([(0,0),(10,0),(10,10),(0,10)],
                    holes=[[(4,4),(6,4),(6,6),(4,6)]])
    road=RoadUnion([polygon])
    # Away from equidistant segments: cost should pull hole point left toward
    # road, and outer right point left toward road, with opposite signed d grads.
    corners=torch.tensor([4.2,5.1],dtype=torch.float64).repeat(41,4,1).requires_grad_()
    cost,distances,_=road.cost(corners)
    gradient,=torch.autograd.grad(cost,corners)
    assert torch.allclose(distances,torch.full_like(distances,-.2))
    assert (gradient[...,0]>0).all() and torch.equal(gradient[...,1],torch.zeros_like(gradient[...,1]))
    corners=torch.tensor([10.2,5.1],dtype=torch.float64).repeat(41,4,1).requires_grad_()
    cost,_,_=road.cost(corners)
    gradient,=torch.autograd.grad(cost,corners)
    assert (gradient[...,0]>0).all()


def test_distance_chunk_boundary_matches_shapely_full_geometry():
    theta=np.linspace(0,2*np.pi,2301)
    polygon=Polygon(np.stack((9*np.cos(theta),7*np.sin(theta)),axis=1))
    road=RoadUnion([polygon])
    assert len(road.segments)>2048
    rng=np.random.default_rng(619)
    points=rng.uniform([-12,-10],[12,10],size=(41,4,2))
    actual,_=road.distances(torch.tensor(points,dtype=torch.float64))
    expected=np.array([(1 if polygon.contains(Point(p)) else -1)*
                       polygon.boundary.distance(Point(p)) for p in points.reshape(-1,2)]).reshape(41,4)
    np.testing.assert_allclose(actual.numpy(),expected,atol=1e-12,rtol=1e-12)


def test_duplicate_road_polygon_does_not_double_objective():
    one=RoadUnion([box(-2,-3,11,4)])
    two=RoadUnion([box(-2,-3,11,4),box(-2,-3,11,4)])
    corners=torch.tensor([1.2,3.7],dtype=torch.float64).repeat(41,4,1)
    assert torch.equal(one.cost(corners)[0],two.cost(corners)[0])


def test_anchor_yaw_periodic_branch_preserves_derivative():
    value=raw();value[:,2]=math.sin(math.pi-1e-8);value[:,3]=math.cos(math.pi-1e-8)
    physical=decode_ego(value);physical[:,2]=-math.pi+2e-8
    decoder=AnchoredDecode(value,physical)
    assert torch.equal(decoder(value),physical)
    x=value.clone().requires_grad_()
    g,=torch.autograd.grad(decoder(x)[:,2].sum(),x)
    expected=torch.zeros_like(x)
    expected[:,2]=value[:,3]/value[:,2:].square().sum(-1)
    expected[:,3]=-value[:,2]/value[:,2:].square().sum(-1)
    torch.testing.assert_close(g,expected,atol=1e-14,rtol=1e-14)


def test_anchor_arbitrary_offset_is_currently_caller_admission_boundary():
    # Explicitly document why real-run approval requires a separate FP32-origin
    # discrepancy gate. The primitive alone is not such an admission guard.
    value=raw();physical=decode_ego(value);physical[:,0]+=100
    decoder=AnchoredDecode(value,physical)
    assert decoder.audit()['cpu_decode_max_xy_difference_m']==pytest.approx(100)
    assert torch.equal(decoder(value),physical)


def test_near_boundary_negative_cost_gradient_increases_margin():
    road=RoadUnion([box(-1,-2,10,3)])
    corners=torch.tensor([2.13,2.95],dtype=torch.float64).repeat(41,4,1).requires_grad_()
    cost,d,_=road.cost(corners)
    g,=torch.autograd.grad(cost,corners)
    moved=corners.detach()-.001*g/torch.linalg.vector_norm(g)
    next_cost,next_d,_=road.cost(moved)
    assert next_cost<cost and (next_d>d).all()


def test_geometric_equivariance_without_changing_action_definition():
    value=raw();physical=decode_ego(value)
    local,_=command_poses(physical,torch.zeros(3,dtype=torch.float64),0)
    origin=torch.tensor([100.,-50.,.7],dtype=torch.float64)
    global_poses,_=command_poses(physical,origin,0)
    vehicle=Footprint(2.588,1.1485,1.461,3.089)
    a=footprint_corners(local,vehicle)
    c,s=math.cos(.7),math.sin(.7)
    rotation=torch.tensor([[c,-s],[s,c]],dtype=torch.float64)
    expected=a@rotation.T+origin[:2]
    torch.testing.assert_close(footprint_corners(global_poses,vehicle),expected,atol=3e-14,rtol=0)


def test_cuda_remains_uninitialized():
    assert not torch.cuda.is_initialized()
