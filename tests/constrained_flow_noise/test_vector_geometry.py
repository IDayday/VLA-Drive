import numpy as np
import pytest
import shapely

from scripts.analysis.constrained_flow_noise.vector_geometry import body_margins
from starVLA.rl.constrained_flow_noise.safety_costs import body_margins as original


def coordinates():
    values=[]
    for t in range(41):
        x=t*.3-2
        values.append([[x,-.5],[x+1,-.5],[x+1,.5],[x,.5],[x+.5,0]])
    return np.array(values,dtype=np.float64)


@pytest.mark.parametrize('road',[
    shapely.box(-3,-3,15,3),
    shapely.Polygon([(-3,-3),(15,-3),(15,3),(-3,3)],holes=[[(2,-1),(3,-1),(3,1),(2,1)]]),
    shapely.union_all([shapely.box(-3,-3,5,0),shapely.box(-3,0,5,3)]),
    shapely.MultiPolygon([shapely.box(-3,-3,0,3),shapely.box(4,-3,15,3)])])
def test_same_body_clearance_holes_union_outside_and_objects(road):
    observations=[[] if t%3==0 else [shapely.box(3+t*.1,-.2,4+t*.1,.8),shapely.box(15,2,16,3)] for t in range(41)]
    assert body_margins(coordinates(),road,observations)==original(coordinates(),road,observations)
    # Exercise immutable context reuse as well as initial preparation.
    assert body_margins(coordinates(),road,observations)==original(coordinates(),road,observations)


@pytest.mark.parametrize('bad',[None,shapely.Polygon(),shapely.Polygon([(0,0),(1,1),(0,1),(1,0)])])
def test_unknown_object_never_becomes_empty(bad):
    observations=[[] for _ in range(41)];observations[7]=[bad]
    with pytest.raises(ValueError):body_margins(coordinates(),shapely.box(-5,-5,20,5),observations)


def test_invalid_body_rejected():
    value=coordinates();value[1]=0
    with pytest.raises(ValueError):body_margins(value,shapely.box(-5,-5,20,5),[[] for _ in range(41)])
