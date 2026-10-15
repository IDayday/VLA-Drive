from types import SimpleNamespace as S
import numpy as np
import pytest
from shapely.geometry import Polygon
from iqe.decision_selection.current_state import extract
from iqe.decision_selection.privileged_probe import normalization,StructuredContext
import torch


def fixture():
    k=np.array([[500.,0,512.],[0,500.,288.],[0,0,1.]])
    e=np.eye(4);e[:3,:3]=np.array([[0.,0.,1.],[-1.,0.,0.],[0.,-1.,0.]])
    e[:3,3]=[0,0,1.6]
    obs=dict(timestamp=1000,global_pose_history=[[0.,0.,0.]],
        current_calibration=dict(intrinsics=[k]*3,extrinsics=[e]*3))
    polygon=Polygon([(8,-1),(12,-1),(12,1),(8,1)])
    def obj(timestamp):
        return S(metadata=S(timestamp_us=timestamp),center=S(x=10.,y=0.,heading=0.),
            box=S(geometry=polygon,length=4.,width=2.,height=1.5),velocity=S(x=3.,y=0.),tracked_object_type=S(value=0))
    class Observation:
        unique_objects={'current':obj(1000),'future':obj(1500)}
        def __getitem__(self,i):
            assert i==0,'future accessed'
            class Occupancy:
                tokens=['current','future']
                def __getitem__(self,token):return polygon
            return Occupancy()
    class Cache:
        ego_state=S(time_point=S(time_us=1000),rear_axle=S(x=0.,y=0.,heading=0.),
            dynamic_car_state=S(rear_axle_velocity_2d=S(x=4.,y=0.),rear_axle_acceleration_2d=S(x=0.,y=0.)))
        observation=Observation()
        drivable_area_map=S(tokens=['lane'],map_types=[S(value=0)],_geometries=[Polygon([(0,-3),(30,-3),(30,3),(0,3)])])
        @property
        def trajectory(self):raise AssertionError('future reference accessed')
        @property
        def centerline(self):raise AssertionError('reference-derived centerline accessed')
    return Cache(),obs


def test_timestamp_filter_rejects_future_object_repeated_into_t0():
    cache,obs=fixture();values,audit=extract(cache,obs)
    assert values['agent_valid'].sum()==1
    assert audit['future_only_objects_rejected']==1
    assert audit['future_occupancy_indices_read']==[] and not audit['reference_or_GT_trajectory_read']
    cache.observation.unique_objects['future'].center=S(x=123.,y=456.,heading=4.)
    other,_=extract(cache,obs)
    assert all(np.array_equal(v,other[k]) for k,v in values.items())
    with pytest.raises(ValueError):extract(cache,obs|dict(timestamp=999))


def test_probe_normalization_fit_only_and_masked_sets():
    cache,obs=fixture();one,_=extract(cache,obs)
    values={k:np.stack([v,v]) for k,v in one.items()}
    first=normalization(values,[0])
    values['agents'][1]*=1000;values['map_points'][1]*=1000;values['current_ego'][1]*=1000
    assert normalization(values,[0])==first
    m=StructuredContext(first)
    context={k:torch.from_numpy(v[:1]) for k,v in values.items()}
    memory,mask=m(context)
    assert memory.shape==(1,321,256) and mask.shape==(1,321)
    context['GT']=torch.zeros(1)
    with pytest.raises(ValueError):m(context)
