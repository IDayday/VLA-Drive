import copy
import numpy as np
import torch
from tools.foresight.prepare_vehicle_trajectories import make_record,timed_frames
from starVLA.model.modules.vehicle_joint.graphs import VehicleGraphConfig


def frame(time=0.,ego_x=0.):
    transform=np.eye(4);transform[0,3]=ego_x
    return {'timestamp':int(time*1e6),'lidar2ego':np.eye(4),'ego2global':transform,
            'driving_command':[0,1,0,0],'ego_dynamic_state':[1.,0.,0.,0.],
            'anns':{'gt_names':np.array(['traffic_cone']*20+['vehicle','vehicle']),
                    'gt_boxes':np.array([[2,0,0,1,1,1,0]]*20+[[10-ego_x,0,0,4,2,1.5,0],[16-ego_x,1,0,4,2,1.5,0]],dtype=np.float32),
                    'track_tokens':[str(i) for i in range(22)]}}


def test_vehicle_filter_before_capacity_and_current_only_selection(monkeypatch):
    monkeypatch.setattr('tools.foresight.prepare_vehicle_trajectories.geometric_fov',lambda points,*_:np.ones(len(points),dtype=bool))
    current=frame();future=[frame(.5*(t+1),t+1) for t in range(8)]
    calibration=dict(intrinsics=[],extrinsics=[],distortion=[])
    cfg=VehicleGraphConfig(max_vehicles=2,max_context=0)
    r,a=make_record(current,future,calibration,cfg)
    assert a['source_vehicles']==2 and a['selected_vehicles']==2
    assert r['active'].sum()==3  # ego is separate, annotations do not add ego.
    for slot in [1,2]:assert torch.allclose(r['future'][slot],r['current'][slot,:2].expand(8,-1))
    altered=copy.deepcopy(future)
    for f in altered:f['anns']['gt_boxes'][:,0]+=1000
    q,b=make_record(current,altered,calibration,cfg)
    assert a['graph_audit']==b['graph_audit'] and torch.equal(r['current'],q['current'])


def test_timestamp_search_not_array_offset():
    frames=[frame(0),frame(.2),frame(.501),frame(1.0),frame(2.001),frame(4.2)]
    selected=timed_frames(frames,0,[.5,1,2,4],.05)
    assert [f['timestamp'] if f else None for f in selected]==[501000,1000000,2001000,None]
