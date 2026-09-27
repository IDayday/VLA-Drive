import json
from pathlib import Path
import numpy as np
import torch
import pytest
from starVLA.model.modules.joint_world.observability import CurrentObservation,measure_observability
from starVLA.model.modules.joint_world.local_graph import build_local_graph,gather_current,rule_identity
from starVLA.model.modules.joint_world.local_targets import local_targets
from starVLA.model.modules.structured_world.contracts import WorldTargets


def inputs():
    k=np.array([[600,0,512],[0,600,288],[0,0,1.]])
    ext=np.eye(4);ext[:3,:3]=np.array([[0,0,1],[-1,0,0],[0,-1,0]]);ext[:3,3]=[0,0,1.5]
    observation=CurrentObservation('synthetic',np.stack([k]*3),np.stack([ext]*3),np.stack([np.eye(3)]*3),np.zeros((3,5)),
        ('CAM_F0','CAM_L0','CAM_R0'),(1024,576),100,(100,100,100),0.,'straight','synthetic')
    config=json.loads(Path('configs/local_interaction_mask_v2/selector_v1.json').read_text())
    return observation,config


def detections(boxes,classes=None):
    boxes=np.asarray(boxes,dtype=float).reshape(-1,8);logits=np.full((len(boxes),8),-5.)
    for i,c in enumerate(classes if classes is not None else [0]*len(boxes)):logits[i,c]=5
    return boxes,logits


def test_real_relations_second_hop_and_weak_exclusion():
    obs,cfg=inputs();boxes,logits=detections([[13,0,1,4,2,1.5,0,1],[28,0,1,4,2,1.5,0,1],[47,16,1,4,2,1.5,0,1]])
    g=build_local_graph(boxes,logits,obs,cfg)
    assert g.graph_provenance[0]['direct_count']==1 and g.graph_provenance[0]['second_hop_count']==1
    assert g.edge_mask[0,0,1] and g.edge_mask[0,1,2] and not g.edge_mask[0,0,2]
    assert g.node_records[0][2]['group']=='C'
    assert g.node_records[0][0]['speed_known'] is False


def test_low_evidence_risk_and_static_context_are_not_padding():
    obs,cfg=inputs();boxes,logits=detections([[12,0,1,1,.2,.2,0,1],[8,2,.5,.5,.5,1,0,1]],classes=[0,3])
    g=build_local_graph(boxes,logits,obs,cfg)
    assert g.active_actor_mask.sum()==1 and g.context_only_mask.sum()==2
    assert all(r['group']=='B' and not r['confirmed_free_space'] for r in g.node_records[0])
    current=dict(actor_features=torch.randn(1,3,16),context=torch.randn(1,4,16),current_xy=torch.randn(1,3,2),existence=torch.ones(1,3))
    packed=gather_current(current,g)
    assert torch.equal(packed['context'][:,:4],current['context'])
    assert torch.equal(packed['context'][:,4:],current['actor_features'][:,1:])
    assert packed['context_mask'].sum()==6


def test_duplicate_capacity_empty_and_unknown_speed():
    obs,cfg=inputs();boxes,logits=detections([[10,0,1,4,2,1.5,0,1],[10.1,0,1,4,2,1.5,0,1]])
    g=build_local_graph(boxes,logits,obs,cfg)
    assert g.active_actor_mask.sum()==2 and g.node_records[0][1]['group']=='D'
    assert g.node_records[0][1]['reason']=='duplicate'
    cfg=dict(cfg,max_non_ego=1,max_direct_neighbors=1)
    boxes,logits=detections([[10,0,1,4,2,1.5,0,1],[13,1,1,4,2,1.5,0,1]])
    g=build_local_graph(boxes,logits,obs,cfg)
    assert g.active_actor_mask.sum()==2 and g.context_only_mask.sum()==1
    boxes,logits=detections([]);g=build_local_graph(boxes,logits,obs,cfg)
    assert g.active_actor_mask.sum()==1 and g.edge_mask[0,0,0]


def test_no_gt_arguments_and_calibration_identity():
    obs,cfg=inputs();boxes,logits=detections([[10,0,1,4,2,1.5,0,1]])
    with pytest.raises(TypeError):build_local_graph(boxes,logits,obs,cfg,future_valid_mask=np.ones((1,8)))
    before=obs.fingerprint();obs.intrinsics[0,0,0]+=1
    assert obs.fingerprint()!=before
    assert rule_identity(cfg)!=rule_identity(dict(cfg,min_pixel_area=cfg['min_pixel_area']+1))
    invalid=CurrentObservation(**dict(vars(obs),camera_timestamps=(101,100,100)))
    with pytest.raises(ValueError):build_local_graph(boxes,logits,invalid,cfg)


def test_source_local_track_mapping_and_missing_future():
    obs,cfg=inputs();boxes,logits=detections([[13,0,1,4,2,1.5,0,1],[28,0,1,4,2,1.5,0,1]])
    g=build_local_graph(boxes,logits,obs,cfg);before=g.source_slot_ids.clone()
    truth=torch.tensor(boxes,dtype=torch.float32)
    target=WorldTargets(truth,torch.zeros(2,dtype=torch.long),('carA','carB'),torch.zeros(2,3,2),
        torch.tensor([[True,False,True],[False,False,False]]),torch.ones(2,dtype=torch.bool),torch.tensor(True),
        torch.ones(2,8,dtype=torch.bool),torch.tensor([0.,-20,50,20]))
    target.future_xy_in_ego_t0[0,1]=float('nan')
    prediction={'boxes':truth[None].clone(),'logits':torch.tensor(logits,dtype=torch.float32)[None]}
    xy,valid,records=local_targets(prediction,g,[target],torch.zeros(1,3,2))
    assert torch.equal(g.source_slot_ids,before) and valid[0,1].tolist()==[True,False,True]
    assert not valid[0,2].any() and torch.isfinite(xy).all()
    assert records[0]['assignments'][0]['track_id']=='carA'
    # A bad full-slot association is rejected; it never reassigns to fabricate a future.
    target.current_boxes[1,:2]+=10
    _,valid,records=local_targets(prediction,g,[target],torch.zeros(1,3,2))
    assert not valid[0,2].any() and records[0]['filtered_current_associations']==1
