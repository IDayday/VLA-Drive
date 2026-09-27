import copy
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from starVLA.model.modules.joint_world.observability import CurrentObservation
from starVLA.model.modules.joint_world.local_cache import cache_identity,build_payload,validate_payload,pack_payload


def test_local_cache_rejects_old_selector_noise_origin_and_retains_current_only():
    k=np.array([[600,0,512],[0,600,288],[0,0,1.]])
    ext=np.eye(4);ext[:3,:3]=[[0,0,1],[-1,0,0],[0,-1,0]];ext[:3,3]=[0,0,1.5]
    obs=CurrentObservation('scene1',np.stack([k]*3),np.stack([ext]*3),np.stack([np.eye(3)]*3),np.zeros((3,5)),
        ('CAM_F0','CAM_L0','CAM_R0'),(1024,576),100,(100,100,100),0.,'straight','fixed_current')
    config=json.loads(Path('configs/local_interaction_mask_v2/selector_v1.json').read_text())
    identity=cache_identity('foundation_digest',{'private_driving_weights_loaded':False},config,{'sampling_seed':2037})
    prediction={'boxes':torch.tensor([[[10.,0,1,4,2,1.5,0,1]]]),'logits':torch.tensor([[[5.,-5,-5,-5,-5,-5,-5,-5]]]),
        'agent_features':torch.randn(1,1,16),'scene_features':torch.randn(1,2,16),'references':torch.tensor([[[10.,0]]])}
    payload=build_payload(torch.randn(1,8,16),prediction,obs,identity)
    graph=validate_payload(payload,identity,'scene1');assert graph.active_actor_mask.sum()==2
    assert pack_payload(payload,identity)['local_graph'].source_slot_ids[0,1]==0
    for key,value in [('schema_version',1),('identity_sha256','old_graph_signature')]:
        bad=copy.deepcopy(payload);bad[key]=value
        with pytest.raises(ValueError):validate_payload(bad,identity,'scene1')
    changed=copy.deepcopy(identity);changed['graph_config']['sampling_seed']=4
    with pytest.raises(ValueError,match='identity mismatch'):validate_payload(payload,changed,'scene1')
    changed=copy.deepcopy(identity);changed['selector']['min_existence']=.1
    with pytest.raises(ValueError,match='identity mismatch'):validate_payload(payload,changed,'scene1')
    bad=copy.deepcopy(payload);bad['local_graph']['graph_provenance'][0]['targets_read']=True
    with pytest.raises(ValueError,match='provenance'):validate_payload(bad,identity,'scene1')
    with pytest.raises(ValueError,match='public-origin'):cache_identity('private',{},config,{})
