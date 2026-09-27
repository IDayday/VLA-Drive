import torch
from tools.local_interaction_mask_v2.graph_runtime import trajectory_metrics
from tools.local_interaction_mask_v2.compare_graphs import common_track_slots
from tools.local_interaction_mask_v2.audit_graphs import role_coverage


def test_static_group_uses_true_current_centre_not_detection_error():
    truth=torch.zeros(1,2,3,2);truth[:,1,:,0]=5.
    valid=torch.ones(1,2,3,dtype=torch.bool)
    predicted_current=torch.tensor([[[0.,0.],[6.5,0.]]])
    target_current=torch.tensor([[[0.,0.],[5.,0.]]])
    result=trajectory_metrics(truth.clone(),truth,valid,predicted_current,target_current)
    assert result['agents_static_valid_points']==3 and result['agents_dynamic_valid_points']==0
    assert result['agents_static_error_sum']==0 and result['agents_static_stationary_error_sum']==4.5
    # Missing coordinates are masked before arithmetic and do not alter groups.
    valid[0,1,1]=False;truth[0,1,1]=float('nan');truth[0,1,2,0]=6.1
    result=trajectory_metrics(truth.clone(),truth,valid,predicted_current,target_current)
    assert result['agents_dynamic_valid_points']==2 and result['agents_static_valid_points']==0
    assert result['agents_dynamic_error_sum']==0 and result['agents_dynamic_fde_count']==1


def test_common_cohort_uses_track_ids_not_local_slot_order():
    def sample(rows):return {'mapping':{'assignments':[dict(track_id=t,local_slot=s,used_for_local_loss=u) for t,s,u in rows]}}
    a=sample([('vehicle-a',1,True),('vehicle-b',2,True),('vehicle-c',3,False)])
    b=sample([('vehicle-b',1,True),('vehicle-c',2,True),('vehicle-a',3,True)])
    tracks,x,y=common_track_slots(a,b)
    assert tracks==['vehicle-a','vehicle-b'] and x==[0,1,2] and y==[0,3,1]


def test_coverage_distinguishes_static_risk_from_trajectory_and_image_context():
    nodes=[dict(source_slot_id=i,group=g) for i,g in enumerate(('A','B','C'))]
    association={'assignments':[dict(track_id=t,source_slot=i,association_accepted=True,used_for_local_loss=i==0)
        for i,t in enumerate(('car','barrier','pedestrian'))]}
    result=role_coverage(association,nodes,['car','barrier','pedestrian'],[0,4,1],[True]*3,[True]*3,[0,1,2])
    assert result['raw_motion_class_supported_relevant_targets']==2
    assert result['retained_motion_class_supported_relevant_targets']==1
    assert result['retained_context_class_in_risk_memory_targets']==1
    assert result['all_class_supported_relevant_trajectory_or_risk_targets']==2
