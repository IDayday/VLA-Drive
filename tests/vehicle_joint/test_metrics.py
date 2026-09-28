import numpy as np
import torch
import pytest
from tools.ddpolicy_vehicle.evaluate_vehicles import evaluate_scene
from tools.ddpolicy_vehicle.paired_results import log_bootstrap
from tools.ddpolicy_vehicle.evaluate_ego import relative_ego_target, trajectory_errors


def test_missed_vehicle_stays_in_denominator_and_motion_keeps_same_track():
    target=dict(current_boxes=torch.tensor([[5.,0,0,4,2,1,0,1],[10.,0,0,4,2,1,0,1]]),
        future_xy_in_ego_t0=torch.tensor([[[5.+i,0] for i in range(8)],[[10.,0] for _ in range(8)]]),
        future_valid_mask=torch.tensor([[True]*8,[True]*4+[False]*4]),
        current_supervision_mask=torch.ones(2,dtype=torch.bool),track_ids=['moving','parked'])
    prediction=dict(vehicle_logits=np.array([[5.,-5.]]),vehicle_boxes=np.array([[10.,0,0,4,2,1,0,1]]),
        vehicle_future_xy=np.array([[[10.,0]]*8]),vehicle_xy=np.array([[[10.,0]]*8]),
        selected_query_indices=np.array([-1,0]),active_actor_mask=np.array([True,True]))
    rows, counts=evaluate_scene(prediction,target)
    assert len(rows)==2 and counts['detected_vehicles']==1
    assert not rows[0]['detected'] and rows[0]['track_id']=='moving'
    assert rows[1]['joint_ADE']==0 and rows[1]['joint_FDE'] is None
    assert rows[1]['joint_last_valid_FDE']==0 and rows[1]['motion_group']=='stationary'
    failed,counts=evaluate_scene(None,target)
    assert len(failed)==2 and counts['detected_vehicles']==0


def test_cluster_bootstrap_retains_scene_weighting_and_one_log_limit():
    result=log_bootstrap([1.,1.,-1.],['a','a','b'],samples=100)
    assert result['mean']==1/3 and result['logs']==2
    assert log_bootstrap([1.,-1.],['a','a'])['ci95'] is None


def test_offline_ego_frame_and_wrapped_heading():
    poses=np.tile([100.,200.,np.pi/2],(12,1))
    poses[4:,1] += np.arange(1,9)
    target=relative_ego_target(poses)
    np.testing.assert_allclose(target[:,:2],np.column_stack((np.arange(1,9),np.zeros(8))),atol=1e-12)
    pred=target.copy();pred[:,2]+=2*np.pi
    assert trajectory_errors(pred,target)['yaw_MAE_rad']==0
    assert trajectory_errors(np.zeros((8,3)),target)['ADE']==4.5
    pred[0,0]=np.nan
    with pytest.raises(ValueError,match='Nonfinite'):trajectory_errors(pred,target)


def test_joint_relative_error_uses_same_sample_ego_not_independent_best():
    target=dict(current_boxes=torch.tensor([[5.,0,0,4,2,1,0,1]]),
        future_xy_in_ego_t0=torch.tensor([[[5.,0]]*8]),future_valid_mask=torch.ones(1,8,dtype=torch.bool),
        current_supervision_mask=torch.ones(1,dtype=torch.bool),track_ids=['vehicle'])
    prediction=dict(vehicle_logits=np.array([[5.,-5.]]),vehicle_boxes=np.array([[5.,0,0,4,2,1,0,1]]),
        vehicle_future_xy=np.array([[[5.,0]]*8]),vehicle_xy=np.array([[[5.,0]]*8]),
        selected_query_indices=np.array([-1,0]),active_actor_mask=np.array([True,True]),
        trajectory=np.tile([2.,0,0],(8,1)))
    rows,_=evaluate_scene(prediction,target,ego_target=np.zeros((8,3)))
    assert rows[0]['joint_ADE']==0
    assert rows[0]['joint_relative_vector_ADE']==2
    assert rows[0]['joint_pair_min_center_distance']==3
    rows,_=evaluate_scene(prediction,target,ego_target=np.zeros((8,3)),ego_valid=np.zeros(8,dtype=bool))
    assert rows[0]['joint_ADE']==0 and rows[0]['joint_relative_valid_points']==0
    assert 'joint_relative_vector_ADE' not in rows[0]
