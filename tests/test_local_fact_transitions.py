from dataclasses import replace
import numpy as np
import pytest
from starVLA.model.modules.decision_local.local_evidence import LocalEvidenceBatch,compare_local_evidence
from tools.decision_local.evaluate_minimal_repair_behavior import plan_changes,aggregate_local_changes


def facts(states,source='FUTURE'):
    states=np.asarray(states);positive=np.repeat((states==0)[:,None],3,axis=1)
    negative=np.repeat((states==1)[:,None],3,axis=1);valid=positive|negative
    return LocalEvidenceBatch(np.zeros(valid.shape,np.float32),valid,positive,negative,
        (~valid).astype(np.uint16),np.arange(len(states))*.5,
        np.full(valid.shape,'actor',object),valid[:,0],valid[:,0],(),source)


def test_every_transition_is_disjoint_and_unknown_never_becomes_repair():
    result=compare_local_evidence(facts([0,0,0,1,1,1,2,2,2]),facts([0,1,2,0,1,2,0,1,2]))
    transitions=result['channels']['annotated_overlap']['transitions']
    assert all(r['sample_count']==1 for r in transitions.values())
    assert transitions['positive_to_negative']['physical_times_s']==[.5]
    assert transitions['positive_to_unknown']['physical_times_s']==[1.]
    assert transitions['negative_to_positive']['physical_times_s']==[1.5]
    assert transitions['unknown_to_positive']['physical_times_s']==[3.]
    assert sum(r['sample_count'] for r in transitions.values())==9


def test_known_positive_is_not_filtered_by_other_unknown_times():
    result=compare_local_evidence(facts([0,2,2]),facts([1,2,2]))
    transitions=result['channels']['road_violation']['transitions']
    assert transitions['positive_to_negative']['sample_count']==1
    assert transitions['unknown_to_unknown']['sample_count']==2


def test_geometry_without_free_evidence_is_unknown_after_avoiding_known_actor():
    from shapely.geometry import box
    from starVLA.dataloader.minimal_repair_scene import AnnotatedScene
    from starVLA.model.modules.decision_local.local_evidence import extract_local_evidence
    actor=dict(corners=np.array([[3.,-1],[5.,-1],[5.,1],[3.,1]]),
               velocity=np.zeros(2),reliable=True,stationary=True)
    scene=AnnotatedScene(road=box(-20,-20,80,20),
        body=dict(length=4.,width=2.,rear_axle_to_center=1.),
        frames=[{'a':actor} for _ in range(9)],observed=np.zeros((9,160,200),bool),road_height_known=True)
    before=extract_local_evidence(scene,np.array([[4.,0,0]]),[1.])
    after=extract_local_evidence(scene,np.array([[4.,6,0]]),[1.])
    result=compare_local_evidence(before,after)['channels']['annotated_overlap']['transitions']
    assert result['positive_to_unknown']['sample_count']==1
    assert result['positive_to_negative']['sample_count']==0
    # Conditional distance improves but cannot create a whole-scene negative.
    assert compare_local_evidence(before,after)['continuous']['annotated_object_distance_m']['mean_change_m']>0


def test_empty_continuous_support_does_not_fabricate_zero_error():
    result=compare_local_evidence(facts([2,2]),facts([2,2]))
    assert result['continuous']['annotated_object_distance_m']['mean_change_m'] is None
    assert result['continuous']['road_margin_m']['paired_samples']==0


def test_time_and_source_must_match():
    before=facts([0,1])
    with pytest.raises(ValueError):compare_local_evidence(before,facts([0,1],source='CV'))
    with pytest.raises(ValueError):compare_local_evidence(before,replace(before,physical_times=np.array([0.,1.])))


def test_sample_counts_do_not_inflate_scene_log_counts():
    change=compare_local_evidence(facts([0,0,0]),facts([1,1,2]))
    rows=[{'token':t,'log':'same','local_fact_transitions':change} for t in ('a','b')]
    result=aggregate_local_changes(rows)['road_violation']['positive_to_negative']
    assert result['samples']==4 and result['scenes']==2 and result['logs']==1
    assert result['input_log_actor_or_map_pairs']==1


def test_trajectory_diagnostics_track_time_position_and_wrapped_yaw():
    before=np.zeros((8,3));before[:,2]=np.pi
    after=before.copy();after[:,2]=-np.pi
    assert plan_changes(before,after)['numerically_unchanged_pose']
    after[3,1]=.7
    changed=plan_changes(before,after)
    assert changed['first_changed_time_s']==changed['max_xy_change_time_s']==2.
    assert changed['xy_displacement_m'][3]==[0.,.7]
    assert not changed['numerically_unchanged_pose']


def test_nuscenes_horizon_is_parameterized_without_training_it():
    result=plan_changes(np.zeros((6,3)),np.zeros((6,3)))
    assert result['physical_times_s'][-1]==3. and result['first_changed_time_s'] is None
