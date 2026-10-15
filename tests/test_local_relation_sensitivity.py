import numpy as np
import pytest
import torch

from tools.decision_local.evaluate_relations import scene_metrics, sensitivity_population, LOCAL_SENSITIVITY
from tools.decision_local.compare_relations import matched_sensitivity_protocol


def example():
    truth=torch.zeros(1,7,8,5)
    truth[...,0]=2.;truth[...,1]=3.;truth[...,3]=3.
    pred=truth.clone();valid=torch.ones_like(truth,dtype=torch.bool)
    poses=torch.zeros(1,7,8,3);poses[:,1:,:,1]=.75
    return pred,truth,valid,poses


def test_partial_local_facts_are_not_insensitivity():
    pred,truth,valid,poses=example()
    valid[:,:,7,4]=False;truth[:,:,7,4]=float('nan')
    old=scene_metrics(pred,truth,valid,poses)
    new=scene_metrics(pred,truth,valid,poses,local_evidence=True)
    assert old['near_insensitive_pairs'][0]==6
    assert new['near_insensitive_pairs'][0]==0
    assert new['near_unknown_pairs'][0]==6
    assert new['road_margin_m_pair_valid_times'][0]==48
    assert new['near_unknown_road_margin_m_valid_times'][0]==48
    assert all(np.isfinite(v).all() for v in new.values())


def test_known_crossing_survives_unrelated_unknown_future():
    pred,truth,valid,poses=example()
    valid[...,4]=False;truth[...,4]=float('nan')
    truth[:,1,0,0]=-1.
    new=scene_metrics(pred,truth,valid,poses,local_evidence=True)
    assert new['near_sensitive_pairs'][0]==1
    assert new['near_unknown_pairs'][0]==5
    assert new['near_insensitive_pairs'][0]==0


def test_complete_noncrossing_is_insensitive_and_legacy_metrics_unchanged():
    args=example();old=scene_metrics(*args);new=scene_metrics(*args,local_evidence=True)
    assert new['near_insensitive_pairs'][0]==6
    assert new['near_unknown_pairs'][0]==0
    for key,value in old.items():np.testing.assert_array_equal(value,new[key])


def test_all_unknown_and_spatial_ambiguity_remain_separate():
    pred,truth,valid,poses=example();valid[:]=False;truth[:]=float('nan')
    poses[:,1:3,:,1]=.1;poses[:,3:5,:,1]=2.
    new=scene_metrics(pred,truth,valid,poses,local_evidence=True)
    assert new['ambiguous_spatial_pairs'][0]==2
    assert new['far_unknown_pairs'][0]==2
    assert new['near_unknown_pairs'][0]==2
    assert new['near_insensitive_pairs'][0]==new['far_insensitive_pairs'][0]==0
    assert new['near_unknown_road_margin_m_valid_times'][0]==0


def test_future_overlap_change_is_known_without_other_criteria():
    pred,truth,valid,poses=example();valid[:]=False;truth[:]=float('nan')
    valid[:,0,2,4]=valid[:,1,2,4]=True
    truth[:,0,2,4]=0.;truth[:,1,2,4]=1.
    new=scene_metrics(pred,truth,valid,poses,local_evidence=True)
    assert new['near_sensitive_pairs'][0]==1
    assert new['near_unknown_pairs'][0]==5


def test_pairs_are_not_independent_scene_or_log_replicates():
    rows=[{'token':'a','log':'x','near_unknown_pairs':4},
          {'token':'b','log':'x','near_unknown_pairs':2},
          {'token':'c','log':'y','near_unknown_pairs':0}]
    assert sensitivity_population(rows)['near_unknown']=={'pairs':6,'scenes':2,'logs':1}


def test_comparison_rejects_changed_group_definition():
    first={'a':{'sensitivity_protocol':LOCAL_SENSITIVITY}}
    assert matched_sensitivity_protocol(first,first)==LOCAL_SENSITIVITY
    assert matched_sensitivity_protocol({'a':{}},{'a':{}})=='any_supported_sample_v1'
    with pytest.raises(ValueError,match='protocols differ'):
        matched_sensitivity_protocol(first,{'a':{}})
