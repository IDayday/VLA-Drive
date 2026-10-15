import numpy as np
import pytest

from tools.decision_local.summarize_natural_local_population import (
    execution_eligibility, plan_facts, verify_scored_q0)
from tools.decision_local.common import digest


def example():
    shape=(1,1,3,7,8,3)
    arrays={name:np.zeros(shape,dtype=dtype) for name,dtype in (
        ('relations',np.float32),('valid',bool),('positive_known',bool),('unknown_reason',np.uint16))}
    arrays.update(common_valid=np.zeros((1,1,7,8,3),bool),
        poses=np.zeros((1,1,7,8,3),np.float32),plan_in_physical=np.zeros((1,1,8,3),np.float32))
    case=dict(anchor_index=0,plan_index=0,target_valid=False,status_name='UNCERTAIN',
        input_check=dict(known_violation=False,road_violation=False,kinematic_ok=True,complete_annotations=False))
    return arrays,case,np.arange(1,9)*.5


def test_neighbor_query_violation_does_not_become_a_native_error():
    arrays,case,times=example()
    arrays['positive_known'][0,0,2,1,0,0]=True
    result=plan_facts(arrays,case,times)
    assert not result['sampled_known_road_violation']
    arrays['positive_known'][0,0,2,0,0,0]=True
    result=plan_facts(arrays,case,times)
    assert result['sampled_local_fact_without_strict_target']
    assert not result['strict_target_eligible']


def test_oracle_dynamic_support_matches_all_three_source_masks():
    arrays,case,times=example()
    arrays['valid'][0,0,1:,0,0,1]=True
    arrays['relations'][0,0,2,0,0,1]=2
    result=plan_facts(arrays,case,times)
    assert result['cv_fut_distance_change_local_support']
    assert not result['dynamic_information_common_oracle_support']
    arrays['common_valid'][0,0,0,0,1]=True
    assert plan_facts(arrays,case,times)['dynamic_information_common_oracle_support']


def test_execution_review_cannot_promote_unknown_and_rejects_bad_target():
    _,case,_=example();record={'token':'s','sha256':'payload'}
    assert not execution_eligibility(record,case,{'cases':{}})
    case.update(status_name='REPAIR',target_valid=True)
    row={'label_sha256':'payload','admitted':False,'checks':{'road':False}}
    assert not execution_eligibility(record,case,{'cases':{'s:0:0':row}})
    with pytest.raises(KeyError):execution_eligibility(record,case,{'cases':{}})


def test_score_join_checks_actual_q0_and_not_just_token(tmp_path):
    bank=tmp_path/'q0';(bank/'proposals').mkdir(parents=True)
    path=bank/'proposals/s.npz';plan=np.zeros((8,3),np.float32)
    np.savez_compressed(path,trajectory=plan)
    row=dict(token='s',variant='q0',status='ok',proposal_sha256=digest(path))
    verify_scored_q0(bank/'scenes.csv',row,plan)
    with pytest.raises(ValueError,match='another actual input'):
        verify_scored_q0(bank/'scenes.csv',row,plan+1)
    with pytest.raises(ValueError,match='original-input'):
        verify_scored_q0(bank/'scenes.csv',dict(row,variant='q_final'),plan)
