"""Explicit retry does not convert failed native comparisons into a PASS."""
import copy

import pytest

from tools.qwen_world_compare.qualify import retry_failed_comparisons


def state():
    return dict(source='/immutable/science',arms={
        'A':dict(stage='FAILED',error=dict(job='old_A_compare',result=dict(returncode=1))),
        'B':dict(stage='FAILED',error=dict(job='old_B_compare',result=dict(returncode=1))),
        'C':dict(stage='RESUME_RUNNING'), 'D':dict(stage='BASELINE_PENDING')},
        jobs={'old_A_compare':dict(stage='COMPARE',architecture='A',handled=True),
              'old_B_compare':dict(stage='COMPARE',architecture='B',handled=True)})


def test_narrow_CPU_retry_preserves_failed_jobs_and_unrelated_training():
    value=state();before=copy.deepcopy(value)
    migration=dict(sha256='explicit_ops_receipt',changes={'tools/qwen_world_compare/qualify.py':dict(before='old',after='fixed')})
    retry_failed_comparisons(value,['A','B'],migration,'/immutable/controller')
    assert value['jobs']==before['jobs']
    assert value['arms']['C']==before['arms']['C']and value['arms']['D']==before['arms']['D']
    assert value['arms']['A']['stage']==value['arms']['B']['stage']=='COMPARE_PENDING'
    assert value['arms']['A']['comparison_retry_history'][0]['previous_error']==before['arms']['A']['error']
    assert all(v.get('stage')!='COMPLETE'for v in value['arms'].values())


@pytest.mark.parametrize('migration',[None,dict(changes={'tools/qwen_world_compare/profile.py':{}},sha256='other_repair')])
def test_retry_requires_the_explicit_qualifier_repair_receipt(migration):
    value=state();before=copy.deepcopy(value)
    with pytest.raises(ValueError,match='qualifier-repair'):retry_failed_comparisons(value,['A'],migration,'/ops')
    assert value==before


def test_CPU_compare_retry_cannot_retry_training_or_partially_mutate_a_batch():
    value=state();before=copy.deepcopy(value)
    migration=dict(sha256='ops',changes={'tools/qwen_world_compare/qualify.py':{}})
    with pytest.raises(ValueError,match='failed CPU comparison'):
        retry_failed_comparisons(value,['A','C'],migration,'/ops')
    assert value==before
