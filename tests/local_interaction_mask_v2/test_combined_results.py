import pytest
from tools.local_interaction_mask_v2.combine_results import VARIANTS, METRICS, combine


def fixture():
    return {name:{'a':dict(token='a',log='log1',status='ok',**{key:value for key in METRICS}),
                  'b':dict(token='b',log='log2',status='ok',**{key:value for key in METRICS})}
            for name,value in zip(VARIANTS,[.5,.6,.65,.7])}


def test_full_rows_failures_and_percentage_point_units():
    groups=fixture();groups['P_LOCAL_MASK']['b']['status']='failed'
    rows=combine(groups)
    assert len(rows)==2
    assert rows[0]['P_LOCAL_MASK_minus_A0_percentage_points']==pytest.approx(20)
    assert rows[0]['P_LOCAL_MASK_no_at_fault_collisions']==.7
    assert rows[1]['P_LOCAL_MASK_PDMS']==0
    assert rows[1]['P_LOCAL_MASK_status']=='failed'
    assert rows[1]['P_LOCAL_MASK_minus_A0_percentage_points']==-50


def test_different_scene_or_log_population_is_rejected():
    groups=fixture();groups['CURRENT_MEMORY'].pop('b')
    with pytest.raises(ValueError,match='populations'):combine(groups)
    groups=fixture();groups['P_LOCAL_ALL']['a']['log']='wrong'
    with pytest.raises(ValueError,match='Log identity'):combine(groups)
