import pytest
from tools.local_interaction_mask_v2.compare_pdms import compare,METRICS
from tools.local_interaction_mask_v2.merge_scores import merge_population


def test_log_cluster_pairing_keeps_scene_weights_and_failed_denominator():
    base={str(i):dict(token=str(i),log='long' if i<9 else 'short',status='ok',**{k:.5 for k in METRICS}) for i in range(10)}
    model={k:dict(v,**{m:.6 for m in METRICS}) for k,v in base.items()}
    result,rows=compare(model,base,bootstrap=500)
    assert len(rows)==10 and result['valid']
    assert result['metrics']['score']['log_cluster_95_interval']==pytest.approx([.1,.1])
    model['9']['status']='failed'
    result,rows=compare(model,base,bootstrap=500)
    assert result['failed_first']==1 and not result['valid']
    assert result['PDMS_first_percent']==pytest.approx(54.)
    assert result['delta_percentage_points']==pytest.approx(4.)
    assert rows[-1]['first_score']==0
    with pytest.raises(ValueError,match='populations differ'):compare({k:v for k,v in model.items() if k!='9'},base)


def test_partition_merge_requires_full_population_and_retains_failure():
    index=[dict(token='a',log='one'),dict(token='b',log='two')]
    first={'a':dict(token='a',log='one',status='ok',**{k:.8 for k in METRICS})}
    second={'b':dict(token='b',log='two',status='failed',score=1.)}
    rows=merge_population(index,[second,first])
    assert [row['token'] for row in rows]==['a','b']
    assert rows[1]['score']==0. and rows[1]['status']=='failed'
    with pytest.raises(ValueError,match='Overlapping'):merge_population(index,[first,first,second])
    with pytest.raises(ValueError,match='Missing'):merge_population(index,[first])
    with pytest.raises(ValueError,match='token/log'):
        merge_population(index,[first,{'b':dict(second['b'],log='wrong')}])
