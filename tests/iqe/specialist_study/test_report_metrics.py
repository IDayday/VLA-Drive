import numpy as np
from iqe.scoring.reference import COMPONENTS
from iqe.specialist_study.metrics import metrics, report_metrics


def test_complete_component_report_preserves_original_selection_metrics():
    scores=np.asarray([[.7,.9],[.8,.6]])
    components=np.ones((2,2,len(COMPONENTS)))
    j=COMPONENTS.index('ego_progress')
    components[:,0,j]=[.2,.4];components[:,1,j]=[.7,.5]
    chosen=np.asarray([1,0]);logs=['a','b']
    original=metrics(scores,components,chosen,logs,repetitions=20)
    result=report_metrics(scores,components,chosen,logs,repetitions=20)
    assert all(result[key]==value for key,value in original.items())
    assert result['components']['ego_progress']['valid_denominator']==2
    assert abs(result['components']['ego_progress']['mean_change']-.25)<1e-12
    assert result['direction_compliance_has_zero_PDMS_weight']


def test_missing_component_is_reported_with_zero_denominator_not_perfect_score():
    scores=np.asarray([[.7,.9]])
    components=np.ones((1,2,len(COMPONENTS)))
    components[:,:,COMPONENTS.index('comfort')]=np.nan
    result=report_metrics(scores,components,[1],['a'],repetitions=10)
    value=result['components']['comfort']
    assert value['valid_denominator']==0 and value['base_mean'] is None and value['selected_mean'] is None
