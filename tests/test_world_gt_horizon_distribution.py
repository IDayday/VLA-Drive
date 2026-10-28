import numpy as np
from tools.qwen_world_postcot.analyze_gt_horizon_distribution import summarize


def test_large_marginal_dispersion_can_have_zero_current_velocity_residual():
    t=np.arange(1,9)*.5;velocity=np.array([1.,2.,3.,4.])
    gt=np.zeros((4,8,3));gt[:,:,0]=velocity[:,None]*t
    valid=np.ones((4,8),dtype=bool)
    population=summarize(gt,valid);residual=summarize(gt-gt,valid)
    assert np.isclose(population[-1]['std_population'][0]/population[0]['std_population'][0],8)
    assert residual[-1]['std_population']==[0.,0.,0.]


def test_invalid_timestamp_values_do_not_become_zero_labels():
    gt=np.zeros((4,8,3));gt[:,:,0]=np.arange(4)[:,None]
    gt[0,0]=np.nan;valid=np.ones((4,8),dtype=bool);valid[0,0]=False
    result=summarize(gt,valid)
    assert result[0]['valid_count']==3
    assert result[0]['mean'][0]==2
    assert np.isclose(result[0]['std_population'][0],np.std([1.,2.,3.]))
