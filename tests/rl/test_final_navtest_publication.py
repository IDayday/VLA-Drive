import pytest
from scripts.analysis.fm_transfer.package_final_navtest import require_complete, match_means, match_safety


@pytest.mark.parametrize('patch', [{'status':'RUNNING'}, {'scenes':12145}, {'logs':135}, {'seed':42}])
def test_partial_or_different_population_rejected(patch):
    x=dict(status='COMPLETE_POPULATION',scenes=12146,logs=136,seed=2101)
    require_complete(x)
    with pytest.raises(ValueError):require_complete(dict(x,**patch))


def test_independent_canonical_results_required():
    d={'versions':{'v1.1':{m:dict(mean={'score':.9}) for m in ['E_NATIVE','E_SHIFT5']}}}
    c={'modes':{m:dict(mean={'score':.9}) for m in ['E_NATIVE','E_SHIFT5']}}
    match_means(d,c)
    c['modes']['E_NATIVE']['mean']['score']=.91
    with pytest.raises(ValueError):match_means(d,c)


def test_canonical_extra_mean_delta_does_not_change_event_count_contract():
    d=dict(safety_transitions={'NC':dict(new_violation=3,recovery=2)},
           reference_safe_new_joint_violation=3,nonzero_to_zero=2,zero_to_nonzero=1)
    c=dict(d,safety_transitions={'NC':dict(new_violation=3,recovery=2,mean_delta=-.001)})
    match_safety(d,c)
    c['safety_transitions']['NC']['recovery']=1
    with pytest.raises(ValueError,match='transition counts'):match_safety(d,c)
