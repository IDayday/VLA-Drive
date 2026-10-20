import pytest
from tools.action_effect.compare_methods import paired_summary

def rows(scores):
    return {str(i):{'scene_id':str(i),'source_log':'log'+str(i//2), 'status':'VALID_SCORED',
        'metrics':{'score':v,'no_at_fault_collisions':float(v>0),'drivable_area_compliance':1.,
            'time_to_collision_within_bound':1.,'ego_progress':v,'comfort':1.,'driving_direction_compliance':1.},
        'distance_m':.1} for i,v in enumerate(scores)}

def test_signed_gains_and_severe_regressions_are_both_retained():
    report=paired_summary(rows([0.,1.,.9,.5]),rows([.8,0.,.8,.9]),repetitions=100,seed=42)
    assert report['mean_delta']==pytest.approx(.025)
    assert report['positive_gain']-report['negative_cost']==pytest.approx(report['mean_delta'])
    assert report['high_to_zero']==1 and report['zero_repaired']==1
    assert report['new_NC_failures']==1 and report['repaired_NC_failures']==1
    assert report['source_logs']==2 and report['full_population_score_claim']

def test_invalid_prediction_does_not_disappear_from_requested_denominator():
    a=rows([.8,.9]);b=rows([.9,.9]);b['1'].update(status='INVALID_PREDICTION',metrics=None)
    r=paired_summary(a,b,repetitions=10)
    assert r['requested_scenes']==2 and r['common_valid_scenes']==1
    assert r['invalid_or_unresolved']==1 and not r['full_population_score_claim']

def test_population_or_group_mismatch_refused():
    with pytest.raises(ValueError,match='scenes'):paired_summary(rows([1]),rows([1,1]))
    b=rows([1]);b['0']['source_log']='wrong'
    with pytest.raises(ValueError,match='logs'):paired_summary(rows([1]),b)

def test_group_bootstrap_fixed_seed_and_negative_results_not_clipped():
    a=rows([1.,1.,1.,1.]);b=rows([.1,.5,.8,.9])
    x=paired_summary(a,b,repetitions=50,seed=8);y=paired_summary(a,b,repetitions=50,seed=8)
    assert x==y and x['mean_delta']<0 and x['delta_ci95'][1]<0
