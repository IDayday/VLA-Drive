import pytest
from scripts.analysis.fm_transfer.package_final_navtest import require_complete, match_means


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
