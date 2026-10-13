import pytest
import torch
from iqe.io import atomic_json,atomic_torch,file_hash
from iqe.internal_baseline.incremental import comparison_population,warm_start_own_parent


def test_round2_distribution_refresh_replays_old_pairs():
    old=comparison_population(['s'],['expert_0','expert_1'],[[.2,.8]],[[True,True]])
    new=comparison_population(['s'],['expert_0','expert_1','expert_2'],[[.2,.8,.9]],[[True,True,True]])
    assert len(old['entries'])==1 and len(new['entries'])==3
    assert {(x['left_id'],x['right_id']) for x in new['entries']}=={('expert_0','expert_1'),('expert_0','expert_2'),('expert_1','expert_2')}
    assert new['population_hash']!=old['population_hash']
    assert sum(x['probability'] for x in new['entries'])==pytest.approx(1.)
    masked=comparison_population(['s'],['expert_0','expert_1','expert_2'],[[.2,.8,.9]],[[True,True,False]])
    assert len(masked['entries'])==1


def test_future_warm_start_accepts_only_declared_own_compatible_parent(tmp_path):
    source=torch.nn.Linear(3,2); target=torch.nn.Linear(3,2)
    atomic_torch(tmp_path/'scorer/best.pt',source.state_dict())
    atomic_json(tmp_path/'BASE_BINDING.json',{'experiment_id':'ours','checkpoint_hash':'base','contract_hash':'features'})
    init={'experiment_id':'ours','external_scorer_weights_loaded':False,'loaded_scorer_weights':[]}
    atomic_json(tmp_path/'SCORER_INITIALIZATION.json',init)
    atomic_json(tmp_path/'scorer/result.json',{'status':'COMPLETE','best_checkpoint_hash':file_hash(tmp_path/'scorer/best.pt')})
    kwargs={'declared_parent_experiment_id':'ours','base_checkpoint_hash':'base','feature_contract_hash':'features'}
    provenance=warm_start_own_parent(target,tmp_path,**kwargs)
    assert provenance['optimizer_reused'] is False
    assert all(torch.equal(v,target.state_dict()[k]) for k,v in source.state_dict().items())
    with pytest.raises(ValueError,match='shared Base'):
        warm_start_own_parent(target,tmp_path,**(kwargs|{'base_checkpoint_hash':'other'}))
    atomic_json(tmp_path/'SCORER_INITIALIZATION.json',init|{'external_scorer_weights_loaded':True})
    with pytest.raises(ValueError,match='external scorer'):
        warm_start_own_parent(target,tmp_path,**kwargs)
