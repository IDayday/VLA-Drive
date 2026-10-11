from copy import deepcopy
import pytest
import torch
from iqe.io import ContractError
from scripts.iqe.fp32_cache.prepare_online_fallback import convert_metadata,state_fingerprint


def fixture():
    cache={'training_visual_source':'cached_fp32','cache_root':'pilot','cache_index_sha256':'x',
           'source_hashes':{'reader':'exact'},'visual_numeric_contract':'legacy_fp32'}
    identity={'production_cache_runtime':cache,'execution_environment':{'host':'same','torch':'same'},
              'optimizer_config':{'steps':100000},'fit_hash':'fit'}
    state={'performance_identity':identity,'training_visual_source':'cached_fp32','mode':'full',
           'model':{'p':torch.randn(4)},'optimizer':{'exp_avg':torch.randn(4),'step':123},
           'scheduler':{'last_epoch':123},'optimizer_step':123,'epoch':2,'offset':6,
           'dependencies':{'fixed':True},'rng_by_rank':[torch.get_rng_state()],
           'fit_scene_ids':['one','two'],'seed':42,'resume_boundary':'optimizer_step',
           'learned_driving_weights_loaded':False}
    online={k:v for k,v in cache.items() if k!='cache_index_sha256'}
    online.update(training_visual_source='online_fp32',cache_root=None)
    return state,{'runtime_identity':deepcopy(identity)},online


def test_fallback_only_changes_execution_metadata_and_retains_diagnostic_mode():
    state,cert,online=fixture();old=deepcopy(state);state['mode']='performance_v2'
    expected=state_fingerprint(state)
    new=convert_metadata(state,cert,online,'actual_parent','sha')
    assert state_fingerprint(new)==expected and new['mode']=='performance_v2'
    assert new['model'] is state['model'] and new['optimizer'] is state['optimizer']
    assert new['performance_identity']!=state['performance_identity']
    assert state['performance_identity']==old['performance_identity']
    assert new['execution_migration']['new_optimizer_steps']==0


@pytest.mark.parametrize('change',['precision','reader','host','terminal'])
def test_fallback_rejects_unqualified_changes(change):
    state,cert,online=fixture()
    if change=='precision':online['visual_numeric_contract']='bf16'
    if change=='reader':online['source_hashes']={'reader':'changed'}
    if change=='host':state['performance_identity']['execution_environment']['host']='foreign'
    if change=='terminal':state['terminal']=True
    with pytest.raises(ContractError):convert_metadata(state,cert,online,'actual_parent','sha')


def test_fallback_fingerprint_covers_adam_rng_cursor_and_small_update():
    state,cert,online=fixture();expected=state_fingerprint(state)
    for key in ['model','optimizer','rng_by_rank','offset']:
        changed=deepcopy(state)
        if key=='model':changed[key]['p'][0]+=1e-5
        elif key=='optimizer':changed[key]['exp_avg'][0]+=1e-5
        elif key=='rng_by_rank':changed[key][0][0]^=1
        else:changed[key]+=1
        assert state_fingerprint(changed)!=expected
