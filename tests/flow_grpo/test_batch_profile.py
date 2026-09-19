"""CPU schema fixtures only; these never publish a production release."""
from copy import deepcopy
import json
from pathlib import Path
import pytest
from starVLA.rl.flow_grpo.batch_profile import validate_batch_profile
from starVLA.rl.flow_grpo.loading import file_sha


def bundle(tmp_path, monkeypatch):
    kernel={'kernel': 'fixture'}
    monkeypatch.setattr('starVLA.rl.flow_grpo.batch_profile.kernel_identity', lambda:kernel)
    names=[f'action_model.fixture{i}' for i in range(359)]
    context={'world_size':8,'config_sha256':'fixture'}
    cfg={'runtime':{'transition_evaluation':'flat_saved_chain','activation_checkpointing':False,'batch_profile_evidence':str(tmp_path/'profile.json')}, 'checkpoint_contract':{'sha256':'F_fixture'}}
    files={}
    def put(name,data):
        path=tmp_path/(name+'.json');path.write_text(json.dumps(data));files[name]=data
        return {'path':str(path),'sha256':file_sha(path)}
    proof={'schema_version':1,'status':'QUALIFIED_FIXED_LAYOUT','context':context,'kernels':kernel,'trainable_names':names}
    for key in ('fp32_serial_run','fp32_batch_run','fp32_rounding_oracle','bf16_batch_run'):
        d='bf16' if key=='bf16_batch_run' else 'fp32'
        proof[key]=put(key,{'status':'MEASURED_NOT_QUALIFIED','device':'fixture GPU','device_type':'cuda','checkpoint_sha256':'F_fixture','kernels':kernel,'head_storage':d,
         'mode':'serial_on' if key=='fp32_serial_run' else 'flat160_off','observed_head_dtypes':['torch.bfloat16' if d=='bf16' else 'torch.float32'],
         'trainable_tensors':359,'trainable_numel':819503620,'preserve_bf16_time_input':key=='fp32_rounding_oracle',
         'scenes':[{'tokens':[str(i)],'official_nonzero_advantage':True,'repeats':[{'gradient_tensors':359,'gradient_finite':True}]*2} for i in (1,2)]})
    proof['fp32_gradient_comparisons']=[put(f'math{i}',{'status':'PASS','scene_position':i,'input_runs':{'left':proof['fp32_serial_run']['sha256'],'right':proof['fp32_batch_run']['sha256']},'parameters':{n:{'allclose':True,'finite':True,'atol':2e-6,'rtol':2e-3} for n in names}}) for i in (1,2)]
    proof['bf16_rounding_comparison']=put('rounding',{'status':'PASS','input_runs':{'oracle':proof['fp32_rounding_oracle']['sha256'],'actual':proof['bf16_batch_run']['sha256']},'scenes':{str(i):{'status':'PASS','forward_exact':True,'parameters':{n:{'equal':True,'nonidentical':0} for n in names}} for i in (1,2)}})
    proof['native_adam']=put('adam',{'status':'PASS','world_size':8,'execution_context':context,'config_hash':'fixture','run':str(tmp_path/'pilot'),'parameters':{n:{'forward_equals_actual_master_cast':True,**{k:{'pass':True} for k in ('master','exp_avg','exp_avg_sq')}} for n in names}})
    proof['export']=put('export',{'status':'TESTED','tensors_identical':989,'output_max_abs':0,'checkpoint':str(tmp_path/'pilot/checkpoints/update_000002')})
    proof['historical_serial_comparison']=put('history',{'status':'FAIL'})
    return cfg,context,proof,files,put


def test_fixed_layout_schema_accepts_consistent_indexed_evidence(tmp_path,monkeypatch):
    cfg,ctx,p,files,put=bundle(tmp_path,monkeypatch)
    validate_batch_profile(put('profile',p),cfg,ctx)


@pytest.mark.parametrize('case', ['cpu','wrong_weight','wrong_dtype','missing_gradient','inner_fail','rounding_fail','detached_rounding','different_scenes','wrong_world','bad_adam','changed_kernel','wrong_export','rewritten_history'])
def test_rejects_semantically_invalid_evidence_even_with_correct_hash(tmp_path,monkeypatch,case):
    cfg,ctx,p,files,put=bundle(tmp_path,monkeypatch)
    if case in ('cpu','wrong_weight','wrong_dtype','different_scenes'):
        key='fp32_batch_run';r=files[key]
        if case=='cpu':r['device_type']='cpu'
        if case=='wrong_weight':r['checkpoint_sha256']='U_fixture'
        if case=='wrong_dtype':r['observed_head_dtypes']=['torch.bfloat16']
        if case=='different_scenes':r['scenes'][0]['tokens']=['different']
        p[key]=put(key,r)
        # Rebind the hashes so this tests CONTENT semantics, not just old hashes.
        for i in (1,2):
            files[f'math{i}']['input_runs']['right']=p[key]['sha256'];p['fp32_gradient_comparisons'][i-1]=put(f'math{i}',files[f'math{i}'])
    elif case in ('missing_gradient','inner_fail'):
        r=files['math1']
        if case=='inner_fail':r['status']='FAIL'
        else:r['parameters'].pop(next(iter(r['parameters'])))
        p['fp32_gradient_comparisons'][0]=put('math1',r)
    elif case in ('rounding_fail','detached_rounding'):
        r=files['rounding']
        if case=='rounding_fail':r['scenes']['1']['parameters'][p['trainable_names'][0]]['equal']=False
        else:r['input_runs']['oracle']='wrong'
        p['bf16_rounding_comparison']=put('rounding',r)
    elif case in ('wrong_world','bad_adam'):
        r=files['adam']
        if case=='wrong_world':r['world_size']=1
        else:r['parameters'][p['trainable_names'][0]]['exp_avg']['pass']=False
        p['native_adam']=put('adam',r)
    elif case=='changed_kernel':p['kernels']={'kernel':'changed'}
    elif case=='wrong_export':
        r=files['export'];r['checkpoint']=str(tmp_path/'another/checkpoints/update_000002');p['export']=put('export',r)
    else:p['historical_serial_comparison']=put('history',{'status':'PASS'})
    with pytest.raises(ValueError):validate_batch_profile(put('profile',p),cfg,ctx)


def test_launch_topology_counts_actual_nodes_instead_of_world_divided_by_eight():
    from starVLA.rl.flow_grpo.full_epoch import launch_peer_count
    spec={'nodes':[{'host':h,'devices':list(range(4))} for h in ('a','b')],
          'entry':['-m','starVLA.rl.flow_grpo.cli','train']}
    assert launch_peer_count(spec,8)==2
    with pytest.raises(ValueError):launch_peer_count(spec,16)
    spec['nodes'][1]['host']='a'
    with pytest.raises(ValueError):launch_peer_count(spec,8)
