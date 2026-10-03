import json
import pytest
from tools.action_video_foresight.summarize_live_campaign import union_seconds,resource_cost,read_steps,development_results


def test_physical_gpu_union_does_not_double_count_shared_jobs(tmp_path):
    for name,start,end in [('trainer',0,3600),('probe',1800,5400)]:
        root=tmp_path/'runs'/name;root.mkdir(parents=True)
        (root/'status.json').write_text(json.dumps({'gpu_count':1,'host':'same_host',
            'cuda_visible_devices':'0','start_unix':start,'end_unix':end,'kind':name}))
    cost=resource_cost(tmp_path,5400)
    assert cost['job_allocated_GPUh']==2.
    assert cost['physical_device_interval_union_GPUh']==1.5
    assert cost['unassigned_allocation_GPUh']==0.
    with pytest.raises(ValueError,match='Negative'):union_seconds([(2,1)])


def test_inflight_jsonl_tail_is_not_a_failed_completed_record(tmp_path):
    path=tmp_path/'steps.jsonl';path.write_text('{"update":1}\n{"update":')
    assert read_steps(path)==[{'update':1}]
    path.write_text('{"update":1}\nBAD\n')
    with pytest.raises(json.JSONDecodeError):read_steps(path)


def test_unmetered_standalone_check_is_explicit_not_invented(tmp_path):
    (tmp_path/'clip_nccl_equivalence_v1.log').write_text('max_gradient_error=0\n')
    result=resource_cost(tmp_path,3600)
    assert result['job_allocated_GPUh']==0
    assert result['physical_device_interval_union_GPUh']==0
    assert result['known_unmetered_checks'][0]['GPUh']=='NOT_MEASURED'


def test_development_snapshot_uses_complete_bound_scores_and_preserves_failures(tmp_path):
    plan={'campaign_root':str(tmp_path),'runs':{'S4':{'run_id':'run'}},
        'evaluation_updates':[5000],'identity':'plan','training_source_sha':'source','dev_scenes':2}
    assert development_results(plan)=={}
    root=tmp_path/'evaluations';root.mkdir()
    state=root/'run_dev5000_seed42_state.json'
    identity={'plan':'plan','arm':'S4','update':5000,'evaluation_source':'source',
        'purpose':'registered complete development; not Navtest'}
    state.write_text(json.dumps({'identity':identity,'status':'FAILED','error':'bad export'}))
    assert development_results(plan)['S4']['5000']['error']=='bad export'
    score=tmp_path/'score.json';ego=tmp_path/'ego.json'
    score.write_text(json.dumps({'valid':True,'failed':0,'scenes':2,'PDMS':.8,'metrics':{'NC':1.}}))
    ego.write_text(json.dumps({'valid':True,'failed':0,'scenes':2,
        'groups':{'all':{'ADE':.5,'FDE':1.,'yaw_MAE_rad':.1}}}))
    state.write_text(json.dumps({'identity':identity,'status':'COMPLETE','scores':str(score),'ego':str(ego),'PDMS_points':80.}))
    assert development_results(plan)['S4']['5000']['PDMS_points']==80.
    value=json.loads(score.read_text());value['failed']=1;score.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='Incomplete development'):development_results(plan)


def test_reconciled_result_is_bound_to_preserved_original_failure(tmp_path):
    from starVLA.model.modules.vehicle_joint.initialization import file_sha256
    plan={'campaign_root':str(tmp_path),'runs':{'S4':{'run_id':'run'}},
        'evaluation_updates':[5000],'identity':'plan','training_source_sha':'source','dev_scenes':2}
    original=tmp_path/'evaluations'/'run_dev5000_seed42_state.json';original.parent.mkdir()
    identity={'plan':'plan','arm':'S4','update':5000,'evaluation_source':'source',
        'purpose':'registered complete development; not Navtest'}
    original.write_text(json.dumps({'identity':identity,'status':'FAILED','error':'SSH'}))
    score=tmp_path/'score.json';ego=tmp_path/'ego.json'
    score.write_text(json.dumps({'valid':True,'failed':0,'scenes':2,'PDMS':.8,'metrics':{'NC':1.}}))
    ego.write_text(json.dumps({'valid':True,'failed':0,'scenes':2,'groups':{'all':{'ADE':.5,'FDE':1.,'yaw_MAE_rad':.1}}}))
    sidecar=tmp_path/'reconciled_development'/'run_dev5000_seed42_state.json';sidecar.parent.mkdir()
    sidecar.write_text(json.dumps({'identity':identity,'status':'COMPLETE','original_failed_state_SHA256':file_sha256(original),
        'scores':str(score),'ego':str(ego),'PDMS_points':80.,'transport_source':'recovery-source'}))
    result=development_results(plan)['S4']['5000']
    assert result['PDMS_points']==80. and result['transport_source']=='recovery-source'
    assert json.loads(original.read_text())['status']=='FAILED'
    original.write_text(original.read_text()+'\n')
    with pytest.raises(ValueError,match='original state changed'):development_results(plan)
