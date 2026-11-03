"""Metadata-only completion callbacks; no model, scorer or active-root reads."""
import copy
import hashlib
import json

import pytest

from tools.qwen_world_performance import complete_events as e


def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))


def seal(value):
    return dict(value,sha256=e.fingerprint(value))


def fixture(root):
    exp='AUG1_FULL';key=exp+'_u000100';cp=root/'runs'/(exp+'_v1')/'checkpoints/update000100_inner0'
    native=seal(dict(algorithm_schema=e.ALGORITHM,experiment_id=exp,parent_model_sha256=e.PARENT,
        qualification=False,scope={'names':['actual-trainable']}))
    put(cp/'identity.json',native)
    files={'identity.json':hashlib.sha256((cp/'identity.json').read_bytes()).hexdigest(),'optimizer.pt':'frozen-payload-sha'}
    put(cp/'files.json',files);inventory=hashlib.sha256((cp/'files.json').read_bytes()).hexdigest()
    put(cp/'COMPLETE',dict(files_sha256=inventory,updates=100,phase='READY',next_inner=0))
    model=e.fingerprint(dict(parent=e.PARENT,trainable_payload=files['optimizer.pt'],scope=native['scope']))
    output=root/'evaluations'/key/'evaluation'
    identity=seal(dict(checkpoint=str(cp),D44_training_identity=native['sha256'],model_sha256=model,
        updates=100,cohort='NAVTEST',qualification=False,seed=2101,count=12146,
        inference='single_candidate_cleanx_K10_shift5',protocol={'sha256':'fixed-official'},token_sha256='full-tokens'))
    put(output/'identity.json',identity)
    summary=seal(dict(status='COMPLETE',scenes=12146,scene_denominator=12146,physical_logs=44,invalid_scenes=0,
        qualification=False,seed=2101,cohort='NAVTEST',architecture_id='D',evaluation=str(output),
        evaluation_identity_sha256=identity['sha256'],model_sha256=model,protocol_sha256='fixed-official',
        token_sha256='full-tokens',PDMS=91.2,means={'score':.912}))
    put(output.parent/'SUMMARY.json',summary)
    case=dict(id=key,status='COMPLETE',world=32,output=str(output),summary=str(output.parent/'SUMMARY.json'),
        binding=dict(experiment=exp,checkpoint=str(cp),updates=100,parent_model_sha256=e.PARENT,
            algorithm_schema=e.ALGORITHM,identity_sha256=native['sha256'],inventory_sha256=inventory,
            payload_sha256=files['optimizer.pt'],policy_signature=model))
    state=root/'operations/AUGUSTUS_CAMPAIGN_20261103/EVAL_STATE.json'
    put(state,dict(schema='d46_augustus_official_evaluation_queue_v1',cases={key:case}))
    return state,case,summary


def test_full_callback_is_idempotent_and_never_writes_evaluation_state(tmp_path):
    state,case,summary=fixture(tmp_path);before=state.read_bytes();request=tmp_path/'original_watcher/refresh_request.json'
    first=e.notify(state,case['id'],summary['sha256'],root=tmp_path,request_path=request)
    assert first['status']=='NOTIFIED'and first['notified_sha']==summary['sha256']
    payload=json.loads(request.read_text());assert payload['scientific_result']is False
    assert payload['validated_point']['checkpoint']==case['binding']['checkpoint']
    written=request.read_bytes();again=e.notify(state,case['id'],summary['sha256'],root=tmp_path,request_path=request)
    assert again['status']=='ALREADY_NOTIFIED'and request.read_bytes()==written and state.read_bytes()==before
    assert len(e.completed_points(tmp_path))==1


def test_partial_qualification_invalid_count_or_parent_never_publishes(tmp_path):
    state,case,summary=fixture(tmp_path);request=tmp_path/'refresh_request.json'
    for key,value in [('scenes',12145),('invalid_scenes',1),('qualification',True),('physical_logs',43)]:
        changed=dict(summary);changed.pop('sha256');changed[key]=value
        put(type(state)(case['summary']),seal(changed))
        with pytest.raises(ValueError):e.notify(state,case['id'],summary['sha256'],root=tmp_path,request_path=request)
        assert not request.exists()and not e.completed_points(tmp_path)
    put(type(state)(case['summary']),summary)
    wrong=copy.deepcopy(case);wrong['binding']['parent_model_sha256']='foreign'
    with pytest.raises(ValueError,match='parent'):e.validated_point(tmp_path,wrong)


def test_pending_case_wrong_event_sha_and_mutated_native_inventory_are_rejected(tmp_path):
    state,case,summary=fixture(tmp_path);request=tmp_path/'refresh_request.json'
    wrong=copy.deepcopy(case);wrong['status']='RUNNING'
    with pytest.raises(ValueError,match='COMPLETE'):e.validated_point(tmp_path,wrong)
    with pytest.raises(ValueError,match='summary SHA'):e.notify(state,case['id'],'stale',root=tmp_path,request_path=request)
    cp=type(state)(case['binding']['checkpoint']);files=json.loads((cp/'files.json').read_text());files['optimizer.pt']='replaced'
    put(cp/'files.json',files)
    with pytest.raises(ValueError,match='seal'):e.notify(state,case['id'],summary['sha256'],root=tmp_path,request_path=request)
    assert not request.exists()
