"""Tiny control fixtures only. No real admission, plan, weights or GPU."""
import copy
import hashlib
import json
from pathlib import Path
import sys

import pytest

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'spatial_probe_actual_preparation_v1'))
import supervise_once as s
import prepare_bindings as b


def counters(hashed,loaded,rchar):
    return dict(hash_logical_bytes_admitted=hashed,torch_load_nominal_bytes=loaded,
                proc_self_io_delta={'rchar':rchar})


@pytest.fixture
def controls(tmp_path,monkeypatch):
    job=tmp_path/'job'
    monkeypatch.setattr(s,'JOB',job)
    monkeypatch.setattr(s,'FAILED_DEV_RESULT',job/'stages/dev_extract/RESULT.json')
    monkeypatch.setattr(s,'DEV_RETRY_DIRECTORY',job/'stages/dev_extract_manual_retry_v1')
    prep={'path':'/fixture/PREPARATION.json','sha256':'a'*64}
    plan=dict(schema='st_SP0_probe_stage_result_v1',status='COMPLETE',stage='dev_plan',
              preparation=prep,resources={'counters':counters(10,2,15)})
    failed=dict(schema='st_SP0_probe_stage_result_v1',status='STOPPED_NO_AUTO_RETRY',
                stage='dev_extract',preparation=prep,resources={'counters':counters(20,3,10),
                'limit_reason':'MONITOR_UNREADABLE','process_cleanup':{'live_owned_remaining':0}})
    plan_pin=b.write_once(job/'stages/dev_plan/RESULT.json',plan)
    failed_pin=b.write_once(s.FAILED_DEV_RESULT,failed)
    monkeypatch.setattr(s,'FAILED_DEV_SHA256',failed_pin['sha256'])
    admission={'preparation':prep,'manual_retry':{'failed_result':failed_pin,'plan_result':plan_pin,
                                              'attempt_directory':str(s.DEV_RETRY_DIRECTORY)}}
    return admission,plan,failed


def test_read_budget_adds_plan_and_failed_attempt_without_double_counting(controls):
    admission,_,_=controls
    before=s.FAILED_DEV_RESULT.read_bytes()
    attempt,account,pin=s.stage_inputs(admission,'dev_extract')
    assert attempt==s.DEV_RETRY_DIRECTORY
    assert account['prior_read_bytes']==15+23
    assert account['remaining_read_bytes']==s.ROLE_READ_LIMIT-38
    assert [x['charged_read_bytes'] for x in account['sources']]==[15,23]
    assert pin==admission['manual_retry']['plan_result']
    assert before==s.FAILED_DEV_RESULT.read_bytes()
    assert not attempt.exists()  # resolving never creates or dispatches work


@pytest.mark.parametrize('field',['stage','directory','failed_path','failed_hash','plan_path','plan_hash_format','extra_key'])
def test_wrong_retry_identity_rejected(controls,field):
    admission,_,_=controls; a=copy.deepcopy(admission); stage='dev_extract'
    if field=='stage': stage='train_extract'
    elif field=='directory': a['manual_retry']['attempt_directory']+='_another_timestamp'
    elif field=='failed_path': a['manual_retry']['failed_result']['path']='/different/RESULT.json'
    elif field=='failed_hash': a['manual_retry']['failed_result']['sha256']='0'*64
    elif field=='plan_path': a['manual_retry']['plan_result']['path']='/different/RESULT.json'
    elif field=='plan_hash_format': a['manual_retry']['plan_result']['sha256']='bad'
    else:a['manual_retry']['automatic_retry']=True
    with pytest.raises(ValueError):s.stage_inputs(a,stage)


def test_hash_and_preparation_changes_rejected(controls):
    admission,_,_=controls
    a=copy.deepcopy(admission);a['manual_retry']['plan_result']['sha256']='0'*64
    with pytest.raises(ValueError,match='hash changed'):s.stage_inputs(a,'dev_extract')
    a=copy.deepcopy(admission);a['preparation']={'path':'/different','sha256':'b'*64}
    with pytest.raises(ValueError,match='different identity'):s.stage_inputs(a,'dev_extract')


def test_retry_is_one_fixed_directory_and_duplicate_fails(controls):
    admission,_,_=controls
    attempt,_,_=s.stage_inputs(admission,'dev_extract')
    attempt.mkdir(exist_ok=False)
    again,_,_=s.stage_inputs(admission,'dev_extract')
    with pytest.raises(FileExistsError):again.mkdir(exist_ok=False)
    assert s.FAILED_DEV_RESULT.is_file()


def test_exhausted_cumulative_budget_and_bad_counters_rejected(controls,monkeypatch):
    admission,_,_=controls
    monkeypatch.setattr(s,'ROLE_READ_LIMIT',38)
    with pytest.raises(ValueError,match='Cumulative'):s.stage_inputs(admission,'dev_extract')
    for value in (-1,True,1.5):
        with pytest.raises(ValueError):s.charged_read_bytes(counters(value,0,0))
    with pytest.raises(KeyError):s.charged_read_bytes({})


def test_train_plan_cannot_skip_failed_dev_without_explicit_success_pin(controls):
    admission,_,_=controls
    a={'preparation':admission['preparation']}
    with pytest.raises(ValueError,match='incomplete'):s.stage_inputs(a,'train_plan')
    success=dict(schema='st_SP0_probe_stage_result_v1',status='COMPLETE',stage='dev_extract',
                 preparation=a['preparation'],manual_retry=admission['manual_retry'],
                 attempt_directory=str(s.DEV_RETRY_DIRECTORY))
    pin=b.write_once(s.DEV_RETRY_DIRECTORY/'RESULT.json',success)
    a['previous_stage_result']=pin
    attempt,account,used_pin=s.stage_inputs(a,'train_plan')
    assert used_pin==pin and account['prior_read_bytes']==0
    assert attempt==s.JOB/'stages/train_plan'
    assert b.read(s.FAILED_DEV_RESULT)[0]['status']=='STOPPED_NO_AUTO_RETRY'
    a['previous_stage_result']={**pin,'sha256':'0'*64}
    with pytest.raises(ValueError,match='hash changed'):s.stage_inputs(a,'train_plan')


def test_downstream_override_not_generic(controls):
    admission,_,_=controls
    a={'preparation':admission['preparation'],'previous_stage_result':{
        'path':str(s.DEV_RETRY_DIRECTORY/'RESULT.json'),'sha256':'a'*64}}
    for stage in ('dev_plan','dev_extract','train_extract','fit'):
        with pytest.raises(ValueError,match='Only train_plan'):s.stage_inputs(a,stage)
