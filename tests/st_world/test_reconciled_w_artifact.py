"""Synthetic metadata fixtures only; no real W validation, extraction or dispatch."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import pytest

HELPERS=Path(__file__).resolve().parents[2]/'reports/st_world_policy/20261017/spatial_probe_actual_preparation_v1'
sys.path.insert(0,str(HELPERS))
import prepare_bindings as b
import reconcile_artifact_exit as r
import supervise_once as s


def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,sort_keys=True)+'\n')
    return {'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}


@pytest.fixture
def fixture(tmp_path,monkeypatch):
    job=tmp_path/'job';cache=tmp_path/'cache';attempt=job/'stages/dev_extract_manual_retry_v1'
    cache.mkdir();attempt.mkdir(parents=True)
    for name,value in [('JOB',job),('DEV_RETRY_DIRECTORY',attempt),('ROOTS',{'dev':cache})]:monkeypatch.setattr(s,name,value)
    prep=put(job/'PREPARATION.json',{'caches':{'dev':{'root':str(cache),'identity':'a'*64}}})
    original=put(job/'stages/dev_extract/RESULT.json',{'fixture':'original closed failure'})
    monkeypatch.setattr(s,'FAILED_DEV_RESULT',Path(original['path']));monkeypatch.setattr(s,'FAILED_DEV_SHA256',original['sha256'])
    plan=put(job/'stages/dev_plan/RESULT.json',{'fixture':'original successful plan'})
    counters={'hash_logical_bytes_admitted':10,'torch_load_nominal_bytes':20,'proc_self_io_delta':{'rchar':25},'limit_exceeded':None}
    res={'exit_code':0,'limit_reason':r.REASON,'first_monitor_error':None,'counters':counters,
         'process_cleanup':{'live_owned_remaining':0,'verified_processes':8}}
    complete={'completed':16,'requested':16,'identity':'a'*64,'scientific_result':False,'status':'COMPLETE'}
    failed={'schema':'st_SP0_probe_stage_result_v1','status':'STOPPED_NO_AUTO_RETRY','stage':'dev_extract',
        'preparation':prep,'resources':res,'cli_result':complete,'read_accounting':{'prior_read_bytes':50,'role_limit_bytes':256<<30},
        'manual_retry':{'failed_result':original,'plan_result':plan,'attempt_directory':str(attempt)},'attempt_directory':str(attempt)}
    failed_pin=put(attempt/'RESULT.json',failed);monkeypatch.setattr(r,'FAILED_SHA',failed_pin['sha256'])
    cli=put(attempt/'CLI_RESULT.json',complete);cp=put(cache/'COMPLETE.json',complete)
    rows=[{'token':f'token{i}','log':f'log{i}'} for i in range(16)]
    sig={'identity':'a'*64,'schema':'st_frozen_spatial_w_v1','targets_read':False,'precision':'FP32_native_master_TF32_off',
        'binding':{'checkpoint_identity':'c'*64,'checkpoint_tag':'fixture5000'},'rows':rows}
    sig_pin=put(cache/'identity.json',sig)
    pins=[failed_pin,cli,cp,prep,sig_pin];records=[]
    for row in rows:
        body=b'CPU fixture; not a real teacher tensor'
        payload=cache/(row['token']+'.safetensors');payload.write_bytes(body)
        sha=hashlib.sha256(body).hexdigest()
        pins.append(put(cache/(row['token']+'.json'),{'identity':'a'*64,**row,'sha256':sha}))
        records.append({**row,'payload_sha256':sha,'shape':[144,2048],'dtype':'torch.float32','mask_shape':[144],
            'finite':True,'mask_all_true':True,'feature_elements':294912,'mask_elements':144})
    review={'schema':r.REVIEW_SCHEMA,'status':'COMMITTED_ARTIFACTS_INDEPENDENTLY_VALIDATED',
        'original_stage_status_unchanged':'STOPPED_NO_AUTO_RETRY','original_supervisor_reason':r.REASON,
        'cache_validity_does_not_rewrite_failed_launch':True,'matching_old_session_processes_at_review':[],
        'gpu_initialized':False,'launch_approved':False,'downstream_transition_approved':False,'automatic_retry':False,
        'compute_resources':res,'input_pins_unchanged':{p['path']:{'sha256':p['sha256']} for p in pins},
        'signature_identity':'a'*64,'checkpoint_identity':'c'*64,'checkpoint_tag':'fixture5000','scenes':16,'records':records,
        'retry_charged_read_bytes':30,'prior_role_read_bytes':50,'remaining_role_read_bytes_before_review':(256<<30)-80,
        'payload_hash_plus_mmap_nominal_bytes':100,'remaining_role_read_bytes_less_review_payload_upper':(256<<30)-180}
    review_pin=put(job/'review.json',review);monkeypatch.setattr(r,'REVIEW_SHA',review_pin['sha256'])
    body=r.create_receipt(review_pin,failed_pin,prep,cache,b.read)
    receipt_pin=put(job/'ADDITIVE_RECONCILIATION.json',body)
    return {'job':job,'cache':cache,'attempt':attempt,'prep':prep,'failed':failed,'failed_pin':failed_pin,
        'review':review,'review_pin':review_pin,'body':body,'receipt_pin':receipt_pin,'monkeypatch':monkeypatch}


def validate(f,pin=None):
    return r.validate_receipt(pin or f['receipt_pin'],failed_pin=f['failed_pin'],preparation_pin=f['prep'],cache_root=f['cache'],read=b.read)


def test_exact_reconciled_train_plan_without_rewriting_failure_or_budget(fixture):
    f=fixture;before=Path(f['failed_pin']['path']).read_bytes()
    a={'preparation':f['prep'],'previous_stage_result':f['failed_pin'],'previous_artifact_reconciliation':f['receipt_pin']}
    attempt,account,pin=s.stage_inputs(a,'train_plan')
    assert attempt==f['job']/'stages/train_plan' and not attempt.exists()
    assert pin==f['failed_pin'] and account['prior_read_bytes']==0 # new train role, not reset dev budget
    assert account['upstream_supervisor_status_preserved']=='STOPPED_NO_AUTO_RETRY'
    assert Path(f['failed_pin']['path']).read_bytes()==before
    assert validate(f)['upstream_accounting']['additional_extraction_attempts']==0
    assert validate(f)['upstream_accounting']['attempt_count_preserved']==2


def test_no_receipt_still_rejects_failed_stage(fixture):
    f=fixture
    with pytest.raises(ValueError,match='incomplete'):
        s.stage_inputs({'preparation':f['prep'],'previous_stage_result':f['failed_pin']},'train_plan')


@pytest.mark.parametrize('stage',['dev_plan','dev_extract','train_extract','fit'])
def test_only_train_plan_can_consume(fixture,stage):
    f=fixture
    with pytest.raises(ValueError):s.check_manual_fields({'previous_stage_result':f['failed_pin'],
        'previous_artifact_reconciliation':f['receipt_pin']},stage)


@pytest.mark.parametrize('field',['status','eligible_consumer','original_failed_result','cache_complete','independent_review',
                                    'payload_snapshots','launch_approved','upstream_accounting'])
def test_resigned_receipt_cannot_change_evidence(fixture,field):
    f=fixture;bad=deepcopy(f['body'])
    if field=='status':bad[field]='COMPLETE'
    elif field=='eligible_consumer':bad[field]='fit'
    elif field in ('original_failed_result','cache_complete','independent_review'):bad[field]['sha256']='0'*64
    elif field=='payload_snapshots':bad[field]=bad[field][:-1]
    elif field=='launch_approved':bad[field]=True
    else:bad[field]['additional_extraction_attempts']=1
    bad['identity']=r.identity({k:v for k,v in bad.items() if k!='identity'})
    pin=put(f['job']/'bad_receipt.json',bad)
    with pytest.raises(ValueError):validate(f,pin)


def test_wrong_file_hash_or_cache_rejected(fixture):
    f=fixture
    with pytest.raises(ValueError):validate(f,{**f['receipt_pin'],'sha256':'0'*64})
    put(f['cache']/'COMPLETE.json',{'status':'COMPLETE','identity':'unverified'})
    with pytest.raises(ValueError):validate(f)


@pytest.mark.parametrize('change',['cleanup','reason','exit','cache_unverified','missing_record','incomplete_record'])
def test_resigned_fixture_review_does_not_override_semantics(fixture,change):
    f=fixture;review=deepcopy(f['review']);failed=deepcopy(f['failed'])
    if change=='cleanup':failed['resources']['process_cleanup']['live_owned_remaining']=1
    elif change=='reason':failed['resources']['limit_reason']='MONITOR_UNREADABLE'
    elif change=='exit':failed['resources']['exit_code']=2
    elif change=='cache_unverified':review['status']='NOT_VERIFIED'
    elif change=='missing_record':review['records']=review['records'][:-1]
    else:review['records'][0]['finite']=False
    new_failed=put(Path(f['failed_pin']['path']),failed)
    review['compute_resources']=failed['resources']
    review['input_pins_unchanged'][new_failed['path']]['sha256']=new_failed['sha256']
    pin=put(Path(f['review_pin']['path']),review)
    # Fixtures change trust anchors to exercise semantic guards, never real pins.
    f['monkeypatch'].setattr(r,'REVIEW_SHA',pin['sha256']);f['monkeypatch'].setattr(r,'FAILED_SHA',new_failed['sha256'])
    with pytest.raises(ValueError):r.create_receipt(pin,new_failed,f['prep'],f['cache'],b.read)


def test_missing_or_changed_payload_stat_rejected_without_payload_read(fixture):
    f=fixture;(f['cache']/'token0.safetensors').write_bytes(b'changed fixture bytes')
    with pytest.raises(ValueError):validate(f)


def test_reconciliation_is_idempotent_metadata(fixture):
    f=fixture
    assert r.create_receipt(f['review_pin'],f['failed_pin'],f['prep'],f['cache'],b.read)==f['body']


def test_consumer_never_opens_payload(fixture,monkeypatch):
    original=Path.open
    def guarded(path,*args,**kwargs):
        if path.suffix=='.safetensors':raise AssertionError('Reconciliation must not reread W')
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'open',guarded)
    assert validate(fixture)['payload_content_revalidated_here'] is False


def test_missing_payload_not_accepted(fixture):
    (fixture['cache']/'token0.safetensors').unlink()
    with pytest.raises(FileNotFoundError):validate(fixture)
