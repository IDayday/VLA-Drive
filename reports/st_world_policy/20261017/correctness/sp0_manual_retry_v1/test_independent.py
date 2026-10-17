"""Independent exact retry admission tests; no production child is dispatched."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

HERE=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(HERE/'sp0_dev_extract_manual_retry_v1'))
from test_manual_retry import controls, counters, s, b


def replace_failed(admission,value,monkeypatch):
    s.FAILED_DEV_RESULT.write_text(json.dumps(value))
    pin=b.read(s.FAILED_DEV_RESULT)[1]
    monkeypatch.setattr(s,'FAILED_DEV_SHA256',pin['sha256'])
    admission['manual_retry']['failed_result']=pin


@pytest.mark.parametrize('mutation',['FAILED','COMPLETE','wrongstage','wrongreason','livechild'])
def test_semantically_wrong_failed_receipt_even_with_matching_pin(controls,monkeypatch,mutation):
    admission,_,failed=deepcopy(controls)
    if mutation in ('FAILED','COMPLETE'):failed['status']=mutation
    elif mutation=='wrongstage':failed['stage']='train_extract'
    elif mutation=='wrongreason':failed['resources']['limit_reason']='GPU_LIMIT'
    else:failed['resources']['process_cleanup']['live_owned_remaining']=1
    replace_failed(admission,failed,monkeypatch)
    with pytest.raises(ValueError,match='pinned closed monitor-failure'):s.stage_inputs(admission,'dev_extract')


@pytest.mark.parametrize('field',['attempt_directory','failed_result','plan_result'])
def test_lexical_traversal_cannot_alias_exact_path(controls,field):
    admission,_,_=deepcopy(controls)
    if field=='attempt_directory':
        p=Path(admission['manual_retry'][field]);admission['manual_retry'][field]=str(p.parent/'ignored')+'/../'+p.name
    else:
        p=Path(admission['manual_retry'][field]['path']);admission['manual_retry'][field]['path']=str(p.parent/'ignored')+'/../'+p.name
    with pytest.raises(ValueError):s.stage_inputs(admission,'dev_extract')


def test_sum_attempt_maxima_not_maximum_of_summed_channels(controls,monkeypatch):
    admission,plan,failed=deepcopy(controls)
    plan['resources']['counters']=counters(0,0,100)
    p=Path(admission['manual_retry']['plan_result']['path']);p.write_text(json.dumps(plan))
    admission['manual_retry']['plan_result']=b.read(p)[1]
    failed['resources']['counters']=counters(200,0,0)
    replace_failed(admission,failed,monkeypatch)
    attempt,accounting,_=s.stage_inputs(admission,'dev_extract')
    assert accounting['prior_read_bytes']==300
    assert accounting['remaining_read_bytes']==s.ROLE_READ_LIMIT-300
    assert [x['charged_read_bytes'] for x in accounting['sources']]==[100,200]
    assert not attempt.exists()


@pytest.mark.parametrize('retry',[False,True])
def test_actual_main_duplicate_attempt_rejected_before_popen(controls,monkeypatch,tmp_path,retry):
    from contextlib import ExitStack
    admission,_,_=deepcopy(controls)
    if not retry:admission.pop('manual_retry')
    preparation={'runtime':{'path':str(tmp_path),'source_sha':'a'*40}}
    admission.update(stage='dev_extract',preparation={'path':'/fixture/prep','sha256':'b'*64},cpu_affinity=[0,1],gpu={'index':0,'uuid':'GPU-fixture'})
    pin={'path':'/fixture/admission','sha256':'c'*64}
    attempt=s.DEV_RETRY_DIRECTORY if retry else s.JOB/'stages/dev_extract'
    attempt.mkdir(parents=True,exist_ok=True)
    before={p.name:p.read_bytes() for p in attempt.iterdir() if p.is_file()}
    monkeypatch.setattr(s.sys,'argv',['supervisor','--admission',pin['path'],'--admission-sha256',pin['sha256']])
    monkeypatch.setattr(s,'read',lambda p,*a,**kw:(admission,pin) if str(p)==pin['path'] else (preparation,{}))
    monkeypatch.setattr(s,'validate',lambda *a:(1,1))
    monkeypatch.setattr(s,'shared_mount',lambda:None)
    monkeypatch.setattr(s,'stage_inputs',lambda *a:(attempt,{'remaining_read_bytes':1},None))
    monkeypatch.setattr(s,'fresh_gpu',lambda *a:None)
    monkeypatch.setattr(s.os,'sched_setaffinity',lambda *a:None)
    monkeypatch.setattr(s.subprocess,'check_output',lambda args,**kw: 'a'*40+'\n' if 'rev-parse' in args else '')
    monkeypatch.setattr(s,'acquire',lambda path,stack:stack.enter_context((tmp_path/'fixturelock').open('a+b')))
    def forbidden(*a,**kw):raise AssertionError('Duplicate must never dispatch')
    monkeypatch.setattr(s,'supervise',forbidden)
    with pytest.raises(FileExistsError):s.main()
    assert before=={p.name:p.read_bytes() for p in attempt.iterdir() if p.is_file()}


def test_explicit_train_predecessor_requires_retry_lineage(controls):
    admission,_,_=controls
    value={'schema':'st_SP0_probe_stage_result_v1','status':'COMPLETE','stage':'dev_extract',
        'preparation':admission['preparation'],'attempt_directory':str(s.DEV_RETRY_DIRECTORY)}
    pin=b.write_once(s.DEV_RETRY_DIRECTORY/'RESULT.json',value)
    with pytest.raises(ValueError,match='Only one explicit'):
        s.stage_inputs({'preparation':admission['preparation'],'previous_stage_result':pin},'train_plan')
