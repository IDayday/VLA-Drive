"""Bounded IO scheduling tests; synthetic receipts do not claim simulator parity."""
import json
import threading
import time
from pathlib import Path
import pytest
from tools.action_effect import compare_methods as cm
from tools.action_effect.common import artifact,digest,read_json,write_json
from tools.action_effect import simulate_interventions as si


def identified(value):return {**value,'identity':digest(value)}


@pytest.fixture
def bank(tmp_path,monkeypatch):
    # Keep real comparison identity, per-scene SHA and join checks. Only the
    # already separately covered full native-wrapper decoder is replaced here.
    def build(name,scores):
        folder=tmp_path/name;folder.mkdir()
        ids=[f'scene_{i}' for i in range(len(scores))]
        population=identified({'base_checkpoint':'fixed','score_protocol_hash':'fixed_protocol',
            'numeric_runtime_hash':'fixed_FP32','context_manifest':'fixed_context','role':'dev','scene_ids':ids})
        write_json(folder/'POPULATION.json',population)
        pred=identified({'requested_scenes':len(ids),'entries':[{'scene_id':s,'identity':digest(s)} for s in ids],
            'registration':artifact(folder/'POPULATION.json')})
        write_json(folder/'PREDICTIONS.json',pred)
        reg=identified({'predictions':artifact(folder/'PREDICTIONS.json')})
        write_json(folder/'EVALUATION_REGISTRATION.json',reg)
        rows=[];pins=[]
        for sid,score in zip(ids,scores):
            metrics={k:1. for k in cm.METRICS};metrics.update(score=score,ego_progress=score)
            row=identified({'scene_id':sid,'source_log':'log_'+str(int(sid.split('_')[-1])//2),
                'status':'VALID_SCORED','metrics':metrics,'distance_m':.1,
                'prediction_identity':digest(sid),'fixture_native':{'status':'VALID_SCORED','metrics':metrics}})
            path=folder/'scenes'/sid/'RESULT.json';write_json(path,row);pins.append(artifact(path));rows.append(row)
        result={'status':'COMPLETE','registration_identity':reg['identity'],'entries':rows}
        path=folder/'RESULT.json';write_json(path,result)
        write_json(folder/'COMPLETE.json',{'result':artifact(path),'registration_identity':reg['identity'],'scenes':pins})
        return path
    monkeypatch.setattr(si,'verified_feedback',lambda row:{'canonical_record':row['fixture_native']})
    return build


def test_ordered_parallel_load_and_comparison_bytes_equal_serial(bank,tmp_path,monkeypatch):
    base=bank('base',[.9,.8,.7,.6,.5,.4,.3,.2]);other=bank('other',[.8,.9,.6,.7,.4,.5,.2,.3])
    calls=[];guard=threading.Lock()
    def native(row):
        with guard:calls.append((row['scene_id'],threading.get_ident()))
        time.sleep(.002*(8-int(row['scene_id'].split('_')[-1])))
        return {'canonical_record':row['fixture_native']}
    monkeypatch.setattr(si,'verified_feedback',native)
    serial=cm.load(base,verification_workers=1);calls.clear()
    parallel=cm.load(base,verification_workers=8)
    assert serial==parallel
    assert {sid for sid,_ in calls}=={r['scene_id'] for r in serial[0]['entries']}
    assert len(calls)==8 and len({worker for _,worker in calls})>1
    for workers in (1,8):
        cm.main(['--baseline',str(base),'--method','method='+str(other),'--bootstrap-repetitions','20',
            '--verification-workers',str(workers),'--output',str(tmp_path/f'out{workers}')])
    for name in ('COMPARISON.json','FINAL_SYSTEM_RESULTS.csv'):
        assert (tmp_path/'out1'/name).read_bytes()==(tmp_path/'out8'/name).read_bytes()
    assert read_json(tmp_path/'out1/VERIFICATION_RUNTIME.json')['verification_workers']==1
    receipt=read_json(tmp_path/'out8/VERIFICATION_RUNTIME.json')
    assert receipt['verification_workers']==8 and receipt['sources']['files']


@pytest.mark.parametrize('workers',[1,8])
def test_bad_scene_checksum_rejects_entire_comparison_without_output(bank,tmp_path,workers):
    base=bank('base',[.9,.8,.7]);other=bank('other',[.8,.9,.6])
    scene=other.parent/'scenes/scene_1/RESULT.json'
    scene.write_bytes(scene.read_bytes()+b' ')
    output=tmp_path/'rejected'
    with pytest.raises(ValueError,match='scene receipt changed'):
        cm.main(['--baseline',str(base),'--method','method='+str(other),
            '--verification-workers',str(workers),'--output',str(output)])
    assert not output.exists()


@pytest.mark.parametrize('workers',[1,8])
def test_native_metric_mismatch_rejects_after_hash_checks(bank,tmp_path,monkeypatch,workers):
    base=bank('base',[.9,.8,.7]);other=bank('other',[.8,.9,.6])
    def wrong(row):
        native=json.loads(json.dumps(row['fixture_native']))
        if row['scene_id']=='scene_1':native['metrics']['score']=.12345
        return {'canonical_record':native}
    monkeypatch.setattr(si,'verified_feedback',wrong)
    output=tmp_path/'rejected'
    with pytest.raises(ValueError,match='independent native receipt'):
        cm.main(['--baseline',str(base),'--method','method='+str(other),
            '--verification-workers',str(workers),'--output',str(output)])
    assert not output.exists()


@pytest.mark.parametrize('workers',[0,9,-1,True,1.0,None])
def test_programmatic_worker_budget_rejected_before_io(workers):
    with pytest.raises(ValueError,match='verification_workers'):cm.load('/not/accessed',verification_workers=workers)


@pytest.mark.parametrize('workers',['0','9','1.0'])
def test_cli_worker_budget_strict(workers):
    with pytest.raises(SystemExit):cm.main(['--baseline','/not/accessed','--output','/not/written','--verification-workers',workers])


def test_default_remains_serial_and_source_mutation_refused(bank,tmp_path,monkeypatch):
    base=bank('base',[.9,.8]);other=bank('other',[.8,.9])
    cm.main(['--baseline',str(base),'--method','m='+str(other),'--output',str(tmp_path/'default')])
    assert read_json(tmp_path/'default/VERIFICATION_RUNTIME.json')['verification_workers']==1
    count=[0]
    def source():count[0]+=1;return {'version':count[0]}
    monkeypatch.setattr(cm,'_verification_source',source)
    with pytest.raises(ValueError,match='source changed'):
        cm.main(['--baseline',str(base),'--method','m='+str(other),'--verification-workers','8','--output',str(tmp_path/'changed')])
    assert not (tmp_path/'changed').exists()
