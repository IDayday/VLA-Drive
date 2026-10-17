"""Synthetic structure/fault tests only. Never loads a real log, image or teacher."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import pickle

import pytest
from PIL import Image

from tools.st_world_policy import prepare_current_teacher as current
from tools.st_world_policy import prepare_future_metadata as future
from tools.st_world_policy.prepare_history import build_target_scene
from starVLA.dataloader.st_world_future_provenance import validate_ledger, LogCoverage
from starVLA.dataloader.st_world_metadata_index import build_index, MetadataIndex
from starVLA.dataloader.st_world_history import file_sha256
from starVLA.dataloader.st_world_targets import atomic_json
from starVLA.dataloader.st_world_schema import identity_hash
from tests.st_world.test_teacher_cache import current_population_fixture, fixture


def setup(tmp_path):
    c=current_population_fixture(tmp_path)
    current.plan(c); current.build(c)
    logs=tmp_path/'raw'; logs.mkdir()
    for token,log in [('a','log_a'),('b','log_b')]:
        frames=[]
        for i in range(9):
            frames.append({'token':token if i==0 else token+str(i),'timestamp':1000000+500000*i,
                'cams':{name:{'data_path':token+'.png'} for name in ('CAM_F0','CAM_L0','CAM_R0')}})
        if token=='b': del frames[-1]['cams']['CAM_R0']
        (logs/(log+'.pkl')).write_bytes(pickle.dumps(frames))
    a=future.parser().parse_args(['plan','--student-root',c.student_root,'--partition',c.partition,
        '--current-population',c.output,'--raw-log-root',str(logs),'--sensor-root',str(tmp_path),
        '--split','train','--output',str(tmp_path/'future'),'--max-scenes','2','--workers','1'])
    plan=future.plan(a)
    reg=future.signed({'schema':future.REG_SCHEMA,'plan_identity':plan['identity'],
        'plan_sha256':file_sha256(Path(a.output)/'PLAN.json'),'phase':'data_preparation',
        'approved_scope':plan['scope'],'gpu_authorized':False,'teacher_extraction_authorized':False,
        'training_authorized':False})
    registration=tmp_path/'REGISTERED.json'; atomic_json(registration,reg)
    run=SimpleNamespace(plan=str(Path(a.output)/'PLAN.json'),registration=str(registration),
        registration_sha256=file_sha256(registration))
    return a,run,plan,reg


def test_plan_no_raw_image_or_tensor_open(tmp_path,monkeypatch):
    a,run,plan,reg=setup(tmp_path)
    original=Path.open
    def no_data(self,*args,**kw):
        assert self.suffix not in ('.png','.pkl','.pt','.safetensors')
        return original(self,*args,**kw)
    monkeypatch.setattr(Path,'open',no_data)
    assert future.plan(a)==plan
    assert plan['status']=='PLAN_ONLY' and plan['selected_scene_count']==2
    assert plan['reference_plus_new_raw_read_estimate_bytes']==4*sum(x['bytes'] for x in plan['raw_log_stats'])


def test_all_future_rows_equal_original_masks_and_ledger(tmp_path,monkeypatch):
    a,run,p,reg=setup(tmp_path)
    monkeypatch.setattr(Image.Image,'load',lambda *_: (_ for _ in ()).throw(AssertionError('No decode')))
    result=future.build(run)
    assert result['status']=='COMPLETE' and result['scenes']==2
    assert result['tensor_disk_bytes']==0 and result['training_admission'] is False
    manifest=json.loads((Path(a.output)/'TEACHER_REQUESTS.json').read_text())
    assert manifest['times_s']==[0.,.5,1.,2.,4.]
    for scene in manifest['scenes']:
        frames=pickle.loads((Path(a.raw_log_root)/(scene['log']+'.pkl')).read_bytes())
        expected=build_target_scene(frames,scene['token'],scene['log'],'train',a.sensor_root)
        assert scene==expected
    assert manifest['scenes'][1]['frames'][-1]['views'][-1] is None
    coverage=LogCoverage()
    for scene in manifest['scenes']: coverage.add(scene['token'],scene['log'])
    assert validate_ledger(manifest,coverage=coverage.records())['sha256']==manifest['source_logs_ledger']['sha256']
    assert 'source_logs' not in manifest
    again=future.build(run); assert again['reused'] is True
    assert again['request_identity']==result['request_identity']


def test_streamed_document_matches_atomic_json_and_identity(tmp_path):
    _,_,p,_=setup(tmp_path)
    budget=future.PreparationBudget(tmp_path,p['limits'])
    rows=[{'token':'一','log':'a','nested':[{'key':2}]},{'token':'b','log':'b'}]
    header={'schema':'example','z':{'other':3},'a':[1,2]}
    path=tmp_path/'stream.json'
    identity=future.write_row_document(path,header,lambda:iter(rows),budget)
    expected=future.signed(dict(header,scenes=rows))
    assert identity==expected['identity']
    atomic_json(tmp_path/'reference.json',expected)
    assert path.read_bytes()==(tmp_path/'reference.json').read_bytes()


def test_one_log_lookup_and_bounded_sha_memo_reuse(tmp_path):
    _,_,p,_=setup(tmp_path); budget=future.PreparationBudget(tmp_path,p['limits'])
    frames=pickle.loads((Path(p['raw_log_root'])/'log_a.pkl').read_bytes())
    memo=future.ImageMemo(budget); lookup=future.TargetLogLookup(frames)
    first=lookup.scene('a','log_a','train',p['sensor_roots'],memo)
    second=lookup.scene('a1','log_a','train',p['sensor_roots'],memo)
    assert memo.hits>20 and len(memo.values)==1
    assert second['frames'][-1]['timestamp_us'] is None
    assert first['frames'][0]['timestamp_us']==1000000
    with pytest.raises(future.BudgetExceeded,match='MEMO'):
        future.ImageMemo(budget,max_bytes=1)(tmp_path/'a.png')


@pytest.mark.parametrize('kind',['duplicate','nonmonotonic','ambiguous','missing_current'])
def test_legacy_selection_errors_remain_errors(tmp_path,kind):
    a,_,p,_=setup(tmp_path)
    frames=pickle.loads((Path(p['raw_log_root'])/'log_a.pkl').read_bytes())
    if kind=='duplicate': frames[1]['token']='a'
    if kind=='nonmonotonic': frames[1]['timestamp']=frames[0]['timestamp']
    if kind=='ambiguous':
        frames[1]['timestamp']=1490000
        frames.insert(2,dict(frames[1],token='other',timestamp=1510000))
    if kind=='missing_current': frames[0]['cams']={}
    with pytest.raises(ValueError):
        future.TargetLogLookup(frames).scene('a','log_a','train',p['sensor_roots'],lambda _: {})


def test_interrupted_log_and_receipt_resume_exact_same_identity(tmp_path,monkeypatch):
    a,run,_,_=setup(tmp_path); original=future.build_log
    def fail(*args,**kw):
        if args[2]=='log_b': raise RuntimeError('injected second-log failure')
        return original(*args,**kw)
    monkeypatch.setattr(future,'build_log',fail)
    with pytest.raises(RuntimeError,match='injected'): future.build(run)
    out=Path(a.output); first=list((out/'log_shards').glob('*.json'))
    assert len(first)==1 and not (out/'COMPLETE.json').exists()
    first_bytes=first[0].read_bytes()
    monkeypatch.setattr(future,'build_log',original)
    writer=future.write_json
    def fail_receipt(path,value,budget):
        if Path(path).name=='COMPLETE.json': raise RuntimeError('completion window')
        return writer(path,value,budget)
    monkeypatch.setattr(future,'write_json',fail_receipt)
    with pytest.raises(RuntimeError,match='completion window'): future.build(run)
    before=(out/'TEACHER_REQUESTS.json').read_bytes()
    monkeypatch.setattr(future,'write_json',writer)
    done=future.build(run)
    assert done['status']=='COMPLETE' and (out/'TEACHER_REQUESTS.json').read_bytes()==before
    assert first[0].read_bytes()==first_bytes


@pytest.mark.parametrize('kind',['raw','image','ego','partition','selection'])
def test_changed_sources_never_reused(tmp_path,kind):
    a,run,_,_=setup(tmp_path); future.build(run)
    target={'raw':Path(a.raw_log_root)/'log_a.pkl','image':tmp_path/'a.png',
        'ego':Path(a.student_root)/'ego/a.pt','partition':Path(a.partition),
        'selection':Path(a.current_population)/'SELECTED_SCENES.json'}[kind]
    target.write_bytes(target.read_bytes()+b' ')
    with pytest.raises(ValueError): future.build(run)


def test_budget_hard_stop_preserves_complete_shard(tmp_path,monkeypatch):
    a,run,_,_=setup(tmp_path); original=future.build_log
    def stop(*args,**kw):
        if args[2]=='log_b': raise future.BudgetExceeded('SOURCE_READ_BYTES_BUDGET')
        return original(*args,**kw)
    monkeypatch.setattr(future,'build_log',stop)
    result=future.build(run)
    assert result['status']=='PARTIAL' and result['reason']=='SOURCE_READ_BYTES_BUDGET'
    assert len(list((Path(a.output)/'log_shards').glob('*.json')))==1
    assert not (Path(a.output)/'COMPLETE.json').exists()


def test_individual_resource_limits_and_unregistered_build_reject(tmp_path):
    a,run,p,_=setup(tmp_path)
    for key,value in [('workers',3),('read_mib_per_second',65),('max_metadata_bytes',2*future.GiB+1),('max_rss_bytes',4*future.GiB+1)]:
        wrong=dict(p['limits'],**{key:value})
        with pytest.raises(ValueError): future.validate_limits(wrong)
    run.registration_sha256='1'*64
    with pytest.raises(ValueError,match='checksum'): future.build(run)
    b=future.PreparationBudget(tmp_path,p['limits']); b.limits=dict(p['limits'],max_source_read_bytes=1)
    with pytest.raises(future.BudgetExceeded,match='READ_BYTES'): b.read_allowance(2)
    b.limits=dict(p['limits'],max_metadata_bytes=b.disk_bytes)
    with pytest.raises(future.BudgetExceeded,match='DISK'): b.reserve(1)


def test_new_request_index_verifies_ledger_and_exact_logs(tmp_path):
    a,run,p,_=setup(tmp_path); future.build(run)
    out=Path(a.output); request=out/'TEACHER_REQUESTS.json'; manifest=json.loads(request.read_text())
    c=fixture(tmp_path)[0]
    r=build_index(tmp_path/'index',requests=request,expected_request_identity=manifest['identity'],contract=c,max_scenes=2)
    assert 'source_logs_ledger' in r['dependencies']
    with MetadataIndex(tmp_path/'index',expected_request_identity=manifest['identity'],expected_index_identity=r['identity']) as idx:
        assert list(idx.manifest['scenes'])==manifest['scenes']
    assert build_index(tmp_path/'index',requests=request,expected_request_identity=manifest['identity'],contract=c,max_scenes=2)==r
    ledger=Path(manifest['source_logs_ledger']['path']); old=ledger.read_bytes()
    ledger.write_bytes(old+b' ')
    with pytest.raises(ValueError,match='changed|checksum'):
        MetadataIndex(tmp_path/'index',expected_request_identity=manifest['identity'],expected_index_identity=r['identity'])


@pytest.mark.parametrize('kind',['hash','source','log','tokens','omitted','both'])
def test_ledger_negative_admission(tmp_path,kind):
    a,run,_,_=setup(tmp_path); future.build(run)
    path=Path(a.output)/'TEACHER_REQUESTS.json'; m=json.loads(path.read_text())
    ref=m['source_logs_ledger']; ledger=Path(ref['path']); body=json.loads(ledger.read_text())
    if kind=='hash': ref['sha256']='0'*64
    elif kind=='both': m['source_logs']=[]
    else:
        if kind=='source': body['partition_sha256']='0'*64
        elif kind=='log': body['logs'][0]['log']='different'
        elif kind=='tokens': body['logs'][0]['tokens_sha256']='0'*64
        elif kind=='omitted': body['logs']=body['logs'][:-1]
        body.pop('identity'); body=future.signed(body); atomic_json(ledger,body)
        ref['sha256']=file_sha256(ledger); ref['identity']=body['identity']
    m.pop('identity'); m=future.signed(m); atomic_json(path,m)
    with pytest.raises(ValueError):
        build_index(tmp_path/'bad_index',requests=path,expected_request_identity=m['identity'],contract=fixture(tmp_path)[0],max_scenes=2)
    assert not (tmp_path/'bad_index/COMPLETE.json').exists()


def test_many_log_ledger_keeps_request_header_bound(tmp_path):
    from starVLA.dataloader.st_world_future_provenance import LEDGER_SCHEMA,REF_SCHEMA
    from starVLA.dataloader.st_world_metadata_index import _atomic_rows,canonical
    from tests.st_world.test_ondemand_teacher import fixture as request_fixture
    c,m=request_fixture(tmp_path)
    shared={k:m[k] for k in ('split','source_student_identity','source_index_sha256','partition_sha256')}
    shared.update(registration_identity='3'*64,selection_identity='4'*64)
    coverage=LogCoverage(); logs=[]; scenes=[]
    for i in range(1176):
        log='log_'+str(i); token='scene_'+str(i)
        coverage.add(token,log)
        scene=deepcopy(m['scenes'][0]); scene.update(log=log,token=token); scenes.append(scene)
        logs.append({'log':log,'path':str(tmp_path/(log+'.pkl')),'sha256':'a'*64,'bytes':10,
            'mtime_ns':1,'scenes':1,'tokens_sha256':identity_hash([token])})
    ledger=future.signed(dict(shared,schema=LEDGER_SCHEMA,logs=logs)); path=tmp_path/'ledger.json'; atomic_json(path,ledger)
    m.update(shared,scenes=scenes,source_logs_ledger={'schema':REF_SCHEMA,'ledger_schema':LEDGER_SCHEMA,
        'path':str(path),'sha256':file_sha256(path),'identity':ledger['identity']})
    m.pop('identity'); m=future.signed(m); request=tmp_path/'many.json'; atomic_json(request,m)
    h,counts,digest=_atomic_rows(request,{'scenes'},lambda *_:None)
    assert counts['scenes']==1176 and digest==m['identity']
    assert len(canonical(h).encode())<4096 and path.stat().st_size>65536
    assert validate_ledger(m,coverage=coverage.records())['sha256']==file_sha256(path)
