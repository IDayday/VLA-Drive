"""CPU control fixtures only; no real builder, model, ledger or process launch."""
import copy
from contextlib import contextmanager
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import pytest

spec=importlib.util.spec_from_file_location('future_build',Path(__file__).with_name('run_build_once.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def admission(monkeypatch,role='dev'):
    monkeypatch.setattr(m.os,'sched_getaffinity',lambda _: {0,1})
    return {'schema':'st_full_future_build_resource_admission_v1','status':'PASS_BOUNDED_'+role.upper()+'_BUILD_ONLY',
        'role':role,'limits':m.limits(role),'hostname':m.socket.gethostname(),'reviewer':'independent_fixture',
        'preparation':{'path':str(m.PREP),'sha256':m.PREP_SHA},
        'issued_at_unix':1,'latest_start_unix':100,'valid_until_unix':20000}


@pytest.mark.parametrize('role',['dev','train'])
def test_fixed_finite_limits(role,monkeypatch):
    a=admission(monkeypatch,role);m.validate(a,role,10)
    assert a['limits']['cpu_affinity']==[0,1] and a['limits']['gpu'] is False
    assert a['limits']['read_bytes']==(16 if role=='dev' else 512)*(1<<30)
    assert a['limits']['wall_seconds']==(1800 if role=='dev' else 14400)


@pytest.mark.parametrize('mutation',['status','role','hostname','own_review','expired','short','nan','prep','gpu','retry','rss','bool_cpu'])
def test_invalid_admission_never_passes(monkeypatch,mutation):
    a=admission(monkeypatch)
    if mutation=='status':a['status']='APPROVED'
    elif mutation=='role':a['role']='index'
    elif mutation=='hostname':a['hostname']='other'
    elif mutation=='own_review':a['reviewer']='/root/st_indexed_composition'
    elif mutation=='expired':a['latest_start_unix']=9
    elif mutation=='short':a['valid_until_unix']=100
    elif mutation=='nan':a['valid_until_unix']=float('nan')
    elif mutation=='prep':a['preparation']['sha256']='0'*64
    elif mutation=='bool_cpu':a['limits']['cpu_affinity']=[False,1]
    elif mutation=='gpu':a['limits']['gpu']=True
    elif mutation=='retry':a['limits']['automatic_retry']=True
    else:a['limits']['rss_bytes']=8<<30
    with pytest.raises(ValueError):m.validate(a,'dev',10)


def test_both_pipe_buffers_are_bounded_and_failure_retains_prefix():
    buffers={'stdout':bytearray(),'stderr':bytearray()}
    for name in buffers:
        assert m.append_pipe(buffers,name,b'a'*m.MiB)
        assert not m.append_pipe(buffers,name,b'excess')
        assert len(buffers[name])==m.MiB


def test_mutable_counter_reads_are_bounded_under_writer_lock(tmp_path):
    (tmp_path/'CONTROL_BUILD_COUNTER_LOCK.json').write_text('')
    (tmp_path/'CONTROL_BUILD_COUNTERS.json').write_bytes(b'x'*65537)
    with pytest.raises(ValueError,match='Counter size'):m.read_counter(tmp_path)


def mock_storage(tmp_path,*,charged=0,cap=4096):
    state={'entries':0,'writes':0}
    ctx=SimpleNamespace(limits={'max_disk_bytes':cap},check=lambda **_: {'charged_disk_bytes':charged})
    @contextmanager
    def writing(**_):
        state['entries']+=1;yield ctx
    def atomic(path,value):state['writes']+=1;path.write_text(json.dumps(value))
    return SimpleNamespace(metadata_writing=writing),SimpleNamespace(_atomic=atomic,json_write_bound=lambda _:128),state


def test_one_shot_refuses_existing_marker_before_write(tmp_path):
    path=tmp_path/'CONTROL_BUILD_STARTED.json';path.write_text('original')
    metadata,storage,state=mock_storage(tmp_path)
    with pytest.raises(ValueError,match='One-attempt'):m.publish_guarded(metadata,storage,{},path,{})
    assert state['writes']==0 and path.read_text()=='original'


def test_full_root_rejects_result_before_atomic_write(tmp_path):
    path=tmp_path/'CONTROL_BUILD_RESULT.json';metadata,storage,state=mock_storage(tmp_path,charged=4000)
    with pytest.raises(ValueError,match='headroom'):m.publish_guarded(metadata,storage,{},path,{})
    assert state['writes']==0 and not path.exists()


def test_successful_start_cannot_be_reused(tmp_path):
    path=tmp_path/'CONTROL_BUILD_STARTED.json';metadata,storage,state=mock_storage(tmp_path)
    m.publish_guarded(metadata,storage,{},path,{'attempt':1})
    with pytest.raises(ValueError):m.publish_guarded(metadata,storage,{},path,{'attempt':2})
    assert state['writes']==1 and json.loads(path.read_text())=={'attempt':1}


def test_control_hash_remains_mandatory(tmp_path):
    path=tmp_path/'control.json';path.write_text('{}')
    with pytest.raises(ValueError,match='changed'):m.pinned({'path':str(path),'sha256':'0'*64})
