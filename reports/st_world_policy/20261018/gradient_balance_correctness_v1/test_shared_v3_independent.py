"""CPU-only mocked nvidia-smi controls. No real GPU query/compute/signaling."""
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import pytest

WRAPPER=Path('/mnt/project/VLA-Drive-st-world-policy-20261016/reports/st_world_policy/20261018/gradient_balance_remote_v3/supervise_gradient_once.py')
spec=importlib.util.spec_from_file_location('gradient_shared_review',WRAPPER)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
GPU='GPU-fixture';HOST='fixture'


def setup(tmp_path,monkeypatch,memory,missing_review=None,missing_admission=False):
    sha=lambda b:hashlib.sha256(b).hexdigest()
    dump=lambda p,v:(p.write_text(json.dumps(v)),sha(p.read_bytes()))[1]
    helpers=tmp_path/'helpers';helpers.mkdir()
    for f in ('supervise_once.py','prepare_bindings.py'):(helpers/f).write_text('fixture')
    reg=tmp_path/'reg';rh=dump(reg,{'resource':{'host':HOST,'gpu_uuid':GPU}})
    policy=tmp_path/'policy';ph=dump(policy,{'allowed_aliases':[HOST]})
    monkeypatch.setattr(m,'REGISTRATION',reg);monkeypatch.setattr(m,'REG_SHA',rh)
    monkeypatch.setattr(m,'POLICY',policy);monkeypatch.setattr(m,'HELPERS',helpers)
    monkeypatch.setattr(m.socket,'gethostname',lambda:HOST)
    monkeypatch.setattr(m.os,'sched_getaffinity',lambda pid:{0,1})
    monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    base={'status':'APPROVE_GRADIENT_BALANCE_EXECUTION','reviewer':'independent','source_sha':m.SOURCE,'registration_sha256':rh,'wrapper_sha256':sha(WRAPPER.read_bytes()),'runtime_environment':m.RUNTIME_ENVIRONMENT,'gpu_admission':m.GPU_ADMISSION}
    pins={}
    for kind in ('correctness','resources'):
        value={**base,'review_kind':kind,'reviewer':'independent_'+kind}
        if missing_review==kind:value.pop('gpu_admission')
        rp=tmp_path/kind;hh=dump(rp,value);pins[kind]={'path':str(rp),'sha256':hh}
    a={'status':'APPROVED','registration_sha256':rh,'source_sha':m.SOURCE,'wrapper_sha256':base['wrapper_sha256'],'helper_sha256':{f.name:sha(f.read_bytes()) for f in helpers.iterdir()},'runtime_environment':m.RUNTIME_ENVIRONMENT,'gpu_admission':m.GPU_ADMISSION,'host':HOST,'gpu_uuid':GPU,'valid_until_unix':m.time.time()+2000,'dispatch_policy_sha256':ph,'reviews':pins,'cpu_affinity':[0,1]}
    if missing_admission:a.pop('gpu_admission')
    path=tmp_path/'admission';dump(path,a);calls=[]
    def output(argv,**kwargs):
        if argv[0]=='git':return m.SOURCE+'\n'
        assert argv==['nvidia-smi','-i',GPU,'--query-gpu=uuid,memory.free','--format=csv,noheader,nounits']
        calls.append(argv);return memory
    monkeypatch.setattr(m.subprocess,'check_output',output)
    monkeypatch.setattr(m.subprocess,'run',lambda *a,**kw:SimpleNamespace(returncode=0))
    return path,calls


def test_accepts_exact_headroom_without_process_preemption(tmp_path,monkeypatch):
    path,calls=setup(tmp_path,monkeypatch,GPU+', 61440\n')
    r,a=m.validate_admission(path)
    assert a['gpu_admission']=={'mode':'shared_spare_capacity','minimum_free_memory_mib':61440,'protect_all_existing_processes':True}
    assert len(calls)==1  # No process-list query and no signal function exists in this path.


def test_insufficient_memory_refused(tmp_path,monkeypatch):
    path,_=setup(tmp_path,monkeypatch,GPU+', 61439')
    with pytest.raises(ValueError,match='lacks admitted spare'):m.validate_admission(path)


def test_wrong_uuid_refused(tmp_path,monkeypatch):
    path,_=setup(tmp_path,monkeypatch,'GPU-other, 80000')
    with pytest.raises(ValueError,match='lacks admitted spare'):m.validate_admission(path)


@pytest.mark.parametrize('response',['','GPU-fixture, N/A','GPU-fixture, -1','GPU-fixture, 80000, extra'])
def test_bad_memory_reply_refused(tmp_path,monkeypatch,response):
    path,_=setup(tmp_path,monkeypatch,response)
    with pytest.raises(ValueError):m.validate_admission(path)


@pytest.mark.parametrize('kind',['correctness','resources'])
def test_review_without_sharing_refused_before_gpu(tmp_path,monkeypatch,kind):
    path,calls=setup(tmp_path,monkeypatch,GPU+', 80000',missing_review=kind)
    with pytest.raises(ValueError,match='Independent exact'):m.validate_admission(path)
    assert calls==[]


def test_lead_without_sharing_refused_before_gpu(tmp_path,monkeypatch):
    path,calls=setup(tmp_path,monkeypatch,GPU+', 80000',missing_admission=True)
    with pytest.raises(ValueError,match='Lead admission'):m.validate_admission(path)
    assert calls==[]
