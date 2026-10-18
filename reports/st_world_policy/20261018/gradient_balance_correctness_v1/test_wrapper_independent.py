"""Synthetic CPU controls only. Never call GPU or real diagnostic."""
import importlib.util
import json
import random
import sys
import types
from pathlib import Path
import pytest

SOURCE=Path('/mnt/project/VLA-Drive-st-road-direction-systems-20261018/reports/st_world_policy/20261018/gradient_balance_systems_v1/supervise_gradient_once.py')
spec=importlib.util.spec_from_file_location('reviewed_gradient_wrapper',SOURCE)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def test_wrap_preserves_result_args_rng_and_preadmission(tmp_path):
    f=tmp_path/'x';f.write_bytes(b'abc');counts=tmp_path/'counts.json'
    before=random.getstate()
    try:
        calls=[]
        def original(path, positional, **kwargs):
            assert json.loads(counts.read_text())['torch_load_nominal_bytes']==3
            calls.append((path,positional,kwargs))
            return random.random(),kwargs['object']
        marker=object(); random.seed(813); expected=random.random();after=random.getstate()
        random.seed(813)
        admission=m.ReadAdmission(counts,limit=3)
        actual=m.ReadAdmission.wrap(admission,original,'torch_load_nominal_bytes')(f,4,mmap=True,object=marker)
        assert actual==(expected,marker) and random.getstate()==after
        assert calls==[(f,4,{'mmap':True,'object':marker})]
        with pytest.raises(RuntimeError,match='admission exhausted'):
            admission.wrap(original,'torch_load_nominal_bytes')(f,4,mmap=True,object=marker)
        assert len(calls)==1 and random.getstate()==after
    finally:random.setstate(before)

def test_original_exception_not_substituted_or_retried(tmp_path):
    f=tmp_path/'x';f.write_bytes(b'abc');ad=m.ReadAdmission(tmp_path/'counter',limit=6)
    error=ValueError('original loader failure');calls=[]
    def original(path):calls.append(path);raise error
    with pytest.raises(ValueError) as got:ad.wrap(original,'hash_nominal_bytes')(f)
    assert got.value is error and calls==[f] and ad.total()==3

def test_from_import_alias_patch_keeps_original_semantics(monkeypatch):
    module=types.ModuleType('independent_fixture_gradient_alias');marker=object()
    def source(a,**kwargs):return marker,a,kwargs
    def wrapper(*args,**kwargs):return source(*args,**kwargs)
    module.alias=source;module.unrelated=lambda:11
    monkeypatch.setitem(sys.modules,module.__name__,module)
    m.replace_aliases(source,wrapper)
    assert module.alias is wrapper
    assert module.alias(7,scale=2)==(marker,7,{'scale':2}) and module.unrelated()==11

@pytest.mark.parametrize('bad',['source_sha','wrapper_sha256','reviewer'])
def test_bad_correctness_pin_refused_before_gpu_query(tmp_path,monkeypatch,bad):
    reg=tmp_path/'registration';policy=tmp_path/'policy';adm=tmp_path/'admission'
    helpers=tmp_path/'helpers';helpers.mkdir()
    for name in ('supervise_once.py','prepare_bindings.py'):(helpers/name).write_text('fixture')
    digest=lambda p:m.hashlib.sha256(p.read_bytes()).hexdigest()
    host='fixture-host';gpu='GPU-fixture'
    reg.write_text(json.dumps({'resource':{'host':host,'gpu_uuid':gpu}}));policy.write_text(json.dumps({'allowed_aliases':[host]}))
    monkeypatch.setattr(m,'REGISTRATION',reg);monkeypatch.setattr(m,'REG_SHA',digest(reg))
    monkeypatch.setattr(m,'POLICY',policy);monkeypatch.setattr(m,'HELPERS',helpers)
    monkeypatch.setattr(m.socket,'gethostname',lambda:host)
    review={'status':'APPROVE_GRADIENT_BALANCE_EXECUTION','review_kind':'correctness','reviewer':'independent','source_sha':m.SOURCE,'registration_sha256':m.REG_SHA,'wrapper_sha256':digest(SOURCE)}
    review[bad]='wrong' if bad!='reviewer' else '/root/road_proxy_systems_review'
    rp=tmp_path/'review';rp.write_text(json.dumps(review))
    a={'status':'APPROVED','registration_sha256':m.REG_SHA,'source_sha':m.SOURCE,'wrapper_sha256':digest(SOURCE),'helper_sha256':{p.name:digest(p) for p in helpers.iterdir()},'host':host,'gpu_uuid':gpu,'valid_until_unix':m.time.time()+2000,'dispatch_policy_sha256':digest(policy),'reviews':{'correctness':{'path':str(rp),'sha256':digest(rp)}}}
    adm.write_text(json.dumps(a))
    def forbidden(*args,**kwargs):raise AssertionError('No git/GPU command allowed before bad review refusal')
    monkeypatch.setattr(m.subprocess,'check_output',forbidden)
    with pytest.raises(ValueError,match='Independent exact correctness review required'):m.validate_admission(adm)

def test_metadata_cli_is_nonexecuting(tmp_path):
    import subprocess
    result=subprocess.run([sys.executable,str(SOURCE)],capture_output=True,text=True,timeout=5)
    assert result.returncode==0
    value=json.loads(result.stdout)
    assert value['status']=='NOT_RUN' and value['independent_reviews_required']
