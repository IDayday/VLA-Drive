"""Minimal env-before-import fixtures; no torch/CUDA/model loads."""
import ast
import builtins
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import pytest

WRAPPER=Path('/mnt/project/VLA-Drive-st-road-direction-systems-20261018/reports/st_world_policy/20261018/gradient_balance_systems_v1/supervise_gradient_once.py')
spec=importlib.util.spec_from_file_location('review_gradient_env_v2',WRAPPER)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

@pytest.mark.parametrize('value',[None,':16:8','bad'])
def test_missing_or_wrong_env_refuses_before_control_reads(value):
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='');env.pop('CUBLAS_WORKSPACE_CONFIG',None)
    if value is not None:env['CUBLAS_WORKSPACE_CONFIG']=value
    p=subprocess.run([sys.executable,str(WRAPPER),'--execute','--admission','/tmp/review-nonexistent-gradient-admission'],env=env,capture_output=True,text=True,timeout=5)
    assert p.returncode!=0 and 'requires explicit CUBLAS_WORKSPACE_CONFIG=:4096:8' in p.stderr
    assert 'FileNotFoundError' not in p.stderr


def test_child_environment_preserves_parent_numeric_flags(monkeypatch):
    monkeypatch.delitem(sys.modules,'torch',raising=False)
    parent={'CUBLAS_WORKSPACE_CONFIG':':4096:8','OTHER_NUMERIC_FLAG':'unaltered','PYTHONHASHSEED':'42'};before=dict(parent)
    child=m.child_environment(parent,{'gpu_uuid':'GPU-fixture'},77)
    assert parent==before and child['CUBLAS_WORKSPACE_CONFIG']==':4096:8'
    assert child['OTHER_NUMERIC_FLAG']=='unaltered' and child['PYTHONHASHSEED']=='42'
    assert child['ST_GRADIENT_LOCK_FD']=='77'


def test_late_torch_env_rejected(monkeypatch):
    monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG',':4096:8');monkeypatch.setitem(sys.modules,'torch',object())
    with pytest.raises(ValueError,match='before importing torch'):m.validate_runtime_environment()


def test_actual_child_records_env_before_first_torch_import(tmp_path,monkeypatch):
    assert 'torch' not in sys.modules
    monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    monkeypatch.setattr(m,'REGISTRATION',tmp_path/'REGISTRATION.json');monkeypatch.setattr(m,'ATTEMPT',tmp_path);monkeypatch.setattr(m,'RUNTIME',tmp_path)
    events=[];monkeypatch.setattr(m,'validate_admission',lambda *a:events.append('admission'))
    old_import=builtins.__import__
    class Intercepted(Exception):pass
    def tracked(name,*args,**kwargs):
        if name=='torch':
            events.append('torch_import')
            assert os.environ['CUBLAS_WORKSPACE_CONFIG']==':4096:8'
            v=json.loads((tmp_path/'RUNTIME_ENVIRONMENT.json').read_text())
            assert v['runtime_environment']==m.RUNTIME_ENVIRONMENT and v['checked_before_import'] and not v['torch_imported']
            raise Intercepted()
        return old_import(name,*args,**kwargs)
    original_dir=Path.cwd()
    with (tmp_path/'RUN.lock').open('a+b') as lock:
        monkeypatch.setenv('ST_GRADIENT_LOCK_FD',str(lock.fileno()))
        monkeypatch.setattr(builtins,'__import__',tracked)
        try:
            with pytest.raises(Intercepted):m.child_main()
        finally:os.chdir(original_dir)
    assert events==['admission','torch_import'] and 'torch' not in sys.modules


def test_numerical_and_cleanup_definitions_unchanged():
    relative=str(WRAPPER.relative_to(WRAPPER.parents[4]))
    old=subprocess.check_output(['git','show','b74c1ae0:'+relative],cwd=WRAPPER.parents[4],text=True)
    def defs(text):return {x.name:ast.dump(x,include_attributes=False) for x in ast.parse(text).body if isinstance(x,(ast.ClassDef,ast.FunctionDef))}
    a,b=defs(old),defs(WRAPPER.read_text())
    for key in ('ReadAdmission','replace_aliases','process_sample','supervise','ownership_helpers','directory_bytes'):assert a[key]==b[key]
    original=json.loads(Path('/mnt/project/st-world-policy-20261016/audits/consequence/gradient_balance_boundary3584_v1_controls/REGISTRATION.json').read_text())
    new=json.loads(m.REGISTRATION.read_text())
    assert [key for key in original if original[key]!=new[key]]==['output']
