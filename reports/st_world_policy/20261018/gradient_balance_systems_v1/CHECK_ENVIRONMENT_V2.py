"""Tiny subprocess environment checks, no PyTorch/model/GPU initialization."""
import ast
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import tempfile
import time

HERE=Path(__file__).resolve().parent
WRAPPER=HERE/'supervise_gradient_once.py'
ROOT=HERE.parents[3]
PYTHON='/root/miniconda3/envs/ddp/bin/python'

def main():
    started=time.monotonic();rows=[];usage_before=resource.getrusage(resource.RUSAGE_CHILDREN)
    for name,value in [('missing',None),('wrong_workspace',':16:8'),('wrong_text','invalid')]:
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='');env.pop('CUBLAS_WORKSPACE_CONFIG',None)
        if value is not None:env['CUBLAS_WORKSPACE_CONFIG']=value
        p=subprocess.run([PYTHON,str(WRAPPER),'--execute','--admission','/tmp/nonexistent-gradient-fixture-admission'],
            env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=10)
        assert p.returncode!=0 and 'requires explicit CUBLAS_WORKSPACE_CONFIG=:4096:8' in p.stdout,p.stdout
        assert 'FileNotFoundError' not in p.stdout,p.stdout
        rows.append({'case':name,'exit_code':p.returncode,'rejected_before_registration_or_admission_read':True})
    loader='''import importlib.util,os,sys,json
from pathlib import Path
spec=importlib.util.spec_from_file_location('w',sys.argv[1]);w=importlib.util.module_from_spec(spec);spec.loader.exec_module(w)
'''
    with tempfile.TemporaryDirectory(prefix='gradient-env-v2-') as folder:
        folder=Path(folder);lock=folder/'RUN.lock'
        with lock.open('a+b') as held:
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='',CUBLAS_WORKSPACE_CONFIG=':4096:8',ST_GRADIENT_LOCK_FD=str(held.fileno()))
            code=loader+'''
assert 'torch' not in sys.modules
original=dict(os.environ); child=w.child_environment(original,{'gpu_uuid':'GPU-fixture'},int(os.environ['ST_GRADIENT_LOCK_FD']))
assert child['CUBLAS_WORKSPACE_CONFIG']==':4096:8' and original==dict(os.environ)
w.REGISTRATION=Path(sys.argv[2])/'REGISTRATION.json';w.ATTEMPT=Path(sys.argv[2]);w.RUNTIME=Path(sys.argv[2])
w.validate_admission=lambda path:None
import builtins
prior=builtins.__import__
class VerifiedBeforeTorch(Exception):pass
def traced(name,*a,**kw):
 if name=='torch':
  assert os.environ['CUBLAS_WORKSPACE_CONFIG']==':4096:8' and 'torch' not in sys.modules
  receipt=json.loads((w.ATTEMPT/'RUNTIME_ENVIRONMENT.json').read_text())
  assert receipt['checked_before_import'] and receipt['torch_imported'] is False
  raise VerifiedBeforeTorch()
 return prior(name,*a,**kw)
builtins.__import__=traced
try:w.child_main()
except VerifiedBeforeTorch:print('PASS_CHILD_ENV_CONFIRMED_BEFORE_TORCH_IMPORT')
else:raise AssertionError('Expected import interception')
'''
            p=subprocess.run([PYTHON,'-c',code,str(WRAPPER),str(folder)],env=env,pass_fds=(held.fileno(),),
                stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=10)
            assert p.returncode==0 and 'PASS_CHILD_ENV_CONFIRMED_BEFORE_TORCH_IMPORT' in p.stdout,p.stdout
            rows.append({'case':'actual_child_path_before_torch_import','exit_code':0,'stdout':p.stdout,'torch_import_intercepted':True})
    code=loader+'''
sys.modules['torch']=object()
try:w.validate_runtime_environment()
except ValueError as e:assert 'before importing torch' in str(e);print('PASS_LATE_ENVIRONMENT_REJECTED')
else:raise AssertionError('late runtime change allowed')
'''
    p=subprocess.run([PYTHON,'-c',code,str(WRAPPER)],env=dict(os.environ,CUDA_VISIBLE_DEVICES='',CUBLAS_WORKSPACE_CONFIG=':4096:8'),
        stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=10)
    assert p.returncode==0,p.stdout
    rows.append({'case':'late_after_torch_rejected','exit_code':0,'stdout':p.stdout})
    path=str(WRAPPER.relative_to(ROOT))
    before=ast.parse(subprocess.check_output(['git','show','b74c1ae0:'+path],cwd=ROOT,text=True))
    after=ast.parse(WRAPPER.read_text())
    def defs(tree):return {n.name:ast.dump(n,include_attributes=False) for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    a,b=defs(before),defs(after)
    unchanged={n:a[n]==b[n] for n in ('ReadAdmission','process_sample','supervise','ownership_helpers','directory_bytes')}
    assert all(unchanged.values()),unchanged
    v1=Path('/mnt/project/st-world-policy-20261016/audits/consequence/gradient_balance_boundary3584_v1_controls/REGISTRATION.json')
    v2=v1.with_name('REGISTRATION.json').parent.parent/'gradient_balance_boundary3584_v2_controls/REGISTRATION.json'
    r1=json.loads(v1.read_text());r2=json.loads(v2.read_text());changes=[k for k in r1 if r1[k]!=r2[k]]
    assert changes==['output'] and r2['output'].endswith('gradient_balance_boundary3584_v2'),changes
    usage=resource.getrusage(resource.RUSAGE_CHILDREN)
    result={'status':'AUTHOR_ENVIRONMENT_FIXTURES_PASS_NOT_APPROVAL','cases':rows,
        'unchanged_AST':unchanged,'registration_changes':changes,
        'runtime_environment':{'CUBLAS_WORKSPACE_CONFIG':':4096:8'},
        'wrapper_sha256':hashlib.sha256(WRAPPER.read_bytes()).hexdigest(),
        'registration_sha256':hashlib.sha256(v2.read_bytes()).hexdigest(),
        'wall_seconds':time.monotonic()-started,
        'CPU_seconds':usage.ru_utime+usage.ru_stime-usage_before.ru_utime-usage_before.ru_stime,
        'peak_child_RSS_bytes':usage.ru_maxrss*1024,'GPU_used':False,'torch_imported_in_fixture':False,
        'real_model_reads':0,'new_scores':0,'optimizer_updates':0,'formal_training_approved':False}
    with (HERE/'ENVIRONMENT_FIX_TESTS_V2.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result))

if __name__=='__main__':main()
