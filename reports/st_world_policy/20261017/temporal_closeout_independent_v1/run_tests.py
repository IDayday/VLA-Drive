"""Independent finite test runner. No real builder, ledger, GPU or SSH."""
import os,time,json,resource,signal,sys
from pathlib import Path
START=time.monotonic()
def io():
    with open('/proc/self/io') as f:return {k:int(v) for k,v in (s.split(':') for s in f)}
BASE=io(); OUT=Path(__file__).resolve().parent
os.environ.update(CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTEST_DISABLE_PLUGIN_AUTOLOAD='1')
os.sched_setaffinity(0,{8,9})
def guard(*_):
    if time.monotonic()-START>175 or io()['rchar']-BASE['rchar']>256<<20 or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>2<<30:
        raise RuntimeError('Independent finite test envelope exhausted')
signal.signal(signal.SIGALRM,guard);signal.setitimer(signal.ITIMER_REAL,1,1)
R=Path('/mnt/project/VLA-Drive-st-temporal-combination-20261017')
sys.path.insert(0,str(R));os.chdir(R)
import pytest
ret=pytest.main(['-q','-p','no:cacheprovider','tests/st_world/test_temporal_composition.py'])
# Run binding tests in the same small process; no subprocess/actual monitor execution.
import importlib.util,unittest
p=Path('/mnt/project/VLA-Drive-st-teacher-channel-control-20261017/reports/st_world_policy/20261017/temporal_history32_preparation_v1/test_history32_binding.py')
spec=importlib.util.spec_from_file_location('independent_bound_tests',p);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(mod))
# Separate additional boundary cases, independent of author's test list.
from copy import deepcopy
m=mod.m
a,cp,now=mod.admission()
extra={}
for key,value in [('reviewer','/root'),('limits',{**a['limits'],'cpu_affinity':[2,2]}),('latest_start_unix',now-1),('valid_until_unix',now+939)]:
    b=deepcopy(a);b[key]=value
    try:m.validate_binding(b,cp,now=now)
    except ValueError:extra[key]='REFUSED'
    else:raise AssertionError('Accepted '+key)
# Runtime source pins are all bounded small files; check actual reviewed identity.
import hashlib
actual={}
for rel,sha in {**m.PINS,**m.BUILDER_PINS}.items():
    blob=m.pinned({'path':str(m.RUNTIME/rel),'sha256':sha},decode=False);actual[rel]=dict(bytes=len(blob),sha256=hashlib.sha256(blob).hexdigest())
guard();signal.setitimer(signal.ITIMER_REAL,0)
metrics=dict(pytest_exit=ret,binding_tests=r.testsRun,binding_failures=len(r.failures),binding_errors=len(r.errors),extra_refusals=extra,actual_source_pins=actual,wall_seconds=time.monotonic()-START,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,logical_read_bytes=io()['rchar']-BASE['rchar'],cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],real_builder_executed=False,ledger_accessed=False,ssh=False)
(OUT/'TEST_METRICS.json').write_text(json.dumps(metrics,indent=2)+'\n')
print(json.dumps(metrics,sort_keys=True));raise SystemExit(int(ret or not r.wasSuccessful()))
