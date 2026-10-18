"""Independent bounded actual native-source-contract regression, no T3 builder."""
import os,time,json,resource,signal,sys
from pathlib import Path
START=time.monotonic()
def rchar():
    with open('/proc/self/io') as f:return int(next(line.split(':')[1] for line in f if line.startswith('rchar:')))
BEFORE=rchar()
OUT=Path(__file__).resolve().parent/'native_migration_patch'
OUT.mkdir(exist_ok=True)
os.environ.update(CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
os.sched_setaffinity(0,{2,3})
def guard(*_):
    if time.monotonic()-START>90 or rchar()-BEFORE>256<<20 or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>2<<30:
        raise RuntimeError('Independent native-contract test budget exhausted')
signal.signal(signal.SIGALRM,guard);signal.setitimer(signal.ITIMER_REAL,.5,.5)
import importlib.util,unittest,hashlib,subprocess
from copy import deepcopy
from unittest.mock import patch
R=Path('/mnt/project/VLA-Drive-st-temporal-packed-migration-fix-20261018')
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip()=='cf516259c8efc24503727f14b9e330ed32231293'
assert not subprocess.check_output(['git','status','--porcelain'],cwd=R).strip()
test=R/'reports/st_world_policy/20261018/temporal_packed_history_migration_v1/test_migration.py'
spec=importlib.util.spec_from_file_location('independently_rerun_native_contract',test)
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
# Redirect all fixture outputs into review ownership; source/runtime remain exact.
mod.HERE=OUT
result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(mod))
assert result.wasSuccessful()
q=mod.Migration.q;m=mod.Migration.migration;plan=mod.Migration.plan
extra=[]
for field,value in [('formal_approved',True),('review',{'path':'/tmp/unreviewed','sha256':'0'*64}),('relative','starVLA/dataloader/st_world_targets.py')]:
    bad=deepcopy(m);bad['mathematics_compatibility'][field]=value
    bad['identity']=q.identity_hash({k:v for k,v in bad.items() if k!='identity'})
    try:q.validate_control_migration(plan,bad)
    except ValueError as e:
        assert 'exact reviewed' in str(e);extra.append('REFUSED forged compatibility '+field)
    else:raise AssertionError('Forged compatibility accepted')
# The original record must remain unchanged after every adversarial copy.
assert m==json.loads((OUT/'SOURCE_MIGRATION_PREFLIGHT.json').read_text())
assert q.validate_control_migration(plan,m)['identity']=='90a908a48eba2e3d94e1d07700178dd08c61cb4aa0943459bbf572bd1b17ff8b'
for pin in (q.PACKED_REVIEW,q.PACKED_SOURCE_BINDING):
    assert hashlib.sha256(Path(pin['path']).read_bytes()).hexdigest()==pin['sha256']
guard();signal.setitimer(signal.ITIMER_REAL,0)
metrics={'status':'PASS','reviewer':'/root/temporal_narrow_review',
 'reviewed_commit':'cf516259c8efc24503727f14b9e330ed32231293',
 'source_sha256':hashlib.sha256((R/'tools/st_world_policy/qualify_indexed_provider.py').read_bytes()).hexdigest(),
 'actual_native_source_migration_cli':mod.CLI_COMPLETED,'original_actual_failure_reproduced':True,
 'focused_tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'extra_refusals':extra,
 'self_logical_read_bytes':rchar()-BEFORE,'wall_seconds':time.monotonic()-START,
 'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
 'payload_or_DB_open_attempts':mod.BLOCKED,'actual_T3_preparation':False,'GPU_access':False,'SSH':False,'ledger_access':False,
 'limits':'Self rchar includes real imports/source checks; short git child source reads not claimed separately measured aggregate. No model/teacher/tensor/DB/image opened.',
 'scope':'Exact legacy-to-packed history storage migration only. Other14 MATH sources byte equal. Strict archived/current pins and original receipt validation preserved. No formal or diagnostic GPU training approval.'}
(OUT/'REVIEW.json').write_text(json.dumps(metrics,indent=2,sort_keys=True)+'\n')
print(json.dumps(metrics,sort_keys=True))
