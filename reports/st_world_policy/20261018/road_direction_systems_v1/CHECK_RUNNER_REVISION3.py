"""Independent resource delta review: small sources/receipts + CPU fixtures."""
import ast
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import time

RUNTIME=Path('/mnt/project/VLA-Drive-st-road-time-parity-runtime-20261018')
BEFORE='83679937fe9702be39f0184154fdf9ebdc4eb98a'
REG=Path('/mnt/project/st-world-policy-20261016/audits/consequence/road_direction_fixed32_v3_controls/REGISTRATION.json')
OUT=Path(__file__).resolve().parent
PYTHON='/root/miniconda3/envs/ddp/bin/python'

def main():
    raw=REG.read_bytes();reg=json.loads(raw)
    path='tools/st_world_policy/run_road_direction.py'
    old=ast.parse(subprocess.check_output(['git','show',BEFORE+':'+path],cwd=RUNTIME,text=True))
    new=ast.parse((RUNTIME/path).read_text())
    def definitions(tree):return {n.name:ast.dump(n,include_attributes=False) for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    a,b=definitions(old),definitions(new)
    names=('Budget','bounded_unpack','owned_output_bytes','check_admission','load_inputs',
           'VerifiedContextEngine','supervise','summarize','_child_entry')
    same={n:a[n]==b[n] for n in names}
    assert all(same.values()),same
    sources={p:{'bytes':(RUNTIME/p).stat().st_size,'sha256':hashlib.sha256((RUNTIME/p).read_bytes()).hexdigest()} for p in reg['source_hashes']}
    assert all(sources[p]['sha256']==reg['source_hashes'][p] for p in sources)
    code='''
import os,sys
def audit(event,args):
 if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
  p=os.fsdecode(args[0])
  if p.startswith('/mnt/project/VLA-Drive-st-road-time-parity-runtime-20261018/') and p.endswith(('.py','.pyc')):return
  if p.startswith('/mnt/project/') and any(k in p for k in ('/metric_cache','/rollouts/','/official_feedback/','/checkpoints/')):
   raise RuntimeError('Real asset access forbidden during resource fixtures')
sys.addaudithook(audit)
import pytest
raise SystemExit(pytest.main(['-q','-p','no:cacheprovider','tests/st_world/test_run_road_direction.py']))
'''
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
        OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    start=time.monotonic();before=resource.getrusage(resource.RUSAGE_CHILDREN)
    p=subprocess.run([PYTHON,'-c',code],cwd=RUNTIME,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=60)
    usage=resource.getrusage(resource.RUSAGE_CHILDREN)
    record={'registration_sha256':hashlib.sha256(raw).hexdigest(),'source_commit':reg['source_commit'],
        'unchanged_resource_and_cleanup_AST':same,'source_verification':sources,
        'fixture_exit_code':p.returncode,'fixture_stdout_stderr':p.stdout,
        'fixture_wall_seconds':time.monotonic()-start,
        'fixture_cpu_seconds':usage.ru_utime+usage.ru_stime-before.ru_utime-before.ru_stime,
        'fixture_peak_child_rss_bytes':usage.ru_maxrss*1024,
        'real_asset_reads':0,'official_calls':0,'GPU_used':False,'reviewer_dispatched':False}
    with (OUT/'RUNNER_TESTS_REVISION3_CORRECTED.json').open('x') as f:json.dump(record,f,indent=2)
    print(json.dumps(record));return p.returncode

if __name__=='__main__':raise SystemExit(main())
