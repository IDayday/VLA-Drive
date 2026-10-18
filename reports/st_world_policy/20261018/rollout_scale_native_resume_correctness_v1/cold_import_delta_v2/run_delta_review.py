"""Only root namespace and cold metadata-import delta; no previous suite rerun."""
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
OWNER=Path('/mnt/project/VLA-Drive-st-rollout-native-resume-smoke-v2-20261018')
FILE='resources/st_rollout_resume_smoke_v1.py'
PARENT='19a5a244cca2a07600339ba9be7c3a150ef61285'
EXPECTED=os.environ['REVIEW_EXPECTED_COMMIT']

def clean():
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=OWNER,text=True).strip()!=EXPECTED:
        raise ValueError('Unexpected final delta source')
    if subprocess.check_output(['git','status','--porcelain=v1'],cwd=OWNER).strip():
        raise ValueError('Delta source must be frozen and clean')
clean()
old=subprocess.check_output(['git','show',PARENT+':'+FILE],cwd=OWNER,text=True)
new=(OWNER/FILE).read_text()
left,right=ast.parse(old),ast.parse(new)
def one(tree,name):return next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
assert len(one(right,'route_checks').body)==len(one(left,'route_checks').body)+1
added=one(right,'route_checks').body[0]
assert ast.dump(added)==ast.dump(ast.parse('if str(RUNTIME) not in sys.path:sys.path.insert(0,str(RUNTIME))').body[0])
one(right,'route_checks').body.pop(0)
for tree,expected_suffix in [(left,'v1'),(right,'v2')]:
    node=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='ROOT' for t in n.targets))
    assert isinstance(node.value,ast.BinOp) and node.value.right.value=='smoke/rollout_scale_native_resume_'+expected_suffix
    node.value.right.value='normalized_namespace'
assert ast.dump(left)==ast.dump(right),'Unexpected unrelated controller AST delta'
started=time.perf_counter();before=resource.getrusage(resource.RUSAGE_CHILDREN);results=[]
with tempfile.TemporaryDirectory(prefix='cold_metadata_review_') as raw:
    temporary=Path(raw);old_path=temporary/'old_controller.py';old_path.write_text(old)
    for path,expected in [(old_path,2),(OWNER/FILE,0)]:
        command=[sys.executable,'-I',str(HERE/'cold_metadata_probe.py'),str(path),
            '/mnt/project/st-world-policy-20261016/smoke/rollout_scale_native_resume_v1_admission/ENVIRONMENT.json']
        env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
        result=subprocess.run(command,cwd=temporary,env=env,capture_output=True,text=True,timeout=25)
        results.append({'source':'old19a5' if expected==2 else EXPECTED,'expected_exit':expected,
            'actual_exit':result.returncode,'stdout':result.stdout,'stderr':result.stderr})
        if result.returncode!=expected:raise AssertionError(results[-1])
        payload=json.loads(result.stdout.strip())
        assert payload['status']==('REPRODUCED_OLD_COLD_IMPORT_FAILURE' if expected==2 else 'PASS_COLD_METADATA_IMPORT')
clean();after=resource.getrusage(resource.RUSAGE_CHILDREN)
output={'schema':'native_resume_cold_import_delta_test_v2','parent':PARENT,'source':EXPECTED,
    'controller_sha256':hashlib.sha256(new.encode()).hexdigest(),'AST_only_changes':['ROOTv1->v2','route_checks imports runtime path before helpers'],
    'tests':results,'wall_seconds':time.perf_counter()-started,'waited_child_CPU_seconds':after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime,
    'maximum_child_RSS_bytes':after.ru_maxrss*1024,'sequential_children_no_concurrent_process_groups':True,
    'previous51_suite_rerun':False,'GPU_calls':0,'official_calls':0,'trainer_calls':0,'model_native_image_payload_loads':0}
(HERE/'TEST_RESULT.json').write_text(json.dumps(output,indent=2)+'\n')
print(json.dumps(output))
