"""Only installed DeepSpeed fragment schema and changed bounded accounting."""
import ast
import ctypes
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
OWNER=Path('/mnt/project/VLA-Drive-st-rollout-native-resume-smoke-v4-20261018')
FILE='resources/st_rollout_resume_smoke_v1.py';PARENT='e631600f13f3a54f9f7bc43f9989a089314b7f9e'
EXPECTED=os.environ['REVIEW_EXPECTED_COMMIT']
def clean():
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=OWNER,text=True).strip()!=EXPECTED or subprocess.check_output(['git','status','--porcelain=v1'],cwd=OWNER).strip():
        raise ValueError('Expected exact final clean v4 source')
clean();old=subprocess.check_output(['git','show',PARENT+':'+FILE],cwd=OWNER,text=True);new=(OWNER/FILE).read_text()
left,right=ast.parse(old),ast.parse(new)
def function(tree,name):return next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name==name)
names=['engine_snapshot','install_observer','worker','consumed_receipt','verify_load_witness','compare',
    'validate_pause_marker','archive_owned_pause','validate_pause_ack','inspect_native','selected_native','validate_result',
    'owned_processes','stop_smoke_owned','monitor_smoke_gpu','group_ready','route_checks']
for name in names:assert ast.dump(function(left,name))==ast.dump(function(right,name)),'Unrelated delta: '+name
serializer=function(right,'logical_state')
indices=[i for i,node in enumerate(serializer.body) if isinstance(node,ast.If) and any(isinstance(child,ast.Constant) and child.value=='deepspeed.utils.tensor_fragment' for child in ast.walk(node.test))]
assert len(indices)==1;serializer.body.pop(indices[0])
assert ast.dump(serializer)==ast.dump(function(left,'logical_state')),'Unexpected tensor/loss/scalar serialization change'
assert hashlib.sha256((OWNER/'resources/st_endpoint_supervisor_v1.py').read_bytes()).hexdigest()=='a2e24537b9749258417e390be705d4d5d4e6f775986ca479a1c67c4330759a3e'
command=[sys.executable,'-m','pytest','-q',str(OWNER/'tests/st_world/test_rollout_resume_smoke.py'),str(HERE/'test_ds_fragment_schema.py'),'-k','ds_fragment or resource_and_scientific_budgets']
env={**os.environ,'PYTHONPATH':str(OWNER),'PYTEST_DISABLE_PLUGIN_AUTOLOAD':'1','CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','TORCHINDUCTOR_COMPILE_THREADS':'1'}
def family():
    found=set();pending=[os.getpid()]
    while pending:
        pid=pending.pop()
        if pid in found:continue
        found.add(pid)
        try:pending.extend(int(item) for item in Path(f'/proc/{pid}/task/{pid}/children').read_text().split())
        except (OSError,ValueError):pass
    return found-{os.getpid()}
def rss(pid):
    try:return next(int(row.split()[1])*1024 for row in Path(f'/proc/{pid}/status').read_text().splitlines() if row.startswith('VmRSS:'))
    except (OSError,StopIteration):return 0
def birth(pid):
    try:return int(Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19])
    except (OSError,IndexError,ValueError):return None
libc=ctypes.CDLL(None,use_errno=True);old_subreaper=ctypes.c_int()
assert libc.prctl(37,ctypes.byref(old_subreaper),0,0,0)==0
assert libc.prctl(36,1,0,0,0)==0
before=resource.getrusage(resource.RUSAGE_CHILDREN);started=time.monotonic();peak=0;reason=None;observed={}
with (HERE/'TEST_OUTPUT.txt').open('w') as output:
    child=subprocess.Popen(command,cwd=OWNER,env=env,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
    while child.poll() is None:
        members=family()
        for pid in members:
            stamp=birth(pid)
            if stamp is not None:observed[pid]=stamp
        peak=max(peak,sum(rss(pid) for pid in members))
        if peak>2*2**30:reason='aggregate RSS cap2GiB'
        if time.monotonic()-started>60:reason='wall cap60s'
        if reason:
            for pid,stamp in list(observed.items()):
                if birth(pid)==stamp:
                    try:os.kill(pid,signal.SIGTERM)
                    except ProcessLookupError:pass
            try:child.wait(8)
            except subprocess.TimeoutExpired:
                for pid,stamp in list(observed.items()):
                    if birth(pid)==stamp:
                        try:os.kill(pid,signal.SIGKILL)
                        except ProcessLookupError:pass
                child.wait()
            break
        time.sleep(.05)
    child.wait()
# Reap/clean any controlled test-import workers, including failure fallback.
for pid in family():
    stamp=birth(pid)
    if stamp is not None:observed[pid]=stamp
for pid,stamp in list(observed.items()):
    if birth(pid)==stamp:
        try:os.kill(pid,signal.SIGKILL)
        except ProcessLookupError:pass
until=time.monotonic()+3
while time.monotonic()<until:
    try:
        waited,_=os.waitpid(-1,os.WNOHANG)
        if not waited:time.sleep(.02)
    except ChildProcessError:break
assert libc.prctl(36,old_subreaper.value,0,0,0)==0
clean();after=resource.getrusage(resource.RUSAGE_CHILDREN)
report={'schema':'independent_real_ds_fragment_v4_test','source':EXPECTED,'controller_sha256':hashlib.sha256(new.encode()).hexdigest(),
 'command':command,'exit_code':child.returncode,'stop_reason':reason,'sampled_50ms_aggregate_RSS_bytes':peak,'test_only_environment':{'TORCHINDUCTOR_COMPILE_THREADS':'1'},'wall_seconds':time.monotonic()-started,
 'waited_child_CPU_seconds':after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime,
 'maximum_child_RSS_bytes':after.ru_maxrss*1024,'unchanged_function_AST':names,
 'serializer_only_adds_explicit_real_fragment_schema':True,'endpoint_helper_unchanged':True,
 'old51_DDP_and_cold_tests_rerun':False,'GPU_calls':0,'official_score_calls':0,'model_or_native_tensor_loads':0,
 'real_class':'Installed DeepSpeed fragment_address and real public ZeroOptimizer.state_dict on4-element CPU carrier',
 'pickle_boundary':'Tests unpickle only their own tiny trusted dataclass fixture, never native data.pkl'}
(HERE/'TEST_RESULT.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report));sys.exit(child.returncode if reason is None else 2)
