"""Bounded review only of DDP process ownership; no old suite or GPU dispatch."""
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
OWNER=Path('/mnt/project/VLA-Drive-st-rollout-native-resume-smoke-v3-20261018')
FILE='resources/st_rollout_resume_smoke_v1.py'
PARENT='01d8dc4733feb98e562eb9a4f66bd4e56c9db114'
EXPECTED=os.environ['REVIEW_EXPECTED_COMMIT']
def clean():
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=OWNER,text=True).strip()!=EXPECTED or subprocess.check_output(['git','status','--porcelain=v1'],cwd=OWNER).strip():
        raise ValueError('Expected exact final clean v3 source')
clean()
old=subprocess.check_output(['git','show',PARENT+':'+FILE],cwd=OWNER,text=True)
new=(OWNER/FILE).read_text();left,right=ast.parse(old),ast.parse(new)
names=['logical_state','engine_snapshot','install_observer','worker','consumed_receipt','verify_load_witness','compare',
    'validate_pause_marker','archive_owned_pause','validate_pause_ack','inspect_native','selected_native','validate_result','run_path']
for name in names:
    get=lambda tree:next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
    assert ast.dump(get(left))==ast.dump(get(right)),'Unexpected scientific/native/pause delta: '+name
assert hashlib.sha256((OWNER/'resources/st_endpoint_supervisor_v1.py').read_bytes()).hexdigest()=='a2e24537b9749258417e390be705d4d5d4e6f775986ca479a1c67c4330759a3e'
command=[sys.executable,'-m','pytest','-q',str(OWNER/'tests/st_world/test_rollout_resume_smoke.py'),str(HERE/'test_ddp_ownership.py'),'-k','ddp_ownership']
env={**os.environ,'PYTHONPATH':str(OWNER),'PYTEST_DISABLE_PLUGIN_AUTOLOAD':'1','CUDA_VISIBLE_DEVICES':'',
    'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}
def tree(pid):
    found=set();pending=[pid]
    while pending:
        current=pending.pop()
        if current in found:continue
        found.add(current)
        try:pending.extend(int(item) for item in Path(f'/proc/{current}/task/{current}/children').read_text().split())
        except (OSError,ValueError):pass
    return found

def rss(pid):
    try:return next(int(row.split()[1])*1024 for row in Path(f'/proc/{pid}/status').read_text().splitlines() if row.startswith('VmRSS:'))
    except (OSError,StopIteration):return 0
libc=ctypes.CDLL(None,use_errno=True);prior_subreaper=ctypes.c_int()
assert libc.prctl(37,ctypes.byref(prior_subreaper),0,0,0)==0
assert libc.prctl(36,1,0,0,0)==0
observed={}
def birth(pid):
    try:return int(Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19])
    except (OSError,IndexError,ValueError):return None
def family():
    members=tree(os.getpid())-{os.getpid()}
    for pid in members:
        stamp=birth(pid)
        if stamp is not None:observed[pid]=stamp
    return members
started=time.monotonic();before=resource.getrusage(resource.RUSAGE_CHILDREN);peak=0;reason=None
with (HERE/'TEST_OUTPUT.txt').open('w') as output:
    child=subprocess.Popen(command,cwd=OWNER,env=env,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
    while child.poll() is None:
        peak=max(peak,sum(rss(pid) for pid in family()))
        if peak>2*2**30:reason='aggregate RSS2GiB'
        if time.monotonic()-started>85:reason='wall85s'
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
# This review driver is a temporary child-subreaper. Reap fixture orphans and
# never leave a new-session CPU process behind if an assertion or timeout failed.
family()
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
assert libc.prctl(36,prior_subreaper.value,0,0,0)==0
clean();after=resource.getrusage(resource.RUSAGE_CHILDREN)
result={'schema':'independent_ddp_ownership_delta_v3_test','source':EXPECTED,'controller_sha256':hashlib.sha256(new.encode()).hexdigest(),
 'command':command,'exit_code':child.returncode,'stop_reason':reason,'wall_seconds':time.monotonic()-started,
 'waited_child_CPU_seconds':after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime,
 'sampled_50ms_owned_tree_peak_RSS_bytes':peak,'scientific_native_pause_functions_AST_unchanged':names,
 'frozen_endpoint_helper_unchanged':True,'old51_or_cold_metadata_tests_rerun':False,
 'GPU_calls':0,'official_score_calls':0,'trainer_calls':0,'model_native_image_GT_payload_loads':0,
 'NVML':'Explicit synthetic output supplied only to original strict monitor; actual Linux process identities used'}
(HERE/'TEST_RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result));sys.exit(child.returncode if not reason else 2)
