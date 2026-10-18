"""Bounded independent CPU-only observer review runner; no experiment dispatch."""
import json
import hashlib
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent
SOURCE=Path('/mnt/project/VLA-Drive-st-rollout-native-resume-smoke-20261018')
expected=os.environ['REVIEW_EXPECTED_COMMIT']
def check_source():
    actual=subprocess.check_output(['git','rev-parse','HEAD'],cwd=SOURCE,text=True).strip()
    if actual!=expected or subprocess.check_output(['git','status','--porcelain=v1'],cwd=SOURCE).strip():
        raise RuntimeError('Final implementation commit must be exact and clean')
check_source()
controller=SOURCE/'resources/st_rollout_resume_smoke_v1.py'
controller_sha=hashlib.sha256(controller.read_bytes()).hexdigest()
command=[sys.executable,'-m','pytest','-q',str(SOURCE/'tests/st_world/test_rollout_resume_smoke.py'),
    str(ROOT/'test_observer_contract.py')]
env={**os.environ,'PYTHONPATH':str(SOURCE),'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1',
    'OPENBLAS_NUM_THREADS':'1','CUDA_VISIBLE_DEVICES':'','PYTEST_DISABLE_PLUGIN_AUTOLOAD':'1'}
def owned_tree(pid):
    found=set();pending=[pid]
    while pending:
        child=pending.pop()
        if child in found:continue
        found.add(child)
        try:pending.extend(int(value) for value in Path(f'/proc/{child}/task/{child}/children').read_text().split())
        except (OSError,ValueError):pass
    return found

def rss(pid):
    try:
        return next(int(row.split()[1])*1024 for row in Path(f'/proc/{pid}/status').read_text().splitlines() if row.startswith('VmRSS:'))
    except (OSError,StopIteration):return 0

started=time.monotonic();before=resource.getrusage(resource.RUSAGE_CHILDREN);peak=0;reason=None
with (ROOT/'NARROW_TEST.log').open('w') as output:
    child=subprocess.Popen(command,cwd=SOURCE,env=env,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
    while child.poll() is None:
        peak=max(peak,sum(rss(pid) for pid in owned_tree(child.pid)))
        if peak>2*2**30:reason='aggregate RSS cap2GiB'
        if time.monotonic()-started>85:reason='wall cap85s'
        if reason:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(8)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
            break
        time.sleep(.05)
    child.wait()
check_source()
after=resource.getrusage(resource.RUSAGE_CHILDREN)
result={'schema':'independent_native_observer_narrow_test_v1','command':command,'controller_commit':expected,'controller_sha256':controller_sha,
    'exit_code':child.returncode,'wall_seconds':time.monotonic()-started,
    'child_user_CPU_seconds':after.ru_utime-before.ru_utime,'child_system_CPU_seconds':after.ru_stime-before.ru_stime,
    'peak_owned_tree_RSS_bytes_sampled_50ms':peak,'stop_reason':reason,
    'GPU_calls':0,'official_score_calls':0,'real_native_or_image_payload_loads':0,
    'test_data':'explicitly synthetic tiny CPU tensors; Gloo ordinary-exception vote only',
    'not_real_native_resume_evidence':True}
(ROOT/'NARROW_TEST_COST.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
sys.exit(child.returncode if not reason else 2)
