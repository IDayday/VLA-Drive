import os,json,time,resource,hashlib,importlib.util
from pathlib import Path
from types import SimpleNamespace
from contextlib import nullcontext
start=time.monotonic();assert sorted(os.sched_getaffinity(0))==[6,7]
resource.setrlimit(resource.RLIMIT_AS,(4<<30,4<<30))
import pytest
code=pytest.main(['-q','-p','no:cacheprovider','reports/st_world_policy/20261016/evaluation/FULL_PAYLOAD_C4_1024_PREPARATION_v1/test_c4_once.py'])
p=Path('reports/st_world_policy/20261016/evaluation/FULL_PAYLOAD_C4_1024_PREPARATION_v1/run_c4_once.py')
sha=hashlib.sha256(p.read_bytes()).hexdigest();assert sha=='e7ead9078ac5f5fe2db159bd4c61c2327b42f708b4f07df678fdf0da3426b164'
spec=importlib.util.spec_from_file_location('cpu_delta',p);r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
m=r.original();core,_,_=m.dependencies();captured=[]
class BeforeLaunch(Exception):pass
def reject_launch(argv,**kwargs):captured.append(argv);raise BeforeLaunch('Intercept before process creation')
original_subprocess=core.subprocess
core.subprocess=SimpleNamespace(Popen=reject_launch,PIPE=original_subprocess.PIPE,DEVNULL=original_subprocess.DEVNULL)
try:
    core.monitor(['/fixture/no-execution'],runtime='/fixture',env={},fd=0,root=Path('/fixture'),s=SimpleNamespace(lifetime_scope=lambda:nullcontext({'signal':None})),m=None,obs=None,budget=m.PAIR_LIMITS)
except BeforeLaunch:pass
finally:core.subprocess=original_subprocess
assert len(captured)==1 and captured[0][:3]==['taskset','-c','0,1']
assert m.PAIR_LIMITS['cpu_affinity']==[6,7]
record={'status':'FAIL_CPU_ROUTING_DELTA','adapter_sha256':sha,'tests_passed':15 if code==0 else None,'existing_test_exit_code':code,'requested_budget_cpu_affinity':m.PAIR_LIMITS['cpu_affinity'],'actual_intercepted_popen_argv':captured[0],'actual_process_launched':False,'review_cpu_affinity':sorted(os.sched_getaffinity(0)),'elapsed_seconds':time.monotonic()-start,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}
Path(__file__).with_name('RESULT.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record));raise SystemExit(code)
