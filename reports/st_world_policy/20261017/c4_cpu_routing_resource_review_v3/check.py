import os,json,time,resource,hashlib,importlib.util
from pathlib import Path
from types import SimpleNamespace
from contextlib import nullcontext
start=time.monotonic();assert sorted(os.sched_getaffinity(0))==[6,7]
def io():return {k:int(v) for k,v in (x.split(':') for x in Path('/proc/self/io').read_text().splitlines())}
before=io();resource.setrlimit(resource.RLIMIT_AS,(4<<30,4<<30))
import pytest
code=pytest.main(['-q','-p','no:cacheprovider','reports/st_world_policy/20261016/evaluation/FULL_PAYLOAD_C4_1024_PREPARATION_v1/test_c4_once.py'])
p=Path('reports/st_world_policy/20261016/evaluation/FULL_PAYLOAD_C4_1024_PREPARATION_v1/run_c4_once.py');sha=hashlib.sha256(p.read_bytes()).hexdigest();assert sha=='c4806201c46caee4a770545fa6e00764a4b5da5ba65ea97b389268b08be14c5f'
spec=importlib.util.spec_from_file_location('cpu_delta',p);r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
m=r.original();core,_,_=m.dependencies();r.route_monitor(core);captured=[]
class BeforeLaunch(Exception):pass
def reject_launch(argv,**kwargs):captured.append({'argv':argv,'kwargs_keys':sorted(kwargs)});raise BeforeLaunch('Intercept before real analysis process creation')
original_popen=r.subprocess.Popen;r.subprocess.Popen=reject_launch
try:
    core.monitor(['/fixture/no-execution'],runtime='/fixture',env={},fd=0,root=Path('/fixture'),s=SimpleNamespace(lifetime_scope=lambda:nullcontext({'signal':None})),m=None,obs=None,budget=m.PAIR_LIMITS)
except BeforeLaunch:pass
finally:r.subprocess.Popen=original_popen
assert len(captured)==1 and captured[0]['argv'][:3]==['taskset','-c','6,7']
after=io();record={'status':'PASS_CPU_ROUTING_DELTA_ONLY','adapter_sha256':sha,'tests_passed':19 if code==0 else None,'test_exit_code':code,'requested_cpu_affinity':m.PAIR_LIMITS['cpu_affinity'],'actual_core_monitor_intercepted_popen':captured[0],'actual_affinity_fixture_executed':True,'actual_analysis_launched':False,'review_cpu_affinity':sorted(os.sched_getaffinity(0)),'elapsed_seconds':time.monotonic()-start,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'rchar_bytes':after['rchar']-before['rchar']}
Path(__file__).with_name('RESULT.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record));raise SystemExit(code)
