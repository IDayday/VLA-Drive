"""Review-only mocked-parent affinity interception; no real analysis launch."""
import json, os, resource, time
from pathlib import Path
start=time.monotonic()
def io():
    return {k:int(v) for k,v in (line.split(':') for line in Path('/proc/self/io').read_text().splitlines())}
before=io(); actual=sorted(os.sched_getaffinity(0)); requests=[]
assert actual==[6,7]
resource.setrlimit(resource.RLIMIT_AS,(4<<30,4<<30))
def retain_review_affinity(pid,cpus):
    assert pid==0 and list(cpus)==[0,1], (pid,cpus)
    assert sorted(os.sched_getaffinity(0))==[6,7]
    requests.append({'requested':list(cpus),'actual_retained':[6,7],'scope':'mocked-parent fixture'})
os.sched_setaffinity=retain_review_affinity
import pytest
code=pytest.main(['-q','-p','no:cacheprovider','reports/st_world_policy/20261016/evaluation/FULL_PAYLOAD_C4_1024_PREPARATION_v1/test_c4_once.py'])
after=io()
record={'status':'PASS' if code==0 else 'FAIL','pytest_exit_code':code,'elapsed_seconds':time.monotonic()-start,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'actual_cpu_affinity':sorted(os.sched_getaffinity(0)),'fixture_affinity_requests':requests,'rchar_bytes':after['rchar']-before['rchar'],'no_actual_analysis':True,'no_gpu':True}
Path(__file__).with_name('TEST_RESULT.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record));raise SystemExit(code)
