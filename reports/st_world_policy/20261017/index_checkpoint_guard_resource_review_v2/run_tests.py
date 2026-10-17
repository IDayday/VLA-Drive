import os,json,time,resource
from pathlib import Path
start=time.monotonic()
def io():return {k:int(v) for k,v in (x.split(':') for x in Path('/proc/self/io').read_text().splitlines())}
before=io();assert sorted(os.sched_getaffinity(0))==[6,7]
resource.setrlimit(resource.RLIMIT_AS,(4<<30,4<<30))
import pytest
code=pytest.main(['-q','-p','no:cacheprovider','reports/st_world_policy/20261017/full_future_index_execution_v1/test_run_index_once.py','reports/st_world_policy/20261017/full_future_index_execution_v1/test_bounded_complete_read.py'])
after=io();r={'status':'PASS' if code==0 else 'FAIL','exit_code':code,'elapsed_seconds':time.monotonic()-start,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'rchar_bytes':after['rchar']-before['rchar'],'cpu_affinity':sorted(os.sched_getaffinity(0)),'no_index_execution':True,'no_payload_read':True}
Path(__file__).with_name('TEST_RESULT.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r));raise SystemExit(code)
