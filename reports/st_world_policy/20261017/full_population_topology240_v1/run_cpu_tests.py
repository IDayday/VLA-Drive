"""Small-source test budget only; no runtime preparation or dispatch."""
import os, resource, time, sys, json, runpy
from pathlib import Path
resource.setrlimit(resource.RLIMIT_AS,(1<<30,1<<30));os.sched_setaffinity(0,{2,3})
def io():return {k:int(v) for k,v in (x.split(':') for x in Path('/proc/self/io').read_text().splitlines())}
start=time.monotonic();before=io();h=Path(__file__).resolve().parent;sys.path.insert(0,str(h));code=0
paths=[Path('/mnt/project/VLA-Drive-st-full-topology-profile-runtime-20261017/tests/st_world/test_full_profile_timing_boundary.py'),h/'test_controls.py']
for path in paths:
    sys.argv=[str(path)]
    try:runpy.run_path(str(path),run_name='__main__')
    except SystemExit as e:code=max(code,e.code or 0)
end=io();result={'command':'taskset -c 2,3 timeout 60s /usr/bin/python3 -B '+str(h/'run_cpu_tests.py'),
    'exit_code':code,'wall_seconds':time.monotonic()-start,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
    'proc_self_io_delta':{k:end[k]-before[k] for k in end},'cpu_affinity':sorted(os.sched_getaffinity(0)),
    'address_space_cap_bytes':1<<30,'real_db_checkpoint_payload_gpu_access':False}
(h/'TEST_RESOURCES.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(json.dumps(result));sys.exit(code)
