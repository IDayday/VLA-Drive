import time
START=time.monotonic()
with open('/proc/self/io') as f:BOOT={k:int(v) for k,v in (line.split(':') for line in f)}
import os,json,resource,signal,sys
from pathlib import Path
resource.setrlimit(resource.RLIMIT_AS,(1<<30,1<<30))
signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('Review deadline')))
signal.alarm(240)
os.environ['PYTEST_DISABLE_PLUGIN_AUTOLOAD']='1'
import pytest
HERE=Path(__file__).resolve().parent
rc=pytest.main(['-v','-p','no:cacheprovider',str(HERE.parent/'matched_spatial_probe_controls_v1/test_control.py'),
    str(HERE/'test_independent.py'),'-k','not test_real_normal_exit_and_real_orphan_still_fails and not test_stale_active_leader_snapshot_is_rechecked'])
with open('/proc/self/io') as f:final={k:int(v) for k,v in (line.split(':') for line in f)}
print('COST_JSON='+json.dumps({'returncode':rc,'wall_seconds':time.monotonic()-START,
    'rss_peak_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
    'logical_read_bytes':final['rchar']-BOOT['rchar'],'logical_written_bytes':final['wchar']-BOOT['wchar'],
    'cpu_affinity':sorted(os.sched_getaffinity(0)),'torch_imported':'torch' in sys.modules,
    'actual_model_data_or_database_reads':False,'actual_process_launches':0,'actual_gpu_or_loader_use':False},sort_keys=True))
raise SystemExit(rc)
