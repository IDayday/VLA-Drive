"""One finite local synthetic pytest invocation; no campaign data reads."""
import json,os,resource,signal,time,sys
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
start=time.monotonic()
def io():return {k:int(v) for k,v in (line.split(':') for line in Path('/proc/self/io').read_text().splitlines())}
boot=io();root=Path(__file__).resolve().parent
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['PYTHONDONTWRITEBYTECODE']='1'
resource.setrlimit(resource.RLIMIT_FSIZE,(16<<20,16<<20))
def deadline(*_):raise RuntimeError('independent180s budget exhausted')
signal.signal(signal.SIGALRM,deadline);signal.alarm(180)
import pytest
from tools.st_world_policy import finalize_future_metadata
startup=io();startup_seconds=time.monotonic()-start
single=os.environ.get('REVIEW_SINGLE')=='1'
fixtures=Path('/tmp/st-finalizer-independent-dispatch-20261017') if single else root/'synthetic_tmp'
targets=['tests/st_world/test_finalize_future_metadata.py::test_actual_storage_dispatcher_new_claim_hooks_and_receipt'] if single else ['tests/st_world/test_finalize_future_metadata.py',str(root/'test_independent.py')]
code=pytest.main(['-vv','-p','no:cacheprovider','--basetemp',str(fixtures),*targets])
end=io();peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
files=[p for p in fixtures.rglob('*') if p.is_file() and not p.is_symlink()]
size=sum(p.stat().st_size for p in files)
report={'pytest_exitcode':code,'elapsed_seconds':time.monotonic()-start,'startup_seconds':startup_seconds,
 'startup_logical_read_bytes':startup['rchar']-boot['rchar'],'test_window_logical_read_bytes':end['rchar']-startup['rchar'],
 'total_process_logical_read_bytes':end['rchar']-boot['rchar'],'startup_wchar_bytes':startup['wchar']-boot['wchar'],
 'test_window_wchar_bytes':end['wchar']-startup['wchar'],'peak_RSS_bytes':peak,
 'synthetic_files':len(files),'synthetic_retained_bytes':size,'cpu_affinity':sorted(os.sched_getaffinity(0)),
 'cuda_visible_devices':os.environ['CUDA_VISIBLE_DEVICES'],'no_real_payload':True,'no_actual_ledger_mutation':True,
 'bounds':{'wall_seconds':180,'RSS_bytes':2<<30,'total_process_reads':256<<20,'fixture_reads_writes_each':64<<20}}
report['bounded_checks_pass']=report['elapsed_seconds']<180 and peak<2<<30 and report['total_process_logical_read_bytes']<256<<20 and report['test_window_logical_read_bytes']<64<<20 and report['test_window_wchar_bytes']<64<<20
(root/('SINGLE_TEST_COST.json' if single else 'TEST_COST.json')).write_text(json.dumps(report,indent=2)+'\n');print('INDEPENDENT_COST',json.dumps(report));signal.alarm(0)
raise SystemExit(code if report['bounded_checks_pass'] else 99)
