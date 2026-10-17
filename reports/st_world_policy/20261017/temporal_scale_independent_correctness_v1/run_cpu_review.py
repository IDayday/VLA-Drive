"""Owned bounded single-process CPU test harness, no GPU or payload access."""
import json,os,resource,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parents[3]))
if len(sys.argv)>1 and sys.argv[1]=='--child':
    os.sched_setaffinity(0,{10,11})
    started=time.monotonic()
    try:
        import pytest
        code=pytest.main(['-q','tests/st_world/test_temporal_scale_calibration.py',str(ROOT/'test_independent.py')])
    finally:
        io=dict(line.split(':',1) for line in Path('/proc/self/io').read_text().splitlines())
        record={'elapsed_seconds':time.monotonic()-started,'max_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
          'logical_read_bytes':int(io['rchar']),'physical_read_bytes':int(io['read_bytes']),
          'CPU_affinity':sorted(os.sched_getaffinity(0)),'CUDA_VISIBLE_DEVICES':os.environ['CUDA_VISIBLE_DEVICES']}
        (ROOT/'TEST_COST.json').write_text(json.dumps(record,indent=2)+'\n')
    raise SystemExit(code)
start=time.monotonic(); before=sum(p.stat().st_size for p in ROOT.iterdir() if p.is_file())
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
with (ROOT/'TESTS.log').open('wb') as log:
    process=subprocess.Popen([sys.executable,__file__,'--child'],stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
    violation=None
    while process.poll() is None:
        try:
            status=Path(f'/proc/{process.pid}/status').read_text()
            rss=int(next(x for x in status.splitlines() if x.startswith('VmRSS:')).split()[1])*1024
            io=dict(line.split(':',1) for line in Path(f'/proc/{process.pid}/io').read_text().splitlines())
            output=sum(p.stat().st_size for p in ROOT.iterdir() if p.is_file())-before
            if time.monotonic()-start>120:violation='wall'
            elif rss>4*1024**3:violation='rss'
            elif int(io['rchar'])>256*1024**2:violation='logical_reads'
            elif output>16*1024**2:violation='output'
            if violation:
                process.terminate();process.wait(timeout=5);break
        except (FileNotFoundError,StopIteration,ProcessLookupError):pass
        time.sleep(.1)
    code=process.wait()
(ROOT/'TEST_TERMINAL.json').write_text(json.dumps({'exit_code':code,'violation':violation,'wall_seconds':time.monotonic()-start,
    'limits':{'seconds':120,'rss_bytes':4*1024**3,'logical_read_bytes':256*1024**2,'output_bytes':16*1024**2}},indent=2)+'\n')
print((ROOT/'TESTS.log').read_text())
print((ROOT/'TEST_TERMINAL.json').read_text())
if (ROOT/'TEST_COST.json').exists():print((ROOT/'TEST_COST.json').read_text())
raise SystemExit(code or bool(violation))
