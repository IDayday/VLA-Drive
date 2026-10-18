"""Independent bounded CPU fixture review; never reads a real scene/cache/model."""
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1',
           MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
           PYTEST_DISABLE_PLUGIN_AUTOLOAD='1')
cmd = [sys.executable, '-m', 'pytest', '-q',
       'tests/st_world/test_audit_flow_gradient_balance.py',
       str(HERE / 'test_independent.py')]
def limits():
    resource.setrlimit(resource.RLIMIT_CPU, (110, 110))
start = time.monotonic()
before = resource.getrusage(resource.RUSAGE_CHILDREN)
with (HERE / 'TEST_OUTPUT.txt').open('w') as out:
    child = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=out,
                             stderr=subprocess.STDOUT, preexec_fn=limits)
    peak = 0
    violation = None
    while child.poll() is None:
        try:
            status = Path(f'/proc/{child.pid}/status').read_text()
            rss = next(int(x.split()[1])*1024 for x in status.splitlines()
                       if x.startswith('VmRSS:'))
            peak = max(peak, rss)
        except (FileNotFoundError, StopIteration):
            pass
        if peak > 2*2**30 or time.monotonic()-start > 115:
            violation = 'RSS_OR_WALL_LIMIT'
            child.kill()
            break
        time.sleep(.05)
    code = child.wait()
after = resource.getrusage(resource.RUSAGE_CHILDREN)
record = dict(schema='flow_gradient_balance_independent_cpu_review_v1', command=cmd,
              implementation_commit='2dcdc52f',
              exit_code=code, violation=violation,
              wall_seconds=time.monotonic()-start,
              child_cpu_seconds=after.ru_utime-before.ru_utime+after.ru_stime-before.ru_stime,
              sampled_peak_rss_bytes=peak, rusage_maxrss_bytes=after.ru_maxrss*1024,
              real_scene_reads=0, metric_cache_reads=0, official_scoring_calls=0,
              model_loads=0, gpu_calls=0, bounded_source_read_budget_bytes=16*2**20)
(HERE / 'TEST_EXECUTION.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
raise SystemExit(code)
