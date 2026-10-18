"""Single resource-guarded actual-class CPU check, not a model experiment."""
import importlib.util
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
SOURCE = Path(sys.argv[1]).resolve()
spec = importlib.util.spec_from_file_location('bounded_serializer_cleanup', SOURCE)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
command = ['/root/miniconda3/envs/ddp/bin/python', str(HERE/'real_installed_fragment_review.py'),
           str(SOURCE), str(HERE/'REAL_FRAGMENT_RESULT.json')]
env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'DS_ACCELERATOR': 'cpu',
       'TORCHINDUCTOR_COMPILE_THREADS': '1', 'OMP_NUM_THREADS': '1',
       'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'PYTHONDONTWRITEBYTECODE': '1'}
started = time.monotonic()
peak = 0
why = None
with (HERE/'REAL_FRAGMENT_LOG.txt').open('w') as log:
    child = subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL, stdout=log,
                             stderr=subprocess.STDOUT, start_new_session=True)
    job = {'child': child, 'owned': {}, 'start_ticks': None}
    try:
        while child.poll() is None:
            owned = m.owned_processes(job)
            rss = sum(row['rss'] for row in owned.values())
            peak = max(peak, rss)
            cpu = sum(row['cpu_seconds'] for row in owned.values())
            if rss > 2*2**30:
                why = 'AGGREGATE_RSS_CAP'
            elif cpu > 60:
                why = 'CPU_CAP'
            elif sum(p.stat().st_size for p in HERE.rglob('*') if p.is_file()) > 8*2**20:
                why = 'OUTPUT_CAP'
            elif time.monotonic()-started > 120:
                why = 'WALL_CAP'
            if why:
                break
            time.sleep(.02)
    finally:
        m.stop_smoke_owned([job])
    child.wait()
usage = resource.getrusage(resource.RUSAGE_CHILDREN)
self_usage = resource.getrusage(resource.RUSAGE_SELF)
result = {'status': 'PASS' if child.returncode == 0 and why is None and job['cleanup_complete'] else 'FAIL',
          'command': command, 'exit_code': child.returncode, 'stop_reason': why,
          'wall_seconds': time.monotonic()-started, 'sampled_aggregate_peak_RSS_bytes': peak,
          'waited_child_CPU_seconds': usage.ru_utime+usage.ru_stime,
          'guardian_CPU_seconds': self_usage.ru_utime+self_usage.ru_stime,
          'owned_cleanup_complete': job['cleanup_complete'],
          'limits': {'CPU_seconds': 60, 'RSS_bytes': 2*2**30, 'output_bytes': 8*2**20, 'wall_seconds': 120},
          'test_only_compile_threads': 1, 'DS_ACCELERATOR': 'cpu', 'CUDA_VISIBLE_DEVICES': '',
          'actual_model_or_official_calls': 0, 'large_payload_reads': 0,
          'scope': 'One installed fragment class and tiny CPU values; no engine/model/native payload.'}
(HERE/'REAL_FRAGMENT_EXECUTION.json').write_text(json.dumps(result, indent=2, sort_keys=True)+'\n')
print(json.dumps(result))
raise SystemExit(0 if result['status'] == 'PASS' else 1)
