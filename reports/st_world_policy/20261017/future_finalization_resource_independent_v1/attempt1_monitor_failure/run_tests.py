"""One bounded subprocess; no actual campaign payload or external dispatch."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
os.sched_setaffinity(0, {8, 9})
def io(pid):
    return {k:int(v) for k,v in (line.split(':',1) for line in Path(f'/proc/{pid}/io').read_text().splitlines())}
def memory(pid):
    data = Path(f'/proc/{pid}/status').read_text()
    return {k:int(v.strip().split()[0])*1024 for k,v in
            (line.split(':',1) for line in data.splitlines() if line.startswith(('VmRSS:','VmHWM:')))}

env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=str(Path.cwd()),
           CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
           MKL_NUM_THREADS='1', INDEPENDENT_FIXTURE_ROOT='/tmp/st-future-finalization-independent-v1')
command = [sys.executable, '-u', str(HERE/'test_resources.py')]
start = time.monotonic(); baseline = io(os.getpid())['rchar']
peak = 0; peak_child = 0; limit_reason = None; sample_count = 0
with (HERE/'TEST_OUTPUT.txt').open('xb') as stream:
    child = subprocess.Popen(command, env=env, stdout=stream, stderr=subprocess.STDOUT)
    while child.poll() is None:
        try:
            cm = memory(child.pid); pm = memory(os.getpid())
            peak_child = max(peak_child, cm['VmHWM:'.rstrip(':')])
            peak = max(peak, cm['VmRSS'] + pm['VmRSS'])
            live_read = io(child.pid)['rchar'] + io(os.getpid())['rchar'] - baseline
            sample_count += 1
            if time.monotonic()-start > 120: limit_reason = 'WALL'
            if peak > 2*(1<<30): limit_reason = 'RSS'
            if live_read > 256*(1<<20): limit_reason = 'LOGICAL_READ'
            if stream.tell() > 1<<20: limit_reason = 'OUTPUT'
            if limit_reason:
                child.terminate(); child.wait(timeout=5); break
        except (FileNotFoundError, ProcessLookupError): pass
        time.sleep(.02)
    code = child.wait()
post_read = io(os.getpid())['rchar'] - baseline
result = {'reviewer':'/root/st_owned_descendants_review_v2',
          'source_commit':'1ce3a44134bee0aa52fd1f6b659103e989a0bbc2',
          'command':command, 'cpu_affinity':sorted(os.sched_getaffinity(0)),
          'exit_code':code, 'elapsed_seconds':time.monotonic()-start,
          'sampled_aggregate_rss_peak_bytes':peak, 'sampled_child_vm_hwm_peak_bytes':peak_child,
          'rss_samples':sample_count, 'post_wait_parent_rchar_delta':post_read,
          'limits':{'seconds':120,'rss_bytes':2<<30,'logical_read_bytes':256<<20,
                    'synthetic_read_bytes':32<<20,'synthetic_write_bytes':32<<20},
          'limit_reason':limit_reason, 'real_payload_reads':False, 'gpu_dispatch':False,
          'production_ledger_mutations':False, 'read_accounting':'post-wait parent delta counts waited child once; imports and monitoring included',
          'rss_accounting':'20ms live /proc VmRSS aggregate plus child VmHWM; no inherited getrusage substitution'}
(HERE/'TEST_RESULT.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
print(json.dumps(result,sort_keys=True,indent=2))
raise SystemExit(code or bool(limit_reason))
