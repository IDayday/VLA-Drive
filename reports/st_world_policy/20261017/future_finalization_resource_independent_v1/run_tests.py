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
           MKL_NUM_THREADS='1', INDEPENDENT_FIXTURE_ROOT='/tmp/st-future-finalization-independent-v2')
command = [sys.executable, '-u', str(HERE/'test_resources.py')]
start = time.monotonic(); baseline = io(os.getpid())['rchar']
peak = None; peak_child = None; limit_reason = None; sample_count = 0; unavailable_samples = 0
with (HERE/'ATTEMPT2_TEST_OUTPUT.txt').open('xb') as stream:
    child = subprocess.Popen(command, env=env, stdout=stream, stderr=subprocess.STDOUT)
    try:
        while child.poll() is None:
            try:
                cm = memory(child.pid); pm = memory(os.getpid())
                if 'VmHWM' in cm and 'VmRSS' in cm and 'VmRSS' in pm:
                    peak_child = max(peak_child or 0, cm['VmHWM'])
                    peak = max(peak or 0, cm['VmRSS'] + pm['VmRSS'])
                    sample_count += 1
                else:
                    unavailable_samples += 1  # Process may become a zombie between poll and status.
                live_read = io(child.pid)['rchar'] + io(os.getpid())['rchar'] - baseline
                if live_read > 256*(1<<20): limit_reason = 'LOGICAL_READ'
            except (FileNotFoundError, ProcessLookupError):
                unavailable_samples += 1
            if time.monotonic()-start > 120: limit_reason = 'WALL'
            if peak is not None and peak > 2*(1<<30): limit_reason = 'RSS'
            if stream.tell() > 1<<20: limit_reason = 'OUTPUT'
            if limit_reason:
                child.terminate(); child.wait(timeout=5); break
            time.sleep(.02)
    except BaseException as error:
        limit_reason = 'MONITOR_EXCEPTION_'+type(error).__name__
    finally:
        if child.poll() is None:
            child.terminate()
            try: child.wait(timeout=5)
            except subprocess.TimeoutExpired: child.kill(); child.wait(timeout=5)
        code = child.wait()
post_read = io(os.getpid())['rchar'] - baseline
result = {'reviewer':'/root/st_owned_descendants_review_v2',
          'source_commit':'1ce3a44134bee0aa52fd1f6b659103e989a0bbc2',
          'command':command, 'cpu_affinity':sorted(os.sched_getaffinity(0)),
          'exit_code':code, 'elapsed_seconds':time.monotonic()-start,
          'sampled_aggregate_rss_peak_bytes':peak, 'sampled_child_vm_hwm_peak_bytes':peak_child,
          'rss_samples':sample_count, 'rss_unavailable_samples':unavailable_samples,
          'rss_evidence':'AVAILABLE_SAMPLED_PEAK' if sample_count else 'NOT_AVAILABLE',
          'post_wait_parent_rchar_delta':post_read,
          'limits':{'seconds':120,'rss_bytes':2<<30,'logical_read_bytes':256<<20,
                    'synthetic_read_bytes':32<<20,'synthetic_write_bytes':32<<20},
          'limit_reason':limit_reason, 'real_payload_reads':False, 'gpu_dispatch':False,
          'production_ledger_mutations':False, 'read_accounting':'post-wait parent delta counts waited child once; imports and monitoring included',
          'rss_accounting':'20ms live /proc VmRSS aggregate plus child VmHWM; no inherited getrusage substitution'}
(HERE/'ATTEMPT2_TEST_RESULT.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
print(json.dumps(result,sort_keys=True,indent=2))
raise SystemExit(code or bool(limit_reason) or not sample_count)
