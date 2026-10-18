"""Bounded independent CPU-only setsid ownership tests; no torch/model/GPU."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace

source = Path(sys.argv[1]).resolve()
spec = importlib.util.spec_from_file_location('setsid_review_source', source)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
started = time.monotonic()
results = []
leaf_code = 'import time; time.sleep(60)'
leader_code = ('import json,os,subprocess,sys,time;from pathlib import Path;'
               'p=subprocess.Popen([sys.executable,"-c",sys.argv[2]]);'
               'Path(sys.argv[1]).write_text(json.dumps({"leader":os.getpid(),"leaf":p.pid}));time.sleep(60)')
root_code = ('import subprocess,sys,time;'
             'subprocess.Popen([sys.executable,"-c",sys.argv[2],sys.argv[1],sys.argv[3]],start_new_session=True);'
             'time.sleep(60)')

def record(pid):
    try:
        return m.process_record(pid)
    except (OSError, ValueError):
        return None

def live(pid):
    row = record(pid)
    return row is not None and row['state'] != 'Z'

def wait_for(test):
    end = time.monotonic()+5
    while not test():
        if time.monotonic() > end:
            raise AssertionError('Bounded CPU fixture timed out')
        time.sleep(.01)

for orphan in (False, True):
    child = foreign = None
    identities = {}
    with tempfile.TemporaryDirectory(prefix='setsid_resource_review_') as folder:
        ready = Path(folder)/'ready.json'
        try:
            child = subprocess.Popen([sys.executable, '-c', root_code, str(ready), leader_code, leaf_code],
                                     start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            foreign = subprocess.Popen([sys.executable, '-c', leaf_code], start_new_session=True,
                                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            wait_for(ready.exists)
            pids = json.loads(ready.read_text())
            expected = {child.pid, pids['leader'], pids['leaf']}
            for pid in expected:
                identities[pid] = record(pid)['start_ticks']
            job = {'child': child, 'owned': {}, 'start_ticks': identities[child.pid]}
            owned = m.owned_processes(job)
            assert expected.issubset(owned)
            assert foreign.pid not in owned
            assert owned[pids['leader']]['sid'] != owned[child.pid]['sid']
            assert sum(row['rss'] for row in owned.values()) >= sum(owned[pid]['rss'] for pid in expected)
            if orphan:
                os.kill(child.pid, signal.SIGTERM)
                os.kill(pids['leader'], signal.SIGTERM)
                child.wait(timeout=5)
                wait_for(lambda: not live(pids['leader']))
                owned = m.owned_processes(job)
                assert pids['leaf'] in owned and live(pids['leaf'])
                assert foreign.pid not in owned
            m.stop_smoke_owned([job])
            assert job['cleanup_complete'] and not any(live(pid) for pid in expected)
            assert foreign.poll() is None
            results.append({'case': 'observed_orphan_new_session' if orphan else 'PPID_cross_setsid_tree',
                            'status': 'PASS', 'expected_real_owned_processes': len(expected),
                            'new_child_session_proven': True, 'foreign_process_survived_owned_cleanup': True,
                            'known_residual_alive': 0})
        finally:
            for pid, stamp in identities.items():
                row = record(pid)
                if row is not None and row['state'] != 'Z' and row['start_ticks'] == stamp:
                    os.kill(pid, signal.SIGKILL)
            for process in (child, foreign):
                if process is not None:
                    if process.poll() is None:
                        process.kill()
                    process.wait(timeout=5)

original_table = m.process_table
def row(pid, ppid, sid, start):
    return {'pid': pid, 'ppid': ppid, 'sid': sid, 'start_ticks': start, 'rss': 1,
            'rss_bytes': 1, 'pgrp': sid, 'state': 'S', 'cpu_seconds': 0}
try:
    job = {'child': SimpleNamespace(pid=100, poll=lambda: 0), 'start_ticks': 1,
           'owned': {}, 'owned_sessions': {900: 2}}
    m.process_table = lambda: {901: row(901, 1, 900, 20)}
    assert not m.owned_processes(job), 'Stale SID alone adopted an unknown process'
    job = {'child': SimpleNamespace(pid=100, poll=lambda: 0), 'start_ticks': 1,
           'owned': {101: row(101, 100, 100, 2)}, 'owned_sessions': {100: 1}}
    m.process_table = lambda: {100: row(100, 1, 100, 99), 101: row(101, 1, 100, 100)}
    assert not m.owned_processes(job), 'Reused PID/start identity adopted foreign state'
    results.append({'case': 'stale_SID_and_PID_reuse_negative_fixture', 'status': 'PASS',
                    'unknown_processes_adopted': 0, 'signals_sent': 0})
finally:
    m.process_table = original_table

usage = resource.getrusage(resource.RUSAGE_SELF)
children = resource.getrusage(resource.RUSAGE_CHILDREN)
result = {'status': 'PASS', 'reviewer': '/root/road_failure_resources_review_v1',
          'source': str(source), 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
          'cases': results, 'wall_seconds': time.monotonic()-started,
          'CPU_seconds_self_and_waited_children': usage.ru_utime+usage.ru_stime+children.ru_utime+children.ru_stime,
          'post_cases_current_RSS_bytes': int(Path('/proc/self/statm').read_text().split()[1])*os.sysconf('SC_PAGE_SIZE'),
          'children_max_peak_RSS_bytes': children.ru_maxrss*1024,
          'rusage_self_highwater_may_include_pre_exec_bytes': usage.ru_maxrss*1024,
          'torch_imported': 'torch' in sys.modules, 'GPU_queries': 0, 'model_calls': 0,
          'official_calls': 0, 'large_payload_reads': 0, 'old_four_lifecycle_tests_rerun': False}
assert not result['torch_imported']
assert result['CPU_seconds_self_and_waited_children'] < 60 and result['wall_seconds'] < 600
Path(sys.argv[2]).write_text(json.dumps(result, indent=2, sort_keys=True)+'\n')
print(json.dumps({'status': 'PASS', 'cases': len(results), 'wall_seconds': result['wall_seconds']}))
