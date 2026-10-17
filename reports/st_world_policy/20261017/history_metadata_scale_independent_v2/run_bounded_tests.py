"""Independent synthetic-only runner. Does not inspect or control other jobs."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]

def io(pid):
    try:return {k:int(v) for k,v in (x.split(':',1) for x in Path(f'/proc/{pid}/io').read_text().splitlines())}
    except (OSError,ValueError):return {}

def tree(pid):
    found={pid};pending=[pid]
    while pending:
        current=pending.pop()
        try:tasks=list(Path(f'/proc/{current}/task').iterdir())
        except OSError:continue
        for task in tasks:
            try:children=map(int,(task/'children').read_text().split())
            except OSError:continue
            for child in children:
                if child not in found:found.add(child);pending.append(child)
    return found

def rss(pid):
    try:return int(Path(f'/proc/{pid}/statm').read_text().split()[1])*os.sysconf('SC_PAGE_SIZE')
    except (OSError,ValueError,IndexError):return 0

def charged(root):
    result=0
    for p in root.rglob('*'):
        try:
            s=p.lstat()
            if p.is_file() and not p.is_symlink():result+=max(s.st_size,s.st_blocks*512)
        except OSError:continue
    return result

assert set(os.sched_getaffinity(0))<={10,11}
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
start_io=0  # Include supervisor interpreter startup/import IO as well as descendants.
start=time.monotonic();peak_rss=peak_read=peak_procs=peak_bytes=0;failure=None
with tempfile.TemporaryDirectory(prefix='history-scale-review-v2-') as temporary:
    fixture=Path(temporary)
    command=[sys.executable,'-m','pytest','-q','-p','no:cacheprovider',
        '--basetemp',str(fixture/'fixtures'),
        str(HERE/'test_independent_fixed_scale.py'),
        'reports/st_world_policy/20261017/history_packed_reader_safety_v1/test_reader.py',
        'reports/st_world_policy/20261017/history_metadata_scale_v1/test_scale.py',
        'tests/st_world/test_history.py']
    with (HERE/'TEST_LOG.txt').open('w') as log:
        proc=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        while proc.poll() is None:
            pids=tree(proc.pid)
            total_rss=sum(rss(p) for p in pids)+rss(os.getpid())
            total_read=sum(io(p).get('rchar',0) for p in pids)+max(io(os.getpid()).get('rchar',0)-start_io,0)
            used=charged(fixture)
            peak_rss=max(peak_rss,total_rss);peak_read=max(peak_read,total_read)
            peak_procs=max(peak_procs,len(pids)+1);peak_bytes=max(peak_bytes,used)
            if time.monotonic()-start>160:failure='wall_seconds'
            elif total_rss>2<<30:failure='process_tree_rss'
            elif total_read>1700<<20:failure='logical_reads'
            elif used>32<<20:failure='fixture_files'
            if failure:
                os.killpg(proc.pid,signal.SIGTERM)
                try:proc.wait(timeout=3)
                except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
                break
            time.sleep(.02)
        exit_code=proc.wait()
    peak_bytes=max(peak_bytes,charged(fixture))
end_io=io(os.getpid())
elapsed=time.monotonic()-start
loghash=hashlib.sha256((HERE/'TEST_LOG.txt').read_bytes()).hexdigest()
result=dict(command=command,exit_code=exit_code,elapsed_seconds=elapsed,
    peak_complete_process_tree_rss_bytes=peak_rss,
    maximum_live_process_tree_rchar_sum_bytes=peak_read,
    charged_parent_after_reap_rchar_bytes=end_io.get('rchar',0)-start_io,
    peak_processes=peak_procs,fixture_peak_file_bytes_charged=peak_bytes,
    resource_limit_failure=failure,cpu_affinity=sorted(os.sched_getaffinity(0)),
    log_sha256=loghash,synthetic_unique_records=256,actual_full_history_build=False,
    prior_review_logical_read_bytes=446024135,prior_review_elapsed_seconds=17.764698671176944,
    prior_review_peak_rss_bytes=1284571136,
    accounting='Supervisor plus all observed descendants; summed /proc RSS sampled at20ms. Logical reads charged through supervisor after child reap; peak snapshot is conservative when parent counters include reaped descendants. Own ephemeral fixtures only. Supervisor rchar charged from process birth, including interpreter startup/imports; no observed production data reads. Wall starts after supervisor setup; all pytest interpreter startup and descendants are inside the timed window.')
result['cumulative_review_logical_read_bytes']=result['charged_parent_after_reap_rchar_bytes']+result['prior_review_logical_read_bytes']
result['cumulative_review_elapsed_seconds']=elapsed+result['prior_review_elapsed_seconds']
(HERE/'TEST_RUN.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
sys.exit(exit_code if not failure else 99)
