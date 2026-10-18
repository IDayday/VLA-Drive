"""One bounded stdlib CPU lifecycle; explicit receipts are fixtures, not model evidence."""
import hashlib
import importlib.util
import json
import os
import resource
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

SOURCE = Path('/mnt/project/VLA-Drive-st-rollout-native-resume-completion-v5-20261018/resources/st_rollout_resume_completion_v5.py')
EXPECTED = '85713b7a2b1ad855e2c24b2dbd8beda71c215859a3dc99a77450878264ede4cb'
OUT = Path(__file__).with_name('NORMAL_EXIT_DRAIN_RESULT.json')
resource.setrlimit(resource.RLIMIT_AS, (2 * 1024**3, 2 * 1024**3))
resource.setrlimit(resource.RLIMIT_CPU, (20, 20))
start = time.monotonic()
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == EXPECTED
spec = importlib.util.spec_from_file_location('independent_completion_v5', SOURCE)
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)
v = c.legacy()
stop_owned = v.stop_smoke_owned
tracked = {os.getpid()}
peak_rss = 0
stop_sampling = threading.Event()

def rss(pid):
    try:
        rows = dict(line.split(':', 1) for line in Path(f'/proc/{pid}/status').read_text().splitlines() if ':' in line)
        return int(rows.get('VmRSS', '0 kB').split()[0]) * 1024
    except (FileNotFoundError, ProcessLookupError):
        return 0

def sample():
    global peak_rss
    while not stop_sampling.is_set():
        peak_rss = max(peak_rss, sum(rss(pid) for pid in tuple(tracked)))
        stop_sampling.wait(.01)

def wait(predicate):
    deadline = time.monotonic() + 8
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(.01)
    raise AssertionError('Fixture did not reach bounded boundary')

thread = threading.Thread(target=sample, daemon=True)
thread.start()
result = {'schema': 'independent_completion_resource_cpu_case_v5', 'controller_sha256': EXPECTED,
          'scope': 'Explicit CPU completion fixtures and real child processes only; no model/replay correctness assertion.',
          'actual_GPU_or_model_or_official_calls': 0}
try:
    with tempfile.TemporaryDirectory(prefix='st-independent-completion-v5-') as temp:
        root = Path(temp)
        c.ROOT = root / 'completion'
        v.ROOT = c.ROOT
        reg = c.ROOT / 'interrupted/REGISTRATION.json'
        reg.parent.mkdir(parents=True)
        reg.write_text(json.dumps({'explicit_fixture': True, 'model_loaded': False}))
        observer = reg.parent / 'observer'
        observer.mkdir()
        for rank in range(4):
            (observer / f'INITIAL_OBSERVER_COMPLETE_rank{rank}.json').write_text(json.dumps({
                'rank': rank, 'observer_sha256': c.OLD_SHA, 'runtime_sha': v.SOURCE,
                'completed': 2, 'registration': v.pin(reg)}))
        validated = []
        def validate(arm, step, updates):
            validated.append([arm, step, updates])
            return {'explicit_fixture': 'completed2_only'}
        v.validate_result = validate
        pidfile, release = root / 'child.pid', root / 'release'
        leafcode = f'import os,time;from pathlib import Path;Path({str(pidfile)!r}).write_text(str(os.getpid()));time.sleep(60)'
        rootcode = f'import subprocess,sys,time;from pathlib import Path;subprocess.Popen([sys.executable,"-c",{leafcode!r}],start_new_session=True);\nwhile not Path({str(release)!r}).exists():time.sleep(.01)'
        parent = subprocess.Popen([sys.executable, '-c', rootcode], start_new_session=True,
                                  stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        foreign = subprocess.Popen([sys.executable, '-c', 'import time;time.sleep(60)'], start_new_session=True,
                                   stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        tracked.update((parent.pid, foreign.pid))
        job = {'child': parent, 'owned': {}, 'start_ticks': None}
        try:
            wait(pidfile.exists)
            leaf = int(pidfile.read_text())
            tracked.add(leaf)
            tree = v.owned_processes(job)
            assert leaf in tree, 'Child ownership must be observed before parent exit'
            assert tree[leaf]['sid'] == leaf, 'Real child must create separate session'
            release.touch()
            wait(lambda: parent.poll() == 0)
            assert v.process_record(leaf)['state'] != 'Z', 'Child must genuinely outlive launcher'
            outcome = c.finish_successful_job(job, 'initial')
            assert outcome == {'explicit_fixture': 'completed2_only'}
            assert validated == [['interrupted', 2, 2]], 'Result must be checked before draining'
            assert job['cleanup_complete']
            assert not any(row['state'] != 'Z' for row in v.owned_processes(job).values())
            assert foreign.poll() is None, 'Unrelated fixture must remain running'
            result.update(status='PASS', launcher_exitcode=parent.returncode,
                          cross_session_child_proved_before_launcher_exit=True,
                          completion_validation_before_drain=True, owned_cleanup_complete=True,
                          unrelated_fixture_preserved=True)
        finally:
            stop_owned([job])
            if foreign.poll() is None:
                foreign.terminate()
            foreign.wait(timeout=5)
except BaseException as error:
    result.update(status='FAIL', error=repr(error))
    raise
finally:
    stop_sampling.set()
    thread.join(timeout=1)
    own = resource.getrusage(resource.RUSAGE_SELF)
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    result.update(wall_seconds=time.monotonic()-start,
                  self_CPU_seconds=own.ru_utime+own.ru_stime,
                  waited_children_CPU_seconds=children.ru_utime+children.ru_stime,
                  CPU_scope='Self plus waited children; orphan CPU is an observed lower bound.',
                  sampled_aggregate_fixture_RSS_bytes=peak_rss,
                  RSS_guard_bytes=2*1024**3,
                  large_payload_reads=0,
                  fixture_receipts_are_actual_model_completion=False)
    assert peak_rss < 2*1024**3
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True)+'\n')
print(json.dumps(result, sort_keys=True))
