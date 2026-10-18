"""One bounded independent CPU lifecycle case against the final clean source."""
import ctypes
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
OWNER = Path('/mnt/project/VLA-Drive-st-rollout-native-resume-completion-v5-20261018')
EXPECTED = '8787b62bd4f897b991ef92a8882b146b9ff42082'
FILE = OWNER / 'resources/st_rollout_resume_completion_v5.py'
EXPECTED_SHA = '85713b7a2b1ad855e2c24b2dbd8beda71c215859a3dc99a77450878264ede4cb'


def clean():
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=OWNER, text=True).strip() == EXPECTED
    assert not subprocess.check_output(['git', 'status', '--porcelain=v1'], cwd=OWNER).strip()
    assert hashlib.sha256(FILE.read_bytes()).hexdigest() == EXPECTED_SHA


def family():
    seen, todo = set(), [os.getpid()]
    while todo:
        pid = todo.pop()
        if pid in seen:
            continue
        seen.add(pid)
        try:
            todo.extend(int(p) for p in Path(f'/proc/{pid}/task/{pid}/children').read_text().split())
        except OSError:
            pass
    return seen - {os.getpid()}


def stamp(pid):
    try:
        return int(Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19])
    except (OSError, ValueError, IndexError):
        return None


def rss(pid):
    try:
        return next(int(row.split()[1])*1024 for row in Path(f'/proc/{pid}/status').read_text().splitlines() if row.startswith('VmRSS:'))
    except (OSError, StopIteration):
        return 0


clean()
libc = ctypes.CDLL(None, use_errno=True)
previous = ctypes.c_int()
assert libc.prctl(37, ctypes.byref(previous), 0, 0, 0) == 0
assert libc.prctl(36, 1, 0, 0, 0) == 0
env = {**os.environ, 'PYTEST_DISABLE_PLUGIN_AUTOLOAD': '1', 'CUDA_VISIBLE_DEVICES': '',
       'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
       'TORCHINDUCTOR_COMPILE_THREADS': '1'}
argv = [sys.executable, '-m', 'pytest', '-q', str(HERE / 'test_completion_lifecycle.py')]
before = resource.getrusage(resource.RUSAGE_CHILDREN)
start = time.monotonic()
peak, observed, reason = 0, {}, None
with (HERE / 'TEST_OUTPUT.txt').open('w') as output:
    child = subprocess.Popen(argv, cwd=OWNER, env=env, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
    while child.poll() is None:
        members = family()
        observed.update({pid: born for pid in members if (born := stamp(pid)) is not None})
        peak = max(peak, sum(rss(pid) for pid in members))
        if peak > 2*2**30:
            reason = 'aggregate_RSS_cap_2GiB'
        if time.monotonic() - start > 60:
            reason = 'wall_cap_60s'
        if (HERE / 'TEST_OUTPUT.txt').stat().st_size > 2**20:
            reason = 'test_output_cap_1MiB'
        if reason:
            break
        time.sleep(.02)
    if reason:
        for pid, born in observed.items():
            if stamp(pid) == born:
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
    child.wait(timeout=5)
# The test-created orphan is adopted here and reaped; unrelated processes are
# not selected by UID, name, GPU or process group.
for pid in family():
    born = stamp(pid)
    if born is not None:
        observed[pid] = born
for pid, born in observed.items():
    if stamp(pid) == born:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
end = time.monotonic() + 3
while time.monotonic() < end:
    try:
        pid, _ = os.waitpid(-1, os.WNOHANG)
        if not pid:
            time.sleep(.01)
    except ChildProcessError:
        break
assert libc.prctl(36, previous.value, 0, 0, 0) == 0
clean()
after = resource.getrusage(resource.RUSAGE_CHILDREN)
record = {
    'schema': 'independent_completion_v5_lifecycle_test', 'source_commit': EXPECTED,
    'controller_sha256': EXPECTED_SHA, 'command': argv, 'exit_code': child.returncode,
    'stop_reason': reason, 'wall_seconds': time.monotonic() - start,
    'waited_child_CPU_seconds': after.ru_utime + after.ru_stime - before.ru_utime - before.ru_stime,
    'sampled_20ms_aggregate_RSS_bytes': peak, 'own_test_processes_reaped': not family(),
    'GPU_calls': 0, 'official_score_calls': 0, 'model_native_tensor_reads': 0,
    'test_scope': 'One CPU process lifecycle, failure exit, missing rank receipt, proven nested-setsid orphan drain, unrelated-process preservation. Fixture receipts are not actual native/GPU evidence.',
    'old_DS_DDP_profile_tests_rerun': False,
}
(HERE / 'TEST_RESULT.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record))
sys.exit(child.returncode if reason is None else 2)
