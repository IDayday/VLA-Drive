"""Independent CPU counterexamples; no real stage/admission/model/storage access.

These tests assert observed candidate defects so a passing reproduction is not
mistaken for production qualification. All signalled processes are created by
this module and pinned by pidfd before signalling.
"""
from contextlib import ExitStack
import ctypes
import json
import os
from pathlib import Path
import signal
import select
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
CANDIDATE = HERE.parent / "spatial_probe_actual_preparation_v1"
sys.path.insert(0, str(CANDIDATE))
import prepare_bindings as b
import supervise_once as s


def pidfd_open(pid):
    # This environment's Python lacks the optional os/signal pidfd wrappers.
    assert os.uname().machine in ('x86_64', 'aarch64')
    libc = ctypes.CDLL(None, use_errno=True)
    fd = libc.syscall(434, int(pid), 0)
    if fd < 0:
        err = ctypes.get_errno(); raise OSError(err, os.strerror(err))
    return fd


def pidfd_signal(fd, sig):
    libc = ctypes.CDLL(None, use_errno=True)
    result = libc.syscall(424, int(fd), int(sig), None, 0)
    if result < 0:
        err = ctypes.get_errno(); raise OSError(err, os.strerror(err))


def alive(pid):
    try:
        return Path(f"/proc/{pid}/stat").read_text().split(") ", 1)[1].split()[0] != "Z"
    except FileNotFoundError:
        return False


def stop_owned(pidfd):
    try:
        pidfd_signal(pidfd, signal.SIGTERM)
    except ProcessLookupError:
        return
    assert select.select([pidfd], [], [], 5)[0], "Owned fixture did not exit"


def wait_path(path, timeout=5):
    deadline = time.monotonic() + timeout
    while not path.exists():
        if time.monotonic() >= deadline:
            raise AssertionError(f"CPU fixture did not publish {path.name}")
        time.sleep(.01)
    return path


def test_reproduce_normal_leader_exit_leaves_own_descendant_and_lock(tmp_path):
    attempt = tmp_path / "stage"; attempt.mkdir()
    pid_file = tmp_path / "descendant.json"
    code = """import json,os,time
pid = os.fork()
if pid:
    os._exit(0)
os.close(1); os.close(2)
with open(os.environ['REVIEW_PID_FILE'], 'w') as f:
    json.dump({'pid':os.getpid(),'sid':os.getsid(0),'pgid':os.getpgrp()},f)
time.sleep(15)
"""
    pidfd = None
    try:
        with ExitStack() as stack:
            lock = s.acquire(tmp_path / "one-loader.lock", stack)
            result = s.supervise([sys.executable, "-c", code], cwd=tmp_path,
                env={**os.environ, "REVIEW_PID_FILE":str(pid_file)},
                cpus=sorted(os.sched_getaffinity(0))[:2], descriptors=(lock.fileno(),),
                attempt=attempt, wall=1, rss=1 << 30, rss_reader=lambda pid: 0)
        state = json.loads(wait_path(pid_file).read_text())
        pidfd = pidfd_open(state['pid'])
        child = json.loads((attempt / "CHILD.json").read_text())
        assert state['sid'] == state['pgid'] == child['pid']
        assert result['exit_code'] == 0 and result['limit_reason'] is None
        assert alive(state['pid']), "Counterexample no longer reproduced"
        with ExitStack() as stack:
            try:
                s.acquire(tmp_path / "one-loader.lock", stack)
            except BlockingIOError:
                pass
            else:
                raise AssertionError("Inherited global lock was unexpectedly released")
    finally:
        if pidfd is not None:
            stop_owned(pidfd); os.close(pidfd)


def test_reproduce_supervisor_sigterm_leaves_own_worker_running(tmp_path):
    attempt = tmp_path / "stage"; attempt.mkdir()
    harness = tmp_path / "harness.py"
    harness.write_text("""import os,sys,time
sys.path.insert(0, sys.argv[1])
import supervise_once as s
s.supervise([sys.executable,'-c','import time; time.sleep(15)'],
 cwd=sys.argv[2],env=dict(os.environ),cpus=sorted(os.sched_getaffinity(0))[:2],
 descriptors=(),attempt=sys.argv[2],wall=10,rss=1<<30,rss_reader=lambda pid:0)
""")
    worker_fd = None
    harness_proc = subprocess.Popen([sys.executable, str(harness), str(CANDIDATE), str(attempt)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    harness_fd = pidfd_open(harness_proc.pid)
    try:
        state = json.loads(wait_path(attempt / "CHILD.json").read_text())
        worker_fd = pidfd_open(state['pid'])
        assert state['owned_new_session'] is True
        pidfd_signal(harness_fd, signal.SIGTERM)
        assert harness_proc.wait(timeout=5) == -signal.SIGTERM
        time.sleep(.05)
        assert alive(state['pid']), "Counterexample no longer reproduced"
    finally:
        if worker_fd is not None:
            stop_owned(worker_fd); os.close(worker_fd)
        if harness_proc.poll() is None:
            stop_owned(harness_fd); harness_proc.wait(timeout=5)
        os.close(harness_fd)


def test_reproduce_ancestor_symlink_control_write(tmp_path):
    actual = tmp_path / "actual"; actual.mkdir()
    alias = tmp_path / "alias"; alias.symlink_to(actual, target_is_directory=True)
    target = alias / "stage" / "RESULT.json"
    pin = b.write_once(target, {"status":"CPU_FIXTURE"})
    assert (actual / "stage" / "RESULT.json").is_file()
    assert pin['sha256'] == b.read(actual / "stage" / "RESULT.json")[1]['sha256']


def test_reproduce_existing_control_reuse_skips_directory_budget(tmp_path):
    target = tmp_path / "SIGNATURE.json"
    first = b.write_once(target, {"status":"CPU_FIXTURE"})
    with (tmp_path / "other.log").open("wb") as f:
        f.truncate(b.DIRECTORY_LIMIT + 1)  # sparse local fixture, not payload bytes
    assert b.observed_bytes(tmp_path) > b.DIRECTORY_LIMIT
    assert b.write_once(target, {"status":"CPU_FIXTURE"}) == first
