"""Independent CPU counterexamples; no real stage/admission/model/storage access.

Adapted regression expectations for the narrow repair; the original independent
counterexample file and its failing-candidate evidence remain unchanged. All signalled processes are created by
this module and pinned by pidfd before signalling.
"""
from contextlib import ExitStack
import ctypes
import pytest
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


def test_normal_leader_exit_closes_own_descendant_and_lock(tmp_path):
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
        try:
            pidfd = pidfd_open(state['pid'])
        except ProcessLookupError:
            pass
        child = json.loads((attempt / "CHILD.json").read_text())
        assert state['sid'] == state['pgid'] == child['pid']
        assert result['exit_code'] == 0 and result['limit_reason'] == 'OWNED_DESCENDANTS_AFTER_LEADER_EXIT'
        assert result['process_cleanup']['live_owned_remaining'] == 0
        assert not alive(state['pid'])
        with ExitStack() as stack:
            s.acquire(tmp_path / 'one-loader.lock', stack)
    finally:
        if pidfd is not None:
            stop_owned(pidfd); os.close(pidfd)


@pytest.mark.parametrize('stop_signal', [signal.SIGTERM, signal.SIGINT])
def test_supervisor_signal_cleans_own_worker(tmp_path, stop_signal):
    attempt = tmp_path / "stage"; attempt.mkdir()
    harness = tmp_path / "harness.py"
    harness.write_text("""import os,sys,time
sys.path.insert(0, sys.argv[1])
import supervise_once as s
result = s.supervise([sys.executable,'-c','import time; time.sleep(15)'],
 cwd=sys.argv[2],env=dict(os.environ),cpus=sorted(os.sched_getaffinity(0))[:2],
 descriptors=(),attempt=sys.argv[2],wall=10,rss=1<<30,rss_reader=lambda pid:0)
s.write_once(__import__('pathlib').Path(sys.argv[2])/'RESULT.json',result)
""")
    worker_fd = None
    harness_proc = subprocess.Popen([sys.executable, str(harness), str(CANDIDATE), str(attempt)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    harness_fd = pidfd_open(harness_proc.pid)
    try:
        state = json.loads(wait_path(attempt / "CHILD.json").read_text())
        worker_fd = pidfd_open(state['pid'])
        assert state['owned_new_session'] is True
        pidfd_signal(harness_fd, stop_signal)
        assert harness_proc.wait(timeout=5) == 0
        result=json.loads((attempt/'RESULT.json').read_text())
        assert result['limit_reason'] == 'SUPERVISOR_SIGNAL_' + str(int(stop_signal))
        assert result['process_cleanup']['live_owned_remaining'] == 0
        assert not alive(state['pid'])
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
    with pytest.raises(ValueError,match='Symlink'):
        b.write_once(target, {'status':'CPU_FIXTURE'})
    assert not (actual/'stage').exists()


def test_reproduce_existing_control_reuse_skips_directory_budget(tmp_path):
    target = tmp_path / "SIGNATURE.json"
    first = b.write_once(target, {"status":"CPU_FIXTURE"})
    with (tmp_path / "other.log").open("wb") as f:
        f.truncate(b.DIRECTORY_LIMIT + 1)  # sparse local fixture, not payload bytes
    assert b.observed_bytes(tmp_path) > b.DIRECTORY_LIMIT
    with pytest.raises(ValueError,match='Existing control directory'):
        b.write_once(target, {'status':'CPU_FIXTURE'})


@pytest.mark.parametrize('code,expected', [
    ('print("normal")', 0),
    ('import os,time; p=os.fork(); time.sleep(.02); os.waitpid(p,0) if p else None', 0),
    ('import os,signal; os.kill(os.getpid(),signal.SIGTERM)', -signal.SIGTERM)])
def test_normal_natural_children_and_signal_exit(tmp_path, code, expected):
    result=s.supervise([sys.executable,'-c',code],cwd=tmp_path,env=dict(os.environ),
        cpus=sorted(os.sched_getaffinity(0))[:2],descriptors=(),attempt=tmp_path,
        wall=3,rss=1<<30,rss_reader=lambda pid:0)
    assert result['exit_code']==expected and result['limit_reason'] is None
    assert result['process_cleanup']['live_owned_remaining']==0


def test_descendant_retaining_pipe_is_closed(tmp_path):
    code='import os,time; p=os.fork(); os._exit(0) if p else time.sleep(15)'
    started=time.monotonic()
    result=s.supervise([sys.executable,'-c',code],cwd=tmp_path,env=dict(os.environ),
        cpus=sorted(os.sched_getaffinity(0))[:2],descriptors=(),attempt=tmp_path,
        wall=2,rss=1<<30,rss_reader=lambda pid:0)
    assert result['limit_reason']=='OWNED_DESCENDANTS_AFTER_LEADER_EXIT'
    assert result['process_cleanup']['live_owned_remaining']==0
    assert time.monotonic()-started<4


def test_changed_spawn_identity_refuses_signal_then_owned_pidfd_cleanup(tmp_path, monkeypatch):
    with s.lifetime_scope():
        child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(15)'],start_new_session=True)
        owner=s.OwnedSession(child)
        actual=s.proc_identity
        try:
            with monkeypatch.context() as m:
                m.setattr(s,'proc_identity',lambda pid:{**actual(pid),'start_ticks':actual(pid)['start_ticks']+1})
                with pytest.raises(RuntimeError,match='anchor changed'): owner.send(signal.SIGTERM)
            owner.finish(terminate=True)
        finally:
            owner.close()
    assert child.returncode==-signal.SIGTERM

@pytest.mark.parametrize('fault', ['owner_init', 'selector_init', 'selector_register', 'proc_identity', 'leader_pidfd'])
def test_post_spawn_setup_failure_closes_fresh_child_and_lock(tmp_path, monkeypatch, fault):
    attempt=tmp_path/'stage'; attempt.mkdir()
    spawned=[]; fixture_fds=[]
    original_open=s.pidfd_open
    def spawn(*args, **kwargs):
        child=subprocess.Popen(*args, **kwargs)
        spawned.append(child); fixture_fds.append(pidfd_open(child.pid))
        return child
    def fail(*args, **kwargs):
        raise OSError('injected '+fault)
    if fault=='owner_init': monkeypatch.setattr(s.OwnedSession,'__init__',fail)
    elif fault=='selector_init': monkeypatch.setattr(s.selectors,'DefaultSelector',fail)
    elif fault=='selector_register': monkeypatch.setattr(s.selectors.DefaultSelector,'register',fail)
    elif fault=='proc_identity': monkeypatch.setattr(s,'proc_identity',fail)
    else:
        monkeypatch.setattr(s,'pidfd_open',lambda pid: fail() if spawned and pid==spawned[0].pid else original_open(pid))
    outsider=subprocess.Popen([sys.executable,'-c','import time; time.sleep(15)'],start_new_session=True)
    outsider_fd=pidfd_open(outsider.pid)
    try:
        with ExitStack() as stack:
            lock=s.acquire(tmp_path/'one-loader.lock',stack)
            with pytest.raises(OSError,match='injected '+fault):
                s.supervise([sys.executable,'-c','import time; time.sleep(15)'],cwd=tmp_path,
                    env=dict(os.environ),cpus=sorted(os.sched_getaffinity(0))[:2],
                    descriptors=(lock.fileno(),),attempt=attempt,wall=3,rss=1<<30,
                    rss_reader=lambda pid:0,popen=spawn)
        assert len(spawned)==1 and spawned[0].returncode is not None
        assert not alive(spawned[0].pid)
        assert alive(outsider.pid)
        with ExitStack() as stack: s.acquire(tmp_path/'one-loader.lock',stack)
    finally:
        for child,fd in zip(spawned,fixture_fds):
            stop_owned(fd); child.wait(timeout=5); os.close(fd)
        stop_owned(outsider_fd); outsider.wait(timeout=5); os.close(outsider_fd)


@pytest.mark.parametrize('new_group', [False, True])
def test_registration_failure_after_spawned_leader_exit_closes_descendant(tmp_path, monkeypatch, new_group):
    pid_file=tmp_path/'tail.json'; attempt=tmp_path/'stage'; attempt.mkdir()
    spawned=[]; fds=[]
    code="""import os,json,time
p=os.fork()
if p: os._exit(0)
if os.environ['NEW_GROUP']=='1': os.setpgid(0,0)
os.close(1); os.close(2)
with open(os.environ['TAIL_FILE'],'w') as f: json.dump({'pid':os.getpid()},f)
time.sleep(15)
"""
    def spawn(*args,**kwargs):
        child=subprocess.Popen(*args,**kwargs); spawned.append(child); fds.append(pidfd_open(child.pid))
        return child
    def fail(self,child):
        wait_path(pid_file)
        state=json.loads(pid_file.read_text()); fds.append(pidfd_open(state['pid']))
        raise OSError('registration failed after leader exit')
    monkeypatch.setattr(s.OwnedSession,'__init__',fail)
    try:
        with ExitStack() as stack:
            lock=s.acquire(tmp_path/'one-loader.lock',stack)
            with pytest.raises(OSError,match='registration failed'):
                s.supervise([sys.executable,'-c',code],cwd=tmp_path,
                    env={**os.environ,'TAIL_FILE':str(pid_file),'NEW_GROUP':str(int(new_group))},cpus=sorted(os.sched_getaffinity(0))[:2],
                    descriptors=(lock.fileno(),),attempt=attempt,wall=3,rss=1<<30,
                    rss_reader=lambda pid:0,popen=spawn)
        assert spawned[0].returncode==0
        assert not alive(json.loads(pid_file.read_text())['pid'])
        with ExitStack() as stack: s.acquire(tmp_path/'one-loader.lock',stack)
    finally:
        for fd in fds: stop_owned(fd); os.close(fd)
        for child in spawned: child.wait(timeout=5)
