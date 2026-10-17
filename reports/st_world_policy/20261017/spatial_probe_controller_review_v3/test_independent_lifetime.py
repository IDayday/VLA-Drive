"""Bounded CPU lifecycle review, no real ST stage or resource admission."""
from contextlib import ExitStack
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

HERE = Path(__file__).resolve().parent
HELPERS = HERE.parent / 'spatial_probe_actual_preparation_v1'
sys.path.insert(0, str(HELPERS))
import supervise_once as s
import prepare_bindings as b


def wait_json(path):
    end = time.monotonic() + 3
    while time.monotonic() < end:
        if path.exists():
            try:
                return json.loads(path.read_text())
            except json.JSONDecodeError:
                pass
        time.sleep(.01)
    raise AssertionError('Fixture metadata not published')


def stop_child(child):
    if child.poll() is None:
        fd = s.pidfd_open(child.pid)
        try:
            s.pidfd_signal(fd, signal.SIGTERM)
        finally:
            os.close(fd)
        child.wait(timeout=3)


@pytest.mark.parametrize('stop_signal', [signal.SIGTERM, signal.SIGINT])
@pytest.mark.parametrize('leader_early_exit', [False, True])
def test_cleanup_nested_owned_session_preserves_foreign_group(tmp_path, stop_signal, leader_early_exit):
    """Exercise independent supervisor plus grandchild and foreign-session control."""
    stage = tmp_path / 'stage'; stage.mkdir()
    marker = tmp_path / 'grandchild.json'
    child_code = """import os,time,json
p=os.fork()
if p:
    while not os.path.exists(os.environ['MARKER']): time.sleep(.005)
    if os.environ['EARLY']=='1': os._exit(0)
    time.sleep(10)
else:
    with open(os.environ['MARKER'],'w') as f: json.dump({'pid':os.getpid()},f)
    time.sleep(10)
"""
    child_path = tmp_path / 'worker.py'; child_path.write_text(child_code)
    harness = tmp_path / 'harness.py'
    harness.write_text("""import sys,os,json
from pathlib import Path
from contextlib import ExitStack
sys.path.insert(0,sys.argv[1])
import supervise_once as s
with ExitStack() as stack:
    lock=s.acquire(Path(sys.argv[2]).parent/'fixture.lock',stack)
    result=s.supervise([sys.executable,sys.argv[3]],cwd=sys.argv[2],env=dict(os.environ),
      cpus=sorted(os.sched_getaffinity(0)),descriptors=(lock.fileno(),),attempt=sys.argv[2],
      wall=4,rss=1<<30,rss_reader=lambda pid:0)
    s.write_once(Path(sys.argv[2])/'RESULT.json',result)
""")
    outsider = subprocess.Popen([sys.executable, '-c', 'import time;time.sleep(10)'], start_new_session=True)
    supervisor = subprocess.Popen([sys.executable, str(harness), str(HELPERS), str(stage), str(child_path)],
        env={**os.environ, 'MARKER':str(marker), 'EARLY':str(int(leader_early_exit))},
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        child = wait_json(stage / 'CHILD.json')
        grandchild = wait_json(marker)
        if not leader_early_exit:
            fd = s.pidfd_open(supervisor.pid)
            try:
                s.pidfd_signal(fd, stop_signal)
            finally:
                os.close(fd)
        assert supervisor.wait(timeout=5) == 0
        result = wait_json(stage / 'RESULT.json')
        assert result['process_cleanup']['live_owned_remaining'] == 0
        assert result['process_cleanup']['verified_processes'] >= 2
        assert result['limit_reason'] == ('OWNED_DESCENDANTS_AFTER_LEADER_EXIT' if leader_early_exit
                                         else 'SUPERVISOR_SIGNAL_' + str(int(stop_signal)))
        assert not Path(f"/proc/{child['pid']}").exists()
        assert not Path(f"/proc/{grandchild['pid']}").exists()
        assert outsider.poll() is None
        with pytest.raises(RuntimeError, match='foreign'):
            s.own_group_signal(outsider, signal.SIGTERM)
        assert outsider.poll() is None
        with ExitStack() as stack:
            s.acquire(tmp_path / 'fixture.lock', stack)
    finally:
        stop_child(supervisor); stop_child(outsider)


@pytest.mark.parametrize('fault', ['owner_init', 'selector_register'])
def test_post_spawn_setup_failure_closes_worker(tmp_path, monkeypatch, fault):
    """The two original setup fault injections must now close the fresh worker."""
    children = []
    def spawn(*a, **kw):
        child = subprocess.Popen(*a, **kw); children.append(child); return child
    def broken_init(self, child):
        raise OSError('fixture ownership initialization failure')
    class BrokenSelector:
        def register(self, *a, **kw):
            raise OSError('fixture selector registration failure')
        def close(self):
            pass
    try:
        with monkeypatch.context() as m:
            if fault == 'owner_init':
                m.setattr(s.OwnedSession, '__init__', broken_init)
            else:
                m.setattr(s.selectors, 'DefaultSelector', BrokenSelector)
            with pytest.raises(OSError, match='fixture'):
                s.supervise([sys.executable,'-c','import time;time.sleep(10)'],cwd=tmp_path,
                    env=dict(os.environ),cpus=sorted(os.sched_getaffinity(0)),descriptors=(),
                    attempt=tmp_path,wall=.2,rss=1<<30,rss_reader=lambda pid:0,popen=spawn)
        assert len(children) == 1 and children[0].returncode is not None
        assert not Path(f'/proc/{children[0].pid}').exists()
    finally:
        for child in children:
            stop_child(child)
            owner = getattr(child, '_probe_owned_session', None)
            if owner is not None:
                owner.close()


def test_control_ancestry_and_reuse_budget_remain_fail_closed(tmp_path):
    real=tmp_path/'real';real.mkdir()
    alias=tmp_path/'alias';alias.symlink_to(real, target_is_directory=True)
    with pytest.raises(ValueError,match='Symlink'):
        b.write_once(alias/'stage'/'RESULT.json',{'status':'fixture'})
    assert not (real/'stage').exists()
    path=real/'RESULT.json';b.write_once(path,{'status':'fixture'})
    with (real/'extra').open('wb') as f:
        f.truncate(b.DIRECTORY_LIMIT+1)
    with pytest.raises(ValueError,match='Existing control directory'):
        b.write_once(path,{'status':'fixture'})
