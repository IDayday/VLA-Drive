"""CPU-only real process fixtures; no torch/model/GPU/checkpoint load."""
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

HERE = Path(__file__).parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT/'spatial_probe_actual_preparation_v1'))
import supervise_once as legacy


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


m = load('new_owned_descendants', HERE/'owned_descendants.py')
adapter = load('future_t1_adapter', ROOT/'official_T1_native4_execution_v1/run_once.py')
core = load('unchanged_io_accounting', ROOT/'full_future_metadata_preparation_v1/run_build_once.py')

TREE = r'''
import json,os,subprocess,sys,time
from pathlib import Path
root=Path(sys.argv[1]); role=sys.argv[2]
child=None
if role in ('root','rank'):
    child=subprocess.Popen([sys.executable,__file__,str(root),'rank' if role=='root' else 'loader'],start_new_session=True)
memory=bytearray((8 if role!='root' else 1)<<20)
for n in range(2):
    with (root/'input.bin').open('rb') as f: f.read()
(root/(role+'.json')).write_text(json.dumps({'pid':os.getpid(),'sid':os.getsid(0)}))
while not (root/'quit').exists():
    if role=='root' and (root/'orphan').exists(): os._exit(7)
    time.sleep(.02)
if child is not None: child.wait(timeout=5)
'''


def ready(owner, root, timeout=5):
    deadline = time.monotonic()+timeout
    while time.monotonic() < deadline:
        owner.discover()
        if len(owner.members) == 3 and (root/'loader.json').exists():
            return
        time.sleep(.02)
    raise AssertionError('Fixture process tree did not become ready')


def spawn(root, facade):
    (root/'input.bin').write_bytes(b'1'*(1 << 20))
    script = root/'tree.py'; script.write_text(TREE)
    child = subprocess.Popen([sys.executable, str(script), str(root), 'root'],
                             start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return child, facade.OwnedSession(child)


@pytest.mark.parametrize('orphan', [False, True])
def test_real_cross_session_rank_loader_rss_io_and_cleanup(tmp_path, orphan):
    baseline = adapter.proc_io()
    with legacy.lifetime_scope():
        facade = m.ancestry_supervisor(legacy)  # before Popen
        child, owner = spawn(tmp_path, facade)
        try:
            ready(owner, tmp_path)
            ids = [json.loads((tmp_path/(role+'.json')).read_text()) for role in ('root','rank','loader')]
            assert len({x['sid'] for x in ids}) == 3
            rss, reads = adapter.live_owned_usage(owner, facade)
            assert rss > (16 << 20) and reads >= (6 << 20)
            assert owner.verified == 3
            assert {r['child']['pid'] for r in owner.evidence} == {x['pid'] for x in ids}
            if orphan:
                (tmp_path/'orphan').touch()
                deadline = time.monotonic()+3
                while not owner.leader_done() and time.monotonic() < deadline:
                    owner.discover(); time.sleep(.02)
                assert owner.leader_done()
                assert len(owner.discover()) == 2  # remembered rank and loader after reparent
                result = owner.finish(terminate=True, deadline=time.monotonic()+3)
                assert result == 7
            else:
                (tmp_path/'quit').touch()
                deadline = time.monotonic()+3
                while not owner.leader_done() and time.monotonic() < deadline:
                    owner.discover(); time.sleep(.02)
                owner.discover()
                terminal = core.owned_proc_io(owner, facade)
                pre = adapter.proc_io()
                assert owner.finish(terminate=False, deadline=deadline) == 0
                post = adapter.proc_io()
                accounting = core.terminal_read_accounting(baseline, pre, post,
                    terminal['rchar'], 64 << 20, final_child_available=True)
                assert accounting['status'] == 'COMPLETE'
                assert accounting['charged_parent_child_rchar_bytes'] == post['rchar']-baseline['rchar']
                assert accounting['charged_parent_child_rchar_bytes'] < (
                    post['rchar']-baseline['rchar']+terminal['rchar'])
            assert all(not Path('/proc/'+str(x['pid'])).exists() for x in ids)
        finally:
            if child.returncode is None:
                owner.finish(terminate=True, deadline=time.monotonic()+3)
            owner.close()


def test_new_session_after_discovery_preserves_pid_identity():
    facade = m.ancestry_supervisor(legacy)
    a = {'pid':12, 'start_ticks':90, 'sid':12}
    assert facade.OwnedSession.same(a, {**a, 'sid':99})
    assert not facade.OwnedSession.same(a, {**a, 'start_ticks':91})


def test_foreign_matching_command_is_not_owned_and_never_signalled(tmp_path, monkeypatch):
    foreign = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(15)'], start_new_session=True)
    child = owner = None
    try:
        with legacy.lifetime_scope():
            facade = m.ancestry_supervisor(legacy)
            child, owner = spawn(tmp_path, facade)
            ready(owner, tmp_path)
            assert foreign.pid not in owner.members
            sent = []
            original = legacy.pidfd_signal
            def checked(fd, sig):
                sent.append(fd)
                return original(fd, sig)
            monkeypatch.setattr(legacy, 'pidfd_signal', checked)
            assert owner.finish(terminate=True, deadline=time.monotonic()+3) < 0
            assert foreign.poll() is None
            assert sent
    finally:
        if owner is not None:
            if child.returncode is None:
                owner.finish(terminate=True, deadline=time.monotonic()+3)
            owner.close()
        foreign.terminate(); foreign.wait(timeout=3)


def test_unwitnessed_direct_child_is_rejected_without_claim(tmp_path):
    child = owner = foreign = None
    with legacy.lifetime_scope():
        facade = m.ancestry_supervisor(legacy)
        try:
            child, owner = spawn(tmp_path, facade)
            ready(owner, tmp_path)
            foreign = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(15)'], start_new_session=True)
            owner.discover()
            assert foreign.pid not in owner.members
            with pytest.raises(RuntimeError, match='Unwitnessed'):
                owner.assert_scope()
            with pytest.raises(RuntimeError, match='Unwitnessed'):
                owner.finish(terminate=True, deadline=time.monotonic()+3)
            assert foreign.poll() is None
        finally:
            if child is not None and child.returncode is None:
                try: owner.finish(terminate=True, deadline=time.monotonic()+3)
                except RuntimeError: pass
            if owner is not None: owner.close()
            if foreign is not None: foreign.terminate(); foreign.wait(timeout=3)


def test_recycled_pid_is_not_admitted_from_stale_parent(monkeypatch):
    facade = m.ancestry_supervisor(legacy)
    owner = object.__new__(facade.OwnedSession)
    owner.members = {}; owner.verified = 0; owner.evidence = []
    parent = {'pid':10001, 'start_ticks':9, 'sid':10001}
    child = {'pid':10002, 'start_ticks':11, 'sid':10002, 'ppid':10001}
    monkeypatch.setattr(legacy, 'proc_identity', lambda pid: {**parent, 'start_ticks':10} if pid==10001 else child)
    monkeypatch.setattr(legacy, 'pidfd_open', lambda pid: pytest.fail('No fd may be acquired for stale ancestry'))
    assert not owner._admit_child(child, parent)
    assert not owner.members


def test_parent_reuse_during_pidfd_open_rejected_and_fd_closed(monkeypatch):
    facade = m.ancestry_supervisor(legacy)
    owner = object.__new__(facade.OwnedSession)
    owner.members = {}; owner.verified = 0; owner.evidence = []
    parent = {'pid':10001, 'start_ticks':9, 'sid':10001}
    child = {'pid':10002, 'start_ticks':11, 'sid':10002, 'ppid':10001}
    calls = []
    def identity(pid):
        if pid == 10001:
            calls.append(pid)
            return parent if len(calls) == 1 else {**parent, 'start_ticks':10}
        return child
    fd = os.open('/dev/null', os.O_RDONLY)
    monkeypatch.setattr(legacy, 'proc_identity', identity)
    monkeypatch.setattr(legacy, 'pidfd_open', lambda pid: fd)
    assert not owner._admit_child(child, parent)
    with pytest.raises(OSError): os.fstat(fd)
    assert not owner.members


def test_future_adapter_uses_real_rank_loader_scope(tmp_path, monkeypatch):
    script = tmp_path/'tree.py'
    script.write_text(TREE.replace("while not (root/'quit').exists():",
        "deadline=time.monotonic()+1.4\nwhile not (root/'quit').exists() and time.monotonic()<deadline:"))
    (tmp_path/'input.bin').write_bytes(b'1'*(1 << 20))
    facade = m.ancestry_supervisor(legacy)
    actual_popen = subprocess.Popen
    cpus = ','.join(map(str, sorted(os.sched_getaffinity(0))))
    assert len(os.sched_getaffinity(0)) == 2
    def tiny_popen(command, **kwargs):
        assert command[:3] == ['taskset', '-c', '2-5']
        command[2] = cpus
        kwargs['cwd'] = tmp_path
        return actual_popen(command, **kwargs)
    monkeypatch.setattr(adapter.subprocess, 'Popen', tiny_popen)
    monkeypatch.setattr(adapter, 'ROOT', tmp_path)
    parent_rss = int(Path('/proc/self/statm').read_text().split()[1])*os.sysconf('SC_PAGE_SIZE')
    with (tmp_path/'lock').open('w') as lock:
        out = adapter.monitor([sys.executable, str(script), str(tmp_path), 'root'],
            dict(os.environ, CUDA_VISIBLE_DEVICES=''), lock.fileno(), facade, core, 64 << 20,
            storage_guard=lambda: {'fixture_only':True}, lease_valid_until=time.time()+5000)
    assert out['failure'] is None and out['returncode'] == 0
    assert out['cleanup']['verified_processes'] == 3
    assert out['cleanup']['live_owned_remaining'] == 0
    assert len(out['cleanup']['ownership_evidence']) == 3
    assert out['peak_rss_bytes'] > parent_rss + (16 << 20)
    assert out['accounting']['charged_parent_child_rchar_bytes'] > (6 << 20)
    assert out['accounting']['status'] == 'COMPLETE'


def test_old_admission_without_new_helper_pin_is_rejected(monkeypatch):
    from types import SimpleNamespace
    a = dict(schema='st_T1_native4_execution_admission_v1', status='APPROVED_BOUNDED_T1_ONLY',
        limits=adapter.LIMITS, hostname=adapter.socket.gethostname(),
        runtime={'path':str(adapter.RUNTIME), 'commit':adapter.SOURCE},
        adapter={'path':str(Path(adapter.__file__).absolute()), 'sha256':'a'*64},
        issued_at_unix=0, latest_start_unix=1e20, valid_until_unix=1e20)
    monkeypatch.setattr(adapter, 'pin_read', lambda pin, decode=True: a if decode else b'fixture')
    with pytest.raises(ValueError, match='ownership helper'):
        adapter.controls(SimpleNamespace(admission='/fixture', admission_sha256='a'*64))


def test_absolute_deadline_never_grants_fresh_grace(tmp_path):
    with legacy.lifetime_scope():
        facade = m.ancestry_supervisor(legacy)
        child, owner = spawn(tmp_path, facade)
        try:
            ready(owner, tmp_path)
            deadline = time.monotonic()-1
            began = time.monotonic()
            with pytest.raises(RuntimeError, match='absolute cleanup deadline'):
                owner.finish(terminate=True, deadline=deadline)
            assert time.monotonic()-began < .5
            # Fixture harness reaps its explicitly owned killed children; this
            # is not a production retry or a second training cleanup budget.
            owner.finish(terminate=False, deadline=time.monotonic()+3)
        finally:
            if child.returncode is None:
                owner.finish(terminate=True, deadline=time.monotonic()+3)
            owner.close()


@pytest.mark.parametrize('deadline', [None, float('nan'), float('inf')])
def test_missing_or_nonfinite_deadline_rejected(deadline):
    facade = m.ancestry_supervisor(legacy)
    owner = object.__new__(facade.OwnedSession)
    with pytest.raises(ValueError, match='absolute cleanup deadline'):
        owner.finish(terminate=True, deadline=deadline)
