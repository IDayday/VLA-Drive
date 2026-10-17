"""Bounded CPU fixtures: stale leader snapshot, verified descendants and identity errors."""
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import time
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location('future_exit_race', Path(__file__).with_name('run_build_once.py'))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def envelope():
    return {'schema': 'st_full_future_child_result_v2',
            'cli_result': {'status': 'GUARDED_METADATA_RETURN'},
            'counters': {'proc_self_io_baseline': {'rchar': 0},
                         'proc_self_io_final': {'rchar': 0},
                         'proc_self_io_delta': {'rchar': 0}}}


@pytest.mark.parametrize('exitcode', [0, 3])
@pytest.mark.parametrize('after_done', ['empty', 'descendant', 'identity_error'])
def test_deterministic_discover_to_done_race(tmp_path, monkeypatch, exitcode, after_done):
    """First snapshot contains leader; WNOWAIT then reports its exit before refresh."""
    pipes = []
    for payload in (json.dumps(envelope()).encode(), b''):
        r, w = os.pipe()
        os.write(w, payload)
        os.close(w)
        pipes.append(os.fdopen(r, 'rb'))
    child = SimpleNamespace(pid=123, stdout=pipes[0], stderr=pipes[1])
    events = []

    class Owner:
        verified = 2 if after_done == 'descendant' else 1
        def __init__(self):
            self.discovers = 0
            self.finishes = 0
            self.signals = []
        def discover(self):
            self.discovers += 1
            events.append(('discover', self.discovers))
            if self.discovers == 1:
                return [({'pid': 123}, 7)]
            if self.discovers == 2 and after_done == 'identity_error':
                raise RuntimeError('Spawn anchor changed; no numeric-session signalling allowed')
            if after_done == 'descendant' and not self.signals:
                return [({'pid': 124}, 8)]
            return []
        def leader_done(self):
            events.append(('leader_done', True))
            return True
        def send(self, sig):
            self.signals.append(sig)
        def finish(self, *, terminate, deadline=None):
            self.finishes += 1
            assert self.finishes == 1
            events.append(('finish', terminate))
            return exitcode
        def close(self):
            events.append(('close', True))

    owner = Owner()
    @contextmanager
    def lifetime():
        yield {'signal': None}
    supervisor = SimpleNamespace(lifetime_scope=lifetime,
        pidfd_open=lambda _: os.open('/dev/null', os.O_RDONLY),
        OwnedSession=lambda _: owner,
        abort_spawn_setup=lambda *_: pytest.fail('Owner exists; setup abort forbidden'))
    monkeypatch.setattr(m.subprocess, 'Popen', lambda *a, **k: child)
    monkeypatch.setattr(m, 'owned_proc_io', lambda *_: {'rchar': 0})
    monkeypatch.setattr(m, 'BOOT_IO', {'rchar': 0})
    monkeypatch.setattr(m, 'START', time.monotonic())
    kwargs = dict(runtime=str(tmp_path), env={}, fd=0, root=tmp_path, s=supervisor,
        m=SimpleNamespace(process_tree_rss=lambda _: 0),
        obs=SimpleNamespace(proc_io=lambda: {'rchar': 0}), budget=m.limits('dev'))
    if after_done == 'identity_error':
        with pytest.raises(RuntimeError, match='Spawn anchor changed'):
            m.monitor(['not-executed'], **kwargs)
        assert ('finish', True) in events
    else:
        result = m.monitor(['not-executed'], **kwargs)
        expected = 'OWNED_DESCENDANTS_AFTER_LEADER_EXIT' if after_done == 'descendant' else None
        assert result['returncode'] == exitcode
        assert result['failure'] == expected
        assert result['cleanup']['live_owned_remaining'] == 0
        assert result['read_accounting']['status'] == 'COMPLETE'
        assert result['read_accounting']['charged_parent_child_rchar_bytes'] == 0
        assert ('finish', after_done == 'descendant') in events
        assert bool(owner.signals) == (after_done == 'descendant')
        if owner.signals:
            assert set(owner.signals) == {signal.SIGTERM}
    assert events[:3] == [('discover', 1), ('leader_done', True), ('discover', 2)]
    assert owner.finishes == 1
    assert events[-1] == ('close', True)


@pytest.mark.parametrize('exitcode', [0, 3])
@pytest.mark.parametrize('descendant', [False, True])
def test_real_exit_with_and_without_live_descendant(tmp_path, monkeypatch, exitcode, descendant):
    helpers = Path('/mnt/project/VLA-Drive-st-spatial-probe-runtime-v2-20261017/reports/st_world_policy/20261017/spatial_probe_actual_preparation_v1')
    for name, digest in m.MONITORS.items():
        assert hashlib.sha256((helpers / name).read_bytes()).hexdigest() == digest
    monkeypatch.syspath_prepend(str(helpers))
    import supervise_once as supervisor
    import observed_cli as observer
    code = """import json,os,sys,time
read=lambda:{k:int(v) for k,v in (line.split(':') for line in open('/proc/self/io'))}
before=read();descendant=None
if sys.argv[2]=='yes':
    descendant=os.fork()
    if descendant==0:
        fd=os.open('/dev/null',os.O_WRONLY)
        os.dup2(fd,1);os.dup2(fd,2);os.close(fd)
        time.sleep(10)
        os._exit(0)
time.sleep(.15)
after=read()
print(json.dumps({'schema':'st_full_future_child_result_v2','cli_result':{'status':'GUARDED_METADATA_RETURN','descendant_pid':descendant},'counters':{'proc_self_io_baseline':before,'proc_self_io_final':after,'proc_self_io_delta':{k:after[k]-before[k] for k in after}}}),flush=True)
raise SystemExit(int(sys.argv[1]))
"""
    monkeypatch.setattr(m, 'BOOT_IO', observer.proc_io())
    monkeypatch.setattr(m, 'START', time.monotonic())
    budget = m.limits('dev')
    budget.update(wall_seconds=10, read_bytes=64 << 20, rss_bytes=1 << 30)
    with (tmp_path / 'owned-fd').open('wb') as fd:
        result = m.monitor([sys.executable, '-c', code, str(exitcode), 'yes' if descendant else 'no'],
            runtime=str(tmp_path), env=dict(os.environ, CUDA_VISIBLE_DEVICES=''), fd=fd.fileno(),
            root=tmp_path, s=supervisor, m=SimpleNamespace(process_tree_rss=lambda _: 32 << 20),
            obs=observer, budget=budget)
    assert result['returncode'] == exitcode
    assert result['failure'] == ('OWNED_DESCENDANTS_AFTER_LEADER_EXIT' if descendant else None)
    assert result['cleanup']['live_owned_remaining'] == 0
    assert result['read_accounting']['status'] == 'COMPLETE'
    a = result['read_accounting']
    assert a['charged_parent_child_rchar_bytes'] == a['parent_post_reap']['rchar'] - m.BOOT_IO['rchar']
    if descendant:
        pid = result['child_accounting']['cli_result']['descendant_pid']
        assert type(pid) is int and not Path('/proc', str(pid)).exists()
        assert result['cleanup']['verified_processes'] >= 2
    else:
        assert result['child_accounting']['cli_result']['descendant_pid'] is None
    assert not list(tmp_path.glob('CONTROL*'))
