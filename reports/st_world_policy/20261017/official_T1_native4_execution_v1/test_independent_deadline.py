# Independent reviewer commit 14831196; only new storage_guard fixture wiring added. Assertions unchanged.
"""A deterministic, no-child/no-model reproduction of cleanup deadline renewal."""
from contextlib import contextmanager
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace
import pytest

SOURCE = Path(__file__).resolve().parents[1] / 'official_T1_native4_execution_v1/run_once.py'
spec = importlib.util.spec_from_file_location('independent_t1_adapter', SOURCE)
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


def test_cleanup_cannot_restart_at_the_already_expired_absolute_deadline(tmp_path, monkeypatch):
    deadlines=[]; signals=[]
    class Owner:
        verified=0
        def discover(self): return [123]
        def leader_done(self): return False
        def send(self, signal): signals.append(signal)
        def finish(self, *, terminate, deadline): deadlines.append(deadline); return 0
        def close(self): pass
    class Selector:
        def __init__(self): self.items={}
        def register(self, stream, event, name): self.items[name]=stream
        def get_map(self): return self.items
        def select(self, timeout): return []
        def close(self): pass
    @contextmanager
    def lifetime_scope(): yield {'signal': None}
    owner=Owner()
    s=SimpleNamespace(lifetime_scope=lifetime_scope, OwnedSession=lambda child: owner,
        pidfd_open=lambda pid: os.open('/dev/null', os.O_RDONLY), abort_spawn_setup=lambda *a: None)
    child=SimpleNamespace(pid=123, stdout=open('/dev/null','rb'),stderr=open('/dev/null','rb'))
    times=iter([0.,3601.,3781.,3781.])
    monkeypatch.setattr(m.time, 'monotonic', lambda: next(times))
    monkeypatch.setattr(m.selectors, 'DefaultSelector', Selector)
    monkeypatch.setattr(m.subprocess, 'Popen', lambda *a,**kw: child)
    monkeypatch.setattr(m, 'proc_io', lambda: {'rchar':0})
    monkeypatch.setattr(m, 'live_owned_usage', lambda *a: (0,0))
    monkeypatch.setattr(m, 'checkpoint_bytes', lambda *a:0)
    with pytest.raises(RuntimeError, match='cleanup exceeded'):
        m.monitor(['/fixture/no-child'],{},-1,s,SimpleNamespace(),1<<30, storage_guard=lambda: {'fixture': True})
    assert len(deadlines)==1
    # First TERM at 3601. Only the original 180-second allowance is admitted.
    assert deadlines[0] <= 3781., f'Cleanup restarted: {deadlines[0]} > original deadline 3781'
