"""Independent CPU-only counterexamples; all policies/files are tmp_path fixtures."""
from pathlib import Path
import time
from types import SimpleNamespace

import pytest

from test_metadata_storage_guard import fixture, write, update_registration
from tools.st_world_policy import cache_storage_budget as b
from tools.st_world_policy import metadata_storage as m


def test_pending_retry_still_requires_unexpired_quiescence(tmp_path, monkeypatch):
    f = fixture(tmp_path)
    now = time.time()
    f.q['valid_until_unix'] = now + 1
    f.kwargs['quiescence_sha256'] = write(f.q_path, f.q)

    def fault(stage):
        if stage == 'after_pending':
            raise RuntimeError('interrupt before changing ledger')

    with pytest.raises(RuntimeError):
        m.migrate_policy(**f.kwargs, fault=fault)
    original = f.old.state_path.read_bytes()
    monkeypatch.setattr(m.time, 'time', lambda: now + 2)
    with pytest.raises(b.StorageBudgetError, match='quiescence expired'):
        m.migrate_policy(**f.kwargs)
    assert f.old.state_path.read_bytes() == original


def test_nonexistent_parent_is_checked_before_write(tmp_path):
    f = fixture(tmp_path, nested=True)
    # file regex permits this path, but no directory pattern permits log_shards.
    m.migrate_policy(**f.kwargs)
    with pytest.raises(b.StorageBudgetError, match='directory'):
        with m.metadata_writing(**f.guard) as guard:
            target = f.root / 'log_shards' / ('a' * 64 + '.json')
            assert not target.parent.exists()
            guard.check(before_commit=True, path=target)
            pytest.fail('unregistered planned directory passed pre-write check')


def test_unwired_runner_rejects_before_artifact_publication(tmp_path, monkeypatch):
    f = fixture(tmp_path)
    f.reg.update(command=['/fixture/python', '-m', 'tools.st_world_policy.index_teacher_metadata',
        'build', '--output', str(f.root)], cwd=str(Path.cwd()),
        callback_protocol='st_metadata_storage_check_v1')
    update_registration(f)
    m.migrate_policy(**f.kwargs)

    class Parser:
        def parse_args(self, _):
            return SimpleNamespace(command='build')

    def unwired(_):
        # Deliberately no hooks: the existing index CLI has this same boundary.
        write(f.root / 'COMPLETE.json', {'status': 'COMPLETE', 'fixture': True})
        return {'status': 'COMPLETE'}

    monkeypatch.setattr(m.importlib, 'import_module',
                        lambda _: SimpleNamespace(parser=Parser, run=unwired))
    with pytest.raises(b.StorageBudgetError):
        m.run_registered_metadata(**f.guard)
    assert not (f.root / 'COMPLETE.json').exists(), 'late hook counting is not pre-write admission'
