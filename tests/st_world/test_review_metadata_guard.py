"""Independent CPU-only counterexamples; all policies/files are tmp_path fixtures."""
from pathlib import Path
import copy
import hashlib
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


@pytest.mark.parametrize('expire_at', ['after_snapshot', 'after_state'])
def test_quiescence_expiring_inside_migration_stops_next_publication(tmp_path, monkeypatch, expire_at):
    f = fixture(tmp_path)
    now = time.time()
    f.q['valid_until_unix'] = now + 1
    f.kwargs['quiescence_sha256'] = write(f.q_path, f.q)
    original = f.old.state_path.read_bytes()

    def expire(stage):
        if stage == expire_at:
            monkeypatch.setattr(m.time, 'time', lambda: now + 2)

    with pytest.raises(b.StorageBudgetError, match='quiescence expired'):
        m.migrate_policy(**f.kwargs, fault=expire)
    assert not list(f.old.state_dir.glob('*.complete.json'))
    assert bool(f.old.state_path.read_bytes() != original) == (expire_at == 'after_state')
    pending = next(f.old.state_dir.glob('*.pending.json'))
    migration_id = pending.name[len('MIGRATION_'):-len('.pending.json')]
    pending_bytes = pending.read_bytes()
    renewed = copy.deepcopy(f.q)
    renewed.update(checked_at_unix=now + 1.5, valid_until_unix=now + 3600,
                   resumes_migration_id=migration_id)
    renewed_path = tmp_path / 'renewed.json'
    renewed_sha = write(renewed_path, renewed)
    result = m.migrate_policy(**f.kwargs, renew_quiescence_receipt=renewed_path,
                              renew_quiescence_sha256=renewed_sha)
    assert result['migration_id'] == migration_id
    assert result['completion_quiescence_sha256'] == renewed_sha
    assert pending.read_bytes() == pending_bytes


def test_renewed_process_evidence_mutation_cannot_resume(tmp_path):
    f = fixture(tmp_path)

    def interrupt(stage):
        if stage == 'after_pending':
            raise RuntimeError('fixture interruption')

    with pytest.raises(RuntimeError):
        m.migrate_policy(**f.kwargs, fault=interrupt)
    before = f.old.state_path.read_bytes()
    pending = next(f.old.state_dir.glob('*.pending.json'))
    renewed = copy.deepcopy(f.q)
    renewed['resumes_migration_id'] = pending.name[len('MIGRATION_'):-len('.pending.json')]
    evidence_path = tmp_path / 'fresh_process_evidence.json'
    evidence_sha = write(evidence_path, {'fixture': True, 'writers': []})
    renewed['process_evidence'] = [{'path': str(evidence_path), 'sha256': evidence_sha}]
    renewed_path = tmp_path / 'renewed.json'
    renewed_sha = write(renewed_path, renewed)
    evidence_path.write_text('{"fixture": "changed after receipt"}\n')
    with pytest.raises(b.StorageBudgetError, match='SHA mismatch'):
        m.migrate_policy(**f.kwargs, renew_quiescence_receipt=renewed_path,
                         renew_quiescence_sha256=renewed_sha)
    assert f.old.state_path.read_bytes() == before
    assert not list(f.old.state_dir.glob('*.complete.json'))


def test_source_changed_during_import_rejected_before_parser_or_admission(tmp_path, monkeypatch):
    f = fixture(tmp_path)
    source = tmp_path / 'source'
    entry = source / 'tools/st_world_policy/index_teacher_metadata.py'
    dependency = source / 'starVLA/dataloader/st_world_metadata_index.py'
    entry.parent.mkdir(parents=True)
    dependency.parent.mkdir(parents=True)
    entry.write_text('# synthetic entry\n')
    dependency.write_text('# synthetic dependency\n')
    pins = [{'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
            for p in (entry, dependency)]
    f.reg.update(command=['/fixture/python', '-m', 'tools.st_world_policy.index_teacher_metadata',
                         'build', '--output', str(f.root)], cwd=str(source),
                 callback_protocol='st_metadata_storage_check_v1', source_pins=pins)
    update_registration(f)
    m.migrate_policy(**f.kwargs)
    monkeypatch.chdir(source)

    def must_not_execute(*_):
        pytest.fail('parser/build ran after registered source changed')

    def importing(_):
        dependency.write_text('# changed during entry import\n')
        return SimpleNamespace(parser=must_not_execute, run=must_not_execute,
            __file__=str(entry), METADATA_STORAGE_CALLBACK_PROTOCOL='st_metadata_storage_check_v1',
            METADATA_STORAGE_CALLBACK_DEPENDENCIES=('starVLA.dataloader.st_world_metadata_index',))

    monkeypatch.setattr(m.importlib, 'import_module', importing)
    before = f.old.state_path.read_bytes()
    with pytest.raises(b.StorageBudgetError, match='SHA mismatch'):
        m.run_registered_metadata(**f.guard)
    assert not f.root.exists()
    assert f.old.state_path.read_bytes() == before
