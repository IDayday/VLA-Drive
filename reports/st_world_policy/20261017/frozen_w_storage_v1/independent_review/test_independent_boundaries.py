"""Independent CPU controls only; every file/ledger is under pytest tmp_path."""
import copy
import hashlib
import json
import multiprocessing
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[5] / 'tests' / 'st_world'))
from test_spatial_w_storage import fixture, refresh, signature_fixture
from test_metadata_storage_guard import write
from tools.st_world_policy import cache_storage_budget as b
from tools.st_world_policy import metadata_storage as m


def _migrate_worker(kwargs, connection):
    try:
        connection.send(('ok', m.migrate_policy(**kwargs)))
    except Exception as exc:
        connection.send(('error', repr(exc)))
    finally:
        connection.close()


def test_two_processes_same_transition_use_one_ledger_history(tmp_path):
    f = fixture(tmp_path, cells=(('S-P0', 'train'),))
    ctx = multiprocessing.get_context('fork')
    children, receivers = [], []
    for _ in range(2):
        parent, child = ctx.Pipe(duplex=False)
        process = ctx.Process(target=_migrate_worker, args=(f.kwargs, child))
        process.start(); child.close()
        children.append(process); receivers.append(parent)
    for child in children:
        child.join(15)
        assert child.exitcode == 0
    results = [pipe.recv() for pipe in receivers]
    assert results[0][0] == results[1][0] == 'ok'
    assert results[0][1] == results[1][1]
    state = json.loads(f.m0.old.state_path.read_text())
    assert state['claims'] == f.before['claims']
    assert len(state['metadata_migrations']) == 2
    assert len(set(state['metadata_migrations'])) == 2


def test_declared_but_unadmitted_w_endpoint_cannot_be_replaced(tmp_path):
    f = fixture(tmp_path, cells=(('S-P0', 'train'),))
    m.migrate_policy(**f.kwargs)
    old = f.signatures[0]
    before = f.m0.old.state_path.read_bytes()
    assert str(old.root) not in json.loads(before)['claims']
    other = signature_fixture(tmp_path, revision='other-valid-endpoint')
    replacement = dict(other.claim, root=str(old.root), claim_id=old.claim['claim_id'])
    f.new_doc['claims'][-1] = replacement
    refresh(f)
    with pytest.raises(b.StorageBudgetError, match='member declarations'):
        b.StorageBudget(f.new_path)
    f.new_doc['frozen_w_package']['members'][str(old.root)] = copy.deepcopy(replacement)
    refresh(f)
    with pytest.raises(b.StorageBudgetError, match='different policy'):
        b.StorageBudget(f.new_path).initialize()
    assert f.m0.old.state_path.read_bytes() == before


def test_pending_w_transition_cannot_resume_with_expired_quiescence(tmp_path, monkeypatch):
    f = fixture(tmp_path, cells=(('S-P0', 'train'),))
    def fault(stage):
        if stage == 'after_pending':
            raise RuntimeError('independent fault after pending')
    with pytest.raises(RuntimeError, match='independent fault'):
        m.migrate_policy(**f.kwargs, fault=fault)
    before = f.m0.old.state_path.read_bytes()
    pending = sorted(f.m0.old.state_dir.glob('*.pending.json'))
    pending_pins = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in pending}
    monkeypatch.setattr(m.time, 'time', lambda: f.q['valid_until_unix'] + 1)
    with pytest.raises(b.StorageBudgetError, match='expired'):
        m.migrate_policy(**f.kwargs)
    assert f.m0.old.state_path.read_bytes() == before
    assert pending_pins == {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in pending}


def test_additive_migration_cannot_replace_old_unadmitted_member(tmp_path):
    f = fixture(tmp_path, cells=(('S-P0', 'train'),))
    m.migrate_policy(**f.kwargs)
    before = f.m0.old.state_path.read_bytes()
    old = f.signatures[0]
    other = signature_fixture(tmp_path, revision='other-valid-endpoint')
    new = copy.deepcopy(f.new_doc)
    new['claims'][-1] = dict(other.claim, root=str(old.root), claim_id=old.claim['claim_id'])
    new['frozen_w_package']['members'][str(old.root)] = copy.deepcopy(new['claims'][-1])
    path = tmp_path / 'replacement-policy.json'
    sha = write(path, new)
    with pytest.raises(b.StorageBudgetError, match='changed/removed an old claim'):
        m.migrate_policy(**{**f.kwargs, 'old_policy': f.new_path, 'old_policy_sha256': f.new_sha,
            'new_policy': path, 'new_policy_sha256': sha,
            'expected_old_ledger_sha256': hashlib.sha256(before).hexdigest()})
    assert f.m0.old.state_path.read_bytes() == before


def test_checkpoint_floor_rejects_only_new_protected_write_not_bare_floor(tmp_path, monkeypatch):
    from tools.st_world_policy import spatial_probe as p
    f = fixture(tmp_path, cells=(('S-P0', 'train'),))
    m.migrate_policy(**f.kwargs)
    s = f.signatures[0]
    budget = b.StorageBudget(f.new_path)
    pin = dict(path=str(f.m0.checkpoint_path), sha256=f.m0.checkpoint_sha)
    floor = p.extraction_free_reserve(budget, pin, s.binding['max_seconds'])
    reservation = budget.reserve(s.root, s.signature['identity'],
        **{k:s.claim[k] for k in ('output_bytes', 'staging_bytes', 'metadata_bytes')},
        free_reserve_bytes=floor)
    before = budget.state_path.read_bytes()
    state = json.loads(before)
    outstanding = budget._check_cap(state) - sum(r['observed_bytes'] for r in state['claims'].values())
    free = budget.core['free_reserve_bytes'] + outstanding + (1 << 19)
    monkeypatch.setattr(b.shutil, 'disk_usage', lambda _: SimpleNamespace(free=free))
    # This is not a generic low-space test: the old bare floor permits it.
    budget._check_disk(state, s.root, 1)
    with pytest.raises(b.StorageBudgetError, match='free-space reserve'):
        with reservation.writing(1):
            pytest.fail('protected write body must not execute')
    assert budget.state_path.read_bytes() == before


def test_actual_extract_passes_protected_floor_before_loader(tmp_path, monkeypatch):
    from tools.st_world_policy import spatial_probe as p, checkpoints
    f = fixture(tmp_path, cells=(('S-P0', 'train'),))
    m.migrate_policy(**f.kwargs)
    s = f.signatures[0]
    class Data:
        rows = s.rows
        def __len__(self): return len(self.rows)
        def __getitem__(self, index): pytest.fail('no image access')
    monkeypatch.setattr(p, 'cache_plan', lambda _: (s.signature, Data(), {k:s.claim[k] for k in
        ('output_bytes', 'staging_bytes', 'metadata_bytes')}))
    monkeypatch.setattr(p, 'source_runtime', lambda *a, **kw: None)
    original = b.StorageBudget.reserve
    observed = []
    def reserve(budget, *args, **kwargs):
        observed.append(kwargs['free_reserve_bytes'])
        return original(budget, *args, **kwargs)
    monkeypatch.setattr(b.StorageBudget, 'reserve', reserve)
    def stop(*a, **kw): raise RuntimeError('stop before actual native load')
    monkeypatch.setattr(checkpoints, 'load_world_checkpoint', stop)
    with pytest.raises(RuntimeError, match='stop before actual native load'):
        p.extract(s.binding, s.root, f.new_path, checkpoint_commitments=dict(
            path=str(f.m0.checkpoint_path), sha256=f.m0.checkpoint_sha))
    assert observed == [f.new_doc['free_reserve_bytes'] + (1 << 20)]
    assert json.loads((s.root / 'identity.json').read_text()) == s.signature
    assert not list(s.root.glob('*.safetensors'))


def test_typed_w_declared_package_cannot_use_second_state_directory(tmp_path):
    f = fixture(tmp_path, cells=(('S-P0', 'train'),))
    before = f.m0.old.state_path.read_bytes()
    f.new_doc['state_dir'] = str(tmp_path / 'new-ledger')
    refresh(f)
    with pytest.raises(b.StorageBudgetError, match='immutable campaign'):
        m.migrate_policy(**f.kwargs)
    assert f.m0.old.state_path.read_bytes() == before
