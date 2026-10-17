"""CPU temporary fixtures only: no campaign ledger, data, teacher or GPUs."""
import copy
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import time
from types import SimpleNamespace

import pytest

from tools.st_world_policy import cache_storage_budget as b
from tools.st_world_policy import metadata_storage as m

MIB = 1 << 20


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture(tmp_path, *, nested=False, initial_plan=False):
    old_path, new_path = tmp_path / "v4.json", tmp_path / "v5.json"
    old_claim = {"claim_id": "old", "root": str(tmp_path / "cache" / "old"),
                 "cache_identity": "a" * 64, "output_bytes": MIB, "staging_bytes": MIB, "metadata_bytes": MIB}
    old_doc = {"schema": b.SCHEMA, "campaign": "fixture", "state_dir": str(tmp_path / "ledger"),
               "allowed_roots": [str(tmp_path / "cache")], "cache_cap_bytes": 32 * MIB,
               "free_reserve_bytes": MIB, "ledger_reserve_bytes": 2 * MIB,
               "legacy_reservations": [], "claims": [old_claim]}
    old_sha = write(old_path, old_doc)
    old = b.StorageBudget(old_path)
    handle = old.reserve(old_claim["root"], "a" * 64, output_bytes=MIB, staging_bytes=MIB, metadata_bytes=MIB)
    with handle.writing(8192):
        pass
    old_ledger_sha = hashlib.sha256(old.state_path.read_bytes()).hexdigest()
    pilot = tmp_path / "audit" / "pilot"
    pilot.mkdir(parents=True)
    (pilot / "old.sqlite").write_bytes(b"original immutable pilot")
    inventory_path = tmp_path / "inventory.json"
    inventory_sha = write(inventory_path, {"schema": "st_metadata_pilot_inventory_v1", "roots": [m.tree_inventory(pilot)]})
    root = tmp_path / "cache" / "metadata"
    reg = {"schema": m.REG_SCHEMA, "output_root": str(root),
           "writer_kind": "future_preparation" if nested else "current_index",
           "storage_requirements": {"output_bytes": 0, "staging_bytes": 2 * MIB, "metadata_bytes": MIB},
           "limits": {"max_disk_bytes": 3 * MIB, "max_rss_bytes": 4 * b.GIB, "max_seconds": 1200,
                      "max_depth": 2 if nested else 0, "max_entries": 100},
           "file_patterns": [r"METADATA_ADMISSION\.json", r"metadata\.sqlite(?:-journal)?", r".*\.tmp",
                             r"PLAN\.json", r"COMPLETE\.json", r"log_shards/[0-9a-f]{64}\.json"]}
    if initial_plan:
        root.mkdir(parents=True)
        plan_sha = write(root / "PLAN.json", {"fixture": "registered plan"})
        reg["preexisting_pins"] = [{"relative_path": "PLAN.json", "sha256": plan_sha}]
    reg_path = tmp_path / "registration.json"
    reg_sha = write(reg_path, reg)
    row = {"claim_id": "metadata", "root": str(root), "cache_identity": reg_sha,
           "identity_kind": m.REG_SCHEMA, "registration_path": str(reg_path), "registration_sha256": reg_sha,
           **reg["storage_requirements"]}
    new_doc = copy.deepcopy(old_doc)
    new_doc["allowed_roots"].append(str(pilot))
    new_doc["legacy_reservations"].append({"claim_id": "adopted-pilot", "root": str(pilot),
        "reserved_bytes": MIB, "evidence_path": str(inventory_path), "evidence_sha256": inventory_sha})
    new_doc["claims"].append(row)
    new_doc["metadata_package"] = {"schema": m.PACKAGE_SCHEMA, "maximum_bytes": 4 * b.GIB,
        "baseline_policy_sha256": old_sha, "baseline_ledger_sha256": old_ledger_sha,
        "baseline_claims": {old_claim["root"]: old_claim}, "baseline_legacy_reservations": []}
    new_sha = write(new_path, new_doc)
    now = time.time()
    checkpoint_path = tmp_path / "checkpoint.json"
    checkpoint_sha = write(checkpoint_path, {"schema": "st_checkpoint_storage_commitments_v1", "status": "CURRENT",
        "remaining_bytes": MIB, "checked_at_unix": now - 10, "valid_until_unix": now + 3600})
    evidence_path = tmp_path / "quiescence_processes.json"
    evidence_sha = write(evidence_path, {"fixture": True, "live_jobs": [], "not_real_evidence": True})
    q_path = tmp_path / "quiescence.json"
    bindings = {"old_policy_sha256": old_sha, "new_policy_sha256": new_sha,
                "old_ledger_sha256": old_ledger_sha, "pilot_inventory_sha256": inventory_sha,
                "checkpoint_sha256": checkpoint_sha}
    q = {"schema": "st_metadata_writer_quiescence_v1", "owner_role": "Lead", "status": "QUIESCENT",
         "bindings": bindings, "checked_writer_roots": [old_claim["root"], str(pilot)],
         "active_or_unknown_writers": [], "queued_writers_disabled": True, "future_admission_owner": "Lead",
         "process_evidence": [{"path": str(evidence_path), "sha256": evidence_sha}],
         "checked_at_unix": now - 10, "valid_until_unix": now + 3600}
    q_sha = write(q_path, q)
    kwargs = {"old_policy": old_path, "old_policy_sha256": old_sha, "new_policy": new_path, "new_policy_sha256": new_sha,
              "expected_old_ledger_sha256": old_ledger_sha, "quiescence_receipt": q_path, "quiescence_sha256": q_sha,
              "pilot_inventory": inventory_path, "pilot_inventory_sha256": inventory_sha,
              "checkpoint_receipt": checkpoint_path, "checkpoint_sha256": checkpoint_sha}
    guard = {"policy_path": new_path, "policy_sha256": new_sha, "registration_path": reg_path,
             "registration_sha256": reg_sha, "checkpoint_receipt": checkpoint_path, "checkpoint_sha256": checkpoint_sha}
    return SimpleNamespace(**locals())


def refreshed_new(f):
    f.new_sha = write(f.new_path, f.new_doc)
    f.kwargs["new_policy_sha256"] = f.new_sha
    f.guard["policy_sha256"] = f.new_sha
    f.q["bindings"]["new_policy_sha256"] = f.new_sha
    f.kwargs["quiescence_sha256"] = write(f.q_path, f.q)


def ledger(f):
    return json.loads(f.old.state_path.read_text())


def test_migration_preserves_state_cap_policy_and_old_writer_rejects(tmp_path):
    f = fixture(tmp_path)
    original = ledger(f)
    result = b.StorageBudget.migrate_metadata_policy(**f.kwargs)
    changed = ledger(f)
    assert changed["claims"] == original["claims"]
    assert changed["core"]["cache_cap_bytes"] == original["core"]["cache_cap_bytes"]
    assert changed["core"]["free_reserve_bytes"] == original["core"]["free_reserve_bytes"]
    assert hashlib.sha256(f.old_path.read_bytes()).hexdigest() == f.old_sha
    assert result["metadata_package_reserved_bytes"] == 4 * MIB
    assert m.migrate_policy(**f.kwargs) == result
    with pytest.raises(b.StorageBudgetError, match="different policy"):
        f.old.initialize()
    assert b.StorageBudget(f.new_path).initialize()["guarded_claims"] == 1


@pytest.mark.parametrize("stage", ["after_pending", "after_snapshot", "after_state", "after_receipt"])
def test_migration_fault_windows_are_idempotent(tmp_path, stage):
    f = fixture(tmp_path)
    before = ledger(f)["claims"]
    def fail(name):
        if name == stage:
            raise RuntimeError("injected " + name)
    with pytest.raises(RuntimeError):
        m.migrate_policy(**f.kwargs, fault=fail)
    result = m.migrate_policy(**f.kwargs)
    assert ledger(f)["claims"] == before
    assert len(ledger(f)["metadata_migrations"]) == 1
    assert len(list(f.old.state_dir.glob("*.snapshot.json"))) == 1
    assert result == m.migrate_policy(**f.kwargs)


def _migrate_worker(kwargs):
    return m.migrate_policy(**kwargs)


def test_two_identical_migrations_race_under_one_lock(tmp_path):
    f = fixture(tmp_path)
    with multiprocessing.get_context("spawn").Pool(2) as pool:
        results = pool.map(_migrate_worker, [f.kwargs, f.kwargs])
    assert results[0] == results[1]
    assert len(ledger(f)["metadata_migrations"]) == 1


@pytest.mark.parametrize("key,value", [("status", "ACTIVE"), ("owner_role", "implementation"),
    ("active_or_unknown_writers", [123]), ("queued_writers_disabled", False), ("checked_writer_roots", []),
    ("valid_until_unix", 0), ("process_evidence", [])])
def test_bad_quiescence_rejects_before_any_state_mutation(tmp_path, key, value):
    f = fixture(tmp_path)
    f.q[key] = value
    f.kwargs["quiescence_sha256"] = write(f.q_path, f.q)
    with pytest.raises(b.StorageBudgetError):
        m.migrate_policy(**f.kwargs)
    assert hashlib.sha256(f.old.state_path.read_bytes()).hexdigest() == f.old_ledger_sha
    assert not list(f.old.state_dir.glob("MIGRATION*"))


@pytest.mark.parametrize("mutation", ["cap", "floor", "lost_claim", "pilot_extra", "pilot_changed", "overlap", "wild_root", "package"])
def test_nonadditive_or_unreceipted_migration_rejected(tmp_path, mutation):
    f = fixture(tmp_path)
    if mutation == "cap": f.new_doc["cache_cap_bytes"] += MIB
    if mutation == "floor": f.new_doc["free_reserve_bytes"] = 0
    if mutation == "lost_claim": f.new_doc["claims"] = f.new_doc["claims"][1:]
    if mutation == "pilot_extra": (f.pilot / "new.json").write_text("new")
    if mutation == "pilot_changed": (f.pilot / "old.sqlite").write_text("changed")
    if mutation == "overlap": f.new_doc["claims"][1]["root"] = f.old_claim["root"]
    if mutation == "wild_root": f.new_doc["allowed_roots"].append(str(tmp_path))
    if mutation == "package": f.new_doc["metadata_package"]["maximum_bytes"] = 4 * b.GIB + 1
    refreshed_new(f)
    with pytest.raises(b.StorageBudgetError): m.migrate_policy(**f.kwargs)
    assert hashlib.sha256(f.old.state_path.read_bytes()).hexdigest() == f.old_ledger_sha


def test_whole_package_counts_all_declared_claims_and_pilots(tmp_path):
    f = fixture(tmp_path)
    f.new_doc["metadata_package"]["maximum_bytes"] = 4 * MIB - 1
    refreshed_new(f)
    with pytest.raises(b.StorageBudgetError, match="aggregate metadata"):
        m.migrate_policy(**f.kwargs)


def test_stale_pin_and_changed_evidence_reject(tmp_path):
    f = fixture(tmp_path)
    f.q_path.write_text(f.q_path.read_text() + " ")
    with pytest.raises(b.StorageBudgetError, match="SHA mismatch"):
        m.migrate_policy(**f.kwargs)


def test_guard_precharges_before_io_and_resumes_partial_without_release(tmp_path):
    f = fixture(tmp_path)
    m.migrate_policy(**f.kwargs)
    with pytest.raises(RuntimeError, match="injected"):
        with m.metadata_writing(**f.guard) as ctx:
            assert ledger(f)["claims"][str(f.root)]["charged_bytes"] == 3 * MIB
            ctx.check(path=f.root / "metadata.sqlite", before_commit=True)
            (f.root / "metadata.sqlite").write_bytes(b"partial")
            raise RuntimeError("injected")
    assert not (f.root / "identity.json").exists()
    with m.metadata_writing(**f.guard) as ctx:
        assert (f.root / "metadata.sqlite").read_bytes() == b"partial"
        ctx.check(before_commit=True)
    assert ledger(f)["claims"][str(f.root)]["charged_bytes"] == 3 * MIB


@pytest.mark.parametrize("initial_plan", [False, True])
def test_only_exact_preexisting_plan_allowed(tmp_path, initial_plan):
    f = fixture(tmp_path, initial_plan=initial_plan)
    m.migrate_policy(**f.kwargs)
    if initial_plan:
        with m.metadata_writing(**f.guard) as ctx: ctx.check()
    else:
        f.root.mkdir(parents=True)
        (f.root / "metadata.sqlite").write_text("foreign")
        with pytest.raises(b.StorageBudgetError, match="foreign files"):
            with m.metadata_writing(**f.guard): pytest.fail("admitted foreign cache")


@pytest.mark.parametrize("kind", ["symlink", "hardlink", "unknown", "nested_flat"])
def test_guard_rejects_links_unknown_files_and_flat_subdirectories(tmp_path, kind):
    f = fixture(tmp_path)
    m.migrate_policy(**f.kwargs)
    with m.metadata_writing(**f.guard): pass
    if kind == "symlink": (f.root / "log_shards").symlink_to(tmp_path, target_is_directory=True)
    if kind == "hardlink": os.link(f.pilot / "old.sqlite", f.root / "metadata.sqlite")
    if kind == "unknown": (f.root / "unknown.bin").write_text("bad")
    if kind == "nested_flat": (f.root / "sub").mkdir()
    with pytest.raises(b.StorageBudgetError):
        with m.metadata_writing(**f.guard): pytest.fail("bad output admitted")


def test_nested_before_write_parent_is_checked(tmp_path):
    f = fixture(tmp_path, nested=True)
    m.migrate_policy(**f.kwargs)
    with pytest.raises(b.StorageBudgetError, match="symlink"):
        with m.metadata_writing(**f.guard) as ctx:
            (f.root / "log_shards").symlink_to(tmp_path, target_is_directory=True)
            ctx.check(before_commit=True, path=f.root / "log_shards" / ("a" * 64 + ".json"))
    assert not (tmp_path / ("a" * 64 + ".json")).exists()


@pytest.mark.parametrize("limit", ["disk", "rss", "deadline", "interval", "free"])
def test_runtime_failures_sticky_and_do_not_write_complete(tmp_path, monkeypatch, limit):
    f = fixture(tmp_path)
    m.migrate_policy(**f.kwargs)
    with pytest.raises(b.StorageBudgetError):
        with m.metadata_writing(**f.guard) as ctx:
            if limit == "disk": (f.root / "metadata.sqlite").write_bytes(b"x" * (3 * MIB))
            if limit == "rss": monkeypatch.setattr(m, "process_tree_rss", lambda _: 5 * b.GIB)
            if limit == "deadline": ctx.started -= 1300
            if limit == "interval": ctx.last_check -= 6
            if limit == "free": monkeypatch.setattr(b.shutil, "disk_usage", lambda _: SimpleNamespace(free=MIB))
            with pytest.raises(b.StorageBudgetError): ctx.check(before_commit=True)
            assert ctx.failed
            ctx.check(before_commit=True)
    assert not (f.root / "COMPLETE.json").exists()
    assert str(f.root) in ledger(f)["claims"]


def test_checkpoint_commitments_included_in_free_floor(tmp_path, monkeypatch):
    f = fixture(tmp_path)
    m.migrate_policy(**f.kwargs)
    # 2 MiB ledger + 1 MiB legacy + 3 MiB old + 3 MiB new + 1 MiB
    # floor + 1 MiB checkpoint = 11 MiB. Omit checkpoint => wrongly fits.
    monkeypatch.setattr(b.shutil, "disk_usage", lambda _: SimpleNamespace(free=10*MIB + 8192))
    with pytest.raises(b.StorageBudgetError, match="free-space"):
        with m.metadata_writing(**f.guard): pytest.fail("checkpoint budget omitted")


def test_pinned_parse_cannot_accept_replaced_bytes(tmp_path, monkeypatch):
    path = tmp_path / "receipt.json"
    sha = write(path, {"value": "original"})
    original = m._json
    def parse(raw):
        write(path, {"value": "replacement"})
        return original(raw)
    monkeypatch.setattr(m, "_json", parse)
    value, _ = m.pinned_json(path, sha)
    assert value == {"value": "original"}  # Hash and parse same captured bytes.


def update_registration(f):
    f.reg_sha = write(f.reg_path, f.reg)
    row = f.new_doc['claims'][1]
    row.update(cache_identity=f.reg_sha, registration_sha256=f.reg_sha)
    f.guard['registration_sha256'] = f.reg_sha
    refreshed_new(f)


def test_preexisting_empty_lock_has_exact_raw_pin(tmp_path):
    f = fixture(tmp_path, initial_plan=True)
    lock = f.root / 'PLAN.lock'; lock.write_bytes(b'')
    f.reg['file_patterns'].append(r'PLAN\.lock')
    f.reg['preexisting_pins'].append({'relative_path': 'PLAN.lock', 'sha256': hashlib.sha256(b'').hexdigest()})
    update_registration(f)
    m.migrate_policy(**f.kwargs)
    with m.metadata_writing(**f.guard) as ctx: ctx.check()
    assert lock.read_bytes() == b''


@pytest.mark.parametrize('count', [9, 16, 17])
def test_bounded_plan_receipts_retained_without_expanding_byte_budget(tmp_path, count):
    f = fixture(tmp_path, initial_plan=True)
    f.reg['file_patterns'].append(r'control_[0-9]+\.json')
    for i in range(count - 1):
        relative = f'control_{i}.json'
        sha = write(f.root / relative, {'fixture': True, 'record': i})
        f.reg['preexisting_pins'].append({'relative_path': relative, 'sha256': sha})
    update_registration(f)
    m.migrate_policy(**f.kwargs)
    if count > 16:
        with pytest.raises(b.StorageBudgetError, match='foreign files'):
            with m.metadata_writing(**f.guard):
                pytest.fail('Over-limit receipt set admitted')
        assert not (f.root / 'METADATA_ADMISSION.json').exists()
    else:
        with m.metadata_writing(**f.guard) as ctx:
            ctx.check()
        for pin in f.reg['preexisting_pins']:
            assert hashlib.sha256((f.root / pin['relative_path']).read_bytes()).hexdigest() == pin['sha256']
        state = json.loads(f.old.state_path.read_text())
        assert state['claims'][str(f.root)]['charged_bytes'] == 3 * MIB


@pytest.mark.parametrize('failure', ['changed_hash', 'unlisted_file', 'duplicate_pin'])
def test_nine_plan_receipts_still_require_exact_complete_pins(tmp_path, failure):
    f = fixture(tmp_path, initial_plan=True)
    f.reg['file_patterns'].append(r'control_[0-9]+\.json')
    for i in range(8):
        relative = f'control_{i}.json'
        sha = write(f.root / relative, {'fixture': True, 'record': i})
        f.reg['preexisting_pins'].append({'relative_path': relative, 'sha256': sha})
    if failure == 'changed_hash':
        f.reg['preexisting_pins'][-1]['sha256'] = '0' * 64
    elif failure == 'unlisted_file':
        f.reg['preexisting_pins'].pop()
    else:
        f.reg['preexisting_pins'][-1] = dict(f.reg['preexisting_pins'][0])
    update_registration(f)
    m.migrate_policy(**f.kwargs)
    with pytest.raises(b.StorageBudgetError):
        with m.metadata_writing(**f.guard):
            pytest.fail('Invalid receipt set admitted')
    assert not (f.root / 'METADATA_ADMISSION.json').exists()


def test_new_metadata_policy_cannot_initialize_a_second_ledger(tmp_path):
    f = fixture(tmp_path)
    # A new policy on the same existing v4 ledger needs migration first.
    with pytest.raises(b.StorageBudgetError, match='different policy'):
        b.StorageBudget(f.new_path).initialize()
    f.old.state_path.unlink()  # Temporary test fixture only.
    with pytest.raises(b.StorageBudgetError, match='explicit migration'):
        b.StorageBudget(f.new_path).initialize()


def test_pilot_changed_in_pending_window_rejects_resume(tmp_path):
    f = fixture(tmp_path)
    def fault(stage):
        if stage == 'after_pending': raise RuntimeError('injected')
    with pytest.raises(RuntimeError): m.migrate_policy(**f.kwargs, fault=fault)
    (f.pilot / 'old.sqlite').write_text('changed after pending')
    with pytest.raises(b.StorageBudgetError, match='pilot inventory changed'):
        m.migrate_policy(**f.kwargs)
    assert hashlib.sha256(f.old.state_path.read_bytes()).hexdigest() == f.old_ledger_sha


def test_wrong_expected_ledger_sha_rejects(tmp_path):
    f = fixture(tmp_path)
    f.kwargs['expected_old_ledger_sha256'] = 'e' * 64
    with pytest.raises(b.StorageBudgetError): m.migrate_policy(**f.kwargs)


def test_nested_unknown_directory_is_not_an_allowed_output(tmp_path):
    f = fixture(tmp_path, nested=True)
    m.migrate_policy(**f.kwargs)
    with pytest.raises(b.StorageBudgetError, match='unregistered metadata directory'):
        with m.metadata_writing(**f.guard) as ctx:
            (f.root / 'unknown').mkdir()
            ctx.check()


def test_threads_use_same_guard_without_nested_flock(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    f = fixture(tmp_path)
    m.migrate_policy(**f.kwargs)
    with m.metadata_writing(**f.guard) as ctx:
        with ThreadPoolExecutor(2) as pool:
            outputs = list(pool.map(lambda _: m.check_active_metadata_storage(), range(8)))
        assert all(o['process_tree_rss_bytes'] > 0 for o in outputs)
        assert ctx.writer_callbacks == 8
    assert m.check_active_metadata_storage() is None


@pytest.mark.parametrize('wired', [True, False])
def test_thin_runner_calls_actual_parser_and_requires_writer_hooks(tmp_path, monkeypatch, wired):
    f = fixture(tmp_path)
    entry = Path.cwd() / 'tools/st_world_policy/index_teacher_metadata.py'
    f.reg.update(command=['/fixture/python', '-m', 'tools.st_world_policy.index_teacher_metadata',
        'build', '--output', str(f.root)], cwd=str(Path.cwd()), callback_protocol='st_metadata_storage_check_v1',
        source_pins=[{'path': str(entry), 'sha256': hashlib.sha256(entry.read_bytes()).hexdigest()}])
    update_registration(f)
    m.migrate_policy(**f.kwargs)
    calls = []
    class Parser:
        def parse_args(self, args):
            calls.append(args)
            return SimpleNamespace(command='build')
    def run(args):
        if wired:
            m.check_active_metadata_storage()
            m.check_active_metadata_storage(before_commit=True, path=f.root / 'COMPLETE.json')
        return {'status': 'fixture', 'not_real_metadata': True}
    monkeypatch.setattr(m.importlib, 'import_module', lambda name: SimpleNamespace(parser=Parser, run=run,
        __file__=str(entry), METADATA_STORAGE_CALLBACK_PROTOCOL='st_metadata_storage_check_v1',
        METADATA_STORAGE_CALLBACK_DEPENDENCIES=()))
    if wired:
        result = m.run_registered_metadata(**f.guard)
        assert result['status'] == 'GUARDED_METADATA_RETURN'
        assert result['writer_callbacks'] == 2 and result['writer_commit_callbacks'] == 1
    else:
        with pytest.raises(b.StorageBudgetError, match='never called'):
            m.run_registered_metadata(**f.guard)
    assert calls == [['build', '--output', str(f.root)]]


def test_full_declared_package_cannot_be_escaped_by_another_root(tmp_path):
    f = fixture(tmp_path)
    f.new_doc['metadata_package']['maximum_bytes'] = 4 * MIB
    refreshed_new(f)
    m.migrate_policy(**f.kwargs)
    second_reg = copy.deepcopy(f.reg)
    second_reg['output_root'] = str(tmp_path / 'cache' / 'another')
    second_path = tmp_path / 'second_registration.json'
    second_sha = write(second_path, second_reg)
    row = copy.deepcopy(f.new_doc['claims'][1])
    row.update(claim_id='another', root=second_reg['output_root'], cache_identity=second_sha,
               registration_path=str(second_path), registration_sha256=second_sha)
    f.new_doc['claims'].append(row)
    write(f.new_path, f.new_doc)
    with pytest.raises(b.StorageBudgetError, match='aggregate metadata'):
        b.StorageBudget(f.new_path)


def test_nested_guard_refuses_before_recursive_lock(tmp_path):
    f = fixture(tmp_path)
    m.migrate_policy(**f.kwargs)
    with m.metadata_writing(**f.guard):
        with pytest.raises(b.StorageBudgetError, match='nested metadata writers'):
            with m.metadata_writing(**f.guard): pytest.fail('nested flock')


@pytest.mark.parametrize('stage', ['after_pending', 'after_snapshot', 'after_state', 'after_receipt'])
def test_expired_pending_requires_explicit_renewal_without_changing_transition(tmp_path, monkeypatch, stage):
    f = fixture(tmp_path)
    now = time.time()
    f.q['valid_until_unix'] = now + 1
    f.kwargs['quiescence_sha256'] = write(f.q_path, f.q)
    def fault(name):
        if name == stage: raise RuntimeError('injected')
    with pytest.raises(RuntimeError): m.migrate_policy(**f.kwargs, fault=fault)
    current = f.old.state_path.read_bytes()
    pending = next(f.old.state_dir.glob('MIGRATION_*.pending.json'))
    original_pending = pending.read_bytes()
    migration_id = pending.name[len('MIGRATION_'):-len('.pending.json')]
    monkeypatch.setattr(m.time, 'time', lambda: now + 2)
    with pytest.raises(b.StorageBudgetError, match='quiescence expired'):
        m.migrate_policy(**f.kwargs)
    assert f.old.state_path.read_bytes() == current
    renewed = copy.deepcopy(f.q)
    renewed.update(checked_at_unix=now + 1.5, valid_until_unix=now + 3600, resumes_migration_id=migration_id)
    renewed_path = tmp_path / 'renewed_quiescence.json'
    renewed_sha = write(renewed_path, renewed)
    result = m.migrate_policy(**f.kwargs, renew_quiescence_receipt=renewed_path, renew_quiescence_sha256=renewed_sha)
    assert result['migration_id'] == migration_id
    assert ledger(f)['metadata_migrations'] == [migration_id]
    assert pending.read_bytes() == original_pending
    assert len(list(f.old.state_dir.glob('MIGRATION_*.pending.json'))) == 1
    if stage != 'after_receipt': assert result['completion_quiescence_sha256'] == renewed_sha
    assert result == m.migrate_policy(**f.kwargs, renew_quiescence_receipt=renewed_path, renew_quiescence_sha256=renewed_sha)


def test_wrong_renewal_transition_and_active_new_writers_reject(tmp_path):
    f = fixture(tmp_path)
    def fault(stage):
        if stage == 'after_pending': raise RuntimeError('injected')
    with pytest.raises(RuntimeError): m.migrate_policy(**f.kwargs, fault=fault)
    original = f.old.state_path.read_bytes()
    q = copy.deepcopy(f.q); q['resumes_migration_id'] = '0' * 64
    path = tmp_path / 'renew.json'; digest = write(path, q)
    with pytest.raises(b.StorageBudgetError, match='same pending transition'):
        m.migrate_policy(**f.kwargs, renew_quiescence_receipt=path, renew_quiescence_sha256=digest)
    pending = next(f.old.state_dir.glob('*.pending.json'))
    q['resumes_migration_id'] = pending.name[len('MIGRATION_'):-len('.pending.json')]
    q['active_or_unknown_writers'] = [{'pid': 123, 'fixture': True}]
    digest = write(path, q)
    with pytest.raises(b.StorageBudgetError, match='same pending transition'):
        m.migrate_policy(**f.kwargs, renew_quiescence_receipt=path, renew_quiescence_sha256=digest)
    assert f.old.state_path.read_bytes() == original


def test_replacing_main_quiescence_is_not_implicit_pending_renewal(tmp_path):
    f = fixture(tmp_path)
    def fault(stage):
        if stage == 'after_pending': raise RuntimeError('injected')
    with pytest.raises(RuntimeError): m.migrate_policy(**f.kwargs, fault=fault)
    other = copy.deepcopy(f.q); other['checked_at_unix'] += .01
    path = tmp_path / 'other_quiescence.json'; digest = write(path, other)
    kwargs = dict(f.kwargs, quiescence_receipt=path, quiescence_sha256=digest)
    with pytest.raises(b.StorageBudgetError, match='conflicting pending'):
        m.migrate_policy(**kwargs)
    assert len(list(f.old.state_dir.glob('*.pending.json'))) == 1


@pytest.mark.parametrize('patterns,relative,allowed', [
    ([r'log_shards'], 'log_shards/' + 'a' * 64 + '.json', True),
    ([r'a/b'], 'a/b/test.tmp', False),
    ([r'a', r'a/b'], 'a/b/test.tmp', True),
    ([r'a', r'a/b', r'a/b/c'], 'a/b/c/test.tmp', False),
])
def test_planned_parent_chain_and_depth_checked_before_creation(tmp_path, patterns, relative, allowed):
    f = fixture(tmp_path, nested=True)
    f.reg['directory_patterns'] = patterns
    update_registration(f)
    m.migrate_policy(**f.kwargs)
    if allowed:
        with m.metadata_writing(**f.guard) as ctx:
            ctx.check(before_commit=True, path=f.root / relative)
        assert not (f.root / relative).parent.exists()
    else:
        with pytest.raises(b.StorageBudgetError, match='directory'):
            with m.metadata_writing(**f.guard) as ctx:
                ctx.check(before_commit=True, path=f.root / relative)
        assert not (f.root / relative).parent.exists()


@pytest.mark.parametrize('failure', ['missing_capability', 'missing_pin', 'changed_source', 'missing_dependency', 'wrong_loaded_path'])
def test_source_and_capability_fail_before_parser_admission_or_builder(tmp_path, monkeypatch, failure):
    f = fixture(tmp_path)
    source = tmp_path / 'source'; entry = source / 'tools/st_world_policy/index_teacher_metadata.py'
    dependency = source / 'starVLA/dataloader/st_world_metadata_index.py'
    entry.parent.mkdir(parents=True); dependency.parent.mkdir(parents=True)
    entry.write_text('# synthetic source fixture\n'); dependency.write_text('# synthetic dependency\n')
    pins = [{'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in (entry, dependency)]
    f.reg.update(command=['/fixture/python', '-m', 'tools.st_world_policy.index_teacher_metadata',
        'build', '--output', str(f.root)], cwd=str(source), callback_protocol='st_metadata_storage_check_v1', source_pins=pins)
    if failure == 'missing_pin': del f.reg['source_pins']
    if failure == 'missing_dependency': f.reg['source_pins'] = pins[:1]
    update_registration(f); m.migrate_policy(**f.kwargs)
    if failure == 'changed_source': entry.write_text('# changed synthetic source\n')
    monkeypatch.chdir(source)
    def must_not_call(*args): pytest.fail('unapproved parser/builder executed')
    fake = SimpleNamespace(parser=must_not_call, run=must_not_call, __file__=str(entry),
        METADATA_STORAGE_CALLBACK_PROTOCOL='st_metadata_storage_check_v1',
        METADATA_STORAGE_CALLBACK_DEPENDENCIES=('starVLA.dataloader.st_world_metadata_index',))
    if failure == 'missing_capability': del fake.METADATA_STORAGE_CALLBACK_PROTOCOL
    if failure == 'wrong_loaded_path': fake.__file__ = str(dependency)
    monkeypatch.setattr(m.importlib, 'import_module', lambda _: fake)
    with pytest.raises(b.StorageBudgetError): m.run_registered_metadata(**f.guard)
    assert not f.root.exists()
    assert str(f.root) not in ledger(f)['claims']
