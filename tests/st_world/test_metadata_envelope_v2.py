"""Temporary CPU fixtures: explicit metadata cap migration, no live storage."""
import copy
import hashlib
import json
import time

import pytest

from tools.st_world_policy import cache_storage_budget as b
from tools.st_world_policy import metadata_storage as m
from tests.st_world.test_metadata_storage_guard import fixture, write


def expansion(tmp_path, old_path, checkpoint_path, checkpoint_sha):
    old = b.StorageBudget(old_path)
    before = old.state_path.read_bytes()
    old_sha = hashlib.sha256(old_path.read_bytes()).hexdigest()
    ledger_sha = hashlib.sha256(before).hexdigest()
    new_doc = copy.deepcopy(old.policy)
    package = new_doc["metadata_package"]
    previous_sha = b._digest(package)
    package.update(schema=m.FULL_METADATA_PACKAGE_SCHEMA,
                   maximum_bytes=m.FULL_METADATA_PACKAGE_MAX,
                   previous_package_sha256=previous_sha)
    new_path = tmp_path / "expanded.json"
    new_sha = write(new_path, new_doc)
    inventory_path = tmp_path / "empty_inventory.json"
    inventory_sha = write(inventory_path, {"schema": "st_metadata_pilot_inventory_v1", "roots": []})
    evidence_path = tmp_path / "fresh_process_evidence.json"
    evidence_sha = write(evidence_path, {"fixture_only": True, "live_jobs": []})
    bindings = dict(old_policy_sha256=old_sha, new_policy_sha256=new_sha,
                    old_ledger_sha256=ledger_sha, pilot_inventory_sha256=inventory_sha,
                    checkpoint_sha256=checkpoint_sha)
    now = time.time()
    q = dict(schema="st_metadata_writer_quiescence_v1", owner_role="Lead", status="QUIESCENT",
             bindings=bindings, checked_writer_roots=sorted(set(old.declarations) |
                 {r["root"] for r in old.core["legacy_reservations"]}),
             active_or_unknown_writers=[], queued_writers_disabled=True, future_admission_owner="Lead",
             process_evidence=[dict(path=str(evidence_path), sha256=evidence_sha)],
             checked_at_unix=now - 1, valid_until_unix=now + 3600)
    q_path = tmp_path / "fresh_quiescence.json"
    q_sha = write(q_path, q)
    kwargs = dict(old_policy=old_path, old_policy_sha256=old_sha, new_policy=new_path,
                  new_policy_sha256=new_sha, expected_old_ledger_sha256=ledger_sha,
                  quiescence_receipt=q_path, quiescence_sha256=q_sha,
                  pilot_inventory=inventory_path, pilot_inventory_sha256=inventory_sha,
                  checkpoint_receipt=checkpoint_path, checkpoint_sha256=checkpoint_sha)
    return old, before, new_doc, q, kwargs


def setup(tmp_path):
    f = fixture(tmp_path)
    m.migrate_policy(**f.kwargs)
    return f, expansion(tmp_path, f.new_path, f.checkpoint_path, f.checkpoint_sha)


@pytest.mark.parametrize("fault_stage", [None, "after_pending", "after_snapshot", "after_state", "after_receipt"])
def test_explicit_expansion_preserves_old_state_and_recovers_each_window(tmp_path, fault_stage):
    f, (old, before, new_doc, q, kwargs) = setup(tmp_path)
    old_policy_bytes = f.new_path.read_bytes()
    def fault(stage):
        if stage == fault_stage:
            raise RuntimeError(stage)
    if fault_stage:
        with pytest.raises(RuntimeError, match=fault_stage):
            m.migrate_policy(**kwargs, fault=fault)
    result = m.migrate_policy(**kwargs)
    assert result == m.migrate_policy(**kwargs)
    after, previous = json.loads(old.state_path.read_text()), json.loads(before)
    assert after["claims"] == previous["claims"]
    assert {k: v for k, v in after["core"].items() if k != "metadata_package"} == {
        k: v for k, v in previous["core"].items() if k != "metadata_package"}
    assert m.original_package(after["core"]["metadata_package"]) == previous["core"]["metadata_package"]
    assert after["core"]["metadata_package"]["maximum_bytes"] == 8 * b.GIB
    assert f.new_path.read_bytes() == old_policy_bytes
    assert len(after["metadata_migrations"]) == 2
    with pytest.raises(b.StorageBudgetError, match="different policy"):
        old.initialize()


@pytest.mark.parametrize("mutation", ["v1_raise", "v2_over", "v2_under", "parent_hash", "baseline",
                                      "floor", "campaign_cap", "removed_claim", "unknown_schema"])
def test_invalid_expansion_rejects_without_mutation(tmp_path, mutation):
    f, (old, before, new_doc, q, kwargs) = setup(tmp_path)
    package = new_doc["metadata_package"]
    if mutation == "v1_raise": package["schema"] = m.PACKAGE_SCHEMA
    if mutation == "v2_over": package["maximum_bytes"] += 1
    if mutation == "v2_under": package["maximum_bytes"] -= 1
    if mutation == "parent_hash": package["previous_package_sha256"] = "f" * 64
    if mutation == "baseline":
        package["baseline_policy_sha256"] = "f" * 64
        parent = {k: v for k, v in package.items() if k != "previous_package_sha256"}
        parent.update(schema=m.PACKAGE_SCHEMA, maximum_bytes=m.PACKAGE_MAX)
        package["previous_package_sha256"] = b._digest(parent)
    if mutation == "floor": new_doc["free_reserve_bytes"] = 0
    if mutation == "campaign_cap": new_doc["cache_cap_bytes"] += b.GIB
    if mutation == "removed_claim": new_doc["claims"].pop()
    if mutation == "unknown_schema": package["schema"] = "unreviewed_v3"
    kwargs["new_policy_sha256"] = write(kwargs["new_policy"], new_doc)
    q["bindings"]["new_policy_sha256"] = kwargs["new_policy_sha256"]
    kwargs["quiescence_sha256"] = write(kwargs["quiescence_receipt"], q)
    with pytest.raises(b.StorageBudgetError):
        m.migrate_policy(**kwargs)
    assert old.state_path.read_bytes() == before


def test_v2_cannot_be_initial_package(tmp_path):
    f = fixture(tmp_path)
    package = f.new_doc["metadata_package"]
    package.update(previous_package_sha256=b._digest(package),
                   schema=m.FULL_METADATA_PACKAGE_SCHEMA, maximum_bytes=8 * b.GIB)
    f.kwargs["new_policy_sha256"] = write(f.new_path, f.new_doc)
    with pytest.raises(b.StorageBudgetError, match="baseline not exact"):
        m.migrate_policy(**f.kwargs)
    assert hashlib.sha256(f.old.state_path.read_bytes()).hexdigest() == f.old_ledger_sha


def test_existing_frozen_w_baseline_unchanged_across_metadata_expansion(tmp_path):
    from tests.st_world.test_spatial_w_storage import fixture as w_fixture
    f = w_fixture(tmp_path)
    m.migrate_policy(**f.kwargs)
    old, before, new_doc, _, kwargs = expansion(
        tmp_path, f.new_path, f.kwargs["checkpoint_receipt"], f.kwargs["checkpoint_sha256"])
    result = m.migrate_policy(**kwargs)
    after = json.loads(old.state_path.read_text())
    assert after["core"]["frozen_w_package"] == json.loads(before)["core"]["frozen_w_package"]
    assert after["claims"] == json.loads(before)["claims"]
    assert result["frozen_w_package_reserved_bytes"] > 0
    assert b.StorageBudget(kwargs["new_policy"]).initialize()["guarded_claims"] >= 1
