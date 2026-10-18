"""Exact v2→v3 migration on temporary ledgers; no campaign state or data."""
import copy
import hashlib
import json
import time

import pytest

from tools.st_world_policy import cache_storage_budget as b
from tools.st_world_policy import metadata_storage as m
from tests.st_world.test_metadata_storage_guard import fixture, write
from tests.st_world.test_metadata_envelope_v2 import expansion as expansion_v2


def expansion_v3(tmp_path, old_path, checkpoint_path, checkpoint_sha):
    old = b.StorageBudget(old_path)
    before = old.state_path.read_bytes()
    new_doc = copy.deepcopy(old.policy)
    package = new_doc["metadata_package"]
    package.update(previous_v2_package_sha256=b._digest(package),
                   schema=m.CAUSAL_HISTORY_PACKAGE_SCHEMA,
                   maximum_bytes=m.CAUSAL_HISTORY_PACKAGE_MAX)
    new_path = tmp_path / "expanded_v3.json"
    new_sha = write(new_path, new_doc)
    inventory = tmp_path / "v3_empty_inventory.json"
    inventory_sha = write(inventory, {"schema": "st_metadata_pilot_inventory_v1", "roots": []})
    evidence = tmp_path / "v3_process_fixture.json"
    evidence_sha = write(evidence, {"fixture_only": True, "live_jobs": []})
    bindings = dict(old_policy_sha256=hashlib.sha256(old_path.read_bytes()).hexdigest(),
                    new_policy_sha256=new_sha, old_ledger_sha256=hashlib.sha256(before).hexdigest(),
                    pilot_inventory_sha256=inventory_sha, checkpoint_sha256=checkpoint_sha)
    now = time.time()
    q = dict(schema="st_metadata_writer_quiescence_v1", owner_role="Lead", status="QUIESCENT",
             bindings=bindings, checked_writer_roots=sorted(set(old.declarations) |
                 {r["root"] for r in old.core["legacy_reservations"]}),
             active_or_unknown_writers=[], queued_writers_disabled=True, future_admission_owner="Lead",
             process_evidence=[dict(path=str(evidence), sha256=evidence_sha)],
             checked_at_unix=now - 1, valid_until_unix=now + 3600)
    q_path = tmp_path / "v3_quiescence_fixture.json"
    q_sha = write(q_path, q)
    kwargs = dict(old_policy=old_path, old_policy_sha256=bindings["old_policy_sha256"],
                  new_policy=new_path, new_policy_sha256=new_sha,
                  expected_old_ledger_sha256=bindings["old_ledger_sha256"],
                  quiescence_receipt=q_path, quiescence_sha256=q_sha,
                  pilot_inventory=inventory, pilot_inventory_sha256=inventory_sha,
                  checkpoint_receipt=checkpoint_path, checkpoint_sha256=checkpoint_sha)
    return old, before, new_doc, q, kwargs


def setup(tmp_path, factory=fixture):
    f = factory(tmp_path)
    m.migrate_policy(**f.kwargs)
    *_, v2_kwargs = expansion_v2(tmp_path, f.new_path,
        f.kwargs["checkpoint_receipt"], f.kwargs["checkpoint_sha256"])
    m.migrate_policy(**v2_kwargs)
    return f, expansion_v3(tmp_path, v2_kwargs["new_policy"],
        f.kwargs["checkpoint_receipt"], f.kwargs["checkpoint_sha256"])


@pytest.mark.parametrize("fault_stage", [None, "after_pending", "after_snapshot", "after_state", "after_receipt"])
def test_v3_exact_additive_transition_and_fault_recovery(tmp_path, fault_stage):
    _, (old, before, new_doc, _, kwargs) = setup(tmp_path)
    old_bytes = kwargs["old_policy"].read_bytes()
    old_package = old.core["metadata_package"]
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
    package = after["core"]["metadata_package"]
    assert package["maximum_bytes"] == 10 * b.GIB
    assert package["previous_v2_package_sha256"] == b._digest(old_package)
    assert package["previous_package_sha256"] == old_package["previous_package_sha256"]
    assert m.original_package(package) == m.original_package(old_package)
    assert kwargs["old_policy"].read_bytes() == old_bytes
    assert len(after["metadata_migrations"]) == 3
    b.StorageBudget(kwargs["new_policy"]).initialize()
    with pytest.raises(b.StorageBudgetError, match="different policy"):
        old.initialize()


@pytest.mark.parametrize("mutation", ["over", "under", "float_max", "v2_hash", "v1_hash",
    "baseline_rehashed", "unknown", "v2_raise", "floor", "cap", "removed_claim", "removed_reservation"])
def test_invalid_v3_rejected_without_ledger_mutation(tmp_path, mutation):
    _, (old, before, new_doc, q, kwargs) = setup(tmp_path)
    p = new_doc["metadata_package"]
    if mutation == "over": p["maximum_bytes"] += 1
    if mutation == "under": p["maximum_bytes"] -= 1
    if mutation == "float_max": p["maximum_bytes"] = float(p["maximum_bytes"])
    if mutation == "v2_hash": p["previous_v2_package_sha256"] = "f" * 64
    if mutation == "v1_hash": p["previous_package_sha256"] = "f" * 64
    if mutation == "unknown": p["schema"] = "st_metadata_storage_package_v4"
    if mutation == "v2_raise": p["schema"] = m.FULL_METADATA_PACKAGE_SCHEMA
    if mutation == "baseline_rehashed":
        p["baseline_policy_sha256"] = "f" * 64
        original = {k: v for k, v in p.items() if k not in (
            "previous_package_sha256", "previous_v2_package_sha256")}
        original.update(schema=m.PACKAGE_SCHEMA, maximum_bytes=m.PACKAGE_MAX)
        p["previous_package_sha256"] = b._digest(original)
        previous = {k: v for k, v in p.items() if k != "previous_v2_package_sha256"}
        previous.update(schema=m.FULL_METADATA_PACKAGE_SCHEMA, maximum_bytes=m.FULL_METADATA_PACKAGE_MAX)
        p["previous_v2_package_sha256"] = b._digest(previous)
    if mutation == "floor": new_doc["free_reserve_bytes"] = 0
    if mutation == "cap": new_doc["cache_cap_bytes"] += b.GIB
    if mutation == "removed_claim": new_doc["claims"].pop()
    if mutation == "removed_reservation": new_doc["legacy_reservations"].pop()
    kwargs["new_policy_sha256"] = write(kwargs["new_policy"], new_doc)
    q["bindings"]["new_policy_sha256"] = kwargs["new_policy_sha256"]
    kwargs["quiescence_sha256"] = write(kwargs["quiescence_receipt"], q)
    with pytest.raises(b.StorageBudgetError):
        m.migrate_policy(**kwargs)
    assert old.state_path.read_bytes() == before


@pytest.mark.parametrize("initial", [False, True])
def test_v3_cannot_skip_v2_or_be_initial_package(tmp_path, initial):
    f = fixture(tmp_path)
    if not initial:
        m.migrate_policy(**f.kwargs)
    p = f.new_doc["metadata_package"]
    p.update(previous_package_sha256=b._digest(p),
             schema=m.FULL_METADATA_PACKAGE_SCHEMA, maximum_bytes=m.FULL_METADATA_PACKAGE_MAX)
    p.update(previous_v2_package_sha256=b._digest(p),
             schema=m.CAUSAL_HISTORY_PACKAGE_SCHEMA, maximum_bytes=m.CAUSAL_HISTORY_PACKAGE_MAX)
    if initial:
        before = f.old.state_path.read_bytes()
        f.kwargs["new_policy_sha256"] = write(f.new_path, f.new_doc)
        kwargs = f.kwargs
    else:
        old, before, _, q, kwargs = expansion_v3(tmp_path, f.new_path, f.checkpoint_path, f.checkpoint_sha)
        kwargs["new_policy_sha256"] = write(kwargs["new_policy"], f.new_doc)
        q["bindings"]["new_policy_sha256"] = kwargs["new_policy_sha256"]
        kwargs["quiescence_sha256"] = write(kwargs["quiescence_receipt"], q)
    with pytest.raises(b.StorageBudgetError):
        m.migrate_policy(**kwargs)
    assert f.old.state_path.read_bytes() == before


def test_frozen_w_original_v1_digest_and_claims_survive_v3(tmp_path):
    from tests.st_world.test_spatial_w_storage import fixture as w_fixture
    _, (old, before, _, _, kwargs) = setup(tmp_path, w_fixture)
    result = m.migrate_policy(**kwargs)
    after, previous = json.loads(old.state_path.read_text()), json.loads(before)
    assert after["core"]["frozen_w_package"] == previous["core"]["frozen_w_package"]
    assert after["claims"] == previous["claims"]
    assert b._digest(m.original_package(after["core"]["metadata_package"])) == after["core"]["frozen_w_package"]["metadata_package_sha256"]
    assert result["frozen_w_package_reserved_bytes"] > 0
    b.StorageBudget(kwargs["new_policy"]).initialize()


def test_aggregate_ten_gib_still_enforced(tmp_path):
    _, (old, _, new_doc, _, _) = setup(tmp_path)
    core = copy.deepcopy(old.core)
    core["metadata_package"] = new_doc["metadata_package"]
    used = m.validate_package(core, old.declarations)
    core["legacy_reservations"].append(dict(claim_id="fixture-only-cap", reserved_bytes=10 * b.GIB - used))
    assert m.validate_package(core, old.declarations) == 10 * b.GIB
    core["legacy_reservations"][-1]["reserved_bytes"] += 1
    with pytest.raises(b.StorageBudgetError, match="aggregate metadata package"):
        m.validate_package(core, old.declarations)


@pytest.mark.parametrize("name,ceiling", [("max_disk_bytes", 4 * b.GIB),
    ("max_rss_bytes", 4 * b.GIB), ("max_seconds", 14400)])
def test_per_run_resource_limits_are_not_enlarged(name, ceiling):
    reg = dict(writer_kind="history_preparation", record_layout="packed_log_v1",
               inventory_strategy="immutable_history_shards_v1",
               file_patterns=[r"history_shards/.*"], directory_patterns=["history_shards"],
               limits=dict(max_disk_bytes=4 * b.GIB, max_rss_bytes=4 * b.GIB,
                           max_seconds=14400, max_depth=1, max_entries=9000))
    assert m._resources(reg) == reg["limits"]
    reg["limits"][name] = ceiling + 1
    with pytest.raises(b.StorageBudgetError, match="resource limit"):
        m._resources(reg)
