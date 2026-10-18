"""Only scope/explicit score-independent smoke prefix may change in diagnostics."""
import hashlib
import json

import pytest

from starVLA.model.modules.vehicle_joint.initialization import identity_hash
from tools.st_world_policy.derive_flow_diagnostic_manifest import derive_manifest


def parent(tmp_path, **changes):
    value = {"schema": "st_flow_training_manifest_v1", "scope": "FORMAL", "split": "train",
             "base_binding_hash": "a"*64, "score_protocol_hash": "b"*64,
             "entries": [{"scene_id": f"scene{i}", "observation_hash": str(i)*64,
                          "source_log": f"log{i//2}"} for i in range(8)],
             "selection": {"seed": 42, "rule": "seeded_hash_before_any_scores"}}
    value.update(changes)
    value["identity"] = identity_hash(value)
    path = tmp_path / "parent.json"
    path.write_text(json.dumps(value))
    return path, hashlib.sha256(path.read_bytes()).hexdigest(), value


def test_profile_full_order_and_all_original_dependencies_retained(tmp_path):
    path, sha, original = parent(tmp_path)
    result = derive_manifest(path, sha, scope="PROFILE")
    assert result["entries"] == original["entries"]
    assert result["diagnostic_derivation"]["parent"]["sha256"] == sha
    assert result["diagnostic_derivation"]["entries_sha256"] == identity_hash(original["entries"])
    assert result["identity"] != original["identity"]
    for key in set(original) - {"scope", "identity"}:
        assert result[key] == original[key]
    assert json.loads(path.read_text()) == original
    assert derive_manifest(path, sha, scope="PROFILE") == result


def test_smoke_has_explicit_prefix_without_changing_members(tmp_path):
    path, sha, original = parent(tmp_path)
    result = derive_manifest(path, sha, scope="SMOKE", max_scenes=3)
    assert result["entries"] == original["entries"][:3]
    assert result["diagnostic_derivation"]["scores_read"] is False


@pytest.mark.parametrize("scope,limit", [("FORMAL", None), ("PROFILE", 4), ("SMOKE", None),
                                          ("SMOKE", 33), ("SMOKE", 9), ("SMOKE", True)])
def test_budget_or_scope_cannot_silently_change(tmp_path, scope, limit):
    path, sha, _ = parent(tmp_path)
    with pytest.raises(ValueError):
        derive_manifest(path, sha, scope=scope, max_scenes=limit)


@pytest.mark.parametrize("change", [{"split": "dev"}, {"scope": "PROFILE"},
                                      {"entries": [{"scene_id": "x"}]*2}])
def test_wrong_role_or_duplicate_parent_refused(tmp_path, change):
    path, sha, _ = parent(tmp_path, **change)
    with pytest.raises(ValueError):
        derive_manifest(path, sha, scope="PROFILE")


def test_changed_content_and_forged_parent_identity_refused(tmp_path):
    path, sha, value = parent(tmp_path)
    value["base_binding_hash"] = "c"*64
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="checksum"):
        derive_manifest(path, sha, scope="PROFILE")
    with pytest.raises(ValueError, match="intact"):
        derive_manifest(path, hashlib.sha256(path.read_bytes()).hexdigest(), scope="PROFILE")
