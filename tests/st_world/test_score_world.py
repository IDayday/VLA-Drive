import copy
import json
from types import SimpleNamespace

import pytest

from starVLA.model.modules.st_world.consequence import fingerprint, METRICS
from tools.st_world_policy.score_world import join_fixed_contexts, read_complete_export, summarize


def test_native_byte_hash_is_distinct_from_deployment_metadata_and_legacy_is_verified(tmp_path):
    from tools.st_world_policy.score_world import native_snapshot_hash
    from tools.st_world_policy.training_state import checkpoint_receipt
    modern = {"sha256": "a" * 64, "native_checkpoint_sha256": "b" * 64,
              "native_receipt_sha256": "c" * 64}
    assert native_snapshot_hash(modern) == "b" * 64
    for value in ({"native_checkpoint_sha256": "b" * 64},
                  dict(modern, native_receipt_sha256="invalid")):
        with pytest.raises(ValueError, match="native checkpoint byte identity"):
            native_snapshot_hash(value)
    run = tmp_path / "run"; folder = run / "checkpoints/step_1"; folder.mkdir(parents=True)
    for name in ("mp_rank_00_model_states.pt", "rng_rank0.pt", "bf16_zero_pp_rank_0_mp_rank_00_optim_states.pt"):
        (folder / name).write_bytes(name.encode())
    receipt = checkpoint_receipt(folder, {"identity": "run", "tag": "step_1", "completed": 1,
                                         "termination_reason": "BUDGET_PAUSED"}, world_size=1)
    legacy = {"run": str(run), "tag": "step_1", "run_identity": "run",
              "receipt_identity": receipt["receipt_identity"], "sha256": "a" * 64}
    assert native_snapshot_hash(legacy) == fingerprint({k: v["sha256"] for k, v in receipt["files"].items()})
    (folder / "rng_rank0.pt").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="corrupt"):
        native_snapshot_hash(legacy)


def test_request_uses_native_hash_as_generator_but_deployment_identity_as_policy_version(tmp_path):
    import numpy as np
    from tools.st_world_policy.score_world import make_requests
    from starVLA.model.modules.st_world.consequence import file_sha256
    root = tmp_path / "export"; (root / "predictions").mkdir(parents=True)
    trajectory = np.zeros((8, 3), dtype=np.float32)
    raw = np.zeros((8, 4), dtype=np.float32); raw[:, 3] = 1
    path = root / "predictions/a.npz"; np.savez(path, raw=raw, trajectory=trajectory)
    path.with_suffix(".json").write_text(json.dumps({"identity": "export", "token": "a", "log": "one",
        "status": "VALID_PREDICTION", "sha256": file_sha256(path)}))
    cache = tmp_path / "cache"; cache.mkdir(); ctx = cache / "context"; ctx.write_bytes(b"context")
    export = {"identity": "export", "checkpoint": {"sha256": "a" * 64,
        "native_checkpoint_sha256": "b" * 64, "native_receipt_sha256": "c" * 64,
        "deployment_source_contract": {"source": "d" * 64}},
        **{k: "fixed" for k in ("environment", "seed", "solver", "precision", "scene_noise")}}
    spec = SimpleNamespace(cache_roots=[str(cache)], campaign_id="unit", score_protocol_hash="e" * 64)
    request = make_requests(root, export, [{"token": "a", "log": "one"}],
                            [{"token": "a", "log": "one", "cache_path": str(ctx)}], spec)[0]
    assert request.policy_version == "a" * 64
    assert request.generator_checkpoint_sha256 == "b" * 64


def test_context_join_is_id_and_log_not_row_order_and_never_shrinks():
    scenes = [{"token": "a", "log": "one"}, {"token": "b", "log": "two"}]
    entries = [dict(scenes[1], cache_path="second"), dict(scenes[0], cache_path="first")]
    assert [r["cache_path"] for r in join_fixed_contexts(scenes, entries)] == ["first", "second"]
    for invalid in (entries[:1], entries * 2, [dict(entries[0], log="wrong"), entries[1]]):
        with pytest.raises(ValueError):
            join_fixed_contexts(scenes, invalid)


def test_complete_summary_cannot_drop_missing_labels_or_replace_them_with_zero():
    requests = [SimpleNamespace(identity=lambda: {"scene": "a"}, source_log="log_a"),
                SimpleNamespace(identity=lambda: {"scene": "b"}, source_log="log_b")]
    records = [{"valid": True, "identity": request.identity(), "metrics": {k: value for k in METRICS},
                "metrics_valid": {k: True for k in METRICS}} for request, value in zip(requests, [0., 1.])]
    result = summarize(records, requests)
    assert result["aggregate"]["metrics"]["score"] == .5
    assert result["aggregate"]["zero_count"] == 1 and result["valid_scored"] == 2
    with pytest.raises(ValueError, match="subset"):
        summarize(records[:1], requests)
    invalid = copy.deepcopy(records); invalid[0].update(valid=False, metrics=None)
    with pytest.raises(ValueError, match="Invalid score"):
        summarize(invalid, requests)
    invalid = copy.deepcopy(records); invalid[0]["metrics_valid"]["comfort"] = False
    with pytest.raises(ValueError, match="metric schema"):
        summarize(invalid, requests)
    invalid = copy.deepcopy(records); invalid[0]["metrics"]["score"] = float("nan")
    with pytest.raises(ValueError, match="Nonfinite"):
        summarize(invalid, requests)


def export_fixture(tmp_path):
    scenes = [{"token": "a", "log": "one"}]
    export = {"schema": "st_single_trajectory_export_v1", "observations": {"split": "dev", "identity": "observations"},
              "selected_scenes_sha256": fingerprint(scenes), "requested_scenes": 1, "world_size": 1,
              "candidates_per_scene": 1, "scorer": None, "targets_read": False, "teacher_cache_read": False}
    export["identity"] = fingerprint(export)
    (tmp_path / "IDENTITY.json").write_text(json.dumps(export))
    (tmp_path / "OBSERVATIONS.json").write_text(json.dumps({"identity": "observations", "scenes": scenes}))
    return export, scenes


def test_complete_export_requires_each_shard_receipt(tmp_path):
    export_fixture(tmp_path)
    with pytest.raises(FileNotFoundError):
        read_complete_export(tmp_path)


def test_changed_scene_manifest_and_test_roles_are_rejected(tmp_path):
    export, scenes = export_fixture(tmp_path)
    scenes[0]["log"] = "different_context"
    (tmp_path / "OBSERVATIONS.json").write_text(json.dumps({"identity": "observations", "scenes": scenes}))
    with pytest.raises(ValueError, match="manifest changed"):
        read_complete_export(tmp_path)
    export["observations"]["split"] = "navtest"
    export["identity"] = fingerprint({k: v for k, v in export.items() if k != "identity"})
    (tmp_path / "IDENTITY.json").write_text(json.dumps(export))
    with pytest.raises(ValueError, match="train/dev"):
        read_complete_export(tmp_path)
