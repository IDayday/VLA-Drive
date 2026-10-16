import copy
import json
from types import SimpleNamespace

import pytest

from starVLA.model.modules.st_world.consequence import fingerprint, METRICS
from tools.st_world_policy.score_world import join_fixed_contexts, read_complete_export, summarize


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
