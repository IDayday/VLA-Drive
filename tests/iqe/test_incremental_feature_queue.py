"""Scheduling-only fixtures; real image acceptance is reported separately."""
from dataclasses import replace
import importlib.util
from pathlib import Path

import pytest

from conftest import scene
from iqe.io import atomic_json, digest, file_hash, read_json
from iqe.pipeline import Pipeline


path = Path(__file__).resolve().parents[2] / "scripts/iqe/fp32_cache/prepare_incremental_features.py"
spec = importlib.util.spec_from_file_location("incremental_feature_queue", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def prepare(tmp_path):
    scenes = [scene(0), scene(1, role="stage_val"), scene(2, role="selector_cal")]
    identity = {"scene_ids": [s.scene_id for s in scenes],
                "roles": ["incremental_fit", "stage_val", "selector_cal"]}
    version = tmp_path / digest(identity)
    atomic_json(version / "IDENTITY.json", identity)
    return version, scenes


def commit_log(version, s, status="ok"):
    cache = version / "cache" / s.source_log_id / "metric_cache.pkl"
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(b"UNIT_ONLY")
    rows = [{"token": s.scene_id, "log": s.source_log_id, "status": status,
             "cache_path": str(cache), "sha256": file_hash(cache), "official_reference_hash": "test"}]
    path = version / "records" / (s.source_log_id + ".json")
    atomic_json(path, rows)
    return path


def test_completed_logs_only_and_resume(tmp_path):
    version, scenes = prepare(tmp_path)
    p = commit_log(version, scenes[0])
    commit_log(version, scenes[1], "failed")
    ready, hashes = module.committed_contexts(version, scenes)
    assert [s.scene_id for s in ready] == [scenes[0].scene_id]
    assert ready[0].observation_hash == scenes[0].observation_hash
    assert ready[0].split_role == scenes[0].split_role
    assert hashes == {scenes[0].source_log_id: file_hash(p)}
    commit_log(version, scenes[1])
    commit_log(version, scenes[2])
    resumed, _ = module.committed_contexts(version, scenes, receipt_hashes=hashes)
    fresh, _ = module.committed_contexts(version, scenes)
    assert resumed == fresh
    assert len(resumed) == 3


@pytest.mark.parametrize("change", ["mutation", "removal", "duplicate", "wrong_log", "outside_version", "missing_reference"])
def test_stale_and_invalid_receipts_rejected(tmp_path, change):
    version, scenes = prepare(tmp_path)
    p = commit_log(version, scenes[0])
    _, hashes = module.committed_contexts(version, scenes)
    rows = read_json(p)
    if change == "removal":
        p.unlink()
    elif change == "mutation":
        rows[0]["sha256"] = "changed"
        atomic_json(p, rows)
    else:
        hashes = {}
        if change == "duplicate":
            rows.append(rows[0])
        elif change == "wrong_log":
            rows[0]["log"] = "other_log"
        elif change == "outside_version":
            rows[0]["cache_path"] = str(tmp_path / "unrelated.pkl")
        else:
            rows[0].pop("official_reference_hash")
        atomic_json(p, rows)
    with pytest.raises(ValueError):
        module.committed_contexts(version, scenes, receipt_hashes=hashes)


def test_no_holdout_injection_or_manifest_substitution(tmp_path):
    version, scenes = prepare(tmp_path)
    with pytest.raises(ValueError, match="manifest mismatch"):
        module.committed_contexts(version, scenes[:2])
    with pytest.raises(ValueError, match="role mismatch"):
        module.committed_contexts(version, [replace(scenes[0], split_role="final_test"), *scenes[1:]])


def test_same_final_feature_key_and_no_pointer_publish(tmp_path):
    version, scenes = prepare(tmp_path)
    p = commit_log(version, scenes[0])
    ready, _ = module.committed_contexts(version, scenes)
    final = replace(scenes[0], metric_context_ref=read_json(p)[0]["cache_path"],
                    metric_context_hash=read_json(p)[0]["sha256"])
    fake = type("Locked", (), {"contract": {k: "locked" for k in (
        "query_checkpoint_hash", "tokenizer_hash", "prompt_hash", "transform_hash",
        "normalizer_hash", "camera_time_hash")}})()
    assert Pipeline.feature_record(fake, ready[0]).cache_key == Pipeline.feature_record(fake, final).cache_key
    assert not (tmp_path / "OFFICIAL_SCENES.json").exists()


def test_late_gpu_owner_rejected_without_signals(monkeypatch):
    import os
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    gpus = "0, GPU-zero\n1, GPU-one\n"
    module.check_gpu_owners("cuda", gpu_rows=gpus, process_rows=f"{os.getpid()}, GPU-one\n999999, GPU-zero\n")
    with pytest.raises(ValueError, match="late unrelated GPU"):
        module.check_gpu_owners("cuda", gpu_rows=gpus, process_rows="999999, GPU-one\n")
