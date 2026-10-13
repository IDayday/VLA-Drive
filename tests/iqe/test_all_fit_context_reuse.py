from dataclasses import replace
from pathlib import Path

import pytest
import torch

from conftest import scene
from iqe.data.manifests import save_scenes
from iqe.internal_baseline.config import Config
from iqe.internal_baseline.expansion import reuse_context
from iqe.internal_baseline.features import JointCache
from iqe.io import atomic_json, file_hash, read_json


def pair(tmp_path, change=None):
    checkpoint = tmp_path/"base.pt"
    checkpoint.write_bytes(b"explicit synthetic checkpoint for cache-contract test")
    rows = [scene(i) for i in range(3)]
    configs = []
    for name, selected in (("old", rows[:2]), ("new", rows)):
        cfg = Config(name, str(tmp_path/name), "test", str(tmp_path), "test",
                     research_protocol="two_candidate_v1", scorer_dropout=0,
                     fit_limit="all_fit" if name == "new" else 20000)
        if name == "new" and change:
            selected = [replace(s, **change) if s.scene_id == "s0" else s for s in selected]
        save_scenes(cfg.root/"scenes.json", selected)
        atomic_json(cfg.root/"BASE_BINDING.json", {
            "experiment_id": name, "config_hash": cfg.identity,
            "checkpoint": str(checkpoint), "checkpoint_hash": file_hash(checkpoint),
            "framework_contract_hash": "framework", "contract_hash": "contract",
            "inference_runtime": {"visual": "legacy_fp32"},
            "scene_manifest_hash": file_hash(cfg.root/"scenes.json")})
        atomic_json(cfg.root/"DATA_PARTITION.json", {"all_assignments": {"g0": "expert_fit", "g1": "scorer_fit"}})
        configs.append(cfg)
    for row in rows[:2]:
        JointCache(configs[0]).write(row, {"world": torch.ones(2, 8), "target_raw": torch.ones(8, 4)})
    return *configs, rows


def test_identical_contexts_reuse_without_duplicate_payload_and_are_idempotent(tmp_path):
    old, new, rows = pair(tmp_path)
    first = reuse_context(old, new, workers=2)
    assert first["count"] == 2 and not first["scores_imported"]
    before = file_hash(new.root/"CONTEXT_REUSE.json")
    assert reuse_context(old, new)["count"] == 2
    assert file_hash(new.root/"CONTEXT_REUSE.json") == before
    for row in rows[:2]:
        src, dst = JointCache(old).path(row), JointCache(new).path(row)
        assert src.stat().st_ino == dst.stat().st_ino
        assert JointCache(new).valid(row)
    assert not JointCache(new).path(rows[2]).exists()


@pytest.mark.parametrize("change", [{"target_id": "wrong-target"}, {"metric_context_hash": "wrong-context"},
                                    {"split_role": "stage_val"}, {"observation_hash": "wrong-observation"}])
def test_scene_target_and_role_changes_rejected_before_copy(tmp_path, change):
    old, new, _ = pair(tmp_path, change)
    with pytest.raises(ValueError, match="scene/target/provenance"):
        reuse_context(old, new)
    assert not (new.root/"features").exists()


def test_corrupt_source_is_not_accepted(tmp_path):
    old, new, rows = pair(tmp_path)
    JointCache(old).path(rows[0]).write_bytes(b"broken")
    with pytest.raises(ValueError, match="corrupt"):
        reuse_context(old, new, workers=1)
    assert not (new.root/"CONTEXT_REUSE.json").exists()


def test_different_runtime_cannot_reuse_old_features(tmp_path):
    old, new, _ = pair(tmp_path)
    path = new.root/"BASE_BINDING.json"
    b = read_json(path)
    b["inference_runtime"] = {"visual": "bf16_compute"}
    atomic_json(path, b)
    with pytest.raises(ValueError, match="Base/runtime mismatch"):
        reuse_context(old, new)


def test_interruption_after_payload_link_recovers_without_overwrite(tmp_path):
    import os
    old, new, rows = pair(tmp_path)
    path = JointCache(new).path(rows[0])
    path.parent.mkdir(parents=True)
    os.link(JointCache(old).path(rows[0]), path)
    assert not Path(str(path)+".COMPLETE.json").exists()
    assert reuse_context(old, new)["count"] == 2
    assert JointCache(new).valid(rows[0])
