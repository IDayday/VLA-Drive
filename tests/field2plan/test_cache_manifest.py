import json
from pathlib import Path

import numpy as np
import pytest

from starVLA.dataloader.field2plan_cache import (
    DraftCacheReader,
    atomic_write_json,
    atomic_write_npz,
    hash_tokens,
    sha256_file,
)
from tools.field2plan.cache_baseline_drafts import build_draft_manifest, validate_cache


def _manifest():
    return {
        "schema_version": 2,
        "cache_type": "baseline_draft",
        "status": "complete",
        "checkpoint": {"sha256": "1" * 64},
        "config": {"sha256": "2" * 64},
        "generator": {"git_commit": "abc123"},
        "inference": {
            "seed": 20260808,
            "steps": 10,
            "num_candidates": 1,
            "world_size": 1,
            "batch_size_per_rank": 2,
            "qwen_forward_mode": "optimized",
        },
        "splits": {
            "train": {
                "entry_count": 1,
                "tokens_sha256": hash_tokens(["token-a"]),
                "datalist_sha256": "3" * 64,
            }
        },
        "tensor_schema": {
            "draft_action": {
                "dtype": "float32",
                "shape": ["M", 8, 4],
                "horizon": 8,
                "last_dim": 4,
            },
            "physical_trajectory": {
                "dtype": "float32",
                "shape": ["M", 8, 3],
            },
        },
        "normalization": {
            "version": "ver_1225_act_norm_1",
            "x_mean": 10.172484,
            "x_std": 8.805105,
            "y_mean": 0.360762,
            "y_std": 2.277741,
            "heading": "sin_cos",
        },
    }


def test_draft_cache_validates_manifest_checksum_and_token(tmp_path) -> None:
    manifest_path = tmp_path / "manifest.json"
    atomic_write_json(manifest_path, _manifest())
    action = np.zeros((1, 8, 4), dtype=np.float32)
    action[..., 3] = 1.0
    atomic_write_npz(
        tmp_path / "train" / "token-a.npz",
        token=np.asarray("token-a"),
        draft_action=action,
        physical_trajectory=np.zeros((1, 8, 3), dtype=np.float32),
    )

    reader = DraftCacheReader(
        str(tmp_path), "train", expected_manifest_sha256=sha256_file(manifest_path)
    )
    loaded = reader.load("token-a")

    np.testing.assert_array_equal(loaded["draft_action"], action)
    assert loaded["physical_trajectory"].shape == (1, 8, 3)
    with pytest.raises(ValueError, match="checksum mismatch"):
        DraftCacheReader(str(tmp_path), "train", expected_manifest_sha256="0" * 64)


def test_draft_cache_binds_full_datalist_or_debug_prefix(tmp_path) -> None:
    datalist = tmp_path / "tokens.json"
    datalist.write_text(json.dumps(["token-a", "token-b"]), encoding="utf-8")
    manifest = _manifest()
    manifest["splits"]["train"]["datalist_sha256"] = sha256_file(datalist)
    atomic_write_json(tmp_path / "manifest.json", manifest)
    reader = DraftCacheReader(str(tmp_path), "train")

    reader.validate_dataset_binding(["token-a"], str(datalist))
    with pytest.raises(ValueError, match="ordered datalist prefix"):
        reader.validate_dataset_binding(["token-b"], str(datalist))
    with pytest.raises(ValueError, match="count/hash"):
        reader.validate_dataset_binding(["token-a", "token-b"], str(datalist))


def test_draft_cache_rejects_datalist_checksum_mismatch(tmp_path) -> None:
    datalist = tmp_path / "tokens.json"
    datalist.write_text(json.dumps(["token-a"]), encoding="utf-8")
    atomic_write_json(tmp_path / "manifest.json", _manifest())
    with pytest.raises(ValueError, match="datalist checksum"):
        DraftCacheReader(str(tmp_path), "train").validate_dataset_binding(
            ["token-a"], str(datalist)
        )


def test_draft_cache_missing_and_corrupt_entries_fail_fast(tmp_path) -> None:
    atomic_write_json(tmp_path / "manifest.json", _manifest())
    reader = DraftCacheReader(str(tmp_path), "train")
    with pytest.raises(FileNotFoundError, match="entry not found"):
        reader.load("missing")

    corrupt = tmp_path / "train" / "broken.npz"
    corrupt.parent.mkdir(parents=True)
    corrupt.write_bytes(b"not-an-npz")
    with pytest.raises(ValueError, match="corrupt"):
        reader.load("broken")


def test_atomic_writers_leave_no_temporary_files(tmp_path) -> None:
    atomic_write_json(tmp_path / "manifest.json", _manifest())
    atomic_write_npz(tmp_path / "train" / "token.npz", token=np.asarray("token"))
    assert not list(tmp_path.rglob("*.tmp-*"))


def test_draft_manifest_requires_reproducibility_metadata(tmp_path) -> None:
    manifest = _manifest()
    del manifest["checkpoint"]["sha256"]
    atomic_write_json(tmp_path / "manifest.json", manifest)
    with pytest.raises(ValueError, match="checkpoint.sha256"):
        DraftCacheReader(str(tmp_path), "train")


def test_manifest_builder_and_full_validation(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    config = tmp_path / "config.yaml"
    datalist = tmp_path / "train_meta.json"
    checkpoint.write_bytes(b"checkpoint")
    config.write_text("framework: QwenOFT\n", encoding="utf-8")
    datalist.write_text(json.dumps(["token-a"]), encoding="utf-8")
    action = np.zeros((2, 8, 4), dtype=np.float32)
    action[..., 3] = 1.0
    physical = np.zeros((2, 8, 3), dtype=np.float32)
    atomic_write_npz(
        tmp_path / "cache" / "train" / "token-a.npz",
        token=np.asarray("token-a"),
        draft_action=action,
        physical_trajectory=physical,
    )
    manifest = build_draft_manifest(
        checkpoint_path=checkpoint,
        config_path=config,
        datalist_path=datalist,
        split="train",
        tokens=["token-a"],
        seed=17,
        inference_steps=10,
        num_candidates=2,
        world_size=1,
        batch_size=4,
        qwen_forward_mode="optimized",
        git_commit="deadbeef",
    )
    atomic_write_json(tmp_path / "cache" / "manifest.json", manifest)

    summary = validate_cache(tmp_path / "cache", "train", ["token-a"])

    assert summary == {"split": "train", "validated_entries": 1}


def test_full_validation_detects_token_set_or_physical_corruption(tmp_path) -> None:
    manifest = _manifest()
    atomic_write_json(tmp_path / "manifest.json", manifest)
    action = np.zeros((1, 8, 4), dtype=np.float32)
    action[..., 3] = 1.0
    atomic_write_npz(
        tmp_path / "train" / "token-a.npz",
        token=np.asarray("token-a"),
        draft_action=action,
        physical_trajectory=np.zeros((1, 7, 3), dtype=np.float32),
    )
    with pytest.raises(ValueError, match="physical_trajectory"):
        validate_cache(tmp_path, "train", ["token-a"])
    with pytest.raises(ValueError, match="token list checksum"):
        validate_cache(tmp_path, "train", ["different-token"])
