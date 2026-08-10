import json

import numpy as np
import pytest

from starVLA.dataloader.field2plan_cache import (
    DynamicsCacheReader,
    atomic_write_json,
    atomic_write_npz_compressed,
    hash_tokens,
    sha256_file,
)


def _manifest(datalist_sha: str = "3" * 64):
    return {
        "schema_version": 1,
        "cache_type": "dynamics_teacher",
        "status": "complete",
        "teacher": {
            "name": "vjepa2_1",
            "version": "vitl16_384",
            "repo_commit": "a" * 40,
            "checkpoint_sha256": "b" * 64,
        },
        "generator": {"git_commit": "deadbeef"},
        "splits": {
            "train": {
                "entry_count": 1,
                "tokens_sha256": hash_tokens(["token-a"]),
                "datalist_sha256": datalist_sha,
            }
        },
        "tensor_schema": {
            "view_names": ["cam_f0", "cam_l0", "cam_r0"],
            "features": {"dtype": "float16", "shape": [8, 3, 6, 4, 5]},
            "confidence": {"dtype": "float32", "shape": [8, 3, 4, 5]},
            "valid_mask": {"dtype": "bool", "shape": [8, 3, 4, 5]},
            "frame_indices": {"dtype": "int64", "shape": [8]},
            "frame_times_s": {"dtype": "float32", "shape": [8]},
            "source_image_hw": {"dtype": "int64", "shape": [8, 3, 2]},
            "feature_hw": {"dtype": "int64", "shape": [8, 3, 2]},
        },
        "temporal": {
            "current_frame_index": 3,
            "history_frame_indices": [0, 1, 2, 3],
            "future_frame_indices": list(range(4, 12)),
            "frame_interval_s": 0.5,
            "teacher_temporal_stride": 2,
        },
        "features": {
            "spatial_layout": "per_view_patch_grid",
            "normalization": "l2",
            "projection": {"algorithm": "seeded_orthogonal", "seed": 17},
        },
        "preprocessing": {
            "input_image_hw": [384, 384],
            "resize_policy": "center_crop_square_then_bilinear",
            "sha256": "4" * 64,
        },
    }


def _write_entry(root, shape=(8, 3, 6, 4, 5)) -> None:
    atomic_write_npz_compressed(
        root / "train" / "token-a.npz",
        token=np.asarray("token-a"),
        features=np.ones(shape, dtype=np.float16),
        confidence=np.ones((8, 3, 4, 5), dtype=np.float32),
        valid_mask=np.ones((8, 3, 4, 5), dtype=np.bool_),
        frame_indices=np.arange(4, 12, dtype=np.int64),
        frame_times_s=np.arange(1, 9, dtype=np.float32) * 0.5,
        source_image_hw=np.full((8, 3, 2), [1080, 1920], dtype=np.int64),
        feature_hw=np.full((8, 3, 2), [4, 5], dtype=np.int64),
    )


def test_dynamics_cache_validates_manifest_entry_and_dataset_binding(tmp_path) -> None:
    datalist = tmp_path / "train_meta.json"
    datalist.write_text(json.dumps(["token-a"]), encoding="utf-8")
    manifest = _manifest(sha256_file(datalist))
    atomic_write_json(tmp_path / "manifest.json", manifest)
    _write_entry(tmp_path)
    checksum = sha256_file(tmp_path / "manifest.json")

    reader = DynamicsCacheReader(
        str(tmp_path), "train", expected_manifest_sha256=checksum
    )
    reader.validate_dataset_binding(["token-a"], str(datalist))
    loaded = reader.load("token-a")

    assert loaded["features"].shape == (8, 3, 6, 4, 5)
    assert loaded["view_names"] == ("cam_f0", "cam_l0", "cam_r0")
    assert loaded["future_frame_indices"] == tuple(range(4, 12))


def test_dynamics_cache_missing_corrupt_and_shape_mismatch_fail_fast(tmp_path) -> None:
    atomic_write_json(tmp_path / "manifest.json", _manifest())
    reader = DynamicsCacheReader(str(tmp_path), "train")
    with pytest.raises(FileNotFoundError, match="entry not found"):
        reader.load("missing")

    broken = tmp_path / "train" / "token-a.npz"
    broken.parent.mkdir(parents=True, exist_ok=True)
    broken.write_bytes(b"not-an-npz")
    with pytest.raises(ValueError, match="corrupt dynamics"):
        reader.load("token-a")

    _write_entry(tmp_path, shape=(8, 3, 7, 4, 5))
    with pytest.raises(ValueError, match="features"):
        reader.load("token-a")
