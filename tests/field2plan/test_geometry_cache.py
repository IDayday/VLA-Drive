import json

import numpy as np
import pytest

from starVLA.dataloader.field2plan_cache import (
    GeometryCacheReader,
    atomic_write_json,
    atomic_write_npz_compressed,
    hash_tokens,
    sha256_file,
)
from tools.field2plan.cache_geometry_da3 import (
    build_geometry_manifest,
    validate_geometry_cache,
)


VIEWS = ("cam_f0", "cam_l0", "cam_r0")


def _manifest():
    return {
        "schema_version": 1,
        "cache_type": "geometry_teacher",
        "status": "complete",
        "teacher": {
            "name": "depth_anything_3_metric_depth",
            "version": "legacy_depth_vis_v1",
            "source_index_sha256": "4" * 64,
        },
        "generator": {"git_commit": "abc123"},
        "splits": {
            "train": {
                "entry_count": 1,
                "tokens_sha256": hash_tokens(["token-a"]),
                "datalist_sha256": "3" * 64,
            }
        },
        "tensor_schema": {
            "view_names": list(VIEWS),
            "depth_m": {"dtype": "float32", "shape": [3, 4, 6]},
            "confidence": {"dtype": "float32", "shape": [3, 4, 6]},
            "valid_mask": {"dtype": "bool", "shape": [3, 4, 6]},
            "source_image_hw": {"dtype": "int64", "shape": [3, 2]},
            "depth_hw": {"dtype": "int64", "shape": [3, 2]},
            "resize_scale_xy": {"dtype": "float32", "shape": [3, 2]},
        },
        "coordinates": {
            "frame": "camera_optical_z_depth_m",
            "frame_index": 3,
            "confidence_source": "finite_positive_validity",
            "resize_policy": "legacy_da3_process_res_252",
        },
    }


def _write_entry(root, token="token-a", depth_shape=(3, 4, 6)) -> None:
    depth = np.full(depth_shape, 8.0, dtype=np.float32)
    confidence = np.ones(depth_shape, dtype=np.float32)
    valid = np.ones(depth_shape, dtype=np.bool_)
    atomic_write_npz_compressed(
        root / "train" / f"{token}.npz",
        token=np.asarray(token),
        depth_m=depth,
        confidence=confidence,
        valid_mask=valid,
        source_image_hw=np.asarray([[1080, 1920]] * 3, dtype=np.int64),
        depth_hw=np.asarray([[4, 6]] * 3, dtype=np.int64),
        resize_scale_xy=np.asarray([[6 / 1920, 4 / 1080]] * 3, dtype=np.float32),
    )


def test_geometry_cache_reader_validates_manifest_and_entry(tmp_path) -> None:
    atomic_write_json(tmp_path / "manifest.json", _manifest())
    _write_entry(tmp_path)
    checksum = sha256_file(tmp_path / "manifest.json")

    loaded = GeometryCacheReader(
        str(tmp_path), "train", expected_manifest_sha256=checksum
    ).load("token-a")

    assert loaded["depth_m"].shape == (3, 4, 6)
    assert loaded["depth_m"].dtype == np.float32
    assert loaded["valid_mask"].dtype == np.bool_
    assert loaded["view_names"] == VIEWS
    assert loaded["coordinate_frame"] == "camera_optical_z_depth_m"


def test_geometry_cache_missing_corrupt_and_shape_mismatch_fail_fast(tmp_path) -> None:
    atomic_write_json(tmp_path / "manifest.json", _manifest())
    reader = GeometryCacheReader(str(tmp_path), "train")
    with pytest.raises(FileNotFoundError, match="entry not found"):
        reader.load("missing")

    broken = tmp_path / "train" / "broken.npz"
    broken.parent.mkdir(parents=True, exist_ok=True)
    broken.write_bytes(b"broken")
    with pytest.raises(ValueError, match="corrupt geometry"):
        reader.load("broken")

    _write_entry(tmp_path, depth_shape=(3, 5, 6))
    with pytest.raises(ValueError, match="depth_m"):
        reader.load("token-a")


def test_geometry_manifest_rejects_incomplete_or_unbound_teacher(tmp_path) -> None:
    manifest = _manifest()
    manifest["status"] = "building"
    atomic_write_json(tmp_path / "manifest.json", manifest)
    with pytest.raises(ValueError, match="complete"):
        GeometryCacheReader(str(tmp_path), "train")

    manifest = _manifest()
    del manifest["teacher"]["source_index_sha256"]
    atomic_write_json(tmp_path / "manifest.json", manifest)
    with pytest.raises(ValueError, match="source_index_sha256"):
        GeometryCacheReader(str(tmp_path), "train")


def test_geometry_manifest_builder_and_full_validation(tmp_path) -> None:
    datalist = tmp_path / "train_meta.json"
    datalist.write_text(json.dumps(["token-a"]), encoding="utf-8")
    source = tmp_path / "legacy"
    source.mkdir()
    (source / "token-a.pkl-depth.pkl").write_bytes(b"source")
    cache = tmp_path / "cache"
    _write_entry(cache)

    manifest = build_geometry_manifest(
        source_root=source,
        datalist_path=datalist,
        split="train",
        tokens=["token-a"],
        view_names=VIEWS,
        depth_shape=(3, 4, 6),
        git_commit="deadbeef",
    )
    atomic_write_json(cache / "manifest.json", manifest)

    assert validate_geometry_cache(cache, "train", ["token-a"]) == {
        "split": "train",
        "validated_entries": 1,
    }


def test_compressed_writer_is_atomic_and_pickle_free(tmp_path) -> None:
    path = tmp_path / "entry.npz"
    atomic_write_npz_compressed(path, value=np.ones((4, 6), dtype=np.float32))
    with np.load(path, allow_pickle=False) as payload:
        assert payload["value"].shape == (4, 6)
    assert not list(tmp_path.rglob("*.tmp-*"))

