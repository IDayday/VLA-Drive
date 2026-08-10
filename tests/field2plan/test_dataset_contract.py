import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from starVLA.dataloader.field2plan_cache import (
    atomic_write_json,
    atomic_write_npz_compressed,
    hash_tokens,
    sha256_file,
)
from starVLA.dataloader.navsim_dataset import NavSimDataset


def _dataset(
    tmp_path, split="test", geometry_cache=None, dynamics_cache=None
) -> NavSimDataset:
    datalist = tmp_path / "tokens.json"
    tokens = ["token-a"] if geometry_cache or dynamics_cache else []
    datalist.write_text(json.dumps(tokens), encoding="utf-8")
    data_cfg = OmegaConf.create(
        {
            "data_root": str(tmp_path),
            "w_neg_traj": None,
            "act_norm": 1,
        }
    )
    all_cfg = OmegaConf.create(
        {
            "field2plan": {
                "enabled": True,
                "proposal": {
                    "source": "cache",
                    "cache_dir": None,
                    "cache_splits": ["train"],
                },
                "camera": {
                    "frame_index": 3,
                    "raw_image_hw": [1080, 1920],
                    "output_image_hw": [576, 1024],
                    "assume_lidar_is_planning_ego": True,
                },
                "geometry": {
                    "teacher_type": "da3" if geometry_cache else "none",
                    "cache_dir": str(geometry_cache) if geometry_cache else None,
                    "cache_splits": [split],
                    "manifest_sha256": None,
                    "supervision": {"enabled": bool(geometry_cache)},
                },
                "dynamics": {
                    "enabled": bool(dynamics_cache),
                    "frame_interval_s": 0.5,
                    "history_frame_indices": [0, 1, 2, 3],
                    "future_frame_indices": list(range(4, 12)),
                    "teacher": {
                        "type": "vjepa2_1" if dynamics_cache else "none",
                        "cache_dir": str(dynamics_cache) if dynamics_cache else None,
                        "cache_splits": [split],
                        "manifest_sha256": None,
                        "input_image_hw": [384, 384],
                    },
                    "supervision": {"enabled": bool(dynamics_cache)},
                },
            },
            "enable_image_aug": 0,
            "w_depth": 0,
            "doing_s2": 0,
            "vit_pre": 0,
        }
    )
    return NavSimDataset(
        datalist_path=str(datalist),
        split=split,
        video_data_cfg=SimpleNamespace(load_2d_data=0),
        gs_data_cfg=SimpleNamespace(load_3d_data=0),
        reward_data_cfg=SimpleNamespace(load_reward_data=0),
        ver_1225=1,
        dataset_cfg=data_cfg,
        all_cfg=all_cfg,
    )


def test_inference_split_does_not_require_training_draft_cache(tmp_path) -> None:
    dataset = _dataset(tmp_path, split="test")
    assert dataset.draft_cache_reader is None


def test_dataset_camera_contract_scales_k_and_keeps_explicit_frames(tmp_path) -> None:
    dataset = _dataset(tmp_path)
    raw = {"glo_images": {}}
    intrinsics = np.repeat(
        np.asarray(
            [[[1000.0, 0.0, 960.0], [0.0, 1000.0, 540.0], [0.0, 0.0, 1.0]]],
            dtype=np.float32,
        ),
        13,
        axis=0,
    )
    rotations = np.repeat(np.eye(3, dtype=np.float32)[None], 13, axis=0)
    translations = np.zeros((13, 3), dtype=np.float32)
    for name in ("cam_f0", "cam_l0", "cam_r0"):
        raw["glo_images"][name] = {
            "intrinsics": intrinsics.copy(),
            "sensor2lidar_rotations": rotations.copy(),
            "sensor2lidar_translations": translations.copy(),
        }

    camera = dataset._build_field2plan_camera(raw)

    assert camera["view_names"] == ["cam_f0", "cam_l0", "cam_r0"]
    assert camera["frame_index"] == 3
    assert camera["transform_status"] == "explicit_identity_assumption"
    assert camera["intrinsics"].shape == (3, 3, 3)
    assert camera["ego_to_camera"].shape == (3, 4, 4)
    torch.testing.assert_close(
        torch.from_numpy(camera["intrinsics"][:, 0, 2]),
        torch.full((3,), 512.0),
    )
    torch.testing.assert_close(
        torch.from_numpy(camera["intrinsics"][:, 1, 2]),
        torch.full((3,), 288.0),
    )


def test_dataset_temporal_contract_has_explicit_history_future_and_transforms(
    tmp_path,
) -> None:
    dataset = _dataset(tmp_path)
    raw = {"glo_images": {}, "glo_status": {}}
    raw["glo_status"]["global_poses"] = np.stack(
        (
            np.arange(12, dtype=np.float32),
            np.zeros(12, dtype=np.float32),
            np.zeros(12, dtype=np.float32),
        ),
        axis=-1,
    )
    intrinsics = np.repeat(
        np.asarray(
            [[[1000.0, 0.0, 960.0], [0.0, 1000.0, 540.0], [0.0, 0.0, 1.0]]],
            dtype=np.float32,
        ),
        13,
        axis=0,
    )
    rotations = np.repeat(np.eye(3, dtype=np.float32)[None], 13, axis=0)
    translations = np.zeros((13, 3), dtype=np.float32)
    for name in ("cam_f0", "cam_l0", "cam_r0"):
        raw["glo_images"][name] = {
            "intrinsics": intrinsics.copy(),
            "sensor2lidar_rotations": rotations.copy(),
            "sensor2lidar_translations": translations.copy(),
        }

    temporal = dataset._build_field2plan_temporal(raw)

    assert temporal["current_frame_index"] == 3
    np.testing.assert_array_equal(temporal["history_frame_indices"], [0, 1, 2, 3])
    np.testing.assert_array_equal(temporal["future_frame_indices"], np.arange(4, 12))
    assert temporal["global_from_ego"].shape == (12, 4, 4)
    assert temporal["current_from_ego"].shape == (12, 4, 4)
    assert temporal["ego_from_current"].shape == (12, 4, 4)
    np.testing.assert_allclose(temporal["frame_times_s"], (np.arange(12) - 3) * 0.5)
    assert temporal["future_camera"]["intrinsics"].shape == (8, 3, 3, 3)
    assert temporal["future_camera"]["ego_to_camera"].shape == (8, 3, 4, 4)



def _write_geometry_cache(root, split, token, datalist) -> None:
    depth_shape = (3, 4, 6)
    manifest = {
        "schema_version": 1,
        "cache_type": "geometry_teacher",
        "status": "complete",
        "teacher": {
            "name": "depth_anything_3_metric_depth",
            "version": "legacy_depth_vis_v1",
            "source_index_sha256": "4" * 64,
        },
        "generator": {"git_commit": "test"},
        "splits": {
            split: {
                "entry_count": 1,
                "tokens_sha256": hash_tokens([token]),
                "datalist_sha256": sha256_file(datalist),
            }
        },
        "tensor_schema": {
            "view_names": ["cam_f0", "cam_l0", "cam_r0"],
            "depth_m": {"dtype": "float32", "shape": list(depth_shape)},
            "confidence": {"dtype": "float32", "shape": list(depth_shape)},
            "valid_mask": {"dtype": "bool", "shape": list(depth_shape)},
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
    atomic_write_json(root / "manifest.json", manifest)
    atomic_write_npz_compressed(
        root / split / f"{token}.npz",
        token=np.asarray(token),
        depth_m=np.ones(depth_shape, dtype=np.float32),
        confidence=np.ones(depth_shape, dtype=np.float32),
        valid_mask=np.ones(depth_shape, dtype=np.bool_),
        source_image_hw=np.asarray([[1080, 1920]] * 3, dtype=np.int64),
        depth_hw=np.asarray([[4, 6]] * 3, dtype=np.int64),
        resize_scale_xy=np.asarray([[6 / 1920, 4 / 1080]] * 3, dtype=np.float32),
    )


def test_dataset_geometry_teacher_is_manifest_validated_and_nested(tmp_path) -> None:
    cache = tmp_path / "geometry-cache"
    datalist = tmp_path / "tokens.json"
    datalist.write_text(json.dumps(["token-a"]), encoding="utf-8")
    _write_geometry_cache(cache, "mini", "token-a", datalist)
    dataset = _dataset(tmp_path, split="mini", geometry_cache=cache)

    teacher = dataset._load_field2plan_geometry("token-a")

    assert teacher["depth_m"].shape == (3, 4, 6)
    assert teacher["view_names"] == ("cam_f0", "cam_l0", "cam_r0")
    assert teacher["coordinate_frame"] == "camera_optical_z_depth_m"


def test_dataset_geometry_supervision_missing_cache_fails_at_init(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="geometry cache manifest"):
        _dataset(tmp_path, split="mini", geometry_cache=tmp_path / "missing")


def _write_dynamics_cache(root, split, token, datalist, input_hw=(384, 384)) -> None:
    feature_shape = (8, 3, 6, 4, 5)
    manifest = {
        "schema_version": 1,
        "cache_type": "dynamics_teacher",
        "status": "complete",
        "teacher": {
            "name": "vjepa2_1",
            "version": "test",
            "repo_commit": "a" * 40,
            "checkpoint_sha256": "b" * 64,
        },
        "generator": {"git_commit": "test"},
        "splits": {
            split: {
                "entry_count": 1,
                "tokens_sha256": hash_tokens([token]),
                "datalist_sha256": sha256_file(datalist),
            }
        },
        "tensor_schema": {
            "view_names": ["cam_f0", "cam_l0", "cam_r0"],
            "features": {"dtype": "float16", "shape": list(feature_shape)},
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
            "input_image_hw": list(input_hw),
            "resize_policy": "center_crop_square_then_bilinear",
            "sha256": "c" * 64,
        },
    }
    atomic_write_json(root / "manifest.json", manifest)
    atomic_write_npz_compressed(
        root / split / f"{token}.npz",
        token=np.asarray(token),
        features=np.ones(feature_shape, dtype=np.float16),
        confidence=np.ones((8, 3, 4, 5), dtype=np.float32),
        valid_mask=np.ones((8, 3, 4, 5), dtype=np.bool_),
        frame_indices=np.arange(4, 12, dtype=np.int64),
        frame_times_s=np.arange(1, 9, dtype=np.float32) * 0.5,
        source_image_hw=np.full((8, 3, 2), [1080, 1920], dtype=np.int64),
        feature_hw=np.full((8, 3, 2), [4, 5], dtype=np.int64),
    )


def test_dataset_dynamics_cache_binds_preprocessing_and_temporal_contract(
    tmp_path,
) -> None:
    datalist = tmp_path / "tokens.json"
    datalist.write_text(json.dumps(["token-a"]), encoding="utf-8")
    cache = tmp_path / "dynamics-cache"
    _write_dynamics_cache(cache, "mini", "token-a", datalist)

    dataset = _dataset(tmp_path, split="mini", dynamics_cache=cache)

    teacher = dataset._load_field2plan_dynamics("token-a")
    assert teacher["features"].shape == (8, 3, 6, 4, 5)
    assert dataset.dynamics_cache_reader.preprocessing_hash == "c" * 64


def test_dataset_rejects_dynamics_preprocessing_camera_mismatch(tmp_path) -> None:
    datalist = tmp_path / "tokens.json"
    datalist.write_text(json.dumps(["token-a"]), encoding="utf-8")
    cache = tmp_path / "dynamics-cache"
    _write_dynamics_cache(cache, "mini", "token-a", datalist, input_hw=(224, 224))

    with pytest.raises(ValueError, match="preprocessing image size"):
        _dataset(tmp_path, split="mini", dynamics_cache=cache)
