import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from starVLA.cache.navsim_feature_cache import (
    NavsimFeatureCacheReader,
    RankCacheWriter,
    write_manifest,
)
from tools.precompute_vggt_dense_cache import (
    build_dense_payload,
    build_patch_geometry,
    load_official_preprocess,
    mirror_vggt_crop_preprocess,
    resolve_lidar_to_ego,
)


def test_vggt_crop_preprocess_preserves_1080p_aspect_ratio(tmp_path):
    path = tmp_path / "synthetic-1080p.jpg"
    Image.new("RGB", (1920, 1080), color=(10, 20, 30)).save(path)

    images = mirror_vggt_crop_preprocess([path])

    assert images.shape == (1, 3, 294, 518)
    assert images.shape[-2:] != (518, 518)
    assert images.shape[-2] // 14 == 21
    assert images.shape[-1] // 14 == 37
    assert abs((518 / 294) - (1920 / 1080)) < 0.02


def test_official_vggt_crop_helper_matches_the_cache_preprocess(tmp_path):
    repo = os.environ.get("VGGT_REPO", "").strip()
    if not repo or not (Path(repo) / "vggt/utils/load_fn.py").is_file():
        pytest.skip("local VGGT_REPO is optional; source load_env.sh to run this contract")
    path = tmp_path / "official-synthetic-1080p.jpg"
    Image.new("RGB", (1920, 1080), color=(40, 50, 60)).save(path)

    official = load_official_preprocess(Path(repo))([str(path)], mode="crop")
    mirrored = mirror_vggt_crop_preprocess([path])

    assert official.shape == mirrored.shape == (1, 3, 294, 518)
    assert official.shape[-2] // 14 == 21
    assert official.shape[-1] // 14 == 37


def test_calibrated_ray_contract_requires_an_explicit_or_processed_ego_chain():
    with pytest.raises(RuntimeError, match="Cannot establish LiDAR→ego"):
        resolve_lidar_to_ego({}, frame_index=3)

    rotation, translation, ray_frame = resolve_lidar_to_ego(
        {
            "glo_images": {},
            "glo_status": {"global_poses": np.zeros((4, 3), dtype=np.float32)},
        },
        frame_index=3,
    )
    np.testing.assert_array_equal(rotation, np.eye(3))
    np.testing.assert_array_equal(translation, np.zeros(3))
    assert ray_frame == "navsim_current_ego_planning_frame"


def test_dense_payload_drops_special_tokens_and_preserves_view_row_col_order():
    patch_grid = torch.tensor([[2, 3], [2, 3], [2, 3]], dtype=torch.int16)
    token_count = 5 + 2 * 3
    last_tokens = torch.arange(3 * token_count * 4, dtype=torch.float32).reshape(
        1, 3, token_count, 4
    )
    rays = []
    for view in range(3):
        rays.append(
            build_patch_geometry(
                patch_h=2,
                patch_w=3,
                intrinsic=np.eye(3, dtype=np.float64),
                distortion=np.zeros(5, dtype=np.float64),
                camera_to_ego_rotation=np.eye(3, dtype=np.float64),
                camera_origin_ego=np.array([view, 0.0, 0.0], dtype=np.float64),
            )
        )

    payload = build_dense_payload(
        last_tokens[0],
        patch_start_idx=5,
        patch_grid_hw=patch_grid,
        patch_geometry=rays,
    )

    assert payload["features"].shape == (18, 4)
    assert payload["features"].dtype == torch.bfloat16
    assert payload["valid_mask"].all()
    assert payload["view_ids"].tolist() == [0] * 6 + [1] * 6 + [2] * 6
    torch.testing.assert_close(
        payload["features"].float()[:6], last_tokens[0, 0, 5:].float()
    )
    assert payload["uv_coords"].shape == (18, 2)
    assert payload["ray_features"].shape == (18, 6)
    torch.testing.assert_close(
        payload["ray_features"][:, 3:].norm(dim=-1),
        torch.ones(18),
    )
    torch.testing.assert_close(payload["patch_grid_hw"], patch_grid)
    assert payload["features"].shape[0] == sum(
        int(h) * int(w) for h, w in patch_grid.tolist()
    )


def test_vggt_dense_component_round_trips_without_mixing_query_cache(tmp_path):
    payload = {
        "features": torch.ones(7, 8, dtype=torch.bfloat16),
        "valid_mask": torch.ones(7, dtype=torch.bool),
        "view_ids": torch.zeros(7, dtype=torch.int16),
        "uv_coords": torch.zeros(7, 2, dtype=torch.float16),
        "ray_features": torch.zeros(7, 6, dtype=torch.float32),
        "patch_grid_hw": torch.tensor([[1, 7], [0, 0], [0, 0]], dtype=torch.int16),
    }
    with RankCacheWriter(
        tmp_path, "vggt_dense", rank=0, map_size_bytes=8 * 1024 * 1024
    ) as writer:
        writer.put("scene", payload)
    write_manifest(
        tmp_path,
        "vggt_dense",
        {
            "world_size": 1,
            "sample_count": 1,
            "feature_dim": 8,
            "component": "vggt_dense",
            "view_order": ["cam_f0", "cam_l0", "cam_r0"],
            "frame_index": 3,
            "teacher_layer_index": 23,
            "teacher_attention_branch": "full_aggregated_feature",
            "include_special_tokens": False,
            "spatial_pooling": None,
            "preprocess": {"mode": "crop"},
            "datalist_sha256": "test",
        },
    )

    reader = NavsimFeatureCacheReader(
        tmp_path, components=("vggt_dense",), strict=True
    )
    loaded = reader.get("vggt_dense", 0, "scene")
    assert loaded is not None
    assert loaded["features"].shape == (7, 8)
    assert not (tmp_path / "vggt_query").exists()


def test_dense_manifest_contract_is_not_v2_contract(tmp_path):
    manifest = {
        "component": "vggt_dense",
        "teacher_layer_index": 23,
        "teacher_layer": "aggregator[-1]",
        "teacher_attention_branch": "full_aggregated_feature",
        "include_special_tokens": False,
        "spatial_pooling": None,
        "preprocess": {
            "mode": "crop",
            "target_long_side": 518,
            "patch_size": 14,
            "preserve_aspect_ratio": True,
        },
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["teacher_layer_index"] == 23
    assert loaded["include_special_tokens"] is False
    assert loaded["spatial_pooling"] is None
    assert "pooled_spatial_grid" not in loaded
