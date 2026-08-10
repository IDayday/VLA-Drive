import json
import pickle

import numpy as np
from PIL import Image

from starVLA.dataloader.field2plan_cache import (
    GeometryCacheReader,
    atomic_write_json,
    atomic_write_npz_compressed,
)
from tools.field2plan.cache_geometry_vggt import (
    build_vggt_manifest,
    load_navsim_vggt_inputs,
)


VIEWS = ("cam_f0", "cam_l0", "cam_r0")


def test_load_navsim_vggt_inputs_preserves_view_order_and_metric_rig(tmp_path) -> None:
    image_paths = []
    for index, view in enumerate(VIEWS):
        path = tmp_path / f"{view}.jpg"
        Image.new("RGB", (40, 24), color=(index * 30, 0, 0)).save(path)
        image_paths.append(path)
    metadata = {"glo_images": {}}
    for index, (view, path) in enumerate(zip(VIEWS, image_paths)):
        paths = [str(path)] * 4
        translations = np.zeros((4, 3), dtype=np.float32)
        translations[3] = np.asarray([index * 0.2, 0.0, 1.5])
        metadata["glo_images"][view] = {
            "image_paths": paths,
            "sensor2lidar_translations": translations,
        }
    meta_path = tmp_path / "token-a.pkl"
    with meta_path.open("wb") as stream:
        pickle.dump(metadata, stream)

    inputs = load_navsim_vggt_inputs(
        token="token-a",
        meta_path=meta_path,
        view_names=VIEWS,
        frame_index=3,
        runtime_raw_root=tmp_path,
        trainval_sensor_root=None,
    )

    assert inputs.token == "token-a"
    assert inputs.image_paths == tuple(image_paths)
    np.testing.assert_array_equal(inputs.source_image_hw, [[24, 40]] * 3)
    np.testing.assert_allclose(
        inputs.known_camera_centers_m,
        [[0.0, 0.0, 1.5], [0.2, 0.0, 1.5], [0.4, 0.0, 1.5]],
    )


def test_build_vggt_manifest_is_bound_to_pinned_assets_and_cache_schema(
    tmp_path,
) -> None:
    datalist = tmp_path / "train_meta.json"
    datalist.write_text(json.dumps(["token-a"]), encoding="utf-8")
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_bytes(b"checkpoint")
    repo = tmp_path / "vggt"
    repo.mkdir()
    metadata = tmp_path / "token-a.pkl"
    metadata.write_bytes(b"metadata")
    manifest = build_vggt_manifest(
        datalist_path=datalist,
        split="train",
        tokens=["token-a"],
        view_names=VIEWS,
        output_hw=(4, 6),
        git_commit="project-commit",
        vggt_repo=repo,
        vggt_repo_commit="vggt-commit",
        checkpoint=checkpoint,
        checkpoint_revision="hf-revision",
        metadata_checksums={"token-a": "1" * 64},
        frame_index=3,
        metricization="da3_scale_anchor",
        scale_anchor_root=tmp_path / "da3",
    )
    assert manifest["teacher"]["name"] == "vggt"
    assert manifest["teacher"]["version"].endswith("da3_anchor_v1")
    assert manifest["teacher"]["checkpoint_revision"] == "hf-revision"
    assert manifest["teacher"]["checkpoint_sha256"]
    assert manifest["teacher"]["repo_commit"] == "vggt-commit"
    assert manifest["coordinates"]["metricization"] == (
        "robust_log_median_da3_metric_depth_ratio"
    )
    assert manifest["teacher"]["scale_anchor"]["name"] == (
        "depth_anything_3_metric_depth"
    )
    assert manifest["coordinates"]["confidence_source"] == "teacher_confidence"

    cache = tmp_path / "cache"
    atomic_write_json(cache / "manifest.json", manifest)
    depth = np.ones((3, 4, 6), dtype=np.float32)
    atomic_write_npz_compressed(
        cache / "train" / "token-a.npz",
        token=np.asarray("token-a"),
        depth_m=depth,
        confidence=depth.copy(),
        valid_mask=np.ones_like(depth, dtype=np.bool_),
        source_image_hw=np.asarray([[24, 40]] * 3, dtype=np.int64),
        depth_hw=np.asarray([[4, 6]] * 3, dtype=np.int64),
        resize_scale_xy=np.asarray([[6 / 40, 4 / 24]] * 3, dtype=np.float32),
    )
    loaded = GeometryCacheReader(cache, "train").load("token-a")
    assert loaded["depth_m"].shape == (3, 4, 6)
