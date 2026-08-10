import json
import pickle
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from starVLA.dataloader.field2plan_cache import DynamicsCacheReader, sha256_file


def _tool():
    from tools.field2plan import cache_dynamics_vjepa

    return cache_dynamics_vjepa


def test_load_navsim_vjepa_inputs_preserves_explicit_view_and_time_order(
    tmp_path: Path,
) -> None:
    tool = _tool()
    views = ("cam_f0", "cam_l0", "cam_r0")
    frames = tuple(range(12))
    sensors = tmp_path / "sensors"
    cameras = {}
    for view_index, view in enumerate(views):
        paths = []
        for frame in frames:
            path = sensors / view / f"{frame}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(
                np.full((6, 10, 3), frame + 20 * view_index, dtype=np.uint8)
            ).save(path)
            paths.append(str(path))
        cameras[view] = {"image_paths": paths}
    meta_path = tmp_path / "token-a.pkl"
    with meta_path.open("wb") as stream:
        pickle.dump({"glo_images": cameras}, stream)

    loaded = tool.load_navsim_vjepa_inputs(
        token="token-a",
        meta_path=meta_path,
        view_names=views,
        input_frame_indices=frames,
        runtime_raw_root=tmp_path / "unused",
        trainval_sensor_root=None,
    )

    assert loaded.frame_indices == frames
    assert loaded.view_names == views
    assert loaded.source_image_hw.shape == (12, 3, 2)
    video = loaded.load_rgb()
    assert video.shape == (3, 12, 6, 10, 3)
    assert video[2, 7, 0, 0, 0] == 47


def test_resample_tokens_maps_tubelet_centers_to_future_frames() -> None:
    tool = _tool()
    # [V,Tt,Ht,Wt,C], values encode temporal center 0.5,2.5,...,10.5.
    centers = torch.arange(0.5, 12.0, 2.0)
    tokens = centers.reshape(1, 6, 1, 1, 1).repeat(2, 1, 2, 3, 4)

    resampled = tool.resample_vjepa_tokens(
        tokens,
        token_center_indices=centers,
        target_frame_indices=torch.arange(4, 12),
    )

    assert resampled.shape == (8, 2, 4, 2, 3)
    expected = torch.tensor([4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 10.5])
    torch.testing.assert_close(resampled[:, 0, 0, 0, 0], expected)


def test_manifest_and_entry_are_strictly_readable(tmp_path: Path) -> None:
    tool = _tool()
    tokens = ["token-a"]
    datalist = tmp_path / "train_meta.json"
    datalist.write_text(json.dumps(tokens), encoding="utf-8")
    checkpoint = tmp_path / "teacher.pt"
    checkpoint.write_bytes(b"pinned-local-weight")
    repo = tmp_path / "vjepa2"
    repo.mkdir()
    metadata = tmp_path / "token-a.pkl"
    metadata.write_bytes(b"metadata")

    manifest = tool.build_dynamics_manifest(
        datalist_path=datalist,
        split="train",
        tokens=tokens,
        view_names=("cam_f0", "cam_l0", "cam_r0"),
        feature_channels=6,
        output_hw=(4, 5),
        git_commit="deadbeef",
        vjepa_repo=repo,
        vjepa_repo_commit="a" * 40,
        checkpoint=checkpoint,
        checkpoint_revision="local-test",
        metadata_checksums={"token-a": sha256_file(metadata)},
        model_variant="vjepa2_1_vit_large_384",
        projection_seed=17,
        current_frame_index=3,
        history_frame_indices=(0, 1, 2, 3),
        future_frame_indices=tuple(range(4, 12)),
        frame_interval_s=0.5,
        input_image_hw=(384, 384),
    )
    tool.atomic_write_json(tmp_path / "manifest.json", manifest)
    fingerprint = "f" * 64
    tool.atomic_write_npz(
        tmp_path / "train" / "token-a.npz",
        token=np.asarray("token-a"),
        cache_fingerprint=np.asarray(fingerprint),
        features=np.ones((8, 3, 6, 4, 5), dtype=np.float16),
        confidence=np.ones((8, 3, 4, 5), dtype=np.float32),
        valid_mask=np.ones((8, 3, 4, 5), dtype=np.bool_),
        frame_indices=np.arange(4, 12, dtype=np.int64),
        frame_times_s=np.arange(1, 9, dtype=np.float32) * 0.5,
        source_image_hw=np.full((8, 3, 2), [6, 10], dtype=np.int64),
        feature_hw=np.full((8, 3, 2), [4, 5], dtype=np.int64),
    )

    reader = DynamicsCacheReader(str(tmp_path), "train")
    reader.validate_dataset_binding(tokens, str(datalist))
    assert reader.input_image_hw == (384, 384)
    assert reader.preprocessing_hash == manifest["preprocessing"]["sha256"]
    assert reader.load("token-a")["features"].shape == (8, 3, 6, 4, 5)
    assert tool.existing_entry_is_valid(
        tmp_path / "train" / "token-a.npz",
        token="token-a",
        expected_feature_shape=(8, 3, 6, 4, 5),
        cache_fingerprint=fingerprint,
    )


def test_manifest_rejects_unpinned_or_inconsistent_temporal_contract(
    tmp_path: Path,
) -> None:
    tool = _tool()
    datalist = tmp_path / "train.json"
    datalist.write_text('["token-a"]', encoding="utf-8")
    checkpoint = tmp_path / "teacher.pt"
    checkpoint.write_bytes(b"weight")
    repo = tmp_path / "repo"
    repo.mkdir()

    with pytest.raises(ValueError, match="future_frame_indices"):
        tool.build_dynamics_manifest(
            datalist_path=datalist,
            split="train",
            tokens=["token-a"],
            view_names=("cam_f0",),
            feature_channels=6,
            output_hw=(4, 5),
            git_commit="deadbeef",
            vjepa_repo=repo,
            vjepa_repo_commit="a" * 40,
            checkpoint=checkpoint,
            checkpoint_revision="test",
            metadata_checksums={"token-a": "b" * 64},
            model_variant="vjepa2_1_vit_large_384",
            projection_seed=17,
            current_frame_index=3,
            history_frame_indices=(0, 1, 2, 3),
            future_frame_indices=(3, 4),
            frame_interval_s=0.5,
            input_image_hw=(384, 384),
        )
