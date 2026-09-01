from __future__ import annotations

from pathlib import Path
import pickle
import gzip
from types import SimpleNamespace

import numpy as np
from omegaconf import OmegaConf
import pytest
import torch

from navsim.agents.EpisodeDrive.drivevla_features import TrajectoryTargetBuilder
from navsim.agents.EpisodeDrive.layers.world_model import (
    decode_path_tensor,
    decode_path_tensor_batch,
    encode_path_tensor,
    encode_path_tensor_batch,
)
from navsim.planning.training.dataset import CacheOnlyDataset


def test_utf8_path_round_trip_and_fixed_shape() -> None:
    paths = ["/tmp/front camera.jpg", "/tmp/未来/前视图.png", "relative/image.jpeg"]
    tensors, lengths = encode_path_tensor_batch(paths)
    assert tensors.shape == (3, 1024)
    assert tensors.dtype == torch.uint8
    assert lengths.shape == (3,)
    assert decode_path_tensor_batch(tensors, lengths) == paths
    for path, tensor, length in zip(paths, tensors, lengths):
        assert decode_path_tensor(tensor, length) == path


class _FakeScene:
    def __init__(self, image_paths) -> None:
        self.scene_metadata = SimpleNamespace(
            num_history_frames=4,
            initial_token="token",
        )
        self.frames = [
            SimpleNamespace(
                cameras=SimpleNamespace(cam_f0=SimpleNamespace(image=path))
            )
            for path in image_paths
        ]

    def get_future_trajectory(self, num_trajectory_frames: int):
        return SimpleNamespace(
            poses=np.zeros((num_trajectory_frames, 3), dtype=np.float32)
        )


def _target_builder():
    action_config = OmegaConf.create(
        {
            "trajectory_sampling": {"num_poses": 8},
            "long_trajectory_additional_poses": -1,
        }
    )
    world_config = OmegaConf.create(
        {"enabled": True, "horizons_sec": [0.5, 1.5, 3.0]}
    )
    return TrajectoryTargetBuilder(action_config, world_config)


def test_target_builder_uses_exact_future_frame_offsets(tmp_path: Path) -> None:
    paths = []
    for index in range(10):
        path = tmp_path / f"frame_{index}.jpg"
        path.touch()
        paths.append(path)
    builder = _target_builder()
    targets = builder.compute_targets(_FakeScene(paths))
    assert builder.get_unique_name() == "trajectory_target_planreg_wm_v1"
    assert targets["future_image_paths"].shape == (3, 1024)
    assert targets["future_image_path_lengths"].shape == (3,)
    assert targets["future_valid_mask"].tolist() == [True, True, True]
    decoded = decode_path_tensor_batch(
        targets["future_image_paths"], targets["future_image_path_lengths"]
    )
    # current_idx=3, then offsets 1/3/6.
    assert decoded == [str(paths[4]), str(paths[6]), str(paths[9])]


def test_future_target_rejects_loaded_rgb_arrays() -> None:
    paths = [None] * 10
    paths[4] = np.zeros((4, 4, 3), dtype=np.uint8)
    with pytest.raises(RuntimeError, match="load_image_path=true"):
        _target_builder().compute_targets(_FakeScene(paths))


def test_stale_trajectory_target_cache_is_not_accepted(tmp_path: Path) -> None:
    cache_root = tmp_path / "cache"
    token_dir = cache_root / "log" / "token"
    token_dir.mkdir(parents=True)
    with gzip.open(token_dir / "dummy_feature.gz", "wb") as stream:
        pickle.dump({"x": torch.tensor(1)}, stream)
    with gzip.open(token_dir / "trajectory_target.gz", "wb") as stream:
        pickle.dump({"trajectory": torch.zeros(8, 3)}, stream)

    feature_builder = SimpleNamespace(get_unique_name=lambda: "dummy_feature")
    dataset = CacheOnlyDataset(
        cache_path=str(cache_root),
        feature_builders=[feature_builder],
        target_builders=[_target_builder()],
    )
    with pytest.raises(FileNotFoundError):
        _ = dataset[0]
