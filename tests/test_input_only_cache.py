from pathlib import Path

from PIL import Image
import torch

from navsim.agents.EpisodeDrive.layers.world_model import encode_path_tensor_batch
from navsim.agents.EpisodeDrive.utils.internvl_preprocess import load_image
from navsim.planning.training.dataset import (
    CacheOnlyDataset,
    drivevla_cached_collate,
    dump_feature_target_to_pickle,
)
from navsim.planning.training.input_only_cache import (
    INPUT_ONLY_CACHE_NAME,
    build_input_only_cache_record,
)


class _Tokenizer:
    padding_side = "left"

    def __call__(self, queries, **_kwargs):
        value = len(queries[0])
        return {
            "input_ids": torch.full((1, 12), value, dtype=torch.long),
            "attention_mask": torch.ones(1, 12, dtype=torch.long),
        }


class _Builder:
    def __init__(self, name):
        self.name = name

    def get_unique_name(self):
        return self.name


def _image(path: Path, size):
    Image.new("RGB", size, color=(23, 45, 67)).save(path)


def _record(tmp_path: Path):
    current = tmp_path / "current.jpg"
    futures = [tmp_path / f"future_{index}.jpg" for index in range(3)]
    _image(current, (640, 360))
    for index, path in enumerate(futures):
        _image(path, (640 + 40 * index, 360))
    current_path = torch.tensor([ord(value) for value in str(current)])
    future_paths, future_lengths = encode_path_tensor_batch(futures)
    features = {
        "history_trajectory": torch.arange(12, dtype=torch.float32).reshape(4, 3),
        "high_command_one_hot": torch.tensor([0.0, 1.0, 0.0, 0.0]),
        "status_feature": torch.arange(8, dtype=torch.float32),
        "image_path_tensor": current_path,
    }
    targets = {
        "trajectory": torch.randn(8, 3),
        "trajectory_long": torch.randn(8, 3),
        "future_image_paths": future_paths,
        "future_image_path_lengths": future_lengths,
        "future_valid_mask": torch.ones(3, dtype=torch.bool),
        "token": "token",
    }
    return features, targets, build_input_only_cache_record(
        features, targets, tokenizer=_Tokenizer()
    )


def test_input_only_record_preserves_data_and_worker_preprocessing(tmp_path: Path):
    original_features, original_targets, record = _record(tmp_path)
    cache_root = tmp_path / "cache"
    token_dir = cache_root / "log" / "token"
    token_dir.mkdir(parents=True)
    dump_feature_target_to_pickle(
        token_dir / f"{INPUT_ONLY_CACHE_NAME}.gz", record
    )
    dataset = CacheOnlyDataset(
        cache_path=str(cache_root),
        feature_builders=[_Builder("unused")],
        target_builders=[_Builder("unused")],
        preprocess_images=True,
        preprocess_future_images=True,
        preprocess_image_dtype="float32",
        pretokenize_inputs=False,
        input_only_cache_name=INPUT_ONLY_CACHE_NAME,
        reject_dynamic_feature_keys=True,
    )
    features, targets = dataset[0]
    torch.testing.assert_close(
        features["history_trajectory"], original_features["history_trajectory"]
    )
    torch.testing.assert_close(targets["trajectory"], original_targets["trajectory"])
    expected_pixels, expected_metadata = load_image(
        tmp_path / "current.jpg", return_tile_metadata=True
    )
    torch.testing.assert_close(features["pixel_values"], expected_pixels)
    torch.testing.assert_close(features["tile_metadata"], expected_metadata)
    assert len(features["future_pixel_values"]) == 3
    assert len(features["future_tile_metadata"]) == 3
    assert features["input_ids"].shape == (12,)
    assert features["attention_mask"].shape == (12,)


def test_cached_collate_preserves_dynamic_future_tile_groups(tmp_path: Path):
    _, _, record = _record(tmp_path)
    cache_root = tmp_path / "cache"
    for token in ("token_a", "token_b"):
        token_dir = cache_root / "log" / token
        token_dir.mkdir(parents=True)
        dump_feature_target_to_pickle(
            token_dir / f"{INPUT_ONLY_CACHE_NAME}.gz", record
        )
    dataset = CacheOnlyDataset(
        cache_path=str(cache_root),
        feature_builders=[],
        target_builders=[],
        preprocess_images=True,
        preprocess_future_images=True,
        preprocess_image_dtype="bfloat16",
        input_only_cache_name=INPUT_ONLY_CACHE_NAME,
        reject_dynamic_feature_keys=True,
    )
    features, targets = drivevla_cached_collate([dataset[0], dataset[1]])
    assert features["pixel_values"].shape[0] == 2
    assert len(features["future_pixel_values"]) == 2
    assert all(len(group) == 3 for group in features["future_pixel_values"])
    assert targets["future_valid_mask"].shape == (2, 3)
