from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import pickle
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch

from local_stage2.export_multiview_dino_observation_replay import (
    CAMERA_NAMES,
    _load_current_batch,
    _load_proposal_inventory,
    _loader_log_mapping,
)
from local_stage2.train_independent_scorer import load_private_observation_table


class _FakeLoader:
    def __init__(self, rows):
        self.rows = rows

    def get_agent_input_from_token(self, token):
        return self.rows[token]


def _agent_input(paths):
    cameras = SimpleNamespace(
        **{
            name: SimpleNamespace(image=path)
            for name, path in zip(CAMERA_NAMES, paths)
        }
    )
    status = SimpleNamespace(
        ego_pose=np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
        ego_velocity=np.asarray([1.0, 0.0], dtype=np.float32),
        ego_acceleration=np.asarray([0.0, 0.0], dtype=np.float32),
        driving_command=np.asarray([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
    )
    return SimpleNamespace(cameras=[cameras], ego_statuses=[status])


def test_threaded_image_loading_preserves_rows_and_values(tmp_path: Path) -> None:
    rows = {}
    for row, token in enumerate(("scene_b", "scene_a")):
        paths = []
        for camera, name in enumerate(CAMERA_NAMES):
            path = tmp_path / f"{token}_{name}.png"
            Image.new(
                "RGB",
                (19, 13),
                color=(20 + row * 60, 30 + camera * 25, 40 + row + camera),
            ).save(path)
            paths.append(path)
        rows[token] = _agent_input(paths)
    loader = _FakeLoader(rows)
    tokens = ["scene_b", "scene_a"]
    serial_images, serial_status = _load_current_batch(tokens, loader, (28, 14))
    with ThreadPoolExecutor(max_workers=4) as executor:
        parallel_images, parallel_status = _load_current_batch(
            tokens,
            loader,
            (28, 14),
            executor,
        )
    assert torch.equal(serial_images, parallel_images)
    assert torch.equal(serial_status, parallel_status)
    assert not torch.equal(parallel_images[0], parallel_images[1])


def test_m0_proposal_inventory_is_sorted_without_loading_score_fields(
    tmp_path: Path,
) -> None:
    path = tmp_path / "proposals.pkl"
    with path.open("wb") as stream:
        pickle.dump(
            {
                "scene_b": {"proposals": np.zeros((64, 8, 3))},
                "scene_a": {"proposals": np.ones((64, 8, 3))},
            },
            stream,
        )
    assert _load_proposal_inventory(path) == ["scene_a", "scene_b"]


def test_scene_loader_log_mapping_rejects_duplicate_tokens() -> None:
    valid = SimpleNamespace(
        get_tokens_list_per_log=lambda: {"log_b": ["b"], "log_a": ["a"]}
    )
    assert _loader_log_mapping(valid) == {"a": "log_a", "b": "log_b"}
    duplicate = SimpleNamespace(
        get_tokens_list_per_log=lambda: {"log_a": ["same"], "log_b": ["same"]}
    )
    with pytest.raises(RuntimeError, match="multiple logs"):
        _loader_log_mapping(duplicate)


@pytest.mark.parametrize(
    ("field", "message"),
    (
        ("future_or_evaluator_input", "future/evaluator"),
        ("official_score_or_factor_input", "official score"),
        ("proposal_input", "proposal input"),
        ("drivor_checkpoint_or_representation_used", "DrivOR representation"),
    ),
)
def test_private_observation_lineage_rejects_forbidden_inputs(
    tmp_path: Path,
    field: str,
    message: str,
) -> None:
    shard = tmp_path / "dino_shard_000-of-001"
    shard.mkdir()
    torch.save(
        {
            "tokens": ["scene"],
            "visual_tokens": torch.zeros(1, 4, 8),
            "visual_valid_mask": torch.ones(1, 4, dtype=torch.bool),
            "status_feature": torch.zeros(1, 11),
            "history_trajectory": torch.empty(1, 0),
            "high_command_one_hot": torch.empty(1, 0),
        },
        shard / "chunk_000000.pt",
    )
    manifest = {
        "shard_count": 1,
        "shard_index": 0,
        "scene_count": 1,
        "checkpoint_sha256": "generic-visual-hash",
        "current_observation_only": True,
        "future_or_evaluator_input": False,
        field: True,
    }
    (shard / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match=message):
        load_private_observation_table(tmp_path)
