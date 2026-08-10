import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from starVLA.dataloader.grounded_world_cache import (
    ConsequenceCacheReader,
    FutureTargetCacheReader,
    PriorCacheReader,
)
from starVLA.dataloader.field2plan_cache import atomic_write_json, atomic_write_npz


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _manifest(cache_type: str, token: str, datalist: Path) -> dict:
    common = {
        "schema_version": 1,
        "cache_type": cache_type,
        "status": "complete",
        "splits": {
            "train": {
                "entry_count": 1,
                "tokens_sha256": _token_hash([token]),
                "datalist_sha256": _file_hash(datalist),
            }
        },
    }
    if cache_type == "grounded_world_prior":
        common.update(
            teacher={
                "name": "driving_jepa",
                "domain": "driving",
                "checkpoint_sha256": _digest("teacher"),
            },
            temporal={
                "current_frame_index": 3,
                "history_frame_indices": [0, 1, 2, 3],
                "frame_interval_s": 0.5,
            },
            tensor_schema={
                "features": {"shape": [4, 3, 6, 2, 2], "dtype": "float16"},
                "confidence": {"shape": [4, 3, 2, 2], "dtype": "float32"},
            },
        )
    else:
        common.update(
            producer={
                "source": "student_ema",
                "checkpoint_sha256": _digest("stage1"),
                "ema_decay": 0.996,
                "shared_across_teacher_controls": True,
            },
            temporal={"future_frame_indices": list(range(4, 12)), "frame_interval_s": 0.5},
            tensor_schema={
                "features": {"shape": [8, 12, 4, 4], "dtype": "float16"},
                "valid_mask": {"shape": [8, 4, 4], "dtype": "bool"},
            },
        )
    return common


def _token_hash(tokens: list[str]) -> str:
    digest = hashlib.sha256()
    for token in tokens:
        encoded = token.encode()
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_prior_cache_contains_only_current_history_external_features(tmp_path: Path) -> None:
    token = "scene-token"
    datalist = tmp_path / "train.json"
    datalist.write_text(json.dumps([token]))
    root = tmp_path / "prior"
    atomic_write_json(root / "manifest.json", _manifest("grounded_world_prior", token, datalist))
    atomic_write_npz(
        root / "train" / f"{token}.npz",
        token=np.asarray(token),
        features=np.ones((4, 3, 6, 2, 2), dtype=np.float16),
        confidence=np.ones((4, 3, 2, 2), dtype=np.float32),
    )
    reader = PriorCacheReader(root, "train")
    reader.validate_dataset_binding([token], datalist)
    entry = reader.load(token)
    assert entry["features"].shape == (4, 3, 6, 2, 2)
    assert entry["frame_indices"].tolist() == [0, 1, 2, 3]


def test_future_target_rejects_external_teacher_source(tmp_path: Path) -> None:
    token = "scene-token"
    datalist = tmp_path / "train.json"
    datalist.write_text(json.dumps([token]))
    root = tmp_path / "future"
    manifest = _manifest("grounded_world_future_target", token, datalist)
    manifest["producer"]["source"] = "driving_jepa"
    atomic_write_json(root / "manifest.json", manifest)
    with pytest.raises(ValueError, match="student_ema"):
        FutureTargetCacheReader(root, "train")


def test_future_target_cache_is_shared_student_ema(tmp_path: Path) -> None:
    token = "scene-token"
    datalist = tmp_path / "train.json"
    datalist.write_text(json.dumps([token]))
    root = tmp_path / "future"
    atomic_write_json(root / "manifest.json", _manifest("grounded_world_future_target", token, datalist))
    atomic_write_npz(
        root / "train" / f"{token}.npz",
        token=np.asarray(token),
        features=np.ones((8, 12, 4, 4), dtype=np.float16),
        valid_mask=np.ones((8, 4, 4), dtype=np.bool_),
    )
    reader = FutureTargetCacheReader(root, "train")
    reader.validate_dataset_binding([token], datalist)
    assert reader.load(token)["features"].shape == (8, 12, 4, 4)


def test_consequence_cache_forbids_aggregate_epdms(tmp_path: Path) -> None:
    token = "scene-token"
    datalist = tmp_path / "train.json"
    datalist.write_text(json.dumps([token]))
    root = tmp_path / "consequence"
    manifest = {
        "schema_version": 1,
        "cache_type": "grounded_world_consequence",
        "status": "complete",
        "producer": {
            "source": "navsim_physical_components",
            "contains_aggregate_epdms": False,
        },
        "splits": {
            "train": {
                "entry_count": 1,
                "tokens_sha256": _token_hash([token]),
                "datalist_sha256": _file_hash(datalist),
            }
        },
        "tensor_schema": {
            "components": [
                "clearance",
                "ttc",
                "collision",
                "lane_distance",
                "progress",
                "comfort",
            ],
            "physical_trajectories": {"shape": [3, 8, 3], "dtype": "float32"},
            "values": {"shape": [3, 6], "dtype": "float32"},
            "valid_mask": {"shape": [3, 6], "dtype": "bool"},
        },
    }
    atomic_write_json(root / "manifest.json", manifest)
    atomic_write_npz(
        root / "train" / f"{token}.npz",
        token=np.asarray(token),
        physical_trajectories=np.zeros((3, 8, 3), dtype=np.float32),
        values=np.zeros((3, 6), dtype=np.float32),
        valid_mask=np.ones((3, 6), dtype=np.bool_),
    )
    reader = ConsequenceCacheReader(root, "train")
    reader.validate_dataset_binding([token], datalist)
    assert reader.load(token)["values"].shape == (3, 6)

    manifest["producer"]["contains_aggregate_epdms"] = True
    atomic_write_json(root / "manifest.json", manifest)
    with pytest.raises(ValueError, match="EPDMS"):
        ConsequenceCacheReader(root, "train")
