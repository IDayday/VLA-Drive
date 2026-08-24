"""Strict loading and validation of GP-SQ3D-Mix training slot statistics."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch


VIEW_ORDER = ["cam_f0", "cam_l0", "cam_r0"]
POOLING_LAYOUT = [3, 6, 10]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_gp_slot_stats(
    stats_root: str | Path,
    *,
    expected_source_cache_manifest_sha256: str | None = None,
    expected_datalist_sha256: str | None = None,
) -> tuple[torch.Tensor, dict]:
    root = Path(stats_root)
    stats_path = root / "gp_sq3dmix_pooled_stats.pt"
    manifest_path = root / "manifest.json"
    if not stats_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(
            f"GP slot stats require {stats_path} and {manifest_path}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        "complete": True,
        "view_order": VIEW_ORDER,
        "pooling_layout": POOLING_LAYOUT,
        "feature_dimension": 2048,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise RuntimeError(
                f"GP slot-stats manifest {key} mismatch: "
                f"expected {value!r}, found {manifest.get(key)!r}"
            )
    if int(manifest.get("sample_count", 0)) <= 0:
        raise RuntimeError("GP slot-stats sample_count must be positive")
    if manifest.get("stats_file_sha256") != sha256_file(stats_path):
        raise RuntimeError("GP slot-stats file SHA256 mismatch")
    if int(manifest.get("schema_version", 1)) >= 2:
        descriptor_path = root / "pooled_scene_descriptors.pt"
        required_v2 = {
            "descriptor_dimension": 128,
            "descriptor_projection_seed": 20260824,
            "descriptor_projection_shape": [2048, 128],
        }
        for key, value in required_v2.items():
            if manifest.get(key) != value:
                raise RuntimeError(
                    f"GP slot-stats manifest {key} mismatch: "
                    f"expected {value!r}, found {manifest.get(key)!r}"
                )
        if not descriptor_path.is_file():
            raise FileNotFoundError(descriptor_path)
        if manifest.get("descriptor_file_sha256") != sha256_file(descriptor_path):
            raise RuntimeError("GP descriptor file SHA256 mismatch")
        shard_count = int(manifest.get("shard_count", 0))
        if shard_count <= 0 or manifest.get("completed_shards") != list(
            range(shard_count)
        ):
            raise RuntimeError("GP slot-stat shards are incomplete")
        projection_sha = manifest.get("descriptor_projection_sha256")
        if not isinstance(projection_sha, str) or len(projection_sha) != 64:
            raise RuntimeError("GP descriptor projection SHA256 is missing")
        token_sha = manifest.get("token_order_sha256")
        if not isinstance(token_sha, str) or len(token_sha) != 64:
            raise RuntimeError("GP descriptor token-order SHA256 is missing")
    for name, expected_value in (
        ("source_cache_manifest_sha256", expected_source_cache_manifest_sha256),
        ("datalist_sha256", expected_datalist_sha256),
    ):
        if expected_value and manifest.get(name) != expected_value:
            raise RuntimeError(f"GP slot-stats {name} does not match the active input")
    payload = torch.load(stats_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or "pooled_feature_slot_mean" not in payload:
        raise RuntimeError("GP slot-stats payload is missing pooled_feature_slot_mean")
    mean = payload["pooled_feature_slot_mean"]
    if mean.shape != (180, 2048) or mean.dtype != torch.float32:
        raise RuntimeError("pooled_feature_slot_mean must be FP32[180,2048]")
    if not torch.isfinite(mean).all():
        raise RuntimeError("pooled_feature_slot_mean must be finite")
    return mean, manifest
