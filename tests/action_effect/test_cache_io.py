from __future__ import annotations

from pathlib import Path

import pytest

from research.action_effect.cache_io import (
    CacheConflictError,
    CacheManifest,
    cache_is_reusable,
    finalize_manifest,
    write_json,
)


def manifest(config_hash: str = "config-a") -> CacheManifest:
    return CacheManifest(
        cache_kind="candidate",
        cache_version="v1",
        dataset_version="dataset-v1",
        code_commit="commit+tree.hash",
        config_hash=config_hash,
        evaluator_hash="not_applicable",
        split="train",
        seed=7,
        inputs={"datalist": "hash"},
    )


def test_compatible_complete_cache_is_reused(tmp_path: Path) -> None:
    write_json(tmp_path / "payload.json", {"ok": True})
    finalize_manifest(tmp_path, manifest())
    assert cache_is_reusable(tmp_path, manifest(), ["payload.json"])


def test_different_manifest_never_silently_overwrites(tmp_path: Path) -> None:
    write_json(tmp_path / "payload.json", {"ok": True})
    finalize_manifest(tmp_path, manifest())
    with pytest.raises(CacheConflictError, match="identity differs"):
        cache_is_reusable(tmp_path, manifest("config-b"), ["payload.json"])


def test_populated_manifestless_directory_is_rejected(tmp_path: Path) -> None:
    write_json(tmp_path / "payload.json", {"ok": True})
    with pytest.raises(CacheConflictError, match="no manifest"):
        cache_is_reusable(tmp_path, manifest(), ["payload.json"])
