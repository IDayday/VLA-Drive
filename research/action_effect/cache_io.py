"""Versioned and atomic cache helpers for action-effect research artifacts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


class CacheConflictError(RuntimeError):
    """Raised when an existing cache has a different compatibility identity."""


def canonical_json(value: Any) -> str:
    """Return a deterministic JSON encoding suitable for content hashes."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def content_hash(value: Any) -> str:
    """Hash a JSON-serializable value with SHA-256."""

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Compute the SHA-256 digest of *path* without loading it all in memory."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class CacheManifest:
    """Compatibility and provenance contract stored beside every cache."""

    cache_kind: str
    cache_version: str
    dataset_version: str
    code_commit: str
    config_hash: str
    evaluator_hash: str
    split: str
    seed: int
    inputs: Mapping[str, str]

    def compatibility_identity(self) -> str:
        """Return the stable identity used to accept or reject cache reuse."""

        return content_hash(asdict(self))


def read_manifest(cache_dir: Path) -> CacheManifest | None:
    """Read a manifest, or return ``None`` when the cache does not exist."""

    path = cache_dir / "manifest.json"
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as stream:
        return CacheManifest(**json.load(stream))


def cache_is_reusable(cache_dir: Path, expected: CacheManifest, required: Sequence[str]) -> bool:
    """Validate an existing cache and report whether it can be reused.

    A populated directory without a manifest, or a manifest with a different
    identity, is rejected. This prevents a rerun from silently mixing versions.
    """

    existing = read_manifest(cache_dir)
    if existing is None:
        if cache_dir.exists() and any(cache_dir.iterdir()):
            raise CacheConflictError(f"cache directory is populated but has no manifest: {cache_dir}")
        return False
    if existing.compatibility_identity() != expected.compatibility_identity():
        raise CacheConflictError(
            "existing cache identity differs; choose a new cache directory: "
            f"{cache_dir}\nexisting={existing.compatibility_identity()}\n"
            f"expected={expected.compatibility_identity()}"
        )
    missing = [name for name in required if not (cache_dir / name).is_file()]
    if missing:
        raise CacheConflictError(f"compatible cache is incomplete ({missing}): {cache_dir}")
    return True


def _atomic_replace_bytes(path: Path, payload: bytes) -> None:
    """Atomically replace one file after flushing it to durable storage."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def write_json(path: Path, value: Any) -> None:
    """Write formatted JSON atomically."""

    payload = (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")
    _atomic_replace_bytes(path, payload)


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    """Write JSON Lines atomically."""

    payload = "".join(canonical_json(dict(row)) + "\n" for row in rows).encode("utf-8")
    _atomic_replace_bytes(path, payload)


def write_npz(path: Path, **arrays: np.ndarray) -> None:
    """Write a compressed NPZ atomically."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".npz", dir=path.parent)
    os.close(fd)
    try:
        np.savez_compressed(temporary_name, **arrays)
        with open(temporary_name, "rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def finalize_manifest(cache_dir: Path, manifest: CacheManifest) -> None:
    """Publish the manifest last, marking all other cache files complete."""

    write_json(cache_dir / "manifest.json", asdict(manifest))
