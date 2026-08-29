"""Deterministic, sharded feature storage with strict identity checks."""

from __future__ import annotations

import hashlib
import io
import json
import os
import uuid
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import numpy.typing as npt


FEATURE_SCHEMA_VERSION = "wote_debug.v1"
BASE_ANCHOR_FEATURE_SCHEMA_VERSION = "wote_debug_base_anchor.v2"


class FeatureStoreError(RuntimeError):
    """Raised when a cache is unsafe, inconsistent, or incomplete."""


def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    """Return the SHA256 of *path* without loading it into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def stable_array_hash(array: npt.NDArray[Any]) -> str:
    """Hash shape, dtype, and C-order bytes of an array."""

    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(json.dumps(contiguous.shape).encode("ascii"))
    # memoryview.cast rejects arrays with a zero-length dimension even though
    # their canonical C-order byte payload is unambiguously empty.
    if contiguous.size:
        digest.update(memoryview(contiguous).cast("B"))
    return digest.hexdigest()


def stable_json_hash(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def atomic_write_json(path: Path, payload: Any) -> None:
    """Atomically write canonical JSON, refusing an existing target."""

    if path.exists():
        raise FeatureStoreError(f"refusing to overwrite existing file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.partial")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _npy_bytes(array: npt.NDArray[Any]) -> bytes:
    buffer = io.BytesIO()
    np.lib.format.write_array(buffer, np.ascontiguousarray(array), allow_pickle=False)
    return buffer.getvalue()


def write_deterministic_npz(
    path: Path, arrays: Mapping[str, npt.NDArray[Any]]
) -> str:
    """Write byte-reproducible NPZ data and return its SHA256.

    NumPy's convenience writer embeds current ZIP timestamps. This writer fixes
    entry ordering and timestamps so rerunning an identical cache produces the
    same file hash.
    """

    if path.exists():
        raise FeatureStoreError(f"refusing to overwrite existing shard: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.partial")
    try:
        with zipfile.ZipFile(
            temporary,
            mode="x",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=6,
            strict_timestamps=True,
        ) as archive:
            for key in sorted(arrays):
                if not key or "/" in key or "\\" in key:
                    raise FeatureStoreError(f"invalid NPZ array key: {key!r}")
                array = np.asarray(arrays[key])
                if array.dtype.hasobject:
                    raise FeatureStoreError(f"object dtype is forbidden for {key}")
                info = zipfile.ZipInfo(f"{key}.npy", date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o600 << 16
                archive.writestr(info, _npy_bytes(array), compress_type=zipfile.ZIP_DEFLATED)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return sha256_file(path)


@dataclass(frozen=True)
class SceneCacheRecord:
    scene_token: str
    candidate_indices: tuple[int, ...]
    trajectory_hash: str
    label_hash: str | None = None
    candidate_bank_hash: str | None = None

    def validate(self, expected_candidates: int) -> None:
        if not self.scene_token:
            raise FeatureStoreError("scene token must be non-empty")
        expected = tuple(range(expected_candidates))
        if self.candidate_indices != expected:
            raise FeatureStoreError(
                f"candidate index mismatch for {self.scene_token}: "
                f"expected 0..{expected_candidates - 1}"
            )
        hashes = [("trajectory_hash", self.trajectory_hash)]
        if self.label_hash is not None:
            hashes.append(("label_hash", self.label_hash))
        if self.candidate_bank_hash is not None:
            hashes.append(("candidate_bank_hash", self.candidate_bank_hash))
        for label, value in hashes:
            if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
                raise FeatureStoreError(
                    f"{label} for {self.scene_token} is not a SHA256"
                )

    def to_sidecar(self) -> dict[str, Any]:
        """Serialize without inventing a sentinel hash for absent labels."""

        payload: dict[str, Any] = {
            "scene_token": self.scene_token,
            "candidate_indices": list(self.candidate_indices),
            "trajectory_hash": self.trajectory_hash,
        }
        if self.candidate_bank_hash is not None:
            payload["candidate_bank_hash"] = self.candidate_bank_hash
        if self.label_hash is not None:
            payload["label_hash"] = self.label_hash
        return payload


@dataclass(frozen=True)
class CacheIdentity:
    run_id: str
    split: str
    checkpoint_sha256: str
    wote_commit_sha: str
    feature_schema_version: str = FEATURE_SCHEMA_VERSION
    candidate_count: int = 256
    horizon: int = 8
    label_source: str = "published"
    candidate_bank_hash: str | None = None

    def validate(self) -> None:
        if not self.run_id or "/" in self.run_id or ".." in self.run_id:
            raise FeatureStoreError(f"unsafe run_id: {self.run_id!r}")
        if self.split not in {"train", "val", "test", "smoke", "headroom"}:
            raise FeatureStoreError(f"unsupported split: {self.split}")
        if self.candidate_count <= 0 or self.horizon <= 0:
            raise FeatureStoreError("candidate_count and horizon must be positive")
        for label, value in (
            ("checkpoint_sha256", self.checkpoint_sha256),
            ("wote_commit_sha", self.wote_commit_sha),
        ):
            if len(value) != 40 and len(value) != 64:
                raise FeatureStoreError(f"invalid {label}: {value!r}")
        if self.label_source not in {"published", "none"}:
            raise FeatureStoreError(f"unsupported label source: {self.label_source!r}")
        if self.candidate_bank_hash is not None and (
            len(self.candidate_bank_hash) != 64
            or any(char not in "0123456789abcdef" for char in self.candidate_bank_hash)
        ):
            raise FeatureStoreError("candidate_bank_hash is not a SHA256")
        if self.label_source == "none" and self.candidate_bank_hash is None:
            raise FeatureStoreError(
                "label-free caches require an explicit candidate_bank_hash"
            )


class FeatureShardWriter:
    """Write a cache once, shard by shard, then atomically finalize it."""

    def __init__(
        self,
        root: Path,
        identity: CacheIdentity,
        *,
        float32_keys: Sequence[str] = (),
    ):
        identity.validate()
        self.root = root
        self.identity = identity
        self.float32_keys = frozenset(float32_keys)
        if root.exists():
            raise FeatureStoreError(f"refusing existing cache root: {root}")
        root.mkdir(parents=True, exist_ok=False)
        self._shards: list[dict[str, Any]] = []
        self._tokens: set[str] = set()
        atomic_write_json(root / "identity.json", asdict(identity))

    def write_shard(
        self,
        shard_index: int,
        arrays: Mapping[str, npt.NDArray[Any]],
        records: Sequence[SceneCacheRecord],
    ) -> Path:
        if shard_index != len(self._shards):
            raise FeatureStoreError(
                f"shards must be contiguous: expected {len(self._shards)}, got {shard_index}"
            )
        if not records:
            raise FeatureStoreError("empty shards are forbidden")
        scene_count = len(records)
        for record in records:
            record.validate(self.identity.candidate_count)
            if self.identity.label_source == "none":
                if record.label_hash is not None:
                    raise FeatureStoreError(
                        f"label-free cache record unexpectedly has label_hash: {record.scene_token}"
                    )
                if record.candidate_bank_hash != self.identity.candidate_bank_hash:
                    raise FeatureStoreError(
                        f"candidate bank mismatch for label-free record: {record.scene_token}"
                    )
            if record.scene_token in self._tokens:
                raise FeatureStoreError(f"duplicate scene token: {record.scene_token}")

        normalized: dict[str, npt.NDArray[Any]] = {}
        for key, value in arrays.items():
            array = np.asarray(value)
            if array.ndim == 0 or array.shape[0] != scene_count:
                raise FeatureStoreError(
                    f"{key} first dimension {array.shape if array.ndim else 'scalar'} "
                    f"does not match {scene_count} scene records"
                )
            if array.dtype.hasobject:
                raise FeatureStoreError(f"object dtype forbidden for {key}")
            if np.issubdtype(array.dtype, np.floating):
                if not np.isfinite(array).all():
                    raise FeatureStoreError(f"NaN/Inf detected in {key}")
                array = array.astype(
                    np.float32 if key in self.float32_keys else np.float16,
                    copy=False,
                )
            normalized[key] = np.ascontiguousarray(array)

        shard_name = f"shard-{shard_index:05d}.npz"
        shard_path = self.root / shard_name
        shard_sha = write_deterministic_npz(shard_path, normalized)
        sidecar = {
            "schema_version": self.identity.feature_schema_version,
            "shard_index": shard_index,
            "shard_sha256": shard_sha,
            "arrays": {
                key: {"shape": list(value.shape), "dtype": str(value.dtype)}
                for key, value in sorted(normalized.items())
            },
            "records": [record.to_sidecar() for record in records],
            "checkpoint_sha256": self.identity.checkpoint_sha256,
            "wote_commit_sha": self.identity.wote_commit_sha,
        }
        sidecar_path = self.root / f"shard-{shard_index:05d}.json"
        atomic_write_json(sidecar_path, sidecar)
        sidecar_sha = sha256_file(sidecar_path)
        self._shards.append(
            {
                "path": shard_name,
                "sha256": shard_sha,
                "sidecar": sidecar_path.name,
                "sidecar_sha256": sidecar_sha,
                "scene_count": scene_count,
            }
        )
        self._tokens.update(record.scene_token for record in records)
        return shard_path

    def finalize(self) -> Path:
        if not self._shards:
            raise FeatureStoreError("cannot finalize a cache without shards")
        manifest = {
            "identity": asdict(self.identity),
            "scene_count": len(self._tokens),
            "shards": self._shards,
        }
        manifest["logical_content_sha256"] = stable_json_hash(manifest)
        path = self.root / "manifest.json"
        atomic_write_json(path, manifest)
        return path


class FeatureShardReader:
    """Validate and iterate a finalized cache without pickle."""

    def __init__(
        self,
        root: Path,
        expected_identity: CacheIdentity | None = None,
        *,
        verify_shard_hashes: bool = True,
    ):
        self.root = root
        self.verify_shard_hashes = bool(verify_shard_hashes)
        manifest_path = root / "manifest.json"
        if not manifest_path.is_file():
            raise FeatureStoreError(f"missing finalized manifest: {manifest_path}")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if expected_identity is not None and self.manifest.get("identity") != asdict(
            expected_identity
        ):
            raise FeatureStoreError("cache identity mismatch")
        logical = dict(self.manifest)
        claimed = logical.pop("logical_content_sha256", None)
        if claimed != stable_json_hash(logical):
            raise FeatureStoreError("manifest logical hash mismatch")

    def iter_shards(
        self, array_keys: Sequence[str] | None = None
    ) -> Iterator[tuple[dict[str, Any], Mapping[str, Any]]]:
        """Iterate validated shards, optionally decoding only selected arrays.

        Selective decoding is important for Gate2O because the required frozen
        future spatial tokens are deliberately cached but are only consumed by
        the WoTE latent controls, not by every A--J baseline epoch.
        """

        for shard in self.manifest["shards"]:
            path = self.root / shard["path"]
            sidecar_path = self.root / shard["sidecar"]
            if self.verify_shard_hashes and sha256_file(path) != shard["sha256"]:
                raise FeatureStoreError(f"shard hash mismatch: {path}")
            if self.verify_shard_hashes and sha256_file(sidecar_path) != shard["sidecar_sha256"]:
                raise FeatureStoreError(f"sidecar hash mismatch: {sidecar_path}")
            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
            with np.load(path, allow_pickle=False) as archive:
                requested = tuple(archive.files) if array_keys is None else tuple(array_keys)
                missing = sorted(set(requested) - set(archive.files))
                if missing:
                    raise FeatureStoreError(
                        f"requested arrays missing from {path}: {missing}"
                    )
                yield sidecar, {key: archive[key] for key in requested}


def fixed_random_projection(
    input_dim: int, output_dim: int, seed: int = 20260827
) -> npt.NDArray[np.float32]:
    """Create a deterministic, non-trainable Gaussian projection matrix."""

    if input_dim <= 0 or output_dim <= 0:
        raise ValueError("projection dimensions must be positive")
    rng = np.random.default_rng(seed)
    matrix = rng.standard_normal((input_dim, output_dim), dtype=np.float32)
    matrix /= np.sqrt(float(output_dim))
    return matrix
