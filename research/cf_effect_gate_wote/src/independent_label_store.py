"""Deterministic independent candidate labels for the fixed 4-second contract."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import numpy.typing as npt

from .feature_store import (
    atomic_write_json,
    sha256_file,
    stable_array_hash,
    stable_json_hash,
    write_deterministic_npz,
)
from .metrics import pdms_from_factors
from .six_factor_metrics import SIX_FACTOR_ORDER, pdms_from_six_factors


LABEL_SCHEMA_VERSION = "independent_wote_labels_4s.v1"
FACTOR_ORDER = ("NC", "DAC", "EP", "TTC", "Comfort")
SIX_FACTOR_LABEL_SCHEMA_VERSION = "independent_wote_labels_4s_six_factor.v2"
CANDIDATE_COUNT = 256


class IndependentLabelStoreError(RuntimeError):
    """An independent label artifact is incomplete or violates its identity."""


class ScoreReconstructionError(IndependentLabelStoreError):
    """The evaluator score cannot be reconstructed from the required five factors."""

    def __init__(
        self,
        max_absolute_error: float,
        mean_absolute_error: float,
        mismatched_indices: npt.NDArray[np.int64],
        evaluator_score: npt.NDArray[np.float32],
        reassembled_score: npt.NDArray[np.float64],
    ) -> None:
        self.max_absolute_error = float(max_absolute_error)
        self.mean_absolute_error = float(mean_absolute_error)
        self.mismatched_indices = np.asarray(mismatched_indices, dtype=np.int64)
        self.evaluator_score = np.asarray(evaluator_score, dtype=np.float32)
        self.reassembled_score = np.asarray(reassembled_score, dtype=np.float64)
        super().__init__(
            "five-factor score reconstruction failed: "
            f"max_abs={self.max_absolute_error:.9f}, "
            f"mean_abs={self.mean_absolute_error:.9f}, "
            f"mismatched_candidates={len(self.mismatched_indices)}"
        )


class SixFactorScoreReconstructionError(IndependentLabelStoreError):
    """The evaluator score cannot be reconstructed from all six factors."""

    def __init__(
        self,
        max_absolute_error: float,
        mean_absolute_error: float,
        mismatched_indices: npt.NDArray[np.int64],
        factors: npt.NDArray[np.float32],
        raw_progress: npt.NDArray[np.float32],
        evaluator_score: npt.NDArray[np.float32],
        reassembled_score: npt.NDArray[np.float64],
    ) -> None:
        self.max_absolute_error = float(max_absolute_error)
        self.mean_absolute_error = float(mean_absolute_error)
        self.mismatched_indices = np.asarray(mismatched_indices, dtype=np.int64)
        self.factors = np.asarray(factors, dtype=np.float32)
        self.raw_progress = np.asarray(raw_progress, dtype=np.float32)
        self.evaluator_score = np.asarray(evaluator_score, dtype=np.float32)
        self.reassembled_score = np.asarray(reassembled_score, dtype=np.float64)
        super().__init__(
            "six-factor score reconstruction failed: "
            f"max_abs={self.max_absolute_error:.9f}, "
            f"mean_abs={self.mean_absolute_error:.9f}, "
            f"mismatched_candidates={len(self.mismatched_indices)}"
        )


def _validate_sha256(name: str, value: str) -> None:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise IndependentLabelStoreError(f"{name} is not a lowercase SHA256: {value!r}")


def validate_label_arrays(
    factors: npt.ArrayLike,
    score: npt.ArrayLike,
    oracle_index: int,
    candidate_indices: npt.ArrayLike,
    reconstruction_tolerance: float = 1e-6,
) -> float:
    """Validate one scene and return its maximum reconstruction error."""

    factor_values = np.asarray(factors, dtype=np.float32)
    score_values = np.asarray(score, dtype=np.float32)
    indices = np.asarray(candidate_indices, dtype=np.int64)
    if factor_values.shape != (CANDIDATE_COUNT, len(FACTOR_ORDER)):
        raise IndependentLabelStoreError(
            f"factors expected [{CANDIDATE_COUNT},5], got {factor_values.shape}"
        )
    if score_values.shape != (CANDIDATE_COUNT,):
        raise IndependentLabelStoreError(
            f"score expected [{CANDIDATE_COUNT}], got {score_values.shape}"
        )
    if indices.shape != (CANDIDATE_COUNT,) or not np.array_equal(
        indices, np.arange(CANDIDATE_COUNT, dtype=np.int64)
    ):
        raise IndependentLabelStoreError("candidate_indices are not exactly 0..255")
    if not np.isfinite(factor_values).all() or not np.isfinite(score_values).all():
        raise IndependentLabelStoreError("candidate labels contain NaN/Inf")
    expected_oracle = int(np.argmax(score_values))
    if int(oracle_index) != expected_oracle:
        raise IndependentLabelStoreError(
            f"oracle_index {oracle_index} does not equal np.argmax(score)={expected_oracle}"
        )
    if reconstruction_tolerance < 0:
        raise ValueError("reconstruction_tolerance must be non-negative")
    reassembled = pdms_from_factors(factor_values)
    errors = np.abs(reassembled - score_values.astype(np.float64))
    mismatched = np.flatnonzero(errors > reconstruction_tolerance).astype(np.int64)
    if len(mismatched):
        raise ScoreReconstructionError(
            max_absolute_error=float(errors.max()),
            mean_absolute_error=float(errors.mean()),
            mismatched_indices=mismatched,
            evaluator_score=score_values,
            reassembled_score=reassembled,
        )
    return float(errors.max(initial=0.0))


@dataclass(frozen=True)
class IndependentLabelRecord:
    scene_token: str
    candidate_bank_hash: str
    trajectory_hash: str
    metric_cache_sha256: str

    def validate(self) -> None:
        if not self.scene_token:
            raise IndependentLabelStoreError("scene token must be non-empty")
        _validate_sha256("candidate_bank_hash", self.candidate_bank_hash)
        _validate_sha256("trajectory_hash", self.trajectory_hash)
        _validate_sha256("metric_cache_sha256", self.metric_cache_sha256)

    def as_dict(self) -> dict[str, str]:
        self.validate()
        return {
            "scene_token": self.scene_token,
            "candidate_bank_hash": self.candidate_bank_hash,
            "trajectory_hash": self.trajectory_hash,
            "metric_cache_sha256": self.metric_cache_sha256,
        }


@dataclass(frozen=True)
class IndependentLabelScene:
    record: IndependentLabelRecord
    factors: npt.NDArray[np.float32]
    score: npt.NDArray[np.float32]
    oracle_index: int
    candidate_indices: npt.NDArray[np.int64]

    def validate(self, reconstruction_tolerance: float = 1e-6) -> float:
        self.record.validate()
        return validate_label_arrays(
            self.factors,
            self.score,
            self.oracle_index,
            self.candidate_indices,
            reconstruction_tolerance=reconstruction_tolerance,
        )


def _shard_logical_payload(
    records: Sequence[IndependentLabelRecord],
    arrays: Mapping[str, npt.NDArray[Any]],
    candidate_bank_hash: str,
    evaluator_contract_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": LABEL_SCHEMA_VERSION,
        "candidate_bank_hash": candidate_bank_hash,
        "evaluator_contract_sha256": evaluator_contract_sha256,
        "records": [record.as_dict() for record in records],
        "array_hashes": {
            name: stable_array_hash(np.asarray(value))
            for name, value in sorted(arrays.items())
        },
    }


def _manifest_logical_payload(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": manifest["schema_version"],
        "factor_order": manifest["factor_order"],
        "candidate_count": manifest["candidate_count"],
        "scene_count": manifest["scene_count"],
        "candidate_bank_hash": manifest["candidate_bank_hash"],
        "evaluator_contract_sha256": manifest["evaluator_contract_sha256"],
        "scene_tokens": manifest["scene_tokens"],
        "shard_logical_content_sha256": [
            shard["logical_content_sha256"] for shard in manifest["shards"]
        ],
    }


class IndependentCandidateLabelWriter:
    """Write a deterministic, sharded label store without pickle or fp16 casts."""

    def __init__(
        self,
        root: Path,
        candidate_bank_hash: str,
        evaluator_contract_sha256: str,
        shard_scenes: int = 16,
    ) -> None:
        if root.exists():
            raise IndependentLabelStoreError(f"refusing existing label store: {root}")
        if shard_scenes <= 0:
            raise ValueError("shard_scenes must be positive")
        _validate_sha256("candidate_bank_hash", candidate_bank_hash)
        _validate_sha256("evaluator_contract_sha256", evaluator_contract_sha256)
        self.root = root
        self.candidate_bank_hash = candidate_bank_hash
        self.evaluator_contract_sha256 = evaluator_contract_sha256
        self.shard_scenes = shard_scenes

    def write(self, scenes: Sequence[IndependentLabelScene]) -> Path:
        if not scenes:
            raise IndependentLabelStoreError("cannot write an empty label store")
        tokens = [scene.record.scene_token for scene in scenes]
        if len(tokens) != len(set(tokens)):
            raise IndependentLabelStoreError(
                "label store contains duplicate scene tokens"
            )
        for scene in scenes:
            if scene.record.candidate_bank_hash != self.candidate_bank_hash:
                raise IndependentLabelStoreError(
                    f"candidate bank mismatch for {scene.record.scene_token}"
                )
            scene.validate()

        self.root.mkdir(parents=True, exist_ok=False)
        shards: list[dict[str, Any]] = []
        for shard_index, start in enumerate(range(0, len(scenes), self.shard_scenes)):
            shard_scenes = scenes[start : start + self.shard_scenes]
            records = [scene.record for scene in shard_scenes]
            arrays: dict[str, npt.NDArray[Any]] = {
                "factors": np.stack(
                    [
                        np.asarray(scene.factors, dtype=np.float32)
                        for scene in shard_scenes
                    ]
                ),
                "score": np.stack(
                    [
                        np.asarray(scene.score, dtype=np.float32)
                        for scene in shard_scenes
                    ]
                ),
                "oracle_index": np.asarray(
                    [scene.oracle_index for scene in shard_scenes], dtype=np.int64
                ),
                "candidate_indices": np.stack(
                    [
                        np.asarray(scene.candidate_indices, dtype=np.int64)
                        for scene in shard_scenes
                    ]
                ),
            }
            logical_payload = _shard_logical_payload(
                records,
                arrays,
                self.candidate_bank_hash,
                self.evaluator_contract_sha256,
            )
            logical_hash = stable_json_hash(logical_payload)
            shard_name = f"shard-{shard_index:05d}.npz"
            shard_path = self.root / shard_name
            shard_sha = write_deterministic_npz(shard_path, arrays)
            sidecar = {
                **logical_payload,
                "logical_content_sha256": logical_hash,
                "shard_index": shard_index,
                "shard_sha256": shard_sha,
                "arrays": {
                    name: {"shape": list(value.shape), "dtype": str(value.dtype)}
                    for name, value in sorted(arrays.items())
                },
            }
            sidecar_name = f"shard-{shard_index:05d}.json"
            sidecar_path = self.root / sidecar_name
            atomic_write_json(sidecar_path, sidecar)
            shards.append(
                {
                    "path": shard_name,
                    "sha256": shard_sha,
                    "sidecar": sidecar_name,
                    "sidecar_sha256": sha256_file(sidecar_path),
                    "logical_content_sha256": logical_hash,
                    "scene_count": len(shard_scenes),
                }
            )

        manifest: dict[str, Any] = {
            "schema_version": LABEL_SCHEMA_VERSION,
            "factor_order": list(FACTOR_ORDER),
            "candidate_count": CANDIDATE_COUNT,
            "scene_count": len(scenes),
            "candidate_bank_hash": self.candidate_bank_hash,
            "evaluator_contract_sha256": self.evaluator_contract_sha256,
            "scene_tokens": tokens,
            "shards": shards,
        }
        manifest["logical_content_sha256"] = stable_json_hash(
            _manifest_logical_payload(manifest)
        )
        manifest_path = self.root / "manifest.json"
        atomic_write_json(manifest_path, manifest)
        return manifest_path


class IndependentCandidateLabelStore:
    """Validate and read finalized independent labels by explicit scene identity."""

    def __init__(self, root: Path):
        self.root = root
        manifest_path = root / "manifest.json"
        if not manifest_path.is_file():
            raise IndependentLabelStoreError(f"missing label manifest: {manifest_path}")
        self.manifest: dict[str, Any] = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )
        if self.manifest.get("schema_version") != LABEL_SCHEMA_VERSION:
            raise IndependentLabelStoreError("unsupported independent label schema")
        if self.manifest.get("factor_order") != list(FACTOR_ORDER):
            raise IndependentLabelStoreError("independent label factor order changed")
        claimed = self.manifest.get("logical_content_sha256")
        if claimed != stable_json_hash(_manifest_logical_payload(self.manifest)):
            raise IndependentLabelStoreError("label manifest logical hash mismatch")
        _validate_sha256(
            "candidate_bank_hash", str(self.manifest["candidate_bank_hash"])
        )
        _validate_sha256(
            "evaluator_contract_sha256",
            str(self.manifest["evaluator_contract_sha256"]),
        )

    @property
    def logical_content_sha256(self) -> str:
        return str(self.manifest["logical_content_sha256"])

    @property
    def scene_tokens(self) -> tuple[str, ...]:
        return tuple(str(token) for token in self.manifest["scene_tokens"])

    def iter_scenes(self) -> Iterator[IndependentLabelScene]:
        count = 0
        for shard in self.manifest["shards"]:
            shard_path = self.root / shard["path"]
            sidecar_path = self.root / shard["sidecar"]
            if sha256_file(shard_path) != shard["sha256"]:
                raise IndependentLabelStoreError(
                    f"label shard hash mismatch: {shard_path}"
                )
            if sha256_file(sidecar_path) != shard["sidecar_sha256"]:
                raise IndependentLabelStoreError(
                    f"label sidecar hash mismatch: {sidecar_path}"
                )
            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
            claimed = sidecar.pop("logical_content_sha256", None)
            logical_payload = {
                key: sidecar[key]
                for key in (
                    "schema_version",
                    "candidate_bank_hash",
                    "evaluator_contract_sha256",
                    "records",
                    "array_hashes",
                )
            }
            if claimed != stable_json_hash(logical_payload):
                raise IndependentLabelStoreError(
                    f"label sidecar logical hash mismatch: {sidecar_path}"
                )
            with np.load(shard_path, allow_pickle=False) as archive:
                arrays = {name: archive[name] for name in archive.files}
            for name, expected_hash in sidecar["array_hashes"].items():
                if (
                    name not in arrays
                    or stable_array_hash(arrays[name]) != expected_hash
                ):
                    raise IndependentLabelStoreError(
                        f"label array logical hash mismatch: {shard_path}:{name}"
                    )
            records = sidecar["records"]
            for index, record_payload in enumerate(records):
                record = IndependentLabelRecord(**record_payload)
                scene = IndependentLabelScene(
                    record=record,
                    factors=np.asarray(arrays["factors"][index], dtype=np.float32),
                    score=np.asarray(arrays["score"][index], dtype=np.float32),
                    oracle_index=int(arrays["oracle_index"][index]),
                    candidate_indices=np.asarray(
                        arrays["candidate_indices"][index], dtype=np.int64
                    ),
                )
                scene.validate()
                count += 1
                yield scene
        if count != int(self.manifest["scene_count"]):
            raise IndependentLabelStoreError(
                f"read {count} scenes, manifest claims {self.manifest['scene_count']}"
            )

    def scene_index(self) -> dict[str, IndependentLabelScene]:
        return {scene.record.scene_token: scene for scene in self.iter_scenes()}

    def join_scene(
        self,
        scene_token: str,
        candidate_bank_hash: str,
        trajectory_hash: str,
    ) -> IndependentLabelScene:
        scenes = self.scene_index()
        if scene_token not in scenes:
            raise IndependentLabelStoreError(
                f"independent labels missing scene token: {scene_token}"
            )
        scene = scenes[scene_token]
        if scene.record.candidate_bank_hash != candidate_bank_hash:
            raise IndependentLabelStoreError(
                f"candidate bank join mismatch for {scene_token}"
            )
        if scene.record.trajectory_hash != trajectory_hash:
            raise IndependentLabelStoreError(
                f"trajectory join mismatch for {scene_token}"
            )
        return scene


def validate_six_factor_label_arrays(
    factors: npt.ArrayLike,
    score: npt.ArrayLike,
    raw_progress: npt.ArrayLike,
    oracle_index: int,
    candidate_indices: npt.ArrayLike,
    reconstruction_tolerance: float = 1e-6,
) -> float:
    """Validate one v2 scene and return its maximum score reconstruction error."""

    factor_values = np.asarray(factors, dtype=np.float32)
    score_values = np.asarray(score, dtype=np.float32)
    progress_values = np.asarray(raw_progress, dtype=np.float32)
    indices = np.asarray(candidate_indices, dtype=np.int64)
    expected_factor_shape = (CANDIDATE_COUNT, len(SIX_FACTOR_ORDER))
    if factor_values.shape != expected_factor_shape:
        raise IndependentLabelStoreError(
            f"six-factor labels expected {expected_factor_shape}, got {factor_values.shape}"
        )
    if score_values.shape != (CANDIDATE_COUNT,):
        raise IndependentLabelStoreError(
            f"score expected [{CANDIDATE_COUNT}], got {score_values.shape}"
        )
    if progress_values.shape != (CANDIDATE_COUNT,):
        raise IndependentLabelStoreError(
            f"raw_progress expected [{CANDIDATE_COUNT}], got {progress_values.shape}"
        )
    if indices.shape != (CANDIDATE_COUNT,) or not np.array_equal(
        indices, np.arange(CANDIDATE_COUNT, dtype=np.int64)
    ):
        raise IndependentLabelStoreError("candidate_indices are not exactly 0..255")
    if not (
        np.isfinite(factor_values).all()
        and np.isfinite(score_values).all()
        and np.isfinite(progress_values).all()
    ):
        raise IndependentLabelStoreError("six-factor labels contain NaN/Inf")
    expected_oracle = int(np.argmax(score_values))
    if int(oracle_index) != expected_oracle:
        raise IndependentLabelStoreError(
            f"oracle_index {oracle_index} does not equal np.argmax(score)={expected_oracle}"
        )
    if reconstruction_tolerance < 0:
        raise ValueError("reconstruction_tolerance must be non-negative")
    reassembled = pdms_from_six_factors(factor_values)
    errors = np.abs(reassembled - score_values.astype(np.float64))
    mismatched = np.flatnonzero(errors > reconstruction_tolerance).astype(np.int64)
    if len(mismatched):
        raise SixFactorScoreReconstructionError(
            max_absolute_error=float(errors.max()),
            mean_absolute_error=float(errors.mean()),
            mismatched_indices=mismatched,
            factors=factor_values,
            raw_progress=progress_values,
            evaluator_score=score_values,
            reassembled_score=reassembled,
        )
    return float(errors.max(initial=0.0))


@dataclass(frozen=True)
class SixFactorIndependentLabelScene:
    """One scene in the explicit v2 six-factor label contract."""

    record: IndependentLabelRecord
    factors: npt.NDArray[np.float32]
    score: npt.NDArray[np.float32]
    raw_progress: npt.NDArray[np.float32]
    oracle_index: int
    candidate_indices: npt.NDArray[np.int64]

    def validate(self, reconstruction_tolerance: float = 1e-6) -> float:
        self.record.validate()
        return validate_six_factor_label_arrays(
            self.factors,
            self.score,
            self.raw_progress,
            self.oracle_index,
            self.candidate_indices,
            reconstruction_tolerance=reconstruction_tolerance,
        )


def _six_factor_shard_logical_payload(
    records: Sequence[IndependentLabelRecord],
    arrays: Mapping[str, npt.NDArray[Any]],
    candidate_bank_hash: str,
    evaluator_contract_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": SIX_FACTOR_LABEL_SCHEMA_VERSION,
        "candidate_bank_hash": candidate_bank_hash,
        "evaluator_contract_sha256": evaluator_contract_sha256,
        "progress_normalization_scope": "all_256_candidates_within_scene",
        "candidate_set_dependent_ep": True,
        "records": [record.as_dict() for record in records],
        "array_hashes": {
            name: stable_array_hash(np.asarray(value))
            for name, value in sorted(arrays.items())
        },
    }


def _six_factor_manifest_logical_payload(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": manifest["schema_version"],
        "factor_order": manifest["factor_order"],
        "candidate_count": manifest["candidate_count"],
        "scene_count": manifest["scene_count"],
        "candidate_bank_hash": manifest["candidate_bank_hash"],
        "evaluator_contract_sha256": manifest["evaluator_contract_sha256"],
        "progress_normalization_scope": manifest["progress_normalization_scope"],
        "candidate_set_dependent_ep": manifest["candidate_set_dependent_ep"],
        "scene_tokens": manifest["scene_tokens"],
        "shard_logical_content_sha256": [
            shard["logical_content_sha256"] for shard in manifest["shards"]
        ],
    }


class SixFactorIndependentCandidateLabelWriter:
    """Write a deterministic v2 store with DDC and diagnostic raw progress."""

    def __init__(
        self,
        root: Path,
        candidate_bank_hash: str,
        evaluator_contract_sha256: str,
        shard_scenes: int = 16,
    ) -> None:
        if root.exists():
            raise IndependentLabelStoreError(f"refusing existing label store: {root}")
        if shard_scenes <= 0:
            raise ValueError("shard_scenes must be positive")
        _validate_sha256("candidate_bank_hash", candidate_bank_hash)
        _validate_sha256("evaluator_contract_sha256", evaluator_contract_sha256)
        self.root = root
        self.candidate_bank_hash = candidate_bank_hash
        self.evaluator_contract_sha256 = evaluator_contract_sha256
        self.shard_scenes = shard_scenes

    def write(self, scenes: Sequence[SixFactorIndependentLabelScene]) -> Path:
        if not scenes:
            raise IndependentLabelStoreError("cannot write an empty label store")
        tokens = [scene.record.scene_token for scene in scenes]
        if len(tokens) != len(set(tokens)):
            raise IndependentLabelStoreError(
                "label store contains duplicate scene tokens"
            )
        for scene in scenes:
            if scene.record.candidate_bank_hash != self.candidate_bank_hash:
                raise IndependentLabelStoreError(
                    f"candidate bank mismatch for {scene.record.scene_token}"
                )
            scene.validate()

        self.root.mkdir(parents=True, exist_ok=False)
        shards: list[dict[str, Any]] = []
        for shard_index, start in enumerate(range(0, len(scenes), self.shard_scenes)):
            shard_scenes = scenes[start : start + self.shard_scenes]
            records = [scene.record for scene in shard_scenes]
            arrays: dict[str, npt.NDArray[Any]] = {
                "factors": np.stack(
                    [
                        np.asarray(scene.factors, dtype=np.float32)
                        for scene in shard_scenes
                    ]
                ),
                "score": np.stack(
                    [
                        np.asarray(scene.score, dtype=np.float32)
                        for scene in shard_scenes
                    ]
                ),
                "raw_progress": np.stack(
                    [
                        np.asarray(scene.raw_progress, dtype=np.float32)
                        for scene in shard_scenes
                    ]
                ),
                "oracle_index": np.asarray(
                    [scene.oracle_index for scene in shard_scenes], dtype=np.int64
                ),
                "candidate_indices": np.stack(
                    [
                        np.asarray(scene.candidate_indices, dtype=np.int64)
                        for scene in shard_scenes
                    ]
                ),
            }
            logical_payload = _six_factor_shard_logical_payload(
                records,
                arrays,
                self.candidate_bank_hash,
                self.evaluator_contract_sha256,
            )
            logical_hash = stable_json_hash(logical_payload)
            shard_name = f"shard-{shard_index:05d}.npz"
            shard_path = self.root / shard_name
            shard_sha = write_deterministic_npz(shard_path, arrays)
            sidecar = {
                **logical_payload,
                "logical_content_sha256": logical_hash,
                "shard_index": shard_index,
                "shard_sha256": shard_sha,
                "arrays": {
                    name: {"shape": list(value.shape), "dtype": str(value.dtype)}
                    for name, value in sorted(arrays.items())
                },
            }
            sidecar_name = f"shard-{shard_index:05d}.json"
            sidecar_path = self.root / sidecar_name
            atomic_write_json(sidecar_path, sidecar)
            shards.append(
                {
                    "path": shard_name,
                    "sha256": shard_sha,
                    "sidecar": sidecar_name,
                    "sidecar_sha256": sha256_file(sidecar_path),
                    "logical_content_sha256": logical_hash,
                    "scene_count": len(shard_scenes),
                }
            )

        manifest: dict[str, Any] = {
            "schema_version": SIX_FACTOR_LABEL_SCHEMA_VERSION,
            "factor_order": list(SIX_FACTOR_ORDER),
            "candidate_count": CANDIDATE_COUNT,
            "scene_count": len(scenes),
            "candidate_bank_hash": self.candidate_bank_hash,
            "evaluator_contract_sha256": self.evaluator_contract_sha256,
            "progress_normalization_scope": "all_256_candidates_within_scene",
            "candidate_set_dependent_ep": True,
            "scene_tokens": tokens,
            "shards": shards,
        }
        manifest["logical_content_sha256"] = stable_json_hash(
            _six_factor_manifest_logical_payload(manifest)
        )
        manifest_path = self.root / "manifest.json"
        atomic_write_json(manifest_path, manifest)
        return manifest_path


class SixFactorIndependentCandidateLabelStore:
    """Strict v2 reader; v1 artifacts are intentionally rejected."""

    def __init__(self, root: Path):
        self.root = root
        manifest_path = root / "manifest.json"
        if not manifest_path.is_file():
            raise IndependentLabelStoreError(f"missing label manifest: {manifest_path}")
        self.manifest: dict[str, Any] = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )
        if self.manifest.get("schema_version") != SIX_FACTOR_LABEL_SCHEMA_VERSION:
            raise IndependentLabelStoreError("unsupported six-factor label schema")
        if self.manifest.get("factor_order") != list(SIX_FACTOR_ORDER):
            raise IndependentLabelStoreError("six-factor label order changed")
        if self.manifest.get("candidate_count") != CANDIDATE_COUNT:
            raise IndependentLabelStoreError("six-factor candidate count changed")
        if (
            self.manifest.get("progress_normalization_scope")
            != "all_256_candidates_within_scene"
            or self.manifest.get("candidate_set_dependent_ep") is not True
        ):
            raise IndependentLabelStoreError(
                "six-factor EP normalization contract changed"
            )
        claimed = self.manifest.get("logical_content_sha256")
        if claimed != stable_json_hash(
            _six_factor_manifest_logical_payload(self.manifest)
        ):
            raise IndependentLabelStoreError(
                "six-factor manifest logical hash mismatch"
            )
        _validate_sha256(
            "candidate_bank_hash", str(self.manifest["candidate_bank_hash"])
        )
        _validate_sha256(
            "evaluator_contract_sha256",
            str(self.manifest["evaluator_contract_sha256"]),
        )

    @property
    def logical_content_sha256(self) -> str:
        return str(self.manifest["logical_content_sha256"])

    @property
    def scene_tokens(self) -> tuple[str, ...]:
        return tuple(str(token) for token in self.manifest["scene_tokens"])

    def iter_scenes(self) -> Iterator[SixFactorIndependentLabelScene]:
        count = 0
        payload_keys = (
            "schema_version",
            "candidate_bank_hash",
            "evaluator_contract_sha256",
            "progress_normalization_scope",
            "candidate_set_dependent_ep",
            "records",
            "array_hashes",
        )
        for shard in self.manifest["shards"]:
            shard_path = self.root / shard["path"]
            sidecar_path = self.root / shard["sidecar"]
            if sha256_file(shard_path) != shard["sha256"]:
                raise IndependentLabelStoreError(
                    f"six-factor shard hash mismatch: {shard_path}"
                )
            if sha256_file(sidecar_path) != shard["sidecar_sha256"]:
                raise IndependentLabelStoreError(
                    f"six-factor sidecar hash mismatch: {sidecar_path}"
                )
            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
            claimed = sidecar.get("logical_content_sha256")
            logical_payload = {key: sidecar[key] for key in payload_keys}
            if claimed != stable_json_hash(logical_payload):
                raise IndependentLabelStoreError(
                    f"six-factor sidecar logical hash mismatch: {sidecar_path}"
                )
            with np.load(shard_path, allow_pickle=False) as archive:
                arrays = {name: archive[name] for name in archive.files}
            required_arrays = {
                "factors",
                "score",
                "raw_progress",
                "oracle_index",
                "candidate_indices",
            }
            if set(arrays) != required_arrays:
                raise IndependentLabelStoreError(
                    f"six-factor shard array schema mismatch: {shard_path}"
                )
            for name, expected_hash in sidecar["array_hashes"].items():
                if stable_array_hash(arrays[name]) != expected_hash:
                    raise IndependentLabelStoreError(
                        f"six-factor array logical hash mismatch: {shard_path}:{name}"
                    )
            records = sidecar["records"]
            for index, record_payload in enumerate(records):
                scene = SixFactorIndependentLabelScene(
                    record=IndependentLabelRecord(**record_payload),
                    factors=np.asarray(arrays["factors"][index], dtype=np.float32),
                    score=np.asarray(arrays["score"][index], dtype=np.float32),
                    raw_progress=np.asarray(
                        arrays["raw_progress"][index], dtype=np.float32
                    ),
                    oracle_index=int(arrays["oracle_index"][index]),
                    candidate_indices=np.asarray(
                        arrays["candidate_indices"][index], dtype=np.int64
                    ),
                )
                scene.validate()
                count += 1
                yield scene
        if count != int(self.manifest["scene_count"]):
            raise IndependentLabelStoreError(
                f"read {count} scenes, manifest claims {self.manifest['scene_count']}"
            )

    def scene_index(self) -> dict[str, SixFactorIndependentLabelScene]:
        return {scene.record.scene_token: scene for scene in self.iter_scenes()}

    def join_scene(
        self,
        scene_token: str,
        candidate_bank_hash: str,
        trajectory_hash: str,
        candidate_indices: npt.ArrayLike | None = None,
    ) -> SixFactorIndependentLabelScene:
        scenes = self.scene_index()
        if scene_token not in scenes:
            raise IndependentLabelStoreError(
                f"six-factor labels missing scene token: {scene_token}"
            )
        scene = scenes[scene_token]
        if scene.record.candidate_bank_hash != candidate_bank_hash:
            raise IndependentLabelStoreError(
                f"candidate bank join mismatch for {scene_token}"
            )
        if scene.record.trajectory_hash != trajectory_hash:
            raise IndependentLabelStoreError(
                f"trajectory join mismatch for {scene_token}"
            )
        if candidate_indices is not None and not np.array_equal(
            scene.candidate_indices, np.asarray(candidate_indices, dtype=np.int64)
        ):
            raise IndependentLabelStoreError(
                f"candidate index join mismatch for {scene_token}"
            )
        return scene
