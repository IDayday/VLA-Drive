"""Offline-only, relocatable per-token candidate cache schema for DDP-DRS."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np


REQUIRED_METRICS = (
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "time_to_collision_within_bound",
    "ego_progress",
    "driving_direction_compliance",
    "comfort",
)
OPTIONAL_METRICS = ("lane_keeping", "traffic_light_compliance")
SUPRIM_METRICS = (
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "time_to_collision_within_bound",
    "ego_progress",
    "driving_direction_compliance",
    "lane_keeping",
    "traffic_light_compliance",
    "history_comfort",
)
TRAINING_METRIC_SCHEMA = SUPRIM_METRICS + ("comfort",)
SCHEMA_VERSION = 1
STATIC_SAMPLE_INDICES = (4, 9, 14, 19, 24, 29, 34, 39)
PROPOSAL_DIRECTORY = "proposals"
TRAINING_RECORD_DIRECTORY = "records"
COMPLETION_FILE = "_SUCCESS.json"


def sha256_file(path: os.PathLike[str] | str, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def stable_config_hash(config: Mapping[str, Any]) -> str:
    serialized = json.dumps(
        config,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


@dataclass(frozen=True)
class CandidateCacheManifest:
    split: str
    ddp_checkpoint_sha: str
    repository_commit_sha: str
    generator_config_hash: str
    seed: int
    metric_schema: Tuple[str, ...]
    label_source_split: str
    schema_version: int = SCHEMA_VERSION

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"candidate cache schema version {self.schema_version} is unsupported"
            )
        missing = set(REQUIRED_METRICS).difference(self.metric_schema)
        if missing:
            raise ValueError(f"candidate cache metric schema missing {sorted(missing)}")
        if not self.ddp_checkpoint_sha or not self.repository_commit_sha:
            raise ValueError("candidate cache hashes must be non-empty")
        if not self.generator_config_hash:
            raise ValueError("candidate cache generator_config_hash must be non-empty")
        if not isinstance(self.seed, int):
            raise TypeError("candidate cache seed must be an integer")
        if "test" in self.label_source_split.lower():
            raise ValueError("candidate cache cannot use test-label metric sources")


@dataclass
class CandidateCacheRecord:
    token: str
    split: str
    ddp_checkpoint_sha: str
    repository_commit_sha: str
    generator_config_hash: str
    seed: int
    candidate_ids: np.ndarray
    trajectory_8: np.ndarray
    trajectory_40: np.ndarray
    metrics: Dict[str, np.ndarray]
    final_score: np.ndarray
    metric_schema: Tuple[str, ...]

    def validate(self, manifest: Optional[CandidateCacheManifest] = None) -> None:
        if not self.token or Path(self.token).name != self.token:
            raise ValueError("candidate cache token must be a non-empty file-safe name")
        if self.trajectory_8.ndim != 3 or self.trajectory_8.shape[1:] != (8, 3):
            raise ValueError("cached trajectory_8 must have shape [K,8,3]")
        if self.trajectory_40.ndim != 3 or self.trajectory_40.shape[1:] != (40, 3):
            raise ValueError("cached trajectory_40 must have shape [K,40,3]")
        candidate_count = self.trajectory_8.shape[0]
        if self.trajectory_40.shape[0] != candidate_count:
            raise ValueError("cached 8/40 trajectory candidate counts differ")
        if self.candidate_ids.shape != (candidate_count,):
            raise ValueError("candidate_ids must have shape [K]")
        if not np.issubdtype(self.candidate_ids.dtype, np.integer):
            raise TypeError("candidate_ids must use an integer dtype")
        if np.unique(self.candidate_ids).size != candidate_count:
            raise ValueError("candidate_ids must be unique to preserve candidate order")
        if self.final_score.shape != (candidate_count,):
            raise ValueError("final_score must have shape [K]")
        if tuple(self.metric_schema) != tuple(self.metrics):
            raise ValueError("metric_schema order does not match stored metric arrays")
        missing = set(REQUIRED_METRICS).difference(self.metric_schema)
        if missing:
            raise ValueError(f"cached metric schema missing {sorted(missing)}")
        for name, values in self.metrics.items():
            if values.shape != (candidate_count,):
                raise ValueError(f"cached metric {name} must have shape [K]")
        arrays = [self.trajectory_8, self.trajectory_40, self.final_score, *self.metrics.values()]
        if not all(np.isfinite(array).all() for array in arrays):
            raise ValueError("candidate cache contains NaN or Inf")
        if not np.allclose(
            self.trajectory_40[:, STATIC_SAMPLE_INDICES, :],
            self.trajectory_8,
            rtol=1e-4,
            atol=1e-5,
        ):
            raise ValueError("cached trajectories violate the fixed 8/40 time convention")
        if manifest is not None:
            manifest.validate()
            comparisons = {
                "split": (self.split, manifest.split),
                "ddp_checkpoint_sha": (
                    self.ddp_checkpoint_sha,
                    manifest.ddp_checkpoint_sha,
                ),
                "repository_commit_sha": (
                    self.repository_commit_sha,
                    manifest.repository_commit_sha,
                ),
                "generator_config_hash": (
                    self.generator_config_hash,
                    manifest.generator_config_hash,
                ),
                "seed": (self.seed, manifest.seed),
                "metric_schema": (tuple(self.metric_schema), tuple(manifest.metric_schema)),
            }
            mismatches = [
                name for name, (actual, expected) in comparisons.items() if actual != expected
            ]
            if mismatches:
                raise ValueError(f"candidate cache manifest mismatch: {mismatches}")


def write_manifest(cache_root: os.PathLike[str] | str, manifest: CandidateCacheManifest) -> Path:
    manifest.validate()
    root = Path(cache_root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "manifest.json"
    temporary = root / f".manifest.json.tmp-{os.getpid()}"
    payload = asdict(manifest)
    payload["metric_schema"] = list(manifest.metric_schema)
    with open(temporary, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, path)
    return path


def read_manifest(cache_root: os.PathLike[str] | str) -> CandidateCacheManifest:
    path = Path(cache_root) / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"candidate cache manifest not found: {path}")
    with open(path, "r", encoding="utf-8") as file:
        payload = json.load(file)
    payload["metric_schema"] = tuple(payload["metric_schema"])
    manifest = CandidateCacheManifest(**payload)
    manifest.validate()
    return manifest


def save_record(
    cache_root: os.PathLike[str] | str,
    record: CandidateCacheRecord,
    manifest: CandidateCacheManifest,
) -> Path:
    record.validate(manifest)
    root = Path(cache_root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{record.token}.npz"
    temporary = root / f".{record.token}.tmp-{os.getpid()}.npz"
    payload: Dict[str, np.ndarray] = {
        "token": np.asarray(record.token),
        "split": np.asarray(record.split),
        "ddp_checkpoint_sha": np.asarray(record.ddp_checkpoint_sha),
        "repository_commit_sha": np.asarray(record.repository_commit_sha),
        "generator_config_hash": np.asarray(record.generator_config_hash),
        "seed": np.asarray(record.seed, dtype=np.int64),
        "candidate_ids": record.candidate_ids,
        "trajectory_8": record.trajectory_8,
        "trajectory_40": record.trajectory_40,
        "final_score": record.final_score,
        "metric_schema": np.asarray(record.metric_schema),
    }
    payload.update(record.metrics)
    np.savez_compressed(temporary, **payload)
    os.replace(temporary, path)
    return path


def load_record(
    cache_root: os.PathLike[str] | str,
    token: str,
    *,
    expected_split: str,
    expected_ddp_checkpoint_sha: str,
) -> CandidateCacheRecord:
    if not token or Path(token).name != token:
        raise ValueError("candidate cache token must be file-safe")
    manifest = read_manifest(cache_root)
    if manifest.split != expected_split:
        raise ValueError(
            f"candidate cache split {manifest.split!r} does not match {expected_split!r}"
        )
    if manifest.ddp_checkpoint_sha != expected_ddp_checkpoint_sha:
        raise ValueError("candidate cache DDP checkpoint hash mismatch")
    path = Path(cache_root) / f"{token}.npz"
    if not path.is_file():
        raise FileNotFoundError(f"candidate cache record not found: {path}")
    with np.load(path, allow_pickle=False) as payload:
        stored_token = str(payload["token"].item())
        if stored_token != token:
            raise ValueError(
                f"candidate cache token {stored_token!r} does not match {token!r}"
            )
        metric_schema = tuple(str(value) for value in payload["metric_schema"].tolist())
        metrics = {name: payload[name].copy() for name in metric_schema}
        record = CandidateCacheRecord(
            token=stored_token,
            split=str(payload["split"].item()),
            ddp_checkpoint_sha=str(payload["ddp_checkpoint_sha"].item()),
            repository_commit_sha=str(payload["repository_commit_sha"].item()),
            generator_config_hash=str(payload["generator_config_hash"].item()),
            seed=int(payload["seed"].item()),
            candidate_ids=payload["candidate_ids"].copy(),
            trajectory_8=payload["trajectory_8"].copy(),
            trajectory_40=payload["trajectory_40"].copy(),
            metrics=metrics,
            final_score=payload["final_score"].copy(),
            metric_schema=metric_schema,
        )
    record.validate(manifest)
    return record


@dataclass
class ProposalCacheRecord:
    """Unscored, deterministic DDP proposals produced by the GPU cache pass."""

    token: str
    split: str
    ddp_checkpoint_sha: str
    repository_commit_sha: str
    generator_config_hash: str
    seed: int
    candidate_ids: np.ndarray
    trajectory_8: np.ndarray
    trajectory_40: np.ndarray
    target_trajectory_8: np.ndarray

    def validate(self, manifest: CandidateCacheManifest) -> None:
        manifest.validate()
        if not self.token or Path(self.token).name != self.token:
            raise ValueError("proposal cache token must be file-safe")
        if self.trajectory_8.ndim != 3 or self.trajectory_8.shape[1:] != (8, 3):
            raise ValueError("proposal trajectory_8 must have shape [K,8,3]")
        candidate_count = self.trajectory_8.shape[0]
        if self.trajectory_40.shape != (candidate_count, 40, 3):
            raise ValueError("proposal trajectory_40 must have shape [K,40,3]")
        if self.candidate_ids.shape != (candidate_count,):
            raise ValueError("proposal candidate_ids must have shape [K]")
        if not np.issubdtype(self.candidate_ids.dtype, np.integer):
            raise TypeError("proposal candidate_ids must use an integer dtype")
        if not np.array_equal(
            self.candidate_ids, np.arange(candidate_count, dtype=self.candidate_ids.dtype)
        ):
            raise ValueError("proposal candidate_ids must preserve generator order 0..K-1")
        if self.target_trajectory_8.shape != (8, 3):
            raise ValueError("target_trajectory_8 must have shape [8,3]")
        arrays = (
            self.trajectory_8,
            self.trajectory_40,
            self.target_trajectory_8,
        )
        if not all(np.isfinite(array).all() for array in arrays):
            raise ValueError("proposal cache contains NaN or Inf")
        if not np.allclose(
            self.trajectory_40[:, STATIC_SAMPLE_INDICES, :],
            self.trajectory_8,
            rtol=1e-4,
            atol=1e-5,
        ):
            raise ValueError("proposal cache violates the fixed 8/40 convention")
        comparisons = {
            "split": (self.split, manifest.split),
            "ddp_checkpoint_sha": (
                self.ddp_checkpoint_sha,
                manifest.ddp_checkpoint_sha,
            ),
            "repository_commit_sha": (
                self.repository_commit_sha,
                manifest.repository_commit_sha,
            ),
            "generator_config_hash": (
                self.generator_config_hash,
                manifest.generator_config_hash,
            ),
            "seed": (self.seed, manifest.seed),
        }
        mismatches = [
            name for name, (actual, expected) in comparisons.items() if actual != expected
        ]
        if mismatches:
            raise ValueError(f"proposal cache manifest mismatch: {mismatches}")


def save_proposal_record(
    cache_root: os.PathLike[str] | str,
    record: ProposalCacheRecord,
    manifest: CandidateCacheManifest,
) -> Path:
    record.validate(manifest)
    root = Path(cache_root) / PROPOSAL_DIRECTORY
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{record.token}.npz"
    temporary = root / f".{record.token}.tmp-{os.getpid()}.npz"
    np.savez_compressed(
        temporary,
        token=np.asarray(record.token),
        split=np.asarray(record.split),
        ddp_checkpoint_sha=np.asarray(record.ddp_checkpoint_sha),
        repository_commit_sha=np.asarray(record.repository_commit_sha),
        generator_config_hash=np.asarray(record.generator_config_hash),
        seed=np.asarray(record.seed, dtype=np.int64),
        candidate_ids=record.candidate_ids,
        trajectory_8=record.trajectory_8,
        trajectory_40=record.trajectory_40,
        target_trajectory_8=record.target_trajectory_8,
    )
    os.replace(temporary, path)
    return path


def load_proposal_record(
    cache_root: os.PathLike[str] | str,
    token: str,
    manifest: Optional[CandidateCacheManifest] = None,
) -> ProposalCacheRecord:
    if not token or Path(token).name != token:
        raise ValueError("proposal cache token must be file-safe")
    manifest = read_manifest(cache_root) if manifest is None else manifest
    path = Path(cache_root) / PROPOSAL_DIRECTORY / f"{token}.npz"
    if not path.is_file():
        raise FileNotFoundError(f"proposal cache record not found: {path}")
    with np.load(path, allow_pickle=False) as payload:
        record = ProposalCacheRecord(
            token=str(payload["token"].item()),
            split=str(payload["split"].item()),
            ddp_checkpoint_sha=str(payload["ddp_checkpoint_sha"].item()),
            repository_commit_sha=str(payload["repository_commit_sha"].item()),
            generator_config_hash=str(payload["generator_config_hash"].item()),
            seed=int(payload["seed"].item()),
            candidate_ids=payload["candidate_ids"].copy(),
            trajectory_8=payload["trajectory_8"].copy(),
            trajectory_40=payload["trajectory_40"].copy(),
            target_trajectory_8=payload["target_trajectory_8"].copy(),
        )
    if record.token != token:
        raise ValueError(
            f"proposal cache token {record.token!r} does not match {token!r}"
        )
    record.validate(manifest)
    return record


@dataclass
class TrainingCacheRecord:
    """Final per-token supervision for DrivoR and DriveSuprim training."""

    proposal: ProposalCacheRecord
    dynamic_metrics: Dict[str, np.ndarray]
    static_metrics: Dict[str, np.ndarray]
    dynamic_final_score: np.ndarray
    static_final_score: np.ndarray

    def validate(
        self,
        manifest: CandidateCacheManifest,
        *,
        expected_static_candidates: int = 8192,
    ) -> None:
        self.proposal.validate(manifest)
        dynamic_count = self.proposal.trajectory_8.shape[0]
        expected_schema = tuple(manifest.metric_schema)
        if expected_schema != tuple(TRAINING_METRIC_SCHEMA):
            raise ValueError(
                "training cache metric schema mismatch: "
                f"expected {TRAINING_METRIC_SCHEMA}, got {expected_schema}"
            )
        if tuple(self.dynamic_metrics) != expected_schema:
            raise ValueError("dynamic metric order does not match manifest")
        if tuple(self.static_metrics) != expected_schema:
            raise ValueError("static metric order does not match manifest")
        for name in expected_schema:
            if self.dynamic_metrics[name].shape != (dynamic_count,):
                raise ValueError(f"dynamic metric {name} must have shape [K]")
            if self.static_metrics[name].shape != (expected_static_candidates,):
                raise ValueError(
                    f"static metric {name} must have shape [{expected_static_candidates}]"
                )
        if self.dynamic_final_score.shape != (dynamic_count,):
            raise ValueError("dynamic_final_score must have shape [K]")
        if self.static_final_score.shape != (expected_static_candidates,):
            raise ValueError("static_final_score must match the static vocabulary")
        arrays = (
            *self.dynamic_metrics.values(),
            *self.static_metrics.values(),
            self.dynamic_final_score,
            self.static_final_score,
        )
        if not all(np.isfinite(array).all() for array in arrays):
            raise ValueError("training cache scores contain NaN or Inf")

    def training_targets(self, stage: str) -> Dict[str, Any]:
        """Return the exact nested target contract expected by the planner."""

        target: Dict[str, Any] = {
            "trajectory": self.proposal.target_trajectory_8.astype(
                np.float32, copy=False
            )
        }
        if stage in {"train_drivor", "joint_finetune"}:
            target["drivor_scores"] = {
                name: self.dynamic_metrics[name].astype(np.float32, copy=False)
                for name in REQUIRED_METRICS
            }
        if stage == "train_suprim_static":
            target["coarse_scores"] = {
                name: self.static_metrics[name].astype(np.float32, copy=False)
                for name in SUPRIM_METRICS
            }
        elif stage in {"train_suprim_joint", "joint_finetune"}:
            target["coarse_scores"] = {
                "static": {
                    name: self.static_metrics[name].astype(np.float32, copy=False)
                    for name in SUPRIM_METRICS
                },
                "dynamic": {
                    name: self.dynamic_metrics[name].astype(np.float32, copy=False)
                    for name in SUPRIM_METRICS
                },
            }
        return target


def save_training_record(
    cache_root: os.PathLike[str] | str,
    record: TrainingCacheRecord,
    manifest: CandidateCacheManifest,
) -> Path:
    record.validate(manifest)
    root = Path(cache_root) / TRAINING_RECORD_DIRECTORY
    root.mkdir(parents=True, exist_ok=True)
    token = record.proposal.token
    path = root / f"{token}.npz"
    temporary = root / f".{token}.tmp-{os.getpid()}.npz"
    payload: Dict[str, np.ndarray] = {
        "token": np.asarray(token),
        "split": np.asarray(record.proposal.split),
        "ddp_checkpoint_sha": np.asarray(record.proposal.ddp_checkpoint_sha),
        "repository_commit_sha": np.asarray(record.proposal.repository_commit_sha),
        "generator_config_hash": np.asarray(record.proposal.generator_config_hash),
        "seed": np.asarray(record.proposal.seed, dtype=np.int64),
        "candidate_ids": record.proposal.candidate_ids,
        "trajectory_8": record.proposal.trajectory_8,
        "trajectory_40": record.proposal.trajectory_40,
        "target_trajectory_8": record.proposal.target_trajectory_8,
        "metric_schema": np.asarray(manifest.metric_schema),
        "dynamic_final_score": record.dynamic_final_score,
        "static_final_score": record.static_final_score,
    }
    payload.update(
        {f"dynamic__{name}": value for name, value in record.dynamic_metrics.items()}
    )
    payload.update(
        {f"static__{name}": value for name, value in record.static_metrics.items()}
    )
    np.savez_compressed(temporary, **payload)
    os.replace(temporary, path)
    return path


def load_training_record(
    cache_root: os.PathLike[str] | str,
    token: str,
    *,
    expected_split: str,
    expected_ddp_checkpoint_sha: Optional[str] = None,
    expected_generator_config_hash: Optional[str] = None,
    require_complete: bool = True,
    manifest: Optional[CandidateCacheManifest] = None,
) -> TrainingCacheRecord:
    root = Path(cache_root)
    manifest = read_manifest(root) if manifest is None else manifest
    if require_complete and not (root / COMPLETION_FILE).is_file():
        raise RuntimeError(f"DDP-DRS training cache is incomplete: {root}")
    if manifest.split != expected_split:
        raise ValueError(
            f"candidate cache split {manifest.split!r} does not match {expected_split!r}"
        )
    if (
        expected_ddp_checkpoint_sha is not None
        and manifest.ddp_checkpoint_sha != expected_ddp_checkpoint_sha
    ):
        raise ValueError("candidate cache DDP checkpoint hash mismatch")
    if (
        expected_generator_config_hash is not None
        and manifest.generator_config_hash != expected_generator_config_hash
    ):
        raise ValueError("candidate cache generator config hash mismatch")
    proposal = load_proposal_record(root, token, manifest)
    path = root / TRAINING_RECORD_DIRECTORY / f"{token}.npz"
    if not path.is_file():
        raise FileNotFoundError(f"training cache record not found: {path}")
    with np.load(path, allow_pickle=False) as payload:
        stored_token = str(payload["token"].item())
        if stored_token != token:
            raise ValueError(
                f"training cache token {stored_token!r} does not match {token!r}"
            )
        metric_schema = tuple(str(value) for value in payload["metric_schema"].tolist())
        dynamic_metrics = {
            name: payload[f"dynamic__{name}"].copy() for name in metric_schema
        }
        static_metrics = {
            name: payload[f"static__{name}"].copy() for name in metric_schema
        }
        record = TrainingCacheRecord(
            proposal=proposal,
            dynamic_metrics=dynamic_metrics,
            static_metrics=static_metrics,
            dynamic_final_score=payload["dynamic_final_score"].copy(),
            static_final_score=payload["static_final_score"].copy(),
        )
    record.validate(manifest)
    return record


def mark_training_cache_complete(
    cache_root: os.PathLike[str] | str,
    *,
    tokens: Sequence[str],
) -> Path:
    """Atomically mark a cache complete only after every requested record exists."""

    root = Path(cache_root)
    manifest = read_manifest(root)
    unique_tokens = list(dict.fromkeys(str(token) for token in tokens))
    if len(unique_tokens) != len(tokens):
        raise ValueError("cannot finalize a cache with duplicate datalist tokens")
    missing = [
        token
        for token in unique_tokens
        if not (root / TRAINING_RECORD_DIRECTORY / f"{token}.npz").is_file()
    ]
    if missing:
        raise RuntimeError(
            f"cannot finalize DDP-DRS cache; {len(missing)} records are missing, "
            f"first={missing[:5]}"
        )
    path = root / COMPLETION_FILE
    temporary = root / f".{COMPLETION_FILE}.tmp-{os.getpid()}"
    payload = {
        "schema_version": manifest.schema_version,
        "split": manifest.split,
        "record_count": len(unique_tokens),
        "ddp_checkpoint_sha": manifest.ddp_checkpoint_sha,
        "generator_config_hash": manifest.generator_config_hash,
    }
    with open(temporary, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, path)
    return path
