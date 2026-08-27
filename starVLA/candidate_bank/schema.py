"""Versioned schema and validation for offline Register64 candidate banks."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor


CANDIDATE_BANK_SCHEMA_VERSION = 1
LABEL_PROTOCOLS = (
    # Deprecated preview name retained only so an old manifest can be opened
    # and diagnosed. New PDMS training must use the explicit two-way protocol.
    "navsim_v1_pdms",
    "navsim_v1_1_pdms_two_way",
    "navsim_v2_epdms",
)
CANDIDATE_METRICS = (
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "time_to_collision_within_bound",
    "ego_progress",
    "driving_direction_compliance",
    "comfort",
    "lane_keeping",
    "traffic_light_compliance",
    "history_comfort",
    "aggregate_score",
)


@dataclass(frozen=True)
class CandidateBankRecordRef:
    token: str
    rank: int


@dataclass(frozen=True)
class CandidateBankBuildIdentity:
    """Immutable inputs that make an interrupted bank build resumable safely."""

    split: str
    world_size: int
    proposal_num: int
    generator_checkpoint_sha256: str
    generator_config_hash: str
    repository_commit: str
    metric_cache_root: str
    datalist_path: str
    scene_dim: int = 256
    scene_queries: int = 16
    include_dense_memory: bool = False
    storage_dtype: str = "float16"
    label_protocol: str = "navsim_v2_epdms"
    schema_version: int = CANDIDATE_BANK_SCHEMA_VERSION

    def validate(self) -> None:
        if self.schema_version != CANDIDATE_BANK_SCHEMA_VERSION:
            raise RuntimeError("candidate-bank build schema version mismatch")
        if self.split not in {"train", "val", "selection"}:
            raise ValueError(
                "candidate-bank build split must be train, val, or selection"
            )
        if self.world_size <= 0:
            raise ValueError("candidate-bank build world_size must be positive")
        if self.proposal_num <= 0 or self.scene_dim <= 0 or self.scene_queries <= 0:
            raise ValueError("candidate-bank build dimensions must be positive")
        if self.storage_dtype not in {"float16", "bfloat16", "float32"}:
            raise ValueError("unsupported candidate-bank storage dtype")
        if self.label_protocol not in LABEL_PROTOCOLS:
            raise ValueError("unsupported candidate-bank label protocol")
        for name, value in (
            ("generator_checkpoint_sha256", self.generator_checkpoint_sha256),
            ("generator_config_hash", self.generator_config_hash),
            ("repository_commit", self.repository_commit),
            ("metric_cache_root", self.metric_cache_root),
            ("datalist_path", self.datalist_path),
        ):
            if not str(value):
                raise ValueError(f"candidate-bank build identity requires {name}")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CandidateBankBuildIdentity":
        identity = cls(**dict(payload))
        identity.validate()
        return identity


@dataclass
class CandidateBankManifest:
    split: str
    num_scenes: int
    proposal_num: int
    world_size: int
    build_identity_hash: str
    generator_checkpoint_sha256: str
    generator_config_hash: str
    repository_commit: str
    metric_cache_root: str
    scene_dim: int = 256
    scene_queries: int = 16
    include_dense_memory: bool = False
    label_protocol: str = "navsim_v2_epdms"
    metric_schema: Sequence[str] = CANDIDATE_METRICS
    trajectory_coordinate_system: str = "ego_relative_x_y_heading"
    trajectory_horizon: float = 4.0
    trajectory_interval: float = 0.5
    records: list[CandidateBankRecordRef] = field(default_factory=list)
    schema_version: int = CANDIDATE_BANK_SCHEMA_VERSION
    complete: bool = True

    def validate(self) -> None:
        if self.schema_version != CANDIDATE_BANK_SCHEMA_VERSION:
            raise RuntimeError("candidate-bank schema version mismatch")
        if not self.complete:
            raise RuntimeError("candidate-bank manifest is incomplete")
        if not self.split:
            raise ValueError("candidate-bank split cannot be empty")
        if self.world_size <= 0:
            raise ValueError("candidate-bank world_size must be positive")
        if self.proposal_num <= 0 or self.scene_dim <= 0 or self.scene_queries <= 0:
            raise ValueError("candidate-bank dimensions must be positive")
        if self.num_scenes != len(self.records):
            raise ValueError("candidate-bank record count does not match num_scenes")
        tokens = [record.token for record in self.records]
        if any(not token for token in tokens) or len(tokens) != len(set(tokens)):
            raise ValueError("candidate-bank tokens must be unique and non-empty")
        if any(record.rank < 0 or record.rank >= self.world_size for record in self.records):
            raise ValueError("candidate-bank record rank lies outside world_size")
        if tuple(self.metric_schema) != CANDIDATE_METRICS:
            raise ValueError("candidate-bank metric schema differs from v1")
        if self.label_protocol not in LABEL_PROTOCOLS:
            raise ValueError("unsupported candidate-bank label protocol")
        if self.trajectory_coordinate_system != "ego_relative_x_y_heading":
            raise ValueError("unsupported candidate trajectory coordinate system")
        if self.trajectory_horizon != 4.0 or self.trajectory_interval != 0.5:
            raise ValueError("candidate-bank v1 requires 8 poses at 0.5 seconds")
        for name, value in (
            ("build_identity_hash", self.build_identity_hash),
            ("generator_checkpoint_sha256", self.generator_checkpoint_sha256),
            ("generator_config_hash", self.generator_config_hash),
            ("repository_commit", self.repository_commit),
            ("metric_cache_root", self.metric_cache_root),
        ):
            if not str(value):
                raise ValueError(f"candidate-bank manifest requires {name}")
        if len(self.build_identity_hash) != 64:
            raise ValueError("candidate-bank build identity hash must be SHA256")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["metric_schema"] = list(self.metric_schema)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CandidateBankManifest":
        values = dict(payload)
        values["records"] = [
            value
            if isinstance(value, CandidateBankRecordRef)
            else CandidateBankRecordRef(**value)
            for value in values.get("records", [])
        ]
        manifest = cls(**values)
        manifest.validate()
        return manifest


def manifest_hash(manifest: CandidateBankManifest | Mapping[str, Any]) -> str:
    payload = manifest.to_dict() if isinstance(manifest, CandidateBankManifest) else dict(manifest)
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_identity_hash(
    identity: CandidateBankBuildIdentity | Mapping[str, Any],
) -> str:
    payload = identity.to_dict() if isinstance(identity, CandidateBankBuildIdentity) else dict(identity)
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_tensor(
    record: Mapping[str, Any], name: str, shape: tuple[int, ...]
) -> Tensor:
    value = record.get(name)
    if not torch.is_tensor(value):
        raise TypeError(f"candidate-bank field {name!r} must be a tensor")
    if tuple(value.shape) != shape:
        raise ValueError(
            f"candidate-bank field {name!r} has shape {tuple(value.shape)}, "
            f"expected {shape}"
        )
    if value.is_floating_point() and not torch.isfinite(value).all():
        raise ValueError(f"candidate-bank field {name!r} contains NaN or Inf")
    return value


def validate_candidate_record(
    record: Mapping[str, Any],
    *,
    proposal_num: int = 64,
    scene_queries: int = 16,
    scene_dim: int = 256,
    include_dense_memory: bool = False,
) -> None:
    token = record.get("token")
    if not isinstance(token, str) or not token:
        raise ValueError("candidate-bank record requires a non-empty token")
    ego_state = _require_tensor(record, "ego_state", (4,))
    scene_tokens = _require_tensor(
        record, "scene_global_tokens", (scene_queries, scene_dim)
    )
    proposals = _require_tensor(record, "proposals", (proposal_num, 8, 3))
    gt_trajectory = _require_tensor(record, "gt_trajectory", (8, 3))
    if ego_state.dtype != torch.float32 or gt_trajectory.dtype != torch.float32:
        raise TypeError("ego_state and gt_trajectory must use float32")
    if scene_tokens.dtype not in {torch.float16, torch.bfloat16, torch.float32}:
        raise TypeError("scene_global_tokens must use fp16, bf16, or fp32")
    if proposals.dtype not in {torch.float16, torch.bfloat16, torch.float32}:
        raise TypeError("proposals must use fp16, bf16, or fp32")
    metrics = record.get("metrics")
    if not isinstance(metrics, Mapping) or set(metrics) != set(CANDIDATE_METRICS):
        raise ValueError("candidate-bank metrics do not match the v1 schema")
    for name in CANDIDATE_METRICS:
        value = metrics[name]
        if not torch.is_tensor(value) or tuple(value.shape) != (proposal_num,):
            raise ValueError(f"candidate metric {name!r} must have shape [{proposal_num}]")
        if value.dtype != torch.float32:
            raise TypeError(f"candidate metric {name!r} must use float32")
        if not torch.isfinite(value).all():
            raise ValueError(f"candidate metric {name!r} contains NaN or Inf")
    has_dense = "scene_dense_memory" in record or "attention_mask" in record
    if include_dense_memory and not has_dense:
        raise ValueError("dense-memory candidate bank record is incomplete")
    if not include_dense_memory and has_dense:
        raise ValueError("dense memory is forbidden by this candidate-bank manifest")
    if has_dense:
        dense = record.get("scene_dense_memory")
        mask = record.get("attention_mask")
        if not torch.is_tensor(dense) or dense.ndim != 2 or dense.shape[-1] != scene_dim:
            raise ValueError(f"scene_dense_memory must have shape [L,{scene_dim}]")
        if dense.dtype not in {torch.float16, torch.bfloat16, torch.float32}:
            raise TypeError("scene_dense_memory must use fp16, bf16, or fp32")
        if not torch.is_tensor(mask) or mask.dtype is not torch.bool:
            raise TypeError("attention_mask must be a boolean tensor")
        if tuple(mask.shape) != (dense.shape[0],):
            raise ValueError("attention_mask length does not match dense memory")


def estimate_record_bytes(
    *,
    proposal_num: int = 64,
    scene_queries: int = 16,
    scene_dim: int = 256,
    proposal_bytes: int = 2,
    scene_bytes: int = 2,
) -> int:
    """Return the raw tensor payload size without LMDB/torch serialization overhead."""

    return (
        4 * 4
        + scene_queries * scene_dim * scene_bytes
        + proposal_num * 8 * 3 * proposal_bytes
        + 8 * 3 * 4
        + len(CANDIDATE_METRICS) * proposal_num * 4
    )
