"""Strict typed records. Scores are always explicitly zero-to-one."""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, Literal
import math
import torch
from .io import ContractError, digest

ROLES = {"incremental_fit", "stage_val", "selector_cal", "dev_report", "final_test"}
TRAIN_ROLES = {"incremental_fit"}
SOURCE_KINDS = {"original", "retrieved", "new_real", "synthetic", "alternative_target"}


def require(condition, message):
    if not condition:
        raise ContractError(message)


def strict_record(cls, value: dict):
    known = {f.name for f in fields(cls)}
    require(not set(value) - known, f"unknown {cls.__name__} fields: {set(value) - known}")
    try:
        return cls(**value)
    except TypeError as e:
        raise ContractError(str(e)) from e


@dataclass(frozen=True)
class SceneRecord:
    schema_version: int
    scene_id: str
    source_log_id: str
    source_group_id: str
    observation_hash: str
    split_role: str
    source_kind: str
    source_scene_id: str
    observation_ref: str
    target_id: str
    target_ref: str
    target_provenance: str
    input_consistency_status: str
    metric_context_ref: str
    metric_context_hash: str
    quality_audit_ref: str
    s0_seen: Literal["yes", "no", "unknown"] = "unknown"
    added_round: int = 0
    parent_group_id: str | None = None
    window_start: float | None = None
    window_end: float | None = None
    consistency_evidence_ref: str | None = None

    def __post_init__(self):
        require(type(self.schema_version) is int and self.schema_version == 1, "unsupported SceneRecord schema")
        require(type(self.added_round) is int and self.added_round >= 0, "source creation round")
        require(self.split_role in ROLES, "unknown split role")
        require(self.source_kind in SOURCE_KINDS, "unknown source kind")
        require(self.target_provenance in {"gt", "external_teacher", "optimized"}, "target provenance")
        require(self.input_consistency_status in {"verified", "rejected", "unverified"}, "consistency status")
        require(self.s0_seen in {"yes", "no", "unknown"}, "S0 exposure status")
        for f in fields(self):
            if f.name in {"scene_id", "source_log_id", "source_group_id", "observation_hash", "observation_ref",
                          "target_id", "target_ref", "source_scene_id", "metric_context_ref", "metric_context_hash"}:
                require(isinstance(getattr(self, f.name), str) and bool(getattr(self, f.name)), f"missing/invalid {f.name}")

    def eligible_input(self, *, external_targets=False):
        require(self.split_role == "incremental_fit", "held-out scene cannot enter training/retrieval")
        require(self.input_consistency_status == "verified", "unverified/rejected observation quarantined")
        if self.source_kind == "synthetic":
            require(bool(self.consistency_evidence_ref), "synthetic data needs collection/render evidence")
        require(external_targets or self.target_provenance == "gt", "external targets explicitly disabled")


@dataclass(frozen=True)
class FeatureRecord:
    schema_version: int
    scene_id: str
    observation_hash: str
    source_context_hash: str
    s0_checkpoint_hash: str
    shared_encoder_hash: str
    tokenizer_hash: str
    prompt_hash: str
    transform_hash: str
    normalizer_hash: str
    camera_time_hash: str
    dtype: str
    precision: str
    augmentation_identity: str
    augmentation_seed: int
    tensor_ref: str
    shapes: dict[str, list[int]]
    valid: bool
    error: str | None
    checksum: str

    def __post_init__(self):
        require(type(self.schema_version) is int and self.schema_version == 1, "feature schema")
        require(type(self.valid) is bool and type(self.augmentation_seed) is int and self.augmentation_seed >= 0, "feature validity/augmentation types")
        require(self.dtype in {"float32", "bfloat16", "float16"}, "feature dtype")
        require(isinstance(self.shapes, dict) and all(isinstance(v,list) and all(type(n) is int and n >= 0 for n in v) for v in self.shapes.values()), "feature shapes")

    @property
    def cache_key(self):
        d = asdict(self)
        for k in ("tensor_ref", "checksum", "valid", "error", "shapes"):
            d.pop(k)
        return digest(d)


@dataclass(frozen=True)
class CandidateRecord:
    schema_version: int
    scene_id: str
    expert_id: str
    expert_checkpoint_hash: str
    feature_contract_hash: str
    trajectory_hash: str
    trajectory_ref: str
    raw_action_representation: str
    physical_frame: str
    horizon: int
    dt: float
    finite_valid: bool
    generation_code_hash: str
    generation_config_hash: str
    generated_at: str

    def __post_init__(self):
        require(type(self.schema_version) is int and self.schema_version == 1 and type(self.horizon) is int and self.horizon > 0 and self.dt > 0, "candidate trajectory contract")
        require(type(self.finite_valid) is bool and all(isinstance(getattr(self,k),str) and getattr(self,k) for k in ("scene_id","expert_id","expert_checkpoint_hash","feature_contract_hash","trajectory_hash","trajectory_ref")), "candidate identity/validity")

    @property
    def key(self):
        return digest({k: getattr(self, k) for k in ("scene_id", "expert_id", "expert_checkpoint_hash",
                      "feature_contract_hash", "trajectory_hash")})


@dataclass(frozen=True)
class ScoreRecord:
    schema_version: int
    candidate_key: str
    score_protocol_hash: str
    metric_context_hash: str
    official_reference_hash: str
    total_score_01: float | None
    named_components: dict[str, float | None]
    component_valid_masks: dict[str, bool]
    scoring_backend: str
    backend_equivalence_version: str
    eval_seed: int
    repetition_index: int
    score_valid: bool
    error_reason: str | None
    first_collision_time: float | None = None
    first_offroad_time: float | None = None

    def __post_init__(self):
        require(type(self.schema_version) is int and self.schema_version == 1, "score schema")
        require(type(self.score_valid) is bool and all(type(v) is bool for v in self.component_valid_masks.values()), "score validity types")
        require(type(self.eval_seed) is int and type(self.repetition_index) is int and self.repetition_index >= 0, "score seed/repetition types")
        require(set(self.named_components) == set(self.component_valid_masks), "named component mask mismatch")
        if self.score_valid:
            require(self.total_score_01 is not None and math.isfinite(self.total_score_01)
                    and 0 <= self.total_score_01 <= 1, "valid score must be explicitly [0,1]")
            require(self.error_reason is None, "valid score cannot have error")
        else:
            require(self.total_score_01 is None and bool(self.error_reason), "failed score is missing, never valid zero")
        for c, valid in self.component_valid_masks.items():
            if valid:
                v = self.named_components[c]
                require(v is not None and math.isfinite(v) and 0 <= v <= 1, f"invalid component {c}")


@dataclass
class FeatureBundle:
    scene: torch.Tensor
    ego: torch.Tensor
    valid_tokens: torch.Tensor  # True means valid; never an implicit padding mask.
    contract_hash: str
    scene_ids: tuple[str, ...]
    conditions: dict[str, torch.Tensor] | None = None

    def __post_init__(self):
        require(self.scene.ndim == 3 and self.ego.ndim in {2, 3}, "feature dimensions")
        require(self.valid_tokens.dtype == torch.bool and self.valid_tokens.shape == self.scene.shape[:2], "feature mask")
        require(len(self.scene_ids) == len(self.scene) == len(self.ego), "feature identity/batch mismatch")
        require(bool(self.valid_tokens.any(-1).all()), "each scene requires valid memory")
        require(bool(torch.isfinite(self.scene).all() and torch.isfinite(self.ego).all()), "nonfinite feature")

    def detached(self):
        # clone outside inference_mode produces normal tensors that can be saved for backward.
        def normal(t):
            return t.detach().clone() if torch.is_inference(t) else t.detach()
        return FeatureBundle(normal(self.scene), normal(self.ego), normal(self.valid_tokens), self.contract_hash,
                             self.scene_ids, {k: normal(v) for k, v in (self.conditions or {}).items()})

    def subset(self, indices):
        ids = indices.detach().cpu().tolist() if isinstance(indices, torch.Tensor) else list(indices)
        return FeatureBundle(self.scene[indices], self.ego[indices], self.valid_tokens[indices], self.contract_hash,
                             tuple(self.scene_ids[i] for i in ids),
                             {k: v[indices] for k, v in (self.conditions or {}).items()})

    def to(self, device):
        return FeatureBundle(self.scene.to(device), self.ego.to(device), self.valid_tokens.to(device),
                             self.contract_hash, self.scene_ids,
                             {k: v.to(device) for k, v in (self.conditions or {}).items()})


@dataclass
class TrajectoryBundle:
    raw: torch.Tensor
    physical: torch.Tensor
    intermediates: tuple[torch.Tensor, ...]
    dt: float


@dataclass(frozen=True)
class TrajectoryContract:
    horizon: int
    dt: float
    raw_representation: str
    frame: str
    origin: str
    yaw_unit: str
    mean: tuple[float, ...]
    scale: tuple[float, ...]
    statistics_source: str

    def __post_init__(self):
        require(self.horizon > 0 and self.dt > 0, "horizon/dt")
        require(self.raw_representation in {"absolute_xy_yaw", "normalized_absolute_xy_yaw", "normalized_xy_sincos"}, "unsupported S0 action representation")
        require(self.frame == "ego_relative" and self.origin == "rear_axle" and self.yaw_unit == "radian", "unsupported physical convention")
        require(len(self.mean) == len(self.scale) == self.raw_dim and all(v > 0 for v in self.scale), "normalizer")

    @property
    def raw_dim(self):
        return 4 if self.raw_representation == "normalized_xy_sincos" else 3

    def physical(self, raw):
        require(raw.shape[-2:] == (self.horizon, self.raw_dim), "trajectory shape disagrees with S0")
        decoded = raw * raw.new_tensor(self.scale) + raw.new_tensor(self.mean)
        if self.raw_dim == 4:
            return torch.cat((decoded[..., :2], torch.atan2(decoded[..., 2:3], decoded[..., 3:4])), -1)
        return decoded

    def raw(self, physical):
        require(physical.shape[-2:] == (self.horizon, 3), "trajectory shape disagrees with S0")
        if self.raw_dim == 4:
            physical = torch.cat((physical[..., :2], physical[..., 2:3].sin(), physical[..., 2:3].cos()), -1)
        return (physical - physical.new_tensor(self.mean)) / physical.new_tensor(self.scale)


def periodic_error(a, b):
    return torch.atan2(torch.sin(a - b), torch.cos(a - b))
