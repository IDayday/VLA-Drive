"""Explicit configuration schema for the opt-in DDP-DRS baseline."""

from __future__ import annotations

from dataclasses import dataclass, field
from os import PathLike
from typing import Any, Mapping, Optional


TRAINING_STAGES = (
    "cache_candidates",
    "train_drivor",
    "train_suprim_static",
    "train_suprim_joint",
    "joint_finetune",
    "inference",
)


def _to_plain_mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    """Convert an OmegaConf/dict/dataclass-like node without loose getattr use."""

    if value is None:
        return {}
    if isinstance(value, Mapping):
        return value
    items = getattr(value, "items", None)
    if callable(items):
        return dict(items())
    values = getattr(value, "__dict__", None)
    if isinstance(values, Mapping):
        return values
    raise TypeError(f"{name} must be a mapping, got {type(value).__name__}")


def _optional_path(value: Any, *, name: str) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, (str, PathLike)):
        raise TypeError(f"{name} must be a path string or null")
    path = str(value).strip()
    return path or None


@dataclass(frozen=True)
class SceneCompressorConfig:
    """Full-Qwen-sequence global scene Q-Former configuration."""

    type: str = "drivevla_global_qformer"
    source_layer: int = -1
    source_token_policy: str = "all_valid_tokens"
    num_queries: int = 16
    scene_dim: int = 2048
    num_layers: int = 4
    num_heads: int = 32
    ffn_dim: int = 8192
    dropout: float = 0.0
    query_init_std: float = 1e-6
    use_padding_mask: bool = True
    detach_qwen_memory: bool = True
    return_dense_memory: bool = True
    checkpoint_path: Optional[str] = None
    debug_validate_finite: bool = False

    @classmethod
    def from_mapping(cls, value: Any) -> "SceneCompressorConfig":
        cfg = _to_plain_mapping(value, name="multi_trajectory.scene_compressor")
        result = cls(
            type=str(cfg.get("type", cls.type)),
            source_layer=int(cfg.get("source_layer", cls.source_layer)),
            source_token_policy=str(
                cfg.get("source_token_policy", cls.source_token_policy)
            ),
            num_queries=int(cfg.get("num_queries", cls.num_queries)),
            scene_dim=int(cfg.get("scene_dim", cls.scene_dim)),
            num_layers=int(cfg.get("num_layers", cls.num_layers)),
            num_heads=int(cfg.get("num_heads", cls.num_heads)),
            ffn_dim=int(cfg.get("ffn_dim", cls.ffn_dim)),
            dropout=float(cfg.get("dropout", cls.dropout)),
            query_init_std=float(cfg.get("query_init_std", cls.query_init_std)),
            use_padding_mask=bool(
                cfg.get("use_padding_mask", cls.use_padding_mask)
            ),
            detach_qwen_memory=bool(
                cfg.get("detach_qwen_memory", cls.detach_qwen_memory)
            ),
            return_dense_memory=bool(
                cfg.get("return_dense_memory", cls.return_dense_memory)
            ),
            checkpoint_path=_optional_path(
                cfg.get("checkpoint_path"),
                name="multi_trajectory.scene_compressor.checkpoint_path",
            ),
            debug_validate_finite=bool(
                cfg.get("debug_validate_finite", cls.debug_validate_finite)
            ),
        )
        result.validate()
        return result

    def validate(self) -> None:
        if self.type != "drivevla_global_qformer":
            raise ValueError("scene_compressor.type must be drivevla_global_qformer")
        if self.source_layer != -1:
            raise ValueError("the fidelity configuration requires source_layer=-1")
        if self.source_token_policy != "all_valid_tokens":
            raise ValueError("scene compressor must consume all valid Qwen tokens")
        if self.num_queries != 16:
            raise ValueError("global scene Q-Former requires num_queries=16")
        if self.scene_dim != 2048:
            raise ValueError("global scene Q-Former requires scene_dim=2048")
        if self.num_layers != 4:
            raise ValueError("global scene Q-Former requires num_layers=4")
        if self.num_heads != 32:
            raise ValueError("global scene Q-Former requires num_heads=32")
        if self.scene_dim % self.num_heads or self.scene_dim // self.num_heads != 64:
            raise ValueError("scene attention must use 32 heads with head_dim=64")
        if self.ffn_dim != 8192:
            raise ValueError("global scene Q-Former requires ffn_dim=8192")
        if self.dropout < 0.0 or self.query_init_std < 0.0:
            raise ValueError("dropout and query_init_std cannot be negative")
        if not self.use_padding_mask or not self.return_dense_memory:
            raise ValueError("padding masks and dense scene memory must remain enabled")


@dataclass(frozen=True)
class PlanningConfig:
    """Trajectory-candidate transformer width, separate from scene width."""

    planning_dim: int = 256
    num_heads: int = 8
    ffn_dim: int = 1024
    dropout: float = 0.0

    @classmethod
    def from_mapping(cls, value: Any) -> "PlanningConfig":
        cfg = _to_plain_mapping(value, name="multi_trajectory.planning")
        result = cls(
            planning_dim=int(cfg.get("planning_dim", cls.planning_dim)),
            num_heads=int(cfg.get("num_heads", cls.num_heads)),
            ffn_dim=int(cfg.get("ffn_dim", cls.ffn_dim)),
            dropout=float(cfg.get("dropout", cls.dropout)),
        )
        result.validate()
        return result

    def validate(self) -> None:
        if self.planning_dim != 256:
            raise ValueError("trajectory planning_dim must remain 256")
        if self.num_heads != 8:
            raise ValueError("planning attention requires num_heads=8")
        if self.planning_dim % self.num_heads or self.planning_dim // self.num_heads != 32:
            raise ValueError("planning attention must use head_dim=32")
        if self.ffn_dim != 1024:
            raise ValueError("planning ffn_dim must remain 1024")
        if self.dropout < 0.0:
            raise ValueError("planning dropout cannot be negative")


@dataclass(frozen=True)
class DrivoRConfig:
    """DrivoR dynamic scorer configuration; no image encoder is present."""

    scorer_layers: int = 4
    dynamic_topk: int = 16
    noc_weight: float = 1.0
    dac_weight: float = 1.0
    ddc_weight: float = 0.0
    ttc_weight: float = 5.0
    ep_weight: float = 5.0
    comfort_weight: float = 2.0
    checkpoint_path: Optional[str] = None
    ego_status_dim: Optional[int] = None
    debug_validate_finite: bool = False

    @classmethod
    def from_mapping(cls, value: Any) -> "DrivoRConfig":
        cfg = _to_plain_mapping(value, name="multi_trajectory.drivor")
        result = cls(
            scorer_layers=int(cfg.get("scorer_layers", cls.scorer_layers)),
            dynamic_topk=int(cfg.get("dynamic_topk", cls.dynamic_topk)),
            noc_weight=float(cfg.get("noc_weight", cls.noc_weight)),
            dac_weight=float(cfg.get("dac_weight", cls.dac_weight)),
            ddc_weight=float(cfg.get("ddc_weight", cls.ddc_weight)),
            ttc_weight=float(cfg.get("ttc_weight", cls.ttc_weight)),
            ep_weight=float(cfg.get("ep_weight", cls.ep_weight)),
            comfort_weight=float(cfg.get("comfort_weight", cls.comfort_weight)),
            checkpoint_path=_optional_path(
                cfg.get("checkpoint_path"),
                name="multi_trajectory.drivor.checkpoint_path",
            ),
            ego_status_dim=(
                None if cfg.get("ego_status_dim") is None else int(cfg["ego_status_dim"])
            ),
            debug_validate_finite=bool(
                cfg.get("debug_validate_finite", cls.debug_validate_finite)
            ),
        )
        result.validate()
        return result

    def validate(self) -> None:
        if self.scorer_layers != 4:
            raise ValueError("DrivoR fidelity configuration requires scorer_layers=4")
        if self.dynamic_topk != 16:
            raise ValueError("DrivoR fidelity configuration requires dynamic_topk=16")


@dataclass(frozen=True)
class SuprimConfig:
    """DriveSuprim static vocabulary, coarse selection, and refinement config."""

    vocab_path: Optional[str] = None
    vocab_size: int = 8192
    num_trajectory_points: int = 40
    coarse_topk: int = 256
    coarse_memory_source: str = "global_scene_tokens"
    fine_memory_source: str = "dense_qwen_memory"
    coarse_layers: int = 3
    num_refinement_stages: int = 1
    refinement_layers: int = 3
    use_mid_output: bool = True
    use_separate_heads: bool = True
    use_imitation_head: bool = True
    checkpoint_path: Optional[str] = None
    normalize_vocab_pos: bool = False
    imitation_sigma: float = 0.5
    debug_validate_finite: bool = False

    @classmethod
    def from_mapping(cls, value: Any) -> "SuprimConfig":
        cfg = _to_plain_mapping(value, name="multi_trajectory.suprim")
        result = cls(
            vocab_path=_optional_path(
                cfg.get("vocab_path"), name="multi_trajectory.suprim.vocab_path"
            ),
            vocab_size=int(cfg.get("vocab_size", cls.vocab_size)),
            num_trajectory_points=int(
                cfg.get("num_trajectory_points", cls.num_trajectory_points)
            ),
            coarse_topk=int(cfg.get("coarse_topk", cls.coarse_topk)),
            coarse_memory_source=str(
                cfg.get("coarse_memory_source", cls.coarse_memory_source)
            ),
            fine_memory_source=str(
                cfg.get("fine_memory_source", cls.fine_memory_source)
            ),
            coarse_layers=int(cfg.get("coarse_layers", cls.coarse_layers)),
            num_refinement_stages=int(
                cfg.get("num_refinement_stages", cls.num_refinement_stages)
            ),
            refinement_layers=int(
                cfg.get("refinement_layers", cls.refinement_layers)
            ),
            use_mid_output=bool(cfg.get("use_mid_output", cls.use_mid_output)),
            use_separate_heads=bool(
                cfg.get("use_separate_heads", cls.use_separate_heads)
            ),
            use_imitation_head=bool(
                cfg.get("use_imitation_head", cls.use_imitation_head)
            ),
            checkpoint_path=_optional_path(
                cfg.get("checkpoint_path"),
                name="multi_trajectory.suprim.checkpoint_path",
            ),
            normalize_vocab_pos=bool(
                cfg.get("normalize_vocab_pos", cls.normalize_vocab_pos)
            ),
            imitation_sigma=float(cfg.get("imitation_sigma", cls.imitation_sigma)),
            debug_validate_finite=bool(
                cfg.get("debug_validate_finite", cls.debug_validate_finite)
            ),
        )
        result.validate()
        return result

    def validate(self) -> None:
        if self.vocab_size != 8192:
            raise ValueError("DriveSuprim fidelity configuration requires vocab_size=8192")
        if self.num_trajectory_points != 40:
            raise ValueError("DriveSuprim trajectories must contain 40 points")
        if self.coarse_topk != 256:
            raise ValueError("DriveSuprim fidelity configuration requires global Top-256")
        if self.coarse_memory_source != "global_scene_tokens":
            raise ValueError("DriveSuprim coarse memory must use global_scene_tokens")
        if self.fine_memory_source not in {
            "dense_qwen_memory",
            "global_scene_tokens",
        }:
            raise ValueError("unsupported DriveSuprim fine_memory_source")
        if self.coarse_layers <= 0:
            raise ValueError("DriveSuprim coarse_layers must be positive")
        if self.num_refinement_stages != 1 or self.refinement_layers != 3:
            raise ValueError("DriveSuprim requires one three-layer refinement stage")
        if not self.use_mid_output or not self.use_separate_heads or not self.use_imitation_head:
            raise ValueError(
                "DriveSuprim fidelity mode requires mid outputs, separate heads, and imitation"
            )
        if self.normalize_vocab_pos:
            raise ValueError("candidate interaction/normalization is disabled")


@dataclass(frozen=True)
class LossConfig:
    """Explicitly disables excluded auxiliary/generator losses."""

    diversity_weight: float = 0.0
    ranking_weight: float = 0.0
    generator_score_weight: float = 0.0

    @classmethod
    def from_mapping(cls, value: Any) -> "LossConfig":
        cfg = _to_plain_mapping(value, name="multi_trajectory.loss")
        result = cls(
            diversity_weight=float(cfg.get("diversity_weight", 0.0)),
            ranking_weight=float(cfg.get("ranking_weight", 0.0)),
            generator_score_weight=float(cfg.get("generator_score_weight", 0.0)),
        )
        if any(
            weight != 0.0
            for weight in (
                result.diversity_weight,
                result.ranking_weight,
                result.generator_score_weight,
            )
        ):
            raise ValueError(
                "DDP-DRS forbids diversity, ranking, and generator-score losses"
            )
        return result


@dataclass(frozen=True)
class CandidateCacheConfig:
    """Offline candidate/metric cache used only by supervised training stages."""

    root_path: Optional[str] = None
    expected_ddp_checkpoint_sha: Optional[str] = None
    expected_generator_config_hash: Optional[str] = None
    require_complete: bool = True

    @classmethod
    def from_mapping(cls, value: Any) -> "CandidateCacheConfig":
        cfg = _to_plain_mapping(value, name="multi_trajectory.cache")
        return cls(
            root_path=_optional_path(
                cfg.get("root_path"), name="multi_trajectory.cache.root_path"
            ),
            expected_ddp_checkpoint_sha=(
                None
                if cfg.get("expected_ddp_checkpoint_sha") is None
                else str(cfg["expected_ddp_checkpoint_sha"]).strip() or None
            ),
            expected_generator_config_hash=(
                None
                if cfg.get("expected_generator_config_hash") is None
                else str(cfg["expected_generator_config_hash"]).strip() or None
            ),
            require_complete=bool(cfg.get("require_complete", True)),
        )


@dataclass(frozen=True)
class MultiTrajectoryConfig:
    """Top-level, opt-in DDP with DrivoR scoring and Suprim refinement."""

    enabled: bool = False
    num_dynamic_candidates: int = 64
    deterministic_seed: Optional[int] = None
    scene_compressor: SceneCompressorConfig = field(
        default_factory=SceneCompressorConfig
    )
    planning: PlanningConfig = field(default_factory=PlanningConfig)
    drivor: DrivoRConfig = field(default_factory=DrivoRConfig)
    suprim: SuprimConfig = field(default_factory=SuprimConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    cache: CandidateCacheConfig = field(default_factory=CandidateCacheConfig)
    training_stage: str = "inference"
    strict_inference: bool = True
    smoke_test_fallback_to_single_ddp: bool = False
    diagnostics_enabled: bool = False

    @classmethod
    def from_mapping(cls, value: Any) -> "MultiTrajectoryConfig":
        cfg = _to_plain_mapping(value, name="multi_trajectory")
        result = cls(
            enabled=bool(cfg.get("enabled", False)),
            num_dynamic_candidates=int(cfg.get("num_dynamic_candidates", 64)),
            deterministic_seed=(
                None
                if cfg.get("deterministic_seed") is None
                else int(cfg["deterministic_seed"])
            ),
            scene_compressor=SceneCompressorConfig.from_mapping(
                cfg.get("scene_compressor")
            ),
            planning=PlanningConfig.from_mapping(cfg.get("planning")),
            drivor=DrivoRConfig.from_mapping(cfg.get("drivor")),
            suprim=SuprimConfig.from_mapping(cfg.get("suprim")),
            loss=LossConfig.from_mapping(cfg.get("loss")),
            cache=CandidateCacheConfig.from_mapping(cfg.get("cache")),
            training_stage=str(cfg.get("training_stage", "inference")),
            strict_inference=bool(cfg.get("strict_inference", True)),
            smoke_test_fallback_to_single_ddp=bool(
                cfg.get("smoke_test_fallback_to_single_ddp", False)
            ),
            diagnostics_enabled=bool(cfg.get("diagnostics_enabled", False)),
        )
        result.validate()
        return result

    @classmethod
    def from_full_config(cls, full_config: Any) -> "MultiTrajectoryConfig":
        root = _to_plain_mapping(full_config, name="config")
        return cls.from_mapping(root.get("multi_trajectory"))

    def validate(self) -> None:
        self.scene_compressor.validate()
        self.planning.validate()
        self.drivor.validate()
        self.suprim.validate()
        if self.training_stage not in TRAINING_STAGES:
            raise ValueError(
                f"training_stage must be one of {TRAINING_STAGES}, got "
                f"{self.training_stage!r}"
            )
        if self.num_dynamic_candidates <= 0:
            raise ValueError("num_dynamic_candidates must be positive")
        if self.drivor.dynamic_topk > self.num_dynamic_candidates:
            raise ValueError("DrivoR dynamic_topk cannot exceed generated candidates")
        if self.suprim.coarse_topk > self.suprim.vocab_size + self.drivor.dynamic_topk:
            raise ValueError("DriveSuprim coarse_topk exceeds the joint candidate pool")
        if self.strict_inference and self.smoke_test_fallback_to_single_ddp:
            raise ValueError(
                "strict_inference cannot silently fall back; disable it for smoke tests"
            )

    @property
    def dynamic_candidates_enabled(self) -> bool:
        return self.training_stage != "train_suprim_static"


def multi_trajectory_enabled(full_config: Any) -> bool:
    """Read only the opt-in bit without constructing any torch module."""

    root = _to_plain_mapping(full_config, name="config")
    cfg = _to_plain_mapping(root.get("multi_trajectory"), name="multi_trajectory")
    return bool(cfg.get("enabled", False))
