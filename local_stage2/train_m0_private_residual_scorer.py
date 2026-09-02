#!/usr/bin/env python3
"""Train an M0-native scorer-private residual on frozen proposal replay."""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from local_stage2.train_independent_scorer import (
    ReplaySource,
    ReplayTensorSet,
    TARGET_FACTOR_KEYS,
    TARGET_TO_MODEL_FACTOR_ORDER,
    _atomic_json_dump,
    _atomic_torch_save,
    _build_sampler,
    _gather_candidates,
    _iter_joined_chunks,
    _mean_details,
    _pairwise_accuracy,
    _sha256,
    load_replay_sources,
)
from local_stage2.train_public_base_residual_scorer import (
    _log_bootstrap_ci,
    base_pairwise_loss,
    expected_regret_loss,
    listwise_loss,
    relative_safety_targets,
    top_set_cross_entropy,
    weighted_pairwise_loss,
)
from navsim.agents.EpisodeDrive.score_module.independent_ranker import (
    FACTOR_KEYS,
    IndependentRankerConfig,
    candidate_relative_consequence_loss,
    current_actor_auxiliary_loss,
    episode_drive_factor_loss,
    masked_pinball_quantile_loss,
    pdms_factor_log_utility,
    shared_future_auxiliary_loss,
    top_regret_rank_loss,
    weighted_pairwise_rank_loss,
)
from navsim.agents.EpisodeDrive.score_module.m0_private_residual_ranker import (
    M0PrivateResidualConfig,
    M0PrivateResidualRanker,
    base_anchored_topk_indices,
)


_M0_REFIT_LOCKED_ARGUMENTS = (
    "seed",
    "batch_size",
    "learning_rate",
    "weight_decay",
    "model_dim",
    "dynamic_queries",
    "private_layers",
    "trajectory_layers",
    "candidate_layers",
    "fine_layers",
    "private_fine_top_k",
    "residual_layers",
    "m0_context_fusion",
    "m0_candidate_fusion",
    "m0_candidate_only",
    "conservative_reference",
    "reference_hidden_dim",
    "reference_layers",
    "reference_gain_quantile_index",
    "reference_minimum_lcb_gain",
    "reference_maximum_safety_worse_probability",
    "reference_minimum_safe_improvement_probability",
    "residual_top_k",
    "score_mode",
    "max_residual",
    "dropout",
    "minimum_pair_delta",
    "factor_rank_minimum_delta",
    "target_temperature",
    "prediction_temperature",
    "top_set_tolerance",
    "pairwise_weight",
    "base_pairwise_weight",
    "listwise_weight",
    "top_set_weight",
    "expected_regret_weight",
    "top_regret_weight",
    "top_regret_minimum_delta",
    "factor_weight",
    "private_factor_weight",
    "factor_rank_weight",
    "relative_safety_weight",
    "residual_l2_weight",
    "reference_weight",
    "reference_quantile_weight",
    "reference_median_rank_weight",
    "reference_safety_weight",
    "reference_improvement_weight",
    "reference_false_switch_weight",
    "reference_missed_improvement_weight",
    "reference_safety_worse_positive_weight",
    "reference_safe_improvement_positive_weight",
    "reference_switch_margin_temperature",
    "reference_minimum_improvement_target",
    "reference_factor_epsilon",
    "shared_future_weight",
    "current_actor_weight",
    "candidate_relative_weight",
    "safety_negative_weight",
    "factor_loss_scope",
    "shared_future_relabeling",
    "shared_future_constant_velocity_residual",
)
_M0_LEGACY_REFIT_ARGUMENT_DEFAULTS = {
    # Wave 1--3 artifacts predate these explicit loss switches but have these
    # exact semantics.  Missing fields may only resolve to these constants.
    "top_regret_weight": 0.0,
    "top_regret_minimum_delta": 0.01,
    "factor_loss_scope": "all",
    "conservative_reference": False,
    "reference_hidden_dim": 512,
    "reference_layers": 2,
    "reference_gain_quantile_index": 1,
    "reference_minimum_lcb_gain": 0.0,
    "reference_maximum_safety_worse_probability": 0.1,
    "reference_minimum_safe_improvement_probability": 0.7,
    "reference_weight": 0.0,
    "reference_quantile_weight": 1.0,
    "reference_median_rank_weight": 0.25,
    "reference_safety_weight": 1.0,
    "reference_improvement_weight": 0.5,
    "reference_false_switch_weight": 0.5,
    "reference_missed_improvement_weight": 0.0,
    "reference_safety_worse_positive_weight": 10.0,
    "reference_safe_improvement_positive_weight": 3.0,
    "reference_switch_margin_temperature": 0.05,
    "reference_minimum_improvement_target": 0.005,
    "reference_factor_epsilon": 1.0e-6,
}
_M0_DEPLOYMENT_ONLY_RESIDUAL_FIELDS = (
    "inference_scale",
    "switch_penalty",
    "safety_floor",
    "safety_relative_tolerance",
    "preserve_ddc",
    "safety_gate_mode",
)


def validate_m0_all_log_refit_provenance(
    selected: Mapping[str, object],
    args: argparse.Namespace,
    private_config: IndependentRankerConfig,
    residual_config: M0PrivateResidualConfig,
) -> Dict[str, object]:
    """Lock an all-log refit to one held-out-log-selected M0 ranker.

    The refit may see the former validation logs, but it cannot change the
    architecture, objective, seed, batch size, stop epoch, scheduler horizon,
    or deployment calibration selected without Navtest.
    """

    if selected.get("architecture") != "M0PrivateResidualRanker":
        raise RuntimeError("M0 refit selection artifact has the wrong architecture")
    if bool(selected.get("refit_all_logs")):
        raise RuntimeError("an M0 all-log refit cannot select another refit")
    source = str(selected.get("checkpoint_selection_source"))
    validation_by_source = selected.get("validation_by_source")
    if not isinstance(validation_by_source, Mapping) or source not in validation_by_source:
        raise RuntimeError("M0 refit artifact lacks held-out source metrics")
    metrics = validation_by_source[source]
    if not isinstance(metrics, Mapping):
        raise RuntimeError("M0 refit validation metrics have the wrong schema")
    interval = metrics.get("selected_delta_log_bootstrap_95ci")
    if not isinstance(interval, Sequence) or len(interval) != 2:
        raise RuntimeError("M0 refit artifact lacks a log-bootstrap interval")
    if float(interval[0]) <= 0.0:
        raise RuntimeError("M0 refit artifact did not pass the held-out CI gate")
    if "policy_calibration" in selected:
        if bool(selected.get("policy_selection_uses_navtest", True)):
            raise RuntimeError("M0 refit calibration must not use Navtest")
        if not bool(selected.get("policy_selection_uses_disjoint_physical_logs")):
            raise RuntimeError(
                "M0 refit calibration was not evaluated on disjoint physical logs"
            )

    if selected.get("private_config") != asdict(private_config):
        raise RuntimeError("M0 refit private configuration differs from selection")
    selected_residual_config = selected.get("residual_config")
    if not isinstance(selected_residual_config, Mapping):
        raise RuntimeError("M0 refit artifact lacks its residual configuration")
    selected_training_residual = dict(selected_residual_config)
    current_training_residual = asdict(residual_config)
    for name in _M0_DEPLOYMENT_ONLY_RESIDUAL_FIELDS:
        selected_training_residual[name] = current_training_residual[name]
    if selected_training_residual != current_training_residual:
        raise RuntimeError("M0 refit residual configuration differs from selection")
    fold_manifest = selected.get("fold_manifest")
    if not isinstance(fold_manifest, Mapping):
        raise RuntimeError("M0 refit artifact lacks fold lineage")
    selected_args = fold_manifest.get("args")
    if not isinstance(selected_args, Mapping):
        raise RuntimeError("M0 refit artifact lacks locked training arguments")
    selected_train_logs = {
        str(value) for value in fold_manifest.get("train_physical_logs", ())
    }
    selected_validation_logs = {
        str(value) for value in fold_manifest.get("validation_physical_logs", ())
    }
    if not selected_train_logs or not selected_validation_logs:
        raise RuntimeError("M0 refit selection did not use a held-out log split")
    if selected_train_logs.intersection(selected_validation_logs):
        raise RuntimeError("M0 refit selection artifact has physical-log leakage")

    missing = object()
    resolved_selected_args = {
        name: selected_args.get(
            name,
            _M0_LEGACY_REFIT_ARGUMENT_DEFAULTS.get(name, missing),
        )
        for name in _M0_REFIT_LOCKED_ARGUMENTS
    }
    unresolved = [
        name for name, value in resolved_selected_args.items() if value is missing
    ]
    if unresolved:
        raise RuntimeError(
            f"M0 refit artifact lacks locked arguments: {sorted(unresolved)}"
        )
    mismatches = {
        name: (getattr(args, name), resolved_selected_args[name])
        for name in _M0_REFIT_LOCKED_ARGUMENTS
        if getattr(args, name) != resolved_selected_args[name]
    }
    if mismatches:
        raise RuntimeError(f"M0 refit arguments differ from selection: {mismatches}")
    selected_epoch = int(selected.get("epoch", -1))
    if selected_epoch < 0 or args.epochs != selected_epoch + 1:
        raise RuntimeError(
            "M0 refit epochs must equal the selected zero-based epoch plus one"
        )
    scheduler_horizon = int(selected_args.get("epochs", 0))
    if scheduler_horizon < args.epochs:
        raise RuntimeError("M0 refit scheduler horizon is shorter than stop epoch")
    return {
        "selection_source": source,
        "selected_epoch": selected_epoch,
        "selected_validation_pdms": float(metrics["selected_pdms"]),
        "selected_validation_base_pdms": float(metrics["base_selected_pdms"]),
        "selected_validation_delta": float(metrics["selected_delta"]),
        "selected_delta_log_bootstrap_95ci": [
            float(interval[0]),
            float(interval[1]),
        ],
        "selected_validation_scene_count": int(metrics["scene_count"]),
        "selected_validation_physical_log_count": int(
            metrics["physical_log_count"]
        ),
        "scheduler_horizon_epochs": scheduler_horizon,
        "locked_training_arguments": resolved_selected_args,
        "selection_train_physical_log_count": len(selected_train_logs),
        "selection_validation_physical_log_count": len(selected_validation_logs),
        "deployment_policy_frozen_from_selection": "policy_calibration" in selected,
        "deployment_residual_config": dict(selected_residual_config),
    }


def load_replay_base_factor_logits(
    sources: Sequence[ReplaySource],
    *,
    max_scenes_per_source: int = 0,
) -> Tuple[List[str], torch.Tensor]:
    """Load only deployable M0 factor logits in replay-loader row order."""

    tokens: List[str] = []
    parts: List[torch.Tensor] = []
    for source in sources:
        source_count = 0
        seen: set[str] = set()
        for feature_path, label_path in _iter_joined_chunks(source):
            features = torch.load(
                feature_path,
                map_location="cpu",
                weights_only=False,
            )
            labels = torch.load(
                label_path,
                map_location="cpu",
                weights_only=False,
            )
            feature_tokens = [str(value) for value in features["tokens"]]
            label_tokens = [str(value) for value in labels["tokens"]]
            if feature_tokens != label_tokens:
                raise RuntimeError(f"feature/label token mismatch: {feature_path}")
            if tuple(features["factor_keys"]) != FACTOR_KEYS:
                raise RuntimeError(f"unexpected Base factor schema: {feature_path}")
            if tuple(labels["target_factor_keys"]) != TARGET_FACTOR_KEYS:
                raise RuntimeError(f"unexpected target factor schema: {label_path}")
            valid = labels["valid_mask"].bool()
            remaining = (
                max_scenes_per_source - source_count
                if max_scenes_per_source > 0
                else len(feature_tokens)
            )
            if remaining <= 0:
                break
            indices = torch.nonzero(valid, as_tuple=False).squeeze(-1)[:remaining]
            selected_tokens = [feature_tokens[int(index)] for index in indices]
            duplicate = seen.intersection(selected_tokens)
            if duplicate:
                raise RuntimeError(
                    f"duplicate factor-logit tokens: {sorted(duplicate)[:3]}"
                )
            seen.update(selected_tokens)
            factor_logits = features["factor_logits"][indices].float()
            if factor_logits.shape[1:] != (64, len(FACTOR_KEYS)):
                raise RuntimeError(
                    f"unexpected Base factor-logit shape: {factor_logits.shape}"
                )
            if not torch.isfinite(factor_logits).all():
                raise RuntimeError("Base factor logits contain non-finite values")
            tokens.extend(selected_tokens)
            parts.append(factor_logits)
            source_count += len(selected_tokens)
            if max_scenes_per_source > 0 and source_count >= max_scenes_per_source:
                break
        if source_count == 0:
            raise RuntimeError(f"source {source.name} has no Base factor logits")
    return tokens, torch.cat(parts)


def load_replay_base_candidate_features(
    sources: Sequence[ReplaySource],
    *,
    max_scenes_per_source: int = 0,
) -> Tuple[List[str], torch.Tensor]:
    """Load M0's deployable candidate-conditioned hidden states in row order."""

    tokens: List[str] = []
    parts: List[torch.Tensor] = []
    expected_shape: Optional[Tuple[int, int]] = None
    for source in sources:
        source_count = 0
        seen: set[str] = set()
        for feature_path, label_path in _iter_joined_chunks(source):
            features = torch.load(
                feature_path,
                map_location="cpu",
                weights_only=False,
            )
            labels = torch.load(
                label_path,
                map_location="cpu",
                weights_only=False,
            )
            feature_tokens = [str(value) for value in features["tokens"]]
            if feature_tokens != [str(value) for value in labels["tokens"]]:
                raise RuntimeError(f"feature/label token mismatch: {feature_path}")
            valid = labels["valid_mask"].bool()
            remaining = (
                max_scenes_per_source - source_count
                if max_scenes_per_source > 0
                else len(feature_tokens)
            )
            if remaining <= 0:
                break
            indices = torch.nonzero(valid, as_tuple=False).squeeze(-1)[:remaining]
            selected_tokens = [feature_tokens[int(index)] for index in indices]
            duplicate = seen.intersection(selected_tokens)
            if duplicate:
                raise RuntimeError(
                    f"duplicate candidate-feature tokens: {sorted(duplicate)[:3]}"
                )
            seen.update(selected_tokens)
            candidate = features.get("candidate_features")
            if candidate is None:
                raise RuntimeError(
                    f"M0 candidate features are absent from {feature_path}"
                )
            candidate = candidate[indices]
            shape = tuple(candidate.shape[1:])
            if len(shape) != 2 or shape[0] != 64:
                raise RuntimeError(
                    f"unexpected M0 candidate-feature shape: {candidate.shape}"
                )
            if expected_shape is None:
                expected_shape = shape
            if shape != expected_shape:
                raise RuntimeError("replay sources have different candidate features")
            if not torch.isfinite(candidate.float()).all():
                raise RuntimeError("M0 candidate features contain non-finite values")
            tokens.extend(selected_tokens)
            parts.append(candidate)
            source_count += len(selected_tokens)
            if max_scenes_per_source > 0 and source_count >= max_scenes_per_source:
                break
        if source_count == 0:
            raise RuntimeError(f"source {source.name} has no M0 candidate features")
    return tokens, torch.cat(parts)


@dataclass(frozen=True)
class SharedFutureTargetTable:
    tokens: List[str]
    actor_future: torch.Tensor
    actor_masks: torch.Tensor
    supervision_valid: torch.Tensor
    lineage: Dict[str, object]


def load_shared_future_target_table(root: Path) -> SharedFutureTargetTable:
    """Load training-only shared logged-future actor supervision."""

    state_path = root / "shared_actor_future.npy"
    mask_path = root / "shared_actor_mask.npy"
    completed_path = root / "completed.npy"
    metadata_path = root / "scene_metadata.parquet"
    manifest_path = root / "manifest.json"
    for path in (
        state_path,
        mask_path,
        completed_path,
        metadata_path,
        manifest_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    manifest = json.loads(manifest_path.read_text())
    if not bool(manifest.get("depends_on_logged_future")):
        raise RuntimeError("shared-future target manifest lacks future provenance")
    if not bool(manifest.get("training_only_target")):
        raise RuntimeError("shared-future targets are not marked training-only")
    if bool(manifest.get("available_as_model_input_at_inference")):
        raise RuntimeError("shared-future targets must be training-only")
    if manifest.get("coordinate_frame") != "current_ego":
        raise RuntimeError("shared-future targets must use the current ego frame")
    declared_hashes = manifest.get("array_sha256", {})
    for path in (state_path, mask_path, completed_path):
        expected_hash = declared_hashes.get(path.name)
        if not expected_hash or _sha256(path) != expected_hash:
            raise RuntimeError(f"shared-future target hash mismatch: {path}")
    actor_future = np.load(state_path, mmap_mode="r")
    actor_masks = np.load(mask_path, mmap_mode="r")
    completed = np.load(completed_path, mmap_mode="r")
    metadata = pd.read_parquet(metadata_path).sort_values("scene_index")
    row_indices = metadata["scene_index"].to_numpy(dtype=np.int64)
    if not np.array_equal(row_indices, np.arange(len(metadata), dtype=np.int64)):
        raise RuntimeError("shared-future metadata scene indices are not contiguous")
    expected_state = (len(metadata), 8, 16, 8)
    expected_mask = (len(metadata), 8, 16)
    if actor_future.shape != expected_state or actor_masks.shape != expected_mask:
        raise RuntimeError(
            "shared-future target shapes are invalid: "
            f"{actor_future.shape}, {actor_masks.shape}"
        )
    if completed.shape != (len(metadata),):
        raise RuntimeError("shared-future completion mask shape is invalid")
    tokens = metadata["scene_token"].astype(str).tolist()
    if len(set(tokens)) != len(tokens):
        raise RuntimeError("shared-future target tokens are not unique")
    preflight = metadata["target_preflight_available"].to_numpy(dtype=bool)
    supervision_valid = np.asarray(completed, dtype=bool) & preflight
    if int(supervision_valid.sum()) != int(manifest.get("valid_scene_count", -1)):
        raise RuntimeError("shared-future valid-scene count disagrees with manifest")
    if not np.isfinite(np.asarray(actor_future[supervision_valid])).all():
        raise RuntimeError("shared-future actor targets contain non-finite values")
    lineage = {
        "name": "training_only_shared_logged_future_actor_supervision",
        "root": str(root.resolve()),
        "scene_count": len(tokens),
        "valid_scene_count": int(supervision_valid.sum()),
        "actor_slots": 16,
        "horizons": 8,
        "coordinate_frame": "current_ego",
        "depends_on_logged_future": True,
        "available_as_model_input_at_inference": False,
        "manifest_sha256": _sha256(manifest_path),
        "metadata_sha256": _sha256(metadata_path),
        "state_sha256": str(manifest["array_sha256"]["shared_actor_future.npy"]),
        "mask_sha256": str(manifest["array_sha256"]["shared_actor_mask.npy"]),
    }
    return SharedFutureTargetTable(
        tokens=tokens,
        actor_future=torch.from_numpy(np.asarray(actor_future).copy()),
        actor_masks=torch.from_numpy(np.asarray(actor_masks).copy()),
        supervision_valid=torch.from_numpy(supervision_valid.copy()),
        lineage=lineage,
    )


class ResidualReplayDataset(Dataset):
    def __init__(
        self,
        data: ReplayTensorSet,
        base_factor_logits: torch.Tensor,
        indices: Sequence[int],
        m0_candidate_features: Optional[torch.Tensor] = None,
        include_m0_context: bool = False,
        include_current_actor_targets: bool = False,
        shared_future_table: Optional[SharedFutureTargetTable] = None,
        shared_future_row_indices: Optional[torch.Tensor] = None,
    ) -> None:
        if base_factor_logits.shape != (len(data), 64, len(FACTOR_KEYS)):
            raise ValueError("Base factor logits do not align with replay rows")
        self.data = data
        self.base_factor_logits = base_factor_logits
        self.indices = torch.as_tensor(indices, dtype=torch.long)
        self.m0_candidate_features = m0_candidate_features
        self.include_m0_context = include_m0_context
        self.include_current_actor_targets = include_current_actor_targets
        self.shared_future_table = shared_future_table
        self.shared_future_row_indices = shared_future_row_indices
        if (shared_future_table is None) != (shared_future_row_indices is None):
            raise ValueError("shared-future table and row indices must be paired")
        if shared_future_row_indices is not None and shared_future_row_indices.shape != (
            len(data),
        ):
            raise ValueError("shared-future row indices must align with replay rows")
        self._empty_future = torch.zeros(8, 16, 8)
        self._empty_future_mask = torch.zeros(8, 16, dtype=torch.bool)
        if include_current_actor_targets:
            if data.current_actor_states.shape != (len(data), 16, 8):
                raise ValueError("current-actor states do not align with replay rows")
            if data.current_actor_masks.shape != (len(data), 16):
                raise ValueError("current-actor masks do not align with replay rows")
        if include_m0_context:
            if data.m0_scene_features is None or data.m0_ego_features is None:
                raise ValueError("released M0 context is absent from replay data")
            if data.m0_scene_features.shape != (len(data), 16, 256):
                raise ValueError("released M0 scene context does not align")
            if data.m0_ego_features.shape != (len(data), 1, 256):
                raise ValueError("released M0 ego context does not align")
        if m0_candidate_features is not None:
            if m0_candidate_features.ndim != 3:
                raise ValueError("M0 candidate features must have shape [N,K,D]")
            if m0_candidate_features.shape[:2] != (len(data), 64):
                raise ValueError("M0 candidate features do not align with replay rows")

    def __len__(self) -> int:
        return int(self.indices.numel())

    def __getitem__(self, index: int):
        source_index = int(self.indices[index])
        observation_index = int(self.data.observation_row_indices[source_index])
        base = (
            self.data.proposals[source_index],
            self.data.observation_tokens[observation_index],
            self.data.observation_valid_masks[observation_index],
            self.data.ego_features[observation_index],
            self.data.base_scores_for_evaluation[source_index],
            self.base_factor_logits[source_index],
            self.data.target_factors[source_index],
            torch.tensor(source_index, dtype=torch.long),
        )
        auxiliary = ()
        if self.m0_candidate_features is not None:
            auxiliary += (self.m0_candidate_features[source_index],)
        if self.include_m0_context:
            auxiliary += (
                self.data.m0_scene_features[source_index],
                self.data.m0_ego_features[source_index],
            )
        if self.include_current_actor_targets:
            auxiliary += (
                self.data.current_actor_states[source_index],
                self.data.current_actor_masks[source_index],
                self.data.current_actor_supervision_valid[source_index],
            )
        if self.shared_future_table is None:
            return base + auxiliary
        target_index = int(self.shared_future_row_indices[source_index])
        if target_index < 0:
            return base + auxiliary + (
                self._empty_future,
                self._empty_future_mask,
                torch.tensor(False),
            )
        return base + auxiliary + (
            self.shared_future_table.actor_future[target_index],
            self.shared_future_table.actor_masks[target_index],
            self.shared_future_table.supervision_valid[target_index],
        )


def _m0_reference_targets(
    target_factors: torch.Tensor,
    reference_indices: torch.Tensor,
    *,
    minimum_improvement: float,
    factor_epsilon: float,
) -> Dict[str, torch.Tensor]:
    """Build Base-relative gain and NC/DAC/TTC regression targets."""

    if target_factors.ndim != 3 or target_factors.shape[-1] != 7:
        raise ValueError("target factors must have shape [B,K,7]")
    if reference_indices.shape != (target_factors.shape[0],):
        raise ValueError("reference indices must have shape [B]")
    reference = target_factors.gather(
        1,
        reference_indices[:, None, None].expand(-1, 1, 7),
    ).expand_as(target_factors)
    gain = target_factors[..., -1] - reference[..., -1]
    safety_indices = (0, 1, 3)
    safety_worse = target_factors[..., list(safety_indices)] < (
        reference[..., list(safety_indices)] - factor_epsilon
    )
    safe_improvement = (
        gain >= minimum_improvement
    ) & ~safety_worse.any(dim=-1)
    reference_mask = torch.zeros_like(gain, dtype=torch.bool)
    reference_mask.scatter_(1, reference_indices[:, None], True)
    return {
        "gain": gain,
        "safety_worse": safety_worse,
        "safe_improvement": safe_improvement,
        "reference_mask": reference_mask,
    }


def _m0_weighted_binary_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor,
    positive_weight: float,
) -> torch.Tensor:
    if positive_weight <= 0:
        raise ValueError("positive weight must be positive")
    element = F.binary_cross_entropy_with_logits(
        logits,
        targets.to(logits.dtype),
        reduction="none",
    )
    weights = torch.where(targets.bool(), positive_weight, 1.0).to(logits.dtype)
    mask = valid_mask
    while mask.ndim < element.ndim:
        mask = mask.unsqueeze(-1)
    mask = mask.expand_as(element).to(logits.dtype)
    return (element * weights * mask).sum() / (
        weights * mask
    ).sum().clamp_min(1.0)


def compute_residual_training_loss(
    model: M0PrivateResidualRanker,
    batch: Sequence[torch.Tensor],
    args: argparse.Namespace,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    base_batch = batch[:8]
    (
        proposals,
        observation,
        observation_valid_mask,
        status,
        base_scores,
        base_factor_logits,
        target_factors,
        _indices,
    ) = base_batch
    cursor = 8
    m0_candidate_features = None
    if model.residual_config.m0_candidate_fusion:
        if len(batch) < cursor + 1:
            raise RuntimeError("M0 candidate fusion lacks frozen candidate features")
        m0_candidate_features = batch[cursor]
        cursor += 1
    m0_scene_features = None
    m0_ego_features = None
    if model.residual_config.m0_context_fusion:
        if len(batch) < cursor + 2:
            raise RuntimeError("M0 context fusion lacks frozen M0 features")
        m0_scene_features = batch[cursor]
        m0_ego_features = batch[cursor + 1]
        cursor += 2
    output = model(
        observation.float(),
        status.float(),
        proposals,
        base_factor_logits,
        base_scores,
        observation_valid_mask=observation_valid_mask,
        m0_scene_features=m0_scene_features,
        m0_ego_features=m0_ego_features,
        m0_candidate_features=m0_candidate_features,
    )
    candidate_indices = base_anchored_topk_indices(
        base_scores,
        model.residual_config.top_k,
    )
    prediction = _gather_candidates(
        output["refined_scores"].unsqueeze(-1), candidate_indices
    ).squeeze(-1)
    target_scores = _gather_candidates(
        target_factors[..., -1].unsqueeze(-1), candidate_indices
    ).squeeze(-1)
    pairwise = weighted_pairwise_loss(
        prediction,
        target_scores,
        args.minimum_pair_delta,
    )
    base_pairwise = base_pairwise_loss(
        prediction,
        target_scores,
        args.minimum_pair_delta,
    )
    listwise = listwise_loss(
        prediction,
        target_scores,
        args.target_temperature,
    )
    top_set = top_set_cross_entropy(
        prediction,
        target_scores,
        tolerance=args.top_set_tolerance,
        prediction_temperature=args.prediction_temperature,
    )
    expected_regret = expected_regret_loss(
        prediction,
        target_scores,
        prediction_temperature=args.prediction_temperature,
    )
    top_regret = top_regret_rank_loss(
        prediction,
        target_scores,
        minimum_target_delta=getattr(
            args,
            "top_regret_minimum_delta",
            0.01,
        ),
    )

    reorder = torch.tensor(
        TARGET_TO_MODEL_FACTOR_ORDER,
        device=target_factors.device,
    )
    target_six = target_factors.index_select(-1, reorder)
    relative_target = relative_safety_targets(target_six, base_scores)
    factor_loss_scope = getattr(args, "factor_loss_scope", "all")
    if factor_loss_scope == "topk":
        refined_factor_logits = _gather_candidates(
            output["refined_factor_logits"],
            candidate_indices,
        )
        private_factor_logits = _gather_candidates(
            output["private_factor_logits"],
            candidate_indices,
        )
        factor_targets = _gather_candidates(target_six, candidate_indices)
        relative_safety_logits = _gather_candidates(
            output["relative_safety_logits"],
            candidate_indices,
        )
        relative_target = _gather_candidates(
            relative_target,
            candidate_indices,
        )
        factor_rank_targets = target_scores
    elif factor_loss_scope == "all":
        refined_factor_logits = output["refined_factor_logits"]
        private_factor_logits = output["private_factor_logits"]
        factor_targets = target_six
        relative_safety_logits = output["relative_safety_logits"]
        factor_rank_targets = target_factors[..., -1]
    else:
        raise ValueError("factor_loss_scope must be all or topk")
    factor = episode_drive_factor_loss(
        refined_factor_logits,
        factor_targets,
        args.safety_negative_weight,
    )
    private_factor = episode_drive_factor_loss(
        private_factor_logits,
        factor_targets,
        args.safety_negative_weight,
    )
    factor_rank = weighted_pairwise_loss(
        pdms_factor_log_utility(refined_factor_logits),
        factor_rank_targets,
        args.factor_rank_minimum_delta,
    )

    relative_element = F.binary_cross_entropy_with_logits(
        relative_safety_logits,
        relative_target,
        reduction="none",
    )
    relative_weight = torch.where(
        relative_target < 0.5,
        args.safety_negative_weight,
        1.0,
    )
    relative_safety = (
        relative_element * relative_weight
    ).sum() / relative_weight.sum().clamp_min(1.0)
    residual_l2 = output["residual"].square().mean()
    zero = output["residual"].sum() * 0.0
    reference = {
        "total": zero,
        "quantile": zero,
        "median_rank": zero,
        "safety": zero,
        "improvement": zero,
        "false_switch": zero,
        "missed_improvement": zero,
    }
    if model.residual_config.conservative_reference:
        reference_indices = base_scores.argmax(dim=1)
        reference_targets = _m0_reference_targets(
            target_factors,
            reference_indices,
            minimum_improvement=args.reference_minimum_improvement_target,
            factor_epsilon=args.reference_factor_epsilon,
        )
        valid = output["shortlist_mask"] & ~reference_targets["reference_mask"]
        quantile = masked_pinball_quantile_loss(
            output["gain_quantiles"],
            reference_targets["gain"],
            valid_mask=valid,
        )
        median_rank = weighted_pairwise_rank_loss(
            output["gain_quantiles"][..., 1],
            reference_targets["gain"],
            valid_mask=valid,
            minimum_target_delta=args.minimum_pair_delta,
        )
        safety = _m0_weighted_binary_loss(
            output["safety_worse_logits"],
            reference_targets["safety_worse"],
            valid,
            args.reference_safety_worse_positive_weight,
        )
        improvement = _m0_weighted_binary_loss(
            output["safe_improvement_logit"],
            reference_targets["safe_improvement"],
            valid,
            args.reference_safe_improvement_positive_weight,
        )
        lower_gain = output["gain_quantiles"][..., 0]
        harmful = (
            reference_targets["gain"] <= 0.0
        ) | reference_targets["safety_worse"].any(dim=-1)
        harmful_mask = valid & harmful
        false_switch = (
            F.softplus(
                lower_gain[harmful_mask]
                / args.reference_switch_margin_temperature
            ).mean()
            if bool(harmful_mask.any())
            else zero
        )
        positive = valid & reference_targets["safe_improvement"]
        missed_improvement = (
            F.softplus(
                -lower_gain[positive]
                / args.reference_switch_margin_temperature
            ).mean()
            if bool(positive.any())
            else zero
        )
        reference = {
            "quantile": quantile,
            "median_rank": median_rank,
            "safety": safety,
            "improvement": improvement,
            "false_switch": false_switch,
            "missed_improvement": missed_improvement,
            "total": (
                args.reference_quantile_weight * quantile
                + args.reference_median_rank_weight * median_rank
                + args.reference_safety_weight * safety
                + args.reference_improvement_weight * improvement
                + args.reference_false_switch_weight * false_switch
                + args.reference_missed_improvement_weight
                * missed_improvement
            ),
        }
    current_actor = {
        "total": zero,
        "presence": zero,
        "type": zero,
        "state": zero,
    }
    if model.private_config.current_actor_auxiliary:
        if len(batch) < cursor + 3:
            raise RuntimeError("current-actor prediction head lacks training targets")
        current_actor = current_actor_auxiliary_loss(
            output,
            batch[cursor],
            batch[cursor + 1],
            batch[cursor + 2],
        )
        cursor += 3

    future_targets = None
    if model.private_config.shared_future_auxiliary:
        if len(batch) != cursor + 3:
            raise RuntimeError("shared-future prediction head lacks training targets")
        future_targets = batch[cursor : cursor + 3]
        future = shared_future_auxiliary_loss(
            output,
            future_targets[0],
            future_targets[1],
            future_targets[2],
        )
    elif len(batch) == cursor:
        future = {
            "total": zero,
            "presence": zero,
            "type": zero,
            "state": zero,
        }
    else:
        raise ValueError(f"unexpected residual training batch width: {len(batch)}")
    consequence = {
        "total": zero,
        "clearance": zero,
        "collision": zero,
        "ttc": zero,
        "occupancy": zero,
        "relative_state": zero,
    }
    candidate_relative_weight = getattr(args, "candidate_relative_weight", 0.0)
    if candidate_relative_weight > 0.0:
        if future_targets is None:
            raise RuntimeError(
                "candidate-relative supervision requires shared-future targets"
            )
        relabeler = model.private_ranker.shared_future_relabeler
        if relabeler is None:
            raise RuntimeError(
                "candidate-relative supervision requires factorized relabeling"
            )
        consequence = candidate_relative_consequence_loss(
            output,
            relabeler,
            proposals,
            future_targets[0],
            future_targets[1],
            future_targets[2],
        )
    total = (
        args.pairwise_weight * pairwise
        + args.base_pairwise_weight * base_pairwise
        + args.listwise_weight * listwise
        + args.top_set_weight * top_set
        + args.expected_regret_weight * expected_regret
        + getattr(args, "top_regret_weight", 0.0) * top_regret
        + args.factor_weight * factor
        + args.private_factor_weight * private_factor
        + args.factor_rank_weight * factor_rank
        + args.relative_safety_weight * relative_safety
        + args.residual_l2_weight * residual_l2
        + getattr(args, "reference_weight", 0.0) * reference["total"]
        + getattr(args, "shared_future_weight", 0.0) * future["total"]
        + getattr(args, "current_actor_weight", 0.0) * current_actor["total"]
        + candidate_relative_weight * consequence["total"]
    )
    details = {
        "loss": total,
        "pairwise": pairwise,
        "base_pairwise": base_pairwise,
        "listwise": listwise,
        "top_set": top_set,
        "expected_regret": expected_regret,
        "top_regret": top_regret,
        "factor": factor,
        "private_factor": private_factor,
        "factor_rank": factor_rank,
        "relative_safety": relative_safety,
        "residual_l2": residual_l2,
        "reference": reference["total"],
        "reference_quantile": reference["quantile"],
        "reference_median_rank": reference["median_rank"],
        "reference_safety": reference["safety"],
        "reference_improvement": reference["improvement"],
        "reference_false_switch": reference["false_switch"],
        "reference_missed_improvement": reference["missed_improvement"],
        "shared_future": future["total"],
        "shared_future_presence": future["presence"],
        "shared_future_type": future["type"],
        "shared_future_state": future["state"],
        "current_actor": current_actor["total"],
        "current_actor_presence": current_actor["presence"],
        "current_actor_type": current_actor["type"],
        "current_actor_state": current_actor["state"],
        "candidate_relative": consequence["total"],
        "candidate_relative_clearance": consequence["clearance"],
        "candidate_relative_collision": consequence["collision"],
        "candidate_relative_ttc": consequence["ttc"],
        "candidate_relative_occupancy": consequence["occupancy"],
        "candidate_relative_state": consequence["relative_state"],
    }
    return total, {
        key: float(value.detach()) for key, value in details.items()
    }


@torch.inference_mode()
def collect_residual_predictions(
    model: M0PrivateResidualRanker,
    data: ReplayTensorSet,
    base_factor_logits: torch.Tensor,
    m0_candidate_features: Optional[torch.Tensor],
    indices: Sequence[int],
    device: torch.device,
    batch_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    model.eval()
    selection_parts: List[torch.Tensor] = []
    factor_parts: List[torch.Tensor] = []
    base_parts: List[torch.Tensor] = []
    target_parts: List[torch.Tensor] = []
    loader = DataLoader(
        ResidualReplayDataset(
            data,
            base_factor_logits,
            indices,
            m0_candidate_features=m0_candidate_features,
            include_m0_context=model.residual_config.m0_context_fusion,
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    for batch in loader:
        base_batch = batch[:8]
        (
            proposals,
            observation,
            observation_valid_mask,
            status,
            base_scores,
            factor_logits,
            target_factors,
            _source_indices,
        ) = base_batch
        cursor = 8
        candidate_features = None
        if model.residual_config.m0_candidate_fusion:
            candidate_features = batch[cursor].to(
                device, non_blocking=True
            ).float()
            cursor += 1
        m0_scene_features = None
        m0_ego_features = None
        if model.residual_config.m0_context_fusion:
            m0_scene_features = batch[cursor].to(device, non_blocking=True).float()
            m0_ego_features = batch[cursor + 1].to(
                device, non_blocking=True
            ).float()
        output = model(
            observation.to(device, non_blocking=True).float(),
            status.to(device, non_blocking=True).float(),
            proposals.to(device, non_blocking=True),
            factor_logits.to(device, non_blocking=True),
            base_scores.to(device, non_blocking=True),
            observation_valid_mask=observation_valid_mask.to(
                device, non_blocking=True
            ),
            m0_scene_features=m0_scene_features,
            m0_ego_features=m0_ego_features,
            m0_candidate_features=candidate_features,
        )
        selection_parts.append(output["selection_scores"].float().cpu())
        factor_parts.append(output["refined_factor_logits"].float().cpu())
        base_parts.append(base_scores.float())
        target_parts.append(target_factors.float())
    return (
        torch.cat(selection_parts),
        torch.cat(factor_parts),
        torch.cat(base_parts),
        torch.cat(target_parts),
    )


def evaluate_residual_predictions(
    selection_scores: torch.Tensor,
    refined_factor_logits: torch.Tensor,
    base_scores: torch.Tensor,
    target_factors: torch.Tensor,
    physical_logs: Sequence[str],
    seed: int,
    bootstrap_replicates: int,
) -> Dict[str, object]:
    target_scores = target_factors[..., -1]
    rows = torch.arange(len(target_scores))
    selected = selection_scores.argmax(dim=1)
    base = base_scores.argmax(dim=1)
    oracle = target_scores.argmax(dim=1)
    selected_values = target_scores[rows, selected]
    base_values = target_scores[rows, base]
    oracle_values = target_scores[rows, oracle]
    delta = (selected_values - base_values).numpy()
    interval = _log_bootstrap_ci(
        delta,
        physical_logs,
        seed,
        replicates=bootstrap_replicates,
    )
    target_six = target_factors[..., list(TARGET_TO_MODEL_FACTOR_ORDER)]
    selected_factors = target_six[rows, selected]
    base_factors = target_six[rows, base]
    factor_prediction = pdms_factor_log_utility(refined_factor_logits)
    wins = int((selected_values > base_values + 1.0e-9).sum())
    losses = int((selected_values < base_values - 1.0e-9).sum())
    return {
        "scene_count": len(target_scores),
        "physical_log_count": len(set(physical_logs)),
        "selected_pdms": float(selected_values.mean()),
        "base_selected_pdms": float(base_values.mean()),
        "best_of_64_pdms": float(oracle_values.mean()),
        "selected_delta": float(delta.mean()),
        "selected_delta_log_bootstrap_95ci": [
            float(interval[0]),
            float(interval[1]),
        ],
        "selected_regret": float((oracle_values - selected_values).mean()),
        "base_regret": float((oracle_values - base_values).mean()),
        "switch_rate": float((selected != base).float().mean()),
        "wins": wins,
        "losses": losses,
        "ties": int(len(target_scores) - wins - losses),
        "pairwise_accuracy_all_non_ties": _pairwise_accuracy(
            selection_scores, target_scores, 1.0e-9
        ),
        "pairwise_accuracy_delta_005": _pairwise_accuracy(
            selection_scores, target_scores, 0.05
        ),
        "factor_pairwise_accuracy_delta_005": _pairwise_accuracy(
            factor_prediction, target_scores, 0.05
        ),
        "selected_factors": {
            key: float(selected_factors[:, index].mean())
            for index, key in enumerate(FACTOR_KEYS)
        },
        "base_selected_factors": {
            key: float(base_factors[:, index].mean())
            for index, key in enumerate(FACTOR_KEYS)
        },
    }


def evaluate_residual_predictions_by_source(
    selection_scores: torch.Tensor,
    refined_factor_logits: torch.Tensor,
    base_scores: torch.Tensor,
    target_factors: torch.Tensor,
    physical_logs: Sequence[str],
    source_names: Sequence[str],
    seed: int,
    bootstrap_replicates: int,
) -> Tuple[Dict[str, object], Dict[str, Dict[str, object]]]:
    """Evaluate a replay mixture without mixing checkpoint-selection domains."""

    scene_count = int(selection_scores.shape[0])
    tensors = (
        refined_factor_logits,
        base_scores,
        target_factors,
    )
    if any(int(value.shape[0]) != scene_count for value in tensors):
        raise ValueError("residual prediction tensors have different row counts")
    if len(physical_logs) != scene_count or len(source_names) != scene_count:
        raise ValueError("residual metadata does not align with prediction rows")

    combined = evaluate_residual_predictions(
        selection_scores,
        refined_factor_logits,
        base_scores,
        target_factors,
        physical_logs,
        seed,
        bootstrap_replicates,
    )
    by_source: Dict[str, Dict[str, object]] = {}
    for source_name in sorted(set(source_names)):
        indices = torch.tensor(
            [
                index
                for index, value in enumerate(source_names)
                if value == source_name
            ],
            dtype=torch.long,
        )
        if not int(indices.numel()):
            continue
        source_logs = [physical_logs[int(index)] for index in indices]
        by_source[source_name] = evaluate_residual_predictions(
            selection_scores.index_select(0, indices),
            refined_factor_logits.index_select(0, indices),
            base_scores.index_select(0, indices),
            target_factors.index_select(0, indices),
            source_logs,
            seed,
            bootstrap_replicates,
        )
    return combined, by_source


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        action="append",
        nargs=3,
        metavar=("NAME", "FEATURE_ROOT", "LABEL_ROOT"),
        required=True,
    )
    parser.add_argument(
        "--private-observation-root",
        type=Path,
        default=None,
        help=(
            "Optional raw/spatial current-observation token cache. When "
            "omitted, train the scorer-private semantic refiner directly on "
            "the source checkpoint's cached scene tokens."
        ),
    )
    parser.add_argument("--current-actor-target-root", type=Path, default=None)
    parser.add_argument("--shared-future-target-root", type=Path, default=None)
    parser.add_argument("--shared-future-relabeling", action="store_true")
    parser.add_argument(
        "--shared-future-constant-velocity-residual", action="store_true"
    )
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--selection-source", default="")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--refit-all-logs",
        action="store_true",
        help=(
            "After held-out-log selection, retrain the identical configuration "
            "on every physical training log for the locked stop epoch."
        ),
    )
    parser.add_argument(
        "--refit-selection-artifact",
        type=Path,
        default=None,
        help=(
            "Held-out-selected M0PrivateResidualRanker artifact that locks an "
            "all-log refit. Required together with --refit-all-logs."
        ),
    )
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--model-dim", type=int, default=256)
    parser.add_argument("--dynamic-queries", type=int, default=16)
    parser.add_argument("--private-layers", type=int, default=2)
    parser.add_argument("--trajectory-layers", type=int, default=2)
    parser.add_argument("--candidate-layers", type=int, default=1)
    parser.add_argument("--fine-layers", type=int, default=2)
    parser.add_argument("--private-fine-top-k", type=int, default=16)
    parser.add_argument("--residual-layers", type=int, default=2)
    parser.add_argument("--m0-context-fusion", action="store_true")
    parser.add_argument("--m0-candidate-fusion", action="store_true")
    parser.add_argument(
        "--m0-candidate-only",
        action="store_true",
        help=(
            "Rank from frozen M0 candidate hidden states plus Base factor "
            "context; excludes the new private candidate feature from ranking."
        ),
    )
    parser.add_argument("--conservative-reference", action="store_true")
    parser.add_argument("--reference-hidden-dim", type=int, default=512)
    parser.add_argument("--reference-layers", type=int, default=2)
    parser.add_argument(
        "--reference-gain-quantile-index", type=int, default=1
    )
    parser.add_argument("--reference-minimum-lcb-gain", type=float, default=0.0)
    parser.add_argument(
        "--reference-maximum-safety-worse-probability",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--reference-minimum-safe-improvement-probability",
        type=float,
        default=0.7,
    )
    parser.add_argument("--residual-top-k", type=int, default=64)
    parser.add_argument(
        "--score-mode",
        choices=("direct", "factor", "hybrid"),
        default="hybrid",
    )
    parser.add_argument("--max-residual", type=float, default=0.5)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--minimum-pair-delta", type=float, default=0.02)
    parser.add_argument("--factor-rank-minimum-delta", type=float, default=0.05)
    parser.add_argument("--target-temperature", type=float, default=0.05)
    parser.add_argument("--prediction-temperature", type=float, default=0.05)
    parser.add_argument("--top-set-tolerance", type=float, default=0.01)
    parser.add_argument("--pairwise-weight", type=float, default=1.0)
    parser.add_argument("--base-pairwise-weight", type=float, default=1.0)
    parser.add_argument("--listwise-weight", type=float, default=0.1)
    parser.add_argument("--top-set-weight", type=float, default=0.5)
    parser.add_argument("--expected-regret-weight", type=float, default=1.0)
    parser.add_argument("--top-regret-weight", type=float, default=0.0)
    parser.add_argument("--top-regret-minimum-delta", type=float, default=0.01)
    parser.add_argument("--factor-weight", type=float, default=1.0)
    parser.add_argument("--private-factor-weight", type=float, default=0.25)
    parser.add_argument("--factor-rank-weight", type=float, default=0.5)
    parser.add_argument("--relative-safety-weight", type=float, default=0.5)
    parser.add_argument("--residual-l2-weight", type=float, default=0.01)
    parser.add_argument("--reference-weight", type=float, default=0.0)
    parser.add_argument("--reference-quantile-weight", type=float, default=1.0)
    parser.add_argument("--reference-median-rank-weight", type=float, default=0.25)
    parser.add_argument("--reference-safety-weight", type=float, default=1.0)
    parser.add_argument("--reference-improvement-weight", type=float, default=0.5)
    parser.add_argument("--reference-false-switch-weight", type=float, default=0.5)
    parser.add_argument(
        "--reference-missed-improvement-weight", type=float, default=0.0
    )
    parser.add_argument(
        "--reference-safety-worse-positive-weight", type=float, default=10.0
    )
    parser.add_argument(
        "--reference-safe-improvement-positive-weight", type=float, default=3.0
    )
    parser.add_argument(
        "--reference-switch-margin-temperature", type=float, default=0.05
    )
    parser.add_argument(
        "--reference-minimum-improvement-target", type=float, default=0.005
    )
    parser.add_argument("--reference-factor-epsilon", type=float, default=1e-6)
    parser.add_argument("--shared-future-weight", type=float, default=0.0)
    parser.add_argument("--current-actor-weight", type=float, default=0.0)
    parser.add_argument("--candidate-relative-weight", type=float, default=0.0)
    parser.add_argument("--safety-negative-weight", type=float, default=1.0)
    parser.add_argument(
        "--factor-loss-scope",
        choices=("all", "topk"),
        default="all",
        help=(
            "Apply factor, factor-rank, and relative-safety supervision to all "
            "64 proposals or only the deployed Base-anchored shortlist."
        ),
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--max-scenes-per-source", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.refit_all_logs != (args.refit_selection_artifact is not None):
        raise ValueError(
            "--refit-all-logs and --refit-selection-artifact must be supplied together"
        )
    if args.refit_selection_artifact is not None and not (
        args.refit_selection_artifact.is_file()
    ):
        raise FileNotFoundError(args.refit_selection_artifact)
    if args.refit_all_logs and args.max_scenes_per_source != 0:
        raise ValueError("M0 all-log refit forbids scene truncation")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    sources = [
        ReplaySource(name, Path(feature_root), Path(label_root))
        for name, feature_root, label_root in args.source
    ]
    if len({source.name for source in sources}) != len(sources):
        raise ValueError("source names must be unique")
    selection_source = args.selection_source or sources[0].name
    if selection_source not in {source.name for source in sources}:
        raise ValueError(f"unknown selection source: {selection_source}")
    required_paths = [args.split_manifest]
    if args.private_observation_root is not None:
        required_paths.append(args.private_observation_root)
    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(path)
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.shared_future_weight < 0:
        raise ValueError("shared_future_weight must be nonnegative")
    if args.current_actor_weight < 0:
        raise ValueError("current_actor_weight must be nonnegative")
    if args.candidate_relative_weight < 0:
        raise ValueError("candidate_relative_weight must be nonnegative")
    if args.reference_weight < 0:
        raise ValueError("reference_weight must be nonnegative")
    if args.conservative_reference != (args.reference_weight > 0.0):
        raise ValueError(
            "--conservative-reference is required exactly when reference loss is active"
        )
    if (args.shared_future_target_root is None) != (
        args.shared_future_weight == 0.0
    ):
        raise ValueError(
            "shared-future target root is required exactly when its weight is positive"
        )
    if (args.current_actor_target_root is None) != (
        args.current_actor_weight == 0.0
    ):
        raise ValueError(
            "current-actor target root is required exactly when its weight is positive"
        )
    if args.shared_future_target_root is not None and args.dynamic_queries != 16:
        raise ValueError(
            "shared-future target schema has 16 fixed current-actor slots; "
            "--dynamic-queries must be 16"
        )
    if args.shared_future_relabeling and args.shared_future_target_root is None:
        raise ValueError(
            "shared-future relabeling requires shared-future supervision"
        )
    if args.candidate_relative_weight > 0 and not args.shared_future_relabeling:
        raise ValueError(
            "candidate-relative loss requires --shared-future-relabeling"
        )
    if (
        args.shared_future_constant_velocity_residual
        and args.current_actor_target_root is None
    ):
        raise ValueError(
            "constant-velocity future residuals require current-actor supervision"
        )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    torch.use_deterministic_algorithms(True)
    device = torch.device(args.device)

    data, source_lineage = load_replay_sources(
        sources,
        max_scenes_per_source=args.max_scenes_per_source,
        private_observation_root=args.private_observation_root,
        current_actor_target_root=args.current_actor_target_root,
        retain_m0_context=args.m0_context_fusion,
    )
    factor_tokens, base_factor_logits = load_replay_base_factor_logits(
        sources,
        max_scenes_per_source=args.max_scenes_per_source,
    )
    if factor_tokens != data.tokens:
        raise RuntimeError("Base factor logits do not match replay token order")
    m0_candidate_features = None
    if args.m0_candidate_fusion:
        candidate_tokens, m0_candidate_features = (
            load_replay_base_candidate_features(
                sources,
                max_scenes_per_source=args.max_scenes_per_source,
            )
        )
        if candidate_tokens != data.tokens:
            raise RuntimeError("M0 candidate features do not match replay token order")
    shared_future_table = None
    shared_future_row_indices = None
    shared_future_lineage = None
    if args.shared_future_target_root is not None:
        shared_future_table = load_shared_future_target_table(
            args.shared_future_target_root
        )
        row_for_token = {
            token: index for index, token in enumerate(shared_future_table.tokens)
        }
        shared_future_row_indices = torch.tensor(
            [row_for_token.get(token, -1) for token in data.tokens],
            dtype=torch.long,
        )
        matched = shared_future_row_indices >= 0
        if not bool(matched.any()):
            raise RuntimeError("no replay row has shared-future supervision")
        shared_future_lineage = shared_future_table.lineage | {
            "matched_replay_rows": int(matched.sum()),
            "total_replay_rows": len(data),
            "matched_supervised_replay_rows": int(
                shared_future_table.supervision_valid.index_select(
                    0, shared_future_row_indices[matched]
                ).sum()
            ),
        }

    split_payload = json.loads(args.split_manifest.read_text())
    declared_train_logs = {
        str(value) for value in split_payload["train_physical_logs"]
    }
    declared_validation_logs = {
        str(value) for value in split_payload["validation_physical_logs"]
    }
    if declared_train_logs.intersection(declared_validation_logs):
        raise RuntimeError("split manifest has overlapping physical logs")
    available_logs = set(data.physical_logs)
    uncovered = available_logs.difference(
        declared_train_logs.union(declared_validation_logs)
    )
    if uncovered:
        raise RuntimeError(f"split omits {len(uncovered)} physical logs")
    train_indices = [
        index
        for index, log_name in enumerate(data.physical_logs)
        if log_name in declared_train_logs
    ]
    validation_indices = [
        index
        for index, log_name in enumerate(data.physical_logs)
        if log_name in declared_validation_logs
    ]
    if not train_indices or not validation_indices:
        raise RuntimeError("training or validation split is empty")
    train_logs = {data.physical_logs[index] for index in train_indices}
    validation_logs = {
        data.physical_logs[index] for index in validation_indices
    }
    if train_logs.intersection(validation_logs):
        raise RuntimeError("physical-log leakage between train and validation")
    selection_split_lineage = {
        "strategy": "external_physical_log_manifest",
        "path": str(args.split_manifest.resolve()),
        "sha256": _sha256(args.split_manifest),
    }
    selection_train_indices = list(train_indices)
    selection_validation_indices = list(validation_indices)
    selection_train_logs = set(train_logs)
    selection_validation_logs = set(validation_logs)

    private_config = IndependentRankerConfig(
        observation_dim=int(data.observation_tokens.shape[-1]),
        max_observation_tokens=int(data.observation_tokens.shape[1]),
        status_dim=int(data.ego_features.shape[-1]),
        model_dim=args.model_dim,
        dynamic_queries=args.dynamic_queries,
        num_private_layers=args.private_layers,
        num_trajectory_layers=args.trajectory_layers,
        num_candidate_layers=args.candidate_layers,
        num_fine_layers=args.fine_layers,
        fine_top_k=args.private_fine_top_k,
        dropout=args.dropout,
        current_actor_auxiliary=args.current_actor_target_root is not None,
        shared_future_auxiliary=shared_future_table is not None,
        shared_future_horizons=8,
        shared_future_relabeling=args.shared_future_relabeling,
        shared_future_constant_velocity_residual=(
            args.shared_future_constant_velocity_residual
        ),
    )
    residual_config = M0PrivateResidualConfig(
        hidden_dim=args.model_dim,
        num_layers=args.residual_layers,
        num_heads=8,
        dropout=args.dropout,
        top_k=args.residual_top_k,
        max_residual=args.max_residual,
        score_mode=args.score_mode,
        m0_context_fusion=args.m0_context_fusion,
        m0_candidate_fusion=args.m0_candidate_fusion,
        m0_candidate_only=args.m0_candidate_only,
        m0_candidate_dim=(
            int(m0_candidate_features.shape[-1])
            if m0_candidate_features is not None
            else 256
        ),
        conservative_reference=args.conservative_reference,
        reference_hidden_dim=args.reference_hidden_dim,
        reference_layers=args.reference_layers,
        gain_quantile_index=args.reference_gain_quantile_index,
        minimum_lcb_gain=args.reference_minimum_lcb_gain,
        maximum_safety_worse_probability=(
            args.reference_maximum_safety_worse_probability
        ),
        minimum_safe_improvement_probability=(
            args.reference_minimum_safe_improvement_probability
        ),
    )
    refit_provenance: Optional[Dict[str, object]] = None
    selected_refit_artifact: Optional[Mapping[str, object]] = None
    split_lineage: Dict[str, object] = dict(selection_split_lineage)
    if args.refit_all_logs:
        assert args.refit_selection_artifact is not None
        loaded = torch.load(
            args.refit_selection_artifact,
            map_location="cpu",
            weights_only=False,
        )
        if not isinstance(loaded, Mapping):
            raise RuntimeError("M0 refit selection artifact has the wrong schema")
        selected_refit_artifact = loaded
        refit_provenance = validate_m0_all_log_refit_provenance(
            selected_refit_artifact,
            args,
            private_config,
            residual_config,
        )
        if refit_provenance["selection_source"] != selection_source:
            raise RuntimeError("M0 refit selection source differs from replay source")
        selected_fold_manifest = selected_refit_artifact["fold_manifest"]
        if selected_fold_manifest.get("split_lineage") != selection_split_lineage:
            raise RuntimeError("M0 refit selection split lineage differs from this run")
        if selected_fold_manifest.get("source_lineage") != source_lineage:
            raise RuntimeError("M0 refit replay/cache lineage differs from selection")
        if (
            selected_fold_manifest.get("shared_future_target_lineage")
            != shared_future_lineage
        ):
            raise RuntimeError("M0 refit shared-future lineage differs from selection")
        refit_provenance.update(
            {
                "selection_artifact": str(args.refit_selection_artifact.resolve()),
                "selection_artifact_sha256": _sha256(
                    args.refit_selection_artifact
                ),
                "selection_split_lineage": selection_split_lineage,
                "selection_train_scene_count": len(selection_train_indices),
                "selection_validation_scene_count": len(
                    selection_validation_indices
                ),
                "selection_train_physical_logs": sorted(selection_train_logs),
                "selection_validation_physical_logs": sorted(
                    selection_validation_logs
                ),
            }
        )
        train_indices = list(range(len(data)))
        validation_indices = []
        train_logs = set(data.physical_logs)
        validation_logs = set()
        split_lineage = {
            "strategy": "all_physical_logs_refit_after_heldout_selection",
            "selection_artifact_sha256": refit_provenance[
                "selection_artifact_sha256"
            ],
            "selection_split_lineage": selection_split_lineage,
        }
    model = M0PrivateResidualRanker(private_config, residual_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(
            int(refit_provenance["scheduler_horizon_epochs"])
            if refit_provenance is not None
            else args.epochs,
            1,
        ),
        eta_min=args.learning_rate * 0.05,
    )
    train_loader = DataLoader(
        ResidualReplayDataset(
            data,
            base_factor_logits,
            train_indices,
            m0_candidate_features=m0_candidate_features,
            include_m0_context=args.m0_context_fusion,
            include_current_actor_targets=(
                args.current_actor_target_root is not None
            ),
            shared_future_table=shared_future_table,
            shared_future_row_indices=shared_future_row_indices,
        ),
        batch_size=args.batch_size,
        sampler=_build_sampler(data, train_indices, args.seed),
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=bool(args.num_workers),
        drop_last=True,
    )

    args.output_dir.mkdir(parents=True, exist_ok=False)
    serialized_args = vars(args) | {
        "output_dir": str(args.output_dir),
        "private_observation_root": (
            str(args.private_observation_root)
            if args.private_observation_root is not None
            else None
        ),
        "current_actor_target_root": (
            str(args.current_actor_target_root)
            if args.current_actor_target_root is not None
            else None
        ),
        "shared_future_target_root": (
            str(args.shared_future_target_root)
            if args.shared_future_target_root is not None
            else None
        ),
        "split_manifest": str(args.split_manifest),
        "refit_selection_artifact": (
            str(args.refit_selection_artifact)
            if args.refit_selection_artifact is not None
            else None
        ),
    }
    fold_manifest: Dict[str, object] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "split_lineage": split_lineage,
        "train_scene_count": len(train_indices),
        "validation_scene_count": len(validation_indices),
        "train_physical_logs": sorted(train_logs),
        "validation_physical_logs": sorted(validation_logs),
        "source_counts": dict(Counter(data.source_names)),
        "checkpoint_selection_source": selection_source,
        "source_lineage": source_lineage,
        "scorer_private_observation_source": (
            "private_current_visual_token_cache"
            if args.private_observation_root is not None
            else "source_checkpoint_current_scene_tokens"
        ),
        "shared_future_target_lineage": shared_future_lineage,
        "private_config": asdict(private_config),
        "residual_config": asdict(residual_config),
        "model_parameter_count": sum(
            parameter.numel() for parameter in model.parameters()
        ),
        "m0_base_factor_logits_used_as_model_input": True,
        "m0_base_numeric_score_used_as_model_input": True,
        "m0_candidate_features_used_as_model_input": args.m0_candidate_fusion,
        "private_candidate_representation_used_for_ranking": not (
            args.m0_candidate_only
        ),
        "released_m0_scene_and_ego_context_used_as_model_input": (
            args.m0_context_fusion
        ),
        "external_model_representation_or_weight_used": False,
        "future_or_evaluator_input": False,
        "logged_future_used_as_training_only_auxiliary_target": (
            shared_future_table is not None
        ),
        "current_actor_annotation_used_as_training_only_target": (
            args.current_actor_target_root is not None
        ),
        "predicted_shared_future_relabeling_used_at_inference": (
            args.shared_future_relabeling
        ),
        "candidate_relative_logged_future_used_as_training_only_target": (
            args.candidate_relative_weight > 0
        ),
        "shared_future_parameterized_as_constant_velocity_residual": (
            args.shared_future_constant_velocity_residual
        ),
        "official_score_input": False,
        "refit_all_logs": args.refit_all_logs,
        "refit_provenance": refit_provenance,
        "args": serialized_args,
    }
    _atomic_json_dump(fold_manifest, args.output_dir / "fold_manifest.json")

    history: List[Dict[str, object]] = []
    best_value = -float("inf")
    best_epoch = -1
    artifact_path = args.output_dir / (
        "refit_m0_private_residual_scorer.pt"
        if refit_provenance is not None
        else "best_m0_private_residual_scorer.pt"
    )
    for epoch in range(args.epochs):
        model.train()
        epoch_details: List[Dict[str, float]] = []
        for batch in train_loader:
            moved = [value.to(device, non_blocking=True) for value in batch]
            optimizer.zero_grad(set_to_none=True)
            loss, details = compute_residual_training_loss(model, moved, args)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            epoch_details.append(details)
        scheduler.step()

        if refit_provenance is not None:
            assert selected_refit_artifact is not None
            record = {
                "epoch": epoch,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "training": _mean_details(epoch_details),
                "validation": None,
                "validation_reason": (
                    "all physical logs are training data; configuration, "
                    "deployment policy and stop epoch were locked by the "
                    "held-out-log selection artifact"
                ),
            }
            history.append(record)
            print(json.dumps(record, sort_keys=True), flush=True)
            summary: Dict[str, object] = {
                "refit_all_logs": True,
                "completed_epochs": epoch + 1,
                "requested_epochs": args.epochs,
                "checkpoint_selection_source": selection_source,
                "refit_provenance": refit_provenance,
                "history": history,
            }
            if epoch == args.epochs - 1:
                deployment_residual_config = dict(
                    refit_provenance["deployment_residual_config"]
                )
                payload: Dict[str, object] = {
                    "schema_version": 2 if args.m0_context_fusion else 1,
                    "architecture": "M0PrivateResidualRanker",
                    "state_dict": {
                        key: value.detach().cpu()
                        for key, value in model.state_dict().items()
                    },
                    "private_config": asdict(private_config),
                    "residual_config": deployment_residual_config,
                    "epoch": epoch,
                    "refit_all_logs": True,
                    "refit_provenance": refit_provenance,
                    "validation_performed": False,
                    "checkpoint_selection_source": selection_source,
                    "fold_manifest": fold_manifest,
                    "inference_input_schema": selected_refit_artifact[
                        "inference_input_schema"
                    ],
                    "forbidden_inputs": selected_refit_artifact[
                        "forbidden_inputs"
                    ],
                }
                for name in (
                    "policy_calibration",
                    "derived_conservative_policy",
                    "policy_selection_uses_disjoint_physical_logs",
                    "policy_selection_uses_navtest",
                ):
                    if name in selected_refit_artifact:
                        payload[name] = selected_refit_artifact[name]
                _atomic_torch_save(payload, artifact_path)
                summary.update(
                    {
                        "artifact": str(artifact_path.resolve()),
                        "artifact_sha256": _sha256(artifact_path),
                        "deployment_residual_config": deployment_residual_config,
                    }
                )
            _atomic_json_dump(summary, args.output_dir / "training_summary.json")
            continue

        predictions = collect_residual_predictions(
            model,
            data,
            base_factor_logits,
            m0_candidate_features,
            validation_indices,
            device,
            args.eval_batch_size,
        )
        validation_physical_logs = [
            data.physical_logs[index] for index in validation_indices
        ]
        validation_source_names = [
            data.source_names[index] for index in validation_indices
        ]
        combined_metrics, validation_by_source = (
            evaluate_residual_predictions_by_source(
                *predictions,
                validation_physical_logs,
                validation_source_names,
                args.seed + epoch,
                args.bootstrap_replicates,
            )
        )
        if selection_source not in validation_by_source:
            raise RuntimeError(
                "checkpoint selection source has no validation predictions: "
                f"{selection_source}"
            )
        metrics = validation_by_source[selection_source]
        record = {
            "epoch": epoch,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "training": _mean_details(epoch_details),
            "validation": metrics,
            "validation_all_sources": combined_metrics,
            "validation_by_source": validation_by_source,
        }
        history.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
        if float(metrics["selected_pdms"]) > best_value:
            best_value = float(metrics["selected_pdms"])
            best_epoch = epoch
            _atomic_torch_save(
                {
                    "schema_version": 2 if args.m0_context_fusion else 1,
                    "architecture": "M0PrivateResidualRanker",
                    "state_dict": {
                        key: value.detach().cpu()
                        for key, value in model.state_dict().items()
                    },
                    "private_config": asdict(private_config),
                    "residual_config": asdict(residual_config),
                    "epoch": epoch,
                    "validation": metrics,
                    "validation_all_sources": combined_metrics,
                    "validation_by_source": validation_by_source,
                    "checkpoint_selection_source": selection_source,
                    "fold_manifest": fold_manifest,
                    "inference_input_schema": (
                        (
                            "m0_current_visual_tokens"
                            if args.private_observation_root is not None
                            else "m0_current_scene_tokens"
                        ),
                        "m0_current_context_feature",
                        *(
                            (
                                "m0_released_scene_features",
                                "m0_released_ego_features",
                            )
                            if args.m0_context_fusion
                            else ()
                        ),
                        *(
                            ("m0_released_candidate_features",)
                            if args.m0_candidate_fusion
                            else ()
                        ),
                        "m0_proposals",
                        "m0_base_factor_logits",
                        "m0_base_scores",
                    ),
                    "forbidden_inputs": (
                        "external_model_representation",
                        "future_annotations",
                        "future_images",
                        "official_pdm_score",
                        "metric_cache",
                    ),
                },
                artifact_path,
            )
        _atomic_json_dump(
            {
                "best_epoch": best_epoch,
                "best_validation_pdms": best_value,
                "artifact": str(artifact_path.resolve()),
                "checkpoint_selection_source": selection_source,
                "history": history,
            },
            args.output_dir / "training_summary.json",
        )

    final = {
        "output_dir": str(args.output_dir.resolve()),
        "artifact": str(artifact_path.resolve()),
        "artifact_sha256": _sha256(artifact_path),
    }
    if refit_provenance is not None:
        final.update(
            {
                "refit_all_logs": True,
                "selection_artifact_sha256": refit_provenance[
                    "selection_artifact_sha256"
                ],
                "selected_epoch": refit_provenance["selected_epoch"],
                "training_physical_log_count": len(train_logs),
                "validation_performed": False,
            }
        )
    else:
        final.update(
            {
                "best_epoch": best_epoch,
                "best_validation_pdms": best_value,
            }
        )
    print(json.dumps(final, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
