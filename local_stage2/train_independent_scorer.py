"""Train a Base-score-independent scorer on immutable proposal replay.

PDM factors are offline training labels only.  The model forward receives
current-observation tokens, current ego features and proposal geometry; it
never receives released scorer outputs or future/evaluator tensors.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from local_stage2.train_public_base_residual_scorer import (
    _log_bootstrap_ci,
    binary_factor_loss,
    expected_regret_loss,
    top_set_cross_entropy,
)
from navsim.agents.EpisodeDrive.score_module.independent_ranker import (
    FACTOR_KEYS,
    FORBIDDEN_INFERENCE_FIELDS,
    IndependentProposalRanker,
    IndependentRankerConfig,
    current_actor_auxiliary_loss,
    pdms_factor_log_utility,
    top_heavy_listwise_loss,
    top_regret_rank_loss,
    weighted_pairwise_rank_loss,
)


TARGET_FACTOR_KEYS = (
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "ego_progress",
    "time_to_collision_within_bound",
    "comfort",
    "driving_direction_compliance",
    "score",
)
TARGET_TO_MODEL_FACTOR_ORDER = (0, 1, 5, 3, 2, 4)
_SEGMENT_SUFFIX = re.compile(r"_\d{5}_\d{5}$")


def physical_log_name(log_name: str) -> str:
    """Map NAVSIM segment directories back to the complete physical log."""

    return _SEGMENT_SUFFIX.sub("", str(log_name))


@dataclass(frozen=True)
class ReplaySource:
    name: str
    feature_root: Path
    label_root: Path


@dataclass
class ReplayTensorSet:
    tokens: List[str]
    log_names: List[str]
    physical_logs: List[str]
    source_names: List[str]
    proposals: torch.Tensor
    observation_tokens: torch.Tensor
    observation_valid_masks: torch.Tensor
    observation_row_indices: torch.Tensor
    ego_features: torch.Tensor
    base_scores_for_evaluation: torch.Tensor
    target_factors: torch.Tensor
    current_actor_states: torch.Tensor
    current_actor_masks: torch.Tensor
    current_actor_supervision_valid: torch.Tensor

    def __len__(self) -> int:
        return len(self.tokens)


@dataclass
class PrivateObservationTable:
    tokens: List[str]
    observation_tokens: torch.Tensor
    observation_valid_masks: torch.Tensor
    status_features: torch.Tensor
    lineage: Dict[str, object]


@dataclass
class CurrentActorTargetTable:
    tokens: List[str]
    actor_states: torch.Tensor
    actor_masks: torch.Tensor
    supervision_valid: torch.Tensor
    lineage: Dict[str, object]


class _ReplayIndexDataset(Dataset):
    def __init__(self, data: ReplayTensorSet, indices: Sequence[int]):
        self.data = data
        self.indices = torch.as_tensor(indices, dtype=torch.long)

    def __len__(self) -> int:
        return int(self.indices.numel())

    def __getitem__(self, index: int):
        source_index = int(self.indices[index])
        observation_index = int(self.data.observation_row_indices[source_index])
        return (
            self.data.proposals[source_index],
            self.data.observation_tokens[observation_index],
            self.data.observation_valid_masks[observation_index],
            self.data.ego_features[observation_index],
            self.data.base_scores_for_evaluation[source_index],
            self.data.target_factors[source_index],
            torch.tensor(source_index, dtype=torch.long),
            self.data.current_actor_states[source_index],
            self.data.current_actor_masks[source_index],
            self.data.current_actor_supervision_valid[source_index],
        )


def _atomic_json_dump(payload: Mapping[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _atomic_torch_save(payload: Mapping[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _sha256(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while block := file.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def _source_lineage(source: ReplaySource) -> Dict[str, object]:
    manifests = sorted(source.feature_root.glob("*_shard_*-of-*/manifest.json"))
    if not manifests:
        raise RuntimeError(f"No feature manifests found for source {source.name}")
    payloads = [json.loads(path.read_text()) for path in manifests]
    checkpoint_hashes = sorted(
        {str(payload.get("checkpoint_sha256")) for payload in payloads}
    )
    if len(checkpoint_hashes) != 1:
        raise RuntimeError(
            f"Source {source.name} has inconsistent checkpoint hashes: {checkpoint_hashes}"
        )
    return {
        "name": source.name,
        "feature_root": str(source.feature_root.resolve()),
        "label_root": str(source.label_root.resolve()),
        "checkpoint": str(payloads[0].get("checkpoint")),
        "checkpoint_sha256": checkpoint_hashes[0],
        "feature_manifest_sha256": {
            str(path.relative_to(source.feature_root)): _sha256(path)
            for path in manifests
        },
    }


def load_private_observation_table(root: Path) -> PrivateObservationTable:
    """Load one deduplicated, current-observation-only visual token table."""

    manifests = sorted(root.glob("*_shard_*-of-*/manifest.json"))
    chunk_paths = sorted(root.glob("*_shard_*-of-*/chunk_*.pt"))
    if not manifests or not chunk_paths:
        raise RuntimeError(f"No private-observation cache found in {root}")
    manifest_payloads = [json.loads(path.read_text()) for path in manifests]
    declared_shards = {int(payload["shard_count"]) for payload in manifest_payloads}
    if len(declared_shards) != 1 or len(manifests) != next(iter(declared_shards)):
        raise RuntimeError("private-observation cache has incomplete shard manifests")
    checkpoint_hashes = {
        str(
            payload.get("checkpoint_sha256")
            or payload.get("m0_checkpoint_sha256")
        )
        for payload in manifest_payloads
    }
    if len(checkpoint_hashes) != 1:
        raise RuntimeError("private-observation shards use different checkpoints")
    for payload in manifest_payloads:
        if not bool(payload.get("current_observation_only")):
            raise RuntimeError("private-observation manifest is not current-only")
        if bool(payload.get("future_or_evaluator_input")):
            raise RuntimeError("private-observation manifest declares future/evaluator input")
        if bool(payload.get("official_score_or_factor_input")):
            raise RuntimeError("private-observation manifest declares official score input")
        if bool(payload.get("proposal_input")):
            raise RuntimeError("private-observation manifest declares proposal input")
        if bool(payload.get("drivor_checkpoint_or_representation_used")):
            raise RuntimeError("private-observation manifest declares DrivOR representation use")

    tokens: List[str] = []
    observations: List[torch.Tensor] = []
    masks: List[torch.Tensor] = []
    statuses: List[torch.Tensor] = []
    seen: set[str] = set()
    forbidden_cache_fields = set(FORBIDDEN_INFERENCE_FIELDS) | {
        "target_factors",
        "base_scores",
        "base_factor_logits",
        "candidate_features",
        "proposals",
    }
    expected_shape: Optional[Tuple[int, int]] = None
    expected_status_width: Optional[int] = None
    for path in chunk_paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        leaked = sorted(forbidden_cache_fields.intersection(payload))
        if leaked:
            raise RuntimeError(f"Forbidden fields in private observation cache: {leaked}")
        chunk_tokens = [str(value) for value in payload["tokens"]]
        duplicate = seen.intersection(chunk_tokens)
        if duplicate:
            raise RuntimeError(
                f"Duplicate private-observation tokens: {sorted(duplicate)[:3]}"
            )
        seen.update(chunk_tokens)
        observation = payload["visual_tokens"]
        valid_mask = payload["visual_valid_mask"].bool()
        status = payload["status_feature"].float()
        history = payload["history_trajectory"].float().flatten(start_dim=1)
        command = payload["high_command_one_hot"].float().flatten(start_dim=1)
        status = torch.cat((status, history, command), dim=-1)
        if observation.ndim != 3 or valid_mask.shape != observation.shape[:2]:
            raise RuntimeError(f"Invalid private-observation tensor shape in {path}")
        if len(chunk_tokens) != len(observation) or len(status) != len(observation):
            raise RuntimeError(f"Private-observation row mismatch in {path}")
        if not valid_mask.any(dim=1).all():
            raise RuntimeError(f"Private-observation row without valid tokens in {path}")
        if not torch.isfinite(observation).all() or not torch.isfinite(status).all():
            raise RuntimeError(f"Non-finite private-observation tensor in {path}")
        if expected_shape is None:
            expected_shape = tuple(observation.shape[1:])
            expected_status_width = int(status.shape[-1])
        if tuple(observation.shape[1:]) != expected_shape:
            raise RuntimeError("Private-observation chunks have different token shapes")
        if status.ndim != 2 or status.shape[-1] != expected_status_width:
            raise RuntimeError("Private-observation chunks have different status shapes")
        tokens.extend(chunk_tokens)
        observations.append(observation)
        masks.append(valid_mask)
        statuses.append(status)

    declared_scene_count = sum(int(payload["scene_count"]) for payload in manifest_payloads)
    if declared_scene_count != len(tokens):
        raise RuntimeError(
            "private-observation manifest scene count mismatch: "
            f"{declared_scene_count} != {len(tokens)}"
        )
    lineage = {
        "name": "scorer_private_current_observation",
        "root": str(root.resolve()),
        "checkpoint_sha256": next(iter(checkpoint_hashes)),
        "scene_count": len(tokens),
        "manifest_sha256": {
            str(path.relative_to(root)): _sha256(path) for path in manifests
        },
        "current_observation_only": True,
        "future_or_evaluator_input": False,
        "official_score_or_factor_input": False,
        "proposal_input": False,
        "drivor_checkpoint_or_representation_used": False,
        "current_context_fields": (
            "status_feature",
            "history_trajectory",
            "high_command_one_hot",
        ),
        "current_context_width": int(statuses[0].shape[-1]),
    }
    return PrivateObservationTable(
        tokens=tokens,
        observation_tokens=torch.cat(observations),
        observation_valid_masks=torch.cat(masks),
        status_features=torch.cat(statuses),
        lineage=lineage,
    )


def load_current_actor_target_table(root: Path) -> CurrentActorTargetTable:
    """Load training-only current actor slots from the Gate-C oracle store."""

    current_path = root / "current.npy"
    completed_path = root / "completed.npy"
    metadata_path = root / "scene_metadata.parquet"
    config_path = root / "store_config.json"
    for path in (current_path, completed_path, metadata_path, config_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    current = np.load(current_path, mmap_mode="r")
    completed = np.load(completed_path, mmap_mode="r")
    metadata = pd.read_parquet(metadata_path)
    required_columns = {"scene_token", "scene_index", "target_preflight_available"}
    missing_columns = sorted(required_columns.difference(metadata.columns))
    if missing_columns:
        raise RuntimeError(
            f"Current-actor metadata misses columns: {missing_columns}"
        )
    if current.ndim != 2 or (current.shape[1] - 6) % 9:
        raise RuntimeError(f"Unexpected current target shape: {current.shape}")
    actor_slots = (current.shape[1] - 6) // 9
    if actor_slots <= 0:
        raise RuntimeError("Current-actor target table has no actor slots")
    row_indices = metadata["scene_index"].to_numpy(dtype=np.int64)
    if len(row_indices) == 0 or row_indices.min() < 0:
        raise RuntimeError("Current-actor metadata has invalid scene indices")
    if row_indices.max() >= len(current) or row_indices.max() >= len(completed):
        raise RuntimeError("Current-actor metadata index exceeds target arrays")
    tokens = metadata["scene_token"].astype(str).tolist()
    if len(set(tokens)) != len(tokens):
        raise RuntimeError("Current-actor metadata contains duplicate scene tokens")

    selected = np.asarray(current[row_indices], dtype=np.float32)
    states = selected[:, 6 : 6 + actor_slots * 8].reshape(
        len(selected), actor_slots, 8
    )
    masks = selected[:, 6 + actor_slots * 8 :].astype(bool)
    preflight = metadata["target_preflight_available"].to_numpy(dtype=bool)
    supervision_valid = np.asarray(completed[row_indices], dtype=bool) & preflight
    if masks.shape != (len(tokens), actor_slots):
        raise RuntimeError("Current-actor mask has an unexpected shape")
    if not np.isfinite(states[supervision_valid]).all():
        raise RuntimeError("Current-actor target table contains non-finite values")
    lineage = {
        "name": "training_only_current_actor_supervision",
        "root": str(root.resolve()),
        "scene_count": len(tokens),
        "valid_scene_count": int(supervision_valid.sum()),
        "actor_slots": actor_slots,
        "coordinate_frame": "current_ego",
        "depends_on_logged_future": False,
        "available_as_model_input_at_inference": False,
        "current_array_sha256": _sha256(current_path),
        "metadata_sha256": _sha256(metadata_path),
        "store_config_sha256": _sha256(config_path),
    }
    return CurrentActorTargetTable(
        tokens=tokens,
        actor_states=torch.from_numpy(states.copy()),
        actor_masks=torch.from_numpy(masks.copy()),
        supervision_valid=torch.from_numpy(supervision_valid.copy()),
        lineage=lineage,
    )


def _iter_joined_chunks(source: ReplaySource) -> Iterable[Tuple[Path, Path]]:
    paths = sorted(source.feature_root.glob("*_shard_*-of-*/chunk_*.pt"))
    if not paths:
        raise RuntimeError(f"No replay chunks found for source {source.name}")
    for feature_path in paths:
        relative = feature_path.relative_to(source.feature_root)
        label_path = source.label_root / relative
        if not label_path.is_file():
            raise RuntimeError(f"Missing label chunk for {source.name}: {relative}")
        yield feature_path, label_path


def load_replay_sources(
    sources: Sequence[ReplaySource],
    *,
    max_scenes_per_source: int = 0,
    private_observation_root: Optional[Path] = None,
    current_actor_target_root: Optional[Path] = None,
) -> Tuple[ReplayTensorSet, List[Dict[str, object]]]:
    tensor_parts: Dict[str, List[torch.Tensor]] = {
        key: []
        for key in (
            "proposals",
            "base_scores_for_evaluation",
            "target_factors",
        )
    }
    source_observations: List[torch.Tensor] = []
    source_ego_features: List[torch.Tensor] = []
    tokens: List[str] = []
    log_names: List[str] = []
    source_names: List[str] = []
    lineage: List[Dict[str, object]] = []
    expected_observation_shape: Optional[Tuple[int, ...]] = None
    expected_ego_shape: Optional[Tuple[int, ...]] = None

    for source in sources:
        lineage.append(_source_lineage(source))
        source_count = 0
        seen_tokens: set[str] = set()
        for feature_path, label_path in _iter_joined_chunks(source):
            features = torch.load(feature_path, map_location="cpu", weights_only=False)
            labels = torch.load(label_path, map_location="cpu", weights_only=False)
            feature_tokens = [str(value) for value in features["tokens"]]
            label_tokens = [str(value) for value in labels["tokens"]]
            if feature_tokens != label_tokens:
                raise RuntimeError(f"Feature/label token mismatch: {feature_path}")
            feature_logs = [str(value) for value in features["log_names"]]
            label_logs = [str(value) for value in labels["log_names"]]
            if feature_logs != label_logs:
                raise RuntimeError(f"Feature/label log mismatch: {feature_path}")
            if tuple(labels["target_factor_keys"]) != TARGET_FACTOR_KEYS:
                raise RuntimeError(
                    f"Unexpected target factor order in {label_path}: "
                    f"{tuple(labels['target_factor_keys'])}"
                )
            valid = labels["valid_mask"].bool()
            if valid.shape != (len(feature_tokens),):
                raise RuntimeError(f"Invalid scene mask shape in {label_path}")
            remaining = (
                max_scenes_per_source - source_count
                if max_scenes_per_source > 0
                else len(feature_tokens)
            )
            if remaining <= 0:
                break
            valid_indices = torch.nonzero(valid, as_tuple=False).squeeze(-1)[:remaining]
            selected_tokens = [feature_tokens[int(index)] for index in valid_indices]
            duplicate = seen_tokens.intersection(selected_tokens)
            if duplicate:
                raise RuntimeError(
                    f"Duplicate tokens within source {source.name}: {sorted(duplicate)[:3]}"
                )
            seen_tokens.update(selected_tokens)
            selected_logs = [feature_logs[int(index)] for index in valid_indices]
            if private_observation_root is None:
                observation = features["scene_features"][valid_indices]
                ego = features["ego_features"][valid_indices].squeeze(1)
                if expected_observation_shape is None:
                    expected_observation_shape = tuple(observation.shape[1:])
                    expected_ego_shape = tuple(ego.shape[1:])
                if tuple(observation.shape[1:]) != expected_observation_shape:
                    raise RuntimeError("Replay sources have different observation-token shapes")
                if tuple(ego.shape[1:]) != expected_ego_shape:
                    raise RuntimeError("Replay sources have different ego-feature shapes")
                source_observations.append(observation)
                source_ego_features.append(ego)

            tensor_parts["proposals"].append(features["proposals"][valid_indices].float())
            tensor_parts["base_scores_for_evaluation"].append(
                features["base_scores"][valid_indices].float()
            )
            tensor_parts["target_factors"].append(
                labels["target_factors"][valid_indices].float()
            )
            tokens.extend(selected_tokens)
            log_names.extend(selected_logs)
            source_names.extend([source.name] * len(selected_tokens))
            source_count += len(selected_tokens)
            if max_scenes_per_source > 0 and source_count >= max_scenes_per_source:
                break
        if source_count == 0:
            raise RuntimeError(f"Source {source.name} has no valid scenes")

    replay_tensors = {key: torch.cat(parts) for key, parts in tensor_parts.items()}
    if private_observation_root is None:
        observation_tokens = torch.cat(source_observations)
        observation_valid_masks = torch.ones(
            observation_tokens.shape[:2], dtype=torch.bool
        )
        ego_features = torch.cat(source_ego_features)
        observation_row_indices = torch.arange(len(tokens), dtype=torch.long)
    else:
        private_table = load_private_observation_table(private_observation_root)
        row_for_token = {token: index for index, token in enumerate(private_table.tokens)}
        missing = sorted({token for token in tokens if token not in row_for_token})
        if missing:
            raise RuntimeError(
                "Private-observation cache is missing replay tokens: "
                f"{missing[:5]} ({len(missing)} total)"
            )
        observation_tokens = private_table.observation_tokens
        observation_valid_masks = private_table.observation_valid_masks
        ego_features = private_table.status_features
        observation_row_indices = torch.tensor(
            [row_for_token[token] for token in tokens], dtype=torch.long
        )
        lineage.append(private_table.lineage)

    if current_actor_target_root is None:
        current_actor_states = torch.empty(len(tokens), 0, 8)
        current_actor_masks = torch.empty(len(tokens), 0, dtype=torch.bool)
        current_actor_supervision_valid = torch.zeros(len(tokens), dtype=torch.bool)
    else:
        actor_table = load_current_actor_target_table(current_actor_target_root)
        actor_row_for_token = {
            token: index for index, token in enumerate(actor_table.tokens)
        }
        actor_slots = int(actor_table.actor_states.shape[1])
        current_actor_states = torch.zeros(len(tokens), actor_slots, 8)
        current_actor_masks = torch.zeros(
            len(tokens), actor_slots, dtype=torch.bool
        )
        current_actor_supervision_valid = torch.zeros(
            len(tokens), dtype=torch.bool
        )
        for replay_index, token in enumerate(tokens):
            actor_index = actor_row_for_token.get(token)
            if actor_index is None:
                continue
            current_actor_states[replay_index] = actor_table.actor_states[
                actor_index
            ]
            current_actor_masks[replay_index] = actor_table.actor_masks[actor_index]
            current_actor_supervision_valid[replay_index] = (
                actor_table.supervision_valid[actor_index]
            )
        if not bool(current_actor_supervision_valid.any()):
            raise RuntimeError(
                "No replay scenes have valid current-actor supervision"
            )
        lineage.append(
            actor_table.lineage
            | {
                "matched_replay_rows": int(
                    current_actor_supervision_valid.sum()
                ),
                "total_replay_rows": len(tokens),
            }
        )

    result = ReplayTensorSet(
        tokens=tokens,
        log_names=log_names,
        physical_logs=[physical_log_name(value) for value in log_names],
        source_names=source_names,
        observation_tokens=observation_tokens,
        observation_valid_masks=observation_valid_masks,
        observation_row_indices=observation_row_indices,
        ego_features=ego_features,
        current_actor_states=current_actor_states,
        current_actor_masks=current_actor_masks,
        current_actor_supervision_valid=current_actor_supervision_valid,
        **replay_tensors,
    )
    expected_shapes = {
        "proposals": (64, 8, 3),
        "base_scores_for_evaluation": (64,),
        "target_factors": (64, 7),
    }
    for key, expected in expected_shapes.items():
        tensor = getattr(result, key)
        if len(tensor) != len(result) or tuple(tensor.shape[1:]) != expected:
            raise RuntimeError(f"Unexpected replay tensor {key}: {tuple(tensor.shape)}")
    if result.observation_row_indices.shape != (len(result),):
        raise RuntimeError("Unexpected observation row-index shape")
    if int(result.observation_row_indices.max()) >= len(result.observation_tokens):
        raise RuntimeError("Observation row index exceeds current-observation table")
    if result.observation_valid_masks.shape != result.observation_tokens.shape[:2]:
        raise RuntimeError("Observation valid-mask shape mismatch")
    if len(result.ego_features) != len(result.observation_tokens):
        raise RuntimeError("Observation/status table row mismatch")
    if result.current_actor_states.shape[:2] != result.current_actor_masks.shape:
        raise RuntimeError("Current-actor state/mask shape mismatch")
    if result.current_actor_supervision_valid.shape != (len(result),):
        raise RuntimeError("Current-actor supervision-valid shape mismatch")
    return result, lineage


def assign_balanced_physical_log_folds(
    physical_logs: Sequence[str],
    num_folds: int,
    seed: int,
) -> Dict[str, int]:
    if num_folds < 2:
        raise ValueError("num_folds must be at least two")
    counts = Counter(str(value) for value in physical_logs)
    if len(counts) < num_folds:
        raise ValueError("fewer physical logs than folds")
    rng = random.Random(seed)
    tie_break = {name: rng.random() for name in sorted(counts)}
    ordered = sorted(counts, key=lambda name: (-counts[name], tie_break[name], name))
    fold_counts = [0] * num_folds
    fold_logs = [0] * num_folds
    assignment: Dict[str, int] = {}
    for name in ordered:
        fold = min(
            range(num_folds),
            key=lambda value: (fold_counts[value], fold_logs[value], value),
        )
        assignment[name] = fold
        fold_counts[fold] += counts[name]
        fold_logs[fold] += 1
    return assignment


def _gather_candidates(value: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    view = indices
    while view.ndim < value.ndim:
        view = view.unsqueeze(-1)
    return value.gather(1, view.expand(-1, -1, *value.shape[2:]))


def _candidate_dropout_indices(
    target_scores: torch.Tensor,
    keep_count: int,
    generator: torch.Generator,
) -> torch.Tensor:
    candidate_count = target_scores.shape[1]
    if keep_count >= candidate_count:
        random_keys = torch.rand(
            target_scores.shape,
            generator=generator,
            device=target_scores.device,
        )
        return torch.argsort(random_keys, dim=1)
    if keep_count <= 1:
        raise ValueError("candidate_keep_count must exceed one")
    random_keys = torch.rand(
        target_scores.shape,
        generator=generator,
        device=target_scores.device,
    )
    oracle = target_scores.argmax(dim=1, keepdim=True)
    random_keys.scatter_(1, oracle, 2.0)
    return torch.topk(random_keys, k=keep_count, dim=1).indices


def _consequence_proxy_loss(
    prediction: torch.Tensor,
    target_six: torch.Tensor,
    positive_weight: float,
) -> torch.Tensor:
    if prediction.shape[-1] != 4:
        raise ValueError("consequence proxy requires four output fields")
    compliance = target_six[..., [0, 3, 1, 2]]
    violations = (compliance < 1.0).to(prediction.dtype)
    element = F.binary_cross_entropy_with_logits(
        prediction,
        violations,
        reduction="none",
    )
    weights = torch.where(violations > 0.5, positive_weight, 1.0)
    return (element * weights).sum() / weights.sum().clamp_min(1.0)


def compute_training_loss(
    model: IndependentProposalRanker,
    batch: Sequence[torch.Tensor],
    args: argparse.Namespace,
    candidate_generator: torch.Generator,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    (
        proposals,
        observation,
        observation_valid_mask,
        ego,
        _base_scores,
        target_factors,
        _indices,
        current_actor_states,
        current_actor_masks,
        current_actor_supervision_valid,
    ) = batch
    target_scores = target_factors[..., -1]
    candidate_indices = _candidate_dropout_indices(
        target_scores,
        args.candidate_keep_count,
        candidate_generator,
    )
    proposals = _gather_candidates(proposals, candidate_indices)
    target_factors = _gather_candidates(target_factors, candidate_indices)
    target_scores = target_factors[..., -1]
    output = model(
        observation.float(),
        ego.float(),
        proposals,
        observation_valid_mask=observation_valid_mask,
    )
    coarse_prediction = output["coarse_utility"]
    fine_indices = output["fine_indices"]
    fine_prediction = output["refined_utility"]
    fine_targets = _gather_candidates(
        target_scores.unsqueeze(-1), fine_indices
    ).squeeze(-1)

    coarse_rank_002 = weighted_pairwise_rank_loss(
        coarse_prediction, target_scores, minimum_target_delta=0.02
    )
    coarse_rank_005 = weighted_pairwise_rank_loss(
        coarse_prediction, target_scores, minimum_target_delta=0.05
    )
    coarse_rank_010 = weighted_pairwise_rank_loss(
        coarse_prediction, target_scores, minimum_target_delta=0.10
    )
    fine_rank_002 = weighted_pairwise_rank_loss(
        fine_prediction, fine_targets, minimum_target_delta=0.02
    )
    fine_rank_005 = weighted_pairwise_rank_loss(
        fine_prediction, fine_targets, minimum_target_delta=0.05
    )
    fine_rank_010 = weighted_pairwise_rank_loss(
        fine_prediction, fine_targets, minimum_target_delta=0.10
    )
    coarse_listwise = top_heavy_listwise_loss(
        coarse_prediction,
        target_scores,
        temperature=args.target_temperature,
    )
    fine_listwise = top_heavy_listwise_loss(
        fine_prediction,
        fine_targets,
        temperature=args.target_temperature,
    )
    coarse_top_set = top_set_cross_entropy(
        coarse_prediction,
        target_scores,
        tolerance=args.top_set_tolerance,
        prediction_temperature=args.prediction_temperature,
    )
    fine_top_set = top_set_cross_entropy(
        fine_prediction,
        fine_targets,
        tolerance=args.top_set_tolerance,
        prediction_temperature=args.prediction_temperature,
    )
    coarse_expected_regret = expected_regret_loss(
        coarse_prediction,
        target_scores,
        prediction_temperature=args.prediction_temperature,
    )
    fine_expected_regret = expected_regret_loss(
        fine_prediction,
        fine_targets,
        prediction_temperature=args.prediction_temperature,
    )
    coarse_top_regret = top_regret_rank_loss(
        coarse_prediction, target_scores, minimum_target_delta=0.01
    )
    fine_top_regret = top_regret_rank_loss(
        fine_prediction, fine_targets, minimum_target_delta=0.01
    )
    reorder = torch.tensor(
        TARGET_TO_MODEL_FACTOR_ORDER,
        device=target_factors.device,
    )
    target_six = target_factors.index_select(-1, reorder)
    factor = binary_factor_loss(
        output["factor_logits"],
        target_six,
        args.safety_negative_weight,
    )
    progress = F.smooth_l1_loss(
        output["factor_logits"][..., 4].sigmoid(),
        target_six[..., 4],
    )
    factor = factor + 2.0 * progress
    factor_rank = weighted_pairwise_rank_loss(
        pdms_factor_log_utility(output["factor_logits"]),
        target_scores,
        minimum_target_delta=0.05,
    )
    consequence = _consequence_proxy_loss(
        output["predicted_consequence"],
        target_six,
        args.safety_negative_weight,
    )
    confidence_target = (
        target_scores >= target_scores.max(dim=1, keepdim=True).values - 0.02
    ).to(coarse_prediction.dtype)
    confidence = F.binary_cross_entropy_with_logits(
        output["confidence_logit"], confidence_target
    )
    if args.current_actor_weight > 0:
        current_actor = current_actor_auxiliary_loss(
            output,
            current_actor_states.float(),
            current_actor_masks,
            current_actor_supervision_valid,
        )
    else:
        zero = coarse_prediction.sum() * 0.0
        current_actor = {
            "total": zero,
            "presence": zero,
            "type": zero,
            "state": zero,
        }
    coarse_objective = (
        args.pairwise_weight * coarse_rank_002
        + args.hard_pairwise_weight * (0.5 * coarse_rank_005 + 0.5 * coarse_rank_010)
        + args.listwise_weight * coarse_listwise
        + args.top_set_weight * coarse_top_set
        + args.expected_regret_weight * coarse_expected_regret
        + args.top_regret_weight * coarse_top_regret
    )
    fine_objective = (
        args.pairwise_weight * fine_rank_002
        + args.hard_pairwise_weight * (0.5 * fine_rank_005 + 0.5 * fine_rank_010)
        + args.listwise_weight * fine_listwise
        + args.top_set_weight * fine_top_set
        + args.expected_regret_weight * fine_expected_regret
        + args.top_regret_weight * fine_top_regret
    )
    total = (
        args.coarse_loss_weight * coarse_objective
        + fine_objective
        + args.factor_weight * factor
        + args.factor_rank_weight * factor_rank
        + args.consequence_weight * consequence
        + args.confidence_weight * confidence
        + args.current_actor_weight * current_actor["total"]
    )
    details = {
        "loss": total,
        "coarse_pairwise_002": coarse_rank_002,
        "coarse_pairwise_005": coarse_rank_005,
        "coarse_pairwise_010": coarse_rank_010,
        "fine_pairwise_002": fine_rank_002,
        "fine_pairwise_005": fine_rank_005,
        "fine_pairwise_010": fine_rank_010,
        "coarse_listwise": coarse_listwise,
        "fine_listwise": fine_listwise,
        "coarse_top_set": coarse_top_set,
        "fine_top_set": fine_top_set,
        "coarse_expected_regret": coarse_expected_regret,
        "fine_expected_regret": fine_expected_regret,
        "coarse_top_regret": coarse_top_regret,
        "fine_top_regret": fine_top_regret,
        "factor": factor,
        "factor_rank": factor_rank,
        "consequence": consequence,
        "confidence": confidence,
        "current_actor": current_actor["total"],
        "current_actor_presence": current_actor["presence"],
        "current_actor_type": current_actor["type"],
        "current_actor_state": current_actor["state"],
    }
    return total, {key: float(value.detach()) for key, value in details.items()}


@torch.inference_mode()
def collect_predictions(
    model: IndependentProposalRanker,
    data: ReplayTensorSet,
    indices: Sequence[int],
    device: torch.device,
    batch_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    model.eval()
    utilities: List[torch.Tensor] = []
    coarse_utilities: List[torch.Tensor] = []
    factor_logits: List[torch.Tensor] = []
    base_scores: List[torch.Tensor] = []
    targets: List[torch.Tensor] = []
    loader = DataLoader(
        _ReplayIndexDataset(data, indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    for (
        proposals,
        observation,
        observation_valid_mask,
        ego,
        base,
        target,
        _source_indices,
        _current_actor_states,
        _current_actor_masks,
        _current_actor_supervision_valid,
    ) in loader:
        output = model(
            observation.to(device, non_blocking=True).float(),
            ego.to(device, non_blocking=True).float(),
            proposals.to(device, non_blocking=True),
            observation_valid_mask=observation_valid_mask.to(
                device, non_blocking=True
            ),
        )
        utilities.append(output["utility"].float().cpu())
        coarse_utilities.append(output["coarse_utility"].float().cpu())
        factor_logits.append(output["factor_logits"].float().cpu())
        base_scores.append(base.float())
        targets.append(target.float())
    return (
        torch.cat(utilities),
        torch.cat(coarse_utilities),
        torch.cat(factor_logits),
        torch.cat(base_scores),
        torch.cat(targets),
    )


def _pairwise_accuracy(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    minimum_delta: float,
    chunk_size: int = 512,
) -> float:
    correct = 0
    total = 0
    candidate_count = predictions.shape[1]
    left, right = torch.triu_indices(candidate_count, candidate_count, offset=1)
    for start in range(0, len(predictions), chunk_size):
        prediction_delta = predictions[start : start + chunk_size, left] - predictions[
            start : start + chunk_size, right
        ]
        target_delta = targets[start : start + chunk_size, left] - targets[
            start : start + chunk_size, right
        ]
        valid = target_delta.abs() >= minimum_delta
        correct += int(((prediction_delta.sign() == target_delta.sign()) & valid).sum())
        total += int(valid.sum())
    return float(correct / max(total, 1))


def evaluate_predictions(
    utilities: torch.Tensor,
    coarse_utilities: torch.Tensor,
    factor_logits: torch.Tensor,
    base_scores: torch.Tensor,
    target_factors: torch.Tensor,
    physical_logs: Sequence[str],
    seed: int,
    bootstrap_replicates: int,
) -> Dict[str, object]:
    target_scores = target_factors[..., -1]
    row = torch.arange(len(target_scores))
    selected = utilities.argmax(dim=1)
    coarse_selected = coarse_utilities.argmax(dim=1)
    base_selected = base_scores.argmax(dim=1)
    factor_utilities = pdms_factor_log_utility(factor_logits)
    factor_selected = factor_utilities.argmax(dim=1)
    oracle = target_scores.argmax(dim=1)
    selected_values = target_scores[row, selected]
    coarse_values = target_scores[row, coarse_selected]
    base_values = target_scores[row, base_selected]
    factor_values = target_scores[row, factor_selected]
    oracle_values = target_scores[row, oracle]
    delta = (selected_values - base_values).numpy()
    coarse_delta = (coarse_values - base_values).numpy()
    factor_delta = (factor_values - base_values).numpy()
    ci = _log_bootstrap_ci(
        delta,
        physical_logs,
        seed,
        replicates=bootstrap_replicates,
    )
    coarse_ci = _log_bootstrap_ci(
        coarse_delta,
        physical_logs,
        seed + 10_000,
        replicates=bootstrap_replicates,
    )
    factor_ci = _log_bootstrap_ci(
        factor_delta,
        physical_logs,
        seed + 20_000,
        replicates=bootstrap_replicates,
    )
    sorted_indices = torch.argsort(utilities, dim=1, descending=True)
    shortlist_mask = utilities > -9999.0
    shortlist_values = target_scores.masked_fill(~shortlist_mask, -1.0)
    shortlist_oracle_values = shortlist_values.max(dim=1).values
    target_six = target_factors[..., list(TARGET_TO_MODEL_FACTOR_ORDER)]
    selected_factors = target_six[row, selected]
    coarse_selected_factors = target_six[row, coarse_selected]
    factor_selected_factors = target_six[row, factor_selected]
    base_factors = target_six[row, base_selected]
    wins = int((selected_values > base_values + 1e-9).sum())
    losses = int((selected_values < base_values - 1e-9).sum())
    return {
        "scene_count": len(target_scores),
        "physical_log_count": len(set(physical_logs)),
        "selected_pdms": float(selected_values.mean()),
        "coarse_selected_pdms": float(coarse_values.mean()),
        "factor_selected_pdms": float(factor_values.mean()),
        "base_selected_pdms": float(base_values.mean()),
        "oracle_best_of_64": float(oracle_values.mean()),
        "shortlist_oracle_pdms": float(shortlist_oracle_values.mean()),
        "shortlist_oracle_recall": float(shortlist_mask[row, oracle].float().mean()),
        "refinement_delta_over_coarse": float((selected_values - coarse_values).mean()),
        "selected_regret": float((oracle_values - selected_values).mean()),
        "coarse_selected_regret": float((oracle_values - coarse_values).mean()),
        "factor_selected_regret": float((oracle_values - factor_values).mean()),
        "base_regret": float((oracle_values - base_values).mean()),
        "selected_delta": float(np.mean(delta)),
        "selected_delta_log_bootstrap_95ci": [float(ci[0]), float(ci[1])],
        "coarse_selected_delta": float(np.mean(coarse_delta)),
        "coarse_selected_delta_log_bootstrap_95ci": [
            float(coarse_ci[0]),
            float(coarse_ci[1]),
        ],
        "factor_selected_delta": float(np.mean(factor_delta)),
        "factor_selected_delta_log_bootstrap_95ci": [
            float(factor_ci[0]),
            float(factor_ci[1]),
        ],
        "wins": wins,
        "losses": losses,
        "ties": int(len(target_scores) - wins - losses),
        "pairwise_accuracy_all_non_ties": _pairwise_accuracy(coarse_utilities, target_scores, 1e-9),
        "pairwise_accuracy_delta_002": _pairwise_accuracy(coarse_utilities, target_scores, 0.02),
        "pairwise_accuracy_delta_005": _pairwise_accuracy(coarse_utilities, target_scores, 0.05),
        "pairwise_accuracy_delta_010": _pairwise_accuracy(coarse_utilities, target_scores, 0.10),
        "oracle_recall_at_1": float((sorted_indices[:, :1] == oracle[:, None]).any(dim=1).float().mean()),
        "oracle_recall_at_2": float((sorted_indices[:, :2] == oracle[:, None]).any(dim=1).float().mean()),
        "oracle_recall_at_4": float((sorted_indices[:, :4] == oracle[:, None]).any(dim=1).float().mean()),
        "predicted_factor_pdms_pairwise_delta_005": _pairwise_accuracy(
            pdms_factor_log_utility(factor_logits), target_scores, 0.05
        ),
        "selected_factors": {
            key: float(selected_factors[:, index].mean())
            for index, key in enumerate(FACTOR_KEYS)
        },
        "coarse_selected_factors": {
            key: float(coarse_selected_factors[:, index].mean())
            for index, key in enumerate(FACTOR_KEYS)
        },
        "factor_selected_factors": {
            key: float(factor_selected_factors[:, index].mean())
            for index, key in enumerate(FACTOR_KEYS)
        },
        "base_selected_factors": {
            key: float(base_factors[:, index].mean())
            for index, key in enumerate(FACTOR_KEYS)
        },
    }


def _build_sampler(
    data: ReplayTensorSet,
    train_indices: Sequence[int],
    seed: int,
) -> WeightedRandomSampler:
    counts = Counter(data.physical_logs[index] for index in train_indices)
    weights = torch.tensor(
        [1.0 / counts[data.physical_logs[index]] for index in train_indices],
        dtype=torch.double,
    )
    generator = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(
        weights,
        num_samples=len(train_indices),
        replacement=True,
        generator=generator,
    )


def _mean_details(values: Sequence[Mapping[str, float]]) -> Dict[str, float]:
    if not values:
        return {}
    return {
        key: float(np.mean([value[key] for value in values]))
        for key in values[0]
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        action="append",
        nargs=3,
        metavar=("NAME", "FEATURE_ROOT", "LABEL_ROOT"),
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--private-observation-root",
        type=Path,
        default=None,
        help=(
            "Optional current-image spatial-token cache. When set, scorer "
            "perception is trained from these tokens instead of the released "
            "16-token Q-Former representation."
        ),
    )
    parser.add_argument(
        "--current-actor-target-root",
        type=Path,
        default=None,
        help=(
            "Optional Gate-C oracle store containing current-only actor slots. "
            "These are training labels, never inference inputs."
        ),
    )
    parser.add_argument("--fold-index", type=int, default=0)
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--fold-seed", type=int, default=20260901)
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=None,
        help=(
            "Optional JSON with train_physical_logs and validation_physical_logs. "
            "This is used for the official train/validation boundary; when absent, "
            "the deterministic balanced fold is used."
        ),
    )
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--eval-batch-size", type=int, default=48)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--candidate-keep-count", type=int, default=48)
    parser.add_argument("--fine-top-k", type=int, default=12)
    parser.add_argument("--model-dim", type=int, default=256)
    parser.add_argument("--dynamic-queries", type=int, default=12)
    parser.add_argument("--private-layers", type=int, default=2)
    parser.add_argument("--trajectory-layers", type=int, default=2)
    parser.add_argument("--candidate-layers", type=int, default=1)
    parser.add_argument("--fine-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--target-temperature", type=float, default=0.05)
    parser.add_argument("--prediction-temperature", type=float, default=0.05)
    parser.add_argument("--top-set-tolerance", type=float, default=0.01)
    parser.add_argument("--pairwise-weight", type=float, default=1.0)
    parser.add_argument("--hard-pairwise-weight", type=float, default=0.5)
    parser.add_argument("--listwise-weight", type=float, default=0.1)
    parser.add_argument("--top-set-weight", type=float, default=0.5)
    parser.add_argument("--expected-regret-weight", type=float, default=1.0)
    parser.add_argument("--top-regret-weight", type=float, default=1.0)
    parser.add_argument("--coarse-loss-weight", type=float, default=0.5)
    parser.add_argument("--factor-weight", type=float, default=0.5)
    parser.add_argument("--factor-rank-weight", type=float, default=0.25)
    parser.add_argument("--consequence-weight", type=float, default=0.5)
    parser.add_argument("--confidence-weight", type=float, default=0.1)
    parser.add_argument("--current-actor-weight", type=float, default=0.0)
    parser.add_argument("--safety-negative-weight", type=float, default=10.0)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--max-scenes-per-source", type=int, default=0)
    parser.add_argument(
        "--selection-source",
        default="",
        help=(
            "Replay source whose held-out-log PDMS selects checkpoints. "
            "Defaults to the first --source; other sources remain augmentation."
        ),
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 <= args.fold_index < args.num_folds:
        raise ValueError("fold-index must be in [0, num-folds)")
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
        raise ValueError(f"Unknown selection source: {selection_source}")
    for source in sources:
        if not source.feature_root.is_dir() or not source.label_root.is_dir():
            raise FileNotFoundError(source)
    if (
        args.private_observation_root is not None
        and not args.private_observation_root.is_dir()
    ):
        raise FileNotFoundError(args.private_observation_root)
    if args.current_actor_weight < 0:
        raise ValueError("current-actor-weight must be non-negative")
    if (args.current_actor_target_root is None) != (
        args.current_actor_weight == 0
    ):
        raise ValueError(
            "current-actor-target-root and a positive current-actor-weight "
            "must be supplied together"
        )
    if args.current_actor_target_root is not None and not (
        args.current_actor_target_root / "current.npy"
    ).is_file():
        raise FileNotFoundError(args.current_actor_target_root / "current.npy")

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
    )
    split_lineage: Dict[str, object]
    if args.split_manifest is not None:
        if not args.split_manifest.is_file():
            raise FileNotFoundError(args.split_manifest)
        split_payload = json.loads(args.split_manifest.read_text())
        declared_train_logs = {
            str(value) for value in split_payload["train_physical_logs"]
        }
        declared_validation_logs = {
            str(value) for value in split_payload["validation_physical_logs"]
        }
        if declared_train_logs.intersection(declared_validation_logs):
            raise RuntimeError("external split manifest has overlapping physical logs")
        available_logs = set(data.physical_logs)
        uncovered = available_logs.difference(
            declared_train_logs.union(declared_validation_logs)
        )
        if uncovered:
            raise RuntimeError(
                f"external split omits {len(uncovered)} available physical logs"
            )
        train_indices = [
            index
            for index, name in enumerate(data.physical_logs)
            if name in declared_train_logs
        ]
        validation_indices = [
            index
            for index, name in enumerate(data.physical_logs)
            if name in declared_validation_logs
        ]
        split_lineage = {
            "strategy": "external_physical_log_manifest",
            "path": str(args.split_manifest.resolve()),
            "sha256": _sha256(args.split_manifest),
        }
    else:
        assignment = assign_balanced_physical_log_folds(
            data.physical_logs,
            args.num_folds,
            args.fold_seed,
        )
        train_indices = [
            index
            for index, name in enumerate(data.physical_logs)
            if assignment[name] != args.fold_index
        ]
        validation_indices = [
            index
            for index, name in enumerate(data.physical_logs)
            if assignment[name] == args.fold_index
        ]
        split_lineage = {
            "strategy": "balanced_physical_log_fold",
            "fold_index": args.fold_index,
            "num_folds": args.num_folds,
            "fold_seed": args.fold_seed,
        }
    if not train_indices or not validation_indices:
        raise RuntimeError("training or validation split is empty")
    train_logs = {data.physical_logs[index] for index in train_indices}
    validation_logs = {data.physical_logs[index] for index in validation_indices}
    if train_logs.intersection(validation_logs):
        raise RuntimeError("physical log leakage between train and validation")

    if args.current_actor_weight > 0 and (
        data.current_actor_states.shape[1] != args.dynamic_queries
    ):
        raise RuntimeError(
            "dynamic query count must equal current actor target slots: "
            f"{args.dynamic_queries} != {data.current_actor_states.shape[1]}"
        )

    config = IndependentRankerConfig(
        observation_dim=int(data.observation_tokens.shape[-1]),
        max_observation_tokens=int(data.observation_tokens.shape[1]),
        status_dim=int(data.ego_features.shape[-1]),
        model_dim=args.model_dim,
        dynamic_queries=args.dynamic_queries,
        num_private_layers=args.private_layers,
        num_trajectory_layers=args.trajectory_layers,
        num_candidate_layers=args.candidate_layers,
        num_fine_layers=args.fine_layers,
        fine_top_k=args.fine_top_k,
        consequence_dim=4,
        dropout=args.dropout,
        current_actor_auxiliary=args.current_actor_weight > 0,
    )
    model = IndependentProposalRanker(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(args.epochs, 1),
        eta_min=args.learning_rate * 0.05,
    )
    train_dataset = _ReplayIndexDataset(data, train_indices)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=_build_sampler(data, train_indices, args.seed),
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=bool(args.num_workers),
        drop_last=True,
    )
    candidate_generator = torch.Generator(device=device).manual_seed(args.seed + 1000)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fold_manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "fold_index": args.fold_index,
        "num_folds": args.num_folds,
        "fold_seed": args.fold_seed,
        "split_lineage": split_lineage,
        "train_scene_count": len(train_indices),
        "validation_scene_count": len(validation_indices),
        "train_physical_logs": sorted(train_logs),
        "validation_physical_logs": sorted(validation_logs),
        "segment_log_count": len(set(data.log_names)),
        "physical_log_count": len(set(data.physical_logs)),
        "source_counts": dict(Counter(data.source_names)),
        "checkpoint_selection_source": selection_source,
        "source_lineage": source_lineage,
        "model_config": asdict(config),
        "model_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "base_score_used_as_model_input": False,
        "base_factor_logits_used_as_model_input": False,
        "base_candidate_features_used_as_model_input": False,
        "observation_source": (
            "scorer_private_current_visual_tokens"
            if args.private_observation_root is not None
            else "released_qformer_scene_tokens_control"
        ),
        "future_or_evaluator_input": False,
        "current_actor_labels_used_as_model_input": False,
        "current_actor_labels_training_only": args.current_actor_weight > 0,
        "args": vars(args)
        | {
            "output_dir": str(args.output_dir),
            "split_manifest": (
                str(args.split_manifest) if args.split_manifest is not None else None
            ),
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
        },
    }
    _atomic_json_dump(fold_manifest, args.output_dir / "fold_manifest.json")

    history: List[Dict[str, object]] = []
    selection_specs = {
        "direct": ("selected_pdms", "best_independent_scorer.pt"),
        "coarse": ("coarse_selected_pdms", "best_coarse_independent_scorer.pt"),
        "factor": ("factor_selected_pdms", "best_factor_independent_scorer.pt"),
    }
    best_values = {mode: -float("inf") for mode in selection_specs}
    best_epochs = {mode: -1 for mode in selection_specs}
    for epoch in range(args.epochs):
        model.train()
        details: List[Dict[str, float]] = []
        for batch in train_loader:
            moved = [value.to(device, non_blocking=True) for value in batch]
            optimizer.zero_grad(set_to_none=True)
            loss, batch_details = compute_training_loss(
                model,
                moved,
                args,
                candidate_generator,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            details.append(batch_details)
        scheduler.step()

        utilities, coarse_utilities, factor_logits, base_scores, target_factors = collect_predictions(
            model,
            data,
            validation_indices,
            device,
            args.eval_batch_size,
        )
        validation_physical_logs = [
            data.physical_logs[index] for index in validation_indices
        ]
        metrics = evaluate_predictions(
            utilities,
            coarse_utilities,
            factor_logits,
            base_scores,
            target_factors,
            validation_physical_logs,
            args.seed + epoch,
            args.bootstrap_replicates,
        )
        validation_source_names = [
            data.source_names[index] for index in validation_indices
        ]
        validation_by_source: Dict[str, Dict[str, object]] = {}
        source_names_in_validation = sorted(set(validation_source_names))
        if len(source_names_in_validation) == 1:
            validation_by_source[source_names_in_validation[0]] = metrics
        sources_to_evaluate = (
            [] if len(source_names_in_validation) == 1 else source_names_in_validation
        )
        for source_name in sources_to_evaluate:
            source_rows = torch.tensor(
                [
                    index
                    for index, value in enumerate(validation_source_names)
                    if value == source_name
                ],
                dtype=torch.long,
            )
            source_logs = [
                validation_physical_logs[index] for index in source_rows.tolist()
            ]
            validation_by_source[source_name] = evaluate_predictions(
                utilities[source_rows],
                coarse_utilities[source_rows],
                factor_logits[source_rows],
                base_scores[source_rows],
                target_factors[source_rows],
                source_logs,
                args.seed + epoch + 100_000,
                args.bootstrap_replicates,
            )
        record = {
            "epoch": epoch,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "training": _mean_details(details),
            "validation": metrics,
            "validation_by_source": validation_by_source,
        }
        history.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
        selection_metrics = validation_by_source[selection_source]
        for mode, (metric_key, filename) in selection_specs.items():
            selection_value = float(selection_metrics[metric_key])
            if selection_value <= best_values[mode]:
                continue
            best_values[mode] = selection_value
            best_epochs[mode] = epoch
            _atomic_torch_save(
                {
                    "schema_version": 1,
                    "architecture": "IndependentProposalRanker",
                    "selection_mode": mode,
                    "state_dict": {
                        key: value.detach().cpu()
                        for key, value in model.state_dict().items()
                    },
                    "model_config": asdict(config),
                    "epoch": epoch,
                    "validation": metrics,
                    "validation_by_source": validation_by_source,
                    "checkpoint_selection_source": selection_source,
                    "fold_manifest": fold_manifest,
                    "inference_input_schema": (
                        "current_observation_tokens",
                        "current_observation_valid_mask",
                        "current_context_feature",
                        "proposals",
                    ),
                    "forbidden_inputs": (
                        "released_base_score",
                        "released_base_factor_logits",
                        "released_candidate_features",
                        "future_annotations",
                        "future_images",
                        "official_pdm_score",
                    ),
                },
                args.output_dir / filename,
            )
        _atomic_json_dump(
            {
                "best_epoch": best_epochs["direct"],
                "best_validation_pdms": best_values["direct"],
                "best_epoch_by_selection_mode": best_epochs,
                "best_validation_pdms_by_selection_mode": best_values,
                "checkpoint_selection_source": selection_source,
                "history": history,
            },
            args.output_dir / "training_summary.json",
        )

    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir.resolve()),
                "best_epoch": best_epochs["direct"],
                "best_validation_pdms": best_values["direct"],
                "best_epoch_by_selection_mode": best_epochs,
                "best_validation_pdms_by_selection_mode": best_values,
                "checkpoint_selection_source": selection_source,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
