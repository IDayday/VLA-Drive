#!/usr/bin/env python3
"""Train an M0-native scorer-private residual on frozen proposal replay."""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
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
    episode_drive_factor_loss,
    pdms_factor_log_utility,
)
from navsim.agents.EpisodeDrive.score_module.m0_private_residual_ranker import (
    M0PrivateResidualConfig,
    M0PrivateResidualRanker,
    base_anchored_topk_indices,
)


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


class ResidualReplayDataset(Dataset):
    def __init__(
        self,
        data: ReplayTensorSet,
        base_factor_logits: torch.Tensor,
        indices: Sequence[int],
    ) -> None:
        if base_factor_logits.shape != (len(data), 64, len(FACTOR_KEYS)):
            raise ValueError("Base factor logits do not align with replay rows")
        self.data = data
        self.base_factor_logits = base_factor_logits
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
            self.base_factor_logits[source_index],
            self.data.target_factors[source_index],
            torch.tensor(source_index, dtype=torch.long),
        )


def compute_residual_training_loss(
    model: M0PrivateResidualRanker,
    batch: Sequence[torch.Tensor],
    args: argparse.Namespace,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    (
        proposals,
        observation,
        observation_valid_mask,
        status,
        base_scores,
        base_factor_logits,
        target_factors,
        _indices,
    ) = batch
    output = model(
        observation.float(),
        status.float(),
        proposals,
        base_factor_logits,
        base_scores,
        observation_valid_mask=observation_valid_mask,
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

    reorder = torch.tensor(
        TARGET_TO_MODEL_FACTOR_ORDER,
        device=target_factors.device,
    )
    target_six = target_factors.index_select(-1, reorder)
    factor = episode_drive_factor_loss(
        output["refined_factor_logits"],
        target_six,
        args.safety_negative_weight,
    )
    private_factor = episode_drive_factor_loss(
        output["private_factor_logits"],
        target_six,
        args.safety_negative_weight,
    )
    factor_rank = weighted_pairwise_loss(
        pdms_factor_log_utility(output["refined_factor_logits"]),
        target_factors[..., -1],
        args.factor_rank_minimum_delta,
    )

    relative_target = relative_safety_targets(target_six, base_scores)
    relative_element = F.binary_cross_entropy_with_logits(
        output["relative_safety_logits"],
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
    total = (
        args.pairwise_weight * pairwise
        + args.base_pairwise_weight * base_pairwise
        + args.listwise_weight * listwise
        + args.top_set_weight * top_set
        + args.expected_regret_weight * expected_regret
        + args.factor_weight * factor
        + args.private_factor_weight * private_factor
        + args.factor_rank_weight * factor_rank
        + args.relative_safety_weight * relative_safety
        + args.residual_l2_weight * residual_l2
    )
    details = {
        "loss": total,
        "pairwise": pairwise,
        "base_pairwise": base_pairwise,
        "listwise": listwise,
        "top_set": top_set,
        "expected_regret": expected_regret,
        "factor": factor,
        "private_factor": private_factor,
        "factor_rank": factor_rank,
        "relative_safety": relative_safety,
        "residual_l2": residual_l2,
    }
    return total, {
        key: float(value.detach()) for key, value in details.items()
    }


@torch.inference_mode()
def collect_residual_predictions(
    model: M0PrivateResidualRanker,
    data: ReplayTensorSet,
    base_factor_logits: torch.Tensor,
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
        ResidualReplayDataset(data, base_factor_logits, indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    for batch in loader:
        (
            proposals,
            observation,
            observation_valid_mask,
            status,
            base_scores,
            factor_logits,
            target_factors,
            _source_indices,
        ) = batch
        output = model(
            observation.to(device, non_blocking=True).float(),
            status.to(device, non_blocking=True).float(),
            proposals.to(device, non_blocking=True),
            factor_logits.to(device, non_blocking=True),
            base_scores.to(device, non_blocking=True),
            observation_valid_mask=observation_valid_mask.to(
                device, non_blocking=True
            ),
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
    parser.add_argument("--private-observation-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--selection-source", default="")
    parser.add_argument("--output-dir", type=Path, required=True)
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
    parser.add_argument("--factor-weight", type=float, default=1.0)
    parser.add_argument("--private-factor-weight", type=float, default=0.25)
    parser.add_argument("--factor-rank-weight", type=float, default=0.5)
    parser.add_argument("--relative-safety-weight", type=float, default=0.5)
    parser.add_argument("--residual-l2-weight", type=float, default=0.01)
    parser.add_argument("--safety-negative-weight", type=float, default=1.0)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--max-scenes-per-source", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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
    for path in (args.private_observation_root, args.split_manifest):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)

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
    )
    factor_tokens, base_factor_logits = load_replay_base_factor_logits(
        sources,
        max_scenes_per_source=args.max_scenes_per_source,
    )
    if factor_tokens != data.tokens:
        raise RuntimeError("Base factor logits do not match replay token order")

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
    )
    residual_config = M0PrivateResidualConfig(
        hidden_dim=args.model_dim,
        num_layers=args.residual_layers,
        num_heads=8,
        dropout=args.dropout,
        top_k=args.residual_top_k,
        max_residual=args.max_residual,
        score_mode=args.score_mode,
    )
    model = M0PrivateResidualRanker(private_config, residual_config).to(device)
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
    train_loader = DataLoader(
        ResidualReplayDataset(data, base_factor_logits, train_indices),
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
        "private_observation_root": str(args.private_observation_root),
        "split_manifest": str(args.split_manifest),
    }
    fold_manifest: Dict[str, object] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "split_lineage": {
            "strategy": "external_physical_log_manifest",
            "path": str(args.split_manifest.resolve()),
            "sha256": _sha256(args.split_manifest),
        },
        "train_scene_count": len(train_indices),
        "validation_scene_count": len(validation_indices),
        "train_physical_logs": sorted(train_logs),
        "validation_physical_logs": sorted(validation_logs),
        "source_counts": dict(Counter(data.source_names)),
        "checkpoint_selection_source": selection_source,
        "source_lineage": source_lineage,
        "private_config": asdict(private_config),
        "residual_config": asdict(residual_config),
        "model_parameter_count": sum(
            parameter.numel() for parameter in model.parameters()
        ),
        "m0_base_factor_logits_used_as_model_input": True,
        "m0_base_numeric_score_used_as_model_input": True,
        "external_model_representation_or_weight_used": False,
        "future_or_evaluator_input": False,
        "official_score_input": False,
        "args": serialized_args,
    }
    _atomic_json_dump(fold_manifest, args.output_dir / "fold_manifest.json")

    history: List[Dict[str, object]] = []
    best_value = -float("inf")
    best_epoch = -1
    artifact_path = args.output_dir / "best_m0_private_residual_scorer.pt"
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

        predictions = collect_residual_predictions(
            model,
            data,
            base_factor_logits,
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
                    "schema_version": 1,
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
                        "m0_current_visual_tokens",
                        "m0_current_context_feature",
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

    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir.resolve()),
                "artifact": str(artifact_path.resolve()),
                "artifact_sha256": _sha256(artifact_path),
                "best_epoch": best_epoch,
                "best_validation_pdms": best_value,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
