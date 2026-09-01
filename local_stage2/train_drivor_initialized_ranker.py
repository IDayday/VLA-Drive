#!/usr/bin/env python3
"""Re-fit a scorer-private DrivOR ranker on immutable EpisodeDrive proposals.

The trainable model receives only current-camera DrivOR registers, current ego
status, and proposal geometry.  PDM factors are offline labels and the released
EpisodeDrive score is retained exclusively for held-out comparison.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from local_stage2.train_independent_scorer import (
    ReplaySource,
    ReplayTensorSet,
    TARGET_TO_MODEL_FACTOR_ORDER,
    _ReplayIndexDataset,
    _atomic_json_dump,
    _atomic_torch_save,
    _build_sampler,
    _candidate_dropout_indices,
    _gather_candidates,
    _mean_details,
    _sha256,
    evaluate_predictions,
    load_replay_sources,
)
from local_stage2.train_public_base_residual_scorer import (
    binary_factor_loss,
    expected_regret_loss,
    top_set_cross_entropy,
)
from navsim.agents.EpisodeDrive.score_module.drivor_ranker import (
    DrivORInitializedProposalRanker,
    DrivORRankerConfig,
)
from navsim.agents.EpisodeDrive.score_module.independent_ranker import (
    pdms_factor_log_utility,
    top_heavy_listwise_loss,
    top_regret_rank_loss,
    weighted_pairwise_rank_loss,
)


def _split_indices(
    data: ReplayTensorSet,
    split_manifest: Path,
) -> Tuple[List[int], List[int], Dict[str, object]]:
    payload = json.loads(split_manifest.read_text())
    train_logs = {str(value) for value in payload["train_physical_logs"]}
    validation_logs = {
        str(value) for value in payload["validation_physical_logs"]
    }
    overlap = train_logs.intersection(validation_logs)
    if overlap:
        raise RuntimeError(f"physical-log split overlaps: {sorted(overlap)[:3]}")
    available = set(data.physical_logs)
    missing = available.difference(train_logs.union(validation_logs))
    if missing:
        raise RuntimeError(f"split omits {len(missing)} available physical logs")
    train = [
        index for index, value in enumerate(data.physical_logs) if value in train_logs
    ]
    validation = [
        index
        for index, value in enumerate(data.physical_logs)
        if value in validation_logs
    ]
    if not train or not validation:
        raise RuntimeError("empty train or validation split")
    actual_train = {data.physical_logs[index] for index in train}
    actual_validation = {data.physical_logs[index] for index in validation}
    if actual_train.intersection(actual_validation):
        raise RuntimeError("physical-log leakage after row selection")
    return train, validation, {
        "path": str(split_manifest.resolve()),
        "sha256": _sha256(split_manifest),
        "train_physical_logs": sorted(actual_train),
        "validation_physical_logs": sorted(actual_validation),
    }


def direct_score_regression_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    beta: float,
) -> torch.Tensor:
    """Calibrate the direct utility to a bounded candidate PDMS estimate."""

    if prediction.shape != target.shape or prediction.ndim != 2:
        raise ValueError("prediction and target must share shape [B,K]")
    if beta <= 0.0:
        raise ValueError("direct regression beta must be positive")
    return F.smooth_l1_loss(prediction.sigmoid(), target, beta=beta)


def compute_training_loss(
    model: DrivORInitializedProposalRanker,
    batch: Sequence[torch.Tensor],
    args: argparse.Namespace,
    candidate_generator: torch.Generator,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    (
        proposals,
        observation,
        observation_valid_mask,
        status,
        _base_scores,
        target_factors,
        _indices,
        *_training_only_targets,
    ) = batch
    candidate_indices = _candidate_dropout_indices(
        target_factors[..., -1],
        args.candidate_keep_count,
        candidate_generator,
    )
    proposals = _gather_candidates(proposals, candidate_indices)
    target_factors = _gather_candidates(target_factors, candidate_indices)
    target_scores = target_factors[..., -1]
    output = model(
        observation.float(),
        status.float(),
        proposals,
        scene_valid_mask=observation_valid_mask,
    )
    factor_logits = output["factor_logits"]
    direct_utility = output["direct_utility"]
    reorder = torch.tensor(
        TARGET_TO_MODEL_FACTOR_ORDER,
        device=target_factors.device,
    )
    target_six = target_factors.index_select(-1, reorder)
    factor = binary_factor_loss(
        factor_logits,
        target_six,
        args.safety_negative_weight,
    )
    progress = F.smooth_l1_loss(
        factor_logits[..., 4].sigmoid(),
        target_six[..., 4],
    )
    factor = factor + args.progress_weight * progress
    factor_utility = pdms_factor_log_utility(factor_logits)
    factor_rank = (
        weighted_pairwise_rank_loss(
            factor_utility, target_scores, minimum_target_delta=0.02
        )
        + weighted_pairwise_rank_loss(
            factor_utility, target_scores, minimum_target_delta=0.05
        )
        + weighted_pairwise_rank_loss(
            factor_utility, target_scores, minimum_target_delta=0.10
        )
    ) / 3.0
    direct_pairwise = (
        weighted_pairwise_rank_loss(
            direct_utility, target_scores, minimum_target_delta=0.02
        )
        + weighted_pairwise_rank_loss(
            direct_utility, target_scores, minimum_target_delta=0.05
        )
        + weighted_pairwise_rank_loss(
            direct_utility, target_scores, minimum_target_delta=0.10
        )
    ) / 3.0
    direct_listwise = top_heavy_listwise_loss(
        direct_utility,
        target_scores,
        temperature=args.target_temperature,
    )
    direct_top_set = top_set_cross_entropy(
        direct_utility,
        target_scores,
        tolerance=args.top_set_tolerance,
        prediction_temperature=args.prediction_temperature,
    )
    direct_expected_regret = expected_regret_loss(
        direct_utility,
        target_scores,
        prediction_temperature=args.prediction_temperature,
    )
    direct_top_regret = top_regret_rank_loss(
        direct_utility,
        target_scores,
        minimum_target_delta=0.01,
    )
    direct_regression = direct_score_regression_loss(
        direct_utility,
        target_scores,
        beta=args.direct_regression_beta,
    )
    total = (
        args.factor_weight * factor
        + args.factor_rank_weight * factor_rank
        + args.direct_pairwise_weight * direct_pairwise
        + args.direct_listwise_weight * direct_listwise
        + args.direct_top_set_weight * direct_top_set
        + args.direct_expected_regret_weight * direct_expected_regret
        + args.direct_top_regret_weight * direct_top_regret
        + args.direct_regression_weight * direct_regression
    )
    details = {
        "loss": total,
        "factor": factor,
        "progress": progress,
        "factor_rank": factor_rank,
        "direct_pairwise": direct_pairwise,
        "direct_listwise": direct_listwise,
        "direct_top_set": direct_top_set,
        "direct_expected_regret": direct_expected_regret,
        "direct_top_regret": direct_top_regret,
        "direct_regression": direct_regression,
    }
    return total, {
        key: float(value.detach()) for key, value in details.items()
    }


@torch.inference_mode()
def collect_predictions(
    model: DrivORInitializedProposalRanker,
    data: ReplayTensorSet,
    indices: Sequence[int],
    device: torch.device,
    batch_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    model.eval()
    direct: List[torch.Tensor] = []
    factors: List[torch.Tensor] = []
    base: List[torch.Tensor] = []
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
        status,
        base_scores,
        target_factors,
        _source_indices,
        *_training_only_targets,
    ) in loader:
        output = model(
            observation.to(device, non_blocking=True).float(),
            status.to(device, non_blocking=True).float(),
            proposals.to(device, non_blocking=True),
            scene_valid_mask=observation_valid_mask.to(device, non_blocking=True),
        )
        direct.append(output["direct_utility"].float().cpu())
        factors.append(output["factor_logits"].float().cpu())
        base.append(base_scores.float())
        targets.append(target_factors.float())
    return (
        torch.cat(direct),
        torch.cat(factors),
        torch.cat(base),
        torch.cat(targets),
    )


def _evaluate_by_source(
    direct: torch.Tensor,
    factors: torch.Tensor,
    base: torch.Tensor,
    targets: torch.Tensor,
    data: ReplayTensorSet,
    validation_indices: Sequence[int],
    seed: int,
    bootstrap_replicates: int,
) -> Tuple[Dict[str, object], Dict[str, Dict[str, object]]]:
    logs = [data.physical_logs[index] for index in validation_indices]
    # evaluate_predictions' coarse and direct paths intentionally receive the
    # same independent utility: this ranker has no shortlist stage.
    overall = evaluate_predictions(
        direct,
        direct,
        factors,
        base,
        targets,
        logs,
        seed,
        bootstrap_replicates,
    )
    source_values = [data.source_names[index] for index in validation_indices]
    by_source: Dict[str, Dict[str, object]] = {}
    for source_name in sorted(set(source_values)):
        rows = torch.tensor(
            [index for index, value in enumerate(source_values) if value == source_name],
            dtype=torch.long,
        )
        source_logs = [logs[index] for index in rows.tolist()]
        by_source[source_name] = evaluate_predictions(
            direct[rows],
            direct[rows],
            factors[rows],
            base[rows],
            targets[rows],
            source_logs,
            seed + 100_000,
            bootstrap_replicates,
        )
    return overall, by_source


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        action="append",
        nargs=3,
        metavar=("NAME", "FEATURE_ROOT", "LABEL_ROOT"),
        required=True,
    )
    parser.add_argument("--selection-source", default="")
    parser.add_argument("--private-observation-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--drivor-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pretrained-learning-rate", type=float, default=3e-5)
    parser.add_argument("--direct-learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--candidate-keep-count", type=int, default=48)
    parser.add_argument("--projection-dropout", type=float, default=0.1)
    parser.add_argument("--drop-path", type=float, default=0.2)
    parser.add_argument("--target-temperature", type=float, default=0.05)
    parser.add_argument("--prediction-temperature", type=float, default=0.05)
    parser.add_argument("--top-set-tolerance", type=float, default=0.01)
    parser.add_argument("--factor-weight", type=float, default=1.0)
    parser.add_argument("--factor-rank-weight", type=float, default=0.5)
    parser.add_argument("--progress-weight", type=float, default=2.0)
    parser.add_argument("--direct-pairwise-weight", type=float, default=1.0)
    parser.add_argument("--direct-listwise-weight", type=float, default=0.1)
    parser.add_argument("--direct-top-set-weight", type=float, default=0.5)
    parser.add_argument("--direct-expected-regret-weight", type=float, default=1.0)
    parser.add_argument("--direct-top-regret-weight", type=float, default=1.0)
    parser.add_argument("--direct-regression-weight", type=float, default=0.0)
    parser.add_argument("--direct-regression-beta", type=float, default=0.1)
    parser.add_argument("--safety-negative-weight", type=float, default=10.0)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--max-scenes-per-source", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sources = [
        ReplaySource(name, Path(feature_root), Path(label_root))
        for name, feature_root, label_root in args.source
    ]
    if len({source.name for source in sources}) != len(sources):
        raise ValueError("source names must be unique")
    selection_source = args.selection_source or sources[0].name
    if selection_source not in {source.name for source in sources}:
        raise ValueError(f"unknown selection source: {selection_source}")
    for path in (
        args.private_observation_root,
        args.split_manifest,
        args.drivor_checkpoint,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
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
    if tuple(data.observation_tokens.shape[1:]) != (64, 256):
        raise RuntimeError(
            "DrivOR current register cache must have shape [scene,64,256]"
        )
    if data.ego_features.shape[-1] != 11:
        raise RuntimeError("DrivOR current ego status must have width 11")
    if not bool(data.observation_valid_masks.all()):
        raise RuntimeError("DrivOR register cache unexpectedly contains padding")
    train_indices, validation_indices, split_lineage = _split_indices(
        data, args.split_manifest
    )

    config = DrivORRankerConfig(
        projection_dropout=args.projection_dropout,
        drop_path=args.drop_path,
    )
    model = DrivORInitializedProposalRanker(config)
    initialization_audit = model.load_drivor_checkpoint(args.drivor_checkpoint)
    model.to(device)
    direct_parameters = list(model.direct_utility_head.parameters())
    direct_ids = {id(parameter) for parameter in direct_parameters}
    pretrained_parameters = [
        parameter for parameter in model.parameters() if id(parameter) not in direct_ids
    ]
    optimizer = torch.optim.AdamW(
        [
            {
                "params": pretrained_parameters,
                "lr": args.pretrained_learning_rate,
            },
            {"params": direct_parameters, "lr": args.direct_learning_rate},
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(args.epochs, 1),
        eta_min=min(args.pretrained_learning_rate, args.direct_learning_rate) * 0.05,
    )
    train_loader = DataLoader(
        _ReplayIndexDataset(data, train_indices),
        batch_size=args.batch_size,
        sampler=_build_sampler(data, train_indices, args.seed),
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=bool(args.num_workers),
        drop_last=True,
    )
    candidate_generator = torch.Generator(device=device).manual_seed(args.seed + 1000)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "architecture": "DrivORInitializedProposalRanker",
        "train_scene_count": len(train_indices),
        "validation_scene_count": len(validation_indices),
        "train_physical_log_count": len(
            {data.physical_logs[index] for index in train_indices}
        ),
        "validation_physical_log_count": len(
            {data.physical_logs[index] for index in validation_indices}
        ),
        "source_counts": dict(Counter(data.source_names)),
        "checkpoint_selection_source": selection_source,
        "source_lineage": source_lineage,
        "split_lineage": split_lineage,
        "drivor_checkpoint": str(args.drivor_checkpoint.resolve()),
        "drivor_checkpoint_sha256": _sha256(args.drivor_checkpoint),
        "initialization_audit": initialization_audit,
        "model_config": asdict(config),
        "model_parameter_count": sum(
            parameter.numel() for parameter in model.parameters()
        ),
        "inference_inputs": [
            "current_camera_scene_registers",
            "current_ego_status",
            "proposal_geometry",
        ],
        "forbidden_inputs": [
            "released_episode_drive_score",
            "future_annotations",
            "future_images",
            "metric_cache",
            "official_pdm_score",
        ],
        "base_score_used_as_model_input": False,
        "future_or_evaluator_input": False,
        "args": vars(args)
        | {
            "private_observation_root": str(args.private_observation_root),
            "split_manifest": str(args.split_manifest),
            "drivor_checkpoint": str(args.drivor_checkpoint),
            "output_dir": str(args.output_dir),
        },
    }
    _atomic_json_dump(manifest, args.output_dir / "training_manifest.json")

    history: List[Dict[str, object]] = []
    selection_specs = {
        "direct": ("selected_pdms", "best_direct_ranker.pt"),
        "factor": ("factor_selected_pdms", "best_factor_ranker.pt"),
    }
    best_values = {mode: -float("inf") for mode in selection_specs}
    best_epochs = {mode: -1 for mode in selection_specs}
    for epoch in range(args.epochs):
        model.train()
        batch_details: List[Mapping[str, float]] = []
        for batch in train_loader:
            moved = [value.to(device, non_blocking=True) for value in batch]
            optimizer.zero_grad(set_to_none=True)
            loss, details = compute_training_loss(
                model, moved, args, candidate_generator
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            batch_details.append(details)
        scheduler.step()

        direct, factors, base, targets = collect_predictions(
            model,
            data,
            validation_indices,
            device,
            args.eval_batch_size,
        )
        validation, validation_by_source = _evaluate_by_source(
            direct,
            factors,
            base,
            targets,
            data,
            validation_indices,
            args.seed + epoch,
            args.bootstrap_replicates,
        )
        record = {
            "epoch": epoch,
            "learning_rates": [
                float(group["lr"]) for group in optimizer.param_groups
            ],
            "training": _mean_details(batch_details),
            "validation": validation,
            "validation_by_source": validation_by_source,
        }
        history.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
        selection_metrics = validation_by_source[selection_source]
        for mode, (metric_key, filename) in selection_specs.items():
            value = float(selection_metrics[metric_key])
            if value <= best_values[mode]:
                continue
            best_values[mode] = value
            best_epochs[mode] = epoch
            _atomic_torch_save(
                {
                    "schema_version": 1,
                    "architecture": "DrivORInitializedProposalRanker",
                    "selection_mode": mode,
                    "state_dict": {
                        key: tensor.detach().cpu()
                        for key, tensor in model.state_dict().items()
                    },
                    "model_config": asdict(config),
                    "epoch": epoch,
                    "validation": validation,
                    "validation_by_source": validation_by_source,
                    "training_manifest": manifest,
                },
                args.output_dir / filename,
            )
        _atomic_json_dump(
            {
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
