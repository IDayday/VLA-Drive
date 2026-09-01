#!/usr/bin/env python3
"""Train a current-observation gate between Base and an independent scorer.

The frozen DrivOR scorer independently selects one proposal from the immutable
EpisodeDrive bank.  A scorer-owned gate then predicts whether that proposal is
a safe improvement over the released Base choice.  Only the Base candidate
index is supplied; its numeric score and all future/evaluator values are absent
from the model forward and are used only for offline supervision/evaluation.
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

from local_stage2.train_conservative_reference_scorer import (
    SAFETY_TARGET_INDICES,
    _reference_positions,
    _reference_targets,
    _threshold_specs,
    _weighted_binary_loss,
)
from local_stage2.train_drivor_initialized_ranker import _split_indices
from local_stage2.train_independent_scorer import (
    ReplaySource,
    ReplayTensorSet,
    _ReplayIndexDataset,
    _atomic_json_dump,
    _atomic_torch_save,
    _build_sampler,
    _gather_candidates,
    _log_bootstrap_ci,
    _mean_details,
    _sha256,
    load_replay_sources,
)
from navsim.agents.EpisodeDrive.score_module.drivor_ranker import (
    DrivORRankerConfig,
    DrivORReferenceGateRanker,
)
from navsim.agents.EpisodeDrive.score_module.independent_ranker import (
    ConservativeReferenceConfig,
    conservative_reference_selection_scores,
    masked_pinball_quantile_loss,
    weighted_pairwise_rank_loss,
)


def _initialize_ranker(
    model: DrivORReferenceGateRanker, checkpoint: Path
) -> Dict[str, object]:
    """Load either a released DrivOR checkpoint or a re-fit ranker artifact."""

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("architecture") == "DrivORInitializedProposalRanker":
        expected = asdict(model.ranker_config)
        if dict(payload["model_config"]) != expected:
            raise RuntimeError("re-fit DrivOR ranker configuration mismatch")
        model.ranker.load_state_dict(payload["state_dict"], strict=True)
        return {
            "format": "refit_drivor_ranker_artifact",
            "selection_mode": str(payload.get("selection_mode")),
            "epoch": int(payload.get("epoch", -1)),
        }
    audit = model.ranker.load_drivor_checkpoint(checkpoint)
    return {"format": "released_drivor_checkpoint", "load_audit": audit}


def _binary_valid_mask(output: Mapping[str, torch.Tensor]) -> torch.Tensor:
    return output["allowed_candidate_mask"].bool() & ~output[
        "reference_mask"
    ].bool()


def _masked_fraction(values: torch.Tensor, mask: torch.Tensor) -> float:
    if not bool(mask.any()):
        return 0.0
    return float(values[mask].float().mean())


def compute_gate_training_loss(
    model: DrivORReferenceGateRanker,
    batch: Sequence[torch.Tensor],
    args: argparse.Namespace,
    candidate_generator: torch.Generator,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    (
        proposals,
        observation,
        observation_valid_mask,
        status,
        base_scores,
        target_factors,
        _source_indices,
    ) = batch
    # Candidate order carries no semantic information.  Randomizing it during
    # training catches accidental index/template leakage and exercises the
    # exact permutation-equivariance required at inference.
    random_keys = torch.rand(
        base_scores.shape,
        generator=candidate_generator,
        device=base_scores.device,
    )
    permutation = torch.argsort(random_keys, dim=1)
    reference_indices = _reference_positions(base_scores, permutation)
    proposals = _gather_candidates(proposals, permutation)
    target_factors = _gather_candidates(target_factors, permutation)
    targets = _reference_targets(
        target_factors,
        reference_indices,
        minimum_improvement=args.minimum_improvement_target,
        factor_epsilon=args.factor_epsilon,
    )
    output = model(
        observation.float(),
        status.float(),
        proposals,
        reference_indices,
        scene_valid_mask=observation_valid_mask,
    )
    valid = _binary_valid_mask(output)
    pair_valid = output["allowed_candidate_mask"].bool()
    quantile = masked_pinball_quantile_loss(
        output["gain_quantiles"], targets["gain"], valid_mask=valid
    )
    median_rank = weighted_pairwise_rank_loss(
        output["gain_quantiles"][..., 1],
        targets["gain"],
        valid_mask=pair_valid,
        minimum_target_delta=args.minimum_pair_delta,
    )
    safety = _weighted_binary_loss(
        output["safety_worse_logits"],
        targets["safety_worse"],
        valid,
        args.safety_worse_positive_weight,
    )
    improvement = _weighted_binary_loss(
        output["safe_improvement_logit"],
        targets["safe_improvement"],
        valid,
        args.safe_improvement_positive_weight,
    )

    lower_gain = output["gain_quantiles"][..., 0]
    harmful = (targets["gain"] <= 0.0) | targets["safety_worse"].any(dim=-1)
    harmful_mask = valid & harmful
    if bool(harmful_mask.any()):
        false_switch = F.softplus(
            lower_gain[harmful_mask] / args.switch_margin_temperature
        ).mean()
    else:
        false_switch = lower_gain.sum() * 0.0
    positive = valid & targets["safe_improvement"]
    if bool(positive.any()):
        missed_improvement = F.softplus(
            -lower_gain[positive] / args.switch_margin_temperature
        ).mean()
    else:
        missed_improvement = lower_gain.sum() * 0.0

    total = (
        args.quantile_weight * quantile
        + args.median_rank_weight * median_rank
        + args.safety_weight * safety
        + args.improvement_weight * improvement
        + args.false_switch_weight * false_switch
        + args.missed_improvement_weight * missed_improvement
    )
    return total, {
        "loss": float(total.detach()),
        "quantile": float(quantile.detach()),
        "median_rank": float(median_rank.detach()),
        "safety": float(safety.detach()),
        "improvement": float(improvement.detach()),
        "false_switch": float(false_switch.detach()),
        "missed_improvement": float(missed_improvement.detach()),
        "alternative_differs_fraction": float(valid.any(dim=1).float().mean()),
        "safe_improvement_fraction": _masked_fraction(
            targets["safe_improvement"], valid
        ),
        "harmful_fraction": _masked_fraction(harmful, valid),
    }


@torch.inference_mode()
def collect_gate_predictions(
    model: DrivORReferenceGateRanker,
    data: ReplayTensorSet,
    indices: Sequence[int],
    device: torch.device,
    batch_size: int,
) -> Dict[str, torch.Tensor]:
    model.eval()
    collected: Dict[str, List[torch.Tensor]] = {
        "gain_quantiles": [],
        "safety_worse_logits": [],
        "safe_improvement_logit": [],
        "alternative_indices": [],
        "allowed_candidate_mask": [],
        "base_scores": [],
        "target_factors": [],
    }
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
    ) in loader:
        reference_indices = base_scores.argmax(dim=1).to(device)
        output = model(
            observation.to(device, non_blocking=True).float(),
            status.to(device, non_blocking=True).float(),
            proposals.to(device, non_blocking=True),
            reference_indices,
            scene_valid_mask=observation_valid_mask.to(device, non_blocking=True),
        )
        for key in (
            "gain_quantiles",
            "safety_worse_logits",
            "safe_improvement_logit",
            "alternative_indices",
            "allowed_candidate_mask",
        ):
            collected[key].append(output[key].cpu())
        collected["base_scores"].append(base_scores.float())
        collected["target_factors"].append(target_factors.float())
    return {key: torch.cat(values) for key, values in collected.items()}


def evaluate_gate_predictions(
    prediction: Mapping[str, torch.Tensor],
    physical_logs: Sequence[str],
    threshold_specs: Sequence[Tuple[int, float, float, float]],
    seed: int,
    bootstrap_replicates: int,
) -> Dict[str, object]:
    base_scores = prediction["base_scores"]
    target_factors = prediction["target_factors"]
    target_scores = target_factors[..., -1]
    references = base_scores.argmax(dim=1)
    alternatives = prediction["alternative_indices"].long()
    rows = torch.arange(len(references))
    base_values = target_scores[rows, references]
    alternative_values = target_scores[rows, alternatives]
    union_oracle_values = torch.maximum(base_values, alternative_values)
    full_oracle_values = target_scores.max(dim=1).values
    target_safety = target_factors[..., list(SAFETY_TARGET_INDICES)]
    base_safety = target_safety[rows, references]

    policies: List[Dict[str, object]] = []
    for quantile_index, minimum_gain, maximum_safety, minimum_improvement in threshold_specs:
        scores = conservative_reference_selection_scores(
            prediction["gain_quantiles"],
            prediction["safety_worse_logits"],
            prediction["safe_improvement_logit"],
            references,
            gain_quantile_index=quantile_index,
            minimum_lcb_gain=minimum_gain,
            maximum_safety_worse_probability=maximum_safety,
            minimum_safe_improvement_probability=minimum_improvement,
            allowed_candidate_mask=prediction["allowed_candidate_mask"].bool(),
        )
        selected = scores.argmax(dim=1)
        selected_values = target_scores[rows, selected]
        selected_safety = target_safety[rows, selected]
        delta = (selected_values - base_values).numpy()
        switched = selected.ne(references)
        switched_delta = delta[switched.numpy()]
        policies.append(
            {
                "gain_quantile_index": quantile_index,
                "minimum_lcb_gain": minimum_gain,
                "maximum_safety_worse_probability": maximum_safety,
                "minimum_safe_improvement_probability": minimum_improvement,
                "selected_pdms": float(selected_values.mean()),
                "base_selected_pdms": float(base_values.mean()),
                "independent_selected_pdms": float(alternative_values.mean()),
                "base_independent_union_oracle_pdms": float(
                    union_oracle_values.mean()
                ),
                "full_candidate_oracle_pdms": float(full_oracle_values.mean()),
                "delta": float(np.mean(delta)),
                "delta_log_bootstrap_95ci": list(
                    _log_bootstrap_ci(
                        delta,
                        physical_logs,
                        seed,
                        replicates=bootstrap_replicates,
                    )
                ),
                "switch_rate": float(switched.float().mean()),
                "switch_count": int(switched.sum()),
                "improved_switch_count": int(np.sum(switched_delta > 0.0)),
                "harmful_switch_count": int(np.sum(switched_delta < 0.0)),
                "catastrophic_switch_count": int(np.sum(switched_delta <= -0.1)),
                "mean_switched_delta": (
                    float(np.mean(switched_delta)) if len(switched_delta) else 0.0
                ),
                "safety_regression_rate": float(
                    (selected_safety < (base_safety - 1.0e-6))
                    .any(dim=-1)
                    .float()
                    .mean()
                ),
            }
        )
    passing = [
        policy
        for policy in policies
        if float(policy["delta_log_bootstrap_95ci"][0]) > 0.0
    ]
    best = max(
        passing or policies,
        key=lambda policy: (
            float(policy["selected_pdms"]),
            -float(policy["catastrophic_switch_count"]),
            -float(policy["switch_rate"]),
        ),
    )
    alternative_differs = alternatives.ne(references)
    true_gain = alternative_values - base_values
    predicted_median = prediction["gain_quantiles"][rows, alternatives, 1]
    sign_accuracy = (
        predicted_median.gt(0).eq(true_gain.gt(0))[alternative_differs]
    )
    union_gain = float(union_oracle_values.mean() - base_values.mean())
    full_headroom = float(full_oracle_values.mean() - base_values.mean())
    return {
        "best_policy": best,
        "policies": policies,
        "any_positive_ci_policy": bool(passing),
        "base_selected_pdms": float(base_values.mean()),
        "independent_selected_pdms": float(alternative_values.mean()),
        "base_independent_union_oracle_pdms": float(union_oracle_values.mean()),
        "base_independent_union_oracle_gain": union_gain,
        "full_candidate_oracle_pdms": float(full_oracle_values.mean()),
        "union_fraction_of_full_headroom": (
            union_gain / full_headroom if full_headroom > 0.0 else 0.0
        ),
        "alternative_differs_fraction": float(alternative_differs.float().mean()),
        "alternative_gain_sign_accuracy": (
            float(sign_accuracy.float().mean()) if sign_accuracy.numel() else 1.0
        ),
    }


def _evaluate_by_source(
    prediction: Mapping[str, torch.Tensor],
    data: ReplayTensorSet,
    validation_indices: Sequence[int],
    thresholds: Sequence[Tuple[int, float, float, float]],
    seed: int,
    bootstrap_replicates: int,
) -> Tuple[Dict[str, object], Dict[str, Dict[str, object]]]:
    logs = [data.physical_logs[index] for index in validation_indices]
    overall = evaluate_gate_predictions(
        prediction, logs, thresholds, seed, bootstrap_replicates
    )
    source_values = [data.source_names[index] for index in validation_indices]
    by_source: Dict[str, Dict[str, object]] = {}
    for source_name in sorted(set(source_values)):
        rows = torch.tensor(
            [index for index, value in enumerate(source_values) if value == source_name],
            dtype=torch.long,
        )
        subset = {key: value[rows] for key, value in prediction.items()}
        source_logs = [logs[index] for index in rows.tolist()]
        by_source[source_name] = evaluate_gate_predictions(
            subset,
            source_logs,
            thresholds,
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
    parser.add_argument("--initialize-ranker-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--alternative-mode",
        choices=("factor", "direct", "all"),
        default="factor",
        help=(
            "factor/direct gate Base against one independent alternative; "
            "all learns reference-relative gain over every proposal"
        ),
    )
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--reference-hidden-dim", type=int, default=512)
    parser.add_argument("--reference-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--minimum-pair-delta", type=float, default=0.02)
    parser.add_argument("--minimum-improvement-target", type=float, default=0.005)
    parser.add_argument("--factor-epsilon", type=float, default=1e-6)
    parser.add_argument("--quantile-weight", type=float, default=1.0)
    parser.add_argument("--median-rank-weight", type=float, default=0.25)
    parser.add_argument("--safety-weight", type=float, default=1.0)
    parser.add_argument("--improvement-weight", type=float, default=0.5)
    parser.add_argument("--false-switch-weight", type=float, default=0.5)
    parser.add_argument("--missed-improvement-weight", type=float, default=0.25)
    parser.add_argument("--safety-worse-positive-weight", type=float, default=10.0)
    parser.add_argument("--safe-improvement-positive-weight", type=float, default=3.0)
    parser.add_argument("--switch-margin-temperature", type=float, default=0.05)
    parser.add_argument("--lcb-gain-grid", default="0.0,0.0025,0.005,0.01")
    parser.add_argument("--gain-quantile-grid", default="0,1")
    parser.add_argument("--safety-probability-grid", default="0.1,0.2,0.3,0.5")
    parser.add_argument("--improvement-probability-grid", default="0.3,0.5,0.7")
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--max-scenes-per-source", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    for path in (
        args.private_observation_root,
        args.split_manifest,
        args.initialize_ranker_checkpoint,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
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

    sources = [
        ReplaySource(name, Path(feature_root), Path(label_root))
        for name, feature_root, label_root in args.source
    ]
    if len({source.name for source in sources}) != len(sources):
        raise ValueError("source names must be unique")
    selection_source = args.selection_source or sources[0].name
    if selection_source not in {source.name for source in sources}:
        raise ValueError(f"unknown selection source: {selection_source}")
    data, source_lineage = load_replay_sources(
        sources,
        max_scenes_per_source=args.max_scenes_per_source,
        private_observation_root=args.private_observation_root,
    )
    if tuple(data.observation_tokens.shape[1:]) != (64, 256):
        raise RuntimeError("DrivOR register cache must have shape [scene,64,256]")
    if data.ego_features.shape[-1] != 11:
        raise RuntimeError("DrivOR current ego status must have width 11")
    if not bool(data.observation_valid_masks.all()):
        raise RuntimeError("DrivOR register cache unexpectedly contains padding")
    train_indices, validation_indices, split_lineage = _split_indices(
        data, args.split_manifest
    )

    ranker_config = DrivORRankerConfig()
    reference_config = ConservativeReferenceConfig(
        model_dim=ranker_config.model_dim,
        hidden_dim=args.reference_hidden_dim,
        num_heads=8,
        num_layers=args.reference_layers,
        dropout=args.dropout,
    )
    model = DrivORReferenceGateRanker(
        ranker_config,
        reference_config,
        alternative_mode=args.alternative_mode,
    )
    initialization_audit = _initialize_ranker(
        model, args.initialize_ranker_checkpoint
    )
    model.ranker.requires_grad_(False)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.reference_head.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(args.epochs, 1),
        eta_min=args.learning_rate * 0.05,
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
        "architecture": "DrivORReferenceGateRanker",
        "alternative_mode": args.alternative_mode,
        "train_scene_count": len(train_indices),
        "validation_scene_count": len(validation_indices),
        "train_physical_log_count": len(
            {data.physical_logs[index] for index in train_indices}
        ),
        "validation_physical_log_count": len(
            {data.physical_logs[index] for index in validation_indices}
        ),
        "source_counts": dict(Counter(data.source_names)),
        "selection_source": selection_source,
        "source_lineage": source_lineage,
        "split_lineage": split_lineage,
        "initializer": str(args.initialize_ranker_checkpoint.resolve()),
        "initializer_sha256": _sha256(args.initialize_ranker_checkpoint),
        "initialization_audit": initialization_audit,
        "ranker_config": asdict(ranker_config),
        "reference_config": asdict(reference_config),
        "ranker_frozen": True,
        "numeric_base_score_used_as_model_input": False,
        "base_selection_index_used_as_fallback": True,
        "future_or_evaluator_input": False,
        "inference_inputs": [
            "current_drivor_scene_registers",
            "current_ego_status",
            "proposal_geometry",
            "base_selected_candidate_index",
        ],
        "forbidden_inputs": [
            "numeric_base_score",
            "future_annotations",
            "future_images",
            "metric_cache",
            "official_pdm_score",
        ],
        "args": vars(args)
        | {
            "private_observation_root": str(args.private_observation_root),
            "split_manifest": str(args.split_manifest),
            "initialize_ranker_checkpoint": str(args.initialize_ranker_checkpoint),
            "output_dir": str(args.output_dir),
        },
    }
    _atomic_json_dump(manifest, args.output_dir / "training_manifest.json")

    thresholds = _threshold_specs(args)
    history: List[Dict[str, object]] = []
    best_pdms = -float("inf")
    best_epoch = -1
    for epoch in range(args.epochs):
        model.train()
        model.ranker.eval()
        batch_details: List[Mapping[str, float]] = []
        for batch in train_loader:
            moved = [value.to(device, non_blocking=True) for value in batch]
            optimizer.zero_grad(set_to_none=True)
            loss, details = compute_gate_training_loss(
                model, moved, args, candidate_generator
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.reference_head.parameters(), max_norm=5.0
            )
            optimizer.step()
            batch_details.append(details)
        scheduler.step()

        prediction = collect_gate_predictions(
            model, data, validation_indices, device, args.eval_batch_size
        )
        validation, validation_by_source = _evaluate_by_source(
            prediction,
            data,
            validation_indices,
            thresholds,
            args.seed + epoch,
            args.bootstrap_replicates,
        )
        record = {
            "epoch": epoch,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "training": _mean_details(batch_details),
            "validation": validation,
            "validation_by_source": validation_by_source,
        }
        history.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
        selection_metrics = validation_by_source[selection_source]
        selected_pdms = float(selection_metrics["best_policy"]["selected_pdms"])
        if selected_pdms > best_pdms:
            best_pdms = selected_pdms
            best_epoch = epoch
            _atomic_torch_save(
                {
                    "schema_version": 1,
                    "architecture": "DrivORReferenceGateRanker",
                    "alternative_mode": args.alternative_mode,
                    "state_dict": {
                        key: value.detach().cpu()
                        for key, value in model.state_dict().items()
                    },
                    "ranker_config": asdict(ranker_config),
                    "reference_config": asdict(reference_config),
                    "epoch": epoch,
                    "validation": validation,
                    "validation_by_source": validation_by_source,
                    "selected_policy": selection_metrics["best_policy"],
                    "training_manifest": manifest,
                    "inference_input_schema": (
                        "current_drivor_scene_registers",
                        "current_ego_status",
                        "proposals",
                        "base_selected_candidate_index",
                    ),
                    "forbidden_inputs": (
                        "numeric_base_score",
                        "future_annotations",
                        "future_images",
                        "official_pdm_score",
                    ),
                },
                args.output_dir / "best_drivor_reference_gate.pt",
            )
        _atomic_json_dump(
            {
                "best_epoch": best_epoch,
                "best_validation_pdms": best_pdms,
                "selection_source": selection_source,
                "history": history,
            },
            args.output_dir / "training_summary.json",
        )

    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir.resolve()),
                "best_epoch": best_epoch,
                "best_validation_pdms": best_pdms,
                "selection_source": selection_source,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
