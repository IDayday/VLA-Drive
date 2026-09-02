#!/usr/bin/env python3
"""Calibrate a conservative M0-private residual policy on physical logs.

The learned residual is frozen.  A balanced half of the predeclared held-out
physical logs selects a deployment-only switch policy; the disjoint half
reports promotion metrics.  Navtest is never read by this program.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from local_stage2.train_independent_scorer import (
    ReplaySource,
    TARGET_FACTOR_KEYS,
    _atomic_json_dump,
    _atomic_torch_save,
    _sha256,
    assign_balanced_physical_log_folds,
    load_replay_sources,
)
from local_stage2.train_m0_private_residual_scorer import (
    ResidualReplayDataset,
    load_replay_base_factor_logits,
)
from local_stage2.train_public_base_residual_scorer import _log_bootstrap_ci
from navsim.agents.EpisodeDrive.score_module.independent_ranker import (
    FACTOR_KEYS,
    IndependentRankerConfig,
)
from navsim.agents.EpisodeDrive.score_module.m0_private_residual_ranker import (
    M0PrivateResidualConfig,
    M0PrivateResidualRanker,
)


def balanced_calibration_split(
    physical_logs: Sequence[str],
    seed: int,
) -> Tuple[List[int], List[int], Dict[str, int]]:
    """Return scene rows for two balanced, disjoint physical-log halves."""

    assignment = assign_balanced_physical_log_folds(physical_logs, 2, seed)
    calibration = [
        index
        for index, name in enumerate(physical_logs)
        if assignment[str(name)] == 0
    ]
    promotion = [
        index
        for index, name in enumerate(physical_logs)
        if assignment[str(name)] == 1
    ]
    if not calibration or not promotion:
        raise RuntimeError("calibration/promotion split is empty")
    calibration_logs = {str(physical_logs[index]) for index in calibration}
    promotion_logs = {str(physical_logs[index]) for index in promotion}
    if calibration_logs.intersection(promotion_logs):
        raise RuntimeError("physical-log leakage in policy calibration split")
    return calibration, promotion, assignment


@torch.inference_mode()
def collect_policy_tensors(
    model: M0PrivateResidualRanker,
    data,
    base_factor_logits: torch.Tensor,
    indices: Sequence[int],
    device: torch.device,
    batch_size: int,
) -> Dict[str, torch.Tensor]:
    model.eval()
    parts: Dict[str, List[torch.Tensor]] = {
        key: []
        for key in (
            "residual",
            "refined_factor_logits",
            "relative_safety_logits",
            "base_scores",
            "target_factors",
        )
    }
    loader = DataLoader(
        ResidualReplayDataset(
            data,
            base_factor_logits,
            indices,
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
        m0_context = {}
        if model.residual_config.m0_context_fusion:
            m0_context = {
                "m0_scene_features": batch[8].to(
                    device, non_blocking=True
                ).float(),
                "m0_ego_features": batch[9].to(
                    device, non_blocking=True
                ).float(),
            }
        output = model(
            observation.to(device, non_blocking=True).float(),
            status.to(device, non_blocking=True).float(),
            proposals.to(device, non_blocking=True),
            factor_logits.to(device, non_blocking=True),
            base_scores.to(device, non_blocking=True),
            observation_valid_mask=observation_valid_mask.to(
                device, non_blocking=True
            ),
            **m0_context,
        )
        parts["residual"].append(output["residual"].float().cpu())
        parts["refined_factor_logits"].append(
            output["refined_factor_logits"].float().cpu()
        )
        parts["relative_safety_logits"].append(
            output["relative_safety_logits"].float().cpu()
        )
        parts["base_scores"].append(base_scores.float())
        parts["target_factors"].append(target_factors.float())
    return {key: torch.cat(values) for key, values in parts.items()}


def policy_selection_indices(
    tensors: Mapping[str, torch.Tensor],
    rows: torch.Tensor,
    policy: Mapping[str, object],
    device: torch.device,
) -> torch.Tensor:
    """Apply one deployment policy using predictions only."""

    base_tensor = tensors["base_scores"].to(device)
    row_indices = rows.to(device)
    base_scores = base_tensor.index_select(0, row_indices)
    residual = tensors["residual"].to(device).index_select(0, row_indices)
    factor_logits = tensors["refined_factor_logits"].to(device).index_select(
        0, row_indices
    )
    relative_logits = tensors["relative_safety_logits"].to(device).index_select(
        0, row_indices
    )
    base_indices = base_scores.argmax(dim=1, keepdim=True)
    base_mask = torch.zeros_like(base_scores, dtype=torch.bool)
    base_mask.scatter_(1, base_indices, True)

    mode = str(policy["safety_gate_mode"])
    floor = float(policy["safety_floor"])
    tolerance = float(policy["safety_relative_tolerance"])
    if mode == "none":
        safety_mask = torch.ones_like(base_mask)
    elif mode == "relative_factor":
        safety_mask = (relative_logits.sigmoid() >= floor).all(dim=-1)
    elif mode == "factor_all":
        safety_indices = [0, 1, 3]
        if bool(policy["preserve_ddc"]):
            safety_indices.append(2)
        probabilities = factor_logits.sigmoid()[..., safety_indices]
        base_probabilities = probabilities.gather(
            1,
            base_indices[..., None].expand(-1, 1, len(safety_indices)),
        )
        safety_mask = (probabilities >= floor).all(dim=-1)
        safety_mask &= (
            probabilities >= base_probabilities - tolerance
        ).all(dim=-1)
    else:
        raise ValueError(f"unknown safety gate mode: {mode}")

    adjusted = (
        base_scores
        + float(policy["inference_scale"]) * residual
        - (~base_mask).to(base_scores.dtype) * float(policy["switch_penalty"])
    )
    eligible = safety_mask | base_mask
    return torch.where(eligible, adjusted, base_scores - 100.0).argmax(dim=1)


@torch.inference_mode()
def summarize_policy_on_device(
    tensors: Mapping[str, torch.Tensor],
    rows: torch.Tensor,
    policy: Mapping[str, object],
    device: torch.device,
) -> Dict[str, object]:
    """Compute grid-selection statistics without repeated host transfers."""

    row_indices = rows.to(device)
    selected = policy_selection_indices(tensors, rows, policy, device)
    base_scores = tensors["base_scores"].to(device).index_select(0, row_indices)
    target_scores = (
        tensors["target_factors"].to(device).index_select(0, row_indices)[..., -1]
    )
    scene_rows = torch.arange(len(rows), device=device)
    base = base_scores.argmax(dim=1)
    selected_values = target_scores[scene_rows, selected]
    base_values = target_scores[scene_rows, base]
    delta = selected_values - base_values
    return {
        "selected_pdms": float(selected_values.mean()),
        "selected_delta": float(delta.mean()),
        "switch_rate": float((selected != base).float().mean()),
        "wins": int((delta > 1.0e-9).sum()),
        "losses": int((delta < -1.0e-9).sum()),
        "policy": dict(policy),
    }


def evaluate_policy(
    tensors: Mapping[str, torch.Tensor],
    rows: Sequence[int],
    physical_logs: Sequence[str],
    policy: Mapping[str, object],
    device: torch.device,
    seed: int,
    bootstrap_replicates: int,
) -> Dict[str, object]:
    row_tensor = torch.as_tensor(rows, dtype=torch.long)
    selected = policy_selection_indices(tensors, row_tensor, policy, device).cpu()
    base_scores = tensors["base_scores"].index_select(0, row_tensor)
    target_factors = tensors["target_factors"].index_select(0, row_tensor)
    target_scores = target_factors[..., -1]
    base = base_scores.argmax(dim=1)
    oracle = target_scores.argmax(dim=1)
    scene_rows = torch.arange(len(row_tensor))
    selected_values = target_scores[scene_rows, selected]
    base_values = target_scores[scene_rows, base]
    oracle_values = target_scores[scene_rows, oracle]
    delta = (selected_values - base_values).numpy()
    selected_factors = target_factors[scene_rows, selected]
    base_factors = target_factors[scene_rows, base]
    selected_logs = [str(physical_logs[index]) for index in rows]
    interval = (
        _log_bootstrap_ci(
            delta,
            selected_logs,
            seed,
            replicates=bootstrap_replicates,
        )
        if bootstrap_replicates > 0
        else (float("nan"), float("nan"))
    )
    wins = int((selected_values > base_values + 1.0e-9).sum())
    losses = int((selected_values < base_values - 1.0e-9).sum())
    return {
        "scene_count": len(rows),
        "physical_log_count": len(set(selected_logs)),
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
        "ties": int(len(rows) - wins - losses),
        "selected_factors": {
            key: float(selected_factors[:, index].mean())
            for index, key in enumerate(TARGET_FACTOR_KEYS)
        },
        "base_selected_factors": {
            key: float(base_factors[:, index].mean())
            for index, key in enumerate(TARGET_FACTOR_KEYS)
        },
        "policy": dict(policy),
    }


def policy_grid() -> List[Dict[str, object]]:
    scales = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)
    penalties = (0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2)
    floors = (0.5, 0.7, 0.85, 0.95, 0.99)
    tolerances = (0.0, 0.02, 0.05, 0.1)
    gates: List[Tuple[str, float, float, bool]] = [("none", 0.0, 1.0, False)]
    gates.extend(("relative_factor", floor, 1.0, False) for floor in floors)
    gates.extend(
        ("factor_all", floor, tolerance, preserve_ddc)
        for floor in floors
        for tolerance in tolerances
        for preserve_ddc in (False, True)
    )
    return [
        {
            "inference_scale": scale,
            "switch_penalty": penalty,
            "safety_floor": floor,
            "safety_relative_tolerance": tolerance,
            "preserve_ddc": preserve_ddc,
            "safety_gate_mode": mode,
        }
        for mode, floor, tolerance, preserve_ddc in gates
        for scale in scales
        for penalty in penalties
    ]


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
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=24)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    sources = [
        ReplaySource(name, Path(feature_root), Path(label_root))
        for name, feature_root, label_root in args.source
    ]
    selection_source = args.selection_source or sources[0].name
    if selection_source not in {source.name for source in sources}:
        raise ValueError(f"unknown selection source: {selection_source}")
    artifact = torch.load(args.artifact, map_location="cpu", weights_only=False)
    if artifact.get("architecture") != "M0PrivateResidualRanker":
        raise RuntimeError("source artifact architecture mismatch")
    if str(artifact["checkpoint_selection_source"]) != selection_source:
        raise RuntimeError("source artifact selection source mismatch")

    data, source_lineage = load_replay_sources(
        sources,
        private_observation_root=args.private_observation_root,
        retain_m0_context=bool(
            artifact["residual_config"].get("m0_context_fusion", False)
        ),
    )
    factor_tokens, base_factor_logits = load_replay_base_factor_logits(sources)
    if factor_tokens != data.tokens:
        raise RuntimeError("Base factor logits do not match replay token order")
    split = json.loads(args.split_manifest.read_text())
    validation_logs = {str(value) for value in split["validation_physical_logs"]}
    validation_indices = [
        index
        for index, (log_name, source_name) in enumerate(
            zip(data.physical_logs, data.source_names)
        )
        if log_name in validation_logs and source_name == selection_source
    ]
    if not validation_indices:
        raise RuntimeError("selection source has no validation scenes")
    selected_physical_logs = [
        data.physical_logs[index] for index in validation_indices
    ]
    calibration_rows, promotion_rows, assignment = balanced_calibration_split(
        selected_physical_logs,
        args.seed,
    )

    device = torch.device(args.device)
    model = M0PrivateResidualRanker(
        IndependentRankerConfig(**dict(artifact["private_config"])),
        M0PrivateResidualConfig(**dict(artifact["residual_config"])),
    ).to(device)
    model.load_state_dict(artifact["state_dict"], strict=True)
    tensors = collect_policy_tensors(
        model,
        data,
        base_factor_logits,
        validation_indices,
        device,
        args.eval_batch_size,
    )

    calibration_summaries: List[Dict[str, object]] = []
    policy_device_tensors = {
        key: value.to(device)
        for key, value in tensors.items()
    }
    calibration_tensor = torch.as_tensor(calibration_rows, dtype=torch.long)
    for policy in policy_grid():
        metrics = summarize_policy_on_device(
            policy_device_tensors,
            calibration_tensor,
            policy,
            device,
        )
        calibration_summaries.append(metrics)
    calibration_summaries.sort(
        key=lambda value: (
            float(value["selected_pdms"]),
            -float(value["switch_rate"]),
        ),
        reverse=True,
    )
    selected_policy = dict(calibration_summaries[0]["policy"])
    calibration_metrics = evaluate_policy(
        tensors,
        calibration_rows,
        selected_physical_logs,
        selected_policy,
        device,
        args.seed,
        args.bootstrap_replicates,
    )
    promotion_metrics = evaluate_policy(
        tensors,
        promotion_rows,
        selected_physical_logs,
        selected_policy,
        device,
        args.seed + 1,
        args.bootstrap_replicates,
    )

    residual_config = dict(artifact["residual_config"])
    residual_config.update(selected_policy)
    derived = dict(artifact)
    derived.update(
        {
            "residual_config": residual_config,
            "validation": promotion_metrics,
            "validation_by_source": {selection_source: promotion_metrics},
            "policy_calibration": calibration_metrics,
            "source_artifact": str(args.artifact.resolve()),
            "source_artifact_sha256": _sha256(args.artifact),
            "derived_conservative_policy": True,
            "policy_selection_uses_navtest": False,
            "policy_selection_uses_disjoint_physical_logs": True,
        }
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    output_artifact = args.output_dir / "best_m0_private_residual_scorer.pt"
    _atomic_torch_save(derived, output_artifact)
    calibration_logs = sorted(
        {selected_physical_logs[index] for index in calibration_rows}
    )
    promotion_logs = sorted(
        {selected_physical_logs[index] for index in promotion_rows}
    )
    fold_manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "args": {"epochs": 1},
        "architecture": "M0PrivateResidualConservativePolicy",
        "checkpoint_selection_source": selection_source,
        "source_artifact": str(args.artifact.resolve()),
        "source_artifact_sha256": _sha256(args.artifact),
        "source_lineage": source_lineage,
        "split_manifest": str(args.split_manifest.resolve()),
        "split_manifest_sha256": _sha256(args.split_manifest),
        "calibration_physical_logs": calibration_logs,
        "promotion_physical_logs": promotion_logs,
        "physical_log_overlap": sorted(set(calibration_logs) & set(promotion_logs)),
        "assignment": assignment,
        "policy_grid_size": len(calibration_summaries),
        "policy_selection_uses_navtest": False,
        "future_or_evaluator_input_at_inference": False,
        "official_score_input_at_inference": False,
        "external_model_representation_or_weight_used": False,
    }
    summary = {
        "best_epoch": int(artifact["epoch"]),
        "best_validation_pdms": promotion_metrics["selected_pdms"],
        "artifact": str(output_artifact.resolve()),
        "checkpoint_selection_source": selection_source,
        "selected_policy": selected_policy,
        "calibration": calibration_metrics,
        "promotion": promotion_metrics,
        "top_calibration_policies": calibration_summaries[:20],
        "history": [
            {
                "epoch": int(artifact["epoch"]),
                "validation": promotion_metrics,
                "validation_by_source": {selection_source: promotion_metrics},
            }
        ],
    }
    _atomic_json_dump(fold_manifest, args.output_dir / "fold_manifest.json")
    _atomic_json_dump(summary, args.output_dir / "training_summary.json")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
