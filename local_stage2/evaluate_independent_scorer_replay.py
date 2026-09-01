#!/usr/bin/env python3
"""Evaluate an independent scorer on immutable held-out-log replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch

from local_stage2.train_independent_scorer import (
    ReplaySource,
    _atomic_json_dump,
    _sha256,
    collect_predictions,
    evaluate_predictions,
    load_replay_sources,
)
from local_stage2.train_public_base_residual_scorer import _log_bootstrap_ci
from navsim.agents.EpisodeDrive.score_module.independent_ranker import (
    IndependentProposalRanker,
    IndependentRankerConfig,
    pdms_factor_log_utility,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument(
        "--source",
        action="append",
        nargs=3,
        metavar=("NAME", "FEATURE_ROOT", "LABEL_ROOT"),
        required=True,
    )
    parser.add_argument("--private-observation-root", type=Path, default=None)
    parser.add_argument("--evaluation-split-manifest", type=Path, default=None)
    parser.add_argument("--subgroup-split-manifest", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _evaluate_rows(
    utilities: torch.Tensor,
    coarse_utilities: torch.Tensor,
    factor_logits: torch.Tensor,
    base_scores: torch.Tensor,
    target_factors: torch.Tensor,
    physical_logs: Sequence[str],
    rows: Sequence[int],
    seed: int,
    bootstrap_replicates: int,
) -> Dict[str, object]:
    indices = torch.as_tensor(rows, dtype=torch.long)
    return evaluate_predictions(
        utilities[indices],
        coarse_utilities[indices],
        factor_logits[indices],
        base_scores[indices],
        target_factors[indices],
        [physical_logs[index] for index in rows],
        seed,
        bootstrap_replicates,
    )


def evaluate_base_shortlist_reranking(
    reranker_utility: torch.Tensor,
    base_scores: torch.Tensor,
    target_factors: torch.Tensor,
    physical_logs: Sequence[str],
    *,
    shortlist_size: int,
    seed: int,
    bootstrap_replicates: int,
) -> Dict[str, object]:
    """Rerank Base top-k without exposing its numeric score to the reranker."""

    if reranker_utility.shape != base_scores.shape:
        raise ValueError("reranker and Base score shapes differ")
    if target_factors.shape[:2] != base_scores.shape:
        raise ValueError("target-factor and Base score shapes differ")
    if len(physical_logs) != len(base_scores):
        raise ValueError("physical-log row count differs")
    candidate_count = base_scores.shape[1]
    if not 1 <= shortlist_size <= candidate_count:
        raise ValueError("shortlist_size must be in [1, candidate_count]")

    shortlist = base_scores.topk(shortlist_size, dim=1).indices
    shortlist_utility = reranker_utility.gather(1, shortlist)
    selected_in_shortlist = shortlist_utility.argmax(dim=1, keepdim=True)
    selected = shortlist.gather(1, selected_in_shortlist).squeeze(1)
    base_selected = base_scores.argmax(dim=1)
    target_scores = target_factors[..., -1]
    row = torch.arange(len(target_scores))
    selected_values = target_scores[row, selected]
    base_values = target_scores[row, base_selected]
    best64_values = target_scores.max(dim=1).values
    shortlist_target = target_scores.gather(1, shortlist)
    shortlist_oracle_values = shortlist_target.max(dim=1).values
    delta = (selected_values - base_values).detach().cpu().numpy()
    ci = _log_bootstrap_ci(
        delta,
        physical_logs,
        seed,
        replicates=bootstrap_replicates,
    )
    wins = int((selected_values > base_values + 1e-9).sum())
    losses = int((selected_values < base_values - 1e-9).sum())
    return {
        "scene_count": len(target_scores),
        "physical_log_count": len(set(physical_logs)),
        "shortlist_size": shortlist_size,
        "base_selected_pdms": float(base_values.mean()),
        "base_shortlist_oracle_pdms": float(shortlist_oracle_values.mean()),
        "best_of_64_pdms": float(best64_values.mean()),
        "selected_pdms": float(selected_values.mean()),
        "selected_delta": float(np.mean(delta)),
        "selected_delta_log_bootstrap_95ci": [float(ci[0]), float(ci[1])],
        "selected_regret_to_best64": float(
            (best64_values - selected_values).mean()
        ),
        "selected_regret_to_shortlist_oracle": float(
            (shortlist_oracle_values - selected_values).mean()
        ),
        "shortlist_headroom_over_base": float(
            (shortlist_oracle_values - base_values).mean()
        ),
        "headroom_recovered": float(
            (selected_values - base_values).mean()
            / (shortlist_oracle_values - base_values).mean().clamp_min(1e-12)
        ),
        "switch_rate": float((selected != base_selected).float().mean()),
        "wins": wins,
        "losses": losses,
        "ties": int(len(target_scores) - wins - losses),
        "base_numeric_score_used_by_reranker": False,
        "base_rank_used_for_shortlist": True,
    }


def collect_base_shortlist_metrics(
    coarse_utilities: torch.Tensor,
    factor_logits: torch.Tensor,
    base_scores: torch.Tensor,
    target_factors: torch.Tensor,
    physical_logs: Sequence[str],
    *,
    seed: int,
    bootstrap_replicates: int,
) -> Dict[str, Dict[str, object]]:
    factor_utilities = pdms_factor_log_utility(factor_logits)
    result: Dict[str, Dict[str, object]] = {}
    for mode, utility in (
        ("independent_coarse", coarse_utilities),
        ("independent_factor", factor_utilities),
    ):
        for shortlist_size in (2, 4, 8, 16):
            key = f"{mode}_base_top{shortlist_size}"
            result[key] = evaluate_base_shortlist_reranking(
                utility,
                base_scores,
                target_factors,
                physical_logs,
                shortlist_size=shortlist_size,
                seed=seed + 100 * shortlist_size + len(result),
                bootstrap_replicates=bootstrap_replicates,
            )
    return result


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if not args.artifact.is_file():
        raise FileNotFoundError(args.artifact)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    sources = [
        ReplaySource(name, Path(feature_root), Path(label_root))
        for name, feature_root, label_root in args.source
    ]
    data, source_lineage = load_replay_sources(
        sources,
        private_observation_root=args.private_observation_root,
    )
    artifact = torch.load(args.artifact, map_location="cpu", weights_only=False)
    if artifact.get("architecture") != "IndependentProposalRanker":
        raise RuntimeError("artifact is not an IndependentProposalRanker")
    model = IndependentProposalRanker(
        IndependentRankerConfig(**artifact["model_config"])
    )
    model.load_state_dict(artifact["state_dict"], strict=True)
    device = torch.device(args.device)
    model.to(device).eval()

    if args.evaluation_split_manifest is not None:
        split = json.loads(args.evaluation_split_manifest.read_text())
        evaluation_logs = {str(value) for value in split["validation_physical_logs"]}
        split_lineage = {
            "path": str(args.evaluation_split_manifest.resolve()),
            "sha256": _sha256(args.evaluation_split_manifest),
        }
    else:
        evaluation_logs = {
            str(value)
            for value in artifact["fold_manifest"]["validation_physical_logs"]
        }
        split_lineage = {"source": "artifact_fold_manifest"}
    evaluation_indices = [
        index
        for index, physical_log in enumerate(data.physical_logs)
        if physical_log in evaluation_logs
    ]
    if not evaluation_indices:
        raise RuntimeError("evaluation split has no replay scenes")
    actual_logs = {data.physical_logs[index] for index in evaluation_indices}
    missing_logs = evaluation_logs.difference(actual_logs)
    if missing_logs:
        raise RuntimeError(f"evaluation split is missing {len(missing_logs)} physical logs")

    utilities, coarse, factors, base, targets = collect_predictions(
        model,
        data,
        evaluation_indices,
        device,
        args.batch_size,
    )
    evaluation_physical_logs = [
        data.physical_logs[index] for index in evaluation_indices
    ]
    metrics = evaluate_predictions(
        utilities,
        coarse,
        factors,
        base,
        targets,
        evaluation_physical_logs,
        args.seed,
        args.bootstrap_replicates,
    )
    base_shortlist_metrics = collect_base_shortlist_metrics(
        coarse,
        factors,
        base,
        targets,
        evaluation_physical_logs,
        seed=args.seed + 50_000,
        bootstrap_replicates=args.bootstrap_replicates,
    )
    source_names = [data.source_names[index] for index in evaluation_indices]
    metrics_by_source: Dict[str, object] = {}
    for source_name in sorted(set(source_names)):
        rows = [index for index, value in enumerate(source_names) if value == source_name]
        metrics_by_source[source_name] = _evaluate_rows(
            utilities,
            coarse,
            factors,
            base,
            targets,
            evaluation_physical_logs,
            rows,
            args.seed + 10_000,
            args.bootstrap_replicates,
        )

    subgroup_metrics: Dict[str, object] = {}
    subgroup_lineage = None
    if args.subgroup_split_manifest is not None:
        subgroup = json.loads(args.subgroup_split_manifest.read_text())
        subgroup_lineage = {
            "path": str(args.subgroup_split_manifest.resolve()),
            "sha256": _sha256(args.subgroup_split_manifest),
        }
        for name, key in (
            ("official_train", "train_physical_logs"),
            ("official_validation", "validation_physical_logs"),
        ):
            allowed = {str(value) for value in subgroup[key]}
            rows = [
                index
                for index, value in enumerate(evaluation_physical_logs)
                if value in allowed
            ]
            if rows:
                subgroup_metrics[name] = _evaluate_rows(
                    utilities,
                    coarse,
                    factors,
                    base,
                    targets,
                    evaluation_physical_logs,
                    rows,
                    args.seed + 20_000,
                    args.bootstrap_replicates,
                )

    summary = {
        "schema_version": 1,
        "artifact": str(args.artifact.resolve()),
        "artifact_sha256": _sha256(args.artifact),
        "artifact_epoch": int(artifact["epoch"]),
        "artifact_selection_mode": artifact.get("selection_mode", "direct"),
        "inference_inputs_only": True,
        "future_or_evaluator_input": False,
        "base_score_used_as_model_input": False,
        "scene_count": len(evaluation_indices),
        "physical_log_count": len(actual_logs),
        "split_lineage": split_lineage,
        "subgroup_lineage": subgroup_lineage,
        "source_lineage": source_lineage,
        "metrics": metrics,
        "base_shortlist_metrics": base_shortlist_metrics,
        "metrics_by_source": metrics_by_source,
        "subgroup_metrics": subgroup_metrics,
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    _atomic_json_dump(summary, args.output_dir / "summary.json")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
