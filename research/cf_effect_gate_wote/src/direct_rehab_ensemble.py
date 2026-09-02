"""Validation-only safe ensemble calibration for frozen Direct V3 checkpoints."""

from __future__ import annotations

import argparse
import json
import os
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .direct_rehab_contracts import AccessPolicy
from .direct_rehab_data import DirectDataset, iter_direct_batches, load_direct_dataset
from .direct_rehab_metrics import (
    aggregate_ranking_metrics,
    paired_scene_bootstrap,
    scene_level_metrics,
)
from .models.top_aware_direct_scorer import load_v3_checkpoint


SCHEMA = "direct_v3_safe_ensemble.v1"
FINAL_EVALUATION_SCHEMA = "direct_v3_final_dual_evaluation.v1"
HARD_FACTOR_INDICES = (0, 1, 2, 4)
FACTOR_FLOORS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
MARGINS = (0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite ensemble output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.partial")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _load_dataset(args: argparse.Namespace) -> DirectDataset:
    policy = AccessPolicy.load(args.access_policy)
    tokens = policy.read_token_file(args.tokens, args.phase)
    if args.limit is not None:
        tokens = tokens[: args.limit]
    return load_direct_dataset(
        feature_root=args.feature_root,
        label_root=args.label_root,
        expected_tokens=tokens,
        access_policy=policy,
        phase=args.phase,
        access_log=args.access_log,
        require_selector_reference=True,
    )


def predict_ensemble(
    checkpoints: Sequence[Path],
    dataset: DirectDataset,
    *,
    device: torch.device,
    batch_scenes: int,
    candidate_chunk: int,
) -> tuple[np.ndarray, np.ndarray]:
    if len(checkpoints) != 3:
        raise ValueError("registered Direct ensemble requires exactly three seeds")
    models = []
    identities = []
    for checkpoint in checkpoints:
        model, payload = load_v3_checkpoint(checkpoint, map_location=device)
        if payload["objective"].get("objective") != "O0":
            raise ValueError("registered ensemble only accepts locked O0 checkpoints")
        model.to(device).eval()
        models.append(model)
        identities.append(
            (
                payload["architecture"]["representation"],
                int(payload["seed"]),
                int(payload["metadata"]["train_scenes"]),
            )
        )
    representations = {identity[0] for identity in identities}
    seeds = {identity[1] for identity in identities}
    scene_counts = {identity[2] for identity in identities}
    if len(representations) != 1 or seeds != {0, 1, 2} or scene_counts != {1024}:
        raise ValueError(f"incompatible ensemble checkpoint identities: {identities}")

    score_batches: list[np.ndarray] = []
    factor_batches: list[np.ndarray] = []
    with torch.inference_mode():
        for raw_batch in iter_direct_batches(
            dataset, batch_scenes=batch_scenes, seed=0, epoch=0, shuffle=False
        ):
            batch = raw_batch.to(device)
            model_scores = []
            model_factors = []
            for model in models:
                output = model(
                    batch.trajectory,
                    batch.ego_status,
                    batch.current_bev_tokens,
                    batch.candidate_current_feature,
                    candidate_chunk=candidate_chunk,
                )
                model_scores.append(output["factor_score"])
                model_factors.append(output["factors"])
            score_batches.append(
                torch.stack(model_scores).mean(dim=0).cpu().numpy()
            )
            factor_batches.append(
                torch.stack(model_factors).mean(dim=0).cpu().numpy()
            )
    return np.concatenate(score_batches), np.concatenate(factor_batches)


def _labels(dataset: DirectDataset) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    return (
        np.stack([scene.factor_labels for scene in dataset.scenes]),
        np.stack([scene.score_labels for scene in dataset.scenes]),
        dataset.tokens,
    )


def _base_indices(dataset: DirectDataset) -> np.ndarray:
    if any(scene.wote_selected_index is None for scene in dataset.scenes):
        raise ValueError("frozen WoTE selector reference is missing")
    return np.asarray(
        [int(scene.wote_selected_index) for scene in dataset.scenes], dtype=np.int64
    )


def policy_selection_values(
    ensemble_score: np.ndarray,
    ensemble_factors: np.ndarray,
    base_indices: np.ndarray,
    *,
    factor_floor: float,
    margin: float,
) -> tuple[np.ndarray, Mapping[str, float]]:
    if ensemble_score.shape != ensemble_factors.shape[:2]:
        raise ValueError("ensemble score/factor candidate shapes differ")
    rows = np.arange(len(ensemble_score))
    hard_min = np.min(ensemble_factors[..., HARD_FACTOR_INDICES], axis=-1)
    eligible = hard_min >= float(factor_floor)
    masked = np.where(eligible, ensemble_score, -1.0e9)
    proposed = np.argmax(masked, axis=1)
    no_eligible = ~eligible.any(axis=1)
    proposed[no_eligible] = base_indices[no_eligible]
    predicted_gain = ensemble_score[rows, proposed] - ensemble_score[rows, base_indices]
    accepted = (~no_eligible) & (predicted_gain >= float(margin))
    selected = np.where(accepted, proposed, base_indices)

    values = masked.copy()
    rejected_rows = rows[~accepted]
    if len(rejected_rows):
        # A no-eligible row is filled with -1e9.  Adding one is below float32
        # resolution at that magnitude, so encode an explicit deterministic
        # fallback row instead of relying on a numerically fragile offset.
        values[rejected_rows] = -1.0e9
        values[rejected_rows, base_indices[rejected_rows]] = 1.0e9
    encoded = np.argmax(values, axis=1)
    if not np.array_equal(encoded, selected):
        bad = np.nonzero(encoded != selected)[0][:5]
        raise AssertionError(
            "safe ensemble selection-value encoding changed indices: "
            f"floor={factor_floor} margin={margin} rows={bad.tolist()} "
            f"encoded={encoded[bad].tolist()} expected={selected[bad].tolist()} "
            f"accepted={accepted[bad].tolist()}"
        )
    return values, {
        "override_fraction": float(np.mean(accepted)),
        "eligible_candidate_fraction": float(np.mean(eligible)),
        "no_eligible_scene_fraction": float(np.mean(no_eligible)),
        "mean_predicted_override_gain": float(
            np.mean(predicted_gain[accepted]) if accepted.any() else 0.0
        ),
    }


def evaluate_policy(
    dataset: DirectDataset,
    selection_values: np.ndarray,
    predicted_factors: np.ndarray,
) -> tuple[Mapping[str, float], list[Mapping[str, Any]]]:
    factors, scores, tokens = _labels(dataset)
    rows = scene_level_metrics(
        tokens, selection_values, predicted_factors, factors, scores
    )
    return aggregate_ranking_metrics(selection_values, scores, rows), rows


def _top1_metrics(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, float]:
    """Aggregate only metrics that remain valid for an arbitrated top-1 policy.

    ``policy_selection_values`` uses a deterministic sentinel encoding so that the
    generic scene evaluator selects the frozen-WoTE fallback exactly.  Those
    sentinels are not calibrated scores or a complete candidate ranking, hence
    pairwise/NDCG/overestimation metrics must not be reported for that policy.
    """

    return {
        "selected_score": float(np.mean([row["selected_score"] for row in rows])),
        "top1_regret": float(np.mean([row["regret"] for row in rows])),
        "mean_selected_candidate_rank": float(
            np.mean([row["selected_rank"] for row in rows])
        ),
        "hard_false_safe": float(np.mean([row["hard_false_safe"] for row in rows])),
        "direction_non_compliance": float(
            np.mean([row["direction_non_compliance"] for row in rows])
        ),
        "zero_score_selection": float(
            np.mean([row["zero_score_selection"] for row in rows])
        ),
        "oracle_capture": float(np.mean([row["oracle_capture"] for row in rows])),
    }


def _wote_rows(dataset: DirectDataset) -> list[Mapping[str, Any]]:
    factors, scores, tokens = _labels(dataset)
    indices = _base_indices(dataset)
    values = np.zeros_like(scores, dtype=np.float64)
    values[np.arange(len(values)), indices] = 1.0
    dummy_factors = np.ones_like(factors, dtype=np.float64)
    return scene_level_metrics(tokens, values, dummy_factors, factors, scores)


def calibrate(
    dataset: DirectDataset,
    ensemble_score: np.ndarray,
    ensemble_factors: np.ndarray,
    *,
    calibration_split: str,
) -> Mapping[str, Any]:
    if calibration_split not in {"val", "dev"}:
        raise ValueError("calibration_split must be 'val' or 'dev'")
    base_indices = _base_indices(dataset)
    base_rows = _wote_rows(dataset)
    base_false = float(np.mean([row["hard_false_safe"] for row in base_rows]))
    base_zero = float(np.mean([row["zero_score_selection"] for row in base_rows]))
    tolerance = 1.0 / len(dataset)
    candidates: list[dict[str, Any]] = []
    for floor in FACTOR_FLOORS:
        for margin in MARGINS:
            values, diagnostics = policy_selection_values(
                ensemble_score,
                ensemble_factors,
                base_indices,
                factor_floor=floor,
                margin=margin,
            )
            _, rows = evaluate_policy(dataset, values, ensemble_factors)
            metrics = _top1_metrics(rows)
            eligible = (
                metrics["hard_false_safe"] <= base_false + tolerance + 1.0e-12
                and metrics["zero_score_selection"] <= base_zero + tolerance + 1.0e-12
            )
            candidates.append(
                {
                    "factor_floor": floor,
                    "margin": margin,
                    "eligible": eligible,
                    "metrics": dict(metrics),
                    "diagnostics": dict(diagnostics),
                }
            )
    eligible_rows = [row for row in candidates if row["eligible"]]
    if not eligible_rows:
        raise RuntimeError("no validation-safe Direct ensemble policy exists")
    chosen = max(
        eligible_rows,
        key=lambda row: (
            row["metrics"]["selected_score"],
            -row["metrics"]["hard_false_safe"],
            -row["metrics"]["zero_score_selection"],
            row["margin"],
            row["factor_floor"],
        ),
    )
    return {
        "schema_version": SCHEMA,
        "calibration_split": calibration_split,
        "scene_count": len(dataset),
        "candidate_grid": candidates,
        "wote_constraints": {
            "hard_false_safe": base_false,
            "zero_score_selection": base_zero,
            "tolerance_scenes": 1,
            "tolerance_fraction": tolerance,
        },
        "chosen_policy": {
            "factor_floor": chosen["factor_floor"],
            "margin": chosen["margin"],
            "metrics": chosen["metrics"],
            "diagnostics": chosen["diagnostics"],
        },
    }


def evaluate_locked(
    dataset: DirectDataset,
    ensemble_score: np.ndarray,
    ensemble_factors: np.ndarray,
    policy: Mapping[str, Any],
) -> Mapping[str, Any]:
    if policy.get("schema_version") != SCHEMA:
        raise ValueError("locked ensemble policy schema changed")
    chosen = policy["chosen_policy"]
    values, diagnostics = policy_selection_values(
        ensemble_score,
        ensemble_factors,
        _base_indices(dataset),
        factor_floor=float(chosen["factor_floor"]),
        margin=float(chosen["margin"]),
    )
    _, rows = evaluate_policy(dataset, values, ensemble_factors)
    metrics = _top1_metrics(rows)
    base_rows = _wote_rows(dataset)
    selected = np.asarray([row["selected_score"] for row in rows])
    base = np.asarray([row["selected_score"] for row in base_rows])
    bootstrap = paired_scene_bootstrap(selected, base, resamples=5000, seed=20260827)
    return {
        "schema_version": SCHEMA,
        "scene_count": len(dataset),
        "locked_policy": {
            "factor_floor": float(chosen["factor_floor"]),
            "margin": float(chosen["margin"]),
        },
        "metrics": dict(metrics),
        "diagnostics": dict(diagnostics),
        "wote_reference": {
            "selected_score": float(base.mean()),
            "top1_regret": float(np.mean([row["regret"] for row in base_rows])),
            "hard_false_safe": float(
                np.mean([row["hard_false_safe"] for row in base_rows])
            ),
            "zero_score_selection": float(
                np.mean([row["zero_score_selection"] for row in base_rows])
            ),
        },
        "selected_score_gain_bootstrap": {
            "mean_delta": bootstrap.mean_delta,
            "ci_lower": bootstrap.ci_lower,
            "ci_upper": bootstrap.ci_upper,
            "resamples": bootstrap.resamples,
            "seed": bootstrap.seed,
        },
        "scene_rows": rows,
    }


def evaluate_final_dual(
    dataset: DirectDataset,
    ensemble_score: np.ndarray,
    ensemble_factors: np.ndarray,
    policy: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Evaluate the two pre-locked outputs from one prediction/label read.

    The raw ensemble is the scientific Direct-quality output and retains complete
    ranking metrics.  The safe fallback is a deployment diagnostic and exposes
    only top-1 metrics because its sentinel values are not a candidate ranking.
    No result-dependent choice is made between the outputs.
    """

    if policy.get("schema_version") != SCHEMA:
        raise ValueError("locked ensemble policy schema changed")
    if policy.get("calibration_split") != "dev":
        raise ValueError("final safe policy must be locked on the designated dev split")

    factors, scores, tokens = _labels(dataset)
    raw_rows = scene_level_metrics(
        tokens, ensemble_score, ensemble_factors, factors, scores
    )
    raw_metrics = aggregate_ranking_metrics(ensemble_score, scores, raw_rows)

    chosen = policy["chosen_policy"]
    safe_values, safe_diagnostics = policy_selection_values(
        ensemble_score,
        ensemble_factors,
        _base_indices(dataset),
        factor_floor=float(chosen["factor_floor"]),
        margin=float(chosen["margin"]),
    )
    safe_rows = scene_level_metrics(
        tokens, safe_values, ensemble_factors, factors, scores
    )
    safe_metrics = _top1_metrics(safe_rows)
    base_rows = _wote_rows(dataset)
    base_metrics = _top1_metrics(base_rows)

    def comparison(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, float | int]:
        selected = np.asarray([row["selected_score"] for row in rows])
        base = np.asarray([row["selected_score"] for row in base_rows])
        result = paired_scene_bootstrap(
            selected, base, resamples=5000, seed=20260827
        )
        base_regret = base_metrics["top1_regret"]
        model_regret = float(np.mean([row["regret"] for row in rows]))
        reduction = (
            (base_regret - model_regret) / base_regret
            if base_regret > 1.0e-12
            else float("nan")
        )
        return {
            "mean_selected_score_delta": result.mean_delta,
            "selected_score_delta_ci_lower": result.ci_lower,
            "selected_score_delta_ci_upper": result.ci_upper,
            "regret_reduction_fraction": float(reduction),
            "bootstrap_resamples": result.resamples,
            "bootstrap_seed": result.seed,
        }

    return {
        "schema_version": FINAL_EVALUATION_SCHEMA,
        "scene_count": len(dataset),
        "prediction_passes": 1,
        "label_store_reads": 1,
        "output_contract": {
            "scientific_direct_quality_output": "raw_direct_ensemble",
            "deployment_diagnostic_output": "safe_fallback_primary",
            "result_dependent_output_selection": False,
        },
        "wote_reference": base_metrics,
        "raw_direct_ensemble": {
            "metrics": dict(raw_metrics),
            "versus_wote": comparison(raw_rows),
            "scene_rows": raw_rows,
        },
        "safe_fallback_primary": {
            "locked_policy": {
                "calibration_split": str(policy["calibration_split"]),
                "factor_floor": float(chosen["factor_floor"]),
                "margin": float(chosen["margin"]),
            },
            "metrics": dict(safe_metrics),
            "ranking_metrics_status": "NOT_APPLICABLE_SENTINEL_TOP1_POLICY",
            "diagnostics": dict(safe_diagnostics),
            "versus_wote": comparison(safe_rows),
            "scene_rows": safe_rows,
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("calibrate", "evaluate", "evaluate-final")
    )
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--label-root", type=Path, required=True)
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--checkpoints", type=Path, nargs=3, required=True)
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--calibration-split-name", choices=("val", "dev"))
    parser.add_argument("--phase", default="development")
    parser.add_argument("--batch-scenes", type=int, default=4)
    parser.add_argument("--candidate-chunk", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--access-policy", type=Path, required=True)
    parser.add_argument("--access-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    dataset = _load_dataset(args)
    score, factors = predict_ensemble(
        args.checkpoints,
        dataset,
        device=torch.device(args.device),
        batch_scenes=args.batch_scenes,
        candidate_chunk=args.candidate_chunk,
    )
    if args.command == "calibrate":
        if args.calibration_split_name is None:
            raise ValueError("calibrate requires --calibration-split-name")
        result = calibrate(
            dataset,
            score,
            factors,
            calibration_split=args.calibration_split_name,
        )
    else:
        if args.policy is None:
            raise ValueError(f"{args.command} requires --policy")
        policy = json.loads(args.policy.read_text(encoding="utf-8"))
        result = (
            evaluate_locked(dataset, score, factors, policy)
            if args.command == "evaluate"
            else evaluate_final_dual(dataset, score, factors, policy)
        )
    _atomic_json(args.output, result)
    display = dict(result)
    display.pop("candidate_grid", None)
    display.pop("scene_rows", None)
    for output_name in ("raw_direct_ensemble", "safe_fallback_primary"):
        if output_name in display:
            display[output_name] = dict(display[output_name])
            display[output_name].pop("scene_rows", None)
    print(json.dumps(display, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
