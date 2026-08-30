#!/usr/bin/env python3
"""Cross-fitted predicted-consequence planning-utility probe.

This probe answers a stricter question than the oracle probe:

    current scene + candidate trajectory -> predicted dynamic consequence
        -> candidate ranking

Static map/route consequence channels are carried as exact online geometry.
Dynamic logged-future channels are predicted. Outer validation logs are never
used to train either the consequence predictor or the downstream scorer, and
outer-training consequence features are out-of-fold predictions.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, ndcg_score, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .common import add_common_arguments, write_dataframe, write_json, write_text
from .mlp_effect_predictor import fit_fixed_epochs, tune_epochs
from .run_oracle_probe import (
    EFFECT_CHANNELS,
    TARGETS,
    TUBE_STATISTICS,
    binary_model_metrics,
    build_dataset,
    fit_regressor,
    ranking_metrics,
    regression_target_metrics,
    split_is_validation,
)


EXACT_ONLINE_CHANNELS = ("drivable_area_sdf", "lane_sdf", "route_sdf")
PREDICTED_DYNAMIC_CHANNELS = (
    "candidate_relative_dynamic_occupancy",
    "relative_longitudinal_velocity",
    "relative_lateral_velocity",
    "dynamic_clearance",
    "dynamic_collision_field",
)
PRIMARY_EFFECT_MODEL_CANDIDATES = ("mlp_delta", "mlp_delta_strong")


def partition_environment_features(
    feature_names: Iterable[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Return exact-static and predicted-dynamic feature indices."""

    names = list(feature_names)
    exact = np.asarray(
        [
            index
            for index, name in enumerate(names)
            if any(f"_{channel}_" in name for channel in EXACT_ONLINE_CHANNELS)
        ],
        dtype=np.int64,
    )
    dynamic = np.asarray(
        [
            index
            for index, name in enumerate(names)
            if any(f"_{channel}_" in name for channel in PREDICTED_DYNAMIC_CHANNELS)
        ],
        dtype=np.int64,
    )
    covered = set(exact.tolist()) | set(dynamic.tolist())
    if len(covered) != len(names) or set(exact.tolist()) & set(dynamic.tolist()):
        missing = [name for index, name in enumerate(names) if index not in covered]
        raise ValueError(f"environment partition is incomplete: {missing}")
    return exact, dynamic


def grouped_crossfit_splits(
    groups: np.ndarray, n_splits: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create deterministic group-disjoint folds over the provided rows."""

    values = np.asarray(groups)
    unique = np.unique(values)
    folds = min(int(n_splits), len(unique))
    if folds < 2:
        raise ValueError("cross-fitting requires at least two distinct logs")
    splitter = GroupKFold(n_splits=folds)
    result = []
    dummy = np.zeros((len(values), 1), dtype=np.float32)
    for train, validation in splitter.split(dummy, groups=values):
        if set(values[train]) & set(values[validation]):
            raise RuntimeError("cross-fit log overlap")
        result.append((train, validation))
    return result


@dataclass
class EffectModel:
    estimator: Any
    target_mean: np.ndarray
    target_scale: np.ndarray
    target_min: np.ndarray
    target_max: np.ndarray

    def predict(self, features: np.ndarray) -> np.ndarray:
        normalized = np.asarray(self.estimator.predict(features), dtype=np.float64)
        prediction = normalized * self.target_scale + self.target_mean
        # Parallel tree reduction can differ below float32 significance. Those
        # differences are physically meaningless but can perturb downstream
        # histogram tie-breaking, so enforce the target schema's audit
        # precision before caching or scoring.
        clipped = np.clip(prediction, self.target_min, self.target_max)
        return np.round(clipped, decimals=6).astype(np.float32)


def fit_effect_model(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    model_kind: str,
    seed: int,
    args: argparse.Namespace,
) -> EffectModel:
    target_mean = np.mean(targets, axis=0, dtype=np.float64)
    target_scale = np.std(targets, axis=0, dtype=np.float64)
    target_scale[target_scale < 1e-6] = 1.0
    normalized = (targets - target_mean) / target_scale
    if model_kind == "ridge":
        estimator = make_pipeline(
            StandardScaler(),
            Ridge(alpha=float(args.ridge_alpha)),
        )
    elif model_kind == "extra_trees":
        estimator = ExtraTreesRegressor(
            n_estimators=int(args.extra_trees_estimators),
            max_depth=int(args.extra_trees_max_depth),
            min_samples_leaf=int(args.extra_trees_min_samples_leaf),
            max_features=float(args.extra_trees_max_features),
            n_jobs=int(args.jobs),
            random_state=int(seed),
        )
    else:
        raise ValueError(f"unknown effect model: {model_kind}")
    estimator.fit(features, normalized)
    return EffectModel(
        estimator=estimator,
        target_mean=target_mean,
        target_scale=target_scale,
        target_min=np.min(targets, axis=0),
        target_max=np.max(targets, axis=0),
    )


def cross_fitted_effect_predictions(
    features: np.ndarray,
    targets: np.ndarray,
    groups: np.ndarray,
    train_mask: np.ndarray,
    validation_mask: np.ndarray,
    *,
    model_kind: str,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Produce OOF outer-train and held-out outer-validation predictions."""

    train_indices = np.flatnonzero(train_mask)
    validation_indices = np.flatnonzero(validation_mask)
    local_groups = groups[train_indices]
    oof = np.full((len(train_indices), targets.shape[1]), np.nan, dtype=np.float32)
    folds: list[dict[str, Any]] = []
    for fold_index, (local_train, local_validation) in enumerate(
        grouped_crossfit_splits(local_groups, args.crossfit_folds)
    ):
        model = fit_effect_model(
            features[train_indices[local_train]],
            targets[train_indices[local_train]],
            model_kind=model_kind,
            seed=args.seed + fold_index,
            args=args,
        )
        oof[local_validation] = model.predict(
            features[train_indices[local_validation]]
        )
        fold_train_logs = set(local_groups[local_train].tolist())
        fold_validation_logs = set(local_groups[local_validation].tolist())
        folds.append(
            {
                "fold": fold_index,
                "train_rows": int(len(local_train)),
                "validation_rows": int(len(local_validation)),
                "train_logs": len(fold_train_logs),
                "validation_logs": len(fold_validation_logs),
                "log_overlap": sorted(fold_train_logs & fold_validation_logs),
            }
        )
        del model
        gc.collect()
    if not np.isfinite(oof).all():
        raise RuntimeError("cross-fit predictions are incomplete or non-finite")
    final_model = fit_effect_model(
        features[train_indices],
        targets[train_indices],
        model_kind=model_kind,
        seed=args.seed,
        args=args,
    )
    held_out = final_model.predict(features[validation_indices])
    if not np.isfinite(held_out).all():
        raise RuntimeError("held-out effect predictions are non-finite")
    return oof, held_out, folds


def inner_monitor_split(
    indices: np.ndarray,
    log_names: np.ndarray,
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Split allowed training logs for epoch selection only."""

    unique = sorted(set(log_names[indices].tolist()))
    ordered = sorted(
        unique,
        key=lambda value: hashlib.sha256(
            f"{seed}:{value}".encode()
        ).digest(),
    )
    monitor_count = max(1, int(round(0.15 * len(ordered))))
    monitor_logs = set(ordered[:monitor_count])
    monitor = indices[
        np.asarray(
            [value in monitor_logs for value in log_names[indices]],
            dtype=bool,
        )
    ]
    train = indices[
        np.asarray(
            [value not in monitor_logs for value in log_names[indices]],
            dtype=bool,
        )
    ]
    if not len(train) or not len(monitor):
        raise RuntimeError("inner convergence split is empty")
    return train, monitor


def cross_fitted_mlp_predictions(
    features: np.ndarray,
    targets: np.ndarray,
    log_names: np.ndarray,
    scene_tokens: np.ndarray,
    train_mask: np.ndarray,
    validation_mask: np.ndarray,
    *,
    model_kind: str,
    args: argparse.Namespace,
) -> tuple[
    np.ndarray,
    np.ndarray,
    list[dict[str, Any]],
    dict[str, Any],
]:
    """Nested log-safe convergence selection plus OOF prediction."""

    delta_weight = {
        "mlp_raw": 0.0,
        "mlp_delta": float(args.mlp_delta_weight),
        "mlp_delta_strong": float(args.mlp_delta_weight_strong),
    }[model_kind]
    hidden_dims = tuple(
        int(value) for value in args.mlp_hidden_dims.split(",") if value
    )
    if not hidden_dims:
        raise ValueError("MLP hidden dimensions are empty")
    device = torch.device(args.mlp_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("MLP requested CUDA but CUDA is unavailable")
    train_indices = np.flatnonzero(train_mask)
    validation_indices = np.flatnonzero(validation_mask)
    local_groups = log_names[train_indices]
    oof = np.full((len(train_indices), targets.shape[1]), np.nan, dtype=np.float32)
    fold_records: list[dict[str, Any]] = []
    for fold_index, (local_train, local_validation) in enumerate(
        grouped_crossfit_splits(local_groups, args.crossfit_folds)
    ):
        global_train = train_indices[local_train]
        global_validation = train_indices[local_validation]
        tune_train, tune_monitor = inner_monitor_split(
            global_train, log_names, seed=args.seed + fold_index
        )
        convergence = tune_epochs(
            features,
            targets,
            scene_tokens,
            tune_train,
            tune_monitor,
            hidden_dims=hidden_dims,
            delta_weight=delta_weight,
            learning_rate=args.mlp_learning_rate,
            weight_decay=args.mlp_weight_decay,
            max_epochs=args.mlp_max_epochs,
            patience=args.mlp_patience,
            seed=args.seed + fold_index,
            device=device,
        )
        tune_train_logs = set(log_names[tune_train].tolist())
        tune_monitor_logs = set(log_names[tune_monitor].tolist())
        convergence["inner_split"] = {
            "train_rows": int(len(tune_train)),
            "monitor_rows": int(len(tune_monitor)),
            "train_logs": len(tune_train_logs),
            "monitor_logs": len(tune_monitor_logs),
            "log_overlap": sorted(tune_train_logs & tune_monitor_logs),
        }
        predictor, refit_history = fit_fixed_epochs(
            features,
            targets,
            scene_tokens,
            global_train,
            hidden_dims=hidden_dims,
            delta_weight=delta_weight,
            learning_rate=args.mlp_learning_rate,
            weight_decay=args.mlp_weight_decay,
            epochs=convergence["selected_epochs"],
            seed=args.seed + fold_index,
            device=device,
        )
        oof[local_validation] = predictor.predict(
            features[global_validation]
        )
        train_logs = set(log_names[global_train].tolist())
        held_out_logs = set(log_names[global_validation].tolist())
        convergence["refit_final_train_loss"] = refit_history[-1][
            "train_total"
        ]
        fold_records.append(
            {
                "fold": fold_index,
                "train_rows": int(len(global_train)),
                "validation_rows": int(len(global_validation)),
                "train_logs": len(train_logs),
                "validation_logs": len(held_out_logs),
                "log_overlap": sorted(train_logs & held_out_logs),
                "convergence": convergence,
            }
        )
        del predictor
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if not np.isfinite(oof).all():
        raise RuntimeError("MLP OOF predictions are incomplete")

    tune_train, tune_monitor = inner_monitor_split(
        train_indices, log_names, seed=args.seed + 1000
    )
    final_convergence = tune_epochs(
        features,
        targets,
        scene_tokens,
        tune_train,
        tune_monitor,
        hidden_dims=hidden_dims,
        delta_weight=delta_weight,
        learning_rate=args.mlp_learning_rate,
        weight_decay=args.mlp_weight_decay,
        max_epochs=args.mlp_max_epochs,
        patience=args.mlp_patience,
        seed=args.seed,
        device=device,
    )
    final_tune_train_logs = set(log_names[tune_train].tolist())
    final_tune_monitor_logs = set(log_names[tune_monitor].tolist())
    final_convergence["inner_split"] = {
        "train_rows": int(len(tune_train)),
        "monitor_rows": int(len(tune_monitor)),
        "train_logs": len(final_tune_train_logs),
        "monitor_logs": len(final_tune_monitor_logs),
        "log_overlap": sorted(
            final_tune_train_logs & final_tune_monitor_logs
        ),
    }
    final_predictor, final_history = fit_fixed_epochs(
        features,
        targets,
        scene_tokens,
        train_indices,
        hidden_dims=hidden_dims,
        delta_weight=delta_weight,
        learning_rate=args.mlp_learning_rate,
        weight_decay=args.mlp_weight_decay,
        epochs=final_convergence["selected_epochs"],
        seed=args.seed,
        device=device,
    )
    held_out = final_predictor.predict(features[validation_indices])
    final_convergence["refit_final_train_loss"] = final_history[-1][
        "train_total"
    ]
    improvements = [
        float(record["convergence"]["relative_monitor_improvement"])
        for record in fold_records
    ]
    best_epochs = [
        int(record["convergence"]["best_epoch"]) for record in fold_records
    ]
    convergence_audit = {
        "model_kind": model_kind,
        "hidden_dims": list(hidden_dims),
        "parameter_count": int(
            sum(
                parameter.numel()
                for parameter in final_predictor.model.parameters()
            )
        ),
        "delta_weight": delta_weight,
        "folds_improved_over_initial": int(
            sum(value >= 0.05 for value in improvements)
        ),
        "fold_count": len(fold_records),
        "median_relative_monitor_improvement": float(
            np.median(improvements)
        ),
        "median_best_epoch": float(np.median(best_epochs)),
        "final_model_convergence": final_convergence,
        "optimization_converged": bool(
            sum(value >= 0.05 for value in improvements)
            >= max(1, len(fold_records) - 1)
            and np.median(best_epochs) > 0
        ),
        "criterion": (
            "at least 4/5 folds improve inner-log monitor loss by >=5% "
            "and median best epoch > 0"
        ),
    }
    return oof, held_out, fold_records, convergence_audit


def safe_spearman(left: np.ndarray, right: np.ndarray) -> float | None:
    if (
        len(left) == 0
        or np.allclose(left, left.flat[0], rtol=0.0, atol=1e-8)
        or np.allclose(right, right.flat[0], rtol=0.0, atol=1e-8)
    ):
        return None
    value = spearmanr(left, right).statistic
    return float(value) if np.isfinite(value) else None


def nested_numeric_difference(left: Any, right: Any) -> tuple[bool, float]:
    """Compare nested metric structures and return max numeric difference."""

    if isinstance(left, dict) and isinstance(right, dict):
        if set(left) != set(right):
            return False, float("inf")
        comparisons = [
            nested_numeric_difference(left[key], right[key])
            for key in left
        ]
    elif isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        if len(left) != len(right):
            return False, float("inf")
        comparisons = [
            nested_numeric_difference(a, b) for a, b in zip(left, right)
        ]
    elif (
        isinstance(left, (int, float))
        and isinstance(right, (int, float))
        and not isinstance(left, bool)
        and not isinstance(right, bool)
    ):
        return True, abs(float(left) - float(right))
    else:
        return left == right, 0.0 if left == right else float("inf")
    if not comparisons:
        return True, 0.0
    return (
        all(item[0] for item in comparisons),
        max(item[1] for item in comparisons),
    )


def verify_determinism(
    reference_dir: Path,
    result: dict[str, Any],
    prediction_payload: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Compare a repeat at physical/schema precision, not JSON bit identity."""

    reference_json = reference_dir / "predicted_consequence_probe_results.json"
    reference_npz = reference_dir / (
        "predicted_consequence_validation_predictions.npz"
    )
    previous = json.loads(reference_json.read_text())
    maximum_difference: dict[str, float] = {}
    with np.load(reference_npz) as previous_arrays:
        for key, value in prediction_payload.items():
            if key not in previous_arrays.files or not np.issubdtype(
                np.asarray(value).dtype, np.number
            ):
                continue
            maximum_difference[key] = float(
                np.max(
                    np.abs(
                        np.asarray(value, dtype=np.float64)
                        - np.asarray(previous_arrays[key], dtype=np.float64)
                    )
                )
            )
    probe_structure, probe_difference = nested_numeric_difference(
        previous["probes"], result["probes"]
    )
    comparison_structure, comparison_difference = nested_numeric_difference(
        previous["primary_comparison"], result["primary_comparison"]
    )
    prediction_tolerance = 1.1e-6
    metric_tolerance = 1e-6
    comparison_tolerance = 1e-12
    prediction_within = all(
        value <= prediction_tolerance for value in maximum_difference.values()
    )
    probe_within = probe_structure and probe_difference <= metric_tolerance
    comparison_within = (
        comparison_structure
        and comparison_difference <= comparison_tolerance
    )
    return {
        "reference": str(reference_dir),
        "planning_probe_metrics_exact": previous["probes"] == result["probes"],
        "planning_probe_metrics_max_abs_difference": probe_difference,
        "planning_probe_metrics_within_tolerance": probe_within,
        "primary_comparison_exact": (
            previous["primary_comparison"] == result["primary_comparison"]
        ),
        "primary_comparison_max_abs_difference": comparison_difference,
        "primary_comparison_within_tolerance": comparison_within,
        "prediction_max_abs_difference": maximum_difference,
        "prediction_within_schema_precision": prediction_within,
        "pass": bool(prediction_within and probe_within and comparison_within),
        "schema_rounding_decimals": 6,
        "tolerances": {
            "prediction_abs": prediction_tolerance,
            "planning_metric_abs": metric_tolerance,
            "primary_comparison_abs": comparison_tolerance,
        },
    }


def center_within_scene(
    values: np.ndarray, scene_tokens: np.ndarray
) -> np.ndarray:
    """Remove scene-shared signal and retain candidate-specific deltas."""

    centered = np.empty_like(values, dtype=np.float32)
    for token in np.unique(scene_tokens):
        mask = scene_tokens == token
        centered[mask] = values[mask] - np.mean(values[mask], axis=0)
    return centered


def aggregate_effect_metrics(
    truth: np.ndarray, prediction: np.ndarray, train_scale: np.ndarray
) -> dict[str, Any]:
    active = np.asarray(train_scale) >= 1e-4
    if not np.any(active):
        active = np.ones_like(train_scale, dtype=bool)
    scale = np.maximum(train_scale[active], 1e-4)
    normalized_error = (prediction[:, active] - truth[:, active]) / scale
    valid_r2 = [
        r2_score(truth[:, index], prediction[:, index])
        for index in range(truth.shape[1])
        if np.std(truth[:, index]) >= 1e-8
    ]
    return {
        "support_rows": int(len(truth)),
        "feature_count": int(truth.shape[1]),
        "normalized_feature_count": int(np.sum(active)),
        "mae_raw": float(mean_absolute_error(truth, prediction)),
        "rmse_raw": float(mean_squared_error(truth, prediction, squared=False)),
        "normalized_mae": float(np.mean(np.abs(normalized_error))),
        "normalized_rmse": float(np.sqrt(np.mean(normalized_error**2))),
        "mean_nonconstant_feature_r2": float(np.mean(valid_r2))
        if valid_r2
        else None,
        "flat_normalized_spearman": safe_spearman(
            (truth[:, active] / scale).ravel(),
            (prediction[:, active] / scale).ravel(),
        ),
    }


def parse_effect_feature_name(name: str) -> tuple[float | None, str, str]:
    horizon_match = re.search(r"_h([0-9.]+)s_", name)
    horizon = float(horizon_match.group(1)) if horizon_match else None
    channel = next(
        (value for value in EFFECT_CHANNELS if f"_{value}_" in name),
        "unknown",
    )
    statistic = next(
        (value for value in TUBE_STATISTICS if name.endswith(f"_{value}")),
        "unknown",
    )
    return horizon, channel, statistic


def feature_effect_metrics(
    *,
    model_kind: str,
    split: str,
    truth: np.ndarray,
    prediction: np.ndarray,
    feature_names: list[str],
    train_scale: np.ndarray,
) -> list[dict[str, Any]]:
    rows = []
    for index, name in enumerate(feature_names):
        horizon, channel, statistic = parse_effect_feature_name(name)
        scale = max(float(train_scale[index]), 1e-6)
        rmse = float(
            mean_squared_error(
                truth[:, index], prediction[:, index], squared=False
            )
        )
        nonconstant = float(np.std(truth[:, index])) >= 1e-8
        rows.append(
            {
                "model": model_kind,
                "split": split,
                "feature": name,
                "horizon_s": horizon,
                "channel": channel,
                "statistic": statistic,
                "support": int(len(truth)),
                "truth_std": float(np.std(truth[:, index])),
                "mae": float(
                    mean_absolute_error(truth[:, index], prediction[:, index])
                ),
                "rmse": rmse,
                "normalized_rmse": rmse / scale,
                "r2": float(
                    r2_score(truth[:, index], prediction[:, index])
                )
                if nonconstant
                else None,
                "spearman": safe_spearman(
                    truth[:, index], prediction[:, index]
                ),
            }
        )
    return rows


def consequence_diversity_recovery(
    truth: np.ndarray,
    prediction: np.ndarray,
    scene_tokens: np.ndarray,
    train_scale: np.ndarray,
) -> dict[str, Any]:
    scale = np.maximum(train_scale, 1e-6)
    true_distances = []
    predicted_distances = []
    scene_correlations = []
    true_variance = []
    predicted_variance = []
    for token in np.unique(scene_tokens):
        mask = scene_tokens == token
        true_scene = truth[mask] / scale
        predicted_scene = prediction[mask] / scale
        true_pairwise = pdist(true_scene)
        predicted_pairwise = pdist(predicted_scene)
        if len(true_pairwise):
            true_distances.append(true_pairwise)
            predicted_distances.append(predicted_pairwise)
            correlation = safe_spearman(true_pairwise, predicted_pairwise)
            if correlation is not None:
                scene_correlations.append(correlation)
        true_variance.append(float(np.mean(np.var(true_scene, axis=0))))
        predicted_variance.append(float(np.mean(np.var(predicted_scene, axis=0))))
    all_truth = np.concatenate(true_distances) if true_distances else np.asarray([])
    all_prediction = (
        np.concatenate(predicted_distances) if predicted_distances else np.asarray([])
    )
    denominator = float(np.mean(true_variance)) if true_variance else 0.0
    return {
        "pairwise_distance_spearman_global": safe_spearman(
            all_truth, all_prediction
        )
        if len(all_truth)
        else None,
        "pairwise_distance_spearman_per_scene_mean": float(
            np.mean(scene_correlations)
        )
        if scene_correlations
        else None,
        "candidate_variance_truth_mean": denominator,
        "candidate_variance_prediction_mean": float(
            np.mean(predicted_variance)
        )
        if predicted_variance
        else None,
        "candidate_variance_recovery_ratio": (
            float(np.mean(predicted_variance)) / denominator
            if predicted_variance and denominator > 1e-12
            else None
        ),
    }


def evaluate_planning_probe(
    features: np.ndarray,
    dataset: dict[str, Any],
    train_mask: np.ndarray,
    validation_mask: np.ndarray,
    *,
    seed: int,
) -> tuple[dict[str, Any], np.ndarray]:
    y_score = dataset["targets"]["aggregate_score"]
    valid_score = np.isfinite(y_score)
    score_train = train_mask & valid_score
    score_validation = validation_mask & valid_score
    model = fit_regressor(
        features[score_train], y_score[score_train], seed
    )
    score_prediction = model.predict(features[score_validation])
    result: dict[str, Any] = {
        "feature_count": int(features.shape[1]),
        "model_family": "sklearn HistGradientBoosting, fixed configuration",
        "aggregate_score_and_ranking": ranking_metrics(
            y_score[score_validation],
            score_prediction,
            dataset["scene_tokens"][score_validation],
        ),
    }
    factor_results: dict[str, Any] = {}
    for target_name, (target_type, _, _) in TARGETS.items():
        if target_name == "aggregate_score":
            continue
        y = dataset["targets"][target_name]
        train = train_mask & np.isfinite(y)
        validation = validation_mask & np.isfinite(y)
        if target_type == "regression":
            factor_model = fit_regressor(
                features[train], y[train], seed
            )
            prediction = factor_model.predict(features[validation])
            factor_results[target_name] = regression_target_metrics(
                y[validation], prediction
            )
            factor_results[target_name]["support"] = int(np.sum(validation))
        else:
            factor_results[target_name] = binary_model_metrics(
                features[train],
                y[train],
                features[validation],
                y[validation],
                seed,
            )
    result["factor_prediction"] = factor_results
    return result, np.asarray(score_prediction, dtype=np.float64)


def scene_ranking_outcomes(
    truth: np.ndarray, prediction: np.ndarray, scene_tokens: np.ndarray
) -> dict[str, np.ndarray]:
    values = {
        "pair_correct": [],
        "pair_count": [],
        "top1_accuracy": [],
        "regret": [],
        "ndcg": [],
        "spearman": [],
    }
    for token in np.unique(scene_tokens):
        mask = scene_tokens == token
        y = truth[mask]
        p = prediction[mask]
        difference = y[:, None] - y[None, :]
        predicted_difference = p[:, None] - p[None, :]
        upper = np.triu_indices(len(y), k=1)
        valid = np.abs(difference[upper]) > 1e-8
        values["pair_correct"].append(
            np.sum(
                np.sign(difference[upper][valid])
                == np.sign(predicted_difference[upper][valid])
            )
        )
        values["pair_count"].append(np.sum(valid))
        selected = int(np.argmax(p))
        best = float(np.max(y))
        values["top1_accuracy"].append(
            float(abs(float(y[selected]) - best) <= 1e-8)
        )
        values["regret"].append(best - float(y[selected]))
        values["ndcg"].append(float(ndcg_score(y[None], p[None])))
        correlation = safe_spearman(y, p)
        values["spearman"].append(
            float(correlation) if correlation is not None else np.nan
        )
    return {key: np.asarray(value, dtype=np.float64) for key, value in values.items()}


def aggregate_scene_outcomes(
    outcomes: dict[str, np.ndarray], indices: np.ndarray
) -> dict[str, float]:
    pair_count = float(np.sum(outcomes["pair_count"][indices]))
    return {
        "pairwise_ranking_accuracy": float(
            np.sum(outcomes["pair_correct"][indices]) / pair_count
        )
        if pair_count
        else float("nan"),
        "top1_accuracy": float(np.mean(outcomes["top1_accuracy"][indices])),
        "top1_score_regret_mean": float(np.mean(outcomes["regret"][indices])),
        "ndcg_mean": float(np.mean(outcomes["ndcg"][indices])),
        "spearman_per_scene_mean": float(
            np.nanmean(outcomes["spearman"][indices])
        ),
    }


def paired_bootstrap_comparison(
    truth: np.ndarray,
    baseline_prediction: np.ndarray,
    candidate_prediction: np.ndarray,
    scene_tokens: np.ndarray,
    *,
    seed: int,
    samples: int,
) -> dict[str, Any]:
    baseline = scene_ranking_outcomes(truth, baseline_prediction, scene_tokens)
    candidate = scene_ranking_outcomes(truth, candidate_prediction, scene_tokens)
    scene_count = len(baseline["regret"])
    observed_indices = np.arange(scene_count)
    observed_baseline = aggregate_scene_outcomes(baseline, observed_indices)
    observed_candidate = aggregate_scene_outcomes(candidate, observed_indices)
    generator = np.random.default_rng(seed)
    deltas: dict[str, list[float]] = {
        key: [] for key in observed_baseline
    }
    for _ in range(samples):
        indices = generator.integers(0, scene_count, size=scene_count)
        left = aggregate_scene_outcomes(baseline, indices)
        right = aggregate_scene_outcomes(candidate, indices)
        for key in deltas:
            deltas[key].append(right[key] - left[key])
    result: dict[str, Any] = {}
    for key, values in deltas.items():
        array = np.asarray(values)
        result[key] = {
            "baseline": observed_baseline[key],
            "candidate": observed_candidate[key],
            "delta_candidate_minus_baseline": (
                observed_candidate[key] - observed_baseline[key]
            ),
            "delta_ci95": [
                float(np.quantile(array, 0.025)),
                float(np.quantile(array, 0.975)),
            ],
        }
    return result


def planning_gain_judgement(comparison: dict[str, Any]) -> str:
    pairwise = comparison["pairwise_ranking_accuracy"]
    regret = comparison["top1_score_regret_mean"]
    pair_delta = float(pairwise["delta_candidate_minus_baseline"])
    regret_delta = float(regret["delta_candidate_minus_baseline"])
    if pairwise["delta_ci95"][0] > 0 and regret["delta_ci95"][1] < 0:
        return "PASS"
    if pair_delta > 0 and regret_delta < 0:
        return "CONDITIONAL_PASS"
    if pairwise["delta_ci95"][1] <= 0 and regret["delta_ci95"][0] >= 0:
        return "FAIL"
    return "INCONCLUSIVE"


def overfit_capacity_sanity(
    features: np.ndarray,
    targets: np.ndarray,
    log_names: np.ndarray,
    scene_tokens: np.ndarray,
    train_mask: np.ndarray,
    *,
    model_kind: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Check that the predictor can memorize a small legal-train subset.

    This is an implementation/capacity diagnostic only. It is deliberately
    evaluated in-sample and must never be reported as generalization evidence.
    """

    selected_tokens: list[str] = []
    train_logs = sorted(set(log_names[train_mask].tolist()))
    for log_name in train_logs:
        tokens = sorted(
            set(scene_tokens[train_mask & (log_names == log_name)].tolist())
        )
        if tokens:
            selected_tokens.append(tokens[0])
        if len(selected_tokens) >= int(args.mlp_overfit_scenes):
            break
    subset_mask = train_mask & np.isin(scene_tokens, selected_tokens)
    subset_indices = np.flatnonzero(subset_mask)
    if len(selected_tokens) < 2 or not len(subset_indices):
        return {
            "pass": False,
            "reason": "insufficient distinct training logs for overfit sanity",
            "scene_count": len(selected_tokens),
        }
    delta_weight = {
        "mlp_raw": 0.0,
        "mlp_delta": float(args.mlp_delta_weight),
        "mlp_delta_strong": float(args.mlp_delta_weight_strong),
    }[model_kind]
    hidden_dims = tuple(
        int(value) for value in args.mlp_hidden_dims.split(",") if value
    )
    device = torch.device(args.mlp_device)
    predictor, history = fit_fixed_epochs(
        features,
        targets,
        scene_tokens,
        subset_indices,
        hidden_dims=hidden_dims,
        delta_weight=delta_weight,
        learning_rate=args.mlp_learning_rate,
        weight_decay=args.mlp_weight_decay,
        epochs=args.mlp_overfit_epochs,
        seed=args.seed + 2000,
        device=device,
    )
    truth = targets[subset_indices]
    prediction = predictor.predict(features[subset_indices])
    tokens = scene_tokens[subset_indices]
    raw_scale = np.std(truth, axis=0)
    centered_truth = center_within_scene(truth, tokens)
    centered_prediction = center_within_scene(prediction, tokens)
    delta_scale = np.std(centered_truth, axis=0)
    delta_metrics = aggregate_effect_metrics(
        centered_truth, centered_prediction, delta_scale
    )
    diversity = consequence_diversity_recovery(
        truth, prediction, tokens, raw_scale
    )
    delta_spearman = delta_metrics.get("flat_normalized_spearman") or 0.0
    pairwise_spearman = (
        diversity.get("pairwise_distance_spearman_global") or 0.0
    )
    variance_recovery = (
        diversity.get("candidate_variance_recovery_ratio") or 0.0
    )
    result = {
        "pass": bool(
            delta_spearman >= 0.80
            and pairwise_spearman >= 0.80
            and 0.50 <= variance_recovery <= 1.50
        ),
        "interpretation": (
            "in-sample implementation/capacity sanity only; not evidence of "
            "held-out prediction quality"
        ),
        "model_kind": model_kind,
        "scene_count": len(selected_tokens),
        "log_count": len(set(log_names[subset_indices].tolist())),
        "candidate_count": int(len(subset_indices)),
        "epochs": int(args.mlp_overfit_epochs),
        "initial_train_loss": float(history[0]["train_total"]),
        "final_train_loss": float(history[-1]["train_total"]),
        "relative_train_loss_reduction": float(
            1.0
            - float(history[-1]["train_total"])
            / max(float(history[0]["train_total"]), 1e-12)
        ),
        "candidate_delta_metrics": delta_metrics,
        "diversity_recovery": diversity,
        "criterion": {
            "candidate_delta_spearman_min": 0.80,
            "pairwise_distance_spearman_min": 0.80,
            "candidate_variance_recovery_range": [0.50, 1.50],
        },
        "scene_tokens": selected_tokens,
    }
    del predictor
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def convergence_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model, records in result.get("crossfit", {}).get(
        "records_by_model", {}
    ).items():
        for record in records:
            convergence = record.get("convergence")
            if not convergence:
                continue
            for point in convergence["history"]:
                rows.append(
                    {
                        "model": model,
                        "fold": int(record["fold"]),
                        **point,
                        "best_epoch": int(convergence["best_epoch"]),
                    }
                )
    return rows


def write_convergence_figures(
    result: dict[str, Any], output_dir: Path
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = sorted(result.get("convergence_audit", {}))
    if not models:
        return
    figure, axes = plt.subplots(
        len(models), 1, figsize=(9, 3.2 * len(models)), squeeze=False
    )
    for axis, model in zip(axes[:, 0], models):
        records = result["crossfit"]["records_by_model"][model]
        for record in records:
            history = record["convergence"]["history"]
            epochs = [point["epoch"] for point in history]
            axis.plot(
                epochs,
                [point["train_total"] for point in history],
                linestyle="--",
                alpha=0.45,
            )
            axis.plot(
                epochs,
                [point["monitor_total"] for point in history],
                alpha=0.75,
                label=f"fold {record['fold']} monitor",
            )
        audit = result["convergence_audit"][model]
        axis.set_title(
            f"{model}: median improvement "
            f"{audit['median_relative_monitor_improvement']:.1%}, "
            f"median best epoch {audit['median_best_epoch']:.0f}"
        )
        axis.set_xlabel("epoch")
        axis.set_ylabel("normalized effect loss")
        axis.set_yscale("log")
        axis.grid(alpha=0.25)
    figure.suptitle(
        "Predicted consequence convergence (solid=inner-log monitor, dashed=train)"
    )
    figure.tight_layout()
    figure_root = output_dir / "figures" / "predicted_consequence"
    figure_root.mkdir(parents=True, exist_ok=True)
    figure.savefig(figure_root / "training_convergence.png", dpi=180)
    plt.close(figure)

    all_models = list(result["effect_prediction"])
    delta_spearman = [
        result["effect_prediction"][model]["validation_candidate_delta"][
            "flat_normalized_spearman"
        ]
        or 0.0
        for model in all_models
    ]
    variance_recovery = [
        result["effect_prediction"][model][
            "validation_diversity_recovery"
        ]["candidate_variance_recovery_ratio"]
        or 0.0
        for model in all_models
    ]
    x = np.arange(len(all_models))
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].bar(x, delta_spearman)
    axes[0].axhline(0.30, color="tab:red", linestyle="--", label="gate")
    axes[0].set_title("Candidate-delta Spearman")
    axes[1].bar(x, variance_recovery)
    axes[1].axhspan(0.5, 1.5, color="tab:green", alpha=0.12, label="gate")
    axes[1].set_title("Candidate-variance recovery")
    for axis in axes:
        axis.set_xticks(x, all_models, rotation=25, ha="right")
        axis.grid(axis="y", alpha=0.25)
        axis.legend()
    figure.suptitle("Held-out predicted-consequence fidelity")
    figure.tight_layout()
    figure.savefig(figure_root / "candidate_fidelity.png", dpi=180)
    plt.close(figure)


def render_report(result: dict[str, Any]) -> str:
    reconstruction_rows = []
    for model, metrics in result["effect_prediction"].items():
        validation = metrics["validation"]
        candidate_delta = metrics["validation_candidate_delta"]
        diversity = metrics["validation_diversity_recovery"]
        reconstruction_rows.append(
            f"| {model} | {validation['normalized_rmse']:.4f} | "
            f"{candidate_delta['normalized_rmse']:.4f} | "
            f"{candidate_delta['flat_normalized_spearman']:.4f} | "
            f"{diversity['pairwise_distance_spearman_global']:.4f} | "
            f"{diversity['candidate_variance_recovery_ratio']:.4f} |"
        )
    planning_rows = []
    for name, probe in result["probes"].items():
        metrics = probe["aggregate_score_and_ranking"]
        planning_rows.append(
            f"| {name} | {metrics['pairwise_ranking_accuracy']:.4f} | "
            f"{metrics['ndcg_mean']:.4f} | "
            f"{metrics['spearman_per_scene_mean']:.4f} | "
            f"{metrics['top1_accuracy']:.4f} | "
            f"{metrics['top1_score_regret_mean']:.5f} | "
            f"{metrics['rmse']:.4f} |"
        )
    comparison = result["primary_comparison"]
    judgement = result["predicted_planning_gain_judgement"]
    convergence = result.get("convergence_audit", {}).get(
        result["primary_effect_model"], {}
    )
    capacity = result.get("overfit_capacity_sanity", {})
    capacity_delta = capacity.get("candidate_delta_metrics", {})
    capacity_diversity = capacity.get("diversity_recovery", {})
    convergence_table = [
        (
            f"| {model} | "
            f"{'PASS' if audit['optimization_converged'] else 'FAIL'} | "
            f"{audit['median_relative_monitor_improvement']:.1%} | "
            f"{audit['median_best_epoch']:.0f} | "
            f"{audit['parameter_count']} |"
        )
        for model, audit in result.get("convergence_audit", {}).items()
    ]
    lines = [
        "# Predicted Candidate-Consequence Probe",
        "",
        f"- Scope: **{result['scene_count']} scenes / {result['candidate_count']} candidates**.",
        f"- Outer split: complete logs, **{result['split']['train_logs']} train / {result['split']['validation_logs']} validation**, overlap **0**.",
        f"- Training consequence features: **{result['crossfit']['folds']} log-group folds, out-of-fold**.",
        f"- Train-OOF-selected primary predictor: **{result['primary_effect_model']}**.",
        f"- Primary optimization convergence: **{'PASS' if convergence.get('optimization_converged') else 'FAIL'}**.",
        f"- Candidate-specific fidelity gate: **{'PASS' if result.get('predictor_fidelity_gate', {}).get('pass') else 'FAIL'}**.",
        f"- Overall prediction gate: **{result.get('overall_prediction_gate')}**.",
        f"- Leakage audit: **{'PASS' if result['leakage_audit']['pass'] else 'FAIL'}**.",
        f"- Fixed-seed repeat: **{'PASS' if result.get('determinism_verification', {}).get('pass') else 'NOT VERIFIED'}**.",
        "",
        "## Dynamic-consequence prediction quality",
        "",
        "| Predictor | Validation NRMSE | Candidate-delta NRMSE | Candidate-delta Spearman | Pairwise-distance Spearman | Candidate-variance recovery |",
        "|---|---:|---:|---:|---:|---:|",
        *reconstruction_rows,
        "",
        "Only dynamic channels are predicted. Drivable-area, lane, and route SDF channels are exact candidate/map geometry and are supplied to every fair online baseline.",
        "",
        "## Predictor convergence",
        "",
        "| Predictor | Optimization gate | Median monitor improvement | Median best epoch | Parameters |",
        "|---|---|---:|---:|---:|",
        *convergence_table,
        "",
        "Epoch selection uses only an inner log-disjoint monitor split inside each OOF training fold. The OOF fold itself and the outer validation logs are not used for early stopping.",
        "",
        f"Small-subset overfit capacity sanity: **{'PASS' if capacity.get('pass') else 'FAIL'}** on {capacity.get('scene_count', 0)} scenes / {capacity.get('candidate_count', 0)} candidates. Train loss reduction **{capacity.get('relative_train_loss_reduction', 0.0):.1%}**; candidate-delta Spearman **{capacity_delta.get('flat_normalized_spearman')}**; pairwise-distance Spearman **{capacity_diversity.get('pairwise_distance_spearman_global')}**; variance recovery **{capacity_diversity.get('candidate_variance_recovery_ratio')}**.",
        "",
        "The overfit check is an implementation/capacity diagnostic only. It is never used as held-out evidence or for model selection.",
        "",
        "## Planning utility",
        "",
        "| Probe | Pairwise | NDCG | Scene Spearman | Top-1 | Regret | Score RMSE |",
        "|---|---:|---:|---:|---:|---:|---:|",
        *planning_rows,
        "",
        f"Primary predicted-vs-direct judgement: **{judgement}**.",
        "",
        "Paired scene-bootstrap deltas (predicted minus fair direct):",
        "",
    ]
    for metric, values in comparison.items():
        lines.append(
            f"- {metric}: **{values['delta_candidate_minus_baseline']:+.6f}**, "
            f"95% CI [{values['delta_ci95'][0]:+.6f}, {values['delta_ci95'][1]:+.6f}]"
        )
    lines += [
        "",
        "## Interpretation boundary",
        "",
        "The direct baseline sees exactly the same current structured actor state, candidate trajectory, constant-velocity candidate/actor interactions, and exact map channels as the consequence predictor. Therefore any predicted-consequence gain is an intermediate-representation/inductive-bias gain, not extra online information.",
        "",
        "Logged-future consequences are supervised labels only. Downstream outer-train features are log-group OOF predictions and outer-validation features are predictions from a model trained only on outer-train logs; oracle future values are never input to the predicted planner.",
        "",
        "Optimization convergence, candidate-specific prediction fidelity, and downstream planning utility are separate gates. A downstream no-gain result does not reject the method unless prediction fidelity first passes.",
        "",
        "Current actors come from planning-instant annotations in this minimal probe. Dynamic future consequences are predictions, but this is still a structured-perception upper bound rather than an end-to-end camera-to-consequence result.",
        "",
        "The oracle-dynamic probe is only a ceiling. It is never used as an online feature and validation oracle targets never train either model.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser, max_scenes=500)
    parser.add_argument("--num-candidates", type=int, default=12, choices=range(8, 17))
    parser.add_argument("--max-scenes-per-log", type=int, default=8)
    parser.add_argument("--crossfit-folds", type=int, default=5)
    parser.add_argument(
        "--effect-models",
        default=(
            "ridge,extra_trees,mlp_raw,mlp_delta,mlp_delta_strong"
        ),
        help="comma-separated fixed model families",
    )
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--extra-trees-estimators", type=int, default=64)
    parser.add_argument("--extra-trees-max-depth", type=int, default=14)
    parser.add_argument("--extra-trees-min-samples-leaf", type=int, default=4)
    parser.add_argument("--extra-trees-max-features", type=float, default=0.7)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--mlp-hidden-dims", default="512,256,128")
    parser.add_argument("--mlp-learning-rate", type=float, default=1e-3)
    parser.add_argument("--mlp-weight-decay", type=float, default=1e-4)
    parser.add_argument("--mlp-max-epochs", type=int, default=160)
    parser.add_argument("--mlp-patience", type=int, default=20)
    parser.add_argument("--mlp-delta-weight", type=float, default=1.0)
    parser.add_argument("--mlp-delta-weight-strong", type=float, default=4.0)
    parser.add_argument("--mlp-overfit-scenes", type=int, default=8)
    parser.add_argument("--mlp-overfit-epochs", type=int, default=400)
    parser.add_argument(
        "--mlp-device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument(
        "--determinism-reference",
        type=Path,
        default=None,
        help="optional prior output directory from the same fixed-seed command",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model_kinds = [
        value.strip() for value in args.effect_models.split(",") if value.strip()
    ]
    missing_primary = [
        value
        for value in PRIMARY_EFFECT_MODEL_CANDIDATES
        if value not in model_kinds
    ]
    if missing_primary:
        raise ValueError(
            f"formal run must include primary candidates {missing_primary}"
        )
    dataset = build_dataset(args)
    validation_mask = np.asarray(
        [
            split_is_validation(str(name), args.seed)
            for name in dataset["log_names"]
        ],
        dtype=bool,
    )
    train_mask = ~validation_mask
    train_logs = set(dataset["log_names"][train_mask].tolist())
    validation_logs = set(dataset["log_names"][validation_mask].tolist())
    if not train_logs or not validation_logs or train_logs & validation_logs:
        raise RuntimeError("invalid outer log split")

    exact_indices, dynamic_indices = partition_environment_features(
        dataset["feature_names"]["environment"]
    )
    exact = dataset["environment"][:, exact_indices]
    dynamic = dataset["environment"][:, dynamic_indices]
    dynamic_names = [
        dataset["feature_names"]["environment"][index]
        for index in dynamic_indices
    ]
    predictor_inputs = np.concatenate(
        [
            dataset["trajectory"],
            dataset["current"],
            dataset["current_candidate"],
            exact,
        ],
        axis=1,
    )
    original_online = np.concatenate(
        [dataset["trajectory"], dataset["current"]], axis=1
    )
    train_indices = np.flatnonzero(train_mask)
    validation_indices = np.flatnonzero(validation_mask)
    train_scale = np.std(dynamic[train_mask], axis=0)
    train_centered_truth = center_within_scene(
        dynamic[train_mask], dataset["scene_tokens"][train_mask]
    )
    candidate_delta_scale = np.std(train_centered_truth, axis=0)

    prediction_payload: dict[str, np.ndarray] = {
        "validation_truth_dynamic": dynamic[validation_mask],
        "validation_scene_tokens": dataset["scene_tokens"][validation_mask],
        "validation_candidate_slots": dataset["candidate_slots"][validation_mask],
    }
    effect_results: dict[str, Any] = {}
    convergence_results: dict[str, Any] = {}
    effect_rows: list[dict[str, Any]] = []
    predicted_features: dict[str, np.ndarray] = {}
    fold_records: dict[str, list[dict[str, Any]]] = {}
    for model_kind in model_kinds:
        if model_kind in {"mlp_raw", "mlp_delta", "mlp_delta_strong"}:
            oof, held_out, folds, convergence = (
                cross_fitted_mlp_predictions(
                    predictor_inputs,
                    dynamic,
                    dataset["log_names"],
                    dataset["scene_tokens"],
                    train_mask,
                    validation_mask,
                    model_kind=model_kind,
                    args=args,
                )
            )
            convergence_results[model_kind] = convergence
        else:
            oof, held_out, folds = cross_fitted_effect_predictions(
                predictor_inputs,
                dynamic,
                dataset["log_names"],
                train_mask,
                validation_mask,
                model_kind=model_kind,
                args=args,
            )
        all_predictions = np.empty_like(dynamic)
        all_predictions[train_indices] = oof
        all_predictions[validation_indices] = held_out
        predicted_features[model_kind] = all_predictions
        fold_records[model_kind] = folds
        effect_results[model_kind] = {
            "outer_train_oof": aggregate_effect_metrics(
                dynamic[train_mask], oof, train_scale
            ),
            "validation": aggregate_effect_metrics(
                dynamic[validation_mask], held_out, train_scale
            ),
            "outer_train_candidate_delta": aggregate_effect_metrics(
                train_centered_truth,
                center_within_scene(
                    oof, dataset["scene_tokens"][train_mask]
                ),
                candidate_delta_scale,
            ),
            "outer_train_diversity_recovery": consequence_diversity_recovery(
                dynamic[train_mask],
                oof,
                dataset["scene_tokens"][train_mask],
                train_scale,
            ),
            "validation_candidate_delta": aggregate_effect_metrics(
                center_within_scene(
                    dynamic[validation_mask],
                    dataset["scene_tokens"][validation_mask],
                ),
                center_within_scene(
                    held_out, dataset["scene_tokens"][validation_mask]
                ),
                candidate_delta_scale,
            ),
            "validation_diversity_recovery": consequence_diversity_recovery(
                dynamic[validation_mask],
                held_out,
                dataset["scene_tokens"][validation_mask],
                train_scale,
            ),
        }
        effect_rows.extend(
            feature_effect_metrics(
                model_kind=model_kind,
                split="outer_train_oof",
                truth=dynamic[train_mask],
                prediction=oof,
                feature_names=dynamic_names,
                train_scale=train_scale,
            )
        )
        effect_rows.extend(
            feature_effect_metrics(
                model_kind=model_kind,
                split="outer_validation",
                truth=dynamic[validation_mask],
                prediction=held_out,
                feature_names=dynamic_names,
                train_scale=train_scale,
            )
        )
        prediction_payload[f"validation_prediction_{model_kind}"] = held_out

    probe_features = {
        "Probe_B_original_current_plus_trajectory": original_online,
        "Probe_D_fair_online_direct": predictor_inputs,
        **{
            f"Probe_E_predicted_dynamic_{model_kind}": np.concatenate(
                [predictor_inputs, prediction], axis=1
            )
            for model_kind, prediction in predicted_features.items()
        },
        "Probe_F_oracle_dynamic_ceiling": np.concatenate(
            [predictor_inputs, dynamic], axis=1
        ),
    }
    probes: dict[str, Any] = {}
    score_predictions: dict[str, np.ndarray] = {}
    for probe_name, features in probe_features.items():
        probe, prediction = evaluate_planning_probe(
            features,
            dataset,
            train_mask,
            validation_mask,
            seed=args.seed,
        )
        probes[probe_name] = probe
        score_predictions[probe_name] = prediction

    score_truth = dataset["targets"]["aggregate_score"][validation_mask]
    validation_tokens = dataset["scene_tokens"][validation_mask]
    comparisons: dict[str, Any] = {}
    for model_kind in model_kinds:
        comparisons[model_kind] = paired_bootstrap_comparison(
            score_truth,
            score_predictions["Probe_D_fair_online_direct"],
            score_predictions[f"Probe_E_predicted_dynamic_{model_kind}"],
            validation_tokens,
            seed=args.seed,
            samples=args.bootstrap_samples,
        )
    oracle_comparison = paired_bootstrap_comparison(
        score_truth,
        score_predictions["Probe_D_fair_online_direct"],
        score_predictions["Probe_F_oracle_dynamic_ceiling"],
        validation_tokens,
        seed=args.seed,
        samples=args.bootstrap_samples,
    )
    # Select delta-loss strength only from outer-train OOF fidelity. Outer
    # validation consequence labels and planning scores are not consulted.
    primary_effect_model = max(
        PRIMARY_EFFECT_MODEL_CANDIDATES,
        key=lambda name: (
            effect_results[name]["outer_train_candidate_delta"][
                "flat_normalized_spearman"
            ]
            or -1.0,
            effect_results[name]["outer_train_diversity_recovery"][
                "pairwise_distance_spearman_global"
            ]
            or -1.0,
        ),
    )
    primary_comparison = comparisons[primary_effect_model]
    judgement = planning_gain_judgement(primary_comparison)
    primary_convergence = convergence_results[primary_effect_model]
    capacity_sanity = overfit_capacity_sanity(
        predictor_inputs,
        dynamic,
        dataset["log_names"],
        dataset["scene_tokens"],
        train_mask,
        model_kind=primary_effect_model,
        args=args,
    )
    primary_fidelity = effect_results[primary_effect_model][
        "validation_diversity_recovery"
    ]
    candidate_delta = effect_results[primary_effect_model][
        "validation_candidate_delta"
    ]
    fidelity_pass = bool(
        (candidate_delta["flat_normalized_spearman"] or 0.0) >= 0.30
        and (
            primary_fidelity["pairwise_distance_spearman_global"] or 0.0
        )
        >= 0.50
        and 0.50
        <= (
            primary_fidelity["candidate_variance_recovery_ratio"] or 0.0
        )
        <= 1.50
    )
    if not primary_convergence["optimization_converged"]:
        overall_gate = "PREDICTOR_NOT_CONVERGED"
    elif not fidelity_pass:
        overall_gate = "PREDICTOR_FIDELITY_NOT_MET"
    elif judgement in {"PASS", "CONDITIONAL_PASS"}:
        overall_gate = "PREDICTED_CONSEQUENCE_GAIN_SUPPORTED"
    else:
        overall_gate = "PREDICTED_CONSEQUENCE_GAIN_NOT_DEMONSTRATED"

    result = {
        "scene_count": int(dataset["selected_scene_count"]),
        "candidate_count": int(len(dataset["scene_tokens"])),
        "candidates_per_scene": int(args.num_candidates),
        "primary_effect_model": primary_effect_model,
        "primary_effect_model_selection": {
            "candidates": list(PRIMARY_EFFECT_MODEL_CANDIDATES),
            "criterion": (
                "maximize outer-train OOF candidate-delta Spearman, then "
                "outer-train OOF pairwise-distance Spearman"
            ),
            "outer_validation_used_for_selection": False,
        },
        "predicted_planning_gain_judgement": judgement,
        "overall_prediction_gate": overall_gate,
        "split": {
            "unit": "complete log_name",
            "train_logs": len(train_logs),
            "validation_logs": len(validation_logs),
            "train_scenes": int(
                len(np.unique(dataset["scene_tokens"][train_mask]))
            ),
            "validation_scenes": int(
                len(np.unique(dataset["scene_tokens"][validation_mask]))
            ),
            "log_overlap": sorted(train_logs & validation_logs),
        },
        "crossfit": {
            "folds": int(args.crossfit_folds),
            "records_by_model": fold_records,
            "outer_train_feature_policy": "out-of-fold predicted dynamic consequence",
            "outer_validation_feature_policy": "model fit only on all outer-train logs",
        },
        "feature_contract": {
            "predictor_inputs": [
                "candidate trajectory",
                "planning-instant aggregate state",
                "planning-instant actor boxes/velocities",
                "constant-velocity candidate-relative interaction features",
                "exact static candidate/map SDF summaries",
            ],
            "predicted_channels": list(PREDICTED_DYNAMIC_CHANNELS),
            "exact_online_channels": list(EXACT_ONLINE_CHANNELS),
            "effect_horizons_s": [1.0, 2.0, 4.0],
            "predictor_input_feature_count": int(predictor_inputs.shape[1]),
            "predicted_feature_count": int(dynamic.shape[1]),
            "exact_map_feature_count": int(exact.shape[1]),
            "current_actor_source": (
                "planning-instant GT annotations; structured-perception upper bound"
            ),
        },
        "effect_prediction": effect_results,
        "convergence_audit": convergence_results,
        "overfit_capacity_sanity": capacity_sanity,
        "predictor_fidelity_gate": {
            "pass": fidelity_pass,
            "criterion": {
                "candidate_delta_spearman_min": 0.30,
                "pairwise_distance_spearman_min": 0.50,
                "candidate_variance_recovery_range": [0.50, 1.50],
            },
            "reason": (
                "Downstream no-gain cannot reject the method unless this "
                "prediction-fidelity gate passes."
            ),
        },
        "probes": probes,
        "comparisons_vs_fair_direct": comparisons,
        "primary_comparison": primary_comparison,
        "oracle_ceiling_comparison_vs_fair_direct": oracle_comparison,
        "leakage_audit": {
            "pass": (
                not (train_logs & validation_logs)
                and all(
                    not record["log_overlap"]
                    for records in fold_records.values()
                    for record in records
                )
                and all(
                    not record.get("convergence", {})
                    .get("inner_split", {})
                    .get("log_overlap", [])
                    for records in fold_records.values()
                    for record in records
                )
                and all(
                    not audit.get("final_model_convergence", {})
                    .get("inner_split", {})
                    .get("log_overlap", [])
                    for audit in convergence_results.values()
                )
            ),
            "outer_log_overlap": sorted(train_logs & validation_logs),
            "crossfit_fold_log_overlap_count": int(
                sum(
                    bool(record["log_overlap"])
                    for records in fold_records.values()
                    for record in records
                )
            ),
            "inner_monitor_log_overlap_count": int(
                sum(
                    bool(
                        record.get("convergence", {})
                        .get("inner_split", {})
                        .get("log_overlap", [])
                    )
                    for records in fold_records.values()
                    for record in records
                )
                + sum(
                    bool(
                        audit.get("final_model_convergence", {})
                        .get("inner_split", {})
                        .get("log_overlap", [])
                    )
                    for audit in convergence_results.values()
                )
            ),
            "downstream_direct_baseline_sees_same_predictor_inputs": True,
            "outer_train_predicted_features_are_out_of_fold": True,
            "outer_validation_oracle_features_train_no_model": True,
            "official_scores_or_factors_in_predictor_inputs": False,
            "candidate_type_or_id_in_predictor_inputs": False,
        },
        "interpretation": {
            "what_is_predicted": (
                "dynamic non-reactive candidate-relative consequence summaries"
            ),
            "what_is_not_predicted": (
                "static map/route geometry, which is computed exactly online"
            ),
            "candidate_source": (
                "deterministic expert-anchor perturbation bank; trajectory source "
                "is not used as a model input label"
            ),
            "current_limit": (
                "current actor state uses annotations, not a learned visual "
                "perception stack"
            ),
            "oracle_is_only_a_ceiling": True,
        },
        "paths": dataset["paths"],
        "failures": dataset["failures"],
    }
    if args.determinism_reference is not None:
        result["determinism_verification"] = verify_determinism(
            args.determinism_reference, result, prediction_payload
        )
    write_json(
        args.output_dir / "predicted_consequence_probe_results.json", result
    )
    write_dataframe(
        pd.DataFrame(effect_rows),
        args.output_dir / "predicted_consequence_feature_metrics.csv",
    )
    write_dataframe(
        pd.DataFrame(convergence_rows(result)),
        args.output_dir / "predicted_consequence_training_curves.csv",
    )
    np.savez_compressed(
        args.output_dir / "predicted_consequence_validation_predictions.npz",
        **prediction_payload,
    )
    write_text(
        args.output_dir / "PREDICTED_CONSEQUENCE_REPORT.md",
        render_report(result),
    )
    write_convergence_figures(result, args.output_dir)
    print(
        json.dumps(
            {
                "scene_count": result["scene_count"],
                "candidate_count": result["candidate_count"],
                "judgement": judgement,
                "overall_prediction_gate": overall_gate,
                "optimization_converged": primary_convergence[
                    "optimization_converged"
                ],
                "predictor_fidelity_pass": fidelity_pass,
                "primary_comparison": primary_comparison,
                "leakage_audit": result["leakage_audit"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
