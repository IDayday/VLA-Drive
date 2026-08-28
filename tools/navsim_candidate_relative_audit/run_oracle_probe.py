#!/usr/bin/env python3
"""Run a lightweight leakage-audited oracle planning-utility probe."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

from .build_candidate_relative_targets import ACTOR_FEATURES, ENVIRONMENT_FEATURES
from .common import (
    add_common_arguments,
    append_command,
    ensure_output_dir,
    read_parquet,
    stable_hash,
    write_json,
    write_markdown,
)


@dataclass
class Standardizer:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray) -> "Standardizer":
        mean = np.nanmean(values, axis=0)
        scale = np.nanstd(values, axis=0)
        scale[~np.isfinite(scale) | (scale < 1e-6)] = 1.0
        mean[~np.isfinite(mean)] = 0.0
        return cls(mean, scale)

    def transform(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        # ``np.nan_to_num(..., nan=array)`` is not supported consistently by
        # the NumPy versions deployed with NAVSIM.  Column-wise replacement is
        # explicit and also handles +/-inf produced by missing map relations.
        filled = np.where(np.isfinite(values), values, self.mean)
        result = (filled - self.mean) / self.scale
        return np.clip(result, -20.0, 20.0)


def fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float = 1.0) -> tuple[Standardizer, np.ndarray]:
    scaler = Standardizer.fit(x)
    z = scaler.transform(x)
    design = np.column_stack([np.ones(len(z)), z])
    regularizer = np.eye(design.shape[1]) * alpha
    regularizer[0, 0] = 0.0
    weights = np.linalg.solve(design.T @ design + regularizer, design.T @ y)
    return scaler, weights


def predict_ridge(model: tuple[Standardizer, np.ndarray], x: np.ndarray) -> np.ndarray:
    scaler, weights = model
    z = scaler.transform(x)
    return np.column_stack([np.ones(len(z)), z]) @ weights


def fit_logistic(
    x: np.ndarray, y: np.ndarray, *, steps: int = 300, learning_rate: float = 0.05, l2: float = 1e-3
) -> tuple[Standardizer, np.ndarray, float | None]:
    scaler = Standardizer.fit(x)
    z = scaler.transform(x)
    if len(np.unique(y)) < 2:
        return scaler, np.zeros(z.shape[1] + 1), float(np.mean(y))
    design = np.column_stack([np.ones(len(z)), z])
    weights = np.zeros(design.shape[1], dtype=np.float64)
    positive = max(float(np.mean(y)), 1e-3)
    sample_weight = np.where(y > 0.5, 0.5 / positive, 0.5 / max(1.0 - positive, 1e-3))
    for _ in range(steps):
        logits = np.clip(design @ weights, -30, 30)
        probability = 1.0 / (1.0 + np.exp(-logits))
        gradient = design.T @ ((probability - y) * sample_weight) / len(y)
        gradient[1:] += l2 * weights[1:]
        weights -= learning_rate * gradient
    return scaler, weights, None


def predict_logistic(model: tuple[Standardizer, np.ndarray, float | None], x: np.ndarray) -> np.ndarray:
    scaler, weights, constant = model
    if constant is not None:
        return np.full(len(x), constant, dtype=np.float64)
    design = np.column_stack([np.ones(len(x)), scaler.transform(x)])
    return 1.0 / (1.0 + np.exp(-np.clip(design @ weights, -30, 30)))


def binary_metrics(y: np.ndarray, probability: np.ndarray) -> dict[str, float | None]:
    y = np.asarray(y, dtype=np.int32)
    probability = np.asarray(probability, dtype=np.float64)
    prediction = probability >= 0.5
    tp = int(np.sum((prediction == 1) & (y == 1)))
    fp = int(np.sum((prediction == 1) & (y == 0)))
    fn = int(np.sum((prediction == 0) & (y == 1)))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    if len(np.unique(y)) < 2:
        auroc = None
    else:
        ranks = rankdata(probability)
        n_pos = int(y.sum())
        n_neg = len(y) - n_pos
        auroc = float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))
    brier = float(np.mean((probability - y) ** 2))
    ece = 0.0
    for lower in np.linspace(0, 0.9, 10):
        mask = (probability >= lower) & (probability < lower + 0.1 + 1e-12)
        if mask.any():
            ece += float(mask.mean() * abs(probability[mask].mean() - y[mask].mean()))
    return {"auroc": auroc, "f1": float(f1), "brier": brier, "ece_10bin": ece}


def ranking_metrics(meta: pd.DataFrame, truth: np.ndarray, prediction: np.ndarray) -> dict[str, float | None]:
    pair_correct, pair_total = 0, 0
    ndcg, top1, regret = [], [], []
    for _, indices in meta.groupby("scene_token", sort=False).groups.items():
        indices = np.asarray(list(indices), dtype=np.int64)
        y, p = truth[indices], prediction[indices]
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                if abs(y[i] - y[j]) <= 1e-9:
                    continue
                pair_total += 1
                pair_correct += int(np.sign(y[i] - y[j]) == np.sign(p[i] - p[j]))
        order = np.argsort(p)[::-1]
        ideal = np.argsort(y)[::-1]
        discounts = 1.0 / np.log2(np.arange(2, len(y) + 2))
        dcg = float(np.sum((2**y[order] - 1) * discounts))
        idcg = float(np.sum((2**y[ideal] - 1) * discounts))
        ndcg.append(dcg / max(idcg, 1e-12))
        selected = order[0]
        top1.append(float(y[selected] >= np.max(y) - 1e-9))
        regret.append(float(np.max(y) - y[selected]))
    correlation = None
    if len(truth) >= 3 and np.ptp(truth) > 0 and np.ptp(prediction) > 0:
        correlation = float(spearmanr(truth, prediction).statistic)
    return {
        "pairwise_ranking_accuracy": pair_correct / max(pair_total, 1),
        "ranked_pair_count": pair_total,
        "ndcg": float(np.mean(ndcg)) if ndcg else None,
        "spearman": correlation,
        "top1_accuracy": float(np.mean(top1)) if top1 else None,
        "top1_score_regret": float(np.mean(regret)) if regret else None,
        "mae": float(np.mean(np.abs(prediction - truth))),
    }


def _masked_actor_summary(arrays: Any) -> np.ndarray:
    values = np.asarray(arrays["candidate_relative_actor"], dtype=np.float64)
    mask = np.asarray(arrays["candidate_relative_actor_mask"], dtype=bool)
    feature_indices = [
        ACTOR_FEATURES.index("relative_x_m"),
        ACTOR_FEATURES.index("relative_y_m"),
        ACTOR_FEATURES.index("relative_vx_mps"),
        ACTOR_FEATURES.index("relative_vy_mps"),
        ACTOR_FEATURES.index("polygon_clearance_m"),
        ACTOR_FEATURES.index("in_candidate_corridor"),
    ]
    summaries = []
    for candidate in range(values.shape[0]):
        per_horizon = []
        for horizon in range(values.shape[1]):
            valid = values[candidate, horizon][mask[candidate, horizon]]
            if len(valid):
                selected = valid[:, feature_indices]
                per_horizon.extend(np.mean(selected, axis=0).tolist())
                per_horizon.extend(np.min(selected, axis=0).tolist())
                per_horizon.append(float(len(valid)))
            else:
                per_horizon.extend([0.0] * (len(feature_indices) * 2 + 1))
        summaries.append(per_horizon)
    return np.asarray(summaries, dtype=np.float64)


def _semantic_action(candidate_type: str) -> str:
    if "left" in candidate_type or candidate_type == "same_endpoint_mid_curve":
        return "left"
    if "right" in candidate_type or candidate_type in {"same_prefix_different_tail", "different_prefix_similar_endpoint"}:
        return "right"
    if candidate_type in {"slow_time_scale", "mild_deceleration"}:
        return "slow"
    if candidate_type in {"fast_time_scale", "mild_acceleration"}:
        return "fast"
    return "factual_or_other"


def nearest_centroid_accuracy(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray) -> dict[str, Any]:
    scaler = Standardizer.fit(x_train)
    train_z, val_z = scaler.transform(x_train), scaler.transform(x_val)
    classes = sorted(set(y_train.tolist()))
    centroids = np.stack([train_z[y_train == label].mean(axis=0) for label in classes])
    distance = ((val_z[:, None, :] - centroids[None, :, :]) ** 2).mean(axis=-1)
    prediction = np.asarray(classes, dtype=object)[np.argmin(distance, axis=1)]
    majority = max(np.mean(y_val == label) for label in set(y_val.tolist()))
    return {
        "accuracy": float(np.mean(prediction == y_val)),
        "majority_baseline": float(majority),
        "classes": classes,
    }


def assemble_dataset(output_dir: Any, max_scenes: int) -> tuple[dict[str, np.ndarray], pd.DataFrame, dict[str, list[str]]]:
    index = read_parquet(output_dir / "targets/index.parquet")
    if max_scenes > 0:
        index = index.head(max_scenes)
    manifest = read_parquet(output_dir / "candidate_manifest.parquet")
    metrics = read_parquet(output_dir / "candidate_metrics.parquet")
    metrics = metrics[metrics["scoring_success"] & (metrics["traffic_policy"] == "non_reactive")]
    feature_a, feature_b, feature_c, interaction = [], [], [], []
    meta_rows, target_rows = [], []
    names_a: list[str] = []
    names_b: list[str] = []
    names_c: list[str] = []
    for index_row in index.itertuples():
        token = index_row.scene_token
        arrays = np.load(output_dir / index_row.target_path)
        group = manifest[manifest["scene_token"] == token].sort_values("candidate_index")
        scored = metrics[metrics["scene_token"] == token].sort_values("candidate_index")
        if len(group) != len(scored) or len(group) != arrays["C_environment_only"].shape[0]:
            continue
        actor_summary = _masked_actor_summary(arrays)
        env_flat = arrays["C_environment_only"].reshape(len(group), -1)
        interaction_features = np.concatenate([env_flat, actor_summary], axis=1)
        for local_index, (manifest_row, score_row) in enumerate(zip(group.itertuples(), scored.itertuples())):
            poses = np.column_stack([manifest_row.pose_x_m, manifest_row.pose_y_m, manifest_row.pose_heading_rad])
            a = np.concatenate(
                [
                    poses.ravel(),
                    np.asarray(
                        [
                            manifest_row.max_speed_mps,
                            manifest_row.max_acceleration_mps2,
                            manifest_row.max_abs_curvature_1pm,
                            manifest_row.terminal_displacement_m,
                        ]
                    ),
                ]
            )
            current = arrays["current_scene_features"].astype(np.float64)
            b = np.concatenate([a, current])
            c = np.concatenate([b, interaction_features[local_index]])
            feature_a.append(a)
            feature_b.append(b)
            feature_c.append(c)
            interaction.append(interaction_features[local_index])
            meta_rows.append(
                {
                    "scene_token": token,
                    "log_name": manifest_row.log_name,
                    "candidate_index": manifest_row.candidate_index,
                    "candidate_type": manifest_row.candidate_type,
                    "semantic_action": _semantic_action(manifest_row.candidate_type),
                }
            )
            target_rows.append(
                {
                    "aggregate_score": score_row.aggregate_score,
                    "collision": float(score_row.no_at_fault_collision < 1.0),
                    "ttc_violation": float(score_row.ttc < 1.0),
                    "dac_violation": float(score_row.dac < 1.0),
                    "ddc_violation": float(score_row.ddc < 1.0),
                    "comfort_violation": float(score_row.comfort < 1.0),
                    "progress": score_row.progress,
                }
            )
        if not names_a:
            names_a = [f"waypoint_{step}_{field}" for step in range(8) for field in ("x", "y", "heading")]
            names_a += ["max_speed", "max_acceleration", "max_abs_curvature", "terminal_displacement"]
            names_b = names_a + [
                "current_ego_speed",
                "current_ego_acceleration",
                "current_actor_count",
                "current_min_actor_distance",
                "current_route_roadblock_count",
                "current_red_light_count",
            ]
            env_names = [f"future_{h}_{name}" for h in range(8) for name in ENVIRONMENT_FEATURES]
            actor_names = [f"actor_summary_{index}" for index in range(actor_summary.shape[1])]
            names_c = names_b + env_names + actor_names
    meta = pd.DataFrame(meta_rows).reset_index(drop=True)
    targets = pd.DataFrame(target_rows).reset_index(drop=True)
    data = {
        "A": np.asarray(feature_a, dtype=np.float64),
        "B": np.asarray(feature_b, dtype=np.float64),
        "C": np.asarray(feature_c, dtype=np.float64),
        "interaction": np.asarray(interaction, dtype=np.float64),
        **{f"target_{column}": targets[column].to_numpy(dtype=np.float64) for column in targets},
    }
    return data, meta, {"A": names_a, "B": names_b, "C": names_c}


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = ensure_output_dir(args.output_dir)
    data, meta, feature_names = assemble_dataset(output_dir, args.max_scenes)
    logs = sorted(meta.log_name.unique().tolist()) if len(meta) else []
    if len(logs) < 2:
        result = {
            "status": "INCONCLUSIVE",
            "reason": "Fewer than two complete logs; log-level train/validation separation is impossible in smoke scope.",
            "scene_count": int(meta.scene_token.nunique()) if len(meta) else 0,
            "log_count": len(logs),
        }
        write_json(output_dir / "oracle_probe_results.json", result)
        write_markdown(
            output_dir / "ORACLE_PROBE_REPORT.md",
            "# Oracle Planning-Utility Probe\n\nINCONCLUSIVE: fewer than two complete logs; a leakage-safe log-level split is impossible for this scope.",
        )
        return result
    val_logs = [log for log in logs if stable_hash(log) % 10 < 2]
    if not val_logs or len(val_logs) == len(logs):
        val_count = max(1, int(round(len(logs) * 0.2)))
        val_logs = logs[-val_count:]
    train_logs = [log for log in logs if log not in set(val_logs)]
    train_mask = meta.log_name.isin(train_logs).to_numpy()
    val_mask = meta.log_name.isin(val_logs).to_numpy()
    score_truth = data["target_aggregate_score"]
    probe_results: dict[str, Any] = {}
    binary_targets = ["collision", "ttc_violation", "dac_violation", "ddc_violation", "comfort_violation"]
    leakage_banned = ("aggregate_score", "official_pdm", "candidate_type", "candidate_index", "pdm_factor")
    leakage_audit = {}
    for probe in ("A", "B", "C"):
        names = feature_names[probe]
        hits = [name for name in names if any(term in name.lower() for term in leakage_banned)]
        leakage_audit[probe] = {"feature_count": len(names), "banned_name_hits": hits, "passed": not hits}
        x = data[probe]
        model = fit_ridge(x[train_mask], score_truth[train_mask], alpha=args.ridge_alpha)
        prediction = predict_ridge(model, x[val_mask])
        result = {
            "aggregate_score": ranking_metrics(
                meta.loc[val_mask].reset_index(drop=True), score_truth[val_mask], prediction
            ),
            "factors": {},
        }
        progress_model = fit_ridge(x[train_mask], data["target_progress"][train_mask], alpha=args.ridge_alpha)
        progress_prediction = predict_ridge(progress_model, x[val_mask])
        progress_truth = data["target_progress"][val_mask]
        progress_spearman = None
        if np.ptp(progress_truth) > 0 and np.ptp(progress_prediction) > 0:
            progress_spearman = float(spearmanr(progress_truth, progress_prediction).statistic)
        result["factors"]["progress"] = {
            "mae": float(np.mean(np.abs(progress_prediction - progress_truth))),
            "spearman": progress_spearman,
        }
        for target in binary_targets:
            truth = data[f"target_{target}"]
            classifier = fit_logistic(x[train_mask], truth[train_mask])
            probability = predict_logistic(classifier, x[val_mask])
            result["factors"][target] = binary_metrics(truth[val_mask], probability)
            result["factors"][target]["positive_rate"] = float(np.mean(truth[val_mask]))
        probe_results[probe] = result

    x_interaction = data["interaction"]
    semantic = meta.semantic_action.to_numpy(dtype=object)
    candidate_id = meta.candidate_index.astype(str).to_numpy(dtype=object)
    semantic_result = nearest_centroid_accuracy(
        x_interaction[train_mask], semantic[train_mask], x_interaction[val_mask], semantic[val_mask]
    )
    candidate_result = nearest_centroid_accuracy(
        x_interaction[train_mask], candidate_id[train_mask], x_interaction[val_mask], candidate_id[val_mask]
    )
    manifest = read_parquet(output_dir / "candidate_manifest.parquet")
    delta_targets = []
    for token, group in manifest[manifest.scene_token.isin(meta.scene_token)].groupby("scene_token", sort=False):
        group = group.sort_values("candidate_index")
        gt = group[group.is_gt].iloc[0]
        gt_xy = np.column_stack([gt.pose_x_m, gt.pose_y_m])
        for row in group.itertuples():
            delta_targets.append((token, int(row.candidate_index), (np.column_stack([row.pose_x_m, row.pose_y_m]) - gt_xy).ravel()))
    delta_map = {(token, index): delta for token, index, delta in delta_targets}
    delta = np.asarray(
        [delta_map[(row.scene_token, int(row.candidate_index))] for row in meta.itertuples()], dtype=np.float64
    )
    delta_model = fit_ridge(x_interaction[train_mask], delta[train_mask], alpha=args.ridge_alpha)
    delta_prediction = predict_ridge(delta_model, x_interaction[val_mask])
    denominator = np.sum((delta[val_mask] - delta[val_mask].mean(axis=0)) ** 2)
    delta_r2 = 1.0 - float(np.sum((delta_prediction - delta[val_mask]) ** 2)) / max(float(denominator), 1e-12)
    semantic_margin = semantic_result["accuracy"] - semantic_result["majority_baseline"]
    candidate_margin = candidate_result["accuracy"] - candidate_result["majority_baseline"]
    inverse_near_random = semantic_margin <= 0.05 and candidate_margin <= 0.05
    strong_inverse_supported = bool(
        delta_r2 > 0.0 and semantic_margin > 0.05 and candidate_margin > 0.05
    )
    interaction_result = {
        "semantic_action": semantic_result,
        "candidate_id": candidate_result,
        "delta_trajectory_r2": delta_r2,
        "near_random": inverse_near_random,
        "semantic_accuracy_margin_over_majority": semantic_margin,
        "candidate_id_accuracy_margin_over_majority": candidate_margin,
        "strong_inverse_supported": strong_inverse_supported,
        "interpretation": (
            "当前数据可支持候选相对风险重标注，但不足以支持强 interaction inverse dynamics。"
            if not strong_inverse_supported
            else "Interaction-only features carry measurable inverse information across classification and trajectory recovery; this remains an oracle association, not causal inverse dynamics."
        ),
        "excluded_inputs": ["candidate x/y/heading", "candidate speed", "candidate curvature", "all trajectory-derived fields", "candidate type"],
    }
    result = {
        "status": "COMPLETE",
        "scene_count": int(meta.scene_token.nunique()),
        "candidate_count": len(meta),
        "train_logs": train_logs,
        "validation_logs": val_logs,
        "train_scene_count": int(meta.loc[train_mask, "scene_token"].nunique()),
        "validation_scene_count": int(meta.loc[val_mask, "scene_token"].nunique()),
        "probe_definitions": {
            "A": "trajectory-only",
            "B": "current structured scene + trajectory",
            "C": "Probe B + C_environment_only + masked actor relation summaries",
        },
        "probes": probe_results,
        "feature_leakage_audit": leakage_audit,
        "interaction_only_inverse_probe": interaction_result,
        "unavailable_target": "EPDMS/TLC/lane-keeping/extended-comfort are not exposed by the deployed v1 training cache/scorer and were not fabricated.",
    }
    write_json(output_dir / "oracle_probe_results.json", result)
    a = probe_results["A"]["aggregate_score"]
    b = probe_results["B"]["aggregate_score"]
    c = probe_results["C"]["aggregate_score"]
    factor_lines = [
        "| Target | A AUROC/F1/ECE | B AUROC/F1/ECE | C AUROC/F1/ECE |",
        "|---|---:|---:|---:|",
    ]
    for target in binary_targets:
        cells = []
        for probe in ("A", "B", "C"):
            value = probe_results[probe]["factors"][target]
            auroc = "NA" if value["auroc"] is None else f"{value['auroc']:.3f}"
            cells.append(f"{auroc}/{value['f1']:.3f}/{value['ece_10bin']:.3f}")
        factor_lines.append(f"| {target} | {cells[0]} | {cells[1]} | {cells[2]} |")
    progress_cells = [
        f"MAE {probe_results[probe]['factors']['progress']['mae']:.3f}; ρ "
        + (
            "NA"
            if probe_results[probe]["factors"]["progress"]["spearman"] is None
            else f"{probe_results[probe]['factors']['progress']['spearman']:.3f}"
        )
        for probe in ("A", "B", "C")
    ]
    factor_lines.append(f"| progress | {progress_cells[0]} | {progress_cells[1]} | {progress_cells[2]} |")
    factor_table = "\n".join(factor_lines)
    report = f"""# Oracle Planning-Utility Probe

## Leakage-safe split

- Scenes / candidates: {result['scene_count']} / {result['candidate_count']}
- Train logs ({len(train_logs)}): `{', '.join(train_logs)}`
- Validation logs ({len(val_logs)}): `{', '.join(val_logs)}`
- No complete log appears on both sides.

## Aggregate PDM ranking

| Probe | Pairwise accuracy | NDCG | Spearman | Top-1 accuracy | Top-1 regret |
|---|---:|---:|---:|---:|---:|
| A trajectory-only | {a['pairwise_ranking_accuracy']:.4f} | {a['ndcg']:.4f} | {a['spearman']:.4f} | {a['top1_accuracy']:.4f} | {a['top1_score_regret']:.4f} |
| B current+trajectory | {b['pairwise_ranking_accuracy']:.4f} | {b['ndcg']:.4f} | {b['spearman']:.4f} | {b['top1_accuracy']:.4f} | {b['top1_score_regret']:.4f} |
| C candidate-relative future | {c['pairwise_ranking_accuracy']:.4f} | {c['ndcg']:.4f} | {c['spearman']:.4f} | {c['top1_accuracy']:.4f} | {c['top1_score_regret']:.4f} |

Probe C contains independently constructed relative actor/map/traffic-light/risk relationships. It excludes official final PDM score, official aggregate factor columns, candidate type and candidate identity. It is an oracle upper-bound probe: the future relationships would need to be predicted at inference.

## Factor prediction and calibration

{factor_table}

Binary cells are `AUROC/F1/10-bin ECE`.  Collision/TTC/DAC/DDC relations in the oracle features are structured per-step geometry/risk targets, not copied official factor or score columns; their semantic proximity to the evaluation factors is the purpose of this upper-bound test and is not evidence that they are available online.

## Interaction-only inverse probe

- Semantic action accuracy / majority baseline: {semantic_result['accuracy']:.4f} / {semantic_result['majority_baseline']:.4f}
- Candidate-ID accuracy / majority baseline: {candidate_result['accuracy']:.4f} / {candidate_result['majority_baseline']:.4f}
- Δtrajectory R²: {delta_r2:.4f}
- Strong interaction inverse supported: {strong_inverse_supported}
- Interpretation: {interaction_result['interpretation']}

EPDMS, TLC, lane keeping and extended comfort were unavailable in the deployed v1 scorer and are explicitly omitted.
"""
    write_markdown(output_dir / "ORACLE_PROBE_REPORT.md", report)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    args = parser.parse_args()
    run(args)
    append_command(args.output_dir.resolve(), "python -m tools.navsim_candidate_relative_audit.run_oracle_probe " + " ".join(sys.argv[1:]))


if __name__ == "__main__":
    main()
