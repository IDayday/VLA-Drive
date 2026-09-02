#!/usr/bin/env python3
"""Full Base-64 audit and descriptive scorer-success analysis."""

from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text

from .candidate_metrics import scene_metrics
from .schema import FACTOR_NAMES, TOP_K_VALUES
from .utils import (
    atomic_json,
    bootstrap_mean,
    cluster_bootstrap_mean,
    load_proposal_pickle,
    physical_log_name,
    safe_divide,
    sha256_file,
    token_index,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proposal-pickle", type=Path, required=True)
    parser.add_argument("--candidate-matrix", type=Path, required=True)
    parser.add_argument("--status-replay-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-scenes", type=int, default=0)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def _load_status(root: Path | None) -> Dict[str, np.ndarray]:
    if root is None:
        return {}
    chunks = sorted(root.glob("**/chunk_*.pt"))
    if not chunks:
        raise FileNotFoundError(f"No status replay chunks under {root}")
    result: Dict[str, np.ndarray] = {}
    for path in chunks:
        chunk = torch.load(path, map_location="cpu")
        tokens = [str(value) for value in chunk["tokens"]]
        status = np.asarray(chunk["status_feature"], dtype=np.float32)
        if status.shape != (len(tokens), 11):
            raise RuntimeError(f"Malformed status tensor in {path}: {status.shape}")
        for index, token in enumerate(tokens):
            if token in result:
                raise RuntimeError(f"Duplicate status token: {token}")
            result[token] = status[index]
    return result


def _bootstrap_difference(
    success: np.ndarray,
    failure: np.ndarray,
    *,
    replicates: int,
    seed: int,
    chunk_size: int = 256,
) -> Dict[str, float]:
    success = np.asarray(success, dtype=np.float64)
    failure = np.asarray(failure, dtype=np.float64)
    success = success[np.isfinite(success)]
    failure = failure[np.isfinite(failure)]
    if not len(success) or not len(failure):
        return {"ci_low": float("nan"), "ci_high": float("nan"), "standard_error": float("nan")}
    rng = np.random.default_rng(seed)
    values = np.empty(replicates, dtype=np.float64)
    for start in range(0, replicates, chunk_size):
        stop = min(replicates, start + chunk_size)
        left = rng.integers(0, len(success), size=(stop - start, len(success)))
        right = rng.integers(0, len(failure), size=(stop - start, len(failure)))
        values[start:stop] = success[left].mean(axis=1) - failure[right].mean(axis=1)
    return {
        "standard_error": float(values.std(ddof=1)),
        "ci_low": float(np.quantile(values, 0.025)),
        "ci_high": float(np.quantile(values, 0.975)),
    }


def _cohen_d(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    left, right = left[np.isfinite(left)], right[np.isfinite(right)]
    if len(left) < 2 or len(right) < 2:
        return float("nan")
    pooled = math.sqrt(
        ((len(left) - 1) * left.var(ddof=1) + (len(right) - 1) * right.var(ddof=1))
        / (len(left) + len(right) - 2)
    )
    return float((left.mean() - right.mean()) / pooled) if pooled > 0 else 0.0


def _status_features(status: np.ndarray) -> Dict[str, float]:
    if status.shape != (11,):
        return {}
    command = status[7:11]
    return {
        "ego_pose_x": float(status[0]),
        "ego_pose_y": float(status[1]),
        "ego_pose_heading": float(status[2]),
        "ego_velocity_x": float(status[3]),
        "ego_velocity_y": float(status[4]),
        "ego_speed": float(np.linalg.norm(status[3:5])),
        "ego_acceleration_x": float(status[5]),
        "ego_acceleration_y": float(status[6]),
        "ego_acceleration_norm": float(np.linalg.norm(status[5:7])),
        "command_index": float(np.argmax(command)),
        **{f"command_{index}": float(value) for index, value in enumerate(command)},
    }


def _failure_flags(selected_factors: np.ndarray) -> Dict[str, bool]:
    values = dict(zip(FACTOR_NAMES, selected_factors.tolist()))
    flags = {
        "collision/NOC failure": values["no_at_fault_collisions"] < 1.0 - 1e-12,
        "DAC failure": values["drivable_area_compliance"] < 1.0 - 1e-12,
        "DDC failure": values["driving_direction_compliance"] < 1.0 - 1e-12,
        "TTC failure": values["time_to_collision_within_bound"] < 1.0 - 1e-12,
        "progress failure": values["ego_progress"] < 0.95,
        "comfort failure": values["comfort"] < 1.0 - 1e-12,
    }
    flags["multi-metric joint failure"] = sum(flags.values()) >= 2
    return flags


def _write_figures(frame: pd.DataFrame, topk: pd.DataFrame, failure: pd.DataFrame, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6.2, 5.2))
    plt.hexbin(frame["O_B"], frame["V_B"], gridsize=55, mincnt=1, cmap="viridis")
    plt.plot([0, 1], [0, 1], "k--", linewidth=1)
    plt.xlabel("Oracle@64 true PDMS")
    plt.ylabel("Selected true PDMS")
    plt.colorbar(label="scene count")
    plt.tight_layout(); plt.savefig(output / "selected_vs_oracle.png", dpi=180); plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4))
    axes[0].hist(frame["G_B"], bins=60, color="#35618f")
    axes[0].set_xlabel("selection regret"); axes[0].set_ylabel("scenes")
    values = np.sort(frame["G_B"].to_numpy())
    axes[1].plot(values, np.arange(1, len(values) + 1) / len(values))
    axes[1].set_xlabel("selection regret"); axes[1].set_ylabel("CDF")
    fig.tight_layout(); fig.savefig(output / "regret_histogram_cdf.png", dpi=180); plt.close(fig)

    grouped = frame.groupby("N_0.95", observed=True)["G_B"].agg(["mean", "median", "count"]).reset_index()
    plt.figure(figsize=(7, 4.5))
    plt.scatter(grouped["N_0.95"], grouped["mean"], s=np.sqrt(grouped["count"]) * 2, label="mean")
    plt.plot(grouped["N_0.95"], grouped["median"], color="#b24c3e", label="median")
    plt.xlabel("N(true PDMS >= 0.95)"); plt.ylabel("selection regret")
    plt.legend(); plt.tight_layout(); plt.savefig(output / "n095_vs_regret.png", dpi=180); plt.close()

    ranks = np.sort(frame["oracle_predicted_rank"].to_numpy())
    plt.figure(figsize=(6.2, 4.2)); plt.step(ranks, np.arange(1, len(ranks) + 1) / len(ranks), where="post")
    plt.xscale("log", base=2); plt.xticks([1, 2, 4, 8, 16, 32, 64], [1, 2, 4, 8, 16, 32, 64])
    plt.xlabel("predicted rank of exact oracle index"); plt.ylabel("CDF")
    plt.tight_layout(); plt.savefig(output / "oracle_predicted_rank_cdf.png", dpi=180); plt.close()

    plt.figure(figsize=(6.2, 4.2)); plt.plot(topk["K"], topk["mean_true_pdms"], marker="o")
    plt.xscale("log", base=2); plt.xticks(topk["K"], topk["K"])
    plt.xlabel("K by predicted scorer rank"); plt.ylabel("Top-K oracle true PDMS")
    plt.tight_layout(); plt.savefig(output / "topk_oracle_curve.png", dpi=180); plt.close()

    gaps = [float((1 - frame["O_B"]).mean()), float(frame["G_B"].mean())]
    plt.figure(figsize=(5.5, 4.2)); plt.bar(["coverage gap", "ranking gap"], gaps, color=["#6c9f58", "#b24c3e"])
    plt.ylabel("mean gap on 0-1 PDMS scale"); plt.tight_layout()
    plt.savefig(output / "gap_decomposition.png", dpi=180); plt.close()

    success = frame["scorer_success_eps01"].astype(bool)
    keys = ["N_0.95", "high_quality_cluster_count", "oracle_second_true_margin", "pairwise_ade", "endpoint_spread", "spearman"]
    plot = pd.DataFrame({
        "success": [frame.loc[success, key].mean() for key in keys],
        "failure": [frame.loc[~success, key].mean() for key in keys],
    }, index=keys)
    plot.plot(kind="bar", figsize=(9, 4.8)); plt.ylabel("raw feature mean")
    plt.tight_layout(); plt.savefig(output / "scorer_success_failure_features.png", dpi=180); plt.close()

    if len(failure):
        pivot = failure.pivot(index="failure_type", columns="limitation", values="share").fillna(0)
        pivot.plot(kind="bar", figsize=(10, 4.8), color=["#6c9f58", "#b24c3e"])
        plt.ylabel("share within failure type"); plt.tight_layout()
        plt.savefig(output / "failure_type_candidate_ranker_share.png", dpi=180); plt.close()


def _subsets(frame: pd.DataFrame) -> List[Tuple[str, np.ndarray]]:
    values = frame["V_B"].to_numpy()
    result = [(f"V_B<{threshold:.2f}", values < threshold) for threshold in (0.50, 0.80, 0.90, 0.95)]
    order = np.argsort(values, kind="stable")
    for fraction in (0.05, 0.10, 0.20):
        mask = np.zeros(len(frame), dtype=bool)
        mask[order[: max(1, int(math.ceil(len(frame) * fraction)))]] = True
        result.append((f"bottom_{int(fraction * 100)}pct", mask))
    return result


def main() -> None:
    args = parse_args()
    summary_path = args.output_dir / "base_audit_summary.json"
    if summary_path.exists() and not args.overwrite:
        print(summary_path.read_text(), end="")
        return
    proposals = load_proposal_pickle(args.proposal_pickle)
    status = _load_status(args.status_replay_root)
    with np.load(args.candidate_matrix, allow_pickle=False) as archive:
        tokens = archive["tokens"].astype(str)
        log_names = archive["log_names"].astype(str)
        true_scores = archive["candidate_scores"].astype(np.float64)
        predicted_scores = archive["predicted_scores"].astype(np.float64)
        selected_indices = archive["selected_indices"].astype(np.int64)
        oracle_indices = archive["oracle_indices"].astype(np.int64)
        factors = archive["candidate_factors"].astype(np.float64)
        factor_names = archive["candidate_factor_names"].astype(str).tolist()
    if factor_names != list(FACTOR_NAMES):
        raise RuntimeError(f"Unexpected true-factor order: {factor_names}")
    if set(tokens) != set(proposals):
        raise RuntimeError("Proposal and score matrices have different token inventories")
    if args.max_scenes:
        tokens, log_names = tokens[: args.max_scenes], log_names[: args.max_scenes]
        true_scores, predicted_scores = true_scores[: args.max_scenes], predicted_scores[: args.max_scenes]
        selected_indices, oracle_indices, factors = selected_indices[: args.max_scenes], oracle_indices[: args.max_scenes], factors[: args.max_scenes]
    if true_scores.shape != (len(tokens), 64) or factors.shape != (len(tokens), 64, 7):
        raise RuntimeError("Expected complete 64-candidate score/factor matrices")
    if not np.isfinite(true_scores).all() or true_scores.min() < -1e-7 or true_scores.max() > 1 + 1e-7:
        raise RuntimeError("True PDM scores are invalid")

    rows: List[Dict[str, object]] = []
    failures: List[Dict[str, object]] = []
    for scene_index, token in enumerate(tokens):
        bank = np.asarray(proposals[str(token)]["proposals"], dtype=np.float64)
        metrics = scene_metrics(true_scores[scene_index], predicted_scores[scene_index], bank)
        if int(metrics["selected_index"]) != int(selected_indices[scene_index]):
            raise RuntimeError(f"Selected-index mismatch for {token}")
        if int(metrics["oracle_index"]) != int(oracle_indices[scene_index]):
            raise RuntimeError(f"Oracle-index mismatch for {token}")
        selected = int(metrics["selected_index"]); oracle = int(metrics["oracle_index"])
        delta = bank[selected, :, :2] - bank[oracle, :, :2]
        row: Dict[str, object] = {
            "token": str(token),
            "log_name": str(log_names[scene_index]),
            "physical_log_name": physical_log_name(str(log_names[scene_index])),
            **metrics,
            "selected_oracle_ade": float(np.linalg.norm(delta, axis=-1).mean()),
            "selected_oracle_fde": float(np.linalg.norm(delta[-1])),
            "scorer_success_eps005": float(metrics["V_B"] >= metrics["O_B"] - 0.005),
            "scorer_success_eps01": float(metrics["V_B"] >= metrics["O_B"] - 0.01),
            "scorer_success_eps02": float(metrics["V_B"] >= metrics["O_B"] - 0.02),
            "selected_true_ge_095": float(metrics["V_B"] >= 0.95),
            "oracle_predicted_rank_le_3": float(metrics["oracle_predicted_rank"] <= 3),
            "oracle_predicted_rank_le_8": float(metrics["oracle_predicted_rank"] <= 8),
        }
        for factor_index, factor_name in enumerate(FACTOR_NAMES):
            row[f"selected_true_{factor_name}"] = float(factors[scene_index, selected, factor_index])
            row[f"oracle_true_{factor_name}"] = float(factors[scene_index, oracle, factor_index])
        if str(token) in status:
            row.update(_status_features(status[str(token)]))
        flags = _failure_flags(factors[scene_index, selected])
        for name, flag in flags.items():
            row[f"failure__{name}"] = float(flag)
            if flag:
                failures.extend(
                    [
                        {"failure_type": name, "limitation": "candidate_limited", "value": float(metrics["is_candidate_limited"])},
                        {"failure_type": name, "limitation": "ranker_limited", "value": float(metrics["is_ranker_limited"])},
                    ]
                )
        rows.append(row)
        if (scene_index + 1) % 1000 == 0:
            print(json.dumps({"analyzed_scenes": scene_index + 1, "total": len(tokens)}), flush=True)
    frame = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.output_dir / "scene_metrics.parquet", index=False)

    # Consistency identities are hard failures, not report footnotes.
    tolerance = 1e-10
    selected_mean = float(frame["V_B"].mean())
    oracle_mean = float(frame["O_B"].mean())
    if abs(float(frame["top1_oracle"].mean()) - selected_mean) > tolerance:
        raise RuntimeError("TopKOracle(1) != selected mean")
    if abs(float(frame["top64_oracle"].mean()) - oracle_mean) > tolerance:
        raise RuntimeError("TopKOracle(64) != oracle mean")
    coverage_gap = float((1.0 - frame["O_B"]).mean())
    ranking_gap = float(frame["G_B"].mean())
    total_gap = float((1.0 - frame["V_B"]).mean())
    if abs(coverage_gap + ranking_gap - total_gap) > tolerance:
        raise RuntimeError("coverage_gap + ranking_gap != total_gap")

    topk_rows = []
    for k in TOP_K_VALUES:
        values = frame[f"top{k}_oracle"].to_numpy()
        stats = bootstrap_mean(values, n_bootstrap=args.bootstrap_replicates, seed=args.seed + k)
        topk_rows.append({"K": k, "mean_true_pdms": float(values.mean()), **{key: value for key, value in stats.items() if key != "mean"}})
    topk = pd.DataFrame(topk_rows)
    topk.to_csv(args.output_dir / "topk_oracle.csv", index=False)

    subset_rows = []
    for name, mask in _subsets(frame):
        subset = frame.loc[mask]
        record: Dict[str, object] = {"subset": name, "scene_count": len(subset), "share_of_all": float(mask.mean())}
        for key in ("is_candidate_limited", "is_sparse_good", "is_ranker_limited", "is_dense_rank_failure", "is_saturated"):
            record[f"share_{key[3:]}"] = float(subset[key].mean()) if len(subset) else float("nan")
        for failure_class, count in subset["failure_class"].value_counts().items():
            record[f"exclusive_{failure_class}"] = int(count)
        record.update({
            "P_O_B_ge_095": float((subset["O_B"] >= 0.95).mean()),
            "P_O_B_lt_090": float((subset["O_B"] < 0.90).mean()),
            "E_N_095": float(subset["N_0.95"].mean()),
            "E_G_B": float(subset["G_B"].mean()),
        })
        subset_rows.append(record)
    pd.DataFrame(subset_rows).to_csv(args.output_dir / "subset_summary.csv", index=False)

    sensitivity_rows = []
    for oracle_bad in (0.85, 0.90, 0.95):
        for good in (0.90, 0.95, 0.99):
            counts = (true_scores >= good).sum(axis=1)
            for sparse_max in (1, 2, 4):
                sensitivity_rows.append({
                    "oracle_bad_threshold": oracle_bad,
                    "good_threshold": good,
                    "sparse_max_count": sparse_max,
                    "candidate_limited_share": float((frame["O_B"] < oracle_bad).mean()),
                    "sparse_good_share": float(((frame["O_B"] >= good) & (counts <= sparse_max)).mean()),
                })
    pd.DataFrame(sensitivity_rows).to_csv(args.output_dir / "threshold_sensitivity.csv", index=False)

    failure_frame = pd.DataFrame(failures)
    failure_summary = (
        failure_frame.groupby(["failure_type", "limitation"], observed=True)["value"]
        .agg(share="mean", scene_occurrences="size").reset_index()
    )
    failure_summary.to_csv(args.output_dir / "failure_type_limitation.csv", index=False)

    feature_columns = [
        "O_B", "N_0.90", "N_0.95", "N_0.99", "high_quality_cluster_count",
        "oracle_second_true_margin", "predicted_top1_top2_margin", "oracle_predicted_rank",
        "M_B", "median_B", "std_B", "pairwise_ade", "pairwise_fde", "endpoint_spread",
        "spearman", "kendall_tau_b", "selected_oracle_ade", "selected_oracle_fde",
        "ego_speed", "ego_acceleration_norm", "command_index",
    ] + [f"selected_true_{name}" for name in FACTOR_NAMES[:-1]]
    feature_columns = [name for name in feature_columns if name in frame]
    success = frame["scorer_success_eps01"].astype(bool).to_numpy()
    comparison_rows = []
    for feature_index, name in enumerate(feature_columns):
        left = frame.loc[success, name].to_numpy(dtype=np.float64)
        right = frame.loc[~success, name].to_numpy(dtype=np.float64)
        ci = _bootstrap_difference(left, right, replicates=args.bootstrap_replicates, seed=args.seed + 1000 + feature_index)
        comparison_rows.append({
            "feature": name,
            "success_mean": float(np.nanmean(left)),
            "failure_mean": float(np.nanmean(right)),
            "mean_difference_success_minus_failure": float(np.nanmean(left) - np.nanmean(right)),
            "standardized_effect_size_cohen_d": _cohen_d(left, right),
            **ci,
        })
    comparison = pd.DataFrame(comparison_rows).sort_values("standardized_effect_size_cohen_d", key=lambda x: x.abs(), ascending=False)
    comparison.to_csv(args.output_dir / "scorer_success_analysis.csv", index=False)

    X = frame[feature_columns].replace([np.inf, -np.inf], np.nan)
    y = success.astype(np.int64)
    groups = frame["log_name"].astype(str).to_numpy()
    if len(np.unique(groups)) < 5 or len(np.unique(y)) < 2:
        raise RuntimeError("Grouped 5-fold scorer analysis lacks groups or both labels")
    cv_rows = []; logistic_importances = []; tree_importances = []; tree_rules = []
    for fold, (train, test) in enumerate(GroupKFold(n_splits=5).split(X, y, groups)):
        logistic = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(C=1.0, penalty="l2", class_weight="balanced", max_iter=2000, random_state=args.seed)),
        ])
        tree = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", DecisionTreeClassifier(max_depth=3, min_samples_leaf=50, class_weight="balanced", random_state=args.seed)),
        ])
        for model_name, model in (("logistic_regression_l2", logistic), ("decision_tree_depth3", tree)):
            model.fit(X.iloc[train], y[train])
            probability = model.predict_proba(X.iloc[test])[:, 1]
            prediction = probability >= 0.5
            cv_rows.append({
                "model": model_name,
                "fold": fold,
                "train_scenes": len(train),
                "test_scenes": len(test),
                "test_logs": len(np.unique(groups[test])),
                "auc": float(roc_auc_score(y[test], probability)),
                "balanced_accuracy": float(balanced_accuracy_score(y[test], prediction)),
            })
        logistic_importances.append(np.abs(logistic.named_steps["model"].coef_[0]))
        tree_importances.append(tree.named_steps["model"].feature_importances_)
        tree_rules.append(export_text(tree.named_steps["model"], feature_names=feature_columns))
    cv_frame = pd.DataFrame(cv_rows)
    cv_frame.to_csv(args.output_dir / "scorer_grouped_cv.csv", index=False)
    importance = pd.DataFrame({
        "feature": feature_columns,
        "logistic_mean_abs_standardized_coefficient": np.mean(logistic_importances, axis=0),
        "tree_mean_importance": np.mean(tree_importances, axis=0),
    }).sort_values("logistic_mean_abs_standardized_coefficient", ascending=False)
    importance.to_csv(args.output_dir / "scorer_feature_importance.csv", index=False)
    (args.output_dir / "decision_tree_rules.txt").write_text("\n\n".join(f"FOLD {i}\n{rule}" for i, rule in enumerate(tree_rules)))

    _write_figures(frame, topk, failure_summary, args.output_dir / "figures")
    low = frame["V_B"] < 0.90
    major_values = {
        "selected_pdms": frame["V_B"].to_numpy(),
        "oracle64_pdms": frame["O_B"].to_numpy(),
        "mean_candidate_pdms": frame["M_B"].to_numpy(),
        "coverage_gap": 1.0 - frame["O_B"].to_numpy(),
        "ranking_gap": frame["G_B"].to_numpy(),
        "total_gap": 1.0 - frame["V_B"].to_numpy(),
    }
    bootstrap = {
        name: {
            "scene": bootstrap_mean(values, n_bootstrap=args.bootstrap_replicates, seed=args.seed + index),
            "cluster_log": cluster_bootstrap_mean(values, frame["log_name"], n_bootstrap=args.bootstrap_replicates, seed=args.seed + index),
        }
        for index, (name, values) in enumerate(major_values.items())
    }
    summary = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "FULL" if len(frame) == 12_146 else "PARTIAL",
        "scene_count": len(frame),
        "log_count": int(frame["log_name"].nunique()),
        "proposal_count": 64,
        "proposal_pickle": str(args.proposal_pickle.resolve()),
        "proposal_pickle_sha256": sha256_file(args.proposal_pickle),
        "candidate_matrix": str(args.candidate_matrix.resolve()),
        "candidate_matrix_sha256": sha256_file(args.candidate_matrix),
        "status_replay_root": str(args.status_replay_root.resolve()) if args.status_replay_root else None,
        "selected_pdms": selected_mean,
        "oracle64_pdms": oracle_mean,
        "mean_candidate_pdms": float(frame["M_B"].mean()),
        "mean_candidate_counts": {threshold: float(frame[f"N_{threshold}"].mean()) for threshold in ("0.50", "0.80", "0.90", "0.95", "0.99")},
        "coverage_gap": coverage_gap,
        "ranking_gap": ranking_gap,
        "total_gap": total_gap,
        "ranking_gap_fraction": safe_divide(ranking_gap, total_gap),
        "coverage_gap_fraction": safe_divide(coverage_gap, total_gap),
        "ranking_coverage_ratio": safe_divide(ranking_gap, coverage_gap),
        "low_vb_scene_count": int(low.sum()),
        "P_O_B_ge_095_given_V_B_lt_090": float((frame.loc[low, "O_B"] >= 0.95).mean()),
        "P_O_B_lt_090_given_V_B_lt_090": float((frame.loc[low, "O_B"] < 0.90).mean()),
        "E_N_095_given_V_B_lt_090": float(frame.loc[low, "N_0.95"].mean()),
        "E_G_B_given_V_B_lt_090": float(frame.loc[low, "G_B"].mean()),
        "candidate_limited_share_low": float(frame.loc[low, "is_candidate_limited"].mean()),
        "ranker_limited_share_low": float(frame.loc[low, "is_ranker_limited"].mean()),
        "scorer_success_eps01_rate": float(frame["scorer_success_eps01"].mean()),
        "topk_oracle": {str(row.K): float(row.mean_true_pdms) for row in topk.itertuples()},
        "grouped_cv_mean": cv_frame.groupby("model")[["auc", "balanced_accuracy"]].mean().to_dict(orient="index"),
        "bootstrap": bootstrap,
        "consistency": {
            "top1_equals_selected": True,
            "top64_equals_oracle": True,
            "gap_identity_abs_error": abs(coverage_gap + ranking_gap - total_gap),
            "no_silently_dropped_token": len(frame) == len(tokens),
            "all_true_scores_in_range": True,
        },
        "analysis_scope_note": "Associations are descriptive and are not causal effects.",
        "failure_thresholds": {
            "binary_compliance": "subscore < 1",
            "progress": "ego_progress < 0.95",
            "joint": "at least two failure flags",
        },
    }
    atomic_json(summary_path, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
