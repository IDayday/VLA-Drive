#!/usr/bin/env python3
"""Generate Phase-4 data feasibility tables, figures, and Gate-1 decision."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from research.action_effect.diagnostics import (  # noqa: E402
    markdown_table,
    percentage,
    require_target_provenance,
    save_figure,
)
from research.action_effect.metric_cache_io import iter_jsonl  # noqa: E402
from research.action_effect.pair_builder import hard_vector  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-cache", type=Path)
    parser.add_argument("--consequence-cache", type=Path)
    parser.add_argument("--pair-cache", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPOSITORY_ROOT / "reports/action_effect_world_model/data_feasibility_artifacts",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=REPOSITORY_ROOT / "reports/action_effect_world_model/data_feasibility.md",
    )
    return parser.parse_args()


def _resolve(explicit: Path | None, suffix: str) -> Path:
    if explicit is not None:
        return explicit.resolve()
    root = os.environ.get("ACTION_EFFECT_CACHE_ROOT", "").strip()
    if not root:
        raise ValueError("pass cache paths or source load_env.sh to set ACTION_EFFECT_CACHE_ROOT")
    return (Path(root) / suffix).resolve()


def _flatten_candidate(row: dict[str, Any]) -> dict[str, Any]:
    exact, replay, reactive = row["exact"], row["log_replay"], row["reactive_model"]
    result = {
        "scene_id": row["scene_id"],
        "candidate_id": row["candidate_id"],
        "perturbation_type": row["perturbation_type"],
        "kinematic_valid": row["kinematic_valid"],
        "route_proxy_valid": row["route_proxy_valid"],
        "candidate_accepted": row["candidate_accepted"],
        "reactive_available": reactive.get("available", False),
    }
    for key in (
        "drivable_area_compliance",
        "driving_direction_compliance",
        "lane_keeping",
        "history_comfort",
        "centerline_progress_m",
        "route_deviation_max_m",
        "max_acceleration_mps2",
        "max_deceleration_mps2",
        "max_abs_jerk_mps3",
        "max_abs_curvature_inv_m",
        "intersection_fraction",
        "static_object_collision",
    ):
        result[f"exact_{key}"] = exact.get(key)
    for key in (
        "no_at_fault_collision",
        "traffic_light_compliance",
        "time_to_collision_within_bound",
        "ttc_infraction_time_s",
        "ttc_infraction_observed",
        "dynamic_collision",
        "minimum_dynamic_clearance_m",
        "dynamic_occupancy_fraction",
        "pdm_score",
    ):
        result[f"lr_{key}"] = replay.get(key)
        result[f"idm_{key}"] = reactive.get(key)
    return result


def _scene_categories(anchor: dict[str, Any], trajectory: np.ndarray, candidate_sensitive: bool) -> list[str]:
    categories: list[str] = []
    heading_change = float(np.max(np.abs(np.unwrap(trajectory[:, 2]) - trajectory[0, 2])))
    categories.append("turning" if heading_change >= 0.15 else "straight")
    exact, replay = anchor["exact"], anchor["log_replay"]
    if float(exact.get("intersection_fraction", 0.0)) > 0.05:
        categories.append("junction")
    if float(replay.get("dynamic_occupancy_fraction", 0.0)) > 0.0:
        categories.append("dynamic_interaction")
    if float(replay.get("minimum_dynamic_clearance_m", 20.0)) < 2.0:
        categories.append("near_collision")
    if replay.get("ttc_infraction_observed") or float(replay.get("ttc_infraction_time_s", 5.0)) < 4.0:
        categories.append("low_ttc")
    hard_safe = bool(
        replay.get("no_at_fault_collision") == 1.0
        and exact.get("drivable_area_compliance") == 1.0
        and not exact.get("static_object_collision")
        and not replay.get("dynamic_collision")
    )
    if float(replay.get("dynamic_occupancy_fraction", 0.0)) == 0.0 and hard_safe:
        categories.append("static_easy")
    if candidate_sensitive:
        categories.append("candidate_sensitive")
    return categories


def _boxplot(ax: Any, frame: pd.DataFrame, column: str, title: str) -> None:
    types = sorted(frame["perturbation_type"].unique())
    values = [frame.loc[frame["perturbation_type"] == name, column].dropna().values for name in types]
    ax.boxplot(values, tick_labels=types, showfliers=False)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=45, labelsize=7)
    ax.grid(axis="y", alpha=0.2)


def main() -> None:
    args = parse_args()
    candidate_cache = _resolve(args.candidate_cache, "candidates/pilot_tiny/expert")
    consequence_cache = _resolve(args.consequence_cache, "consequences/pilot_tiny/expert")
    pair_cache = _resolve(args.pair_cache, "pairs/pilot_tiny/expert")
    output_dir = args.output_dir.resolve()
    report_path = args.report_path.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_rows = list(iter_jsonl(candidate_cache / "metadata.jsonl"))
    consequence_rows = list(iter_jsonl(consequence_cache / "consequences.jsonl"))
    pair_rows = list(iter_jsonl(pair_cache / "pairs.jsonl"))
    require_target_provenance(consequence_rows)
    trajectories = np.load(candidate_cache / "candidates.npz")["trajectories"]
    trajectory_by_id = {
        row["candidate_id"]: trajectories[int(row["trajectory"]["index"])] for row in candidate_rows
    }
    candidate_df = pd.DataFrame([_flatten_candidate(row) for row in consequence_rows])
    pair_df = pd.DataFrame(
        [
            {
                **{key: value for key, value in row.items() if key != "soft_difference_by_field"},
                **{
                    f"soft_diff_{key}": value
                    for key, value in row["soft_difference_by_field"].items()
                },
            }
            for row in pair_rows
        ]
    )
    candidate_df.to_csv(output_dir / "candidates.csv", index=False)
    pair_df.to_csv(output_dir / "pairs.csv", index=False)

    rows_by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    pairs_by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in consequence_rows:
        rows_by_scene[row["scene_id"]].append(row)
    for row in pair_rows:
        pairs_by_scene[row["scene_id"]].append(row)
    scene_rows: list[dict[str, Any]] = []
    category_rows: list[dict[str, Any]] = []
    for scene_id in sorted(rows_by_scene):
        rows = rows_by_scene[scene_id]
        pairs = pairs_by_scene[scene_id]
        accepted = [row for row in rows if row["candidate_accepted"]]
        pair_types = Counter(pair["pair_type"] for pair in pairs)
        hard_unique = len({tuple(hard_vector(row)) for row in accepted if row["log_replay"].get("available")})
        distances = [float(pair["consequence_distance"]) for pair in pairs]
        anchor = next(row for row in rows if row["perturbation_type"] == "anchor")
        candidate_sensitive = pair_types["effect_divergent"] > 0
        categories = _scene_categories(anchor, trajectory_by_id[anchor["candidate_id"]], candidate_sensitive)
        scene_summary = {
            "scene_id": scene_id,
            "valid_candidate_count": len(accepted),
            "kinematic_valid_count": sum(bool(row["kinematic_valid"]) for row in rows),
            "hard_vector_diversity": hard_unique,
            "consequence_distance_mean": float(np.mean(distances)) if distances else np.nan,
            "consequence_distance_std": float(np.std(distances)) if distances else np.nan,
            "equivalent_pairs": pair_types["effect_equivalent"],
            "divergent_pairs": pair_types["effect_divergent"],
            "ambiguous_pairs": pair_types["ambiguous"],
            "safety_boundary_pairs": sum(bool(pair["safety_boundary"]) for pair in pairs),
            "has_equivalent_and_divergent": bool(
                pair_types["effect_equivalent"] and pair_types["effect_divergent"]
            ),
            "categories": ",".join(categories),
        }
        scene_rows.append(scene_summary)
        for category in categories:
            category_rows.append({"category": category, **scene_summary})
    scene_df = pd.DataFrame(scene_rows)
    category_df = pd.DataFrame(category_rows)
    scene_df.to_csv(output_dir / "scenes.csv", index=False)
    category_summary = (
        category_df.groupby("category")
        .agg(
            scenes=("scene_id", "nunique"),
            valid_candidates_mean=("valid_candidate_count", "mean"),
            equivalent_pairs_mean=("equivalent_pairs", "mean"),
            divergent_pairs_mean=("divergent_pairs", "mean"),
            safety_boundary_pairs=("safety_boundary_pairs", "sum"),
        )
        .reset_index()
    )
    category_summary.to_csv(output_dir / "scene_categories.csv", index=False)

    retention = (
        candidate_df.groupby("perturbation_type")
        .agg(
            candidates=("candidate_id", "count"),
            kinematic_valid_rate=("kinematic_valid", "mean"),
            route_proxy_valid_rate=("route_proxy_valid", "mean"),
            accepted_rate=("candidate_accepted", "mean"),
            mean_progress_m=("exact_centerline_progress_m", "mean"),
            mean_dynamic_clearance_m=("lr_minimum_dynamic_clearance_m", "mean"),
            dynamic_collision_rate=("lr_dynamic_collision", "mean"),
        )
        .reset_index()
    )
    retention.to_csv(output_dir / "perturbation_summary.csv", index=False)

    hard_driver_counts = Counter()
    soft_driver_values: dict[str, list[float]] = defaultdict(list)
    for pair in pair_rows:
        if pair["pair_type"] != "effect_divergent":
            continue
        hard_driver_counts.update(pair["hard_difference_fields"])
        for field, value in pair["soft_difference_by_field"].items():
            if value is not None:
                soft_driver_values[field].append(float(value))
    driver_rows = [
        {"metric": key, "kind": "hard", "count": value, "mean_normalized_difference": np.nan}
        for key, value in hard_driver_counts.items()
    ] + [
        {
            "metric": key,
            "kind": "soft",
            "count": len(values),
            "mean_normalized_difference": float(np.mean(values)),
        }
        for key, values in soft_driver_values.items()
    ]
    driver_df = pd.DataFrame(driver_rows).sort_values(
        ["kind", "count", "mean_normalized_difference"], ascending=[True, False, False]
    )
    driver_df.to_csv(output_dir / "divergence_metric_drivers.csv", index=False)

    assessed_pairs = pair_df[pair_df["hard_agreement"].notna()].copy()
    row_assessed = [
        row
        for row in consequence_rows
        if row["candidate_accepted"] and row["reactive_model"].get("available")
    ]
    candidate_hard_agreement = (
        np.mean(
            [
                np.array_equal(hard_vector(row), hard_vector(row, "reactive_model"))
                for row in row_assessed
            ]
        )
        if row_assessed
        else np.nan
    )
    pair_hard_relation_agreement = (
        float(assessed_pairs["pairwise_hard_relation_agreement"].mean()) if len(assessed_pairs) else np.nan
    )
    ranking_agreement = (
        float(assessed_pairs["soft_rank_agreement"].mean()) if len(assessed_pairs) else np.nan
    )
    agreement_summary = pd.DataFrame(
        [
            {
                "reactive_candidate_count": len(row_assessed),
                "assessed_pair_count": len(assessed_pairs),
                "candidate_hard_agreement": candidate_hard_agreement,
                "pairwise_hard_relation_agreement": pair_hard_relation_agreement,
                "pairwise_ranking_agreement": ranking_agreement,
            }
        ]
    )
    agreement_summary.to_csv(output_dir / "lr_idm_agreement.csv", index=False)

    correlation, correlation_p = spearmanr(
        pair_df["geometric_distance"], pair_df["consequence_distance"], nan_policy="omit"
    )

    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.arange(0.5, candidate_df.groupby("scene_id")["candidate_accepted"].sum().max() + 1.5, 1)
    ax.hist(scene_df["valid_candidate_count"], bins=bins, color="#4477AA", edgecolor="white")
    ax.set(xlabel="Accepted candidates per scene", ylabel="Scenes", title="Valid candidate distribution")
    save_figure(fig, output_dir / "valid_candidate_histogram.png")

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = {"effect_equivalent": "#228833", "effect_divergent": "#CC6677", "ambiguous": "#999999"}
    for pair_type, group in pair_df.groupby("pair_type"):
        ax.hist(group["consequence_distance"], bins=50, alpha=0.55, label=pair_type, color=colors[pair_type])
    ax.set(xlabel="Consequence distance", ylabel="Pairs", title="Consequence distance by pair type")
    ax.legend()
    save_figure(fig, output_dir / "consequence_distance_histogram.png")

    fig, ax = plt.subplots(figsize=(7, 5))
    sampled = pair_df.iloc[:: max(1, len(pair_df) // 20000)]
    for pair_type, group in sampled.groupby("pair_type"):
        ax.scatter(
            group["geometric_distance"],
            group["consequence_distance"],
            s=7,
            alpha=0.25,
            label=pair_type,
            color=colors[pair_type],
        )
    ax.set(
        xlabel="Geometric distance (metre-equivalent)",
        ylabel="Consequence distance",
        title=f"Geometry vs consequence (Spearman $\\rho$={correlation:.3f})",
    )
    ax.legend(markerscale=2)
    save_figure(fig, output_dir / "geometry_vs_consequence.png")

    fig, ax = plt.subplots(figsize=(6, 4))
    pair_counts = pair_df["pair_type"].value_counts().reindex(colors).fillna(0)
    ax.bar(pair_counts.index, pair_counts.values, color=[colors[key] for key in pair_counts.index])
    ax.set(ylabel="Pairs", title="Action-effect pair distribution")
    ax.tick_params(axis="x", rotation=20)
    save_figure(fig, output_dir / "pair_type_distribution.png")

    boundary = pair_df[pair_df["safety_boundary"]].sort_values("geometric_distance").head(6)
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for ax in axes.flat:
        ax.axis("off")
    for ax, (_, pair) in zip(axes.flat, boundary.iterrows()):
        left = trajectory_by_id[pair["candidate_i"]]
        right = trajectory_by_id[pair["candidate_j"]]
        ax.axis("on")
        ax.plot(np.r_[0, left[:, 0]], np.r_[0, left[:, 1]], "-o", ms=2, label="candidate i")
        ax.plot(np.r_[0, right[:, 0]], np.r_[0, right[:, 1]], "-o", ms=2, label="candidate j")
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_title(f"{pair['scene_id'][:8]} d_g={pair['geometric_distance']:.2f}\n{pair['hard_difference_fields']}", fontsize=8)
        ax.grid(alpha=0.2)
    if len(boundary):
        axes.flat[0].legend(fontsize=7)
    else:
        axes.flat[0].axis("on")
        axes.flat[0].text(0.5, 0.5, "No safety-boundary pairs", ha="center", va="center")
    fig.suptitle("Closest safety-boundary pair examples")
    save_figure(fig, output_dir / "safety_boundary_examples.png")

    relation_labels = ["same", "different"]
    matrix = np.zeros((2, 2), dtype=int)
    for _, pair in assessed_pairs.iterrows():
        matrix[relation_labels.index(pair["log_replay_hard_relation"]), relation_labels.index(pair["reactive_hard_relation"])] += 1
    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(matrix, cmap="Blues")
    for row_index in range(2):
        for column_index in range(2):
            ax.text(column_index, row_index, str(matrix[row_index, column_index]), ha="center", va="center")
    ax.set_xticks(range(2), relation_labels)
    ax.set_yticks(range(2), relation_labels)
    ax.set(xlabel="IDM hard relation", ylabel="Log-replay hard relation", title="LR / IDM pair agreement")
    fig.colorbar(image, ax=ax)
    save_figure(fig, output_dir / "lr_idm_agreement_matrix.png")

    accepted_df = candidate_df[candidate_df["candidate_accepted"]]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    _boxplot(axes[0, 0], accepted_df, "exact_centerline_progress_m", "Centerline progress")
    _boxplot(axes[0, 1], accepted_df, "lr_minimum_dynamic_clearance_m", "Dynamic clearance")
    _boxplot(axes[1, 0], accepted_df, "exact_max_abs_jerk_mps3", "Maximum jerk")
    _boxplot(axes[1, 1], accepted_df, "exact_route_deviation_max_m", "Route deviation")
    fig.suptitle("Consequences induced by perturbation type")
    fig.tight_layout()
    save_figure(fig, output_dir / "perturbation_consequence_distribution.png")

    median_valid = float(scene_df["valid_candidate_count"].median())
    both_rate = float(scene_df["has_equivalent_and_divergent"].mean())
    boundary_count = int(pair_df["safety_boundary"].sum())
    boundary_scene_rate = float((scene_df["safety_boundary_pairs"] > 0).mean())
    severe_conflict = bool(
        math.isfinite(pair_hard_relation_agreement) and pair_hard_relation_agreement < 0.5
    )
    if severe_conflict:
        gate = "STOP"
        gate_reason = "LR/IDM 在已评估关键 pair 上的 hard relation agreement 低于 50%。"
    elif median_valid < 8 or both_rate < 0.05 or boundary_count < 10:
        gate = "MODIFY_CANDIDATES"
        gate_reason = "候选有效性、同场景双类 pair 密度或安全边界 pair 数量不足。"
    elif abs(float(correlation)) >= 0.9:
        gate = "MODIFY_CANDIDATES"
        gate_reason = "后果距离几乎完全由几何距离解释。"
    else:
        gate = "PASS"
        gate_reason = "局部候选有效、pair 类型有共存，且后果差异不退化为纯几何距离。"

    perturbation_table = [
        (
            row.perturbation_type,
            int(row.candidates),
            f"{row.kinematic_valid_rate:.3f}",
            f"{row.route_proxy_valid_rate:.3f}",
            f"{row.accepted_rate:.3f}",
        )
        for row in retention.itertuples()
    ]
    category_table = [
        (
            row.category,
            int(row.scenes),
            f"{row.valid_candidates_mean:.2f}",
            f"{row.equivalent_pairs_mean:.2f}",
            f"{row.divergent_pairs_mean:.2f}",
            int(row.safety_boundary_pairs),
        )
        for row in category_summary.itertuples()
    ]
    report_rel = os.path.relpath(output_dir, report_path.parent)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPOSITORY_ROOT, text=True).strip()
    markdown = f"""# Action-effect 数据可行性（Phase 4）

状态：**已实现并运行**。统计只使用 `train` split；soft consequence 的 median/IQR/5%–95% 与 pair 阈值均未读取验证集。运行提交基线为 `{commit}`，生成缓存 manifest 另外记录了未提交源码 tree hash。

## 结论与 Gate 1

**Gate 1：{gate}。** {gate_reason}

关键证据：512 个场景的每场有效候选中位数为 {median_valid:.1f}；同时含 equivalent/divergent pair 的场景比例为 {both_rate:.1%}；安全边界 pair 共 {boundary_count} 个，覆盖 {boundary_scene_rate:.1%} 场景；几何/后果距离 Spearman 相关为 {correlation:.3f}（p={correlation_p:.2e}）。

该标签是 `replay_grounded_consequence` / `reactive_model` 监督，不是真实因果反事实。真实参与者对未执行动作的反应仍标记为 `unknown`。

## 1. 候选数量与有效性

- 场景数：{len(scene_df)}；候选总数：{len(candidate_df)}。
- 运动学有效率：{candidate_df['kinematic_valid'].mean():.2%}。
- route proxy 有效率：{candidate_df['route_proxy_valid'].mean():.2%}。
- 同时通过两项过滤：{candidate_df['candidate_accepted'].mean():.2%}。
- 每场有效候选：均值 {scene_df['valid_candidate_count'].mean():.2f}，中位数 {median_valid:.1f}，最小 {int(scene_df['valid_candidate_count'].min())}。

![valid candidate histogram]({report_rel}/valid_candidate_histogram.png)

### 各扰动保留率

{markdown_table(['扰动', '数量', '运动学有效率', 'route proxy 有效率', '最终保留率'], perturbation_table)}

## 2. Consequence 多样性与 pair 密度

- pair 总数：{len(pair_df)}。
- equivalent：{int((pair_df['pair_type'] == 'effect_equivalent').sum())}（{percentage((pair_df['pair_type'] == 'effect_equivalent').sum(), len(pair_df))}）。
- divergent：{int((pair_df['pair_type'] == 'effect_divergent').sum())}（{percentage((pair_df['pair_type'] == 'effect_divergent').sum(), len(pair_df))}）。
- ambiguous：{int((pair_df['pair_type'] == 'ambiguous').sum())}（{percentage((pair_df['pair_type'] == 'ambiguous').sum(), len(pair_df))}）。
- 同时含 equivalent/divergent 的场景：{int(scene_df['has_equivalent_and_divergent'].sum())}/{len(scene_df)}（{both_rate:.1%}）。
- 每场 hard consequence 向量种类：均值 {scene_df['hard_vector_diversity'].mean():.2f}，最大 {int(scene_df['hard_vector_diversity'].max())}。

![consequence distance]({report_rel}/consequence_distance_histogram.png)

![pair types]({report_rel}/pair_type_distribution.png)

## 3. 安全边界与主要差异来源

安全边界定义为 hard consequence 不同且几何距离不超过配置阈值；共 {boundary_count} 个。下图固定选择几何距离最近的样本，不做有利案例筛选。

![safety boundaries]({report_rel}/safety_boundary_examples.png)

Hard 差异的主要计数与 soft 归一化差异均保存在 `divergence_metric_drivers.csv`。前五项为：

{markdown_table(['metric', 'kind', 'count', 'mean normalized difference'], [(row.metric, row.kind, int(row.count), 'n/a' if pd.isna(row.mean_normalized_difference) else f'{row.mean_normalized_difference:.3f}') for row in driver_df.head(5).itertuples()])}

## 4. 几何距离与后果距离

Spearman $\\rho={correlation:.3f}$。散点保留所有 pair 的确定性下采样，不只展示成功案例。

![geometry consequence scatter]({report_rel}/geometry_vs_consequence.png)

## 5. Log-replay 与 IDM 一致性

IDM 固定哈希子集包含 {candidate_df.loc[candidate_df['reactive_available'], 'scene_id'].nunique()} 个场景、{len(row_assessed)} 条有效候选、{len(assessed_pairs)} 个 pair。

- candidate hard agreement：{candidate_hard_agreement:.2%}。
- pairwise hard-relation agreement：{pair_hard_relation_agreement:.2%}。
- pairwise PDMS ranking agreement（含共同 tie）：{ranking_agreement:.2%}。

![LR IDM agreement]({report_rel}/lr_idm_agreement_matrix.png)

这是 NAVSIM-v2 同一 MetricCache 上的 `log_replay` 与 `reactive_model` 对照；本机没有 pilot train 的官方 v1 MetricCache，因此不能把它夸大成完整 v1/v2 evaluator 复现。它直接覆盖本课题最关心的 traffic-assumption 冲突。

## 6. 扰动类型产生的后果

![perturbation consequences]({report_rel}/perturbation_consequence_distribution.png)

## 7. 场景类别标签密度

类别由 anchor 的轨迹曲率、intersection occupancy、动态 clearance/TTC 与 pair 多样性确定；一个场景可属于多个类别。

{markdown_table(['类别', '场景', '平均有效候选', '平均 equivalent', '平均 divergent', '安全边界 pair'], category_table)}

## 8. 已知限制与假设

- `extended_comfort` 需要跨相邻帧聚合，单场景 cache 中明确为缺失并被 robust-scale coverage 过滤；未用常数伪造。
- TTC 无事件样本是右删失值，缓存使用配置的 5 s 下界并同时保存 `ttc_infraction_observed`。
- `route_proxy_valid` 只是 Phase-1 快速过滤；DAC/DDC/LK 和 centerline 指标以官方 v2 PDM scorer 为准。
- IDM 只覆盖显式、确定性抽取的 64 场景，未计算项为 `reactive_model.available=false`。
- 固定日志无法辨识真实交互响应；对应字段始终属于 `unknown`。

## 9. 可复核产物

- `candidates.csv`：逐候选有效性与 consequence。
- `pairs.csv`：逐 pair 类型、距离、置信度与 LR/IDM order。
- `scenes.csv`：逐场景多样性。
- `perturbation_summary.csv`、`scene_categories.csv`、`divergence_metric_drivers.csv`、`lr_idm_agreement.csv`。
"""
    report_path.write_text(markdown, encoding="utf-8")
    gate_payload = {
        "gate": gate,
        "reason": gate_reason,
        "median_valid_candidates": median_valid,
        "scenes_with_both_pair_types_rate": both_rate,
        "safety_boundary_pair_count": boundary_count,
        "geometry_consequence_spearman": float(correlation),
        "candidate_hard_agreement": float(candidate_hard_agreement),
        "pairwise_ranking_agreement": float(ranking_agreement),
    }
    (output_dir / "gate1.json").write_text(json.dumps(gate_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(report_path), "artifacts": str(output_dir), **gate_payload}, indent=2))


if __name__ == "__main__":
    main()
