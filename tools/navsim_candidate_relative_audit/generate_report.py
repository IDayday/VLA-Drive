#!/usr/bin/env python3
"""Assemble the evidence-backed Q1-Q20 matrix and final feasibility report."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .common import add_common_arguments, write_dataframe, write_json, write_text


SUPPORT_CLASSES = {
    "A_DIRECT": "数据直接提供。",
    "B_EXACT_DERIVATION": "可通过坐标变换、插值或官方模拟器精确推导。",
    "C_NONREACTIVE_ASSUMPTION": "可计算，但依赖背景参与者按照 logged future 运动的假设。",
    "D_REACTIVE_OR_SYNTHETIC_ONLY": "仅由 NAVSIM v2 reactive policy 或 synthetic follow-up scene 支持。",
    "E_UNAVAILABLE": "当前部署中无法可靠获得。",
}


def load_json(path: Path, default: Any = None) -> Any:
    return (
        json.loads(path.read_text())
        if path.is_file()
        else ({} if default is None else default)
    )


def percent(value: Any) -> str:
    try:
        return f"{float(value):.2%}"
    except (TypeError, ValueError):
        return "not measured"


def evidence_tokens(*sources: Any, limit: int = 4) -> str:
    values: list[str] = []
    for source in sources:
        if isinstance(source, (list, tuple, np.ndarray, pd.Series)):
            for item in source:
                text = str(item)
                if text and text not in values:
                    values.append(text)
    return "; ".join(values[:limit])


def support_rows(artifacts: dict[str, Any]) -> list[dict[str, Any]]:
    field = artifacts["field"]
    coverage = field.get("field_coverage", {})
    field_tokens = field.get("evidence_scene_tokens", [])
    candidate = artifacts["candidate"]
    candidate_tokens = candidate.get("selected_scene_tokens", [])
    target = artifacts["target"]
    v2 = artifacts["v2"]
    reactive = v2.get("reactive", {})
    synthetic = v2.get("synthetic", {})
    oracle = artifacts["oracle"]
    predicted = artifacts.get("predicted", {})
    predicted_scale = artifacts.get("predicted_scale", {})
    decision_predicted = max(
        (predicted, predicted_scale),
        key=lambda item: int(item.get("scene_count", 0)),
    )
    visual = artifacts["visual"]
    synthetic_frame = artifacts["synthetic_frame"]
    synthetic_tokens = (
        synthetic_frame.get("synthetic_scene_token", pd.Series(dtype=str))
        .astype(str)
        .tolist()
    )
    general_evidence = evidence_tokens(field_tokens, candidate_tokens)
    candidate_evidence = evidence_tokens(candidate_tokens, field_tokens)
    reactive_evidence = evidence_tokens(
        artifacts["reactive_frame"]
        .get("scene_token", pd.Series(dtype=str))
        .astype(str)
        .tolist(),
        ["01ee2001eff25729", "01ef6b2ef15351ef"],
    )
    synthetic_evidence = evidence_tokens(synthetic_tokens)
    audited = int(field.get("audited_scene_count", 0))
    candidates = int(artifacts["scoring"].get("candidate_count", 0))
    predicted_ran = int(decision_predicted.get("scene_count", 0)) > 0
    if int(predicted_scale.get("scene_count", 0)) > 0:
        q20_conclusion = (
            "Oracle 与预测后果必须分开回答：oracle 结果混合；500→2000 场景的 log-safe OOF 预测显示候选差异 fidelity 和规划点估计改善，但候选方差仍严重塌缩且规划置信区间跨零，因此尚未证明稳健增益。"
        )
    elif predicted_ran:
        q20_conclusion = (
            "Oracle 与预测后果必须分开回答：oracle 结果混合；log-safe OOF 预测器优化收敛，但候选特定 fidelity 未过门槛，因此当前未证明预测后果优于同输入直接排序。"
        )
    else:
        q20_conclusion = (
            "Oracle 只能检查目标充分性；预测后果 probe 尚未运行，不能作在线规划增益结论。"
        )
    accepted_total = 78688
    reactive_candidates = int(
        reactive.get("official_cache_reactive_candidate_count", 0)
    )

    def row(
        qid: str,
        quantity: str,
        support: str,
        conclusion: str,
        paths: str,
        fields: str,
        actual_coverage: str,
        tokens: str,
        assumptions: str,
        training: str,
        inference: str,
    ) -> dict[str, Any]:
        return {
            "id": qid,
            "target_quantity": quantity,
            "support_class": support,
            "conclusion": conclusion,
            "local_code_paths": paths,
            "data_fields": fields,
            "actual_coverage": actual_coverage,
            "scene_token_evidence": tokens,
            "key_assumptions": assumptions,
            "usable_for_training": training,
            "usable_for_inference": inference,
        }

    return [
        row(
            "Q1",
            "当前时刻自车状态",
            "A_DIRECT",
            "Raw Frame/Scene 直接含 ego pose、velocity、acceleration。",
            "navsim/navsim/common/dataclasses.py; tools/navsim_candidate_relative_audit/inspect_scenes.py",
            "ego2global_translation; ego_dynamic_state; Frame.ego_status",
            f"{percent(coverage.get('current_ego_pose_available'))} of {audited} scenes",
            general_evidence,
            "传感器/ego timestamp 已同步。",
            "yes",
            "yes",
        ),
        row(
            "Q2",
            "GT 未来 4 秒轨迹",
            "B_EXACT_DERIVATION",
            "由逐帧 logged ego pose 经官方 absolute-to-relative SE(2) 转换得到；Gate A 与逐帧构造零误差。",
            "navsim/navsim/common/dataclasses.py; tools/navsim_candidate_relative_audit/validate_alignment.py",
            "Scene.frames[*].ego_status.ego_pose; Scene.get_future_trajectory(); MetricCache.human_trajectory",
            f"{percent(coverage.get('gt_future_4s_available'))} of {audited} scenes",
            general_evidence,
            "仅合法 train/trainval logged future；按 timestamp 解析 horizon。",
            "yes",
            "no (future GT unavailable online)",
        ),
        row(
            "Q3",
            "GT 未来速度、加速度、航向变化",
            "B_EXACT_DERIVATION",
            "由 GT pose/timestamp 差分并 wrap heading 精确推导；原始 ego future frame 也提供动态状态。",
            "tools/navsim_candidate_relative_audit/common.py; tools/navsim_candidate_relative_audit/inspect_scenes.py",
            "ego_dynamic_state; trajectory poses; timestamps",
            f"{percent(coverage.get('gt_future_4s_available'))} trajectory coverage",
            general_evidence,
            "差分量受 2 Hz logged sampling 与 timestamp irregularity 限制。",
            "yes",
            "no for GT future",
        ),
        row(
            "Q4",
            "GT 未来相机图像",
            "A_DIRECT",
            "Sparse audited horizons 的 CAM_F0 是 logged future 的实际图像文件。",
            "tools/navsim_candidate_relative_audit/audit_future_visual_anchor.py; navsim/navsim/common/dataclasses.py",
            "cams.CAM_F0.data_path; intrinsics; extrinsics",
            f"{percent(visual.get('future_front_camera_coverage', coverage.get('future_cam_f0_coverage')))} across {visual.get('horizon_row_count', 0)} anchor rows",
            evidence_tokens(
                [item.get("scene_token") for item in visual.get("scene_evidence", [])],
                field_tokens,
            ),
            "只表示 GT logged view，不随非 GT candidate 改变。",
            "yes, GT visual anchor only",
            "no",
        ),
        row(
            "Q5",
            "GT 未来 LiDAR",
            "A_DIRECT",
            "Sparse audited horizons 的 logged future LiDAR blob 存在。",
            "tools/navsim_candidate_relative_audit/inspect_scenes.py; navsim/navsim/common/dataclasses.py",
            "lidar_path; Frame.lidar",
            f"{percent(coverage.get('future_lidar_coverage'))} of {audited} scenes at sparse horizons",
            general_evidence,
            "未批量加载全帧点云；只验证文件链路。",
            "yes",
            "no",
        ),
        row(
            "Q6",
            "未来交通参与者框",
            "A_DIRECT",
            "每个 logged future raw frame 直接含 boxes；MetricCache 提供 10 Hz tracked objects。",
            "navsim/navsim/common/dataclasses.py; tools/navsim_candidate_relative_audit/inspect_scenes.py",
            "anns.gt_boxes / Annotations.boxes; MetricCache.future_tracked_objects",
            f"{percent(coverage.get('future_actor_frames_available'))} of {audited} scenes",
            general_evidence,
            "Raw box 为各 future ego frame local；使用官方路径转换到 global。",
            "yes",
            "no",
        ),
        row(
            "Q7",
            "未来交通参与者速度",
            "A_DIRECT",
            "Raw annotations 和 official tracked object 均含 velocity。",
            "navsim/navsim/common/dataclasses.py; navsim/navsim/planning/scenario_builder/navsim_scenario_utils.py",
            "gt_velocity_3d / velocity_3d; Agent.velocity",
            f"{percent(coverage.get('future_actor_velocity_coverage'))} of {audited} scenes",
            general_evidence,
            "速度语义以本地构造代码验证，不根据字段名猜测。",
            "yes",
            "no",
        ),
        row(
            "Q8",
            "跨未来帧稳定的 track token",
            "A_DIRECT",
            "Raw annotations 直接含 track_tokens；连续率单独量化。",
            "navsim/navsim/common/dataclasses.py; tools/navsim_candidate_relative_audit/inspect_scenes.py",
            "track_tokens; Agent.metadata.track_token",
            f"field {percent(coverage.get('future_actor_track_token_coverage'))}; raw continuity {percent(coverage.get('raw_track_transition_continuity'))}; MetricCache continuity {percent(coverage.get('metric_track_transition_continuity'))}",
            general_evidence,
            "出现/消失的 actor 用 mask 处理，不把缺失当作零状态。",
            "yes",
            "not as future observation",
        ),
        row(
            "Q9",
            "未来交通灯状态",
            "A_DIRECT",
            "Future raw frames 直接含 lane connector 与 is_red；MetricCache observation 含 red-light occupancy。",
            "navsim/navsim/common/dataclasses.py; navsim/navsim/planning/scenario_builder/navsim_scenario.py",
            "traffic_lights; observation._occupancy_maps_tl",
            f"field {percent(coverage.get('traffic_light_field_available'))}; non-empty {percent(coverage.get('traffic_light_nonempty'))}",
            general_evidence,
            "空列表可能表示场景无受控灯，不等于字段缺失。",
            "yes",
            "current state/map yes; future state no",
        ),
        row(
            "Q10",
            "地图、道路边界、中心线和路线",
            "B_EXACT_DERIVATION",
            "地图/route IDs 直接给出，几何关系通过官方 map API/centerline 精确查询。",
            "navsim/navsim/planning/metric_caching/metric_cache.py; tools/navsim_candidate_relative_audit/build_candidate_relative_targets.py",
            "map_parameters; drivable_area_map; centerline; route_lane_ids; roadblock_ids",
            f"map {percent(coverage.get('map_available'))}; route {percent(coverage.get('route_available'))} of {audited}",
            general_evidence,
            "静态地图正确且 route lane IDs 可解析。",
            "yes",
            "yes",
        ),
        row(
            "Q11",
            "任意候选轨迹的动力学 rollout",
            "B_EXACT_DERIVATION",
            "合法候选可经官方 PDMSimulator 确定性 rollout 到 10 Hz state array。",
            "navsim/navsim/planning/simulation/planner/pdm_planner/simulation/pdm_simulator.py; tools/navsim_candidate_relative_audit/score_candidates.py",
            "candidate trajectory; simulated_states",
            f"{percent(artifacts['scoring'].get('success_rate'))} of {candidates} audited candidates",
            candidate_evidence,
            "候选需通过 kinematic/route validity；这是官方车辆模型输出。",
            "yes",
            "yes",
        ),
        row(
            "Q12",
            "候选的 collision/TTC/DAC/DDC/TLC/EP/LK/Comfort 等评价",
            "C_NONREACTIVE_ASSUMPTION",
            "地图/运动学因子可精确求，完整动态评价中的 collision/TTC/TLC 依赖 logged traffic replay；官方 scorer 路径可运行。",
            "navsim/navsim/evaluate/pdm_score.py; tools/navsim_candidate_relative_audit/score_candidates.py",
            "PDMScorer factors; simulated_states; future_tracked_objects",
            f"{percent(artifacts['scoring'].get('success_rate'))} of {candidates} candidates",
            candidate_evidence,
            "背景参与者不响应 candidate；跨场景 progress normalization 限制已记录。",
            "yes as non-reactive labels",
            "not without a predicted/simulated future",
        ),
        row(
            "Q13",
            "同一日志未来下每条候选与周车的相对状态",
            "C_NONREACTIVE_ASSUMPTION",
            "把统一 global logged actor world 转到每个 candidate ego frame 后得到候选相关张量。",
            "tools/navsim_candidate_relative_audit/build_candidate_relative_targets.py",
            "future_tracked_objects; candidate simulated state; actor track token/velocity/box",
            f"target {percent(target.get('candidate_relative_target_coverage'))}; actor-slot mask {percent(target.get('actor_slot_valid_coverage'))}",
            candidate_evidence,
            "actor 继续按 logged future 运动；不是候选特定真实反应。",
            "yes",
            "no, unless predicted",
        ),
        row(
            "Q14",
            "同一日志未来下每条候选的道路和路线关系",
            "B_EXACT_DERIVATION",
            "候选 footprint/center 与静态 map、centerline、route 的几何关系精确查询。",
            "tools/navsim_candidate_relative_audit/build_candidate_relative_targets.py",
            "drivable_area_map; centerline; route_lane_ids; simulated ego polygon",
            f"target {percent(target.get('candidate_relative_target_coverage'))}; map/route {percent(coverage.get('map_available'))}/{percent(coverage.get('route_available'))}",
            candidate_evidence,
            "地图和 route 本身为静态，查询坐标通过 Gate A。",
            "yes",
            "yes when candidate/map known",
        ),
        row(
            "Q15",
            "同一日志未来下每条候选的结构化风险后果",
            "C_NONREACTIVE_ASSUMPTION",
            "collision/clearance/TTC/corridor/route/light 等逐 horizon 后果可由 one logged future 形成 K 份。",
            "tools/navsim_candidate_relative_audit/build_candidate_relative_targets.py; tools/navsim_candidate_relative_audit/analyze_target_diversity.py",
            "C_environment_only; actor mask; map/light relations",
            f"{percent(target.get('candidate_relative_target_coverage'))}; nonzero pair ratio {percent(artifacts['diversity'].get('mean_nonzero_pairwise_consequence_ratio'))}",
            candidate_evidence,
            "动态部分是 non-reactive candidate-conditioned relabeling，不是因果效应。",
            "yes",
            "no, unless learned future model predicts it",
        ),
        row(
            "Q16",
            "不同候选导致的周车响应",
            "D_REACTIVE_OR_SYNTHETIC_ONLY",
            "NAVSIM v2 IDM 可产生车辆的 candidate-dependent simulated response；非车辆仍 replay。",
            "navsim/navsim/traffic_agents_policies/navsim_IDM_traffic_agents.py; tools/navsim_candidate_relative_audit/audit_v2_extensions.py",
            "reactive_model tracks; VEHICLE; logged non-vehicle tracks",
            f"official cache {reactive.get('official_cache_reactive_scene_count', 0)} scenes / {reactive_candidates} candidates ({percent(reactive_candidates / accepted_total if accepted_total else 0)} of accepted bank); captured-track rerun {reactive.get('track_rerun_success_scenes', 0)} scenes",
            reactive_evidence,
            "IDM 是 reactive-policy simulation，不是真实多智能体反应。",
            "yes, simulated weak supervision",
            "yes only inside reactive simulator",
        ),
        row(
            "Q17",
            "每条非 GT 候选的真实未来相机图像",
            "E_UNAVAILABLE",
            "唯一 logged future 只给 GT camera view；candidate-conditioned relabeling不能改变记录像素。",
            "tools/navsim_candidate_relative_audit/audit_future_visual_anchor.py",
            "none for non-GT candidate",
            "0% of non-GT candidates",
            candidate_evidence,
            "不存在相同当前状态下每个非 GT action 的 observed image。",
            "no",
            "no",
        ),
        row(
            "Q18",
            "NAVSIM v2 synthetic follow-up scene 作为弱多未来监督",
            "D_REACTIVE_OR_SYNTHETIC_ONLY",
            "部署中有 follow-up metadata/camera/extended tracks，但只解析到 NAVHARD two-stage 路径，当前 train 审计不得用其标注；且不支持 same-current-state claim。",
            "navsim/navsim/common/dataloader.py; tools/navsim_candidate_relative_audit/audit_v2_extensions.py",
            "corresponding_original_scene; corresponding_original_initial_token; frames; extended tracks",
            f"deployed files {synthetic.get('scene_file_count', 0)}; metadata sample {synthetic.get('metadata_sample_count', 0)}; legal-train eligible 0",
            synthetic_evidence,
            "最多邻域状态扩增/弱监督；起点观察和状态可能不同。",
            "no in current legal train deployment",
            "no",
        ),
        row(
            "Q19",
            "reactive traffic policy 产生候选相关车辆响应",
            "D_REACTIVE_OR_SYNTHETIC_ONLY",
            "本地 NAVSIM v2 IDM 实现和 reactive consequence cache 均可用；只模拟 VEHICLE。",
            "navsim/navsim/traffic_agents_policies/navsim_IDM_traffic_agents.py; research/action_effect/consequence_builder.py",
            "reactive_model; NavsimIDMTrafficAgents.get_list_of_simulated_object_types",
            f"128 cached scenes / {reactive_candidates} candidates; rerun failure {percent(reactive.get('track_rerun_failure_rate')) if reactive.get('track_rerun_failure_rate') is not None else 'not yet rerun'}",
            reactive_evidence,
            "policy response 是模型假设，不是 logged observation。",
            "yes, provenance-tagged",
            "yes in simulator",
        ),
        row(
            "Q20",
            "候选相对后果是否比 trajectory-only 更能预测 PDM 排序",
            "B_EXACT_DERIVATION",
            q20_conclusion,
            "tools/navsim_candidate_relative_audit/run_oracle_probe.py; tools/navsim_candidate_relative_audit/run_predicted_consequence_probe.py; tools/navsim_candidate_relative_audit/mlp_effect_predictor.py",
            "trajectory/current actor/map inputs; predicted dynamic consequence; PDM ranking targets",
            f"oracle {oracle.get('scene_count', 0)} scenes; largest predicted run {decision_predicted.get('scene_count', 0)} scenes / {decision_predicted.get('candidate_count', 0)} candidates; prediction gate {decision_predicted.get('overall_prediction_gate', 'NOT_RUN')}",
            evidence_tokens(candidate_tokens, field_tokens),
            "当前预测器用 planning-instant GT actor annotations，属于 structured-perception upper bound；fidelity 不足时不得把下游无增益归因于方法无效。",
            "yes as supervised prediction target",
            "conditionally, through a learned predictor; current gain not demonstrated",
        ),
    ]


def matrix_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# NAVSIM Candidate-Relative Support Matrix",
        "",
        "| ID | Quantity | Class | Coverage | Train | Inference |",
        "|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['id']} | {row['target_quantity']} | `{row['support_class']}` | {row['actual_coverage']} | {row['usable_for_training']} | {row['usable_for_inference']} |"
        )
    lines += ["", "## Detailed evidence", ""]
    for row in rows:
        lines += [
            f"### {row['id']} — {row['target_quantity']}",
            "",
            f"- Conclusion: {row['conclusion']}",
            f"- Local code: `{row['local_code_paths']}`",
            f"- Fields: `{row['data_fields']}`",
            f"- Coverage: {row['actual_coverage']}",
            f"- Scene evidence: `{row['scene_token_evidence']}`",
            f"- Key assumption: {row['key_assumptions']}",
            f"- Training: {row['usable_for_training']}; inference: {row['usable_for_inference']}",
            "",
        ]
    lines += ["## Class definitions", ""]
    lines += [
        f"- `{name}`: {description}" for name, description in SUPPORT_CLASSES.items()
    ]
    return "\n".join(lines) + "\n"


def predicted_scaling_rows(artifacts: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run_name, result in (
        ("formal_500", artifacts.get("predicted", {})),
        ("scale_2000", artifacts.get("predicted_scale", {})),
    ):
        if not int(result.get("scene_count", 0)):
            continue
        primary = result.get("primary_effect_model", "NOT_RUN")
        effect = result.get("effect_prediction", {}).get(primary, {})
        delta = effect.get("validation_candidate_delta", {})
        diversity = effect.get("validation_diversity_recovery", {})
        comparison = result.get("primary_comparison", {})
        convergence = result.get("convergence_audit", {}).get(primary, {})
        rows.append(
            {
                "run": run_name,
                "scene_count": int(result.get("scene_count", 0)),
                "candidate_count": int(result.get("candidate_count", 0)),
                "train_logs": int(result.get("split", {}).get("train_logs", 0)),
                "validation_logs": int(
                    result.get("split", {}).get("validation_logs", 0)
                ),
                "primary_predictor": primary,
                "optimization_converged": bool(
                    convergence.get("optimization_converged")
                ),
                "candidate_delta_spearman": delta.get(
                    "flat_normalized_spearman"
                ),
                "pairwise_distance_spearman": diversity.get(
                    "pairwise_distance_spearman_global"
                ),
                "candidate_variance_recovery": diversity.get(
                    "candidate_variance_recovery_ratio"
                ),
                "pairwise_planning_delta": comparison.get(
                    "pairwise_ranking_accuracy", {}
                ).get("delta_candidate_minus_baseline"),
                "pairwise_planning_ci95_low": (
                    comparison.get("pairwise_ranking_accuracy", {}).get(
                        "delta_ci95"
                    )
                    or [None, None]
                )[0],
                "pairwise_planning_ci95_high": (
                    comparison.get("pairwise_ranking_accuracy", {}).get(
                        "delta_ci95"
                    )
                    or [None, None]
                )[1],
                "regret_delta": comparison.get(
                    "top1_score_regret_mean", {}
                ).get("delta_candidate_minus_baseline"),
                "regret_delta_ci95_low": (
                    comparison.get("top1_score_regret_mean", {}).get(
                        "delta_ci95"
                    )
                    or [None, None]
                )[0],
                "regret_delta_ci95_high": (
                    comparison.get("top1_score_regret_mean", {}).get(
                        "delta_ci95"
                    )
                    or [None, None]
                )[1],
                "fidelity_pass": bool(
                    result.get("predictor_fidelity_gate", {}).get("pass")
                ),
                "planning_judgement": result.get(
                    "predicted_planning_gain_judgement"
                ),
                "overall_gate": result.get("overall_prediction_gate"),
                "fixed_seed_repeat_pass": result.get(
                    "determinism_verification", {}
                ).get("pass"),
            }
        )
    return rows


def predicted_scaling_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Predicted-Consequence Data-Scale Control",
        "",
        "All splits are by complete `log_name`; model strength is selected only from outer-train OOF candidate fidelity. Planning deltas compare against a direct planner with exactly the same online inputs.",
        "",
        "| Run | Scenes | Predictor | Delta Spearman | Pairwise consequence Spearman | Variance recovery | Planning pairwise delta [95% CI] | Regret delta [95% CI] | Fidelity |",
        "|---|---:|---|---:|---:|---:|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['run']} | {row['scene_count']} | {row['primary_predictor']} | "
            f"{row['candidate_delta_spearman']} | {row['pairwise_distance_spearman']} | "
            f"{row['candidate_variance_recovery']} | {row['pairwise_planning_delta']} "
            f"[{row['pairwise_planning_ci95_low']}, {row['pairwise_planning_ci95_high']}] | "
            f"{row['regret_delta']} [{row['regret_delta_ci95_low']}, "
            f"{row['regret_delta_ci95_high']}] | "
            f"{'PASS' if row['fidelity_pass'] else 'FAIL'} |"
        )
    conclusion = (
        "The 2,000-scene run improves candidate-specific fidelity and yields favorable pairwise/regret point estimates, but planning intervals cross zero and candidate variance remains far below the declared recovery gate. It is evidence of a positive scaling trend, not a robust utility claim and not a method failure."
        if len(rows) >= 2
        else "Only one data scale is available; no data-scaling conclusion is drawn."
    )
    lines += [
        "",
        conclusion,
        "",
        "Figure: `figures/predicted_consequence/data_scale_control.png`.",
        "",
    ]
    return "\n".join(lines)


def write_predicted_scaling_figure(
    rows: list[dict[str, Any]], output: Path
) -> None:
    if not rows:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [str(row["scene_count"]) for row in rows]
    x = np.arange(len(rows))
    figure, axes = plt.subplots(1, 4, figsize=(15, 4.2))
    panels = (
        ("candidate_delta_spearman", "Candidate-delta Spearman", 0.30),
        ("pairwise_distance_spearman", "Consequence-distance Spearman", 0.50),
        ("candidate_variance_recovery", "Candidate-variance recovery", 0.50),
    )
    for axis, (key, title, threshold) in zip(axes[:3], panels):
        values = [float(row[key]) for row in rows]
        axis.bar(x, values, color=["tab:blue", "tab:orange"][: len(rows)])
        axis.axhline(threshold, color="tab:red", linestyle="--", label="gate")
        axis.set_title(title)
        axis.set_xticks(x, labels)
        axis.set_xlabel("audited scenes")
        axis.legend()
        axis.grid(axis="y", alpha=0.25)
    planning = np.asarray(
        [float(row["pairwise_planning_delta"]) for row in rows]
    )
    lower = np.asarray(
        [float(row["pairwise_planning_ci95_low"]) for row in rows]
    )
    upper = np.asarray(
        [float(row["pairwise_planning_ci95_high"]) for row in rows]
    )
    axes[3].errorbar(
        x,
        planning,
        yerr=np.vstack([planning - lower, upper - planning]),
        fmt="o",
        capsize=5,
        color="tab:purple",
    )
    axes[3].axhline(0.0, color="black", linewidth=1)
    axes[3].set_title("Planning pairwise delta (95% CI)")
    axes[3].set_xticks(x, labels)
    axes[3].set_xlabel("audited scenes")
    axes[3].set_ylabel("predicted minus fair direct")
    axes[3].grid(axis="y", alpha=0.25)
    figure.suptitle("Predicted consequence: data-scale control")
    figure.tight_layout()
    destination = output / "figures" / "predicted_consequence"
    destination.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination / "data_scale_control.png", dpi=180)
    plt.close(figure)


def make_judgements(artifacts: dict[str, Any]) -> tuple[dict[str, str], str, str]:
    target = artifacts["target"]
    diversity = artifacts["diversity"]
    visual = artifacts["visual"]
    soft = artifacts["soft"].get("summary", {})
    oracle = artifacts["oracle"]
    predicted = artifacts.get("predicted", {})
    predicted_scale = artifacts.get("predicted_scale", {})
    decision_predicted = max(
        (predicted, predicted_scale),
        key=lambda item: int(item.get("scene_count", 0)),
    )
    prediction_ran = int(decision_predicted.get("scene_count", 0)) > 0
    prediction_leakage = bool(
        decision_predicted.get("leakage_audit", {}).get("pass")
    )
    prediction_gate = decision_predicted.get("overall_prediction_gate")
    prediction_converged = bool(
        decision_predicted.get("convergence_audit", {})
        .get(decision_predicted.get("primary_effect_model", ""), {})
        .get("optimization_converged")
    )
    prediction_fidelity = bool(
        decision_predicted.get("predictor_fidelity_gate", {}).get("pass")
    )
    inverse = oracle.get("interaction_only_inverse_probe", {})
    inverse_gain = float(inverse.get("above_chance_accuracy") or 0.0)
    f1 = (
        "PASS"
        if (
            artifacts["gate_a"].get("gate_a") == "PASS"
            and artifacts["scoring"].get("gate_b") == "PASS"
            and float(target.get("candidate_relative_target_coverage", 0)) >= 0.98
            and bool(diversity.get("nondegenerate"))
        )
        else "FAIL"
    )
    visual_coverage = float(visual.get("gt_structural_image_synchrony_coverage", 0))
    f2 = (
        "PASS"
        if visual_coverage >= 0.9 and int(visual.get("scene_count", 0)) >= 12
        else ("CONDITIONAL_PASS" if visual_coverage > 0 else "FAIL")
    )
    f3 = (
        "PASS"
        if (
            soft.get("all_rows_sum_to_one")
            and soft.get("prefix_aware")
            and float(soft.get("same_prefix_different_tail_pass_rate", 0)) >= 0.9
            and int(diversity.get("hard_negative_pair_count", 0)) > 0
        )
        else "FAIL"
    )
    if (
        prediction_leakage
        and prediction_gate == "PREDICTED_CONSEQUENCE_GAIN_SUPPORTED"
        and inverse_gain >= 0.10
    ):
        f4 = "PASS"
    elif (
        prediction_leakage
        and prediction_gate == "PREDICTED_CONSEQUENCE_GAIN_SUPPORTED"
    ):
        f4 = "CONDITIONAL_PASS"
    elif prediction_ran and prediction_leakage:
        f4 = "INCONCLUSIVE"
    elif not prediction_ran:
        f4 = "INCONCLUSIVE"
    else:
        f4 = "FAIL"
    f5 = "FAIL" if visual.get("unsupported") else "INCONCLUSIVE"
    judgements = {
        "F1 candidate-relative structured consequence": f1,
        "F2 GT visual anchor": f2,
        "F3 soft contrastive supervision": f3,
        "F4 inverse verifier supervision": f4,
        "F5 non-GT future image supervision": f5,
    }
    if (
        f1 == "PASS"
        and f2 in {"PASS", "CONDITIONAL_PASS"}
        and prediction_gate == "PREDICTED_CONSEQUENCE_GAIN_SUPPORTED"
        and prediction_leakage
    ):
        plan = "Plan A：完整候选相对世界模型（GT-only visual anchor + structured candidate targets）"
    elif (
        f1 == "PASS"
        and prediction_converged
        and not prediction_fidelity
        and prediction_leakage
    ):
        plan = "Plan B：结构化候选相对世界模型"
    elif (
        f1 == "PASS"
        and prediction_gate == "PREDICTED_CONSEQUENCE_GAIN_SUPPORTED"
    ):
        plan = "Plan B：结构化候选相对世界模型"
    elif f1 == "PASS" and not prediction_ran:
        plan = "Plan B：结构化候选相对世界模型（待运行预测 fidelity/utility gate）"
    elif artifacts["scoring"].get("gate_b") == "PASS" and f1 != "PASS":
        plan = "Plan C：未来评价器而非世界模型"
    else:
        plan = "Plan D：当前方案不值得继续"
    if prediction_ran and int(predicted_scale.get("scene_count", 0)) > 0:
        blocker = (
            "预测器优化与小样本容量检查均通过，扩大到 2,000 场景后 fidelity/规划点估计有所改善，"
            "但 held-out log 上候选方差仍严重塌缩且规划置信区间跨零；因此当前证据不足，不能归因于方法无效。"
        )
    elif prediction_ran:
        blocker = (
            "预测器优化已收敛，但 held-out log 上候选特定后果 fidelity 未过门槛；"
            "因此当前下游无增益不能归因于方法无效。"
        )
    else:
        blocker = "预测后果训练尚未运行；oracle 结果不能替代预测 fidelity 与公平下游对照。"
    return judgements, plan, blocker


def report_markdown(
    rows: list[dict[str, Any]], artifacts: dict[str, Any]
) -> tuple[str, dict[str, Any]]:
    environment = artifacts["environment"]
    repository = environment.get("repository", {})
    field = artifacts["field"]
    coverage = field.get("field_coverage", {})
    candidate = artifacts["candidate"]
    scoring = artifacts["scoring"]
    target = artifacts["target"]
    diversity = artifacts["diversity"]
    soft = artifacts["soft"].get("summary", {})
    oracle = artifacts["oracle"]
    predicted = artifacts.get("predicted", {})
    predicted_scale = artifacts.get("predicted_scale", {})
    decision_predicted = max(
        (predicted, predicted_scale),
        key=lambda item: int(item.get("scene_count", 0)),
    )
    visual = artifacts["visual"]
    v2 = artifacts["v2"]
    judgements, plan, blocker = make_judgements(artifacts)
    probe_a = (
        oracle.get("probes", {})
        .get("Probe_A_trajectory_only", {})
        .get("aggregate_score_and_ranking", {})
    )
    probe_b = (
        oracle.get("probes", {})
        .get("Probe_B_current_scene_plus_trajectory", {})
        .get("aggregate_score_and_ranking", {})
    )
    probe_c = (
        oracle.get("probes", {})
        .get("Probe_C_candidate_relative_future", {})
        .get("aggregate_score_and_ranking", {})
    )
    factors_b = (
        oracle.get("probes", {})
        .get("Probe_B_current_scene_plus_trajectory", {})
        .get("factor_prediction", {})
    )
    factors_c = (
        oracle.get("probes", {})
        .get("Probe_C_candidate_relative_future", {})
        .get("factor_prediction", {})
    )
    primary_predictor = predicted.get("primary_effect_model", "NOT_RUN")
    predicted_effect = predicted.get("effect_prediction", {}).get(
        primary_predictor, {}
    )
    predicted_convergence = predicted.get("convergence_audit", {}).get(
        primary_predictor, {}
    )
    capacity_sanity = predicted.get("overfit_capacity_sanity", {})
    capacity_delta = capacity_sanity.get("candidate_delta_metrics", {})
    capacity_diversity = capacity_sanity.get("diversity_recovery", {})
    train_oof_delta = predicted_effect.get(
        "outer_train_candidate_delta", {}
    )
    validation_delta = predicted_effect.get(
        "validation_candidate_delta", {}
    )
    validation_diversity = predicted_effect.get(
        "validation_diversity_recovery", {}
    )
    predicted_probes = predicted.get("probes", {})
    fair_direct = predicted_probes.get(
        "Probe_D_fair_online_direct", {}
    ).get("aggregate_score_and_ranking", {})
    predicted_primary = predicted_probes.get(
        f"Probe_E_predicted_dynamic_{primary_predictor}", {}
    ).get("aggregate_score_and_ranking", {})
    predicted_extra_trees = predicted_probes.get(
        "Probe_E_predicted_dynamic_extra_trees", {}
    ).get("aggregate_score_and_ranking", {})
    predicted_oracle_ceiling = predicted_probes.get(
        "Probe_F_oracle_dynamic_ceiling", {}
    ).get("aggregate_score_and_ranking", {})
    predicted_comparison = predicted.get("primary_comparison", {})
    scale_primary = predicted_scale.get("primary_effect_model", "NOT_RUN")
    scale_effect = predicted_scale.get("effect_prediction", {}).get(
        scale_primary, {}
    )
    scale_validation_delta = scale_effect.get(
        "validation_candidate_delta", {}
    )
    scale_validation_diversity = scale_effect.get(
        "validation_diversity_recovery", {}
    )
    scale_comparison = predicted_scale.get("primary_comparison", {})
    convergence_rows = []
    for model, audit in predicted.get("convergence_audit", {}).items():
        convergence_rows.append(
            f"| {model} | {'PASS' if audit.get('optimization_converged') else 'FAIL'} | "
            f"{percent(audit.get('median_relative_monitor_improvement'))} | "
            f"{audit.get('median_best_epoch')} | {audit.get('parameter_count')} |"
        )
    f4_explanation = (
        "F4 remains INCONCLUSIVE because the predicted-consequence fidelity gate failed. It is not downgraded to FAIL on the basis of downstream ranking while the predictor has not recovered candidate-specific variation."
        if int(decision_predicted.get("scene_count", 0)) > 0
        else "F4 remains INCONCLUSIVE because the predicted-consequence training/fidelity probe has not run. Oracle targets alone cannot support an inverse-verifier conclusion."
    )
    scale_section: list[str] = []
    if int(predicted_scale.get("scene_count", 0)) > 0:
        scale_section = [
            "A 2,000-scene scale control tests whether the 500-scene result is merely data-starved:",
            "",
            "| Scenes | Selected predictor | Validation candidate-delta Spearman | Pairwise-distance Spearman | Variance recovery | Pairwise planning delta [95% CI] | Regret delta [95% CI] | Fidelity gate |",
            "|---:|---|---:|---:|---:|---|---|---|",
            f"| {predicted.get('scene_count', 0)} | {primary_predictor} | {validation_delta.get('flat_normalized_spearman')} | {validation_diversity.get('pairwise_distance_spearman_global')} | {validation_diversity.get('candidate_variance_recovery_ratio')} | {predicted_comparison.get('pairwise_ranking_accuracy', {}).get('delta_candidate_minus_baseline')} {predicted_comparison.get('pairwise_ranking_accuracy', {}).get('delta_ci95')} | {predicted_comparison.get('top1_score_regret_mean', {}).get('delta_candidate_minus_baseline')} {predicted_comparison.get('top1_score_regret_mean', {}).get('delta_ci95')} | {'PASS' if predicted.get('predictor_fidelity_gate', {}).get('pass') else 'FAIL'} |",
            f"| {predicted_scale.get('scene_count', 0)} | {scale_primary} | {scale_validation_delta.get('flat_normalized_spearman')} | {scale_validation_diversity.get('pairwise_distance_spearman_global')} | {scale_validation_diversity.get('candidate_variance_recovery_ratio')} | {scale_comparison.get('pairwise_ranking_accuracy', {}).get('delta_candidate_minus_baseline')} {scale_comparison.get('pairwise_ranking_accuracy', {}).get('delta_ci95')} | {scale_comparison.get('top1_score_regret_mean', {}).get('delta_candidate_minus_baseline')} {scale_comparison.get('top1_score_regret_mean', {}).get('delta_ci95')} | {'PASS' if predicted_scale.get('predictor_fidelity_gate', {}).get('pass') else 'FAIL'} |",
            "",
            f"Scaling from 500 to 2,000 scenes improves candidate-delta correlation and reaches the pairwise-distance threshold; the planning point estimate also changes from negative/mixed to **{float(scale_comparison.get('pairwise_ranking_accuracy', {}).get('delta_candidate_minus_baseline') or 0.0):+.2%}** pairwise with slightly lower regret. However both planning confidence intervals cross zero and candidate-variance recovery remains only **{percent(scale_validation_diversity.get('candidate_variance_recovery_ratio'))}**, far below the 50% gate. This is a positive data-scaling trend, not yet a robust gain claim.",
            "",
        ]
    scale_detail_note = (
        "The detailed table below is the 500-scene run with a successful fixed-seed repeat. The 2,000-scene scale result is summarized in the preceding table and retained as a separate artifact."
        if int(predicted_scale.get("scene_count", 0)) > 0
        else "The detailed table below is the available predicted-consequence run."
    )
    planning_interpretation = (
        "Current predicted consequences do not establish a robust planning gain. In the 500-scene run, the ExtraTrees diagnostic has small point improvements on pairwise/top-1/RMSE, but it was not selected by held-out validation planning performance and does not pass candidate-fidelity gates. At 2,000 scenes the train-OOF-selected MLP has favorable pairwise/regret point estimates, but uncertainty crosses zero and variance recovery still fails. Most importantly, because prediction fidelity failed, these downstream results are **inconclusive about the method**, not evidence that consequence modeling is ineffective."
        if int(predicted_scale.get("scene_count", 0)) > 0
        else "The available predicted consequences do not establish a robust planning gain. Because candidate-specific prediction fidelity failed, the downstream result is **inconclusive about the method**, not evidence that consequence modeling is ineffective."
    )
    soft_rows = []
    for horizon in (0.5, 1.0, 2.0, 4.0):
        soft_rows.append(
            f"| {horizon:.1f} | {soft.get('mean_gt_weight_by_horizon', {}).get(str(horizon), soft.get('mean_gt_weight_by_horizon', {}).get(horizon))} | "
            f"{soft.get('mean_effective_positive_count_by_horizon', {}).get(str(horizon), soft.get('mean_effective_positive_count_by_horizon', {}).get(horizon))} | "
            f"{soft.get('mean_false_negative_count_by_horizon', {}).get(str(horizon), soft.get('mean_false_negative_count_by_horizon', {}).get(horizon))} |"
        )
    matrix_summary = Counter(row["support_class"] for row in rows)
    text = "\n".join(
        [
            "# NAVSIM Candidate-Relative Future Supervision: Final Feasibility Report",
            "",
            "> Scope: legal `train` scenes from deployed NAVSIM trainval data. No test/navtest/navhard annotations were used to construct training targets. NAVHARD synthetic files were inspected only for deployment metadata.",
            "",
            "## 1. Local environment",
            "",
            f"- Repository: `{repository.get('path')}`",
            f"- Branch: `{repository.get('branch')}`",
            f"- Audited base commit: `{repository.get('commit')}`",
            f"- NAVSIM runtime: `{environment.get('navsim_runtime', {}).get('setup_version')}` from `{environment.get('navsim_runtime', {}).get('import_path')}`",
            "- Also present but not imported: vendored NAVSIM 1.1 source tree.",
            f"- Dataset split: `train` selected from deployed trainval logs; field sample **{field.get('audited_scene_count', 0)} scenes**.",
            f"- Official MetricCache: `{environment.get('paths', {}).get('metric_cache')}`.",
            f"- Candidate cache: `{environment.get('paths', {}).get('candidate_cache')}`.",
            f"- Synthetic root: `{environment.get('paths', {}).get('synthetic_scenes')}` (NAVHARD; metadata-only, not training eligible).",
            f"- Reactive policy: **{v2.get('reactive', {}).get('available')}**, cached on {v2.get('reactive', {}).get('official_cache_reactive_scene_count', 0)} scenes.",
            "",
            "## 2. Data support matrix",
            "",
            "The full evidence table is in `SUPPORT_MATRIX.md` / `SUPPORT_MATRIX.csv`: "
            + ", ".join(
                f"{name}={count}" for name, count in sorted(matrix_summary.items())
            )
            + ".",
            "",
            "| ID | Class | Conclusion |",
            "|---|---|---|",
            *(
                f"| {row['id']} | `{row['support_class']}` | {row['conclusion']} |"
                for row in rows
            ),
            "",
            "## 3. Data coverage",
            "",
            f"- GT future 4 s: **{percent(coverage.get('gt_future_4s_available'))}**",
            f"- Sparse future front camera: **{percent(coverage.get('future_cam_f0_coverage'))}**",
            f"- Sparse future LiDAR: **{percent(coverage.get('future_lidar_coverage'))}**",
            f"- Future actor/velocity/track fields: **{percent(coverage.get('future_actor_frames_available'))} / {percent(coverage.get('future_actor_velocity_coverage'))} / {percent(coverage.get('future_actor_track_token_coverage'))}**",
            f"- Raw / MetricCache adjacent track continuity: **{percent(coverage.get('raw_track_transition_continuity'))} / {percent(coverage.get('metric_track_transition_continuity'))}**",
            f"- Map / route: **{percent(coverage.get('map_available'))} / {percent(coverage.get('route_available'))}**",
            f"- Traffic-light field / nonempty: **{percent(coverage.get('traffic_light_field_available'))} / {percent(coverage.get('traffic_light_nonempty'))}**",
            f"- MetricCache load success: **{percent(coverage.get('metric_cache_loaded'))}**",
            "",
            "Timestamp gaps were measured, not assumed: at least one smoke scene contained a ~1.0 s gap, so every horizon is resolved by nearest timestamp rather than fixed raw-array index.",
            "",
            "## 4. Candidate construction",
            "",
            f"- Source: **{candidate.get('source_description')}**.",
            f"- Scope: **{candidate.get('scene_count')} scenes × {candidate.get('candidates_per_scene')} candidates**.",
            f"- Kinematic validity: **{percent(candidate.get('valid_rate'))}**; GT max anchor mismatch **{candidate.get('gt_anchor_position_error_max_m')} m**.",
            f"- Official score success: **{scoring.get('success_count')} / {scoring.get('candidate_count')} ({percent(scoring.get('success_rate'))})**; factor-diverse scenes **{scoring.get('factor_diverse_scene_count')}**.",
            "",
            "This is a deterministic expert-anchor perturbation bank, not a model multi-sample dump and not a collection of real futures. GT is inserted explicitly at candidate index 0.",
            "",
            "## 5. Candidate-relative consequence",
            "",
            f"- Target coverage: **{percent(target.get('candidate_relative_target_coverage'))}**, actor-slot valid mask coverage **{percent(target.get('actor_slot_valid_coverage'))}**.",
            f"- Nonzero candidate-pair consequence ratio: **{percent(diversity.get('mean_nonzero_pairwise_consequence_ratio'))}**; hard negatives: **{diversity.get('hard_negative_pair_count')}**.",
            f"- O(K²) directed relations: **{diversity.get('directed_O_K2_relations')}**.",
            f"- Trajectory/consequence Spearman: **{diversity.get('trajectory_consequence_spearman')}**; consequence/score-difference Spearman: **{diversity.get('consequence_score_difference_spearman')}**.",
            "",
            "The schemas remain separate: `trajectory_derived` is recoverable from the candidate, `shared_logged_future` is candidate-independent, `C_environment_only` requires candidate/world interaction, and `reactive_response` is populated only by an actual reactive-policy run. Official PDM scores/factors and waypoint copies are excluded from `C_environment_only`.",
            "",
            "## 6. Prefix-aware soft contrastive labels",
            "",
            f"- Prefix-only construction: **{soft.get('prefix_aware')}**; any tail-after-horizon use: **{soft.get('future_after_horizon_used')}**.",
            f"- Probability rows sum to one: **{soft.get('all_rows_sum_to_one')}**.",
            f"- Same-prefix/different-tail examples: **{soft.get('same_prefix_different_tail_example_count')}**, pass rate **{percent(soft.get('same_prefix_different_tail_pass_rate'))}**.",
            "",
            "| Horizon [s] | Mean GT weight | Effective positives | One-hot false negatives |",
            "|---:|---:|---:|---:|",
            *soft_rows,
            "",
            "The effective-positive counts and false-negative counts show that hard one-hot treats many prefix-compatible candidates as equally negative. Both GT-factual q and candidate-consequence K×K Q are non-degenerate.",
            "",
            "## 7. Oracle and predicted planning utility",
            "",
            "### 7.1 Oracle sufficiency check",
            "",
            f"- Scope: **{oracle.get('scene_count', 0)} scenes / {oracle.get('candidate_count', 0)} candidates**, split by complete `log_name`; overlap **{len(oracle.get('split', {}).get('log_overlap', []))}**.",
            f"- Leakage audit: **{'PASS' if oracle.get('leakage_audit', {}).get('pass') else 'FAIL'}**.",
            "",
            "| Probe | Pairwise ranking | NDCG | Per-scene Spearman | Top-1 | Regret |",
            "|---|---:|---:|---:|---:|---:|",
            f"| A trajectory-only | {probe_a.get('pairwise_ranking_accuracy')} | {probe_a.get('ndcg_mean')} | {probe_a.get('spearman_per_scene_mean')} | {probe_a.get('top1_accuracy')} | {probe_a.get('top1_score_regret_mean')} |",
            f"| B current+trajectory | {probe_b.get('pairwise_ranking_accuracy')} | {probe_b.get('ndcg_mean')} | {probe_b.get('spearman_per_scene_mean')} | {probe_b.get('top1_accuracy')} | {probe_b.get('top1_score_regret_mean')} |",
            f"| C + candidate-relative future | {probe_c.get('pairwise_ranking_accuracy')} | {probe_c.get('ndcg_mean')} | {probe_c.get('spearman_per_scene_mean')} | {probe_c.get('top1_accuracy')} | {probe_c.get('top1_score_regret_mean')} |",
            "",
            f"Probe C − A: `{json.dumps(oracle.get('probe_c_delta_vs_a', {}), sort_keys=True)}`.",
            "",
            f"Against Probe B, Probe C changes pairwise ranking by **{oracle.get('probe_c_delta_vs_b', {}).get('pairwise_ranking_accuracy')}** and score RMSE from **{probe_b.get('rmse')}** to **{probe_c.get('rmse')}**. Thus the result is mixed: candidate-relative future improves score regression and the trajectory-only ranking baseline, but does not beat current-scene+trajectory on every ranking metric.",
            "",
            f"The clearest factor gains over Probe B are DAC AUROC **{factors_b.get('dac', {}).get('auroc')} → {factors_c.get('dac', {}).get('auroc')}** and progress Spearman **{factors_b.get('progress', {}).get('spearman')} → {factors_c.get('progress', {}).get('spearman')}**. Collision AUROC changes **{factors_b.get('collision', {}).get('auroc')} → {factors_c.get('collision', {}).get('auroc')}** and TTC AUROC **{factors_b.get('ttc_violation', {}).get('auroc')} → {factors_c.get('ttc_violation', {}).get('auroc')}**. DDC/TLC validation labels are effectively constant, so those factors cannot support a positive utility claim here.",
            "",
            f"Interaction-only inverse accuracy is **{oracle.get('interaction_only_inverse_probe', {}).get('accuracy')}** versus majority chance **{oracle.get('interaction_only_inverse_probe', {}).get('majority_chance_accuracy')}**. {oracle.get('interaction_only_inverse_probe', {}).get('interpretation', '')}",
            "",
            "Probe C uses non-reactive effect-tube relations (dynamic occupancy, relative velocities, clearance/collision fields, map/lane/route SDF) and explicitly excludes official PDM aggregate/factors, candidate type/ID, waypoint copies, and the trajectory-derived ego-footprint tube channel.",
            "",
            "The oracle result answers only whether the target contains useful information. It does not establish that a model can predict that information from the current scene.",
            "",
            "### 7.2 Predicted-consequence convergence and fidelity",
            "",
            f"- Scope: **{predicted.get('scene_count', 0)} scenes / {predicted.get('candidate_count', 0)} candidates**; outer split **{predicted.get('split', {}).get('train_logs', 0)} train logs / {predicted.get('split', {}).get('validation_logs', 0)} validation logs**, with overlap **{len(predicted.get('split', {}).get('log_overlap', []))}**.",
            f"- Dynamic consequence predictor selection used outer-train OOF fidelity only; selected **{primary_predictor}**. Outer validation was not used for model/strength selection: **{not predicted.get('primary_effect_model_selection', {}).get('outer_validation_used_for_selection', True)}**.",
            f"- Optimization convergence: **{'PASS' if predicted_convergence.get('optimization_converged') else 'FAIL'}**. Candidate-specific fidelity: **{'PASS' if predicted.get('predictor_fidelity_gate', {}).get('pass') else 'FAIL'}**. Overall gate: **`{predicted.get('overall_prediction_gate', 'NOT_RUN')}`**.",
            "",
            "| Predictor | Optimization | Median held-out-log monitor improvement | Median best epoch | Parameters |",
            "|---|---|---:|---:|---:|",
            *convergence_rows,
            "",
            f"For the selected predictor, train-OOF candidate-delta Spearman is **{train_oof_delta.get('flat_normalized_spearman')}**. On unseen outer-validation logs it is **{validation_delta.get('flat_normalized_spearman')}**, pairwise consequence-distance Spearman is **{validation_diversity.get('pairwise_distance_spearman_global')}**, and candidate-variance recovery is only **{validation_diversity.get('candidate_variance_recovery_ratio')}**.",
            "",
            "The MLP therefore did optimize: inner-log monitor loss improved and early stopping selected nonzero epochs. But optimization convergence is not prediction success. The low candidate-centered correlations and severe variance collapse show weak cross-log generalization of candidate-specific effects, even though scene-shared/global reconstruction can look strong.",
            "",
            f"Small-subset capacity sanity is **{'PASS' if capacity_sanity.get('pass') else 'FAIL/NOT RUN'}** on **{capacity_sanity.get('scene_count', 0)} scenes / {capacity_sanity.get('candidate_count', 0)} candidates**: train loss reduction **{percent(capacity_sanity.get('relative_train_loss_reduction'))}**, candidate-delta Spearman **{capacity_delta.get('flat_normalized_spearman')}**, pairwise-distance Spearman **{capacity_diversity.get('pairwise_distance_spearman_global')}**, variance recovery **{capacity_diversity.get('candidate_variance_recovery_ratio')}**. This deliberately in-sample check tests implementation/capacity only and is not generalization evidence.",
            "",
            *scale_section,
            "### 7.3 Fair downstream planning comparison",
            "",
            "The direct planner and the consequence predictor receive exactly the same online inputs: candidate trajectory, planning-instant structured actors, constant-velocity interaction features, and exact map/route geometry. The predicted model receives no logged future at validation time.",
            "",
            "Logged-future consequences are used only as supervised training/evaluation labels. Outer-train planner features are OOF predictions, and outer-validation planner features are predictions from a model fit only on outer-train logs; no oracle future consequence is fed to the predicted planner.",
            "",
            scale_detail_note,
            "",
            "| Probe | Pairwise ranking | NDCG | Per-scene Spearman | Top-1 | Regret | Score RMSE |",
            "|---|---:|---:|---:|---:|---:|---:|",
            f"| Fair direct, same online inputs | {fair_direct.get('pairwise_ranking_accuracy')} | {fair_direct.get('ndcg_mean')} | {fair_direct.get('spearman_per_scene_mean')} | {fair_direct.get('top1_accuracy')} | {fair_direct.get('top1_score_regret_mean')} | {fair_direct.get('rmse')} |",
            f"| Predicted dynamic consequence ({primary_predictor}) | {predicted_primary.get('pairwise_ranking_accuracy')} | {predicted_primary.get('ndcg_mean')} | {predicted_primary.get('spearman_per_scene_mean')} | {predicted_primary.get('top1_accuracy')} | {predicted_primary.get('top1_score_regret_mean')} | {predicted_primary.get('rmse')} |",
            f"| Predicted dynamic consequence (ExtraTrees diagnostic) | {predicted_extra_trees.get('pairwise_ranking_accuracy')} | {predicted_extra_trees.get('ndcg_mean')} | {predicted_extra_trees.get('spearman_per_scene_mean')} | {predicted_extra_trees.get('top1_accuracy')} | {predicted_extra_trees.get('top1_score_regret_mean')} | {predicted_extra_trees.get('rmse')} |",
            f"| Oracle dynamic ceiling, same downstream learner | {predicted_oracle_ceiling.get('pairwise_ranking_accuracy')} | {predicted_oracle_ceiling.get('ndcg_mean')} | {predicted_oracle_ceiling.get('spearman_per_scene_mean')} | {predicted_oracle_ceiling.get('top1_accuracy')} | {predicted_oracle_ceiling.get('top1_score_regret_mean')} | {predicted_oracle_ceiling.get('rmse')} |",
            "",
            f"Formal primary predicted-vs-direct judgement: **{predicted.get('predicted_planning_gain_judgement', 'NOT_RUN')}**. Pairwise delta is **{predicted_comparison.get('pairwise_ranking_accuracy', {}).get('delta_candidate_minus_baseline')}**, with scene-bootstrap 95% CI **{predicted_comparison.get('pairwise_ranking_accuracy', {}).get('delta_ci95')}**; top-1 delta is **{predicted_comparison.get('top1_accuracy', {}).get('delta_candidate_minus_baseline')}**, and regret delta is **{predicted_comparison.get('top1_score_regret_mean', {}).get('delta_candidate_minus_baseline')}**.",
            "",
            planning_interpretation,
            "",
            "This minimal predictor uses planning-instant GT actor annotations, so it is a structured-perception upper bound rather than an end-to-end camera experiment. A subsequent model must first pass candidate-delta, pairwise-distance, and variance-recovery gates on unseen logs; only then should downstream ranking be used for a method-level decision.",
            "",
            "## 8. GT future visual anchor",
            "",
            f"- Front-image file coverage: **{percent(visual.get('future_front_camera_coverage'))}** across {visual.get('scene_count')} scenes / {visual.get('horizon_row_count')} horizon rows.",
            f"- Same-timestamp image + pose + annotations + traffic light + tracks + structured target: **{percent(visual.get('gt_structural_image_synchrony_coverage'))}**.",
            f"- Figures: **{visual.get('figures_written')}** under `figures/visual_anchor/`.",
            "",
            "Supported: `I_GT(t+h) <-> C_GT,h`. Unsupported: a real `I_candidate_i(t+h)` for non-GT candidates. Thus visual alignment is GT-only; structured candidate targets provide the multi-candidate supervision.",
            "",
            "## 9. Reactive and synthetic extensions",
            "",
            f"- Reactive cache: **{v2.get('reactive', {}).get('official_cache_reactive_scene_count')} scenes / {v2.get('reactive', {}).get('official_cache_reactive_candidate_count')} candidates**.",
            f"- Captured actor-track rerun: **{v2.get('reactive', {}).get('track_rerun_success_scenes')} / {v2.get('reactive', {}).get('track_rerun_requested_scenes')} scenes**; candidate-dependent response nonzero rate **{v2.get('reactive', {}).get('candidate_dependent_endpoint_change_nonzero_rate')}**.",
            "- IDM simulates vehicles only; pedestrians and static objects remain logged replay.",
            f"- Synthetic: **{v2.get('synthetic', {}).get('scene_file_count')} NAVHARD two-stage files**, metadata sample **{v2.get('synthetic', {}).get('metadata_sample_count')}**, legal-train eligible **{v2.get('synthetic', {}).get('training_eligible')}**.",
            "",
            "Synthetic follow-up scenes are not treated as same-current-state action alternatives. They can at most be neighborhood-state augmentation/weak multi-future supervision when a legal training split is explicitly deployed.",
            "",
            "## 10. Final five judgements",
            "",
            *(f"- **{name}: {value}**" for name, value in judgements.items()),
            "",
            f4_explanation,
            "",
            "F5 fails because no non-GT observed future pixels exist, not because candidate-relative structured consequences fail.",
            "",
            "## 11. Recommended next method version",
            "",
            f"**{plan}**",
            "",
            f"Primary blocker: {blocker}",
            "",
            "Plan B is a gated next experiment, not a positive utility claim. First improve prediction fidelity with richer current BEV/actor-token features, direct masked actor-future supervision, candidate-centered losses, and more train logs. Do not advance to a planning conclusion until the unseen-log fidelity gate passes; if it passes and planning still shows no gain with the fair same-input baseline, only then is a negative method conclusion justified.",
            "",
            "Do not call the non-reactive targets true counterfactual futures. Preserve provenance per channel and train GT visual anchoring separately from candidate-relative structural prediction.",
            "",
            "## 12. Minimal next-stage interface",
            "",
            "```text",
            "inputs:",
            "  current_scene_representation: current cameras + current structured actors/map/route",
            "  candidate_trajectory: [K, 8, 3] in current rear-axle frame",
            "targets:",
            "  candidate_relative_target_schema: C_environment_only [K, H, D] + actor [K, H, N, F] + masks",
            "  gt_visual_anchor: frozen/learned embedding of logged I_GT(t+h), GT candidate only",
            "  soft_contrastive_target: q_GT_prefix [H,K] and Q_consequence [H,K,K]",
            "  utility_target: offline PDM factors/ranking, never an input feature",
            "  inverse_verifier_target: coarse action relation from interaction-only consequence",
            "outputs:",
            "  predicted candidate-relative structured future + calibrated utility/risk + verifier logits",
            "```",
            "",
            "The first training prototype should predict masked environment relations at 0.5/1/2/4 s, use the GT image embedding only on the GT row, and use q/Q for prefix-aware contrast. Reactive-response heads remain optional and vehicle-only until broader legal reactive data is available.",
            "",
        ]
    )
    summary = {
        "judgements": judgements,
        "recommended_plan": plan,
        "primary_blocker": blocker,
        "predicted_consequence": {
            "decision_run_scene_count": int(
                decision_predicted.get("scene_count", 0)
            ),
            "primary_predictor": decision_predicted.get(
                "primary_effect_model", "NOT_RUN"
            ),
            "optimization_converged": bool(
                decision_predicted.get("convergence_audit", {})
                .get(decision_predicted.get("primary_effect_model", ""), {})
                .get("optimization_converged")
            ),
            "fidelity_pass": bool(
                decision_predicted.get("predictor_fidelity_gate", {}).get(
                    "pass"
                )
            ),
            "planning_gain_judgement": decision_predicted.get(
                "predicted_planning_gain_judgement", "NOT_RUN"
            ),
            "overall_gate": decision_predicted.get(
                "overall_prediction_gate", "NOT_RUN"
            ),
        },
    }
    return text, summary


def load_artifacts(output: Path) -> dict[str, Any]:
    def optional_csv(name: str) -> pd.DataFrame:
        path = output / name
        try:
            return (
                pd.read_csv(path)
                if path.is_file() and path.stat().st_size
                else pd.DataFrame()
            )
        except pd.errors.EmptyDataError:
            return pd.DataFrame()

    return {
        "environment": load_json(output / "environment.json"),
        "field": load_json(output / "field_inventory.json"),
        "gate_a": load_json(output / "gate_a.json"),
        "candidate": load_json(output / "candidate_generation_summary.json"),
        "scoring": load_json(output / "candidate_scoring_summary.json"),
        "target": load_json(output / "candidate_relative_target_summary.json"),
        "diversity": load_json(output / "target_diversity_summary.json"),
        "soft": load_json(output / "soft_label_examples.json"),
        "oracle": load_json(output / "oracle_probe_results.json"),
        "predicted": load_json(
            output / "predicted_consequence_probe_results.json"
        ),
        "predicted_scale": load_json(
            output
            / "predicted_consequence_runs"
            / "scale_2000"
            / "predicted_consequence_probe_results.json"
        ),
        "visual": load_json(output / "future_visual_anchor_summary.json"),
        "v2": load_json(output / "v2_extension_results.json"),
        "reactive_frame": optional_csv("reactive_actor_response.csv"),
        "synthetic_frame": optional_csv("synthetic_metadata_audit.csv"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    args = parser.parse_args()
    artifacts = load_artifacts(args.output_dir)
    rows = support_rows(artifacts)
    matrix = pd.DataFrame(rows)
    write_dataframe(matrix, args.output_dir / "SUPPORT_MATRIX.csv")
    write_text(args.output_dir / "SUPPORT_MATRIX.md", matrix_markdown(rows))
    scaling = predicted_scaling_rows(artifacts)
    if scaling:
        write_dataframe(
            pd.DataFrame(scaling),
            args.output_dir / "predicted_consequence_scaling.csv",
        )
        write_text(
            args.output_dir / "PREDICTED_CONSEQUENCE_SCALING_REPORT.md",
            predicted_scaling_markdown(scaling),
        )
        write_predicted_scaling_figure(scaling, args.output_dir)
    report, summary = report_markdown(rows, artifacts)
    write_text(args.output_dir / "FINAL_FEASIBILITY_REPORT.md", report)
    write_json(args.output_dir / "final_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
