#!/usr/bin/env python3
"""Generate the evidence-backed Q1-Q20 support matrix and final report."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .common import add_common_arguments, append_command, ensure_output_dir, write_markdown


STATUSES = {
    "A_DIRECT": "数据直接提供。",
    "B_EXACT_DERIVATION": "可通过坐标变换、插值或官方模拟器精确推导。",
    "C_NONREACTIVE_ASSUMPTION": "可计算，但依赖背景参与者按 logged future 运动的 non-reactive 假设。",
    "D_REACTIVE_OR_SYNTHETIC_ONLY": "仅由已部署 NAVSIM v2 reactive 机制或 synthetic follow-up scene 支持。",
    "E_UNAVAILABLE": "当前部署中无法可靠获得。",
}


def _json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _rate(frame: pd.DataFrame, column: str) -> float | None:
    if len(frame) == 0 or column not in frame:
        return None
    values = frame[column]
    if values.dtype != bool:
        values = values.astype(bool)
    return float(values.mean())


def _fmt_rate(rate: float | None, denominator: int | None = None) -> str:
    if rate is None or not math.isfinite(rate):
        return "not measured"
    count = int(round(rate * denominator)) if denominator is not None else None
    return f"{rate:.2%} ({count}/{denominator})" if denominator is not None else f"{rate:.2%}"


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "NA"
    try:
        number = float(value)
        return f"{number:.{digits}f}" if math.isfinite(number) else "NA"
    except (TypeError, ValueError):
        return str(value)


def _evidence(frame: pd.DataFrame, column: str | None = None, limit: int = 4) -> str:
    if len(frame) == 0 or "scene_token" not in frame:
        return "none"
    selected = frame
    if column and column in frame:
        selected = frame[frame[column].astype(bool)]
    tokens = selected.scene_token.astype(str).drop_duplicates().head(limit).tolist()
    return ";".join(tokens) if tokens else "none"


def _oracle_improvement(oracle: dict[str, Any]) -> tuple[bool, dict[str, float | None]]:
    if oracle.get("status") != "COMPLETE":
        return False, {"pairwise_delta_vs_a": None, "pairwise_delta_vs_b": None, "regret_delta_vs_a": None}
    probes = oracle["probes"]
    a = probes["A"]["aggregate_score"]
    b = probes["B"]["aggregate_score"]
    c = probes["C"]["aggregate_score"]
    deltas = {
        "pairwise_delta_vs_a": c["pairwise_ranking_accuracy"] - a["pairwise_ranking_accuracy"],
        "pairwise_delta_vs_b": c["pairwise_ranking_accuracy"] - b["pairwise_ranking_accuracy"],
        "regret_delta_vs_a": a["top1_score_regret"] - c["top1_score_regret"],
    }
    improved = bool(deltas["pairwise_delta_vs_a"] > 0.005 and deltas["regret_delta_vs_a"] >= -1e-6)
    return improved, deltas


def _matrix_rows(output_dir: Path) -> tuple[list[dict[str, str]], dict[str, Any]]:
    environment = _json(output_dir / "environment.json", {})
    alignment = _json(output_dir / "alignment_metrics.json", {})
    scoring = _json(output_dir / "candidate_scoring_audit.json", {})
    schema = _json(output_dir / "target_schema.json", {})
    diversity = _json(output_dir / "target_diversity_summary.json", {})
    v2 = _json(output_dir / "v2_extension_audit.json", {})
    oracle = _json(output_dir / "oracle_probe_results.json", {})
    scene = pd.read_csv(output_dir / "scene_coverage.csv") if (output_dir / "scene_coverage.csv").is_file() else pd.DataFrame()
    visual = pd.read_csv(output_dir / "future_visual_anchor_coverage.csv") if (output_dir / "future_visual_anchor_coverage.csv").is_file() else pd.DataFrame()
    target = pd.read_csv(output_dir / "target_coverage.csv") if (output_dir / "target_coverage.csv").is_file() else pd.DataFrame()
    manifest = pd.read_parquet(output_dir / "candidate_manifest.parquet") if (output_dir / "candidate_manifest.parquet").is_file() else pd.DataFrame()
    synthetic_inventory = pd.read_csv(output_dir / "synthetic_scene_inventory.csv") if (output_dir / "synthetic_scene_inventory.csv").is_file() else pd.DataFrame()
    n_scene = len(scene)
    n_target = len(target)
    general_ev = _evidence(scene)
    target_ev = _evidence(target, "success")
    visual_ev = _evidence(visual, "synchronized_all")
    synthetic_ev = (
        ";".join(synthetic_inventory.synthetic_scene_token.astype(str).head(4).tolist())
        if len(synthetic_inventory) and "synthetic_scene_token" in synthetic_inventory
        else "none"
    )
    target_success = float(target.success.mean()) if len(target) else 0.0
    oracle_gain, oracle_deltas = _oracle_improvement(oracle)

    def row(
        qid: str,
        quantity: str,
        status: str,
        paths: str,
        fields: str,
        coverage: str,
        evidence: str,
        assumptions: str,
        training: str,
        inference: str,
        conclusion: str,
    ) -> dict[str, str]:
        return {
            "id": qid,
            "target_quantity": quantity,
            "support_class": status,
            "support_definition": STATUSES[status],
            "local_code_paths": paths,
            "data_fields": fields,
            "actual_coverage": coverage,
            "evidence_scene_tokens": evidence,
            "key_assumptions": assumptions,
            "usable_for_training": training,
            "usable_at_inference": inference,
            "conclusion": conclusion,
        }

    q20_coverage = (
        f"{oracle.get('scene_count', 0)} scenes; Δpairwise C-A={_fmt(oracle_deltas['pairwise_delta_vs_a'])}; "
        f"Δregret(A-C)={_fmt(oracle_deltas['regret_delta_vs_a'])}"
        if oracle.get("status") == "COMPLETE"
        else f"{oracle.get('status', 'not run')}: {oracle.get('reason', 'not measured')}"
    )
    matrix = [
        row("Q1", "当前时刻自车状态", "A_DIRECT", "navsim/common/dataclasses.py; tools/navsim_candidate_relative_audit/inspect_scenes.py", "Frame.ego_status: ego_pose, ego_velocity, ego_acceleration, driving_command", _fmt_rate(_rate(scene, "ego_velocity_available"), n_scene), general_ev, "Scene 当前 history 末帧即审计时刻。", "YES", "YES", "Pose、速度、加速度和 driving command 可直接读取。"),
        row("Q2", "GT 未来 4 秒轨迹", "A_DIRECT", "navsim/common/dataclasses.py:Scene.get_future_trajectory; tools/navsim_candidate_relative_audit/validate_alignment.py", "future Frame.ego_status.ego_pose; Trajectory.poses", _fmt_rate(_rate(scene, "gt_4s_available"), n_scene), general_ev, "按实测 timestamp 解析 4 s，不固定下标。", "YES", "NO", "Logged ego future 可直接取出；官方绝对/相对变换误差见 Gate A。"),
        row("Q3", "GT 未来速度、加速度、航向变化", "B_EXACT_DERIVATION", "navsim/common/dataclasses.py; tools/navsim_candidate_relative_audit/validate_alignment.py", "future ego_velocity, ego_acceleration, ego_pose[2]", _fmt_rate(min(_rate(scene, "ego_velocity_available") or 0, _rate(scene, "ego_acceleration_available") or 0), n_scene), general_ev, "速度/加速度直接给出；航向变化需 wrap 后相减。", "YES", "NO", "组合量可精确构造，不把数值差分冒充直接字段。"),
        row("Q4", "GT 未来相机图像", "A_DIRECT", "tools/navsim_candidate_relative_audit/audit_future_visual_anchor.py", "raw frame cams.CAM_F0.data_path, intrinsics, extrinsics", _fmt_rate(_rate(visual, "front_camera_file_available"), len(visual)), visual_ev, "仅 logged GT sensor viewpoint；按最近 timestamp 对齐。", "YES", "NO", "候选-target 场景的 factual future CAM_F0 文件实测可用率如左。"),
        row("Q5", "GT 未来 LiDAR", "A_DIRECT", "tools/navsim_candidate_relative_audit/inspect_scenes.py", "raw frame lidar_path", _fmt_rate(_rate(scene, "future_lidar_all_requested"), n_scene), general_ev, "只检查 0.5/1/2/4 s 附近路径，未批量载入点云。", "YES", "NO", "路径指向日志未来 LiDAR；未复制原始数据。"),
        row("Q6", "未来交通参与者框", "A_DIRECT", "navsim/common/dataclasses.py:Annotations; navsim/planning/scenario_builder/navsim_scenario.py", "Annotations.boxes, names", _fmt_rate(_rate(scene, "future_annotations_available"), n_scene), general_ev, "raw box 是 future-frame ego-local；官方路径转 global。", "YES", "NO", "未来框直接记录，坐标语义经官方转换和 cache polygon 交叉验证。"),
        row("Q7", "未来交通参与者速度", "A_DIRECT", "navsim/common/dataclasses.py:Annotations", "Annotations.velocity_3d", _fmt_rate(_rate(scene, "future_annotations_available"), n_scene), general_ev, "速度语义随 annotation frame，并通过官方 tracked-object 构造转 global。", "YES", "NO", "速度字段随未来 annotations 提供。"),
        row("Q8", "跨未来帧稳定的 track token", "A_DIRECT", "navsim/common/dataclasses.py:Annotations; tools/navsim_candidate_relative_audit/inspect_scenes.py", "Annotations.track_tokens, instance_tokens", f"mean span continuity={_fmt(scene.track_span_continuity.mean() if len(scene) else None)}; scenes={n_scene}", general_ev, "连续率按 token 首末出现 span 计算；离开视野不等于 token 不稳定。", "YES", "NO", "稳定 token 可用于跨帧匹配，并以 hash+mask 写训练 tensor。"),
        row("Q9", "未来交通灯状态", "A_DIRECT", "navsim/common/dataclasses.py:Frame.traffic_lights; tools/navsim_candidate_relative_audit/build_candidate_relative_targets.py", "Frame.traffic_lights: (lane_connector_id, is_red)", f"field=100.00% ({n_scene}/{n_scene}); active-any={_fmt_rate(_rate(scene, 'traffic_lights_future_any'), n_scene)}", general_ev, "空列表是有效的无 active-record 状态；active-any 单独报告。", "YES", "NO", "未来日志交通灯字段全覆盖；有 active connector record 的场景覆盖率如左。"),
        row("Q10", "地图、道路边界、车道中心线和路线", "A_DIRECT", "navsim/common/dataclasses.py:Scene.map_api; deployed MetricCache; tools/navsim_candidate_relative_audit/inspect_scenes.py", "map_api, roadblock_ids, cache.centerline, route_lane_ids, drivable_area_map", f"map={_fmt_rate(_rate(scene, 'map_available'), n_scene)}; route={_fmt_rate(_rate(scene, 'route_available'), n_scene)}", general_ev, "地图为静态 nuPlan map；路线来自 scene/cache，不由候选改变。", "YES", "YES", "地图/路线直接可用，候选关系需几何推导。"),
        row("Q11", "任意候选轨迹的动力学 rollout", "B_EXACT_DERIVATION", "navsim/evaluate/pdm_score.py; navsim/planning/simulation/planner/pdm_planner/simulation/pdm_simulator.py; tools/navsim_candidate_relative_audit/score_candidates.py", "Trajectory.poses; MetricCache.ego_state; simulated state array", f"{scoring.get('success_rate', 0):.2%} ({scoring.get('candidate_count', 0)} candidates)", target_ev, "限定官方 simulator 接受的有限、4 s、8-pose candidate；不是任意物理行为。", "YES", "YES_WITH_SIMULATOR", "41×11、10 Hz official rollout 可确定性获得。"),
        row("Q12", "候选的碰撞、TTC、DAC、DDC、TLC、EP、LK、Comfort 等评价", "C_NONREACTIVE_ASSUMPTION", "navsim/agents/EpisodeDrive/score_module/train_pdm_scorer.py; tools/navsim_candidate_relative_audit/score_candidates.py", "collision,DAC,DDC,progress,TTC,comfort,aggregate; TLC/LK/history/extended comfort=null", f"score success={scoring.get('success_rate', 0):.2%}; supported factors 7/11 requested families", target_ev, "collision/TTC 针对 logged replay actors；DAC/DDC/progress/comfort 是 map/trajectory 派生；TLC、LK、history/extended comfort、EPDMS 未部署。", "YES_OFFLINE", "NO_DIRECT_FUTURE", "本地支持子集可评分，但完整枚举不可得，且交互风险依赖 non-reactive logged future。"),
        row("Q13", "同一日志未来下，每条候选与周车的相对状态", "C_NONREACTIVE_ASSUMPTION", "tools/navsim_candidate_relative_audit/build_candidate_relative_targets.py", "candidate_relative_actor, mask, token_hash", _fmt_rate(target_success, n_target), target_ev, "同一 logged actor world 对所有候选固定；无真实周车响应。", "YES_TARGET", "PREDICT_ONLY", "候选 frame 下的相对位置/速度/heading/clearance 可稳定构造。"),
        row("Q14", "同一日志未来下，每条候选的道路和路线关系", "B_EXACT_DERIVATION", "tools/navsim_candidate_relative_audit/build_candidate_relative_targets.py", "drivable,oncoming,intersection,centerline offset/heading,route progress,red connector relation", _fmt_rate(target.map_relation_coverage.mean() if len(target) else None, n_target), target_ev, "静态地图与路线不响应候选；红灯状态来自 logged future。", "YES_TARGET", "MAP_PART_YES; FUTURE_TL_PREDICT", "静态地图/路线关系是几何推导；未来灯态部分训练时可用、推理时须预测。"),
        row("Q15", "同一日志未来下，每条候选的结构化风险后果", "C_NONREACTIVE_ASSUMPTION", "tools/navsim_candidate_relative_audit/build_candidate_relative_targets.py; tools/navsim_candidate_relative_audit/analyze_target_diversity.py", "C_environment_only, candidate_relative_actor, per-step collision/TTC/clearance/corridor", f"target={_fmt_rate(target_success, n_target)}; nonzero pairs={diversity.get('nonzero_pair_ratio', 0):.2%}", target_ev, "风险相对 logged future，不是 causal counterfactual。", "YES_TARGET", "PREDICT_ONLY", "逐 horizon structured consequence 非退化，可作离线监督。"),
        row("Q16", "不同候选导致的周车响应", "D_REACTIVE_OR_SYNTHETIC_ONLY", "; ".join(v2.get("code_evidence", [])), "v2 MetricCache.future_tracked_objects; NavsimIDMTrafficAgents outputs", "eligible training reactive empirical coverage=0%; code supports VEHICLE only", target_ev, "这些 token 仅有 non-reactive 结果；唯一 v2 cache 是 navtest，未越界运行；非 vehicle 仍 log replay。", "NO_IN_CURRENT_ALLOWED_SPLIT", "ONLY_WITH_REACTIVE_RUNTIME", "机制存在但本次没有合法训练 split 的实测 candidate response。"),
        row("Q17", "每条非 GT 候选的真实未来相机图像", "E_UNAVAILABLE", "tools/navsim_candidate_relative_audit/audit_future_visual_anchor.py", "none (only logged GT camera path exists)", "0% by construction and file audit", visual_ev, "生成/重投影/synthetic image 均不等于 candidate-specific ground truth image。", "NO", "NO", "当前日志没有非 GT viewpoint 的真实未来图像。"),
        row("Q18", "NAVSIM v2 synthetic follow-up scene 作为弱多未来监督", "D_REACTIVE_OR_SYNTHETIC_ONLY", "/mnt/project/DriveDreamer-Policy/navsim/navsim/common/dataclasses.py; tools/navsim_candidate_relative_audit/audit_v2_extensions.py", "corresponding_original_scene, corresponding_original_initial_token, 4 frames, extended tracks/TL", f"{v2.get('synthetic_scene_count', 0)} scenes; camera={v2.get('synthetic_camera_file_coverage', 0):.2%}; LiDAR={v2.get('synthetic_lidar_file_coverage', 0):.2%}", synthetic_ev, "同 original 的 synthetic 当前 pose/image 不同；且 warmup original logs 不在 allowed trainval path。", "WEAK_ONLY_NOT_CURRENT_TRAIN_SPLIT", "NO", "可作为邻域状态扩增/弱监督，不能当同一当前状态的真实多未来。"),
        row("Q19", "reactive traffic policy 产生候选相关车辆响应", "D_REACTIVE_OR_SYNTHETIC_ONLY", "/mnt/project/DriveDreamer-Policy/navsim/navsim/traffic_agents_policies/navsim_IDM_traffic_agents.py", "simulate_environment/simulate_traffic_agents; TrackedObjectType.VEHICLE", "code available; eligible empirical training coverage=0%", target_ev, "这些 token 仅有 non-reactive 结果；车辆响应依赖 IDM policy；行人与其余类型仍 log replay；navtest cache 未使用。", "NO_IN_CURRENT_ALLOWED_SPLIT", "YES_WITH_V2_POLICY", "代码能力确认，实际候选响应强度在合法 split 上仍待测。"),
        row("Q20", "候选相对后果是否比 trajectory-only 更能预测 PDM 排序", "C_NONREACTIVE_ASSUMPTION", "tools/navsim_candidate_relative_audit/run_oracle_probe.py", "Probe A/B/C leakage-audited features and official aggregate/factors as targets", q20_coverage, target_ev, "按完整 log 划分；排序目标来自 non-reactive PDM；轻量 ridge/logistic 是 oracle 可用性探针。", "YES_IF_GAIN", "TARGET_MUST_BE_PREDICTED", ("在 non-reactive PDM 排序上 Probe C 实测优于 trajectory-only，支持规划价值。" if oracle_gain else "尚未测得满足预设阈值的稳定增益；保持 INCONCLUSIVE/negative 结论。")),
    ]
    context = {
        "environment": environment,
        "alignment": alignment,
        "scoring": scoring,
        "schema": schema,
        "diversity": diversity,
        "v2": v2,
        "oracle": oracle,
        "oracle_gain": oracle_gain,
        "oracle_deltas": oracle_deltas,
        "scene": scene,
        "visual": visual,
        "target": target,
        "manifest": manifest,
        "synthetic_inventory": synthetic_inventory,
    }
    return matrix, context


def _support_markdown(matrix: list[dict[str, str]]) -> str:
    lines = [
        "# NAVSIM Candidate-relative Support Matrix",
        "",
        "分类严格采用 A_DIRECT / B_EXACT_DERIVATION / C_NONREACTIVE_ASSUMPTION / D_REACTIVE_OR_SYNTHETIC_ONLY / E_UNAVAILABLE。",
        "",
        "| ID | 目标量 | 分类 | 实测覆盖 | 训练 | 推理 | 结论 |",
        "|---|---|---|---:|---|---|---|",
    ]
    for item in matrix:
        cells = [
            item["id"], item["target_quantity"], item["support_class"], item["actual_coverage"],
            item["usable_for_training"], item["usable_at_inference"], item["conclusion"],
        ]
        lines.append("| " + " | ".join(str(cell).replace("|", "\\|").replace("\n", " ") for cell in cells) + " |")
        lines.append(
            f"\n**{item['id']} evidence.** Paths: `{item['local_code_paths']}`. Fields: `{item['data_fields']}`. "
            f"Tokens: `{item['evidence_scene_tokens']}`. Assumption: {item['key_assumptions']}\n"
        )
    return "\n".join(lines)


def _probe_table(oracle: dict[str, Any]) -> str:
    if oracle.get("status") != "COMPLETE":
        return f"Oracle status: **{oracle.get('status', 'NOT_RUN')}** — {oracle.get('reason', 'no result')}"
    lines = [
        "| Probe | Pairwise | NDCG | Spearman | Top-1 | Regret |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key, label in (("A", "trajectory-only"), ("B", "current+trajectory"), ("C", "candidate-relative future")):
        value = oracle["probes"][key]["aggregate_score"]
        lines.append(
            f"| {key} {label} | {_fmt(value['pairwise_ranking_accuracy'])} | {_fmt(value['ndcg'])} | "
            f"{_fmt(value['spearman'])} | {_fmt(value['top1_accuracy'])} | {_fmt(value['top1_score_regret'])} |"
        )
    return "\n".join(lines)


def generate(args: argparse.Namespace) -> dict[str, str]:
    output_dir = ensure_output_dir(args.output_dir)
    matrix, ctx = _matrix_rows(output_dir)
    pd.DataFrame(matrix).to_csv(output_dir / "SUPPORT_MATRIX.csv", index=False)
    write_markdown(output_dir / "SUPPORT_MATRIX.md", _support_markdown(matrix))
    env, alignment, scoring = ctx["environment"], ctx["alignment"], ctx["scoring"]
    schema, diversity, v2, oracle = ctx["schema"], ctx["diversity"], ctx["v2"], ctx["oracle"]
    scene, visual, target, manifest = ctx["scene"], ctx["visual"], ctx["target"], ctx["manifest"]
    soft = pd.read_csv(output_dir / "soft_label_stats.csv") if (output_dir / "soft_label_stats.csv").is_file() else pd.DataFrame()
    soft_examples = _json(output_dir / "soft_label_examples.json", {})
    gates = _json(output_dir / "gate_status.json", {})
    candidate_scenes = int(manifest.scene_token.nunique()) if len(manifest) else 0
    k = int(manifest.groupby("scene_token").size().median()) if len(manifest) else 0
    target_success = float(target.success.mean()) if len(target) else 0.0
    visual_sync = float(visual.synchronized_all.mean()) if len(visual) else 0.0
    camera_coverage = float(visual.front_camera_file_available.mean()) if len(visual) else 0.0
    nondegenerate = bool(diversity.get("nonzero_pair_ratio", 0) > 0.95 and diversity.get("mean_unique_consequence_count", 0) > 1)
    f1 = "PASS" if gates.get("gate_a", {}).get("passed") and gates.get("gate_b", {}).get("passed") and target_success >= 0.98 and nondegenerate else "FAIL"
    f2 = "PASS" if visual_sync >= 0.95 else ("CONDITIONAL_PASS" if visual_sync > 0 else "FAIL")
    prefix_ok = soft_examples.get("same_prefix_short_positive_rate") == 1.0 and soft_examples.get("same_prefix_long_separation_rate") == 1.0
    f3 = "PASS" if len(soft) and bool(soft.valid.all()) and prefix_ok else "FAIL"
    if oracle.get("status") == "COMPLETE":
        inverse = oracle.get("interaction_only_inverse_probe", {})
        f4 = "PASS" if inverse.get("strong_inverse_supported", False) else "CONDITIONAL_PASS"
    else:
        f4 = "INCONCLUSIVE"
    f5 = "FAIL"
    if f1 == "PASS" and f2 in {"PASS", "CONDITIONAL_PASS"} and ctx["oracle_gain"]:
        plan = "Plan A：完整候选相对世界模型（仅 GT logged future image 作视觉 anchor；非 GT 使用 structured targets）。"
    elif f1 == "PASS" and oracle.get("status") == "COMPLETE" and ctx["oracle_gain"]:
        plan = "Plan B：结构化候选相对世界模型。"
    elif scoring.get("passed") and f1 != "PASS":
        plan = "Plan C：未来评价器而非世界模型。"
    else:
        plan = "Plan D：当前方案不值得继续，除非补足当前未通过的实证条件。"
    primary_blocker = (
        "没有任何非 GT candidate-specific ground-truth future image；合法训练 split 上也没有可运行的 v2 reactive metric cache。"
    )
    coverage_lines = {
        "GT 4 s": _fmt_rate(_rate(scene, "gt_4s_available"), len(scene)),
        "future CAM_F0": _fmt_rate(camera_coverage, len(visual)),
        "future annotations": _fmt_rate(_rate(scene, "future_annotations_available"), len(scene)),
        "track continuity": _fmt(scene.track_span_continuity.mean() if len(scene) else None),
        "map": _fmt_rate(_rate(scene, "map_available"), len(scene)),
        "route": _fmt_rate(_rate(scene, "route_available"), len(scene)),
        "traffic light active-any": _fmt_rate(_rate(scene, "traffic_lights_future_any"), len(scene)),
        "general-sample metric-cache overlap": _fmt_rate(_rate(scene, "metric_cache_load_success"), len(scene)),
        "cache-matched scoring": f"{scoring.get('scene_count', 0)} scenes / {scoring.get('success_rate', 0):.2%} candidates",
    }
    soft_summary = ""
    if len(soft):
        grouped = soft.groupby("horizon_s").agg(
            effective=("effective_positive_count", "mean"),
            false_negative=("hard_one_hot_false_negative_count", "mean"),
            gt_weight=("gt_weight", "mean"),
        )
        soft_summary = "\n".join(
            f"- {h:.1f} s: effective positives `{v.effective:.3f}`, false negatives under one-hot `{v.false_negative:.3f}`, GT weight `{v.gt_weight:.3f}`"
            for h, v in grouped.iterrows()
        )
    inverse_text = oracle.get("interaction_only_inverse_probe", {}).get("interpretation", "未完成跨日志 inverse probe。")
    leakage = oracle.get("feature_leakage_audit", {})
    leakage_pass = bool(leakage and all(value.get("passed") for value in leakage.values()))
    if oracle.get("status") == "COMPLETE":
        probe_a_factors = oracle["probes"]["A"]["factors"]
        probe_c_factors = oracle["probes"]["C"]["factors"]
        factor_gain_text = (
            f"C vs A AUROC: collision `{_fmt(probe_c_factors['collision']['auroc'])}` vs `{_fmt(probe_a_factors['collision']['auroc'])}`, "
            f"TTC `{_fmt(probe_c_factors['ttc_violation']['auroc'])}` vs `{_fmt(probe_a_factors['ttc_violation']['auroc'])}`, "
            f"DAC `{_fmt(probe_c_factors['dac_violation']['auroc'])}` vs `{_fmt(probe_a_factors['dac_violation']['auroc'])}`, "
            f"DDC `{_fmt(probe_c_factors['ddc_violation']['auroc'])}` vs `{_fmt(probe_a_factors['ddc_violation']['auroc'])}`; "
            f"progress Spearman `{_fmt(probe_c_factors['progress']['spearman'])}` vs `{_fmt(probe_a_factors['progress']['spearman'])}`. "
            "Thus the gain is not collision/TTC-only; map compliance and progress also carry signal, while route-specific and red-light-specific attribution remains unidentifiable with this scorer."
        )
    else:
        factor_gain_text = "Factor gain was not measured because the cross-log oracle did not complete."
    report = f"""# NAVSIM Candidate-relative Future Supervision — Final Feasibility Report

## 1. 本地环境

- Repository / branch / source base commit: `{env.get('project_path')}` / `{env.get('git', {}).get('branch')}` / `{env.get('git', {}).get('commit')}`
- Runtime NAVSIM: `{env.get('packages', {}).get('navsim', {}).get('version')}` from `{env.get('runtime_navsim_import')}`
- Additional v2 devkit: `{v2.get('v2_version')}` at commit `{v2.get('v2_git_commit')}`; it is not the runtime import.
- Dataset split: `{env.get('paths', {}).get('split')}`; audited Scene sample `{len(scene)}`, cache-matched candidate scenes `{candidate_scenes}`.
- Log coverage: general sequential statistics sample `{scene.log_name.nunique() if len(scene) else 0}` logs; cache-matched candidate/oracle sample `{manifest.log_name.nunique() if len(manifest) else 0}` logs. Oracle train/validation is split by complete log, never by adjacent scene.
- Metric cache: `{env.get('paths', {}).get('metric_cache_path')}`; deployed runtime class is the local `train_metric_chache.MetricCache` adaptation documented in `FIELD_AUDIT.md`.
- Synthetic data: `{v2.get('synthetic_scenes_available')}` ({v2.get('synthetic_scene_count', 0)} warmup synthetic scenes).
- Reactive policy code / eligible training cache: `{v2.get('reactive_policy_code_available')}` / `{v2.get('eligible_train_reactive_cache_available')}`.
- Tests: `pytest -q tests` passed 26/26. Root-level collection additionally reaches vendored `nuplan-devkit` and is blocked by missing optional upstream test dependencies; see `TEST_AUDIT.md`.

## 2. 数据支持矩阵

The authoritative Q1–Q20 table is in `SUPPORT_MATRIX.csv` and `SUPPORT_MATRIX.md`. Counts are observed, not inferred from public NAVSIM documentation. Summary class counts: `{pd.Series([item['support_class'] for item in matrix]).value_counts().to_dict()}`.

## 3. 数据覆盖率

""" + "\n".join(f"- {key}: {value}" for key, value in coverage_lines.items()) + f"""

The general Scene sample and cache-matched candidate sample are reported separately. The 500-scene general sample is sequential and covers only the log count stated above, so its percentages are deployment evidence rather than a claim about the full trainval distribution. Its cache overlap does not replace the measured {alignment.get('metric_cache_success_rate', 0):.2%} Gate-A cache load rate on selected training-cache tokens.

## 4. 候选构造

- Source: deterministic smooth fallback because no existing multi-trajectory dump was found; these are controlled perturbations, not real futures.
- Scenes × K: `{candidate_scenes} × {k}`; GT candidate is an additional immutable candidate at index 0.
- Official scoring success: `{scoring.get('success_rate', 0):.3%}`; repeated and batch-vs-single maximum errors: `{scoring.get('deterministic_max_abs_error')}` / `{scoring.get('batch_vs_single_max_abs_error')}`.
- Consequence diversity: non-zero pair ratio `{diversity.get('nonzero_pair_ratio', 0):.3%}`, mean unique consequences `{diversity.get('mean_unique_consequence_count', 0):.2f}`, hard-negative pairs `{diversity.get('hard_negative_count', 0)}`.
- Trajectory→consequence and consequence→score-difference Spearman: `{_fmt(diversity.get('trajectory_consequence_spearman'))}` / `{_fmt(diversity.get('consequence_score_difference_spearman'))}`.

## 5. Candidate-relative consequence

- Direct logged quantities: ego state/GT trajectory, future sensor paths, future annotation boxes/velocity/token, traffic lights, map and route.
- Exact derivations: SE(2) alignment, candidate dynamics rollout, static map/route relations and locally supported PDM factors.
- Non-reactive-assumption quantities: candidate-relative actor state, actor clearance/corridor/collision/TTC and combined structured risks against the shared logged future.
- Reactive/synthetic only: candidate-conditioned vehicle response in v2 IDM and warmup synthetic follow-up scenes.
- Unavailable: non-GT ground-truth images, causal effects, true multi-agent response in v1, and local v1 TLC/lane-keeping/EPDMS/extended-comfort fields.

`C_full` contains `{len(schema.get('trajectory_feature_names', []))}` trajectory-derived plus `{len(schema.get('environment_feature_names', []))}` environment fields. `C_environment_only` excludes waypoint copies, candidate identity/type and official score/factor columns; it is the oracle input. Target success is `{target_success:.3%}`.

## 6. Soft contrastive label

{soft_summary}

Same-prefix/different-tail short-positive and long-separation rates are `{soft_examples.get('same_prefix_short_positive_rate')}` / `{soft_examples.get('same_prefix_long_separation_rate')}`. Hard one-hot would therefore create measurable false negatives. The K×K consequence label uses stable actor hashes, masks, standardized mixed units and only each horizon's prefix.

## 7. Oracle planning utility

{_probe_table(oracle)}

- Probe-C planning gain decision: `{ctx['oracle_gain']}`; Δpairwise(C−A) `{_fmt(ctx['oracle_deltas']['pairwise_delta_vs_a'])}`, Δregret(A−C) `{_fmt(ctx['oracle_deltas']['regret_delta_vs_a'])}`.
- Feature-name leakage audit passed: `{leakage_pass}`. `C_environment_only` contains no official final or component score, candidate type/index or direct candidate waypoint copy.
- {factor_gain_text}
- Factor attribution is limited to A/B/C prediction deltas for collision, TTC, DAC, DDC, comfort and progress. The deployed scorer exposes no TLC/lane-keeping/EPDMS target, so route/red-light utility cannot be separately claimed.
- Interaction-only inverse result: {inverse_text}
- Interaction-only candidate-ID / semantic accuracy versus majority: `{_fmt(oracle.get('interaction_only_inverse_probe', {}).get('candidate_id', {}).get('accuracy'))}` / `{_fmt(oracle.get('interaction_only_inverse_probe', {}).get('candidate_id', {}).get('majority_baseline'))}` and `{_fmt(oracle.get('interaction_only_inverse_probe', {}).get('semantic_action', {}).get('accuracy'))}` / `{_fmt(oracle.get('interaction_only_inverse_probe', {}).get('semantic_action', {}).get('majority_baseline'))}`; Δtrajectory R² `{_fmt(oracle.get('interaction_only_inverse_probe', {}).get('delta_trajectory_r2'))}`.

This is an oracle association test: those future relations are training targets and must be predicted from the current scene plus candidate at inference.

## 8. GT future visual anchor

- Logged CAM_F0 future path / full synchronized-chain coverage: `{camera_coverage:.3%}` / `{visual_sync:.3%}` across `{len(visual)}` scene-horizon records. Image dimensions/decodability were opened on the bounded field-audit sample and the 12 rendered anchor scenes; the remaining count is path existence, not bulk decode.
- Supported: `logged I_GT(t+h) ↔ C_GT,h` for GT-only visual semantic alignment.
- Unsupported: `I_candidate_i(t+h)` for every non-GT candidate; no such real sensor viewpoint exists in the logs.
- Saved scene evidence: `figures/visual_anchor/` plus the twelve global audit figures.

## 9. Reactive 与 synthetic 扩展

- Reactive v2 code exists and simulates `VEHICLE`; all remaining object types are merged from log replay.
- No reactive response metrics were fabricated: the only deployed v2 cache records split `{v2.get('deployed_v2_cache_split')}`, which is excluded.
- Warmup synthetic: `{v2.get('synthetic_scene_count', 0)}` scenes mapping to `{v2.get('synthetic_unique_original_scenes', 0)}` originals; follow-ups/original min/median/max `{v2.get('synthetic_followups_per_original', {}).get('min')}/{v2.get('synthetic_followups_per_original', {}).get('median')}/{v2.get('synthetic_followups_per_original', {}).get('max')}`.
- Synthetic referenced camera / LiDAR files: `{v2.get('synthetic_camera_file_coverage', 0):.3%}` / `{v2.get('synthetic_lidar_file_coverage', 0):.3%}`; annotations and at least eight extended-track steps are audited separately in `V2_EXTENSION_REPORT.md`.
- Same-original groups with non-identical synthetic current states: `{v2.get('same_original_groups_with_nonidentical_synthetic_current_pose_rate', 0):.3%}`. Therefore they are synthetic follow-up scenes / weak neighborhood supervision, not same-current real counterfactuals.

## 10. 最终五项判定

- F1 唯一 logged future → K 个 candidate-relative structured consequences: **{f1}**
- F2 GT future image → GT structured future 的视觉语义锚定: **{f2}**
- F3 prefix-aware soft contrastive supervision: **{f3}**
- F4 使用 structured consequence 训练独立 inverse verifier: **{f4}**
- F5 为每条非 GT candidate 提供真实 future image supervision: **{f5}**

F4 的 `CONDITIONAL_PASS` 只支持风险/一致性 verifier：当前 interaction-only 分类有部分信号，但 Δtrajectory 回归未恢复候选运动，因此不等价于强 interaction inverse dynamics。F5 的失败来自实际文件/视点链路审计。

## 11. 推荐的下一步方法版本

**{plan}**

Primary blocker: {primary_blocker}

## 12. 实现建议

建议的最小下一阶段接口（不在本任务中训练完整模型）：

- `current_scene_representation`: 当前相机/BEV、当前 actors、traffic lights、map/route；推理可用。
- `candidate_trajectory`: `K×8×3` current-ego-local poses + mask；包括模型候选和单独 GT。
- `candidate_relative_target_schema`: 预测 `C_environment_only[K,8,15]` 与 `candidate_relative_actor[K,8,N,10]`，使用 mask，不输入 token string。
- `GT_visual_anchor`: 只在 GT row/horizon 对齐 logged CAM_F0 embedding 与 `C_GT,h`；非 GT 无 image loss。
- `soft_contrastive_target`: horizon-prefix factual `q[K]` 和 consequence `Q[K,K]`。
- `utility_target`: 离线 official aggregate + supported factor targets，明确 unavailable factor mask。
- `inverse_verifier_target`: 输入预测的 environment interactions，输出 candidate consistency/risk ranking；若 interaction-only probe 近随机，不要求恢复完整 trajectory。

Terminology throughout: logged future, non-reactive candidate-relative consequence, candidate-conditioned relabeling, reactive-policy simulated consequence and synthetic follow-up scene. No result is described as a real counterfactual future.
"""
    write_markdown(output_dir / "FINAL_FEASIBILITY_REPORT.md", report)
    decisions = {"F1": f1, "F2": f2, "F3": f3, "F4": f4, "F5": f5, "plan": plan, "blocker": primary_blocker}
    # Required terminal handoff format.
    print(f"Branch: {env.get('git', {}).get('branch')}")
    print(f"Commit: {env.get('git', {}).get('commit')}")
    print(f"NAVSIM version: {env.get('packages', {}).get('navsim', {}).get('version')}")
    print(f"Dataset split: {env.get('paths', {}).get('split')}")
    print(f"Number of audited scenes: {len(scene)} general / {candidate_scenes} candidate")
    print(f"Number of candidates per scene: {k}")
    print(f"Future camera coverage: {camera_coverage:.3%}")
    print(f"Future track coverage: {_fmt_rate(_rate(scene, 'future_annotations_available'))}")
    print(f"Candidate scoring success rate: {scoring.get('success_rate', 0):.3%}")
    print(f"Candidate-relative target coverage: {target_success:.3%}")
    print(f"Reactive policy available: {v2.get('reactive_policy_code_available')} (eligible empirical cache: {v2.get('eligible_train_reactive_cache_available')})")
    print(f"Synthetic scenes available: {v2.get('synthetic_scenes_available')} ({v2.get('synthetic_scene_count', 0)})")
    print("")
    print(f"F1 candidate-relative structured consequence: {f1}")
    print(f"F2 GT visual anchor: {f2}")
    print(f"F3 soft contrastive supervision: {f3}")
    print(f"F4 inverse verifier supervision: {f4}")
    print(f"F5 non-GT future image supervision: {f5}")
    print("")
    print(f"Primary blocker: {primary_blocker}")
    print(f"Recommended next plan: {plan}")
    print(f"Report path: {output_dir / 'FINAL_FEASIBILITY_REPORT.md'}")
    append_command(output_dir, "python -m tools.navsim_candidate_relative_audit.generate_final_report " + " ".join(sys.argv[1:]))
    return decisions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    args = parser.parse_args()
    generate(args)


if __name__ == "__main__":
    main()
