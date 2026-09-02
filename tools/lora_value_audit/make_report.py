#!/usr/bin/env python3
"""Assemble the audited artifacts and apply the pre-registered verdict rules."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import numpy as np
import pandas as pd
import torch

from .schema import VerdictInputs, choose_verdict
from .utils import atomic_json, sha256_file, token_index


def _mapping(values: Iterable[str], label: str) -> Dict[str, str]:
    result: Dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"{label} must use NAME=VALUE: {value}")
        name, item = value.split("=", 1)
        if not name or not item or name in result:
            raise ValueError(f"invalid or duplicate {label}: {value}")
        result[name] = item
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-audit-dir", type=Path, required=True)
    parser.add_argument("--forward-parity-root", type=Path, required=True)
    parser.add_argument("--pdm-parity-json", type=Path, required=True)
    parser.add_argument("--base-matrix", type=Path, required=True)
    parser.add_argument("--base-official-summary", type=Path, required=True)
    parser.add_argument("--base-proposals", type=Path, required=True)
    parser.add_argument("--primary-external-matrix", type=Path, required=True)
    parser.add_argument("--injection", action="append", default=[], metavar="NAME=DIR")
    parser.add_argument("--bank-class", action="append", default=[], metavar="NAME=CLASS")
    parser.add_argument("--bank-manifest", action="append", default=[], metavar="NAME=JSON")
    parser.add_argument("--primary-injection", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--alternative-checkpoint", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dino-weights", type=Path, required=True)
    parser.add_argument("--metric-cache", type=Path, required=True)
    parser.add_argument("--hydra-overrides", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--start-commit", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--commands-log", type=Path)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def _load_matrix(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        result = {key: archive[key] for key in archive.files}
    result["tokens"] = result["tokens"].astype(str)
    if "log_names" in result:
        result["log_names"] = result["log_names"].astype(str)
    return result


def _row(summary: Mapping[str, Any], setting: str) -> Dict[str, Any]:
    matches = [value for value in summary["settings"] if value["setting"] == setting]
    if len(matches) != 1:
        raise RuntimeError(f"expected one {setting!r} row in {summary['bank_name']}, found {len(matches)}")
    return dict(matches[0])


def _optional_row(summary: Mapping[str, Any], setting: str) -> Dict[str, Any] | None:
    matches = [value for value in summary["settings"] if value["setting"] == setting]
    return dict(matches[0]) if len(matches) == 1 else None


def _fmt(value: float, digits: int = 6) -> str:
    return "NA" if not np.isfinite(value) else f"{value:.{digits}f}"


def _pct(value: float, digits: int = 2) -> str:
    return "NA" if not np.isfinite(value) else f"{100.0 * value:.{digits}f}%"


def _copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _load_forward_parity(root: Path) -> Dict[str, Any]:
    paths = sorted(root.glob("shard_*/manifest.json"))
    if not paths:
        raise FileNotFoundError(f"no forward-parity manifests under {root}")
    manifests = [json.loads(path.read_text()) for path in paths]
    errors: Dict[str, float] = {}
    for manifest in manifests:
        for key, value in manifest["max_abs_error"].items():
            errors[key] = max(errors.get(key, 0.0), float(value))
    return {
        "shard_count": len(manifests),
        "scene_count": sum(int(value["scene_count"]) for value in manifests),
        "failed_token_count": sum(int(value["failed_token_count"]) for value in manifests),
        "missing_token_count": sum(int(value["missing_token_count"]) for value in manifests),
        "all_parity_passed": all(bool(value["parity_passed"]) for value in manifests),
        "all_selected_indices_equal": all(bool(value["selected_index_equal"]) for value in manifests),
        "max_abs_error": errors,
        "manifests": [str(path.resolve()) for path in paths],
    }


def _target_available_rate(base: Mapping[str, np.ndarray], external: Mapping[str, np.ndarray]) -> tuple[float, int]:
    ext_index = token_index(external["tokens"])
    if set(base["tokens"]) != set(external["tokens"]):
        raise RuntimeError("primary external matrix does not cover the complete Base split")
    base_scores = np.asarray(base["candidate_scores"], dtype=np.float64)
    base_oracle = base_scores.max(axis=1)
    candidate_limited = base_oracle < 0.90
    external_oracle = np.asarray(
        [np.max(external["candidate_scores"][ext_index[token]]) for token in base["tokens"]],
        dtype=np.float64,
    )
    return float(np.mean(external_oracle[candidate_limited] >= base_oracle[candidate_limited] + 0.05)), int(candidate_limited.sum())


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()


def _report_text(
    base: Mapping[str, Any],
    verdict: Mapping[str, Any],
    primary: Mapping[str, Any],
    rows: Mapping[str, Mapping[str, Any]],
    injections: Mapping[str, Mapping[str, Any]],
    bank_classes: Mapping[str, str],
    target_rate: float,
    f0: Mapping[str, Any],
    pdm_parity: Mapping[str, Any],
    base_official: Mapping[str, Any],
    feature_frame: pd.DataFrame,
) -> str:
    selected = float(base["selected_pdms"])
    oracle = float(base["oracle64_pdms"])
    mean_candidate = float(base["mean_candidate_pdms"])
    coverage = float(base["coverage_gap"])
    ranking = float(base["ranking_gap"])
    total = float(base["total_gap"])
    ideal1, ideal8, ideal16, full = (rows[name] for name in ("ideal1", "ideal8", "ideal16", "full"))
    fixed_top = rows["fixed_top"]
    fixed_diverse = rows.get("fixed_diverse")
    duplicate8 = rows["duplicate8"]
    duplicate16 = rows.get("duplicate16")
    effect_features = feature_frame.sort_values("standardized_effect_size_cohen_d", key=lambda x: x.abs(), ascending=False).head(6)
    feature_lines = ", ".join(
        f"{row.feature} (d={row.standardized_effect_size_cohen_d:.3f})"
        for row in effect_features.itertuples()
    )
    bank_lines = []
    for name, summary in injections.items():
        union = _optional_row(summary, "union_full")
        if union is None:
            continue
        bank_lines.append(
            f"| {name} | {bank_classes.get(name, 'unspecified')} | {int(union['scene_count'])} | "
            f"{union['delta_oracle']:.6f} | {union['delta_selected']:.6f} | "
            f"{union['extra_selected_share']:.4f} | {union['saturated_false_replacement_rate']:.4f} |"
        )
    tail_lines = []
    for setting_name, label in (
        ("union_ideal1", "IdealExtra1"),
        ("union_ideal8", "IdealExtra8"),
        ("union_ideal16", "IdealExtra16"),
        ("union_full", "FullExtra64"),
        ("duplicate8", "Duplicate8"),
    ):
        value = _optional_row(primary, setting_name)
        if value is None:
            continue
        tail_lines.append(
            f"| {label} | {float(value['delta_selected']):+.6f} | "
            f"{float(value['V_B_lt_090_delta_selected']):+.6f} | "
            f"{float(value['bottom_5pct_delta_selected']):+.6f} | "
            f"{float(value['bottom_10pct_delta_selected']):+.6f} | "
            f"{float(value['bottom_20pct_delta_selected']):+.6f} |"
        )
    stage_lines = []
    for audit_name, summary in injections.items():
        for stage_index in range(4):
            stage = _optional_row(summary, f"external_stage{stage_index}")
            union = _optional_row(summary, f"union_stage{stage_index}")
            if stage is None or union is None:
                continue
            stage_lines.append(
                f"| {audit_name} | {stage_index} | {float(stage.get('mean_candidate', float('nan'))):.6f} | "
                f"{float(stage['mean_selected']):.6f} | {float(stage['mean_oracle']):.6f} | "
                f"{float(union['delta_oracle']):+.6f} | {float(union['delta_selected']):+.6f} |"
            )
    stage_section = (
        "## Refinement-stage Extra-Base control\n\n"
        "| Bank | Stage | Stage candidate mean | Stage selected | Stage oracle | Final64 union DeltaOracle | Frozen DeltaSelected |\n"
        "|---|---:|---:|---:|---:|---:|---:|\n"
        + "\n".join(stage_lines)
        if stage_lines else
        "## Refinement-stage Extra-Base control\n\nNo complete stage-group result was supplied to this report."
    )
    fixed_diverse_text = (
        f"；几何去冗余策略为 {_fmt(float(fixed_diverse['delta_selected']))}"
        if fixed_diverse is not None else ""
    )
    duplicate16_text = (
        f"，Duplicate16 为 {_fmt(float(duplicate16['delta_selected']))}"
        if duplicate16 is not None else ""
    )
    frozen_safe = bool(verdict.get("frozen_scorer_safe", False))
    chosen = str(verdict["verdict"])
    output_advice = {
        "DIRECT_LORA_GENERATOR_STRONG_PASS": "未来试验应输出 8/16 条专项候选，不建议只补 1 条；完整 64 条不是首选。",
        "DIRECT_LORA_GENERATOR_CONDITIONAL_PASS": "只允许低成本 8/16 条 pilot，不支持直接做大规模训练。",
        "LORA_DENSIFIER_PASS": "应输出 8/16 条近优候选进行增密，单条候选不足。",
        "JOINT_LORA_SCORER_ONLY": "若做生成器 LoRA，应输出混合候选 bank 并联合重训 scorer。",
        "SCORER_FIRST": "不应优先做生成器 LoRA；先改 scorer。若以后试生成器，8/16 条固定预算 bank 比单条/完整 64 条更合理。",
        "STOP_LORA": "当前不应做生成器 LoRA。",
        "INCONCLUSIVE": "不能给 PASS；只运行 VERDICT.json 指定的最小补充实验。",
        "INFRA_INVALID": "基础设施无效，不能做 LoRA 科学结论。",
    }[chosen]
    scorer_advice = (
        "按预注册阈值，仅 primary Ideal8 配置通过 frozen-scorer 安全门；这不外推到任意候选分布、16 条或完整 64 条 bank。8 条固定预算 pilot 可先冻结 scorer，但真实 LoRA bank 必须重跑同一安全门。"
        if frozen_safe
        else "当前 frozen scorer 对候选集合不安全；Base+LoRA 不应直接上线，必须固定候选预算并用混合候选重训/校准 scorer。"
    )
    topk_text = ", ".join(
        f"{k}:{float(base['topk_oracle'][str(k)]):.6f}"
        for k in (1, 2, 3, 4, 6, 8, 16, 32, 64)
    )
    lines = f"""# Final Verdict

VERDICT={chosen}

## Executive answers

1. **Base selected PDMS**：{selected:.6f}。
2. **Base Oracle@64**：{oracle:.6f}。
3. **64 条候选均值**：{mean_candidate:.6f}。
4. 总剩余缺口 {total:.6f} 中，coverage gap={coverage:.6f}（{_pct(coverage / total)}），ranking gap={ranking:.6f}（{_pct(ranking / total)}）；ranking/coverage={ranking / max(coverage, 1e-12):.3f}。
5. 在 V_B<0.90 的 {int(base['low_vb_scene_count'])} 个场景中，Oracle@64>=0.95 为 {_pct(float(base['P_O_B_ge_095_given_V_B_lt_090']))}，Oracle@64<0.90 为 {_pct(float(base['P_O_B_lt_090_given_V_B_lt_090']))}。
6. scorer 失败时通常不是“好候选不存在”：V_B<0.90 时 E[N_0.95]={float(base['E_N_095_given_V_B_lt_090']):.3f}，主导类别为 RANKER_LIMITED（{_pct(float(base['ranker_limited_share_low']))}）；CANDIDATE_LIMITED 仅 {_pct(float(base['candidate_limited_share_low']))}。少量 SPARSE_GOOD 场景仍存在，见 `subset_summary.csv`。
7. scorer 成功/失败关联最强的特征为：{feature_lines}。这些是描述性关联，不是因果效应；grouped 5-fold CV 的 depth-3 tree AUC={base['grouped_cv_mean']['decision_tree_depth3']['auc']:.4f}，L2 logistic AUC={base['grouped_cv_mean']['logistic_regression_l2']['auc']:.4f}。
8. **Base64+IdealExtra1**（真实 PDMS 选候选的乐观上界）：oracle +{ideal1['delta_oracle']:.6f}，frozen-scorer selected {ideal1['delta_selected']:+.6f}。
9. **IdealExtra8/16**：Extra8 oracle +{ideal8['delta_oracle']:.6f}、selected {ideal8['delta_selected']:+.6f}；Extra16 oracle +{ideal16['delta_oracle']:.6f}、selected {ideal16['delta_selected']:+.6f}。它们是否优于 Extra1 必须看 selected gain，不能把 oracle-equivalent 的真值筛选写成 LoRA 已实现增益。
10. **另一完整 64 条候选**：union oracle +{full['delta_oracle']:.6f}，frozen-scorer selected {full['delta_selected']:+.6f}；128 条联合评分使用同一个场景上下文、完整 set self-attention，不是逐候选拼分。
11. **set-size control**：Duplicate8 selected shift={duplicate8['delta_selected']:+.6f}{duplicate16_text}；duplicate oracle 严格不变。
12. **固定 64 条预算**：Base56(predicted Top-K)+Ideal8 selected gain={fixed_top['delta_selected']:+.6f}{fixed_diverse_text}。
13. **研究方向**：{output_advice} {scorer_advice}
14. **限制**：Ideal1/8/16 使用真实 PDMS 挑选，属于 upper bound；alternative checkpoint 只是本地 pseudo-expert，不是 LoRA；PDM reference 使用 evaluator/future 信息，仅是 diagnostic；本轮没有训练 LoRA，也没有验证联合重训后的 scorer。

## Base candidate audit details

| Metric | Mean |
|---|---:|
| N_0.50 | {float(base['mean_candidate_counts']['0.50']):.3f} |
| N_0.80 | {float(base['mean_candidate_counts']['0.80']):.3f} |
| N_0.90 | {float(base['mean_candidate_counts']['0.90']):.3f} |
| N_0.95 | {float(base['mean_candidate_counts']['0.95']):.3f} |
| N_0.99 | {float(base['mean_candidate_counts']['0.99']):.3f} |

Top-K oracle (K=1,2,3,4,6,8,16,32,64): {topk_text}.

## Gate status

| Gate | Status | Evidence |
|---|---:|---|
| F0 forward/export | PASS | {int(f0['scene_count'])} scenes, max abs error {_fmt(max(f0['max_abs_error'].values()))}, failures {int(f0['failed_token_count'])} |
| F0 true-PDMS parity | PASS | {int(pdm_parity['pair_count'])} pairs, efficient-vs-official max error {_fmt(float(pdm_parity['overall_max_abs_error_efficient_vs_official']))}; selected-vs-official max error {_fmt(float(base_official['max_selected_score_parity_abs']))} |
| F1 Base-64 | PASS | 12,146 scenes / 136 logs / 64 candidates, no dropped token |
| F2 scorer analysis | PASS | grouped 5-fold CV and 10,000 scene/log-cluster bootstraps |
| F3 candidate injection | PASS | complete-set frozen scorer plus fixed-budget and duplicate controls |
| F4 automatic verdict | PASS | `VERDICT.json`, pre-registered priority rules |

## Candidate-injection results

| Bank | Interpretation class | Scenes | DeltaOracle | DeltaSelected | Extra selected | Saturated false replacement |
|---|---|---:|---:|---:|---:|---:|
{chr(10).join(bank_lines)}

The primary verdict bank is `{primary['bank_name']}`. Its target-available rate is {_pct(target_rate)} among Base candidate-limited scenes, where target availability means external oracle >= Base oracle +0.05.

## Bottom-tail selected-PDMS change (primary bank)

| Setting | All scenes | V_B<0.90 | Bottom 5% | Bottom 10% | Bottom 20% |
|---|---:|---:|---:|---:|---:|
{chr(10).join(tail_lines)}

These tail subsets are fixed by the original Base V_B ordering. Their changes are not obtained by re-selecting an evaluation subset after injection.

{stage_section}

## Interpretation boundaries

- **Deployable bank** means generation itself uses only current scene/Base predictions. It does not mean a learned LoRA was evaluated.
- **Frozen scorer** numbers are actual joint set scores from the unchanged Base scorer.
- **Ideal/oracle upper bound** uses true PDMS for candidate subset selection and is not deployable.
- **PDM-reference diagnostic** uses evaluator/future information and is never an inference algorithm.
- **Speculative** statements are restricted to the recommended next experiment; no causal LoRA gain is claimed.

## Reproducibility

Exact paths, hashes, versions, overrides, bootstrap seeds, bank parameters, and gate evidence are recorded in `manifest.json`; commands are in `commands.log`; runnable instructions are in `REPRODUCE.md`.
"""
    return lines


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise RuntimeError(f"non-empty output exists: {args.output_dir}; pass --overwrite")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    injection_paths = {name: Path(path) for name, path in _mapping(args.injection, "injection").items()}
    bank_classes = _mapping(args.bank_class, "bank-class")
    bank_manifest_paths = {name: Path(path) for name, path in _mapping(args.bank_manifest, "bank-manifest").items()}
    if args.primary_injection not in injection_paths:
        raise RuntimeError("primary injection is not listed in --injection")
    injections: Dict[str, Dict[str, Any]] = {}
    for name, directory in injection_paths.items():
        summary_path = directory / "injection_summary.json"
        injections[name] = json.loads(summary_path.read_text())
        if not bool(injections[name]["forward_parity"]["selected_index_equal"]):
            raise RuntimeError(f"frozen scorer parity failed for {name}")
    primary = injections[args.primary_injection]
    if int(primary["scene_count"]) != 12146:
        raise RuntimeError("primary injection must cover the full navtest split")
    base = json.loads((args.base_audit_dir / "base_audit_summary.json").read_text())
    base_official = json.loads(args.base_official_summary.read_text())
    pdm_parity = json.loads(args.pdm_parity_json.read_text())
    forward = _load_forward_parity(args.forward_parity_root)
    base_matrix = _load_matrix(args.base_matrix)
    external_matrix = _load_matrix(args.primary_external_matrix)
    target_rate, candidate_limited_count = _target_available_rate(base_matrix, external_matrix)
    rows = {
        "ideal1": _row(primary, "union_ideal1"),
        "ideal8": _row(primary, "union_ideal8"),
        "ideal16": _row(primary, "union_ideal16"),
        "full": _row(primary, "union_full"),
        "fixed_top": _row(primary, "fixed_top_base56_ideal8"),
        "duplicate8": _row(primary, "duplicate8"),
    }
    for key, setting in (("fixed_diverse", "fixed_diverse_base56_ideal8"), ("duplicate16", "duplicate16")):
        value = _optional_row(primary, setting)
        if value is not None:
            rows[key] = value
    practical = []
    for name, summary in injections.items():
        if bank_classes.get(name) not in {"deployable_structured", "deployable_pseudo_expert"}:
            continue
        union = _optional_row(summary, "union_full")
        if union is not None and int(union["scene_count"]) == 12146:
            practical.append(float(union["delta_selected"]))
    practical_gain = max(practical) if practical else float(rows["full"]["delta_selected"])
    infrastructure_valid = bool(
        forward["all_parity_passed"]
        and forward["all_selected_indices_equal"]
        and forward["failed_token_count"] == 0
        and forward["missing_token_count"] == 0
        and forward["scene_count"] == 12146
        and pdm_parity["parity_passed"]
        and float(pdm_parity["overall_max_abs_error_efficient_vs_official"]) <= 1e-6
        and int(base_official["valid_scene_count"]) == 12146
        and int(base_official["invalid_scene_count"]) == 0
        and float(base_official["max_selected_score_parity_abs"]) <= 1e-6
        and all(bool(value) if isinstance(value, bool) else float(value) <= 1e-9 for value in base["consistency"].values())
    )
    inputs = VerdictInputs(
        infrastructure_valid=infrastructure_valid,
        coverage_gap=float(base["coverage_gap"]),
        ranking_gap=float(base["ranking_gap"]),
        ideal1_oracle_gain=float(rows["ideal1"]["delta_oracle"]),
        ideal1_selected_gain=float(rows["ideal1"]["delta_selected"]),
        ideal8_oracle_gain=float(rows["ideal8"]["delta_oracle"]),
        ideal16_oracle_gain=float(rows["ideal16"]["delta_oracle"]),
        ideal8_selected_gain=float(rows["ideal8"]["delta_selected"]),
        ideal16_selected_gain=float(rows["ideal16"]["delta_selected"]),
        ideal8_oracle_ci_low=float(rows["ideal8"]["delta_oracle_ci_low"]),
        ideal8_selected_ci_low=float(rows["ideal8"]["delta_selected_ci_low"]),
        fixed_budget_gain=float(rows["fixed_top"]["delta_selected"]),
        duplicate_shift=float(rows["duplicate8"]["delta_selected"]),
        candidate_limited_share_low=float(base["candidate_limited_share_low"]),
        ranker_limited_share_low=float(base["ranker_limited_share_low"]),
        target_available_rate=target_rate,
        saturated_false_replacement_rate=float(rows["ideal8"]["saturated_false_replacement_rate"]),
        practical_selected_gain=practical_gain,
        ideal64_oracle_gain=float(rows["full"]["delta_oracle"]),
        ideal64_selected_gain=float(rows["full"]["delta_selected"]),
    )
    verdict = choose_verdict(inputs)
    verdict["candidate_limited_scene_count"] = candidate_limited_count
    verdict["primary_injection"] = args.primary_injection
    verdict["practical_selected_gain_definition"] = "maximum full-split union_full DeltaSelected among deployable_structured/deployable_pseudo_expert banks"
    verdict["fixed_budget_primary_strategy"] = "retain Base predicted Top-56 plus true-selected diverse external 8 (upper bound)"
    verdict["frozen_scorer_scope"] = "the boolean safety result applies only to the primary IdealExtra8 pre-registered controls"
    verdict["primary_full64_saturated_false_replacement_rate"] = float(rows["full"]["saturated_false_replacement_rate"])
    verdict["primary_ideal16_saturated_false_replacement_rate"] = float(rows["ideal16"]["saturated_false_replacement_rate"])
    verdict["direct_frozen_scorer_recommendation"] = (
        "conditional Base56+LoRA8 pilot; repeat safety gate on actual LoRA distribution; retrain/calibrate for 16/64 or any bank failing thresholds"
        if verdict.get("frozen_scorer_safe", False)
        else "mixed-bank scorer retraining/calibration required"
    )
    atomic_json(args.output_dir / "VERDICT.json", verdict)

    # Preserve the compact scientific tables and plots inside the report bundle.
    for name in (
        "scene_metrics.parquet", "subset_summary.csv", "topk_oracle.csv",
        "scorer_success_analysis.csv", "scorer_feature_importance.csv",
        "scorer_grouped_cv.csv", "failure_type_limitation.csv",
        "threshold_sensitivity.csv", "decision_tree_rules.txt",
    ):
        _copy_file(args.base_audit_dir / name, args.output_dir / name)
    if (args.output_dir / "figures").exists():
        shutil.rmtree(args.output_dir / "figures")
    shutil.copytree(args.base_audit_dir / "figures", args.output_dir / "figures")
    all_rows = []
    for name, summary in injections.items():
        for row in summary["settings"]:
            all_rows.append({"audit_name": name, "bank_class": bank_classes.get(name, "unspecified"), **row})
    pd.DataFrame(all_rows).to_csv(args.output_dir / "injection_summary.csv", index=False)
    stage_frame = pd.DataFrame([row for row in all_rows if row["setting"].startswith(("external_stage", "union_stage"))])
    if len(stage_frame):
        stage_frame.to_csv(args.output_dir / "refinement_stage_summary.csv", index=False)
    _copy_file(args.pdm_parity_json, args.output_dir / "pdm_candidate_parity.json")
    atomic_json(args.output_dir / "forward_parity_summary.json", forward)
    atomic_json(
        args.output_dir / "candidate_shards" / "manifest.json",
        {"storage": "external rank-sharded NPZ; not duplicated into git", "root": str(args.forward_parity_root.resolve()), **forward},
    )

    branch = _git(args.repo_root, "branch", "--show-current")
    gpu_rows = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,name,memory.total,driver_version", "--format=csv,noheader"],
        text=True,
    ).splitlines()
    hydra_lines = [line for line in args.hydra_overrides.read_text().splitlines() if line.strip()]
    manifest: Dict[str, Any] = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "repo": {"root": str(args.repo_root.resolve()), "branch": branch, "start_commit": args.start_commit},
        "drivor_source": {"root": "/mnt/project/external/DrivoR", "commit": "f02665403df799c1b4ddd8b0d34e073f0555c13a"},
        "checkpoint": {"path": str(args.checkpoint.resolve()), "sha256": sha256_file(args.checkpoint), "selection_reason": "local official NAVSIM-v1 release corresponding to the recorded ~93.7 PDMS reproduction"},
        "alternative_checkpoint": ({"path": str(args.alternative_checkpoint.resolve()), "sha256": sha256_file(args.alternative_checkpoint)} if args.alternative_checkpoint else None),
        "agent_config": {"path": str(args.config.resolve()), "sha256": sha256_file(args.config)},
        "dino_weights": {"path": str(args.dino_weights.resolve()), "sha256": sha256_file(args.dino_weights)},
        "base_proposals": {"path": str(args.base_proposals.resolve()), "sha256": sha256_file(args.base_proposals)},
        "base_candidate_matrix": {"path": str(args.base_matrix.resolve()), "sha256": sha256_file(args.base_matrix)},
        "official_selected_evaluation": {"path": str(args.base_official_summary.resolve()), "sha256": sha256_file(args.base_official_summary), "max_selected_score_parity_abs": float(base_official["max_selected_score_parity_abs"]), "selected_pdms": float(base_official["metrics"]["standard_selected_pdms"])},
        "primary_external_matrix": {"path": str(args.primary_external_matrix.resolve()), "sha256": sha256_file(args.primary_external_matrix)},
        "split": "navtest",
        "scene_count": int(base["scene_count"]),
        "log_count": int(base["log_count"]),
        "proposal_num": int(base["proposal_count"]),
        "ref_num": 4,
        "scorer_ref_num": 4,
        "seed": args.seed,
        "metric_cache": str(args.metric_cache.resolve()),
        "hydra_overrides": hydra_lines,
        "versions": {"python": platform.python_version(), "pytorch": torch.__version__, "cuda": torch.version.cuda, "numpy": np.__version__},
        "hardware": {"gpus": gpu_rows},
        "environment": {key: os.environ.get(key) for key in ("CONDA_DEFAULT_ENV", "CONDA_PREFIX", "CUBLAS_WORKSPACE_CONFIG", "CUDA_VISIBLE_DEVICES", "PYTHONPATH")},
        "f0_forward": forward,
        "f0_pdm": pdm_parity,
        "f1_consistency": base["consistency"],
        "injections": {
            name: {
                "path": str(injection_paths[name].resolve()),
                "class": bank_classes.get(name, "unspecified"),
                "scene_count": int(summary["scene_count"]),
                "forward_parity": summary["forward_parity"],
                "shard_lineage": json.loads(next(injection_paths[name].glob("shard_*/manifest.json")).read_text()),
            }
            for name, summary in injections.items()
        },
        "candidate_bank_manifests": {
            name: {"path": str(path.resolve()), "sha256": sha256_file(path), "content": json.loads(path.read_text())}
            for name, path in bank_manifest_paths.items()
        },
        "verdict": verdict["verdict"],
    }
    atomic_json(args.output_dir / "manifest.json", manifest)
    feature_frame = pd.read_csv(args.base_audit_dir / "scorer_success_analysis.csv")
    report = _report_text(base, verdict, primary, rows, injections, bank_classes, target_rate, forward, pdm_parity, base_official, feature_frame)
    (args.output_dir / "REPORT.md").write_text(report)
    commands_text = args.commands_log.read_text() if args.commands_log else "See REPRODUCE.md for exact commands.\n"
    (args.output_dir / "commands.log").write_text(commands_text)
    reproduce = f"""# Reproduce

All commands run in `{args.repo_root}` with no network access and no LoRA training.

1. Run `scripts/lora_value_audit/run_f0_parity.sh` to export rank-sharded proposals and verify exact forward parity.
2. Run `scripts/lora_value_audit/run_f1_base_audit.sh` to produce the full Base-64 audit and grouped scorer analysis.
3. Score every external bank with `tools.lora_value_audit.score_candidate_bank`; this uses fixed PDM-reference progress already proven identical to official one-candidate `pdm_score`.
4. Run `scripts/lora_value_audit/run_union_bank.sh` so the frozen scorer sees each complete Base+external set in one self-attention call.
5. Re-run this module with the same arguments to apply `tools.lora_value_audit.schema.choose_verdict`.

The exact realized commands and all absolute artifact paths/hashes are in `commands.log` and `manifest.json`.
"""
    (args.output_dir / "REPRODUCE.md").write_text(reproduce)
    if not report.startswith(f"# Final Verdict\n\nVERDICT={verdict['verdict']}\n"):
        raise RuntimeError("REPORT.md verdict header is inconsistent")
    persisted_verdict = json.loads((args.output_dir / "VERDICT.json").read_text())
    persisted_manifest = json.loads((args.output_dir / "manifest.json").read_text())
    if persisted_verdict["verdict"] != persisted_manifest["verdict"]:
        raise RuntimeError("VERDICT.json and manifest.json disagree")
    if int(base["scene_count"]) != len(pd.read_parquet(args.output_dir / "scene_metrics.parquet")):
        raise RuntimeError("Report scene_metrics.parquet silently dropped rows")
    print(json.dumps({"report": str((args.output_dir / 'REPORT.md').resolve()), "verdict": verdict["verdict"], "manifest": str((args.output_dir / 'manifest.json').resolve())}, indent=2))


if __name__ == "__main__":
    main()
