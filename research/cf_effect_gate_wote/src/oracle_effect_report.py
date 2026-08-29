"""Automatic Gate2O v2 comparisons, verdict, and immutable report surface."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import numpy.typing as npt
import pandas as pd

from .effect_tokenizer import MODEL_VARIANTS
from .feature_store import FeatureShardReader, atomic_write_json, sha256_file, stable_json_hash
from .independent_label_store import SixFactorIndependentCandidateLabelStore
from .metrics import paired_scene_bootstrap
from .models.structured_six_factor_probe import (
    CHECKPOINT_SCHEMA,
    SIX_FACTOR_ORDER,
    StructuredProbeConfig,
    StructuredSixFactorProbe,
    trainable_parameter_count,
)
from .oracle_effect_data import validate_effect_cache, validate_label_free_feature_cache


BOOTSTRAP_SAMPLES = 5000
BOOTSTRAP_CONFIDENCE = 0.95
BOOTSTRAP_SEED = 20260827
EXPECTED_OLD_REPORT_HASHES = {
    "reports/cf_effect_gate_wote": "3903ddd28803fffe9ddeceeb89ad2968f4e6bcf0496cd82aac0776d7cc6842fa",
    "reports/cf_effect_wote_relabel": "7d79931636b35cd19a16c8ba3f3ae1dd10e3c78f5c5843ccc2caef9e40fdd255",
    "reports/cf_effect_wote_six_factor": "bd2d9d88881ea0e538ad18a867972d7e0c100dc7059737d5dd430e8664ee4298",
}

NEXT_EXPERIMENTS = {
    "ORACLE_PRIMITIVE_ACTION_EFFECT_VIABLE": "train lightweight candidate-conditioned primitive effect forward predictor",
    "INTERACTION_CONDITIONAL_EFFECT": "train interaction-gated selective effect predictor",
    "STATIC_GEOMETRY_ONLY": "stop counterfactual world modeling; study static geometry scorer/refiner",
    "SHARED_FUTURE_SUFFICIENT": "study shared factual future latent without candidate-specific rollout",
    "WOTE_LATENT_SIGNAL_ONLY": "analyze useful WoTE latent information and action sensitivity",
    "DIRECT_SCORER_SUFFICIENT": "stop the explicit effect bottleneck",
    "METRIC_PROXY_DEPENDENT": "stop the current effect schema",
    "INTERACTION_MASK_PROXY": "treat the mask as an engineered risk feature, not a world-model contribution",
    "DIRECT_BASELINE_UNDERFIT": "repair the scorer before evaluating the direction",
    "CANDIDATE_SPECIFICITY_UNPROVEN": "redesign candidate-specificity evidence before any predictor training",
    "ORACLE_EFFECT_PIPELINE_FAILED": "repair and rerun the oracle-effect data/implementation pipeline",
}


class OracleEffectReportError(RuntimeError):
    """The automatic report cannot establish the registered contract."""


def hard_false_safe(factors: npt.ArrayLike) -> npt.NDArray[np.bool_]:
    values = np.asarray(factors, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 6:
        raise ValueError("hard false-safe expects [N,6] six-factor labels")
    return (
        (values[:, 0] == 0)
        | (values[:, 1] == 0)
        | (values[:, 2] == 0)
        | (values[:, 4] == 0)
    )


def direction_non_compliance(factors: npt.ArrayLike) -> npt.NDArray[np.bool_]:
    values = np.asarray(factors, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 6:
        raise ValueError("direction diagnostics expect [N,6]")
    return values[:, 2] < 1.0


def logical_report_tree_sha256(repo_root: Path, relative: str) -> str:
    root = repo_root / relative
    lines = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        lines.append(f"{sha256_file(path)}  {path.relative_to(repo_root).as_posix()}\n")
    return hashlib.sha256("".join(lines).encode("utf-8")).hexdigest()


def audit_legacy_reports(repo_root: Path) -> Mapping[str, Any]:
    actual = {
        relative: logical_report_tree_sha256(repo_root, relative)
        for relative in EXPECTED_OLD_REPORT_HASHES
    }
    status = all(actual[key] == expected for key, expected in EXPECTED_OLD_REPORT_HASHES.items())
    return {
        "status": "UNCHANGED" if status else "CHANGED",
        "expected": EXPECTED_OLD_REPORT_HASHES,
        "actual": actual,
    }


@dataclass(frozen=True)
class Comparison:
    first: str
    second: str
    first_intervention: str
    second_intervention: str
    scenes: int
    score_delta: float
    score_ci_lower: float
    score_ci_upper: float
    first_regret: float
    second_regret: float
    regret_reduction: float
    seed_score_deltas: tuple[float, ...]


def paired_comparison(
    scene_results: pd.DataFrame,
    first: str,
    second: str,
    *,
    first_intervention: str = "none",
    second_intervention: str = "none",
) -> Comparison:
    values: dict[str, pd.DataFrame] = {}
    for label, model, intervention in (
        ("first", first, first_intervention),
        ("second", second, second_intervention),
    ):
        subset = scene_results[
            (scene_results["model_type"] == model)
            & (scene_results["intervention"] == intervention)
        ].copy()
        if subset.empty:
            raise OracleEffectReportError(f"missing comparison model {model}/{intervention}")
        if model == "wote_base_selector":
            averaged = subset[["scene_token", "selected_score", "regret"]].drop_duplicates(
                "scene_token"
            )
        else:
            averaged = subset.groupby("scene_token", as_index=False)[
                ["selected_score", "regret"]
            ].mean()
        values[label] = averaged
    merged = values["first"].merge(
        values["second"], on="scene_token", suffixes=("_first", "_second"), validate="one_to_one"
    )
    if len(merged) != 512:
        raise OracleEffectReportError(
            f"comparison {first}/{second} expected 512 scenes, got {len(merged)}"
        )
    interval = paired_scene_bootstrap(
        merged["selected_score_first"].to_numpy(dtype=np.float64),
        merged["selected_score_second"].to_numpy(dtype=np.float64),
        samples=BOOTSTRAP_SAMPLES,
        confidence=BOOTSTRAP_CONFIDENCE,
        seed=BOOTSTRAP_SEED,
    )
    first_regret = float(merged["regret_first"].mean())
    second_regret = float(merged["regret_second"].mean())
    regret_reduction = (
        (second_regret - first_regret) / second_regret
        if second_regret > 1.0e-12
        else 0.0
    )
    seed_deltas: list[float] = []
    if first != "wote_base_selector" and second != "wote_base_selector":
        first_seed = scene_results[
            (scene_results["model_type"] == first)
            & (scene_results["intervention"] == first_intervention)
        ]
        second_seed = scene_results[
            (scene_results["model_type"] == second)
            & (scene_results["intervention"] == second_intervention)
        ]
        for seed in (0, 1, 2):
            a = first_seed[first_seed["seed"] == seed][["scene_token", "selected_score"]]
            b = second_seed[second_seed["seed"] == seed][["scene_token", "selected_score"]]
            pair = a.merge(b, on="scene_token", suffixes=("_a", "_b"), validate="one_to_one")
            seed_deltas.append(float((pair["selected_score_a"] - pair["selected_score_b"]).mean()))
    return Comparison(
        first=first,
        second=second,
        first_intervention=first_intervention,
        second_intervention=second_intervention,
        scenes=len(merged),
        score_delta=interval.estimate,
        score_ci_lower=interval.lower,
        score_ci_upper=interval.upper,
        first_regret=first_regret,
        second_regret=second_regret,
        regret_reduction=float(regret_reduction),
        seed_score_deltas=tuple(seed_deltas),
    )


def _gate_statuses(
    scene: pd.DataFrame, interaction: pd.DataFrame
) -> tuple[Mapping[str, Any], Mapping[str, Comparison]]:
    comparisons = {
        "direct_vs_wote": paired_comparison(scene, "direct_current", "wote_base_selector"),
        "full_vs_direct": paired_comparison(
            scene, "full_primitive_action_effect", "direct_current"
        ),
        "full_vs_static": paired_comparison(
            scene, "full_primitive_action_effect", "static_primitive_effect"
        ),
        "full_vs_shared": paired_comparison(
            scene, "full_primitive_action_effect", "shared_logged_future"
        ),
        "full_vs_swap": paired_comparison(
            scene,
            "full_primitive_action_effect",
            "full_primitive_action_effect",
            second_intervention="full_effect_swap",
        ),
        "engineered_vs_direct": paired_comparison(
            scene, "full_engineered_action_effect", "direct_current"
        ),
        "engineered_vs_static": paired_comparison(
            scene, "full_engineered_action_effect", "static_primitive_effect"
        ),
        "mask_vs_direct": paired_comparison(scene, "interaction_mask_only", "direct_current"),
        "nomask_vs_direct": paired_comparison(
            scene, "full_primitive_no_interaction_mask", "direct_current"
        ),
        "wote_full_vs_direct": paired_comparison(scene, "wote_full_future", "direct_current"),
        "wote_env_vs_direct": paired_comparison(
            scene, "wote_environment_only", "direct_current"
        ),
    }
    direct = comparisons["direct_vs_wote"]
    full_direct = comparisons["full_vs_direct"]
    full_static = comparisons["full_vs_static"]
    full_shared = comparisons["full_vs_shared"]
    swap = comparisons["full_vs_swap"]
    direct_pass = direct.score_delta >= 0.05 or direct.regret_reduction >= 0.20
    full_direct_pass = (
        full_direct.score_delta >= 0.005
        and full_direct.regret_reduction >= 0.10
        and full_direct.score_ci_lower > 0
        and len(full_direct.seed_score_deltas) == 3
        and all(value > 0 for value in full_direct.seed_score_deltas)
    )
    full_static_pass = (
        full_static.score_ci_lower > 0
        and (full_static.score_delta >= 0.003 or full_static.regret_reduction >= 0.075)
        and sum(value > 0 for value in full_static.seed_score_deltas) >= 2
        and all(value >= -0.001 for value in full_static.seed_score_deltas)
    )
    full_shared_pass = full_shared.score_ci_lower > 0 and (
        full_shared.score_delta >= 0.003 or full_shared.regret_reduction >= 0.075
    )
    eliminated_fraction = (
        swap.score_delta / full_direct.score_delta if full_direct.score_delta > 1.0e-12 else 0.0
    )
    swap_regret_increase = (
        (swap.second_regret - swap.first_regret) / swap.first_regret
        if swap.first_regret > 1.0e-12
        else 0.0
    )
    specificity_pass = swap.score_ci_lower > 0 and (
        swap.score_delta >= 0.003
        or swap_regret_increase >= 0.10
        or eliminated_fraction >= 0.50
    )
    engineered = comparisons["engineered_vs_direct"]
    engineered_static = comparisons["engineered_vs_static"]
    engineered_pass = (
        engineered.score_ci_lower > 0
        and engineered.score_delta >= 0.005
        and engineered_static.score_ci_lower > 0
        and engineered_static.score_delta >= 0.003
    )
    primitive_pass = full_direct_pass and full_static_pass and full_shared_pass and specificity_pass
    gain_full = full_direct.score_delta
    gain_mask = comparisons["mask_vs_direct"].score_delta
    gain_nomask = comparisons["nomask_vs_direct"].score_delta
    mask_proxy = gain_full > 0 and gain_mask >= 0.8 * gain_full and gain_nomask <= 0.2 * gain_full

    rich = interaction[interaction["subset"] == "interaction_rich"]
    non = interaction[interaction["subset"] == "non_interaction"]
    interaction_conditional = False
    rich_count = 0
    if len(rich) == 1 and len(non) == 1:
        rich_row, non_row = rich.iloc[0], non.iloc[0]
        rich_count = int(rich_row["scenes"])
        interaction_conditional = (
            not full_static_pass
            and full_static.score_delta >= -0.001
            and rich_count >= 64
            and float(rich_row["full_vs_static_gain"]) >= 0.005
            and float(rich_row["full_vs_static_ci_lower"]) > 0
            and (
                float(non_row["full_vs_static_gain"]) <= 0.001
                or float(non_row["full_vs_static_ci_lower"]) <= 0
            )
        )
    statuses = {
        "direct_baseline_quality": direct_pass,
        "full_vs_direct": full_direct_pass,
        "full_vs_static": full_static_pass,
        "full_vs_shared_future": full_shared_pass,
        "candidate_specificity": specificity_pass,
        "primitive_requirement": primitive_pass,
        "engineered_pass": engineered_pass,
        "mask_proxy": mask_proxy,
        "interaction_conditional": interaction_conditional,
        "interaction_rich_scene_count": rich_count,
        "full_effect_swap_regret_increase": swap_regret_increase,
        "interaction_subset_status": (
            "PASS" if interaction_conditional else ("FAIL" if rich_count >= 64 else "NOT_APPLICABLE")
        ),
    }
    return statuses, comparisons


def automatic_verdict(
    *,
    data_contract_pass: bool,
    probe_contract_pass: bool,
    statuses: Mapping[str, Any],
    comparisons: Mapping[str, Comparison],
) -> Mapping[str, Any]:
    positive: list[str] = []
    blocking: list[str] = []
    if not data_contract_pass or not probe_contract_pass:
        verdict = "ORACLE_EFFECT_PIPELINE_FAILED"
        blocking.append("data or six-factor probe contract failed")
    elif not statuses["direct_baseline_quality"]:
        verdict = "DIRECT_BASELINE_UNDERFIT"
        blocking.append("direct current-BEV scorer did not clear its quality gate")
    elif statuses["engineered_pass"] and not statuses["primitive_requirement"]:
        verdict = "METRIC_PROXY_DEPENDENT"
        blocking.append("only the quarantined engineered representation cleared the effect gate")
    elif statuses["mask_proxy"]:
        verdict = "INTERACTION_MASK_PROXY"
        blocking.append("interaction mask explains the registered fraction of the full gain")
    elif not statuses["full_vs_direct"]:
        latent = any(
            comparisons[name].score_ci_lower > 0
            for name in ("wote_full_vs_direct", "wote_env_vs_direct")
        )
        verdict = "WOTE_LATENT_SIGNAL_ONLY" if latent else "DIRECT_SCORER_SUFFICIENT"
        blocking.append("full primitive did not outperform direct current scoring")
    elif not statuses["full_vs_static"]:
        if statuses["interaction_conditional"]:
            verdict = "INTERACTION_CONDITIONAL_EFFECT"
            positive.append("pre-registered interaction-rich subset cleared the conditional gate")
        else:
            verdict = "STATIC_GEOMETRY_ONLY"
            blocking.append("dynamic replay did not improve over ego plus static map primitives")
    elif not statuses["full_vs_shared_future"]:
        verdict = "SHARED_FUTURE_SUFFICIENT"
        blocking.append("candidate-specific transformation did not improve over shared logged future")
    elif not statuses["candidate_specificity"]:
        verdict = "CANDIDATE_SPECIFICITY_UNPROVEN"
        blocking.append("effect swapping did not significantly damage candidate ranking")
    elif statuses["interaction_conditional"]:
        verdict = "INTERACTION_CONDITIONAL_EFFECT"
        positive.append("effect value is isolated to the registered interaction-rich subset")
    else:
        verdict = "ORACLE_PRIMITIVE_ACTION_EFFECT_VIABLE"
        positive.extend(
            [
                "full primitive cleared full-vs-direct",
                "full primitive cleared full-vs-static",
                "full primitive cleared full-vs-shared-future",
                "full primitive cleared candidate-specificity intervention",
            ]
        )
    if verdict in {"ORACLE_EFFECT_PIPELINE_FAILED", "DIRECT_BASELINE_UNDERFIT"}:
        hypothesis = "UNTESTED"
    elif verdict == "ORACLE_PRIMITIVE_ACTION_EFFECT_VIABLE":
        hypothesis = "POSITIVE"
    elif verdict in {
        "INTERACTION_CONDITIONAL_EFFECT",
        "WOTE_LATENT_SIGNAL_ONLY",
        "STATIC_GEOMETRY_ONLY",
        "SHARED_FUTURE_SUFFICIENT",
        "CANDIDATE_SPECIFICITY_UNPROVEN",
    }:
        hypothesis = "PARTIAL"
    else:
        hypothesis = "NEGATIVE"
    return {
        "data_contract": "PASS" if data_contract_pass else "FAIL",
        "six_factor_probe_contract": "PASS" if probe_contract_pass else "FAIL",
        "direct_baseline_quality": "PASS" if statuses.get("direct_baseline_quality") else "FAIL",
        "full_vs_direct": "PASS" if statuses.get("full_vs_direct") else "FAIL",
        "full_vs_static": "PASS" if statuses.get("full_vs_static") else "FAIL",
        "full_vs_shared_future": "PASS" if statuses.get("full_vs_shared_future") else "FAIL",
        "candidate_specificity": "PASS" if statuses.get("candidate_specificity") else "FAIL",
        "primitive_requirement": "PASS" if statuses.get("primitive_requirement") else "FAIL",
        "interaction_subset_status": statuses.get("interaction_subset_status", "NOT_APPLICABLE"),
        "final_verdict": verdict,
        "scientific_hypothesis_status": hypothesis,
        "positive_evidence": positive,
        "blocking_evidence": blocking,
        "next_recommended_experiment": NEXT_EXPERIMENTS[verdict],
    }


def parameter_audit() -> pd.DataFrame:
    rows = []
    for model_type in MODEL_VARIANTS:
        model = StructuredSixFactorProbe()
        rows.append(
            {
                "model_type": model_type,
                "trainable_parameters": trainable_parameter_count(model),
                "architecture_class": type(model).__name__,
                "checkpoint_schema": CHECKPOINT_SCHEMA,
            }
        )
    frame = pd.DataFrame(rows)
    if frame["trainable_parameters"].nunique() != 1:
        raise OracleEffectReportError("A-L trainable parameter counts differ")
    return frame


def _data_contract(args: argparse.Namespace) -> Mapping[str, Any]:
    feature = {
        split: validate_label_free_feature_cache(getattr(args, f"{split}_cache"))
        for split in ("train", "val", "test")
    }
    effect = {
        split: validate_effect_cache(getattr(args, f"{split}_effects"))
        for split in ("train", "val", "test")
    }
    labels = {
        split: SixFactorIndependentCandidateLabelStore(getattr(args, f"{split}_labels"))
        for split in ("train", "val", "test")
    }
    expected_counts = {"train": 1024, "val": 256, "test": 512}
    counts_ok = all(
        feature[split]["scene_count"] == expected_counts[split]
        and effect[split]["scene_count"] == expected_counts[split]
        and int(labels[split].manifest["scene_count"]) == expected_counts[split]
        for split in expected_counts
    )
    label_hashes = {
        split: labels[split].logical_content_sha256 for split in expected_counts
    }
    feature_hashes = {
        split: feature[split]["logical_content_sha256"] for split in expected_counts
    }
    effect_hashes = {
        split: effect[split]["logical_content_sha256"] for split in expected_counts
    }
    return {
        "status": "PASS" if counts_ok else "FAIL",
        "candidate_count": 256,
        "trajectory_offsets": False,
        "oracle_candidate_forced": False,
        "label_schema": "independent_wote_labels_4s_six_factor.v2",
        "factor_order": list(SIX_FACTOR_ORDER),
        "score_source": "stored independent six-factor label store",
        "score_reconstruction_tolerance": 1.0e-6,
        "feature_cache_label_source": "none",
        "effect_reactive_response": False,
        "primitive_effect_schema": "primitive_effect.v1",
        "engineered_effect_schema": "engineered_effect.v1",
        "actor_selection": "first_valid_distance_then_track_token",
        "counts": expected_counts,
        "label_store_hashes": label_hashes,
        "feature_cache_hashes": feature_hashes,
        "effect_cache_hashes": effect_hashes,
        "six_factor_label_store_sha256": stable_json_hash(label_hashes),
        "feature_cache_sha256": stable_json_hash(feature_hashes),
        "effect_cache_sha256": stable_json_hash(effect_hashes),
    }


def _architecture_contract() -> Mapping[str, Any]:
    config = StructuredProbeConfig()
    params = parameter_audit()
    return {
        "status": "PASS",
        "checkpoint_schema": CHECKPOINT_SCHEMA,
        "factor_order": list(SIX_FACTOR_ORDER),
        "architecture": asdict(config),
        "trajectory_tokens": [8, 128],
        "current_bev_tokens": [64, 256],
        "projected_current_bev_tokens": [64, 128],
        "auxiliary_tokens": [32, 64],
        "parameter_difference_allowed": 0,
        "trainable_parameters": int(params["trainable_parameters"].iloc[0]),
        "all_model_counts_equal": True,
    }


def _copy_required(source: Path, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(f"refusing existing report artifact: {destination}")
    shutil.copy2(source, destination)


def _format(value: Any, digits: int = 6) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    return "N/A" if not np.isfinite(numeric) else f"{numeric:.{digits}f}"


def _metric_row(aggregate: pd.DataFrame, model: str) -> pd.Series:
    rows = aggregate[
        (aggregate["model_type"] == model) & (aggregate["intervention"] == "none")
    ]
    if len(rows) != 1:
        raise OracleEffectReportError(f"expected one aggregate row for {model}")
    return rows.iloc[0]


def _markdown_report(
    aggregate: pd.DataFrame,
    factor: pd.DataFrame,
    intervention: pd.DataFrame,
    interaction: pd.DataFrame,
    comparisons: Mapping[str, Comparison],
    verdict: Mapping[str, Any],
) -> str:
    lines = [
        "# Six-Factor Oracle Primitive Action-Effect Gate",
        "",
        "本报告只评价 oracle replay-grounded primitive action-effect representation 的候选排序价值；没有训练 forward/inverse/VLA/WoTE/trajectory 模型。",
        "",
        "## 表一：基础 scorer",
        "",
        "| Model | Selected score | Regret | Rank | Oracle capture | False-safe |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model, label in (
        ("wote_base_selector", "WoTE base selector"),
        ("trajectory_only", "Trajectory-only"),
        ("direct_current", "Direct current"),
        ("ego_kinematic_effect", "Ego kinematic"),
    ):
        row = _metric_row(aggregate, model)
        lines.append(
            f"| {label} | {_format(row.selected_score)} | {_format(row.top1_regret)} | "
            f"{_format(row.mean_selected_candidate_rank, 2)} | {_format(row.oracle_capture_mean)} | "
            f"{_format(row.hard_false_safe_rate)} |"
        )
    lines.extend(
        [
            "",
            "## 表二：Action-effect 分解",
            "",
            "| Model | Selected score | Delta vs Direct | Delta vs Static | Regret | Pairwise acc. |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    direct_score = float(_metric_row(aggregate, "direct_current").selected_score)
    static_score = float(_metric_row(aggregate, "static_primitive_effect").selected_score)
    for model, label in (
        ("static_primitive_effect", "Static primitive"),
        ("shared_logged_future", "Shared logged future"),
        ("dynamic_replay_effect", "Dynamic replay primitive"),
        ("full_primitive_action_effect", "Full primitive"),
        ("full_engineered_action_effect", "Full engineered"),
    ):
        row = _metric_row(aggregate, model)
        lines.append(
            f"| {label} | {_format(row.selected_score)} | {_format(float(row.selected_score)-direct_score)} | "
            f"{_format(float(row.selected_score)-static_score)} | {_format(row.top1_regret)} | "
            f"{_format(row.pairwise_ranking_accuracy)} |"
        )
    lines.extend(
        [
            "",
            "## 表三：现有 latent world model",
            "",
            "| Model | Selected score | Delta vs Direct | Regret | Candidate-specific |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for model, label in (
        ("wote_full_future", "WoTE full future"),
        ("wote_environment_only", "WoTE environment-only"),
    ):
        row = _metric_row(aggregate, model)
        lines.append(
            f"| {label} | {_format(row.selected_score)} | {_format(float(row.selected_score)-direct_score)} | "
            f"{_format(row.top1_regret)} | Yes |"
        )
    lines.extend(
        [
            "",
            "## 表四：干预实验",
            "",
            "| Intervention | Selected score | Drop vs Full | Regret increase |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    labels = {
        "none": "Full primitive",
        "full_effect_swap": "Full effect swap",
        "actor_only_swap": "Actor-only swap",
        "static_only_swap": "Static-only swap",
        "scene_mean_effect": "Scene-mean effect",
        "no_interaction_mask": "No interaction mask",
        "interaction_mask_only": "Interaction mask only",
    }
    for _, row in intervention.iterrows():
        lines.append(
            f"| {labels.get(row.intervention, row.intervention)} | {_format(row.selected_score)} | "
            f"{_format(row.drop_vs_full)} | {_format(row.regret_increase)} |"
        )
    lines.extend(
        [
            "",
            "## 表五：六因子 MAE",
            "",
            "| Model | NC | DAC | DDC | EP | TTC | Comfort |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for model, label in (
        ("direct_current", "Direct current"),
        ("static_primitive_effect", "Static primitive"),
        ("full_primitive_action_effect", "Full primitive"),
    ):
        subset = factor[factor["model_type"] == model].groupby("factor")["mae"].mean()
        lines.append(
            f"| {label} | " + " | ".join(_format(subset[name]) for name in SIX_FACTOR_ORDER) + " |"
        )
    lines.extend(
        [
            "",
            "## 核心比较",
            "",
            "| Comparison | Score delta | Regret reduction | 95% CI |",
            "| --- | ---: | ---: | --- |",
        ]
    )
    for key, label in (
        ("full_vs_direct", "Full vs Direct"),
        ("full_vs_static", "Full vs Static"),
        ("full_vs_shared", "Full vs Shared"),
        ("full_vs_swap", "Full vs Swap"),
    ):
        value = comparisons[key]
        lines.append(
            f"| {label} | {_format(value.score_delta)} | {_format(value.regret_reduction)} | "
            f"[{_format(value.score_ci_lower)}, {_format(value.score_ci_upper)}] |"
        )
    lines.extend(
        [
            "",
            "## Interaction subset",
            "",
            "| Subset | Scenes | Full | Static | Delta | CI |",
            "| --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for _, row in interaction.iterrows():
        lines.append(
            f"| {row['subset']} | {int(row.get('scenes', 0))} | {_format(row.get('full_score'))} | "
            f"{_format(row.get('static_score'))} | {_format(row.get('full_vs_static_gain'))} | "
            f"[{_format(row.get('full_vs_static_ci_lower'))}, {_format(row.get('full_vs_static_ci_upper'))}] |"
        )
    lines.extend(
        [
            "",
            "## 自动判定",
            "",
            f"- `final_verdict`: `{verdict['final_verdict']}`",
            f"- `scientific_hypothesis_status`: `{verdict['scientific_hypothesis_status']}`",
            f"- 下一实验（本任务未运行）：{verdict['next_recommended_experiment']}",
            "",
            "即使 verdict 为 `ORACLE_PRIMITIVE_ACTION_EFFECT_VIABLE`，其含义也仅限于：在单专家日志合法构造的 replay-grounded primitive effect 上，候选特定 action effect 含有超越相应控制组的规划信息，值得进入轻量 forward prediction 阶段；这不表示完整世界模型已经成立。",
            "",
            "## 明确 NOT_RUN",
            "",
            "`forward_effect`、`effect_predictor`、`inverse`、`trajectory_refinement`、`policy_distillation` 均未运行，因为任务边界要求停在 Oracle Effect Gate。",
        ]
    )
    return "\n".join(lines) + "\n"


def build_reports(args: argparse.Namespace) -> Mapping[str, Any]:
    if args.output.exists():
        raise FileExistsError(f"refusing existing report directory: {args.output}")
    args.output.mkdir(parents=True, exist_ok=False)
    data_contract = _data_contract(args)
    architecture = _architecture_contract()
    legacy = audit_legacy_reports(args.repo_root)
    if legacy["status"] != "UNCHANGED":
        data_contract = dict(data_contract)
        data_contract["status"] = "FAIL"
        data_contract["legacy_report_hash_status"] = "CHANGED"
    else:
        data_contract = dict(data_contract)
        data_contract["legacy_report_hash_status"] = "UNCHANGED"
    atomic_write_json(args.output / "DATA_CONTRACT.json", data_contract)
    atomic_write_json(args.output / "PROBE_ARCHITECTURE.json", architecture)

    asset_manifest = json.loads(args.asset_manifest.read_text(encoding="utf-8"))
    asset_manifest["legacy_report_hash_audit"] = legacy
    atomic_write_json(args.output / "ASSET_MANIFEST.json", asset_manifest)
    for source, name in (
        (args.hyperparameters, "GLOBAL_HYPERPARAMETER_SELECTION.json"),
        (args.split_manifest, "split_manifest.json"),
        (args.relabel_audit, "relabel_determinism_audit.json"),
        (args.effect_audit, "effect_cache_determinism_audit.json"),
    ):
        _copy_required(source, args.output / name)
    parameter_audit().to_csv(args.output / "parameter_audit.csv", index=False)

    required_eval = (
        "probe_metrics_per_seed.csv",
        "probe_metrics_aggregate.csv",
        "scene_level_results.parquet",
        "factor_metrics.csv",
        "intervention_ablation.csv",
        "interaction_subset.csv",
        "ddc_diagnostics.csv",
        "oracle_capture.csv",
        "failure_cases.csv",
    )
    for name in required_eval:
        _copy_required(args.evaluation / name, args.output / name)
    scene = pd.read_parquet(args.output / "scene_level_results.parquet")
    aggregate = pd.read_csv(args.output / "probe_metrics_aggregate.csv")
    factor = pd.read_csv(args.output / "factor_metrics.csv")
    intervention = pd.read_csv(args.output / "intervention_ablation.csv")
    interaction = pd.read_csv(args.output / "interaction_subset.csv")
    statuses, comparisons = _gate_statuses(scene, interaction)
    verdict = automatic_verdict(
        data_contract_pass=data_contract["status"] == "PASS",
        probe_contract_pass=architecture["status"] == "PASS",
        statuses=statuses,
        comparisons=comparisons,
    )
    verdict["comparisons"] = {key: asdict(value) for key, value in comparisons.items()}
    verdict["registered_gate_statuses"] = dict(statuses)
    atomic_write_json(args.output / "VERDICT.json", verdict)
    (args.output / "ORACLE_EFFECT_REPORT.md").write_text(
        _markdown_report(
            aggregate, factor, intervention, interaction, comparisons, verdict
        ),
        encoding="utf-8",
    )
    reproduction = """# Reproduction

Run `research/cf_effect_gate_wote/scripts/run_gate2o_all.sh` from the isolated
Gate2O worktree with explicit `--data-root`, `--map-root`, `--wote-root`, and
`--release-root` arguments.  Paths follow CLI > one-shot environment > shared
repository defaults; no personal mount is committed.

The launcher enforces: preflight, fixed split, deterministic 16-scene relabel,
full six-factor relabel, label-free frozen features, deterministic primitive
effects, overfit smoke, shared pilot, locked A--L training, full-256 evaluation,
interventions, automatic verdict, and stop.

No test label is used for hyperparameter selection.  EP is always evaluated on
the complete 256-candidate set.  The published `formatted_pdm_score_256.npy`
is not loaded by this pipeline.

NOT_RUN by design: forward effect prediction, inverse dynamics, VLA/WoTE
training, trajectory refinement, and policy distillation.
"""
    (args.output / "REPRODUCTION.md").write_text(reproduction, encoding="utf-8")
    return verdict


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--hyperparameters", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--relabel-audit", type=Path, required=True)
    parser.add_argument("--effect-audit", type=Path, required=True)
    parser.add_argument("--asset-manifest", type=Path, required=True)
    for split in ("train", "val", "test"):
        parser.add_argument(f"--{split}-cache", type=Path, required=True)
        parser.add_argument(f"--{split}-effects", type=Path, required=True)
        parser.add_argument(f"--{split}-labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    verdict = build_reports(args)
    print(json.dumps(verdict, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
