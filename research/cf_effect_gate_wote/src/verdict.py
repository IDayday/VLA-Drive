"""Build the evidence-only Gate report and decision-matrix verdict."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .feature_store import FeatureShardReader, atomic_write_json


MODEL_DISPLAY = {
    "trajectory_only": "Trajectory-only",
    "direct_current": "Direct current-state",
    "shared_logged_future": "Shared logged future",
    "oracle_replay_effect": "Oracle replay effect",
    "predicted_replay_effect": "Predicted replay effect",
    "wote_full_future": "WoTE full future",
    "wote_environment_only": "WoTE environment-only future",
    "effect_swap": "Effect swap",
    "predicted_replay_effect_swap": "Predicted effect swap",
}


def _json(path: Path) -> Mapping[str, Any] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _status(payload: Mapping[str, Any] | None, key: str) -> str:
    if payload is None:
        return "NOT_RUN"
    return "PASS" if bool(payload.get(key, False)) else "FAIL"


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite report CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _copy_if_absent(source: Path, target: Path) -> None:
    if target.exists():
        return
    if source.is_file():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)


def _simulator_supervision_dependent(metrics_path: Path) -> tuple[bool, dict[str, float]]:
    if not metrics_path.is_file():
        return False, {}
    metrics = pd.read_csv(metrics_path)
    averaged = metrics.groupby("model").mean(numeric_only=True)
    if not {"direct_current", "wote_full_future"}.issubset(averaged.index):
        return False, {}
    direct = averaged.loc["direct_current"]
    full = averaged.loc["wote_full_future"]
    direct_regret = float(direct["top1_regret"])
    regret_reduction = (
        float((direct_regret - full["top1_regret"]) / direct_regret)
        if direct_regret > 0
        else -float("inf")
    )
    pdms_gain = float(full["selected_pdms"] - direct["selected_pdms"])
    evidence = {
        "wote_full_vs_direct_regret_reduction_fraction": regret_reduction,
        "wote_full_vs_direct_selected_pdms_gain_raw": pdms_gain,
    }
    return regret_reduction >= 0.20 or pdms_gain >= 0.005, evidence


def determine_verdict(
    g0: Mapping[str, Any] | None,
    g1: Mapping[str, Any] | None,
    g2: Mapping[str, Any] | None,
    g3: Mapping[str, Any] | None,
    g4: Mapping[str, Any] | None,
    g2_metrics: Path,
) -> dict[str, Any]:
    statuses = {
        "gate_g0_reproduction": _status(g0, "gate_g0_pass"),
        "gate_g1_candidate_headroom": _status(g1, "gate_g1_pass"),
        "gate_g2_replay_effect_value": _status(g2, "gate_g2_pass"),
        "gate_g3_effect_prediction": _status(g3, "gate_g3_pass"),
        "gate_g4_inverse_planning": _status(g4, "gate_g4_pass"),
    }
    simulator_dependent, simulator_evidence = _simulator_supervision_dependent(
        g2_metrics
    )
    blocking: list[str] = []
    positive: list[str] = []
    recommended: list[str] = []

    if statuses["gate_g0_reproduction"] == "NOT_RUN":
        final = "NOT_RUN"
        blocking.append("G0 has not run; no direction verdict is available.")
    elif statuses["gate_g0_reproduction"] == "FAIL":
        final = "STOP_DIRECTION"
        mismatch = float(g0.get("alignment_mismatched_candidate_fraction", float("nan")))
        maximum = float(g0.get("alignment_maximum_absolute_error", float("nan")))
        tolerance = float(g0.get("alignment_tolerance", float("nan")))
        blocking.append(
            "G0 candidate-label alignment failed: "
            f"mismatched candidate fraction={mismatch:.6f}, "
            f"maximum absolute error={maximum:.9f}, tolerance={tolerance:.1e}."
        )
        if bool(g0.get("alignment_published_generator_default_cache_conflict", False)):
            blocking.append(
                "The published score generator requests 80 proposal poses, while the "
                "official default metric cache exposes 40 proposal poses and 50 future poses; "
                "the released score-generation horizon is not reproducible from the default cache."
            )
        if bool(g0.get("official_debug_equivalence", False)):
            positive.append(
                f"Patched/default WoTE outputs remained equivalent across {int(g0.get('scene_count', 0))} smoke scenes."
            )
        if bool(g0.get("cache_reproducible", False)):
            positive.append(
                "Two frozen-feature cache runs have the same logical content SHA256."
            )
    elif statuses["gate_g1_candidate_headroom"] == "NOT_RUN":
        final = "NOT_RUN"
        blocking.append("G1 has not run; no direction verdict is available.")
    elif statuses["gate_g1_candidate_headroom"] == "FAIL":
        final = "STOP_DIRECTION"
        blocking.append("Candidate bank has insufficient oracle headroom.")
    elif statuses["gate_g2_replay_effect_value"] == "NOT_RUN":
        final = "NOT_RUN"
        blocking.append("G2 has not run; replay-effect value is unknown.")
    elif statuses["gate_g2_replay_effect_value"] == "FAIL":
        if simulator_dependent:
            final = "SIMULATOR_SUPERVISION_DEPENDENT"
            blocking.append(
                "WoTE full-future features help, but legal single-log replay effects do not."
            )
        else:
            final = "STOP_DIRECTION"
            blocking.append(
                "Replay-grounded action effects do not outperform direct trajectory scoring."
            )
    elif statuses["gate_g3_effect_prediction"] == "NOT_RUN":
        final = "NOT_RUN"
        positive.append("Candidate-specific oracle replay effects pass G2.")
        blocking.append("G3 has not run; prediction viability is unknown.")
    elif statuses["gate_g3_effect_prediction"] == "FAIL":
        final = "EFFECT_TARGET_VALID_BUT_PREDICTION_BOTTLENECK"
        positive.append("Candidate-specific oracle replay effects pass G2.")
        blocking.append("The lightweight current-only forward predictor does not recover enough gain.")
    elif statuses["gate_g4_inverse_planning"] == "NOT_RUN":
        final = "NOT_RUN"
        positive.extend(
            [
                "Candidate-specific replay effects pass G2.",
                "Predicted replay effects pass G3 without future-rollout input.",
            ]
        )
        blocking.append("G4 has not run; inverse remains unevaluated.")
    elif statuses["gate_g4_inverse_planning"] == "FAIL":
        final = "EFFECT_MODEL_ONLY"
        positive.extend(
            [
                "Candidate-specific replay effects pass G2.",
                "Predicted replay effects pass G3 without future-rollout input.",
            ]
        )
        blocking.append("Environment-only inverse consistency is diagnostic only.")
        recommended.extend(
            [
                "candidate-conditioned effect latent from current observation and trajectory",
                "no inverse term in the planning objective",
            ]
        )
    else:
        final = "EFFECT_PLUS_INVERSE"
        positive.extend(
            [
                "Candidate-specific replay effects pass G2.",
                "Predicted replay effects pass G3 without future-rollout input.",
                "Frozen environment-only inverse consistency improves planning on G4.",
            ]
        )
        recommended.extend(
            [
                "candidate-conditioned effect latent from current observation and trajectory",
                "frozen environment-only inverse consistency",
                "validation-frozen inverse fusion or rejection rule",
            ]
        )
    if g1 is not None and bool(g1.get("gate_g1_pass")):
        positive.append(
            f"Best-of-256 oracle gap is {float(g1['oracle_gap_raw']):.6f} raw "
            f"({float(g1['oracle_gap_points']):.3f} points)."
        )
    if g2 is not None and bool(g2.get("gate_g2_pass")):
        positive.append(
            f"Oracle effect vs direct PDMS gain is "
            f"{float(g2['mean_direct_selected_pdms_gain_raw']):.6f} raw."
        )
    if simulator_evidence:
        positive.append(
            "WoTE full-future control evidence: "
            + json.dumps(simulator_evidence, sort_keys=True)
        )
    return {
        **statuses,
        "final_verdict": final,
        "recommended_vla_baseline_requirements": recommended,
        "blocking_evidence": blocking,
        "positive_evidence": positive,
    }


def _candidate_table(g1: Mapping[str, Any] | None) -> str:
    if g1 is None:
        return "| WoTE fixed base anchors (K=256) | NOT_RUN | — | — | — |"
    return (
        f"| WoTE fixed base anchors (K=256) | {g1['selected_score_raw']:.6f} "
        f"| {g1['oracle_score_raw']:.6f} | {g1['oracle_gap_raw']:.6f} "
        f"({g1['oracle_gap_points']:.3f} pt) | "
        f"{g1['failed_scene_recovery_fraction']:.3%} |"
    )


def _effect_table(metrics: pd.DataFrame) -> list[str]:
    order = [
        "trajectory_only",
        "direct_current",
        "shared_logged_future",
        "oracle_replay_effect",
        "predicted_replay_effect",
        "wote_full_future",
        "wote_environment_only",
        "effect_swap",
    ]
    if metrics.empty:
        return [
            f"| {MODEL_DISPLAY[model]} | NOT_RUN | — | — | — | — | — |"
            for model in order
        ]
    averaged = metrics.groupby("model", sort=False).mean(numeric_only=True)
    lines: list[str] = []
    for model in order:
        if model not in averaged.index:
            lines.append(f"| {MODEL_DISPLAY[model]} | NOT_RUN | — | — | — | — | — |")
            continue
        row = averaged.loc[model]
        source = metrics[metrics["model"] == model].iloc[-1]
        lines.append(
            f"| {MODEL_DISPLAY[model]} | {source['future_input']} | "
            f"{'yes' if bool(source['candidate_specific']) else 'no'} | "
            f"{row['selected_pdms']:.6f} | {row['top1_regret']:.6f} | "
            f"{row['pairwise_accuracy']:.3f} | {row['false_safe_rate']:.3%} |"
        )
    return lines


def _inverse_table(metrics: pd.DataFrame) -> list[str]:
    if metrics.empty:
        return [
            f"| {mode} | NOT_RUN | — | — | — | — |"
            for mode in ("ego_only", "environment_only", "full_effect")
        ]
    averaged = metrics.groupby("effect_input").mean(numeric_only=True)
    lines: list[str] = []
    for mode in ("ego_only", "environment_only", "full_effect"):
        if mode not in averaged.index:
            lines.append(f"| {mode} | NOT_RUN | — | — | — | — |")
            continue
        row = averaged.loc[mode]
        pdms = "—" if not np.isfinite(row.get("pdms_with_gate", np.nan)) else f"{row['pdms_with_gate']:.6f}"
        false_safe = "—" if not np.isfinite(row.get("false_safe", np.nan)) else f"{row['false_safe']:.3%}"
        lines.append(
            f"| {mode} | {row['top1_retrieval']:.3%} | {row['mrr']:.3f} | "
            f"{row['delta_sign_accuracy']:.3%} | {pdms} | {false_safe} |"
        )
    return lines


def _gate_evidence(
    statuses: Mapping[str, str],
    g0: Mapping[str, Any] | None,
    g1: Mapping[str, Any] | None,
    g2: Mapping[str, Any] | None,
    g3: Mapping[str, Any] | None,
    g4: Mapping[str, Any] | None,
) -> list[str]:
    rows = []
    if g0:
        if bool(g0.get("candidate_alignment_pass", False)):
            g0_evidence = (
                f"{int(g0.get('scene_count', 0))} scenes; equivalence/cache/alignment passed"
            )
        else:
            g0_evidence = (
                f"alignment mismatch={float(g0.get('alignment_mismatched_candidate_fraction', float('nan'))):.3%}; "
                f"max error={float(g0.get('alignment_maximum_absolute_error', float('nan'))):.6f}; "
                f"tolerance={float(g0.get('alignment_tolerance', float('nan'))):.1e}"
            )
    else:
        g0_evidence = "artifact absent"
    evidence = {
        "G0": (
            statuses["gate_g0_reproduction"],
            g0_evidence,
        ),
        "G1": (
            statuses["gate_g1_candidate_headroom"],
            f"oracle gap={g1['oracle_gap_raw']:.6f}" if g1 else "not run",
        ),
        "G2": (
            statuses["gate_g2_replay_effect_value"],
            f"effect-direct gain={g2['mean_direct_selected_pdms_gain_raw']:.6f}"
            if g2
            else "not run",
        ),
        "G3": (
            statuses["gate_g3_effect_prediction"],
            f"gain recovery={max(g3['oracle_regret_gain_recovered_fraction'], g3['oracle_pdms_gain_recovered_fraction']):.3f}"
            if g3
            else "not run",
        ),
        "G4": (
            statuses["gate_g4_inverse_planning"],
            f"environment inverse PDMS gain={g4['environment_pdms_gain_raw']:.6f}"
            if g4
            else "not run",
        ),
    }
    for gate, (status, primary) in evidence.items():
        decision = "continue" if status == "PASS" else "stop dependent Gates" if status == "FAIL" else "dependent Gates NOT_RUN"
        rows.append(f"| {gate} | {status} | {primary} | {decision} |")
    return rows


def _g0_detail_table(g0: Mapping[str, Any] | None) -> list[str]:
    if g0 is None:
        return ["| Smoke export | NOT_RUN | — |", "| Candidate-label alignment | NOT_RUN | — |"]
    cache_hash = str(g0.get("cache_first_logical_sha256", "—"))
    mismatch = float(g0.get("alignment_mismatched_candidate_fraction", float("nan")))
    maximum = float(g0.get("alignment_maximum_absolute_error", float("nan")))
    mean = float(g0.get("alignment_mean_absolute_error", float("nan")))
    tolerance = float(g0.get("alignment_tolerance", float("nan")))
    generator_horizon = g0.get("alignment_published_score_generator_proposal_num_poses", "—")
    cache_proposal = g0.get("alignment_default_metric_cache_proposal_num_poses", "—")
    cache_future = g0.get("alignment_default_metric_cache_future_num_poses", "—")
    return [
        f"| Smoke feature export | {'PASS' if int(g0.get('scene_count', 0)) == 200 else 'FAIL'} | "
        f"{int(g0.get('scene_count', 0))} scenes; {int(g0.get('scene_failures', 0))} failures |",
        f"| Debug-output equivalence | {'PASS' if bool(g0.get('official_debug_equivalence', False)) else 'FAIL'} | "
        "trajectory, all_trajectory, and final_rewards checked at 1e-6 |",
        f"| Cache determinism | {'PASS' if bool(g0.get('cache_reproducible', False)) else 'FAIL'} | "
        f"logical SHA256 `{cache_hash}` |",
        f"| Candidate-label alignment | {'PASS' if bool(g0.get('candidate_alignment_pass', False)) else 'FAIL'} | "
        f"mismatch {mismatch:.3%}; max/mean error {maximum:.9f}/{mean:.9f}; tolerance {tolerance:.1e} |",
        f"| Published horizon consistency | {'FAIL' if bool(g0.get('alignment_published_generator_default_cache_conflict', False)) else 'PASS'} | "
        f"score generator={generator_horizon}; default cache proposal/future={cache_proposal}/{cache_future} poses |",
    ]


def _interaction_sensitivity(effect_cache: Path) -> pd.DataFrame:
    if not (effect_cache / "manifest.json").is_file():
        return pd.DataFrame(columns=["ablation", "value", "unit"])
    sums: dict[str, int] = {}
    counts: dict[str, int] = {}
    for _, arrays in FeatureShardReader(effect_cache).iter_shards():
        for key, value in arrays.items():
            if key.startswith("interaction_mask_clearance_"):
                sums[key] = sums.get(key, 0) + int(np.asarray(value, dtype=bool).sum())
                counts[key] = counts.get(key, 0) + int(np.asarray(value).size)
    return pd.DataFrame(
        [
            {"ablation": key, "value": sums[key] / counts[key], "unit": "masked_fraction"}
            for key in sorted(sums)
        ]
    )


def build_report(args: argparse.Namespace) -> None:
    report = args.report_dir
    report.mkdir(parents=True, exist_ok=True)
    gate01 = args.experiment_root / args.gate01_run_id
    gate2 = args.experiment_root / args.gate2_run_id
    gate3 = args.experiment_root / args.gate3_run_id
    gate4 = args.experiment_root / args.gate4_run_id
    g0 = _json(gate01 / "g0_smoke_summary.json")
    g1 = _json(gate01 / "g1_candidate_oracle_summary.json")
    g2 = _json(gate2 / "evaluation/g2_summary.json")
    g3 = _json(gate3 / "evaluation/g3_summary.json")
    g4 = _json(gate4 / "evaluation/g4_summary.json")

    _copy_if_absent(gate01 / "g0_candidate_alignment.csv", report / "candidate_alignment.csv")
    _copy_if_absent(gate01 / "candidate_oracle_summary.csv", report / "candidate_oracle_summary.csv")
    if not (report / "candidate_alignment.csv").exists():
        _write_csv(
            report / "candidate_alignment.csv",
            pd.DataFrame(
                columns=[
                    "scene_token",
                    "candidate_index",
                    "factor",
                    "precomputed",
                    "recomputed",
                    "absolute_error",
                    "trajectory_hash",
                ]
            ),
        )
    if not (report / "candidate_oracle_summary.csv").exists():
        _write_csv(
            report / "candidate_oracle_summary.csv",
            pd.DataFrame(columns=["scene_token", "selected_score_raw", "oracle_score_raw", "regret_raw"]),
        )

    metric_paths = [
        gate2 / "evaluation/probe_metrics.csv",
        gate3 / "evaluation/probe_metrics_g3.csv",
    ]
    metric_frames = [pd.read_csv(path) for path in metric_paths if path.is_file()]
    probe_metrics = (
        pd.concat(metric_frames, ignore_index=True).drop_duplicates(
            subset=["gate", "model", "seed"], keep="last"
        )
        if metric_frames
        else pd.DataFrame(
            columns=[
                "gate",
                "model",
                "seed",
                "future_input",
                "candidate_specific",
                "selected_pdms",
                "top1_regret",
                "pairwise_accuracy",
                "false_safe_rate",
            ]
        )
    )
    _write_csv(report / "probe_metrics.csv", probe_metrics)
    _copy_if_absent(
        gate3 / "evaluation/effect_prediction_metrics.csv",
        report / "effect_prediction_metrics.csv",
    )
    if not (report / "effect_prediction_metrics.csv").exists():
        _write_csv(
            report / "effect_prediction_metrics.csv",
            pd.DataFrame(columns=["seed", "split", "scene_count", "ego_effect_mae", "map_effect_mae"]),
        )
    _copy_if_absent(gate4 / "evaluation/inverse_metrics.csv", report / "inverse_metrics.csv")
    if not (report / "inverse_metrics.csv").exists():
        _write_csv(
            report / "inverse_metrics.csv",
            pd.DataFrame(columns=["effect_input", "seed", "top1_retrieval", "mrr", "delta_sign_accuracy", "pdms_with_gate", "false_safe"]),
        )
    inverse_metrics = pd.read_csv(report / "inverse_metrics.csv")

    scene_paths = [
        gate2 / "evaluation/scene_level_g2.parquet",
        gate3 / "evaluation/scene_level_g3.parquet",
        gate4 / "evaluation/scene_level_g4.parquet",
    ]
    scene_frames = [pd.read_parquet(path) for path in scene_paths if path.is_file()]
    scene_results = (
        pd.concat(scene_frames, ignore_index=True, sort=False)
        if scene_frames
        else pd.DataFrame(columns=["scene_token", "gate", "model", "seed"])
    )
    scene_path = report / "scene_level_results.parquet"
    if scene_path.exists():
        raise FileExistsError(f"refusing existing scene report: {scene_path}")
    scene_results.to_parquet(scene_path, index=False)

    if not scene_results.empty and {"model", "false_safe"}.issubset(scene_results.columns):
        failures = scene_results[
            scene_results["false_safe"].fillna(False).astype(bool)
            | (scene_results.get("regret", pd.Series(0, index=scene_results.index)).fillna(0) > 0.05)
        ].copy()
    else:
        alignment_frame = pd.read_csv(report / "candidate_alignment.csv")
        tolerance = float(g0.get("alignment_tolerance", 1e-6)) if g0 else 1e-6
        failures = alignment_frame[alignment_frame.get("absolute_error", 0) > tolerance].copy()
        if not failures.empty:
            failures.insert(1, "gate", "G0")
            failures.insert(2, "model", "candidate_alignment")
            failures.insert(3, "seed", np.nan)
            failures["reason"] = "precomputed/recomputed candidate factor mismatch"
        else:
            failures = pd.DataFrame(columns=["scene_token", "gate", "model", "seed", "reason"])
    _write_csv(report / "failure_cases.csv", failures)

    ablations = _interaction_sensitivity(gate2 / "effects-test")
    if not probe_metrics.empty:
        selected_ablations = probe_metrics[
            probe_metrics["model"].isin(
                ["shared_logged_future", "effect_swap", "wote_full_future", "wote_environment_only"]
            )
        ][["gate", "model", "seed", "selected_pdms", "top1_regret"]].copy()
        selected_ablations["ablation"] = selected_ablations["model"]
        ablations = pd.concat([ablations, selected_ablations], ignore_index=True, sort=False)
    _write_csv(report / "ablation_summary.csv", ablations)

    verdict = determine_verdict(
        g0, g1, g2, g3, g4, gate2 / "evaluation/probe_metrics.csv"
    )
    atomic_write_json(report / "VERDICT.json", verdict)
    statuses = {
        key: str(verdict[key])
        for key in (
            "gate_g0_reproduction",
            "gate_g1_candidate_headroom",
            "gate_g2_replay_effect_value",
            "gate_g3_effect_prediction",
            "gate_g4_inverse_planning",
        )
    }
    gate_report = "\n".join(
        [
            "# Frozen-WoTE Counterfactual Effect Gate",
            "",
            "This is a direction Gate, not a named algorithm. Replay effects hold logged actor futures fixed and are not true counterfactual futures.",
            "",
            f"Final verdict: `{verdict['final_verdict']}`.",
            "",
            "## G0 reproduction detail",
            "",
            "| Check | Status | Evidence |",
            "| --- | --- | --- |",
            *_g0_detail_table(g0),
            "",
            "The released score-generation script uses an 80-pose proposal horizon, but the official default metric cache uses 40 proposal poses and 50 future poses. This conflict is also recorded in [WoTE issue #16](https://github.com/liyingyanUCAS/WoTE/issues/16). The formal audit used the explicit official default 40-pose proposal horizon; it did not reproduce the released factors, and the 1e-6 threshold was not relaxed.",
            "",
            "## Table 1: Candidate headroom",
            "",
            "| Candidate set | Selected score | Oracle score | Regret | Recoverable scenes |",
            "| --- | ---: | ---: | ---: | ---: |",
            _candidate_table(g1),
            "",
            "## Table 2: Effect representation value",
            "",
            "| Model | Future input | Candidate-specific | Selected PDMS | Regret | Pairwise acc. | False-safe |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
            *_effect_table(probe_metrics),
            "",
            "## Table 3: Inverse",
            "",
            "| Effect input | Top-1 retrieval | MRR | Delta sign acc. | PDMS with gate | False-safe |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
            *_inverse_table(inverse_metrics),
            "",
            "## Table 4: Gate decision",
            "",
            "| Gate | Status | Primary evidence | Decision |",
            "| --- | --- | --- | --- |",
            *_gate_evidence(statuses, g0, g1, g2, g3, g4),
            "",
            "## Evidence boundaries",
            "",
            "- All reported confidence intervals use paired scene-level bootstrap units.",
            "- Validation alone selects loss weights, fusion coefficients, and rejection thresholds.",
            "- A missing dependent artifact is `NOT_RUN`, never silently converted to a failed experiment.",
            "- Because candidate-label alignment failed G0, G1 through G4 were not run and make no positive or negative claim about effect modeling.",
            "- No module generates actor braking, yielding, or other reactive pseudo-labels.",
            "",
        ]
    )
    gate_path = report / "GATE_REPORT.md"
    if gate_path.exists():
        raise FileExistsError(f"refusing existing Gate report: {gate_path}")
    gate_path.write_text(gate_report, encoding="utf-8")

    split_dir = args.project_root / "research/cf_effect_gate_wote/configs/splits"
    split_counts = {
        split: len(
            [line for line in (split_dir / f"{split}_tokens.txt").read_text().splitlines() if line]
        )
        if (split_dir / f"{split}_tokens.txt").is_file()
        else 0
        for split in ("train", "val", "test")
    }
    baseline_path = (
        args.project_root
        / "research/cf_effect_gate_wote/configs/worktree_baseline.json"
    )
    baseline = _json(baseline_path)
    if baseline is None:
        baseline_lines = ["Worktree baseline audit: unavailable."]
    else:
        source = baseline["source_repository"]
        baseline_lines = [
            "## Worktree isolation audit",
            "",
            f"Source checkout: `{source['logical_name']}` on `{source['branch']}` at "
            f"`{source['head']}`; dirty=`{str(bool(source['dirty'])).lower()}` with "
            f"{len(source['dirty_paths'])} recorded paths.",
            "The exact portable snapshot is tracked at "
            "`research/cf_effect_gate_wote/configs/worktree_baseline.json`.",
            "No pre-existing worktree was reset, cleaned, pruned, or reused.",
            "",
            "| Existing worktree | Branch | HEAD | Prunable at capture |",
            "| --- | --- | --- | --- |",
            *[
                f"| {entry['logical_name']} | {entry.get('branch') or 'detached'} | "
                f"`{entry['head']}` | {'yes' if entry.get('prunable', False) else 'no'} |"
                for entry in baseline["existing_worktrees"]
            ],
            "",
        ]
    reproduction = "\n".join(
        [
            "# Reproduction",
            "",
            "Runtime path precedence is CLI > one-shot `CF_GATE_*` environment > task-local defaults.",
            "Every launcher supports `--dry-run` and `--preflight-only` and refuses existing outputs.",
            "",
            f"Fixed split counts: `{json.dumps(split_counts, sort_keys=True)}`.",
            "",
            "## G0 stop boundary",
            "",
            "The 200-scene feature export and deterministic cache checks completed. Candidate-label auditing stopped the experiment because the released factors could not be reproduced at the required 1e-6 tolerance. The published generator/default-cache horizon conflict is tracked upstream in [WoTE issue #16](https://github.com/liyingyanUCAS/WoTE/issues/16). G1-G4 therefore remain `NOT_RUN`.",
            "",
            *baseline_lines,
            "```bash",
            "bash research/cf_effect_gate_wote/scripts/setup_wote_gate.sh --help",
            "bash research/cf_effect_gate_wote/scripts/run_gate0_smoke.sh --help",
            "bash research/cf_effect_gate_wote/scripts/run_gate1_candidate_oracle.sh --help",
            "bash research/cf_effect_gate_wote/scripts/run_gate2_replay_effect.sh --help",
            "bash research/cf_effect_gate_wote/scripts/run_gate3_effect_prediction.sh --help",
            "bash research/cf_effect_gate_wote/scripts/run_gate4_inverse.sh --help",
            "bash research/cf_effect_gate_wote/scripts/build_report.sh --help",
            "pytest research/cf_effect_gate_wote/tests -q",
            "```",
            "",
        ]
    )
    reproduction_path = report / "REPRODUCTION.md"
    if reproduction_path.exists():
        raise FileExistsError(f"refusing existing reproduction report: {reproduction_path}")
    reproduction_path.write_text(reproduction, encoding="utf-8")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--gate01-run-id", default="gate0-smoke")
    parser.add_argument("--gate2-run-id", default="gate2-main")
    parser.add_argument("--gate3-run-id", default="gate3-main")
    parser.add_argument("--gate4-run-id", default="gate4-main")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    build_report(_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
