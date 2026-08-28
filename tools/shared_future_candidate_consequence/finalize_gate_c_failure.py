#!/usr/bin/env python3
"""Finalize transparent NOT-RUN reports when formal Gate C1 fails."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd

from .common import (
    DEFAULT_CACHE_DIR,
    DEFAULT_REPORT_DIR,
    append_command,
    write_json,
    write_markdown,
)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(value: float | None, digits: int = 4) -> str:
    return "NOT RUN" if value is None else f"{value:.{digits}f}"


def run(args: argparse.Namespace) -> dict[str, Any]:
    report = args.output_dir
    oracle = _load(report / "oracle_decomposition_results.json")
    if oracle["gate_c1"] != "FAIL":
        raise RuntimeError("Failure finalizer is only valid after a formal Gate C1 FAIL")
    reproduction = _load(report / "reproduction_results.json")
    split = _load(report / "dataset_split_summary.json")
    target = _load(report / "all_log_pipeline_summary.json")
    store = _load(report / "oracle_store_summary.json")
    model = _load(report / "model_candidate_oracle_results.json")
    candidate = _load(report / "candidates/model_candidate_bank_summary.json")
    primary = oracle["primary"]
    model_o5_selected = (
        model["best_of_16_mean_official_score"]
        - model["primary"]["O5"]["top1_regret_mean"]
    )
    static_gain = primary["O3"]["pairwise_mean"] - primary["O2"]["pairwise_mean"]
    raw_state_gain = primary["O4"]["pairwise_mean"] - primary["O3"]["pairwise_mean"]
    direct_risk_gain = primary["O5"]["pairwise_mean"] - primary["O3"]["pairwise_mean"]
    signal_gain = primary["O6"]["pairwise_mean"] - primary["O3"]["pairwise_mean"]
    failed_criteria = [name for name, passed in oracle["criteria"].items() if not passed]
    blocker = "; ".join(failed_criteria)

    training_results = {
        "status": "NOT_RUN",
        "reason": "Formal Gate C1 failed; protocol forbids current-observation model development.",
        "gate_c1": "FAIL",
        "failed_criteria": failed_criteria,
        "inference_future_inputs_used": False,
        "models": {
            "M0_original_episode_drive_scorer": "AUDITED_OFFLINE",
            "M1_direct_scene_trajectory_scorer": "NOT_RUN",
            "M2_direct_consequence": "NOT_RUN",
            "M3_shared_future_factorized": "NOT_RUN",
            "M4_visual_anchor": "NOT_RUN",
            "M5_consistency_verifier": "NOT_RUN",
            "M6_oracle_structured_consequence": "UPPER_BOUND_ONLY",
            "M7_shuffled_consequence": "ORACLE_CONTROL_ONLY",
            "M8_random_high_dimensional": "ORACLE_CONTROL_ONLY",
        },
        "predicted_consequence_pairwise": None,
        "visual_anchor": False,
        "consistency_verifier": False,
        "num_training_seeds_completed": 0,
    }
    write_json(report / "training_results.json", training_results)

    selection_results = {
        "status": "OFFLINE_BASELINE_AND_ORACLE_ONLY",
        "scene_count": model["scene_count"],
        "log_count": model["log_count"],
        "candidates_per_scene": model["candidates_per_scene"],
        "original_scorer_mean_official_score": model["baseline_selected_mean_official_score"],
        "random_mean_official_score": candidate["random_expected_mean_score"],
        "o8_oracle_ranker_mean_official_score": model["o8_ranker_selected_mean_official_score"],
        "o5_direct_risk_ranker_mean_official_score": model_o5_selected,
        "best_of_16_mean_official_score": model["best_of_16_mean_official_score"],
        "best_of_16_headroom": model["best_of_16_headroom"],
        "o8_oracle_ranker_beats_original": model["o8_ranker_beats_original_scorer"],
        "predicted_model_selection": "NOT_RUN",
        "official_scorer_used_at_inference": False,
    }
    write_json(report / "selection_results.json", selection_results)

    selection = pd.read_parquet(report / "candidates/episode_drive_selection_evaluation.parquet")
    numeric = [
        "baseline_selected_official_score_selected16",
        "random_expected_official_score_selected16",
        "oracle_best_official_score_selected16",
        "baseline_top1_regret_selected16",
        "baseline_selected_collision_free",
        "baseline_selected_ttc",
        "baseline_selected_dac",
        "baseline_selected_ddc",
        "baseline_selected_progress",
        "baseline_selected_comfort",
    ]
    per_log = selection.groupby(["log_name", "fold"], as_index=False)[numeric].mean()
    per_log["scene_count"] = selection.groupby(["log_name", "fold"]).size().to_numpy()
    per_log["predicted_model_status"] = "NOT_RUN_GATE_C1"
    per_log.to_csv(report / "per_log_results.csv", index=False)
    pd.DataFrame(
        [
            {
                "seed": 20260828 + offset,
                "status": "NOT_RUN_GATE_C1",
                "predicted_pairwise": None,
                "selected_mean_official_score": None,
            }
            for offset in range(3)
        ]
    ).to_csv(report / "per_seed_results.csv", index=False)

    write_markdown(
        report / "DYNAMIC_PREDICTION_REPORT.md",
        f"""# Dynamic Prediction Report

## Gate C2: NOT RUN

Formal Gate C1 failed before current-observation model development. No direct
candidate-consequence head or shared-future head was trained, so this audit does
not claim that current images can or cannot predict the logged future.

- Oracle O3/O8 pairwise: {primary['O3']['pairwise_mean']:.4f} / {primary['O8']['pairwise_mean']:.4f}
- Oracle dynamic gain: {oracle['dynamic_gain']:.4f}
- Failed Gate C1 criteria: {', '.join(failed_criteria)}
- Future files read by a deployable inference path: no such path was created
""",
    )
    write_markdown(
        report / "SHARED_VS_DIRECT_REPORT.md",
        f"""# Shared Future vs Direct Consequence

## Deployable comparison: NOT RUN

The protocol permits this comparison only after Gate C1. The available numbers
are logged-future upper bounds, not current-observation predictions:

| Oracle input | Pairwise |
|---|---:|
| O4 raw dynamic actor state | {primary['O4']['pairwise_mean']:.4f} |
| O5 direct physical risk | {primary['O5']['pairwise_mean']:.4f} |
| O8 full dynamic consequence | {primary['O8']['pairwise_mean']:.4f} |
| O9 actor state + recomputed physical risk | {primary['O9']['pairwise_mean']:.4f} |

These results cannot establish superiority of `SharedFutureFactorized` or
`DirectCandidateConsequenceHead` because neither model was trained.
""",
    )
    write_markdown(
        report / "VISUAL_ANCHOR_ABLATION.md",
        """# GT Future Visual-anchor Ablation

## Decision: NOT RUN / EXCLUDED FROM THE FINAL METHOD

The prior feasibility audit established the GT-only future-image data path, but
Gate C1 failed before prediction training. No future visual embedding was used,
no non-GT candidate received an image target, and no planning improvement is
claimed. The teacher branch therefore remains absent at inference and from the
recommended method.
""",
    )
    write_markdown(
        report / "CONSISTENCY_VERIFIER_REPORT.md",
        """# Candidate–Consequence Consistency Verifier

## Decision: NOT RUN

No predicted consequence exists after Gate C1 failure, so training a verifier
would test synthetic mismatch recognition without a deployable upstream signal.
AUROC, confidence correlation and planning gain are intentionally unreported.
The strong-inverse-dynamics task remains out of scope.
""",
    )
    write_markdown(
        report / "MODEL_CANDIDATE_SELECTION_REPORT.md",
        f"""# Frozen EpisodeDrive Candidate Selection

## Gate C3: NOT RUN

| Selector / upper bound | Mean offline official score |
|---|---:|
| Random retained proposal | {candidate['random_expected_mean_score']:.4f} |
| Original frozen EpisodeDrive scorer | {model['baseline_selected_mean_official_score']:.4f} |
| O5 logged-future direct-risk ranker | {model_o5_selected:.4f} |
| O8 logged-future oracle ranker | {model['o8_ranker_selected_mean_official_score']:.4f} |
| Best of retained K=16 | {model['best_of_16_mean_official_score']:.4f} |

- Scenes/logs: {model['scene_count']:,}/{model['log_count']:,}
- Original-scorer best-of-K headroom: {model['best_of_16_headroom']:.4f}
- O8 oracle ranker beats original scorer: {model['o8_ranker_beats_original_scorer']}
- Predicted shared/direct model: NOT RUN

Official scores above are offline evaluation labels. No inference selector called
the official scorer, and no deployable planning-gain claim is made.
""",
    )

    group_rows = "\n".join(
        f"| {group} | {primary[group]['pairwise_mean']:.4f} | {primary[group]['top1_regret_mean']:.4f} |"
        for group in ("O0", "O1", "O2", "O3", "O4", "O5", "O6", "O7", "O8", "O9", "O10", "O11", "O12", "O13")
    )
    criterion_rows = "\n".join(
        f"| `{name}` | {'PASS' if passed else 'FAIL'} |"
        for name, passed in oracle["criteria"].items()
    )
    write_markdown(
        report / "FINAL_GATE_C_REPORT.md",
        f"""# Shared Future–Candidate Consequence Gate C Final Report

## Outcome

- Gate C0 reproduction: PASS ({reproduction['scene_count']} scenes, exact max error 0)
- Gate C1 oracle dynamic incremental value: FAIL
- Gate C2 current-observation predictability: NOT RUN
- Gate C3 real-candidate planning gain: NOT RUN
- Final route: **Route E** under the predeclared decision rule

Route E does not mean the measured dynamic signal is exactly zero. It means the
expanded, log-balanced and real-proposal evidence does not satisfy the minimum
support required to spend model-training budget or claim a shared-future world
model. A narrower physical-risk distillation study would be a different method.

## Environment and data

- Base commit: `6e96cf7321b134c42c2cf0fbbc315cd61c925b11`
- Branch: `feature/shared-future-candidate-consequence-gate-c`
- Legal split: `trainval`
- Selected/scanned scenes: {split['selected_scene_count']:,}/{split['metric_cache_rows']:,}
- Selected logs: {split['selected_log_count']:,}; maximum {split['per_log_max']} scenes/log
- Five log-disjoint folds: {split['fold_scene_counts']}
- Candidate bank: randomized controlled K=16 and frozen EpisodeDrive K=16
- Traffic setting: non-reactive candidate-conditioned relabeling of trainval logged future
- Reactive/navtest response cache: NOT RUN (forbidden for training/tuning)
- Synthetic follow-up data: discovered in environment audit but not mixed into any Gate C result
- EpisodeDrive checkpoint SHA256: `7f7ce61b9492936d9af15c7bb68dc57d53c6629fccd23ac5ae530058286a807d`
- Candidate/target coverage: {target['candidate_coverage']:.3%}/{target['target_coverage']:.3%}
- Oracle completed prefix: {store['completed_scene_count']:,}/{store['scene_count']:,}
- Audited construction failure: {target.get('failure_examples', [])[:1]}

## Formal oracle decomposition

| Group | Pairwise | Top-1 regret |
|---|---:|---:|
{group_rows}

- O8−O3 dynamic gain: {oracle['dynamic_gain']:.4f}
- Equal-log point estimate: {oracle['equal_log_dynamic_gain']:.4f}
- Equal-log bootstrap 95% CI: [{oracle['dynamic_gain_log_bootstrap_95ci'][0]:.4f}, {oracle['dynamic_gain_log_bootstrap_95ci'][1]:.4f}]
- Statistical note: the Gate threshold uses the mean of five fold-level pairwise accuracies; the bootstrap gives every log equal weight, so these two point estimates need not coincide.
- Top-1 regret reduction: {oracle['top1_regret_reduction']:.2%}
- O9 state/recomputed-risk gain retention: {oracle['state_recomputed_risk_gain_retention']:.3f}
- Held-out candidate-family mean/worst gain: {oracle['heldout_candidate_type_gain_mean']:.4f}/{oracle['heldout_candidate_type_gain_worst']:.4f}

| Gate C1 criterion | Result |
|---|---|
{criterion_rows}

## Frozen EpisodeDrive proposal evidence

- O3/O4/O5/O8/O9 pairwise: {model['primary']['O3']['pairwise_mean']:.4f} / {model['primary']['O4']['pairwise_mean']:.4f} / {model['primary']['O5']['pairwise_mean']:.4f} / {model['primary']['O8']['pairwise_mean']:.4f} / {model['primary']['O9']['pairwise_mean']:.4f}
- O8−O3 gain and 95% CI: {model['dynamic_pairwise_gain']:.4f}, [{model['dynamic_gain_log_bootstrap_95ci'][0]:.4f}, {model['dynamic_gain_log_bootstrap_95ci'][1]:.4f}]
- Raw-state/direct-risk/recomputed-risk gains: {model['raw_dynamic_state_pairwise_gain']:.4f} / {model['direct_physical_risk_pairwise_gain']:.4f} / {model['state_recomputed_risk_pairwise_gain']:.4f}
- Original scorer / O5 risk ranker / O8 full-dynamic ranker / best-of-K mean score: {model['baseline_selected_mean_official_score']:.4f} / {model_o5_selected:.4f} / {model['o8_ranker_selected_mean_official_score']:.4f} / {model['best_of_16_mean_official_score']:.4f}

## Required questions

1. **Does 0.764 reproduce on more logs?** {'Yes for the absolute O8 metric' if primary['O8']['pairwise_mean'] >= 0.764 else 'No'}. The formal O0 trajectory-only and O8 values are {primary['O0']['pairwise_mean']:.4f} and {primary['O8']['pairwise_mean']:.4f}; the more stringent conditional dynamic increment over O3 is {oracle['dynamic_gain']:.4f}. The predeclared +0.03 increment criterion is {'PASS' if oracle['criteria']['dynamic_pairwise_gain_at_least_0p03'] else 'FAIL'}; the overall Gate C1 result also includes the independent controls below.
2. **Where does the gain come from?** Static-map O3 adds {static_gain:+.4f} over O2; raw actor state O4 adds {raw_state_gain:+.4f}, direct collision/TTC-adjacent physical risk O5 adds {direct_risk_gain:+.4f}, and future signal O6 adds {signal_gain:+.4f} over O3. Direct risk is therefore reported separately from raw actor state rather than packaged as a generic world-model gain.
3. **How much survives without direct collision/TTC?** O9 retention is {oracle['state_recomputed_risk_gain_retention']:.3f}; O9 recomputes risk from actor state and masks rather than ingesting official factors.
4. **Can current vision predict dynamic consequence?** INCONCLUSIVE / NOT RUN because Gate C1 failed.
5. **Is shared-future prediction better than direct prediction?** INCONCLUSIVE / NOT RUN.
6. **Does the GT image anchor improve planning?** NOT RUN; it is excluded from the final method.
7. **Does the consistency verifier identify unreliable consequences?** NOT RUN.
8. **Are actual EpisodeDrive proposal choices improved?** No deployable model was trained. The O8 logged-future oracle ranker mean {model['o8_ranker_selected_mean_official_score']:.4f} does not beat the original scorer {model['baseline_selected_mean_official_score']:.4f}.
9. **Does the result depend on fixed candidate templates?** Candidate parameters/order/GT index were randomized. Held-out-family worst gain is {oracle['heldout_candidate_type_gain_worst']:.4f}; the corresponding Gate criterion is {'PASS' if oracle['criteria']['every_heldout_candidate_type_has_positive_gain'] else 'FAIL'}.
10. **Route?** Route E. Stop shared-future model integration under this protocol.

## Leakage and terminology

All official aggregate/factor values are physically isolated as offline targets.
O0–O13 receive no official score, future image or candidate-type label. The
structured current-actor features and HD-map relations in the oracle table are
explicitly oracle-only because EpisodeDrive does not consume them directly.
No deployable model was allowed to inherit those fields. The
construction is a non-reactive candidate-conditioned relabeling of one shared
logged future; it is not a true counterfactual future or true multi-agent
response. One audited invalid map-geometry scene remains a reported failure and
was not repaired by modifying source data.

## Primary blocker

{blocker}
""",
    )
    return {
        "gate_c1": "FAIL",
        "gate_c2": "NOT_RUN",
        "gate_c3": "NOT_RUN",
        "route": "E",
        "failed_criteria": failed_criteria,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR / "all")
    args = parser.parse_args()
    print(json.dumps(run(args), sort_keys=True))
    append_command(
        args.output_dir,
        "python -m tools.shared_future_candidate_consequence.finalize_gate_c_failure "
        + " ".join(sys.argv[1:]),
    )


if __name__ == "__main__":
    main()
