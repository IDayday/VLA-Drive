"""Headroom/oracle-effect contracts and fail-closed relabel Gate reporting."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import numpy.typing as npt
import pandas as pd

from .feature_store import atomic_write_json, sha256_file
from .independent_label_store import IndependentCandidateLabelStore
from .metrics import FACTOR_NAMES, candidate_ranks, paired_scene_bootstrap
from .replay_effect_builder import (
    ENGINEERED_EFFECT_NAMES,
    PRIMITIVE_ACTOR_EFFECT_NAMES,
    PRIMITIVE_EGO_EFFECT_NAMES,
    PRIMITIVE_MAP_EFFECT_NAMES,
)


BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_SEED = 20260827
FORBIDDEN_PRIMITIVE_NAMES = {
    "nc",
    "dac",
    "ep",
    "ttc",
    "comfort",
    "pdms",
    "epdms",
    "score",
    "selected_index",
    "oracle_index",
    "collision_flag",
    "time_to_collision_final_metric",
    "footprint_outside_drivable_ratio",
    "aggregated_evaluator_progress",
}


class OracleEffectGateError(RuntimeError):
    """A headroom/probe identity or decision contract was violated."""


@dataclass(frozen=True)
class ProbeSplit:
    train: tuple[str, ...]
    val: tuple[str, ...]
    test: tuple[str, ...]

    def validate(
        self,
        original_train: Sequence[str],
        original_val: Sequence[str],
        original_test: Sequence[str],
        headroom: Sequence[str],
    ) -> None:
        expected = {"train": 1024, "val": 256, "test": 512}
        values = asdict(self)
        for name, count in expected.items():
            if len(values[name]) != count or len(set(values[name])) != count:
                raise OracleEffectGateError(
                    f"probe {name} expected {count} unique tokens, got {len(values[name])}"
                )
        sets = {name: set(tokens) for name, tokens in values.items()}
        if sets["train"] & sets["val"] or sets["train"] & sets["test"] or sets[
            "val"
        ] & sets["test"]:
            raise OracleEffectGateError("probe train/val/test splits overlap")
        if sets["test"] & set(headroom):
            raise OracleEffectGateError("probe test overlaps G1 headroom tokens")
        if not sets["train"] <= set(original_train):
            raise OracleEffectGateError("probe train escaped original train split")
        if not sets["val"] <= set(original_val):
            raise OracleEffectGateError("probe val escaped original val split")
        if not sets["test"] <= set(original_test):
            raise OracleEffectGateError("probe test escaped original test split")


def _read_tokens(path: Path) -> list[str]:
    values = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    if not values or any(not value for value in values) or len(values) != len(set(values)):
        raise OracleEffectGateError(f"invalid fixed token file: {path}")
    return values


def build_probe_split(split_dir: Path, headroom_path: Path) -> ProbeSplit:
    original_train = _read_tokens(split_dir / "train_tokens.txt")
    original_val = _read_tokens(split_dir / "val_tokens.txt")
    original_test = _read_tokens(split_dir / "test_tokens.txt")
    headroom = _read_tokens(headroom_path)
    split = ProbeSplit(
        train=tuple(original_train[:1024]),
        val=tuple(original_val[:256]),
        test=tuple(original_test[200:712]),
    )
    split.validate(original_train, original_val, original_test, headroom)
    return split


def write_probe_split(split: ProbeSplit, output_dir: Path) -> dict[str, str]:
    paths = {
        "train": output_dir / "probe_v1_train_tokens.txt",
        "val": output_dir / "probe_v1_val_tokens.txt",
        "test": output_dir / "probe_v1_test_tokens.txt",
    }
    for name, path in paths.items():
        if path.exists():
            raise FileExistsError(f"refusing existing probe split: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(f"{token}\n" for token in getattr(split, name)), encoding="utf-8")
    return {name: sha256_file(path) for name, path in paths.items()}


def primitive_schema() -> dict[str, Any]:
    groups = {
        "ego": tuple(PRIMITIVE_EGO_EFFECT_NAMES),
        "map": tuple(PRIMITIVE_MAP_EFFECT_NAMES),
        "actor_replay": tuple(PRIMITIVE_ACTOR_EFFECT_NAMES),
        "masks": ("actor_validity_mask", "interaction_uncertainty_mask"),
    }
    lowered = {name.lower() for values in groups.values() for name in values}
    overlap = lowered & FORBIDDEN_PRIMITIVE_NAMES
    if overlap:
        raise OracleEffectGateError(f"primitive schema leaks forbidden keys: {sorted(overlap)}")
    return {
        "schema_version": "primitive_effect.v1",
        "groups": groups,
        "engineered_schema_version": "engineered_effect.v1",
        "engineered_only": tuple(ENGINEERED_EFFECT_NAMES),
    }


def deterministic_effect_permutation(
    scene_token: str, candidates: int = 256, seed: int = BOOTSTRAP_SEED
) -> npt.NDArray[np.int64]:
    if candidates <= 1:
        raise ValueError("candidate effect shuffle requires at least two candidates")
    token_seed = int(hashlib.sha256(scene_token.encode("utf-8")).hexdigest()[:16], 16)
    rng = np.random.default_rng(token_seed ^ seed)
    permutation = rng.permutation(candidates).astype(np.int64)
    if np.array_equal(permutation, np.arange(candidates, dtype=np.int64)):
        permutation = np.roll(permutation, 1)
    return permutation


def shuffle_effect_only(
    effect: Mapping[str, npt.ArrayLike],
    permutation: npt.ArrayLike,
) -> dict[str, npt.NDArray[Any]]:
    order = np.asarray(permutation, dtype=np.int64)
    if order.ndim != 1 or not np.array_equal(np.sort(order), np.arange(len(order))):
        raise OracleEffectGateError("effect shuffle is not a candidate permutation")
    output: dict[str, npt.NDArray[Any]] = {}
    for name, value in effect.items():
        array = np.asarray(value)
        if name == "shared_logged_future":
            output[name] = array.copy()
        else:
            if array.shape[0] != len(order):
                raise OracleEffectGateError(
                    f"candidate-specific effect {name} lacks candidate axis"
                )
            output[name] = array[order].copy()
    return output


def candidate_headroom(
    label_root: Path,
    selected_indices: Mapping[str, int],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Evaluate the frozen WoTE base-anchor selector on independent labels."""

    store = IndependentCandidateLabelStore(label_root)
    rows: list[dict[str, Any]] = []
    factor_oracles: list[npt.NDArray[np.float32]] = []
    for scene in store.iter_scenes():
        token = scene.record.scene_token
        if token not in selected_indices:
            raise OracleEffectGateError(f"WoTE selection missing scene {token}")
        selected_index = int(selected_indices[token])
        if not 0 <= selected_index < 256:
            raise OracleEffectGateError(f"invalid WoTE selected index for {token}")
        selected_score = float(scene.score[selected_index])
        oracle_score = float(scene.score[scene.oracle_index])
        rank = int(candidate_ranks(scene.score[None])[0, selected_index])
        gap = oracle_score - selected_score
        rows.append(
            {
                "scene_token": token,
                "selected_index": selected_index,
                "oracle_index": scene.oracle_index,
                "selected_score_raw": selected_score,
                "selected_score_points": selected_score * 100.0,
                "oracle_score_raw": oracle_score,
                "oracle_score_points": oracle_score * 100.0,
                "oracle_gap_raw": gap,
                "oracle_gap_points": gap * 100.0,
                "selected_rank": rank,
                "has_better_candidate": bool(gap > 0),
                "improvement_gt_0_01": bool(gap > 0.01),
                "improvement_gt_0_02": bool(gap > 0.02),
                "improvement_gt_0_05": bool(gap > 0.05),
                "zero_score_selected": bool(selected_score == 0),
                "zero_to_positive_recovery": bool(
                    selected_score == 0 and oracle_score > 0
                ),
                "low_score_selected": bool(selected_score < 0.30),
                "low_score_improvement_gt_0_10": bool(
                    selected_score < 0.30 and gap > 0.10
                ),
            }
        )
        factor_oracles.append(scene.factors.max(axis=0))
    frame = pd.DataFrame(rows)
    selected_values = frame["selected_score_raw"].to_numpy(dtype=np.float64)
    oracle_values = frame["oracle_score_raw"].to_numpy(dtype=np.float64)
    gap_interval = paired_scene_bootstrap(
        oracle_values,
        selected_values,
        samples=BOOTSTRAP_SAMPLES,
        seed=BOOTSTRAP_SEED,
    )
    zero_count = int(frame["zero_score_selected"].sum())
    recovery_fraction = (
        float(frame["zero_to_positive_recovery"].sum() / zero_count)
        if zero_count
        else 0.0
    )
    if zero_count >= 10:
        recovery_ok = recovery_fraction >= 0.15
        recovery_status = "APPLICABLE"
    else:
        recovery_ok = True
        recovery_status = "NOT_APPLICABLE_TOO_FEW_ZERO_SCENES"
    mean_gap = float(frame["oracle_gap_raw"].mean())
    fraction_gt_001 = float(frame["improvement_gt_0_01"].mean())
    headroom_ok = mean_gap >= 0.020 or fraction_gt_001 >= 0.20
    summary = {
        "selector": "WoTE base-anchor selector",
        "scenes": len(frame),
        "selected_score_raw": float(selected_values.mean()),
        "selected_score_points": float(selected_values.mean() * 100.0),
        "oracle_score_raw": float(oracle_values.mean()),
        "oracle_score_points": float(oracle_values.mean() * 100.0),
        "mean_oracle_gap_raw": mean_gap,
        "mean_oracle_gap_points": mean_gap * 100.0,
        "oracle_gap_ci95": asdict(gap_interval),
        "better_scene_fraction": float(frame["has_better_candidate"].mean()),
        "fraction_improvement_gt_0_01": fraction_gt_001,
        "fraction_improvement_gt_0_02": float(frame["improvement_gt_0_02"].mean()),
        "fraction_improvement_gt_0_05": float(frame["improvement_gt_0_05"].mean()),
        "zero_score_scene_count": zero_count,
        "zero_to_positive_recovery_fraction": recovery_fraction,
        "recovery_status": recovery_status,
        "low_score_scene_count": int(frame["low_score_selected"].sum()),
        "low_score_improvement_gt_0_10_fraction": float(
            frame.loc[
                frame["low_score_selected"], "low_score_improvement_gt_0_10"
            ].mean()
        )
        if frame["low_score_selected"].any()
        else None,
        "oracle_factor_ceilings": {
            name: float(value)
            for name, value in zip(FACTOR_NAMES, np.stack(factor_oracles).mean(axis=0))
        },
        "headroom_ok": headroom_ok,
        "recovery_ok": recovery_ok,
        "gate_g1r": "PASS" if headroom_ok and recovery_ok else "FAIL",
    }
    return frame, summary


def oracle_effect_verdict(
    *,
    direct_pass: bool,
    static_pass: bool,
    shared_pass: bool,
    specificity_pass: bool,
    primitive_pass: bool,
    engineered_pass: bool,
) -> str:
    if engineered_pass and not primitive_pass:
        return "METRIC_PROXY_DEPENDENT"
    if not direct_pass:
        return "ORACLE_EFFECT_NOT_USEFUL"
    if not static_pass:
        return "STATIC_GEOMETRY_ONLY"
    if not shared_pass or not specificity_pass:
        return "CANDIDATE_SPECIFICITY_UNPROVEN"
    if primitive_pass:
        return "ORACLE_ACTION_EFFECT_VIABLE"
    return "ORACLE_EFFECT_NOT_USEFUL"


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing existing report artifact: {path}")
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_text(path: Path, text: str) -> None:
    if path.exists():
        raise FileExistsError(f"refusing existing report artifact: {path}")
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def build_g0_failure_reports(
    report_dir: Path,
    failure_path: Path,
    token_path: Path,
    checkpoint_path: Path,
    anchor_path: Path,
    evaluator_contract_path: Path,
) -> None:
    """Materialize the complete report surface after the mandatory early stop."""

    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    report_dir.mkdir(parents=True, exist_ok=True)
    contract_sha = sha256_file(evaluator_contract_path)
    evaluator_contract = json.loads(
        evaluator_contract_path.read_text(encoding="utf-8")
    )
    assets = {
        "status": "VERIFIED",
        "label_source": "independent_navsim_recompute",
        "wote_commit": evaluator_contract["wote_commit"],
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "candidate_bank_sha256": sha256_file(anchor_path),
        "candidate_bank_logical_sha256": evaluator_contract[
            "candidate_bank_logical_sha256"
        ],
        "evaluator_contract_sha256": contract_sha,
        "evaluator_source_hashes": evaluator_contract["evaluator_source_hashes"],
        "relabel_headroom_tokens_sha256": sha256_file(token_path),
        "relabel_headroom_token_count": 200,
        "published_candidate_scores_required": False,
    }
    atomic_write_json(report_dir / "ASSET_MANIFEST.json", assets)
    verdict = {
        "upstream_published_label_contract": "FAILED_PREVIOUS_AUDIT",
        "independent_relabel_contract": "FAIL",
        "candidate_headroom_gate": "NOT_RUN",
        "oracle_effect_gate": "NOT_RUN",
        "static_geometry_control": "NOT_RUN",
        "candidate_specificity_control": "NOT_RUN",
        "final_verdict": "RELABEL_CONTRACT_FAILED",
        "scientific_hypothesis_status": "UNTESTED",
        "blocking_evidence": [
            {
                "scene_token": failure.get("scene_token"),
                "reason": failure.get("reason", failure.get("error")),
                "maximum_score_reconstruction_error": failure.get(
                    "max_absolute_error"
                ),
                "mismatched_candidates": failure.get(
                    "mismatched_candidate_count"
                ),
            }
        ],
        "positive_evidence": [],
        "next_recommended_experiment": (
            "Resolve the evaluator/five-factor label contract explicitly; do not run "
            "candidate-headroom or action-effect probes until score reconstruction is exact."
        ),
    }
    atomic_write_json(report_dir / "VERDICT.json", verdict)
    consistency_summary = {
        "status": "FAIL",
        "gate": "G0-R",
        "run1": "FAIL",
        "run2": "NOT_RUN",
        "reason": failure.get("reason", failure.get("error")),
        "attempted_scenes": failure.get("attempted_scenes", 0),
        "completed_scenes": failure.get("completed_scenes", 0),
        "candidates_evaluated": failure.get("candidates_evaluated", 0),
        "run1_logical_sha256": None,
        "run2_logical_sha256": None,
        "max_run_to_run_error": None,
        "score_reconstruction_max_absolute_error": failure.get(
            "max_absolute_error"
        ),
    }
    atomic_write_json(
        report_dir / "relabel_consistency_summary.json", consistency_summary
    )
    _write_csv(
        report_dir / "relabel_consistency.csv",
        (
            "scene_token",
            "run1_status",
            "run2_status",
            "max_run_to_run_error",
            "score_reconstruction_error",
        ),
        [
            {
                "scene_token": failure.get("scene_token"),
                "run1_status": "FAIL",
                "run2_status": "NOT_RUN",
                "max_run_to_run_error": "",
                "score_reconstruction_error": failure.get("max_absolute_error"),
            }
        ],
    )
    atomic_write_json(
        report_dir / "published_vs_independent_summary.json",
        {
            "status": "UPSTREAM_REPRODUCTION_AUDIT_ONLY",
            "comparison_status": "NOT_RUN",
            "reason": "independent label store did not pass G0-R",
            "gate_blocking": False,
        },
    )
    _write_csv(
        report_dir / "published_vs_independent_audit.csv",
        ("scene_token", "factor", "status", "reason"),
        (),
    )
    _write_csv(
        report_dir / "candidate_headroom_summary.csv",
        (
            "selector",
            "status",
            "selected_score_raw",
            "oracle_score_raw",
            "gap_raw",
            "better_scene_fraction",
            "recovery",
        ),
        [
            {
                "selector": "WoTE base-anchor selector",
                "status": "NOT_RUN",
                "selected_score_raw": "",
                "oracle_score_raw": "",
                "gap_raw": "",
                "better_scene_fraction": "",
                "recovery": "",
            }
        ],
    )
    _write_csv(
        report_dir / "oracle_effect_probe_metrics.csv",
        (
            "model",
            "seed",
            "status",
            "selected_score",
            "regret",
            "pairwise_accuracy",
            "false_safe",
            "delta_vs_direct",
        ),
        (),
    )
    _write_csv(
        report_dir / "oracle_effect_ablation.csv",
        ("ablation", "status", "selected_score_delta", "regret_delta"),
        (),
    )
    _write_csv(
        report_dir / "failure_cases.csv",
        (
            "gate",
            "scene_token",
            "candidate_index",
            "evaluator_score",
            "reassembled_score",
            "absolute_error",
            "driving_direction_compliance",
        ),
        [
            {
                "gate": "G0-R",
                "scene_token": failure.get("scene_token"),
                **row,
            }
            for row in failure.get("candidate_differences", [])
        ],
    )
    empty_headroom = pd.DataFrame(
        columns=["scene_token", "selected_score_raw", "oracle_score_raw"]
    )
    empty_probe = pd.DataFrame(
        columns=["scene_token", "model", "seed", "selected_score", "regret"]
    )
    for path, frame in (
        (report_dir / "candidate_headroom_scene_level.parquet", empty_headroom),
        (report_dir / "oracle_effect_scene_level.parquet", empty_probe),
    ):
        if path.exists():
            raise FileExistsError(f"refusing existing report artifact: {path}")
        frame.to_parquet(path, index=False)

    relabel_report = f"""# Independent Relabel Report

G0-R is **FAIL**. The run stopped at the first fixed scene, as required by the
score-reconstruction hard gate.

| Check | Result |
| --- | ---: |
| Requested scenes | 200 |
| Attempted scenes | {failure.get('attempted_scenes', 0)} |
| Completed scenes | {failure.get('completed_scenes', 0)} |
| Candidates evaluated before stop | {failure.get('candidates_evaluated', 0)} |
| Run1 logical SHA256 | NOT_AVAILABLE |
| Run2 logical SHA256 | NOT_RUN |
| Max run-to-run error | NOT_RUN |
| Score reconstruction error | {failure.get('max_absolute_error')} |
| G0-R | FAIL |

The official scorer includes Driving Direction Compliance as a multiplicative
term. The required independent store contains exactly NC, DAC, EP, TTC and
Comfort, so its required five-factor reconstruction cannot reproduce those
candidate scores within `1e-6`. No factor was merged, dropped, or redefined.
"""
    _write_text(report_dir / "RELABEL_REPORT.md", relabel_report)
    _write_text(
        report_dir / "G1_HEADROOM_REPORT.md",
        """# G1-R Candidate Headroom Report

Status: **NOT_RUN**.

G0-R failed its mandatory independent-label reconstruction contract. Therefore
no candidate score, oracle gap, rank, or recovery statistic is reported.
""",
    )
    _write_text(
        report_dir / "G2_ORACLE_EFFECT_REPORT.md",
        """# G2-O Oracle Action-Effect Report

Status: **NOT_RUN**.

G2-O was gated first by G0-R and then by G1-R. Neither independent training
labels nor a passed candidate-headroom Gate exist, so no probe was trained and
no action-effect conclusion is drawn.
""",
    )
    _write_text(
        report_dir / "REPRODUCTION.md",
        f"""# Reproduction

The run used the fixed first 200 tokens in `relabel_headroom_tokens.txt`, the
released 256×8×3 base-anchor bank, proposal sampling 40×0.1 s, and the pinned
evaluator sources in `EVALUATOR_CONTRACT.json`.

Evaluator contract SHA256: `{contract_sha}`.

Run order stopped during G0-R run1 on scene
`{failure.get('scene_token')}` after all 256 candidates were evaluated. Run2,
the published-label audit, G1-R, feature caching, effect construction, and G2-O
were not run because the five-factor score reconstruction error exceeded
`1e-6`.
""",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    split = commands.add_parser("build-probe-split")
    split.add_argument("--split-dir", type=Path, required=True)
    split.add_argument("--headroom-tokens", type=Path, required=True)
    split.add_argument("--output-dir", type=Path, required=True)
    headroom = commands.add_parser("headroom")
    headroom.add_argument("--labels", type=Path, required=True)
    headroom.add_argument("--selected-indices", type=Path, required=True)
    headroom.add_argument("--output-parquet", type=Path, required=True)
    headroom.add_argument("--output-summary", type=Path, required=True)
    reports = commands.add_parser("build-g0-failure-reports")
    reports.add_argument("--report-dir", type=Path, required=True)
    reports.add_argument("--failure", type=Path, required=True)
    reports.add_argument("--tokens", type=Path, required=True)
    reports.add_argument("--checkpoint", type=Path, required=True)
    reports.add_argument("--anchors", type=Path, required=True)
    reports.add_argument("--evaluator-contract", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "build-probe-split":
        split = build_probe_split(args.split_dir, args.headroom_tokens)
        print(json.dumps(write_probe_split(split, args.output_dir), indent=2))
        return 0
    if args.command == "headroom":
        selected = json.loads(args.selected_indices.read_text(encoding="utf-8"))
        frame, summary = candidate_headroom(args.labels, selected)
        if args.output_parquet.exists():
            raise FileExistsError(f"refusing existing output: {args.output_parquet}")
        args.output_parquet.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(args.output_parquet, index=False)
        atomic_write_json(args.output_summary, summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0 if summary["gate_g1r"] == "PASS" else 4
    build_g0_failure_reports(
        args.report_dir,
        args.failure,
        args.tokens,
        args.checkpoint,
        args.anchors,
        args.evaluator_contract,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
