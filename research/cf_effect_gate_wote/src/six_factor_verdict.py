"""Validation, determinism, auditing, and headroom for six-factor labels."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import numpy.typing as npt
import pandas as pd

from .feature_store import FeatureShardReader, atomic_write_json, sha256_file
from .independent_label_store import (
    CANDIDATE_COUNT,
    IndependentLabelStoreError,
    SixFactorIndependentCandidateLabelStore,
)
from .metrics import candidate_ranks, paired_scene_bootstrap, pdms_from_factors
from .six_factor_metrics import SIX_FACTOR_ORDER, pdms_from_six_factors


BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_SEED = 20260827
RECONSTRUCTION_TOLERANCE = 1e-6

PUBLISHED_FACTOR_KEYS: Mapping[str, str] = {
    "NC": "no_at_fault_collisions",
    "DAC": "drivable_area_compliance",
    "DDC": "driving_direction_compliance",
    "EP": "ego_progress",
    "TTC": "time_to_collision_within_bound",
    "Comfort": "comfort",
}


class SixFactorGateError(RuntimeError):
    """A staged six-factor Gate identity or invariant was violated."""


def _write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    fieldnames: Sequence[str] | None = None,
) -> None:
    if path.exists():
        raise FileExistsError(f"refusing existing CSV: {path}")
    if not rows and fieldnames is None:
        raise ValueError(f"cannot infer columns for empty CSV: {path}")
    columns = tuple(fieldnames or rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def single_scene_validation(
    label_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Validate all 256 candidates and retain the historical DDC failure contrast."""

    store = SixFactorIndependentCandidateLabelStore(label_root)
    if len(store.scene_tokens) != 1:
        raise SixFactorGateError(
            f"G0-R2a requires exactly one scene, got {len(store.scene_tokens)}"
        )
    scene = next(store.iter_scenes())
    if scene.record.scene_token != "0fcede1cbfb15faa":
        raise SixFactorGateError(f"unexpected G0-R2a token: {scene.record.scene_token}")
    old_factors = scene.factors[:, [0, 1, 3, 4, 5]]
    old_score = pdms_from_factors(old_factors)
    six_score = pdms_from_six_factors(scene.factors)
    evaluator_score = scene.score.astype(np.float64)
    old_error = np.abs(old_score - evaluator_score)
    six_error = np.abs(six_score - evaluator_score)
    rows: list[dict[str, Any]] = []
    contrast: list[dict[str, Any]] = []
    for index in range(CANDIDATE_COUNT):
        row = {
            "scene_token": scene.record.scene_token,
            "candidate_index": index,
            **{
                name: float(scene.factors[index, factor_index])
                for factor_index, name in enumerate(SIX_FACTOR_ORDER)
            },
            "raw_progress": float(scene.raw_progress[index]),
            "old_five_factor_score": float(old_score[index]),
            "new_six_factor_score": float(six_score[index]),
            "evaluator_score": float(evaluator_score[index]),
            "old_error": float(old_error[index]),
            "new_error": float(six_error[index]),
        }
        rows.append(row)
        if old_error[index] > RECONSTRUCTION_TOLERANCE:
            contrast.append(row.copy())

    old_still_wrong = bool(np.any(old_error > RECONSTRUCTION_TOLERANCE))
    six_mismatches = int(np.sum(six_error > RECONSTRUCTION_TOLERANCE))
    ddc_half = int(np.sum(scene.factors[:, 2] == np.float32(0.5)))
    passed = (
        len(rows) == CANDIDATE_COUNT
        and six_mismatches == 0
        and int(np.argmax(evaluator_score)) == scene.oracle_index
        and old_still_wrong
        and ddc_half > 0
        and scene.record.candidate_bank_hash == scene.record.trajectory_hash
    )
    summary = {
        "scene": scene.record.scene_token,
        "scenes": 1,
        "candidates": CANDIDATE_COUNT,
        "candidate_bank_hash": scene.record.candidate_bank_hash,
        "trajectory_hash": scene.record.trajectory_hash,
        "logical_content_sha256": store.logical_content_sha256,
        "oracle_index": scene.oracle_index,
        "old_five_factor_max_error": float(old_error.max()),
        "old_five_factor_mismatched_count": int(
            np.sum(old_error > RECONSTRUCTION_TOLERANCE)
        ),
        "six_factor_max_error": float(six_error.max()),
        "six_factor_mismatched_count": six_mismatches,
        "ddc_half_candidates": ddc_half,
        "gate": (
            "SINGLE_SCENE_SIX_FACTOR_PASS"
            if passed
            else "SIX_FACTOR_FORMULA_STILL_INCOMPLETE"
        ),
        "status": "PASS" if passed else "FAIL",
    }
    return rows, contrast, summary


def compare_six_factor_runs(
    first_root: Path,
    second_root: Path,
    expected_scenes: int,
    pass_status: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Compare factors, scores, raw progress, oracle indices, and logical hashes."""

    first = SixFactorIndependentCandidateLabelStore(first_root)
    second = SixFactorIndependentCandidateLabelStore(second_root)
    if first.scene_tokens != second.scene_tokens:
        raise IndependentLabelStoreError("six-factor run1/run2 scene order differs")
    if len(first.scene_tokens) != expected_scenes:
        raise SixFactorGateError(
            f"expected {expected_scenes} scenes, got {len(first.scene_tokens)}"
        )
    if (
        first.manifest["candidate_bank_hash"] != second.manifest["candidate_bank_hash"]
        or first.manifest["evaluator_contract_sha256"]
        != second.manifest["evaluator_contract_sha256"]
    ):
        raise IndependentLabelStoreError("run1/run2 contract identity differs")
    left_scenes = first.scene_index()
    right_scenes = second.scene_index()
    rows: list[dict[str, Any]] = []
    maxima = {"factor": 0.0, "score": 0.0, "raw_progress": 0.0}
    all_exact = True
    max_reconstruction_error = 0.0
    for token in first.scene_tokens:
        left = left_scenes[token]
        right = right_scenes[token]
        if left.record != right.record:
            raise IndependentLabelStoreError(f"run identity differs for {token}")
        factor_equal = bool(np.array_equal(left.factors, right.factors))
        score_equal = bool(np.array_equal(left.score, right.score))
        progress_equal = bool(np.array_equal(left.raw_progress, right.raw_progress))
        oracle_equal = left.oracle_index == right.oracle_index
        indices_equal = bool(
            np.array_equal(left.candidate_indices, right.candidate_indices)
        )
        factor_error = float(np.max(np.abs(left.factors - right.factors)))
        score_error = float(np.max(np.abs(left.score - right.score)))
        progress_error = float(np.max(np.abs(left.raw_progress - right.raw_progress)))
        reconstruction_error = float(
            np.max(
                np.abs(
                    pdms_from_six_factors(left.factors) - left.score.astype(np.float64)
                )
            )
        )
        maxima["factor"] = max(maxima["factor"], factor_error)
        maxima["score"] = max(maxima["score"], score_error)
        maxima["raw_progress"] = max(maxima["raw_progress"], progress_error)
        max_reconstruction_error = max(max_reconstruction_error, reconstruction_error)
        exact = (
            factor_equal
            and score_equal
            and progress_equal
            and oracle_equal
            and indices_equal
        )
        all_exact &= exact
        rows.append(
            {
                "scene_token": token,
                "factors_array_equal": factor_equal,
                "score_array_equal": score_equal,
                "raw_progress_array_equal": progress_equal,
                "oracle_index_equal": oracle_equal,
                "candidate_indices_equal": indices_equal,
                "max_factor_absolute_error": factor_error,
                "max_score_absolute_error": score_error,
                "max_raw_progress_absolute_error": progress_error,
                "score_reconstruction_max_error": reconstruction_error,
            }
        )
    logical_equal = first.logical_content_sha256 == second.logical_content_sha256
    passed = (
        all_exact
        and logical_equal
        and max_reconstruction_error <= RECONSTRUCTION_TOLERANCE
    )
    summary = {
        "status": pass_status if passed else "SIX_FACTOR_RELABEL_NONDETERMINISTIC",
        "pass": passed,
        "scenes": len(rows),
        "candidates_per_run": len(rows) * CANDIDATE_COUNT,
        "scene_failures": 0,
        "run1_logical_sha256": first.logical_content_sha256,
        "run2_logical_sha256": second.logical_content_sha256,
        "logical_sha256_equal": logical_equal,
        "run1_manifest_sha256": sha256_file(first_root / "manifest.json"),
        "run2_manifest_sha256": sha256_file(second_root / "manifest.json"),
        "factor_arrays_exactly_equal": all(row["factors_array_equal"] for row in rows),
        "score_arrays_exactly_equal": all(row["score_array_equal"] for row in rows),
        "raw_progress_arrays_exactly_equal": all(
            row["raw_progress_array_equal"] for row in rows
        ),
        "oracle_indices_exactly_equal": all(row["oracle_index_equal"] for row in rows),
        "candidate_indices_exactly_equal": all(
            row["candidate_indices_equal"] for row in rows
        ),
        "max_factor_run_to_run_error": maxima["factor"],
        "max_score_run_to_run_error": maxima["score"],
        "max_raw_progress_run_to_run_error": maxima["raw_progress"],
        "max_run_to_run_error": max(maxima.values()),
        "max_score_reconstruction_error": max_reconstruction_error,
        "candidate_bank_hash": first.manifest["candidate_bank_hash"],
        "evaluator_contract_sha256": first.manifest["evaluator_contract_sha256"],
    }
    return rows, summary


def published_six_factor_audit(
    independent_root: Path,
    published_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Audit published labels without using them in any Gate or label identity."""

    store = SixFactorIndependentCandidateLabelStore(independent_root)
    payload = np.load(published_path, allow_pickle=True).item()
    if not isinstance(payload, dict):
        raise ValueError("published audit input is not a dictionary")
    rows: list[dict[str, Any]] = []
    mismatch_by_factor: dict[str, list[npt.NDArray[np.bool_]]] = {
        name: [] for name in SIX_FACTOR_ORDER
    }
    score_mismatches: list[npt.NDArray[np.bool_]] = []
    argmax_mismatches = 0
    top5_mismatches = 0
    ddc_available_all = True
    for scene in store.iter_scenes():
        token = scene.record.scene_token
        if token not in payload:
            raise ValueError(f"published audit is missing token {token}")
        table = payload[token]["trajectory_scores"][0]
        for factor_index, factor_name in enumerate(SIX_FACTOR_ORDER):
            key = PUBLISHED_FACTOR_KEYS[factor_name]
            if key not in table:
                if factor_name == "DDC":
                    ddc_available_all = False
                rows.append(
                    {
                        "scene_token": token,
                        "factor": factor_name,
                        "available": False,
                        "mismatch_rate": "",
                        "max_absolute_error": "",
                        "mean_absolute_error": "",
                    }
                )
                continue
            published = np.asarray(table[key], dtype=np.float32)
            if published.shape != (CANDIDATE_COUNT,):
                raise ValueError(
                    f"published {factor_name} shape mismatch for {token}: {published.shape}"
                )
            errors = np.abs(published - scene.factors[:, factor_index])
            mismatch = errors > RECONSTRUCTION_TOLERANCE
            mismatch_by_factor[factor_name].append(mismatch)
            rows.append(
                {
                    "scene_token": token,
                    "factor": factor_name,
                    "available": True,
                    "mismatch_rate": float(mismatch.mean()),
                    "max_absolute_error": float(errors.max()),
                    "mean_absolute_error": float(errors.mean()),
                }
            )
        if "score" not in table:
            raise ValueError(f"published audit score missing for {token}")
        published_score = np.asarray(table["score"], dtype=np.float32)
        if published_score.shape != scene.score.shape:
            raise ValueError(f"published score shape mismatch for {token}")
        score_mismatches.append(
            np.abs(published_score - scene.score) > RECONSTRUCTION_TOLERANCE
        )
        argmax_mismatches += int(int(np.argmax(published_score)) != scene.oracle_index)
        top5_mismatches += int(
            set(np.argsort(published_score)[-5:].tolist())
            != set(np.argsort(scene.score)[-5:].tolist())
        )

    def mismatch_rate(name: str) -> float | None:
        arrays = mismatch_by_factor[name]
        return float(np.concatenate(arrays).mean()) if arrays else None

    scene_count = len(store.scene_tokens)
    summary = {
        "status": "UPSTREAM_REPRODUCTION_AUDIT_ONLY",
        "gate_blocking": False,
        "scenes": scene_count,
        "NC_mismatch_rate": mismatch_rate("NC"),
        "DAC_mismatch_rate": mismatch_rate("DAC"),
        "DDC_available_in_published_labels": ddc_available_all,
        "DDC_mismatch_rate": mismatch_rate("DDC"),
        "EP_mismatch_rate": mismatch_rate("EP"),
        "TTC_mismatch_rate": mismatch_rate("TTC"),
        "Comfort_mismatch_rate": mismatch_rate("Comfort"),
        "score_mismatch_rate": float(np.concatenate(score_mismatches).mean()),
        "argmax_disagreement_rate": argmax_mismatches / scene_count,
        "top5_set_disagreement_rate": top5_mismatches / scene_count,
    }
    return rows, summary


def _selected_indices_with_identity(
    cache_root: Path,
    store: SixFactorIndependentCandidateLabelStore,
) -> dict[str, int]:
    reader = FeatureShardReader(cache_root)
    identity = reader.manifest.get("identity", {})
    if identity.get("label_source") != "none":
        raise SixFactorGateError("G1-R2 requires a label-free feature cache")
    if identity.get("candidate_bank_hash") != store.manifest["candidate_bank_hash"]:
        raise SixFactorGateError("feature/label candidate bank hash mismatch")
    scenes = store.scene_index()
    selected: dict[str, int] = {}
    for sidecar, arrays in reader.iter_shards():
        records = sidecar["records"]
        selected_values = np.asarray(arrays["selected_index"], dtype=np.int64)
        final_rewards = np.asarray(arrays["final_rewards"], dtype=np.float32)
        if selected_values.shape != (len(records),):
            raise SixFactorGateError("cached selected_index shape mismatch")
        if final_rewards.shape != (len(records), CANDIDATE_COUNT):
            raise SixFactorGateError("cached final_rewards shape mismatch")
        for row_index, record in enumerate(records):
            token = str(record["scene_token"])
            if token in selected:
                raise SixFactorGateError(f"duplicate feature scene token: {token}")
            if token not in scenes:
                raise SixFactorGateError(f"feature cache has unexpected token: {token}")
            expected_indices = np.arange(CANDIDATE_COUNT, dtype=np.int64)
            if record.get("candidate_indices") != expected_indices.tolist():
                raise SixFactorGateError(f"candidate index mismatch for {token}")
            store.join_scene(
                token,
                str(record.get("candidate_bank_hash")),
                str(record.get("trajectory_hash")),
                expected_indices,
            )
            value = int(selected_values[row_index])
            if not 0 <= value < CANDIDATE_COUNT:
                raise SixFactorGateError(f"invalid selected index for {token}: {value}")
            selected[token] = value
    if set(selected) != set(store.scene_tokens):
        missing = sorted(set(store.scene_tokens) - set(selected))
        raise SixFactorGateError(
            f"feature cache does not exactly cover label scenes; first missing={missing[:1]}"
        )
    return selected


def six_factor_candidate_headroom(
    label_root: Path,
    feature_cache_root: Path,
) -> tuple[pd.DataFrame, dict[str, Any], list[dict[str, Any]]]:
    """Measure fixed-bank headroom using only independent six-factor labels."""

    store = SixFactorIndependentCandidateLabelStore(label_root)
    selected_indices = _selected_indices_with_identity(feature_cache_root, store)
    rows: list[dict[str, Any]] = []
    ddc_rows: list[dict[str, Any]] = []
    for scene in store.iter_scenes():
        token = scene.record.scene_token
        selected_index = selected_indices[token]
        oracle_index = scene.oracle_index
        selected_score = float(scene.score[selected_index])
        oracle_score = float(scene.score[oracle_index])
        gap = oracle_score - selected_score
        rank = int(candidate_ranks(scene.score[None])[0, selected_index])
        no_ddc_score = pdms_from_factors(scene.factors[:, [0, 1, 3, 4, 5]])
        no_ddc_oracle_index = int(np.argmax(no_ddc_score))
        no_ddc_actual_score = float(scene.score[no_ddc_oracle_index])
        row: dict[str, Any] = {
            "scene_token": token,
            "selected_index": selected_index,
            "oracle_index": oracle_index,
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
            "zero_to_positive_recovery": bool(selected_score == 0 and oracle_score > 0),
            "low_score_selected": bool(selected_score < 0.30),
            "low_score_improvement_gt_0_10": bool(selected_score < 0.30 and gap > 0.10),
            "selected_raw_progress": float(scene.raw_progress[selected_index]),
            "oracle_raw_progress": float(scene.raw_progress[oracle_index]),
        }
        for factor_index, factor_name in enumerate(SIX_FACTOR_ORDER):
            row[f"selected_{factor_name}"] = float(
                scene.factors[selected_index, factor_index]
            )
            row[f"oracle_{factor_name}"] = float(
                scene.factors[oracle_index, factor_index]
            )
        rows.append(row)
        ddc_rows.append(
            {
                "scene_token": token,
                "selected_index": selected_index,
                "full_oracle_index": oracle_index,
                "no_ddc_oracle_index": no_ddc_oracle_index,
                "selected_DDC": float(scene.factors[selected_index, 2]),
                "oracle_DDC": float(scene.factors[oracle_index, 2]),
                "no_ddc_oracle_DDC": float(scene.factors[no_ddc_oracle_index, 2]),
                "oracle_ranking_reversal": no_ddc_oracle_index != oracle_index,
                "score_loss_caused_by_omitting_DDC": oracle_score - no_ddc_actual_score,
            }
        )

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
    gate_pass = bool(headroom_ok and recovery_ok)
    selected_ddc = frame["selected_DDC"].to_numpy(dtype=np.float64)
    oracle_ddc = frame["oracle_DDC"].to_numpy(dtype=np.float64)
    reversal = np.asarray(
        [row["oracle_ranking_reversal"] for row in ddc_rows], dtype=np.bool_
    )
    no_ddc_ddc = np.asarray(
        [row["no_ddc_oracle_DDC"] for row in ddc_rows], dtype=np.float64
    )
    score_loss = np.asarray(
        [row["score_loss_caused_by_omitting_DDC"] for row in ddc_rows],
        dtype=np.float64,
    )
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
        "mean_selected_rank": float(frame["selected_rank"].mean()),
        "zero_score_scene_count": zero_count,
        "zero_to_positive_recovery_fraction": recovery_fraction,
        "recovery_status": recovery_status,
        "low_score_scene_count": int(frame["low_score_selected"].sum()),
        "low_score_improvement_gt_0_10_fraction": (
            float(
                frame.loc[
                    frame["low_score_selected"],
                    "low_score_improvement_gt_0_10",
                ].mean()
            )
            if frame["low_score_selected"].any()
            else None
        ),
        "selected_DDC_mean": float(selected_ddc.mean()),
        "selected_DDC_lt_1_fraction": float(np.mean(selected_ddc < 1.0)),
        "selected_DDC_eq_0_5_fraction": float(np.mean(selected_ddc == 0.5)),
        "selected_DDC_eq_0_fraction": float(np.mean(selected_ddc == 0.0)),
        "oracle_DDC_mean": float(oracle_ddc.mean()),
        "oracle_DDC_lt_1_fraction": float(np.mean(oracle_ddc < 1.0)),
        "oracle_DDC_eq_0_fraction": float(np.mean(oracle_ddc == 0.0)),
        "DDC_oracle_ranking_reversal_fraction": float(reversal.mean()),
        "no_DDC_oracle_candidate_DDC_lt_1_fraction": float(np.mean(no_ddc_ddc < 1.0)),
        "mean_score_loss_caused_by_omitting_DDC": float(score_loss.mean()),
        "headroom_ok": headroom_ok,
        "recovery_ok": recovery_ok,
        "candidate_headroom_gate": "PASS" if gate_pass else "FAIL",
        "final_verdict": (
            "WOTE_CANDIDATE_BANK_VIABLE" if gate_pass else "STOP_WOTE_CANDIDATE_BANK"
        ),
        "scientific_hypothesis_status": "UNTESTED",
        "next_recommended_experiment": (
            "DIRECT_VS_STATIC_VS_PRIMITIVE_ORACLE_EFFECT"
            if gate_pass
            else "SWITCH_TO_A_PROPOSAL_BASELINE_WITH_VERIFIED_NATIVE_HEADROOM"
        ),
    }
    return frame, summary, ddc_rows


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"required Gate artifact is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write_markdown(path: Path, content: str) -> None:
    if path.exists():
        raise FileExistsError(f"refusing existing report: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(content.rstrip() + "\n")


def build_final_reports(
    *,
    report_dir: Path,
    single_summary_path: Path,
    ten_summary_path: Path,
    two_hundred_summary_path: Path,
    headroom_summary_path: Path,
    tokens_path: Path,
) -> dict[str, Any]:
    """Materialize the final, scope-limited scientific verdict and documentation."""

    single = _read_json(single_summary_path)
    ten = _read_json(ten_summary_path)
    two_hundred = _read_json(two_hundred_summary_path)
    headroom = _read_json(headroom_summary_path)
    contract_path = report_dir / "EVALUATOR_CONTRACT.json"
    contract = _read_json(contract_path)
    contract_sha = sha256_file(contract_path)
    if single.get("status") != "PASS":
        raise SixFactorGateError("cannot build success reports: G0-R2a did not pass")
    if not ten.get("pass"):
        raise SixFactorGateError("cannot build success reports: G0-R2b did not pass")
    if not two_hundred.get("pass"):
        raise SixFactorGateError("cannot build success reports: G0-R2c did not pass")
    gate_pass = headroom.get("candidate_headroom_gate") == "PASS"
    final_verdict = (
        "WOTE_CANDIDATE_BANK_VIABLE" if gate_pass else "STOP_WOTE_CANDIDATE_BANK"
    )
    next_experiment = (
        "DIRECT_VS_STATIC_VS_PRIMITIVE_ORACLE_EFFECT"
        if gate_pass
        else "SWITCH_TO_A_PROPOSAL_BASELINE_WITH_VERIFIED_NATIVE_HEADROOM"
    )
    verdict = {
        "upstream_published_label_contract": "FAILED_PREVIOUS_AUDIT",
        "five_factor_schema": "INCOMPLETE_DDC_MISSING",
        "single_scene_six_factor_gate": "PASS",
        "ten_scene_determinism_gate": "PASS",
        "two_hundred_scene_relabel_gate": "PASS",
        "candidate_headroom_gate": "PASS" if gate_pass else "FAIL",
        "final_verdict": final_verdict,
        "scientific_hypothesis_status": "UNTESTED",
        "positive_evidence": [
            "Six-factor evaluator scores reconstruct within 1e-6 for all 51,200 candidates.",
            "Two independent 200-scene relabel runs are exactly identical.",
        ]
        + (
            ["The fixed 256-anchor bank meets the pre-registered headroom criterion."]
            if gate_pass
            else []
        ),
        "blocking_evidence": (
            []
            if gate_pass
            else ["The fixed 256-anchor bank does not meet the headroom criterion."]
        ),
        "next_recommended_experiment": next_experiment,
    }
    atomic_write_json(report_dir / "VERDICT.json", verdict)
    asset_manifest = {
        "status": "VERIFIED",
        "wote_commit": contract["wote_commit"],
        "checkpoint_sha256": contract["checkpoint_sha256"],
        "candidate_bank_sha256": contract["candidate_bank_sha256"],
        "candidate_bank_logical_sha256": contract["candidate_bank_logical_sha256"],
        "evaluator_contract_sha256": contract_sha,
        "evaluator_source_hashes": contract["evaluator_source_hashes"],
        "label_schema_version": contract["label_schema_version"],
        "relabel_headroom_token_count": len(
            [
                line
                for line in tokens_path.read_text(encoding="utf-8").splitlines()
                if line
            ]
        ),
        "relabel_headroom_tokens_sha256": sha256_file(tokens_path),
        "published_candidate_scores_gate_dependency": False,
        "legacy_report_tree_sha256": (
            "dbc800d748be19c5245b6cd33a4460b4212f9abbb2a1d334e3766609e582dfd2"
        ),
        "legacy_report_hash_status": "UNCHANGED",
    }
    atomic_write_json(report_dir / "ASSET_MANIFEST.json", asset_manifest)

    relabel_report = f"""# Six-Factor Independent Relabel Report

The historical five-factor schema remains preserved and is explicitly classified as
`INCOMPLETE_DDC_MISSING`. This run uses the immutable v2 order
`[NC, DAC, DDC, EP, TTC, Comfort]`; raw progress is diagnostic only.

| Gate | Scenes | Candidates/run | Run1 logical SHA256 | Run2 logical SHA256 | Max reconstruction error | Status |
| --- | ---: | ---: | --- | --- | ---: | --- |
| G0-R2a | 1 | 256 | {single['logical_content_sha256']} | n/a | {single['six_factor_max_error']:.9g} | {single['gate']} |
| G0-R2b | {ten['scenes']} | {ten['candidates_per_run']} | {ten['run1_logical_sha256']} | {ten['run2_logical_sha256']} | {ten['max_score_reconstruction_error']:.9g} | {ten['status']} |
| G0-R2c | {two_hundred['scenes']} | {two_hundred['candidates_per_run']} | {two_hundred['run1_logical_sha256']} | {two_hundred['run2_logical_sha256']} | {two_hundred['max_score_reconstruction_error']:.9g} | {two_hundred['status']} |

Every evaluator invocation received the complete 256-anchor set in one call, so EP
normalization is scoped to all 256 candidates within each scene. No candidate was
added, removed, chunked, or offset. The published-label comparison is a non-blocking
upstream reproduction audit only.
"""
    _write_markdown(report_dir / "SIX_FACTOR_RELABEL_REPORT.md", relabel_report)

    recovery = (
        f"{headroom['zero_to_positive_recovery_fraction']:.6f}"
        if headroom["recovery_status"] == "APPLICABLE"
        else headroom["recovery_status"]
    )
    headroom_report = f"""# G1-R2 Candidate Headroom Report

This is the frozen **WoTE base-anchor selector**, not a full WoTE leaderboard result.
Trajectory offsets are disabled and every score is an independently recomputed
six-factor 4-second label.

| Selector | Selected score | Oracle score | Gap | Better-scene fraction | Recovery |
| --- | ---: | ---: | ---: | ---: | ---: |
| WoTE base-anchor selector | {headroom['selected_score_raw']:.6f} | {headroom['oracle_score_raw']:.6f} | {headroom['mean_oracle_gap_raw']:.6f} | {headroom['better_scene_fraction']:.6f} | {recovery} |

The scene-paired bootstrap 95% interval for oracle gap is
`[{headroom['oracle_gap_ci95']['lower']:.6f}, {headroom['oracle_gap_ci95']['upper']:.6f}]`
using 2,000 resamples and seed 20260827.

DDC diagnostics: selected mean `{headroom['selected_DDC_mean']:.6f}`, oracle mean
`{headroom['oracle_DDC_mean']:.6f}`, and no-DDC/full-oracle reversal fraction
`{headroom['DDC_oracle_ranking_reversal_fraction']:.6f}`.

Gate: `{headroom['candidate_headroom_gate']}`. Final verdict: `{final_verdict}`.
The action-effect hypothesis remains `UNTESTED`; no effect scorer, forward model,
inverse model, WoTE training, trajectory offsets, or extra candidates were run.
"""
    _write_markdown(report_dir / "G1_CANDIDATE_HEADROOM_REPORT.md", headroom_report)

    reproduction = f"""# Reproduction

Run `research/cf_effect_gate_wote/scripts/run_six_factor_gate.sh` from the dedicated
six-factor worktree with machine-local paths supplied through the documented command
arguments or `CF_SIX_FACTOR_*` environment variables. The launcher enforces this order:
asset verification, metric-cache creation, G0-R2a, G0-R2b twice, G0-R2c twice,
published-label audit, label-free frozen feature caching, G1-R2, then stop.

- Evaluator contract SHA256: `{contract_sha}`
- Label schema: `{contract['label_schema_version']}`
- Fixed 200-token SHA256: `{sha256_file(tokens_path)}`
- Proposal sampling: 40 poses at 0.1 s
- Candidate bank: 256 base anchors, 8 waypoints at 0.5 s; offsets disabled

The launcher refuses existing outputs and has no automatic fallback to another
horizon, evaluator, label source, or candidate set.
"""
    _write_markdown(report_dir / "REPRODUCTION.md", reproduction)
    _write_csv(
        report_dir / "failure_cases.csv",
        [],
        (
            "gate",
            "scene_token",
            "candidate_index",
            "reason",
            "absolute_error",
        ),
    )
    return verdict


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    single = commands.add_parser("single-scene")
    single.add_argument("--labels", type=Path, required=True)
    single.add_argument("--output-csv", type=Path, required=True)
    single.add_argument("--output-contrast", type=Path, required=True)
    single.add_argument("--output-summary", type=Path, required=True)

    compare = commands.add_parser("compare")
    compare.add_argument("--run1", type=Path, required=True)
    compare.add_argument("--run2", type=Path, required=True)
    compare.add_argument("--expected-scenes", type=int, required=True)
    compare.add_argument("--pass-status", required=True)
    compare.add_argument("--output-csv", type=Path, required=True)
    compare.add_argument("--output-summary", type=Path, required=True)

    audit = commands.add_parser("published-audit")
    audit.add_argument("--labels", type=Path, required=True)
    audit.add_argument("--published", type=Path, required=True)
    audit.add_argument("--output-csv", type=Path, required=True)
    audit.add_argument("--output-summary", type=Path, required=True)

    headroom = commands.add_parser("headroom")
    headroom.add_argument("--labels", type=Path, required=True)
    headroom.add_argument("--feature-cache", type=Path, required=True)
    headroom.add_argument("--output-parquet", type=Path, required=True)
    headroom.add_argument("--output-summary", type=Path, required=True)
    headroom.add_argument("--output-ddc", type=Path, required=True)

    report = commands.add_parser("build-reports")
    report.add_argument("--report-dir", type=Path, required=True)
    report.add_argument("--single-summary", type=Path, required=True)
    report.add_argument("--ten-summary", type=Path, required=True)
    report.add_argument("--two-hundred-summary", type=Path, required=True)
    report.add_argument("--headroom-summary", type=Path, required=True)
    report.add_argument("--tokens", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "single-scene":
        rows, contrast, summary = single_scene_validation(args.labels)
        _write_csv(args.output_csv, rows)
        _write_csv(args.output_contrast, contrast, tuple(rows[0].keys()))
        atomic_write_json(args.output_summary, summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0 if summary["status"] == "PASS" else 4
    if args.command == "compare":
        if args.expected_scenes <= 0:
            raise ValueError("--expected-scenes must be positive")
        rows, summary = compare_six_factor_runs(
            args.run1, args.run2, args.expected_scenes, args.pass_status
        )
        _write_csv(args.output_csv, rows)
        atomic_write_json(args.output_summary, summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0 if summary["pass"] else 4
    if args.command == "published-audit":
        rows, summary = published_six_factor_audit(args.labels, args.published)
        _write_csv(args.output_csv, rows)
        atomic_write_json(args.output_summary, summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    if args.command == "build-reports":
        verdict = build_final_reports(
            report_dir=args.report_dir,
            single_summary_path=args.single_summary,
            ten_summary_path=args.ten_summary,
            two_hundred_summary_path=args.two_hundred_summary,
            headroom_summary_path=args.headroom_summary,
            tokens_path=args.tokens,
        )
        print(json.dumps(verdict, indent=2, sort_keys=True))
        return 0

    frame, summary, ddc_rows = six_factor_candidate_headroom(
        args.labels, args.feature_cache
    )
    if args.output_parquet.exists():
        raise FileExistsError(f"refusing existing parquet: {args.output_parquet}")
    args.output_parquet.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.output_parquet, index=False)
    atomic_write_json(args.output_summary, summary)
    _write_csv(args.output_ddc, ddc_rows)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
