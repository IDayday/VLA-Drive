"""Diagnose temporal-consequence scorer failures on the locked Navtest bank.

This is an offline reporting tool.  Model inference consumes only the frozen
current-observation cache; the all-candidate PDM matrix is joined afterwards to
measure calibration and characterize selection errors.  In particular, no
PDM value is passed to :class:`TemporalConsequenceRanker`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

# Deterministic CUDA must be configured before importing torch.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

from local_stage2.public_base_residual_scorer import FACTOR_KEYS
from local_stage2.temporal_consequence_scorer import (
    TemporalConsequenceConfig,
    TemporalConsequenceRanker,
    TemporalConsequenceScorerAgent,
)


EXPECTED_SCENES = 12_146
EXPECTED_LOGS = 136
EXPECTED_CANDIDATES = 64
MATRIX_FACTOR_KEYS = (
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "ego_progress",
    "time_to_collision_within_bound",
    "comfort",
    "driving_direction_compliance",
    "score",
)
SAFETY_FACTOR_KEYS = (
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "time_to_collision_within_bound",
)


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while block := file.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _parse_artifacts(values: Sequence[str]) -> List[Tuple[str, Path]]:
    result: List[Tuple[str, Path]] = []
    names = set()
    for value in values:
        if "=" not in value:
            raise ValueError(f"Artifact must be NAME=PATH, got {value!r}")
        name, raw_path = value.split("=", 1)
        path = Path(raw_path)
        if not name or name in names:
            raise ValueError(f"Empty or duplicate artifact name: {name!r}")
        if not path.is_file():
            raise FileNotFoundError(path)
        names.add(name)
        result.append((name, path))
    if not result:
        raise ValueError("At least one --artifact NAME=PATH is required")
    return result


def _batches(values: Sequence[str], size: int) -> Iterable[Sequence[str]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _stack(
    cache: Mapping[str, Mapping[str, np.ndarray]],
    tokens: Sequence[str],
    key: str,
    device: torch.device,
) -> torch.Tensor:
    value = np.stack(
        [np.asarray(cache[token][key], dtype=np.float32) for token in tokens]
    )
    return torch.from_numpy(value).to(device)


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Return one-based average ranks with exact handling of score ties."""

    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    order = np.argsort(flat, kind="stable")
    ranks = np.empty(len(flat), dtype=np.float64)
    start = 0
    while start < len(flat):
        end = start + 1
        while end < len(flat) and flat[order[end]] == flat[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + 1 + end)
        start = end
    return ranks


def binary_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=bool).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    positive = int(labels.sum())
    negative = len(labels) - positive
    if positive == 0 or negative == 0:
        return float("nan")
    ranks = _average_ranks(scores)
    rank_sum = float(ranks[labels].sum())
    return (rank_sum - positive * (positive + 1) / 2.0) / (positive * negative)


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=bool).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    positive = int(labels.sum())
    if positive == 0:
        return float("nan")
    order = np.argsort(-scores, kind="stable")
    sorted_labels = labels[order]
    precision = np.cumsum(sorted_labels) / np.arange(1, len(labels) + 1)
    return float(precision[sorted_labels].sum() / positive)


def expected_calibration_error(
    labels: np.ndarray,
    probabilities: np.ndarray,
    bins: int = 20,
) -> float:
    labels = np.asarray(labels, dtype=np.float64).reshape(-1)
    probabilities = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    edges = np.linspace(0.0, 1.0, bins + 1)
    # Include probability exactly equal to one in the final bin.
    assignment = np.minimum(np.digitize(probabilities, edges[1:-1]), bins - 1)
    error = 0.0
    for index in range(bins):
        mask = assignment == index
        if mask.any():
            error += float(mask.mean()) * abs(
                float(probabilities[mask].mean()) - float(labels[mask].mean())
            )
    return error


def binary_calibration(labels: np.ndarray, probabilities: np.ndarray) -> Dict[str, float]:
    labels = np.asarray(labels, dtype=bool).reshape(-1)
    probabilities = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if labels.shape != probabilities.shape:
        raise ValueError("labels and probabilities must have equal flattened shape")
    return {
        "count": int(len(labels)),
        "target_rate": float(labels.mean()),
        "prediction_mean": float(probabilities.mean()),
        "auroc": binary_auroc(labels, probabilities),
        "average_precision": average_precision(labels, probabilities),
        "brier": float(np.square(probabilities - labels).mean()),
        "ece_20bin": expected_calibration_error(labels, probabilities, bins=20),
    }


def _load_feature_cache(path: Path) -> Dict[str, Dict[str, np.ndarray]]:
    with path.open("rb") as file:
        cache = pickle.load(file)
    required = {
        "proposals",
        "predicted_scores",
        "base_factor_logits",
        "candidate_features",
        "scene_features",
        "ego_features",
    }
    if not isinstance(cache, dict) or not cache:
        raise ValueError(f"Malformed feature cache: {path}")
    for token, item in cache.items():
        missing = required - set(item)
        if missing:
            raise ValueError(f"Feature token {token} lacks {sorted(missing)}")
    return cache


def _load_matrix(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        matrix = {key: archive[key] for key in archive.files}
    required = {
        "tokens",
        "log_names",
        "candidate_scores",
        "predicted_scores",
        "candidate_factors",
        "candidate_factor_names",
    }
    missing = required - set(matrix)
    if missing:
        raise ValueError(f"Candidate matrix lacks {sorted(missing)}")
    factor_names = tuple(matrix["candidate_factor_names"].astype(str))
    if factor_names != MATRIX_FACTOR_KEYS:
        raise ValueError(f"Unexpected candidate factor order: {factor_names}")
    return matrix


@torch.inference_mode()
def collect_temporal_outputs(
    artifact_path: Path,
    cache: Mapping[str, Mapping[str, np.ndarray]],
    tokens: Sequence[str],
    *,
    device: torch.device,
    batch_size: int,
) -> Tuple[Dict[str, np.ndarray], Mapping[str, object]]:
    payload = torch.load(artifact_path, map_location="cpu")
    if payload.get("artifact_type") != TemporalConsequenceScorerAgent.ARTIFACT_TYPE:
        raise ValueError(f"Not a temporal consequence artifact: {artifact_path}")
    model = TemporalConsequenceRanker(
        TemporalConsequenceConfig(**dict(payload["model_config"]))
    )
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device).eval()
    keys = (
        "selection_scores",
        "refined_scores",
        "residual",
        "refined_factor_logits",
        "risk_logits",
        "area_logits",
        "predicted_safety",
        "top_k_mask",
        "eligible_mask",
    )
    values: Dict[str, List[np.ndarray]] = {key: [] for key in keys}
    for batch_tokens in _batches(tokens, batch_size):
        output = model(
            _stack(cache, batch_tokens, "candidate_features", device),
            _stack(cache, batch_tokens, "proposals", device),
            _stack(cache, batch_tokens, "base_factor_logits", device),
            _stack(cache, batch_tokens, "predicted_scores", device),
            _stack(cache, batch_tokens, "scene_features", device),
            _stack(cache, batch_tokens, "ego_features", device),
        )
        for key in keys:
            value = output[key].detach().cpu()
            if value.dtype == torch.bool:
                values[key].append(value.numpy())
            else:
                values[key].append(value.float().numpy())
    return {key: np.concatenate(parts) for key, parts in values.items()}, payload


def _factor_index() -> Dict[str, int]:
    return {name: index for index, name in enumerate(MATRIX_FACTOR_KEYS)}


def _scorer_factor_diagnostics(
    factors: np.ndarray,
    base_logits: np.ndarray,
    refined_logits: np.ndarray,
    top_mask: np.ndarray,
) -> Dict[str, object]:
    matrix_index = _factor_index()
    output: Dict[str, object] = {}
    for scorer_index, key in enumerate(FACTOR_KEYS):
        target = factors[..., matrix_index[key]][top_mask]
        base_probability = 1.0 / (1.0 + np.exp(-base_logits[..., scorer_index][top_mask]))
        refined_probability = 1.0 / (
            1.0 + np.exp(-refined_logits[..., scorer_index][top_mask])
        )
        if key == "ego_progress":
            output[key] = {
                "target_mean": float(target.mean()),
                "base_mae": float(np.abs(base_probability - target).mean()),
                "refined_mae": float(np.abs(refined_probability - target).mean()),
                "mae_delta_refined_minus_base": float(
                    np.abs(refined_probability - target).mean()
                    - np.abs(base_probability - target).mean()
                ),
            }
        else:
            binary_target = target >= 1.0 - 1e-6
            base = binary_calibration(binary_target, base_probability)
            refined = binary_calibration(binary_target, refined_probability)
            output[key] = {
                "base": base,
                "refined": refined,
                "brier_delta_refined_minus_base": refined["brier"] - base["brier"],
            }
    return output


def _selection_diagnostics(
    tokens: Sequence[str],
    log_names: np.ndarray,
    candidate_scores: np.ndarray,
    factors: np.ndarray,
    base_scores: np.ndarray,
    outputs: Mapping[str, np.ndarray],
) -> Tuple[Dict[str, object], pd.DataFrame, pd.DataFrame]:
    rows = np.arange(len(tokens))
    base_index = base_scores.argmax(axis=1)
    selected_index = outputs["selection_scores"].argmax(axis=1)
    base_pdms = candidate_scores[rows, base_index]
    selected_pdms = candidate_scores[rows, selected_index]
    delta = selected_pdms - base_pdms
    switched = selected_index != base_index
    matrix_index = _factor_index()
    selected_factors = factors[rows, selected_index]
    base_factors = factors[rows, base_index]
    factor_delta = selected_factors - base_factors

    safety_columns = np.asarray([matrix_index[key] for key in SAFETY_FACTOR_KEYS])
    safety_worse = (factor_delta[:, safety_columns] < -1e-6).any(axis=1)
    base_strict_safe = (
        base_factors[:, safety_columns] >= 1.0 - 1e-6
    ).all(axis=1)
    selected_strict_safe = (
        selected_factors[:, safety_columns] >= 1.0 - 1e-6
    ).all(axis=1)
    safe_to_unsafe = switched & base_strict_safe & ~selected_strict_safe
    progress_up = factor_delta[:, matrix_index["ego_progress"]] > 1e-6
    progress_up_safety_down = switched & progress_up & safety_worse

    predicted_event = 1.0 - outputs["predicted_safety"]
    selected_predicted_event = predicted_event[rows, selected_index]
    base_predicted_event = predicted_event[rows, base_index]
    predicted_no_worse = (
        selected_predicted_event <= base_predicted_event + 1e-8
    ).all(axis=1)
    false_safety_approval = switched & safety_worse & predicted_no_worse

    frame = pd.DataFrame(
        {
            "token": np.asarray(tokens),
            "log_name": log_names.astype(str),
            "base_index": base_index,
            "selected_index": selected_index,
            "switched": switched,
            "base_pdms": base_pdms,
            "selected_pdms": selected_pdms,
            "pdms_delta": delta,
            "base_strict_safe": base_strict_safe,
            "selected_strict_safe": selected_strict_safe,
            "safety_worse": safety_worse,
            "safe_to_unsafe": safe_to_unsafe,
            "progress_up": progress_up,
            "progress_up_safety_down": progress_up_safety_down,
            "predicted_no_worse": predicted_no_worse,
            "false_safety_approval": false_safety_approval,
            "pred_collision_event_base": base_predicted_event[:, 0],
            "pred_collision_event_selected": selected_predicted_event[:, 0],
            "pred_ttc_event_base": base_predicted_event[:, 1],
            "pred_ttc_event_selected": selected_predicted_event[:, 1],
        }
    )
    for index, key in enumerate(MATRIX_FACTOR_KEYS):
        frame[f"base_{key}"] = base_factors[:, index]
        frame[f"selected_{key}"] = selected_factors[:, index]
        frame[f"delta_{key}"] = factor_delta[:, index]

    per_log = (
        frame.groupby("log_name", sort=True)
        .agg(
            scene_count=("token", "size"),
            pdms_delta=("pdms_delta", "mean"),
            switch_rate=("switched", "mean"),
            loss_rate=("pdms_delta", lambda value: float((value < -1e-8).mean())),
            safe_to_unsafe_rate=("safe_to_unsafe", "mean"),
            false_safety_approval_rate=("false_safety_approval", "mean"),
        )
        .reset_index()
        .sort_values("pdms_delta")
    )

    switch_count = int(switched.sum())
    selection = {
        "selected_pdms": float(selected_pdms.mean()),
        "base_pdms": float(base_pdms.mean()),
        "pdms_delta": float(delta.mean()),
        "switch_count": switch_count,
        "switch_rate": float(switched.mean()),
        "win_count": int((delta > 1e-8).sum()),
        "loss_count": int((delta < -1e-8).sum()),
        "tie_count": int((np.abs(delta) <= 1e-8).sum()),
        "safe_to_unsafe_count": int(safe_to_unsafe.sum()),
        "safe_to_unsafe_fraction_of_switches": float(
            safe_to_unsafe.sum() / max(switch_count, 1)
        ),
        "progress_up_safety_down_count": int(progress_up_safety_down.sum()),
        "progress_up_safety_down_fraction_of_switches": float(
            progress_up_safety_down.sum() / max(switch_count, 1)
        ),
        "false_safety_approval_count": int(false_safety_approval.sum()),
        "false_safety_approval_fraction_of_safety_worse_switches": float(
            false_safety_approval.sum() / max(int((switched & safety_worse).sum()), 1)
        ),
        "switched_factor_delta": {
            key: float(factor_delta[switched, index].mean()) if switch_count else 0.0
            for index, key in enumerate(MATRIX_FACTOR_KEYS)
        },
        "worst_10_logs": per_log.head(10).to_dict(orient="records"),
        "best_10_logs": per_log.tail(10).iloc[::-1].to_dict(orient="records"),
    }
    return selection, frame, per_log


def analyze_artifact(
    name: str,
    artifact_path: Path,
    cache: Mapping[str, Mapping[str, np.ndarray]],
    tokens: Sequence[str],
    matrix: Mapping[str, np.ndarray],
    matrix_rows: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
    output_dir: Path,
) -> Dict[str, object]:
    outputs, payload = collect_temporal_outputs(
        artifact_path,
        cache,
        tokens,
        device=device,
        batch_size=batch_size,
    )
    candidate_scores = matrix["candidate_scores"][matrix_rows].astype(np.float64)
    factors = matrix["candidate_factors"][matrix_rows].astype(np.float64)
    base_scores = matrix["predicted_scores"][matrix_rows].astype(np.float64)
    cache_base_logits = np.stack(
        [np.asarray(cache[token]["base_factor_logits"], dtype=np.float64) for token in tokens]
    )
    top_mask = outputs["top_k_mask"].astype(bool)
    collision_event = factors[..., _factor_index()["no_at_fault_collisions"]] < 1.0 - 1e-6
    ttc_event = (
        factors[..., _factor_index()["time_to_collision_within_bound"]] < 1.0 - 1e-6
    )
    final_risk_probability = 1.0 - outputs["predicted_safety"]
    risk = {
        "target_semantics": (
            "Navtest final-factor proxy: collision iff NOC<1; TTC event iff TTC<1. "
            "Validation metadata uses exact event-by-horizon labels."
        ),
        "collision_top16": binary_calibration(
            collision_event[top_mask], final_risk_probability[..., 0][top_mask]
        ),
        "ttc_top16": binary_calibration(
            ttc_event[top_mask], final_risk_probability[..., 1][top_mask]
        ),
    }
    area_probability = outputs["area_logits"]
    area_probability = 1.0 / (1.0 + np.exp(-area_probability))
    # Heads predict per-horizon occupancy.  Max is the deployable estimate that
    # a violation appears at any sampled horizon.
    area_any = area_probability.max(axis=2)
    area = {
        "target_semantics": (
            "Approximate final-factor proxy: non-drivable iff DAC<1; oncoming "
            "iff DDC<1. These are not exact per-horizon area labels."
        ),
        "non_drivable_top16": binary_calibration(
            (
                factors[..., _factor_index()["drivable_area_compliance"]]
                < 1.0 - 1e-6
            )[top_mask],
            area_any[..., 0][top_mask],
        ),
        "oncoming_top16": binary_calibration(
            (
                factors[..., _factor_index()["driving_direction_compliance"]]
                < 1.0 - 1e-6
            )[top_mask],
            area_any[..., 1][top_mask],
        ),
    }
    selection, per_scene, per_log = _selection_diagnostics(
        tokens,
        matrix["log_names"][matrix_rows],
        candidate_scores,
        factors,
        base_scores,
        outputs,
    )
    validation = dict(payload.get("metadata", {}).get("validation", {}))
    method_dir = output_dir / name
    method_dir.mkdir(parents=True, exist_ok=True)
    per_scene.to_csv(method_dir / "per_scene_failure_diagnostics.csv", index=False)
    per_log.to_csv(method_dir / "per_log_failure_diagnostics.csv", index=False)
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": name,
        "artifact_path": str(artifact_path.resolve()),
        "artifact_sha256": _sha256(artifact_path),
        "artifact_type": payload.get("artifact_type"),
        "model_config": payload.get("model_config"),
        "scene_count": len(tokens),
        "log_count": int(len(set(matrix["log_names"][matrix_rows].astype(str)))),
        "candidate_count": int(candidate_scores.shape[1]),
        "top_k_count": int(top_mask.sum(axis=1).min()),
        "future_inputs_used": False,
        "official_pdm_joined_after_model_inference": True,
        "validation_reference": {
            "selected_pdms_delta": validation.get("selected_pdms_delta"),
            "selection_switch_rate": validation.get("selection_switch_rate"),
            "risk_prediction": validation.get("risk_prediction"),
        },
        "navtest_risk_calibration": risk,
        "navtest_area_calibration": area,
        "navtest_factor_head": _scorer_factor_diagnostics(
            factors,
            cache_base_logits,
            outputs["refined_factor_logits"],
            top_mask,
        ),
        "navtest_selection": selection,
    }
    _atomic_json(method_dir / "failure_diagnostics.json", summary)
    return summary


def _format_metric(value: object, digits: int = 6) -> str:
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{number:.{digits}f}"


def write_report(path: Path, summaries: Sequence[Mapping[str, object]]) -> None:
    lines = [
        "# Temporal-consequence Navtest failure diagnosis",
        "",
        "The temporal models consume only current-observation features and frozen "
        "proposals. Official all-candidate factors are joined after inference for "
        "this diagnostic and are never scorer inputs.",
        "",
        "| Method | Val delta | Navtest delta | Switch | Collision AUROC/Brier | TTC AUROC/Brier | Safe→unsafe | False safety approval |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for summary in summaries:
        val = summary["validation_reference"]
        test = summary["navtest_selection"]
        risk = summary["navtest_risk_calibration"]
        collision = risk["collision_top16"]
        ttc = risk["ttc_top16"]
        lines.append(
            "| {name} | {val} | {test_delta} | {switch:.2%} | {c_auc}/{c_brier} | "
            "{t_auc}/{t_brier} | {unsafe} | {false} |".format(
                name=summary["method"],
                val=_format_metric(val.get("selected_pdms_delta")),
                test_delta=_format_metric(test["pdms_delta"]),
                switch=float(test["switch_rate"]),
                c_auc=_format_metric(collision["auroc"], 4),
                c_brier=_format_metric(collision["brier"], 4),
                t_auc=_format_metric(ttc["auroc"], 4),
                t_brier=_format_metric(ttc["brier"], 4),
                unsafe=test["safe_to_unsafe_count"],
                false=test["false_safety_approval_count"],
            )
        )
    lines.extend(
        [
            "",
            "`Safe→unsafe` counts switches from a Base candidate with NOC=DAC=TTC=1 "
            "to a candidate violating at least one of those factors. `False safety "
            "approval` counts safety-worse switches for which predicted collision/TTC "
            "risk did not increase relative to the Base choice.",
            "",
            "Collision/TTC Navtest labels above are final-factor event proxies because "
            "the locked Navtest matrix intentionally stores no future annotations. "
            "Training/validation event labels are exact per-horizon PDM events.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text("\n".join(lines) + "\n")
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--feature-manifest", type=Path, required=True)
    parser.add_argument("--candidate-matrix", type=Path, required=True)
    parser.add_argument("--candidate-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--artifact", action="append", default=[])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifacts = _parse_artifacts(args.artifact)
    feature_manifest = json.loads(args.feature_manifest.read_text())
    candidate_summary = json.loads(args.candidate_summary.read_text())
    if not feature_manifest.get("inference_inputs_only"):
        raise RuntimeError("Feature cache is not marked inference-input-only")
    if not feature_manifest.get("reference_parity", {}).get("passes_1e_8"):
        raise RuntimeError("Feature cache failed released-reference parity")
    if (
        candidate_summary.get("proposal_predictions_sha256")
        != feature_manifest.get("reference_predictions_sha256")
    ):
        raise RuntimeError("Candidate matrix does not trace to the locked proposal bank")

    cache = _load_feature_cache(args.feature_cache)
    matrix = _load_matrix(args.candidate_matrix)
    tokens = sorted(cache)
    matrix_index = {
        token: index for index, token in enumerate(matrix["tokens"].astype(str))
    }
    if set(tokens) != set(matrix_index):
        raise RuntimeError("Feature cache and candidate matrix token sets differ")
    matrix_rows = np.asarray([matrix_index[token] for token in tokens])
    if len(tokens) != EXPECTED_SCENES:
        raise RuntimeError(f"Expected {EXPECTED_SCENES} scenes, found {len(tokens)}")
    if len(set(matrix["log_names"][matrix_rows].astype(str))) != EXPECTED_LOGS:
        raise RuntimeError("Unexpected Navtest log count")
    if matrix["candidate_scores"].shape[1] != EXPECTED_CANDIDATES:
        raise RuntimeError("Unexpected candidate count")
    cache_base = np.stack(
        [np.asarray(cache[token]["predicted_scores"], dtype=np.float32) for token in tokens]
    )
    matrix_base = matrix["predicted_scores"][matrix_rows].astype(np.float32)
    base_max_abs = float(np.abs(cache_base.astype(np.float64) - matrix_base).max())
    if base_max_abs > 1e-8:
        raise RuntimeError(f"Base score cache/matrix mismatch: {base_max_abs}")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.use_deterministic_algorithms(True, warn_only=False)
    summaries = [
        analyze_artifact(
            name,
            path,
            cache,
            tokens,
            matrix,
            matrix_rows,
            device=device,
            batch_size=args.batch_size,
            output_dir=args.output_dir,
        )
        for name, path in artifacts
    ]
    campaign = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "feature_cache": str(args.feature_cache.resolve()),
        "feature_cache_sha256": _sha256(args.feature_cache),
        "candidate_matrix": str(args.candidate_matrix.resolve()),
        "candidate_matrix_sha256": _sha256(args.candidate_matrix),
        "base_score_max_abs": base_max_abs,
        "methods": summaries,
    }
    _atomic_json(args.output_dir / "failure_diagnostics_campaign.json", campaign)
    write_report(args.output_dir / "TEMPORAL_NAVTEST_FAILURE_DIAGNOSIS.md", summaries)
    print(json.dumps(campaign, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
