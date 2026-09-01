#!/usr/bin/env python3
"""Compare two complete 64-proposal Navtest banks on identical scene tokens."""

from __future__ import annotations

import argparse
import json
import pickle
import re
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

_SEGMENT_SUFFIX = re.compile(r"_\d{5}_\d{5}$")


def _physical_log_name(value: str) -> str:
    return _SEGMENT_SUFFIX.sub("", str(value))


def _log_bootstrap_ci(
    values: np.ndarray,
    log_names: Sequence[str],
    seed: int,
    replicates: int,
) -> Tuple[float, float]:
    """Bootstrap physical logs while retaining the scene-weighted estimand."""

    if replicates <= 0:
        return float("nan"), float("nan")
    grouped: Dict[str, List[float]] = {}
    for value, log_name in zip(values, log_names):
        grouped.setdefault(str(log_name), []).append(float(value))
    ordered_logs = sorted(grouped)
    if not ordered_logs:
        return float("nan"), float("nan")
    log_sums = np.asarray([np.sum(grouped[name]) for name in ordered_logs])
    log_counts = np.asarray([len(grouped[name]) for name in ordered_logs])
    rng = np.random.default_rng(seed)
    sampled = rng.integers(
        0,
        len(ordered_logs),
        size=(replicates, len(ordered_logs)),
    )
    samples = log_sums[sampled].sum(axis=1) / log_counts[sampled].sum(axis=1)
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def _load_matrix(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        required = {
            "tokens",
            "log_names",
            "candidate_scores",
            "predicted_scores",
            "candidate_factors",
            "candidate_factor_names",
        }
        if not required.issubset(archive.files):
            raise RuntimeError(f"Candidate matrix is missing fields: {path}")
        return {key: archive[key] for key in archive.files}


def _align_matrix(
    matrix: Mapping[str, np.ndarray], ordered_tokens: Sequence[str]
) -> Dict[str, np.ndarray]:
    tokens = matrix["tokens"].astype(str)
    if len(tokens) != len(set(tokens.tolist())):
        raise RuntimeError("Candidate matrix contains duplicate tokens")
    index = {token: row for row, token in enumerate(tokens.tolist())}
    if set(index) != set(ordered_tokens):
        raise RuntimeError(
            "Candidate matrices cover different tokens: "
            f"missing={len(set(ordered_tokens) - set(index))}, "
            f"extra={len(set(index) - set(ordered_tokens))}"
        )
    rows = np.asarray([index[token] for token in ordered_tokens], dtype=np.int64)
    return {
        key: value[rows] if value.shape[:1] == tokens.shape else value
        for key, value in matrix.items()
    }


def _load_proposals(path: Path) -> Dict[str, np.ndarray]:
    with path.open("rb") as stream:
        payload = pickle.load(stream)
    if not isinstance(payload, dict) or not payload:
        raise RuntimeError(f"Malformed proposal cache: {path}")
    proposals: Dict[str, np.ndarray] = {}
    for token, row in payload.items():
        value = np.asarray(row["proposals"], dtype=np.float32)
        if value.shape != (64, 8, 3) or not np.isfinite(value).all():
            raise RuntimeError(f"Malformed proposals for {token}: {value.shape}")
        proposals[str(token)] = value
    return proposals


def _bank_rows(
    matrix: Mapping[str, np.ndarray], prefix: str
) -> Dict[str, np.ndarray]:
    scores = matrix["candidate_scores"].astype(np.float64)
    predictions = matrix["predicted_scores"].astype(np.float64)
    factors = matrix["candidate_factors"].astype(np.float64)
    selected = predictions.argmax(axis=1)
    oracle = scores.argmax(axis=1)
    rows = np.arange(len(scores))
    top_count = min(5, scores.shape[1])
    top5 = np.partition(scores, -top_count, axis=1)[:, -top_count:].mean(axis=1)
    output = {
        f"{prefix}_selected_pdms": scores[rows, selected],
        f"{prefix}_oracle_pdms": scores[rows, oracle],
        f"{prefix}_regret": scores[rows, oracle] - scores[rows, selected],
        f"{prefix}_mean_candidate_pdms": scores.mean(axis=1),
        f"{prefix}_median_candidate_pdms": np.median(scores, axis=1),
        f"{prefix}_candidate_p10": np.quantile(scores, 0.10, axis=1),
        f"{prefix}_candidate_p90": np.quantile(scores, 0.90, axis=1),
        f"{prefix}_top5_oracle_mean": top5,
        f"{prefix}_fraction_ge_0_8": (scores >= 0.8).mean(axis=1),
        f"{prefix}_fraction_ge_0_9": (scores >= 0.9).mean(axis=1),
        f"{prefix}_selected_index": selected,
        f"{prefix}_oracle_index": oracle,
    }
    factor_names = matrix["candidate_factor_names"].astype(str).tolist()
    for factor_index, factor_name in enumerate(factor_names):
        safe_name = factor_name.replace("score", "aggregate_score")
        output[f"{prefix}_candidate_mean_{safe_name}"] = factors[
            :, :, factor_index
        ].mean(axis=1)
        output[f"{prefix}_oracle_{safe_name}"] = factors[
            rows, oracle, factor_index
        ]
        output[f"{prefix}_selected_{safe_name}"] = factors[
            rows, selected, factor_index
        ]
    return output


def _cross_bank_geometry(
    tokens: Sequence[str],
    proposals_a: Mapping[str, np.ndarray],
    proposals_b: Mapping[str, np.ndarray],
    oracle_a: np.ndarray,
    oracle_b: np.ndarray,
) -> Dict[str, np.ndarray]:
    metrics = {
        "a_to_b_mean_nearest_ade_m": np.empty(len(tokens), dtype=np.float64),
        "b_to_a_mean_nearest_ade_m": np.empty(len(tokens), dtype=np.float64),
        "a_to_b_mean_nearest_endpoint_m": np.empty(len(tokens), dtype=np.float64),
        "b_to_a_mean_nearest_endpoint_m": np.empty(len(tokens), dtype=np.float64),
        "oracle_cross_bank_ade_m": np.empty(len(tokens), dtype=np.float64),
        "oracle_cross_bank_endpoint_m": np.empty(len(tokens), dtype=np.float64),
    }
    for row, token in enumerate(tokens):
        a = proposals_a[token]
        b = proposals_b[token]
        delta = a[:, None, :, :2] - b[None, :, :, :2]
        point_distance = np.linalg.norm(delta, axis=-1)
        ade = point_distance.mean(axis=-1)
        endpoint = point_distance[:, :, -1]
        metrics["a_to_b_mean_nearest_ade_m"][row] = ade.min(axis=1).mean()
        metrics["b_to_a_mean_nearest_ade_m"][row] = ade.min(axis=0).mean()
        metrics["a_to_b_mean_nearest_endpoint_m"][row] = endpoint.min(axis=1).mean()
        metrics["b_to_a_mean_nearest_endpoint_m"][row] = endpoint.min(axis=0).mean()
        left = int(oracle_a[row])
        right = int(oracle_b[row])
        metrics["oracle_cross_bank_ade_m"][row] = ade[left, right]
        metrics["oracle_cross_bank_endpoint_m"][row] = endpoint[left, right]
    return metrics


def _delta_summary(
    values: np.ndarray,
    physical_logs: Sequence[str],
    seed: int,
    replicates: int,
) -> Dict[str, object]:
    low, high = _log_bootstrap_ci(values, physical_logs, seed, replicates)
    return {
        "mean": float(values.mean()),
        "log_bootstrap_95ci": [float(low), float(high)],
        "wins": int((values > 1.0e-9).sum()),
        "losses": int((values < -1.0e-9).sum()),
        "ties": int((np.abs(values) <= 1.0e-9).sum()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a-name", required=True)
    parser.add_argument("--a-matrix", type=Path, required=True)
    parser.add_argument("--a-proposals", type=Path, required=True)
    parser.add_argument("--b-name", required=True)
    parser.add_argument("--b-matrix", type=Path, required=True)
    parser.add_argument("--b-proposals", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260901)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    matrix_b_raw = _load_matrix(args.b_matrix)
    tokens = sorted(matrix_b_raw["tokens"].astype(str).tolist())
    matrix_b = _align_matrix(matrix_b_raw, tokens)
    matrix_a = _align_matrix(_load_matrix(args.a_matrix), tokens)
    if len(tokens) != 12_146:
        raise RuntimeError(f"Expected 12146 scenes, got {len(tokens)}")
    if not np.array_equal(
        matrix_a["candidate_factor_names"], matrix_b["candidate_factor_names"]
    ):
        raise RuntimeError("Candidate factor schemas differ")
    log_names = matrix_b["log_names"].astype(str).tolist()
    if matrix_a["log_names"].astype(str).tolist() != log_names:
        raise RuntimeError("Aligned candidate matrices disagree on log names")
    physical_logs = [_physical_log_name(value) for value in log_names]

    rows: Dict[str, np.ndarray] = {
        "token": np.asarray(tokens),
        "log_name": np.asarray(log_names),
        "physical_log": np.asarray(physical_logs),
        **_bank_rows(matrix_a, "a"),
        **_bank_rows(matrix_b, "b"),
    }
    proposals_a = _load_proposals(args.a_proposals)
    proposals_b = _load_proposals(args.b_proposals)
    if set(proposals_a) != set(tokens) or set(proposals_b) != set(tokens):
        raise RuntimeError("Proposal cache token set differs from candidate matrices")
    rows.update(
        _cross_bank_geometry(
            tokens,
            proposals_a,
            proposals_b,
            rows["a_oracle_index"],
            rows["b_oracle_index"],
        )
    )

    frame = pd.DataFrame(rows)
    frame["oracle_delta_a_minus_b"] = frame.a_oracle_pdms - frame.b_oracle_pdms
    frame["mean_candidate_delta_a_minus_b"] = (
        frame.a_mean_candidate_pdms - frame.b_mean_candidate_pdms
    )
    frame["selected_delta_a_minus_b"] = (
        frame.a_selected_pdms - frame.b_selected_pdms
    )
    frame["regret_delta_a_minus_b"] = frame.a_regret - frame.b_regret
    frame["union_oracle_pdms"] = np.maximum(
        frame.a_oracle_pdms, frame.b_oracle_pdms
    )
    frame.to_csv(args.output_dir / "per_scene_bank_comparison.csv", index=False)

    delta_fields = {
        "selected_pdms": "selected_delta_a_minus_b",
        "oracle_pdms": "oracle_delta_a_minus_b",
        "mean_candidate_pdms": "mean_candidate_delta_a_minus_b",
        "median_candidate_pdms": None,
        "top5_oracle_mean": None,
        "fraction_ge_0_8": None,
        "fraction_ge_0_9": None,
        "regret": "regret_delta_a_minus_b",
    }
    comparisons: Dict[str, object] = {}
    for offset, (name, ready_field) in enumerate(delta_fields.items()):
        values = (
            frame[ready_field].to_numpy()
            if ready_field is not None
            else (frame[f"a_{name}"] - frame[f"b_{name}"]).to_numpy()
        )
        comparisons[name] = _delta_summary(
            values,
            physical_logs,
            args.seed + offset,
            args.bootstrap_replicates,
        )

    candidate_factor_names = matrix_a["candidate_factor_names"].astype(str).tolist()
    factor_comparisons: Dict[str, object] = {}
    for offset, factor_name in enumerate(candidate_factor_names):
        safe_name = factor_name.replace("score", "aggregate_score")
        factor_comparisons[factor_name] = {
            "candidate_mean_delta": _delta_summary(
                (frame[f"a_candidate_mean_{safe_name}"] - frame[f"b_candidate_mean_{safe_name}"]).to_numpy(),
                physical_logs,
                args.seed + 100 + offset,
                args.bootstrap_replicates,
            ),
            "oracle_factor_delta": _delta_summary(
                (frame[f"a_oracle_{safe_name}"] - frame[f"b_oracle_{safe_name}"]).to_numpy(),
                physical_logs,
                args.seed + 200 + offset,
                args.bootstrap_replicates,
            ),
            "selected_factor_delta": _delta_summary(
                (frame[f"a_selected_{safe_name}"] - frame[f"b_selected_{safe_name}"]).to_numpy(),
                physical_logs,
                args.seed + 300 + offset,
                args.bootstrap_replicates,
            ),
        }

    selected_delta = comparisons["selected_pdms"]["mean"]
    oracle_delta = comparisons["oracle_pdms"]["mean"]
    regret_delta = comparisons["regret"]["mean"]
    decomposition_error = selected_delta - (oracle_delta - regret_delta)
    payload = {
        "a_name": args.a_name,
        "b_name": args.b_name,
        "delta_convention": "a_minus_b",
        "scene_count": len(tokens),
        "segment_log_count": len(set(log_names)),
        "physical_log_count": len(set(physical_logs)),
        "candidate_count_per_bank": 64,
        "comparisons": comparisons,
        "factor_comparisons": factor_comparisons,
        "means": {
            key: float(frame[key].mean())
            for key in frame.columns
            if key.startswith("a_")
            or key.startswith("b_")
            or key.startswith("union_")
            or key.endswith("_m")
        },
        "selection_gap_decomposition": {
            "selected_delta": selected_delta,
            "oracle_ceiling_delta": oracle_delta,
            "regret_delta": regret_delta,
            "identity": "selected_delta = oracle_ceiling_delta - regret_delta",
            "identity_error": float(decomposition_error),
        },
        "union_oracle_gain_over_a": float(
            frame.union_oracle_pdms.mean() - frame.a_oracle_pdms.mean()
        ),
        "union_oracle_gain_over_b": float(
            frame.union_oracle_pdms.mean() - frame.b_oracle_pdms.mean()
        ),
    }
    (args.output_dir / "comparison.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )

    lines = [
        f"# Native proposal-bank comparison: {args.a_name} vs {args.b_name}",
        "",
        "All values use identical complete Navtest scene tokens and official PDM scoring.",
        "Best-of-64 is an offline oracle candidate-bank upper bound.",
        "",
        "| Quantity | A | B | A - B | 95% physical-log bootstrap CI |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name in (
        "selected_pdms",
        "oracle_pdms",
        "mean_candidate_pdms",
        "median_candidate_pdms",
        "top5_oracle_mean",
        "fraction_ge_0_8",
        "fraction_ge_0_9",
        "regret",
    ):
        result = comparisons[name]
        low, high = result["log_bootstrap_95ci"]
        lines.append(
            f"| {name} | {frame[f'a_{name}'].mean():.6f} | "
            f"{frame[f'b_{name}'].mean():.6f} | {result['mean']:+.6f} | "
            f"[{low:+.6f}, {high:+.6f}] |"
        )
    lines.extend(
        [
            "",
            "## Selection-gap decomposition",
            "",
            f"- Selected PDMS delta: `{selected_delta:+.6f}`",
            f"- Oracle ceiling delta: `{oracle_delta:+.6f}`",
            f"- Regret delta: `{regret_delta:+.6f}`",
            "- Identity: selected delta = oracle ceiling delta - regret delta.",
            f"- Union best-of-128: `{frame.union_oracle_pdms.mean():.6f}`.",
            "",
            "## Cross-bank geometry",
            "",
            f"- Mean A-to-B nearest ADE: `{frame.a_to_b_mean_nearest_ade_m.mean():.3f} m`.",
            f"- Mean B-to-A nearest ADE: `{frame.b_to_a_mean_nearest_ade_m.mean():.3f} m`.",
            f"- Oracle-trajectory cross-bank ADE: `{frame.oracle_cross_bank_ade_m.mean():.3f} m`.",
            "",
            "## Selected-factor comparison",
            "",
            "| Factor | A | B | A - B | 95% physical-log bootstrap CI |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for factor_name in candidate_factor_names:
        safe_name = factor_name.replace("score", "aggregate_score")
        result = factor_comparisons[factor_name]["selected_factor_delta"]
        low, high = result["log_bootstrap_95ci"]
        lines.append(
            f"| {factor_name} | {frame[f'a_selected_{safe_name}'].mean():.6f} | "
            f"{frame[f'b_selected_{safe_name}'].mean():.6f} | "
            f"{result['mean']:+.6f} | [{low:+.6f}, {high:+.6f}] |"
        )
    (args.output_dir / "COMPARISON.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
