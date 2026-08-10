#!/usr/bin/env python3
"""Aggregate only actual GroundedWorld NAVSIM results with paired bootstrap."""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


SUMMARY_TOKENS = {
    "average_all_frames",
    "extended_pdm_score_stage_one",
    "extended_pdm_score_stage_two",
    "extended_pdm_score_combined",
}


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON result: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"JSON result must be an object: {path}")
    return payload


def _score_keys(summary: Mapping[str, Any]) -> tuple[str, ...]:
    suite = summary.get("suite")
    if suite == "navtest":
        return ("official_summary_score",)
    if suite == "navhard_two_stage":
        return ("stage_one_score", "stage_two_score", "combined_score")
    raise ValueError(f"unsupported result suite: {suite}")


def _scene_scores(summary: Mapping[str, Any]) -> dict[str, float]:
    csv_path = Path(str(summary.get("csv", "")))
    if not csv_path.is_file():
        raise FileNotFoundError(f"per-scene result CSV not found: {csv_path}")
    frame = pd.read_csv(csv_path)
    if not {"token", "valid", "score"}.issubset(frame.columns):
        raise ValueError(f"per-scene CSV lacks token/valid/score: {csv_path}")
    frame = frame[~frame["token"].isin(SUMMARY_TOKENS)]
    valid = frame["valid"] if frame["valid"].dtype == bool else frame["valid"].astype(str).str.lower().isin(("true", "1"))
    if not valid.all():
        raise ValueError(f"per-scene CSV contains invalid scenarios: {csv_path}")
    scores = pd.to_numeric(frame["score"], errors="coerce")
    if not scores.map(math.isfinite).all():
        raise ValueError(f"per-scene CSV contains non-finite scores: {csv_path}")
    tokens = frame["token"].astype(str).tolist()
    if len(tokens) != len(set(tokens)):
        raise ValueError(f"per-scene CSV contains duplicate tokens: {csv_path}")
    return dict(zip(tokens, scores.astype(float).tolist()))


def _paired_bootstrap(
    candidate: Mapping[str, float],
    reference: Mapping[str, float],
    *,
    samples: int,
    seed: int,
) -> dict[str, float | int]:
    if set(candidate) != set(reference):
        missing = sorted(set(reference) - set(candidate))
        extra = sorted(set(candidate) - set(reference))
        raise ValueError(
            f"paired token sets differ: missing={missing[:10]} extra={extra[:10]}"
        )
    tokens = sorted(candidate)
    difference = np.asarray(
        [candidate[token] - reference[token] for token in tokens], dtype=np.float64
    )
    generator = np.random.default_rng(int(seed))
    indices = generator.integers(0, len(tokens), size=(int(samples), len(tokens)))
    boot = difference[indices].mean(axis=1)
    return {
        "paired_tokens": len(tokens),
        "mean_difference": float(difference.mean()),
        "ci95_low": float(np.quantile(boot, 0.025)),
        "ci95_high": float(np.quantile(boot, 0.975)),
    }


def aggregate_matrix(
    matrix: Mapping[str, Any],
    *,
    reference_arm: str,
    bootstrap_samples: int = 10000,
    seed: int = 20260810,
) -> dict[str, Any]:
    """Load a declared run matrix; never synthesize scores for missing files."""

    runs = matrix.get("runs")
    if not isinstance(runs, list) or not runs:
        raise ValueError("matrix requires a non-empty runs list")
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    loaded: dict[tuple[str, int, int, str], dict[str, Any]] = {}
    run_rows: list[dict[str, Any]] = []
    for declared in runs:
        if not isinstance(declared, dict):
            raise ValueError("each matrix run must be an object")
        arm = str(declared.get("arm", ""))
        suite = str(declared.get("suite", ""))
        run_seed = int(declared.get("seed"))
        step = int(declared.get("step"))
        summary_path = Path(str(declared.get("summary", "")))
        key = (arm, run_seed, step, suite)
        if not arm or key in loaded:
            raise ValueError(f"invalid or duplicate run identity: {key}")
        row: dict[str, Any] = {
            "arm": arm,
            "seed": run_seed,
            "step": step,
            "suite": suite,
            "summary": str(summary_path),
        }
        if not summary_path.is_file():
            row["status"] = "MISSING"
            loaded[key] = {"status": "MISSING", "row": row}
            run_rows.append(row)
            continue
        summary = _load_json(summary_path)
        if summary.get("suite") != suite:
            raise ValueError(f"declared/result suite mismatch: {summary_path}")
        for score_key in _score_keys(summary):
            value = float(summary.get(score_key))
            if not math.isfinite(value):
                raise ValueError(f"non-finite {score_key}: {summary_path}")
            row[score_key] = value
        scenes = _scene_scores(summary)
        row["status"] = "OK"
        loaded[key] = {"status": "OK", "row": row, "scenes": scenes}
        run_rows.append(row)

    aggregates: list[dict[str, Any]] = []
    groups: dict[tuple[str, int, str, str], list[float]] = {}
    declared_counts: dict[tuple[str, int, str], int] = {}
    for row in run_rows:
        base = (row["arm"], row["step"], row["suite"])
        declared_counts[base] = declared_counts.get(base, 0) + 1
        if row["status"] != "OK":
            continue
        for score_key in (
            "official_summary_score",
            "stage_one_score",
            "stage_two_score",
            "combined_score",
        ):
            if score_key in row:
                groups.setdefault((*base, score_key), []).append(float(row[score_key]))
    for (arm, step, suite, score_key), values in sorted(groups.items()):
        declared = declared_counts[(arm, step, suite)]
        aggregates.append(
            {
                "arm": arm,
                "step": step,
                "suite": suite,
                "metric": score_key,
                "status": "OK" if len(values) == declared else "PARTIAL",
                "seeds_present": len(values),
                "seeds_declared": declared,
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else None,
            }
        )

    paired: list[dict[str, Any]] = []
    comparison_index = 0
    for key, candidate in sorted(loaded.items()):
        arm, run_seed, step, suite = key
        if arm == reference_arm or candidate["status"] != "OK":
            continue
        reference = loaded.get((reference_arm, run_seed, step, suite))
        if reference is None or reference["status"] != "OK":
            continue
        stats = _paired_bootstrap(
            candidate["scenes"],
            reference["scenes"],
            samples=bootstrap_samples,
            seed=seed + comparison_index,
        )
        comparison_index += 1
        paired.append(
            {
                "arm": arm,
                "reference_arm": reference_arm,
                "seed": run_seed,
                "step": step,
                "suite": suite,
                **stats,
            }
        )
    return {
        "reference_arm": reference_arm,
        "bootstrap_samples": int(bootstrap_samples),
        "bootstrap_seed": int(seed),
        "runs": run_rows,
        "aggregates": aggregates,
        "paired": paired,
    }


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise


def _markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# GroundedWorld NAVSIM result report",
        "",
        "Missing results remain `MISSING`; no score is imputed.",
        "",
        "## Declared runs",
        "",
        "| Arm | Seed | Step | Suite | Status |",
        "|---|---:|---:|---|---|",
    ]
    for row in report["runs"]:
        lines.append(
            f"| {row['arm']} | {row['seed']} | {row['step']} | {row['suite']} | {row['status']} |"
        )
    lines.extend(("", "## Aggregate", "", "| Arm | Step | Suite | Metric | Seeds | Mean | Std |", "|---|---:|---|---|---:|---:|---:|"))
    for row in report["aggregates"]:
        std = "NA" if row["std"] is None else f"{row['std']:.6g}"
        lines.append(
            f"| {row['arm']} | {row['step']} | {row['suite']} | {row['metric']} | "
            f"{row['seeds_present']}/{row['seeds_declared']} | {row['mean']:.6g} | {std} |"
        )
    lines.extend(("", "## Paired scene bootstrap", "", "| Arm - reference | Seed | Step | Suite | N | Mean diff | 95% CI |", "|---|---:|---:|---|---:|---:|---:|"))
    for row in report["paired"]:
        lines.append(
            f"| {row['arm']} - {row['reference_arm']} | {row['seed']} | {row['step']} | "
            f"{row['suite']} | {row['paired_tokens']} | {row['mean_difference']:.6g} | "
            f"[{row['ci95_low']:.6g}, {row['ci95_high']:.6g}] |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--reference-arm", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260810)
    args = parser.parse_args()
    report = aggregate_matrix(
        _load_json(args.matrix),
        reference_arm=args.reference_arm,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    _atomic_text(args.output_dir / "report.json", json.dumps(report, indent=2, sort_keys=True) + "\n")
    _atomic_text(args.output_dir / "report.md", _markdown(report))
    pd.DataFrame(report["runs"]).to_csv(args.output_dir / "runs.csv", index=False)
    pd.DataFrame(report["aggregates"]).to_csv(args.output_dir / "aggregates.csv", index=False)
    pd.DataFrame(report["paired"]).to_csv(args.output_dir / "paired.csv", index=False)
    print(_markdown(report), end="")


if __name__ == "__main__":
    main()
