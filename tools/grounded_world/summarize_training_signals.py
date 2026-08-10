#!/usr/bin/env python3
"""Summarize learned-module diagnostics from an actual training JSONL file."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


def load_metric_records(path: str | Path) -> list[dict[str, Any]]:
    """Load strict scalar metric records from ``training_metrics.jsonl``."""

    metric_path = Path(path)
    if not metric_path.is_file():
        raise FileNotFoundError(f"training metric file not found: {metric_path}")
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        metric_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"invalid JSON at {metric_path}:{line_number}"
            ) from error
        if not isinstance(payload, dict) or not isinstance(
            payload.get("step"), (int, float)
        ):
            raise ValueError(
                f"metric record requires numeric step at {metric_path}:{line_number}"
            )
        records.append(payload)
    if not records:
        raise ValueError(f"training metric file is empty: {metric_path}")
    records.sort(key=lambda value: float(value["step"]))
    return records


def _finite_values(
    records: Sequence[Mapping[str, Any]], key: str
) -> list[float]:
    values = []
    for record in records:
        value = record.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            numeric = float(value)
            if math.isfinite(numeric):
                values.append(numeric)
    return values


def _window_mean(values: Sequence[float], window: int) -> float | None:
    if not values:
        return None
    selected = values[-min(window, len(values)) :]
    return sum(selected) / len(selected)


def _threshold_check(
    records: Sequence[Mapping[str, Any]],
    key: str,
    *,
    window: int,
    minimum: float,
    description: str,
) -> dict[str, Any]:
    values = _finite_values(records, key)
    mean = _window_mean(values, window)
    if mean is None:
        return {
            "status": "MISSING",
            "metric": key,
            "last_window_mean": None,
            "criterion": f"> {minimum}",
            "description": description,
        }
    return {
        "status": "PASS" if mean > minimum else "WARN",
        "metric": key,
        "last_window_mean": mean,
        "criterion": f"> {minimum}",
        "description": description,
    }


def summarize_training_signals(
    records: Sequence[Mapping[str, Any]],
    stage: str,
    window: int = 20,
) -> dict[str, Any]:
    """Build a non-evaluative learning audit from real logged scalar records."""

    if stage not in {"prior", "predictive", "planning"}:
        raise ValueError("stage must be prior, predictive, or planning")
    if window <= 0:
        raise ValueError("window must be positive")
    if not records:
        raise ValueError("records must not be empty")
    prior_metric = (
        "retention_scene_shuffle_margin"
        if stage in {"predictive", "planning"}
        and _finite_values(records, "retention_scene_shuffle_margin")
        else "prior_scene_shuffle_margin"
    )
    checks = {
        "prior_alignment": _threshold_check(
            records,
            prior_metric,
            window=window,
            minimum=0.0,
            description="real current/history prior is more identifiable than scene shuffle",
        ),
        "geometry_coverage": _threshold_check(
            records,
            "geometry_valid_ratio",
            window=window,
            minimum=0.01,
            description="geometry targets cover a nontrivial fraction of the field",
        ),
        "future_temporal_identity": _threshold_check(
            records,
            "future_temporal_margin",
            window=window,
            minimum=0.0,
            description="future targets match the correct time better than a temporal shuffle",
        ),
        "trajectory_tube_coverage": _threshold_check(
            records,
            "world_tube_valid_ratio",
            window=window,
            minimum=0.05,
            description="the planned tube lands inside readable world memory",
        ),
        "refiner_nonzero_update": _threshold_check(
            records,
            "world_delta_norm",
            window=window,
            minimum=1.0e-4,
            description="the zero-initialized refiner has begun learning nonzero corrections",
        ),
    }
    all_keys = {str(key) for record in records for key in record}
    suffix = "_target_std"
    prefix = "consequence_"
    for key in sorted(all_keys):
        if not key.startswith(prefix) or not key.endswith(suffix):
            continue
        name = key[len(prefix) : -len(suffix)]
        prediction_key = f"consequence_{name}_prediction_std"
        target = _window_mean(_finite_values(records, key), window)
        prediction = _window_mean(
            _finite_values(records, prediction_key), window
        )
        ratio = (
            prediction / target
            if target is not None
            and prediction is not None
            and target > 1.0e-8
            else None
        )
        checks[f"consequence_{name}_noncollapse"] = {
            "status": (
                "MISSING" if ratio is None else "PASS" if ratio >= 0.05 else "WARN"
            ),
            "metric": prediction_key,
            "last_window_mean": prediction,
            "target_std": target,
            "prediction_to_target_std_ratio": ratio,
            "criterion": ">= 0.05",
            "description": "prediction varies across fixed candidates instead of collapsing",
        }
    return {
        "stage": stage,
        "record_count": len(records),
        "first_step": int(float(records[0]["step"])),
        "last_step": int(float(records[-1]["step"])),
        "window": int(window),
        "checks": checks,
        "interpretation": (
            "PASS/WARN only audits whether a module appears to learn. It is not a "
            "planning-gain claim; B0-B5 paired NAVSIM evaluation is still required."
        ),
    }


def _markdown(report: Mapping[str, Any]) -> str:
    lines = [
        f"# GroundedWorld training-signal audit (step {report['last_step']})",
        "",
        "| Check | Status | Last-window value | Criterion |",
        "|---|---:|---:|---:|",
    ]
    for name, result in report["checks"].items():
        value = result.get("last_window_mean")
        rendered = "MISSING" if value is None else f"{float(value):.6g}"
        lines.append(
            f"| {name} | {result['status']} | {rendered} | {result['criterion']} |"
        )
    lines.extend(("", str(report["interpretation"])))
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run-dir", type=Path)
    source.add_argument("--metrics", type=Path)
    parser.add_argument("--stage", choices=("prior", "predictive", "planning"), required=True)
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metric_path = (
        args.metrics
        if args.metrics is not None
        else args.run_dir / "training_metrics.jsonl"
    )
    report = summarize_training_signals(
        load_metric_records(metric_path), args.stage, args.window
    )
    rendered = _markdown(report)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_name(args.output.name + f".tmp-{os.getpid()}")
        temporary.write_text(rendered, encoding="utf-8")
        os.replace(temporary, args.output)
    print(rendered, end="")


if __name__ == "__main__":
    main()
