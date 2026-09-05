#!/usr/bin/env python3
"""Audit PlanReg-WM training dynamics from one or more TensorBoard runs.

The script deliberately reports task-facing quantities (proposal score, oracle
gap, per-component losses, horizon losses, gradient health, and register
geometry) instead of treating the scalar total loss as sufficient evidence.
Restarted TensorBoard streams are merged by optimizer step; the newest event
at a duplicated step wins.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


TASK_TAGS = (
    "train/loss",
    "train/trajectory_loss",
    "train/min_loss",
    "train/inter_loss",
    "train/final_score_loss",
    "train/da_loss",
    "train/ttc_loss",
    "train/noc_loss",
    "train/progress_loss",
    "train/ddc_loss",
    "train/comfort_loss",
    "train/score",
    "train/best_score",
)
WM_TAGS = (
    "train/weighted_wm_loss",
    "train/wm_loss",
    "train/wm_abs_loss",
    "train/wm_delta_loss",
    "train/wm_abs_0p5",
    "train/wm_abs_1p5",
    "train/wm_abs_3p0",
    "train/wm_delta_0p5",
    "train/wm_delta_1p5",
    "train/wm_delta_3p0",
    "train/wm_weight_current",
    "train/ema_momentum_current",
)
GRAD_TAGS = (
    "train/vision_lora_grad_norm",
    "train/register_grad_norm",
    "train/future_predictor_grad_norm",
    "train/action_head_grad_norm",
    "train/scorer_grad_norm",
)
REGISTER_TAGS = (
    "train/register_effective_rank",
    "train/register_mean_pairwise_cosine",
    "train/register_std",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_float(value: float):
    return float(value) if math.isfinite(float(value)) else None


def _find_event_files(roots: Iterable[Path]) -> List[Path]:
    files: List[Path] = []
    for root in roots:
        if root.is_file() and root.name.startswith("events.out.tfevents"):
            files.append(root)
        elif root.exists():
            files.extend(root.rglob("events.out.tfevents.*"))
        else:
            raise FileNotFoundError(root)
    unique = sorted({path.resolve() for path in files})
    if not unique:
        raise RuntimeError("No TensorBoard event files were found")
    return unique


def _load_series(files: Iterable[Path]) -> Dict[str, List[dict]]:
    merged: Dict[str, Dict[int, dict]] = {}
    for path in files:
        accumulator = EventAccumulator(str(path), size_guidance={"scalars": 0})
        accumulator.Reload()
        for tag in accumulator.Tags().get("scalars", []):
            by_step = merged.setdefault(tag, {})
            for event in accumulator.Scalars(tag):
                record = {
                    "step": int(event.step),
                    "wall_time": float(event.wall_time),
                    "value": float(event.value),
                    "source": str(path),
                }
                previous = by_step.get(record["step"])
                if previous is None or record["wall_time"] >= previous["wall_time"]:
                    by_step[record["step"]] = record
    return {
        tag: [by_step[step] for step in sorted(by_step)]
        for tag, by_step in merged.items()
    }


def _window(values: np.ndarray, fraction: float = 0.1) -> int:
    return max(3, min(30, int(math.ceil(len(values) * fraction))))


def _summarize(records: List[dict]) -> dict:
    steps = np.asarray([record["step"] for record in records], dtype=np.float64)
    values = np.asarray([record["value"] for record in records], dtype=np.float64)
    finite = np.isfinite(values)
    if not finite.any():
        return {"count": int(len(values)), "finite_fraction": 0.0}
    steps = steps[finite]
    values = values[finite]
    count = len(values)
    window = _window(values)
    early = values[:window]
    late = values[-window:]
    late_steps = steps[-window:]
    slope = 0.0
    if len(np.unique(late_steps)) > 1:
        slope = float(np.polyfit(late_steps, late, 1)[0] * 1000.0)
    early_mean = float(early.mean())
    late_mean = float(late.mean())
    scale = max(abs(early_mean), 1e-12)
    return {
        "count": int(count),
        "first_step": int(steps[0]),
        "last_step": int(steps[-1]),
        "first": float(values[0]),
        "last": float(values[-1]),
        "min": float(values.min()),
        "min_step": int(steps[int(values.argmin())]),
        "max": float(values.max()),
        "early_window_count": int(window),
        "early_mean": early_mean,
        "late_mean": late_mean,
        "late_std": float(late.std()),
        "relative_late_vs_early": float((late_mean - early_mean) / scale),
        "late_slope_per_1000_steps": slope,
        "finite_fraction": float(finite.mean()),
        "zero_fraction": float(np.isclose(values, 0.0, atol=1e-15).mean()),
    }


def _series_arrays(series: Mapping[str, List[dict]], tag: str):
    records = series.get(tag, [])
    return (
        np.asarray([item["step"] for item in records]),
        np.asarray([item["value"] for item in records]),
    )


def _nearest_join(series: Mapping[str, List[dict]], left: str, right: str):
    left_map = {item["step"]: item["value"] for item in series.get(left, [])}
    right_map = {item["step"]: item["value"] for item in series.get(right, [])}
    steps = sorted(set(left_map) & set(right_map))
    return (
        np.asarray(steps, dtype=np.int64),
        np.asarray([left_map[step] for step in steps], dtype=np.float64),
        np.asarray([right_map[step] for step in steps], dtype=np.float64),
    )


def _plot(series: Mapping[str, List[dict]], output: Path) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(15, 13), constrained_layout=True)
    panels = (
        (axes[0, 0], ("train/loss", "train/trajectory_loss", "train/final_score_loss"), "Optimization losses"),
        (axes[0, 1], ("train/score", "train/best_score"), "Batch PDM target: selected vs offline oracle"),
        (axes[1, 0], ("train/da_loss", "train/ttc_loss", "train/noc_loss", "train/progress_loss", "train/ddc_loss", "train/comfort_loss"), "Scorer component BCE"),
        (axes[1, 1], ("train/wm_abs_0p5", "train/wm_abs_1p5", "train/wm_abs_3p0", "train/wm_delta_0p5", "train/wm_delta_1p5", "train/wm_delta_3p0"), "World-model horizons"),
        (axes[2, 0], GRAD_TAGS, "Gradient norms"),
        (axes[2, 1], REGISTER_TAGS, "Planning-register geometry"),
    )
    for axis, tags, title in panels:
        for tag in tags:
            steps, values = _series_arrays(series, tag)
            if len(values):
                axis.plot(steps, values, linewidth=1.4, label=tag.removeprefix("train/"))
        axis.set_title(title)
        axis.set_xlabel("optimizer step")
        axis.grid(alpha=0.25)
        if "Gradient" in title:
            axis.set_yscale("log")
        axis.legend(fontsize=7, ncols=2)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def audit(event_roots: Iterable[Path], output_dir: Path, target_step: int | None) -> dict:
    files = _find_event_files(event_roots)
    series = _load_series(files)
    summaries = {tag: _summarize(records) for tag, records in sorted(series.items())}

    steps, selected, oracle = _nearest_join(series, "train/score", "train/best_score")
    gap = oracle - selected
    trajectory_task = {}
    if len(gap):
        window = _window(gap)
        trajectory_task = {
            "count": int(len(gap)),
            "early_selected_score": float(selected[:window].mean()),
            "late_selected_score": float(selected[-window:].mean()),
            "early_oracle_score": float(oracle[:window].mean()),
            "late_oracle_score": float(oracle[-window:].mean()),
            "early_selection_gap": float(gap[:window].mean()),
            "late_selection_gap": float(gap[-window:].mean()),
            "last_selection_gap": float(gap[-1]),
            "interpretation": (
                "best_score measures proposal-bank quality on training batches; "
                "score measures the scorer-selected candidate on those same batches."
            ),
        }

    gradient_health = {}
    for tag in GRAD_TAGS:
        _, values = _series_arrays(series, tag)
        if len(values):
            finite = np.isfinite(values)
            gradient_health[tag.removeprefix("train/")] = {
                "observations": int(len(values)),
                "finite_fraction": float(finite.mean()),
                "nonzero_fraction": float((np.abs(values[finite]) > 0).mean()) if finite.any() else 0.0,
                "median": _finite_float(np.median(values[finite])) if finite.any() else None,
                "p95": _finite_float(np.percentile(values[finite], 95)) if finite.any() else None,
                "max": _finite_float(np.max(values[finite])) if finite.any() else None,
            }

    wall_records = series.get("train/loss", [])
    throughput = {}
    if len(wall_records) >= 2:
        tail = wall_records[-min(20, len(wall_records)):]
        delta_steps = tail[-1]["step"] - tail[0]["step"]
        delta_time = tail[-1]["wall_time"] - tail[0]["wall_time"]
        seconds_per_step = delta_time / delta_steps if delta_steps > 0 else float("nan")
        throughput = {
            "last_logged_step": int(tail[-1]["step"]),
            "seconds_per_optimizer_step_recent": _finite_float(seconds_per_step),
        }
        if target_step is not None and seconds_per_step > 0:
            throughput.update(
                {
                    "target_step": int(target_step),
                    "remaining_steps_at_last_event": max(0, int(target_step - tail[-1]["step"])),
                    "estimated_remaining_seconds_at_last_event": max(0.0, float((target_step - tail[-1]["step"]) * seconds_per_step)),
                }
            )

    report = {
        "schema_version": 1,
        "event_files": [
            {"path": str(path), "sha256": _sha256(path), "size_bytes": path.stat().st_size}
            for path in files
        ],
        "optimizer_step_range": [
            min((record["step"] for records in series.values() for record in records), default=None),
            max((record["step"] for records in series.values() for record in records), default=None),
        ],
        "scalar_tag_count": len(series),
        "scalar_summaries": summaries,
        "trajectory_and_selection_task": trajectory_task,
        "gradient_health": gradient_health,
        "throughput": throughput,
        "caveat": (
            "Training-batch score is not held-out Navtest PDMS. Task learning is "
            "only established when combined with immutable Navtest candidate scoring."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "training_dynamics.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (output_dir / "training_scalars.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("tag", "step", "wall_time", "value", "source"))
        for tag, records in sorted(series.items()):
            for record in records:
                writer.writerow((tag, record["step"], record["wall_time"], record["value"], record["source"]))
    _plot(series, output_dir / "training_dynamics.png")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--event-root", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-step", type=int)
    args = parser.parse_args()
    report = audit(args.event_root, args.output_dir, args.target_step)
    print(json.dumps(report["trajectory_and_selection_task"], indent=2, sort_keys=True))
    print(json.dumps(report["throughput"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
