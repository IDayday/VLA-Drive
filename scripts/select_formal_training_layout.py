#!/usr/bin/env python3
"""Select and lock one shared Base/VQA formal PlanReg-WM GPU layout."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
from typing import Any, Dict, Mapping, Optional, Tuple


LAYOUT_SPECS = {
    "8x2": {"gpu_count": 8, "per_gpu_batch_size": 2, "global_batch_size": 16, "scorer_processes_per_rank": 8},
    "8x4": {"gpu_count": 8, "per_gpu_batch_size": 4, "global_batch_size": 32, "scorer_processes_per_rank": 8},
    "16x2": {"gpu_count": 16, "per_gpu_batch_size": 2, "global_batch_size": 32, "scorer_processes_per_rank": 4},
    "16x4": {"gpu_count": 16, "per_gpu_batch_size": 4, "global_batch_size": 64, "scorer_processes_per_rank": 4},
}
REFERENCE_LRS = {
    "planning_adapter": 2.0e-4,
    "semantic_fusion": 2.0e-4,
    "action_generator": 2.0e-4,
    "scorer": 2.0e-4,
    "future_predictor": 2.0e-4,
    "semantic_qformer": 1.0e-4,
    "vision_qv_lora": 4.0e-5,
}
LR_CAPS = {
    "planning_adapter": 3.0e-4,
    "semantic_fusion": 3.0e-4,
    "future_predictor": 3.0e-4,
    "action_generator": 3.0e-4,
    "scorer": 3.0e-4,
    "semantic_qformer": 1.5e-4,
    "vision_qv_lora": 5.0e-5,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_layout_metrics(
    layout: str,
    metrics: Mapping[str, Any],
    *,
    memory_limit_gib: float = 72.0,
) -> Tuple[bool, str]:
    if layout not in LAYOUT_SPECS:
        return False, f"unknown layout {layout!r}"
    expected = LAYOUT_SPECS[layout]
    if metrics.get("layout") != layout:
        return False, "layout label mismatch"
    if metrics.get("status") != "success":
        return False, f"status={metrics.get('status')!r}"
    if int(metrics.get("timed_optimizer_steps", 0)) < 300:
        return False, "fewer than 300 timed optimizer steps"
    for name in ("gpu_count", "global_batch_size", "per_gpu_batch_size"):
        if int(metrics.get(name, -1)) != int(expected[name]):
            return False, f"{name} mismatch"
    if int(metrics.get("scorer_processes_per_rank", -1)) != int(
        expected["scorer_processes_per_rank"]
    ):
        return False, "scorer process count mismatch"
    if bool(metrics.get("oom")) or bool(metrics.get("deadlock")):
        return False, "OOM/deadlock recorded"
    if int(metrics.get("nonfinite_count", 0)) != 0:
        return False, "non-finite values recorded"
    peak = float(metrics.get("peak_allocated_gib", math.inf))
    if not math.isfinite(peak) or peak >= memory_limit_gib:
        return False, f"peak allocated memory {peak} GiB is not < {memory_limit_gib}"
    throughput = float(metrics.get("samples_per_second", float("nan")))
    if not math.isfinite(throughput) or throughput <= 0:
        return False, "invalid samples_per_second"
    return True, "eligible"


def validate_global64_stability(stability: Optional[Mapping[str, Any]]) -> Tuple[bool, str]:
    if stability is None:
        return False, "missing 1000-step global64 stability audit"
    if int(stability.get("optimizer_steps", 0)) < 1000:
        return False, "global64 stability audit has fewer than 1000 steps"
    required_true = (
        "trajectory_loss_no_clear_degradation",
        "scorer_loss_no_clear_degradation",
        "gradient_norm_stable",
    )
    failed = [name for name in required_true if not bool(stability.get(name, False))]
    if failed:
        return False, f"global64 stability gates failed: {failed}"
    if int(stability.get("nonfinite_count", 0)) != 0:
        return False, "global64 stability audit contains non-finite values"
    return True, "eligible"


def select_layout(
    metrics_by_layout: Mapping[str, Mapping[str, Any]],
    *,
    global64_stability: Optional[Mapping[str, Any]] = None,
) -> Tuple[str, Dict[str, str]]:
    decisions: Dict[str, str] = {}
    eligible: Dict[str, Mapping[str, Any]] = {}
    for layout in LAYOUT_SPECS:
        valid, reason = validate_layout_metrics(layout, metrics_by_layout[layout])
        decisions[layout] = reason
        if valid:
            eligible[layout] = metrics_by_layout[layout]

    global32 = [layout for layout in ("8x4", "16x2") if layout in eligible]
    if global32:
        selected = max(
            global32,
            key=lambda name: float(eligible[name]["samples_per_second"]),
        )
    elif "8x2" in eligible:
        selected = "8x2"
    else:
        raise RuntimeError(
            "No eligible formal layout: neither global-batch-32 layout nor 8x2 passed"
        )

    if "16x4" in eligible:
        baseline_throughput = float(eligible[selected]["samples_per_second"])
        global64_throughput = float(eligible["16x4"]["samples_per_second"])
        speedup = global64_throughput / baseline_throughput
        stability_ok, stability_reason = validate_global64_stability(
            global64_stability
        )
        if speedup >= 1.25 and stability_ok:
            selected = "16x4"
            decisions["16x4"] = (
                f"selected: {speedup:.4f}x throughput and 1000-step stability passed"
            )
        else:
            decisions["16x4"] = (
                f"not selected: throughput ratio={speedup:.4f}; {stability_reason}"
            )
    return selected, decisions


def build_layout_lock(
    selected: str,
    metrics_by_layout: Mapping[str, Mapping[str, Any]],
    decisions: Mapping[str, str],
    *,
    dataset_length: int = 103288,
    epochs: int = 27,
    source_commit: str,
    metrics_sha256: Mapping[str, str],
) -> Dict[str, Any]:
    spec = dict(LAYOUT_SPECS[selected])
    global_batch = int(spec["global_batch_size"])
    steps_per_epoch = math.ceil(dataset_length / global_batch)
    lr_scale = math.sqrt(global_batch / 32.0)
    logical_lrs = {
        name: min(reference * lr_scale, LR_CAPS[name])
        for name, reference in REFERENCE_LRS.items()
    }
    metrics = metrics_by_layout[selected]
    return {
        "schema_version": 1,
        "selected_layout": selected,
        **spec,
        "num_nodes": 2 if spec["gpu_count"] == 16 else 1,
        "devices_per_node": 8,
        "num_workers_per_rank": int(metrics["num_workers_per_rank"]),
        "lr_scale_multiplier": lr_scale,
        "logical_peak_learning_rates": logical_lrs,
        "ema_actual_start_momentum": 0.996 ** (global_batch / 16.0),
        "ema_actual_end_momentum": 0.9999 ** (global_batch / 16.0),
        "dataset_length": dataset_length,
        "dataset_epochs": epochs,
        "steps_per_epoch": steps_per_epoch,
        "total_steps": steps_per_epoch * epochs,
        "sampler_padding_per_epoch": steps_per_epoch * global_batch - dataset_length,
        "source_git_commit": source_commit,
        "selection_decisions": dict(decisions),
        "benchmark_metrics_sha256": dict(metrics_sha256),
        "shared_between_base_and_vqa": True,
    }


def _git_commit(repo_root: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True
    ).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dataset-length", type=int, default=103288)
    parser.add_argument("--epochs", type=int, default=27)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--global64-stability", type=Path)
    args = parser.parse_args()

    metrics_by_layout: Dict[str, Mapping[str, Any]] = {}
    hashes: Dict[str, str] = {}
    for layout in LAYOUT_SPECS:
        path = args.metrics_root / layout / "metrics.json"
        if not path.is_file():
            raise FileNotFoundError(f"Missing required benchmark metrics: {path}")
        metrics_by_layout[layout] = json.loads(path.read_text(encoding="utf-8"))
        hashes[layout] = _sha256(path)
    stability = None
    if args.global64_stability is not None:
        stability = json.loads(args.global64_stability.read_text(encoding="utf-8"))
    selected, decisions = select_layout(
        metrics_by_layout, global64_stability=stability
    )
    lock = build_layout_lock(
        selected,
        metrics_by_layout,
        decisions,
        dataset_length=args.dataset_length,
        epochs=args.epochs,
        source_commit=_git_commit(args.repo_root.resolve()),
        metrics_sha256=hashes,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(lock, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
