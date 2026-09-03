#!/usr/bin/env python3
"""Select and lock one shared Base/VQA formal PlanReg-WM GPU layout."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
from typing import Any, Dict, Mapping, Tuple


LAYOUT_SPECS = {
    "8x2": {"gpu_count": 8, "per_gpu_batch_size": 2, "global_batch_size": 16, "scorer_processes_per_rank": 8, "scorer_partitions_per_scene": 8},
    "8x4": {"gpu_count": 8, "per_gpu_batch_size": 4, "global_batch_size": 32, "scorer_processes_per_rank": 8, "scorer_partitions_per_scene": 8},
    "16x2": {"gpu_count": 16, "per_gpu_batch_size": 2, "global_batch_size": 32, "scorer_processes_per_rank": 4, "scorer_partitions_per_scene": 8},
    "16x4": {"gpu_count": 16, "per_gpu_batch_size": 4, "global_batch_size": 64, "scorer_processes_per_rank": 4, "scorer_partitions_per_scene": 8},
    "16x6": {
        "gpu_count": 16,
        "per_gpu_batch_size": 6,
        "global_batch_size": 96,
        "scorer_processes_per_rank": 8,
        "scorer_partitions_per_scene": 8,
        "gradient_checkpointing": False,
        "read_only_attention_backend": "split_sdpa",
    },
    "16x8": {
        "gpu_count": 16,
        "per_gpu_batch_size": 8,
        "global_batch_size": 128,
        "scorer_processes_per_rank": 8,
        "scorer_partitions_per_scene": 2,
        "gradient_checkpointing": False,
        "read_only_attention_backend": "split_sdpa",
    },
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
    reserved_memory_limit_gib: float = 76.0,
    max_p90_to_median_ratio: float = 1.35,
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
    if int(metrics.get("scorer_partitions_per_scene", -1)) != int(
        expected["scorer_partitions_per_scene"]
    ):
        return False, "scorer partition count mismatch"
    for name in ("gradient_checkpointing", "read_only_attention_backend"):
        if name in expected and metrics.get(name) != expected[name]:
            return False, f"{name} mismatch"
    if bool(metrics.get("oom")) or bool(metrics.get("deadlock")):
        return False, "OOM/deadlock recorded"
    if int(metrics.get("nonfinite_count", 0)) != 0:
        return False, "non-finite values recorded"
    peak = float(metrics.get("peak_allocated_gib", math.inf))
    if not math.isfinite(peak) or peak >= memory_limit_gib:
        return False, f"peak allocated memory {peak} GiB is not < {memory_limit_gib}"
    reserved = float(metrics.get("peak_reserved_gib", math.inf))
    if not math.isfinite(reserved) or reserved >= reserved_memory_limit_gib:
        return False, (
            f"peak reserved memory {reserved} GiB is not < "
            f"{reserved_memory_limit_gib}"
        )
    throughput = float(metrics.get("samples_per_second", float("nan")))
    if not math.isfinite(throughput) or throughput <= 0:
        return False, "invalid samples_per_second"
    median_step = float(metrics.get("median_step_time", float("nan")))
    p90_step = float(metrics.get("p90_step_time", float("nan")))
    if not math.isfinite(median_step) or not math.isfinite(p90_step):
        return False, "invalid median/p90 step time"
    if p90_step > median_step * max_p90_to_median_ratio:
        return False, (
            f"step-time tail ratio={p90_step / median_step:.4f} exceeds "
            f"{max_p90_to_median_ratio:.4f}"
        )
    return True, "eligible"


def select_layout(
    metrics_by_layout: Mapping[str, Mapping[str, Any]],
    *,
    near_peak_throughput_ratio: float = 0.95,
) -> Tuple[str, Dict[str, str]]:
    """Choose minimum total wall time while preserving useful update count.

    Layouts within five percent of peak sample throughput are effectively tied
    for a multi-day run. Among those, the smaller global batch is selected to
    retain more optimizer updates and more memory headroom.
    """
    if not 0.0 < near_peak_throughput_ratio <= 1.0:
        raise ValueError("near_peak_throughput_ratio must be in (0, 1]")
    decisions: Dict[str, str] = {}
    eligible: Dict[str, Mapping[str, Any]] = {}
    for layout, metrics in metrics_by_layout.items():
        valid, reason = validate_layout_metrics(layout, metrics)
        decisions[layout] = reason
        if valid:
            eligible[layout] = metrics
    if not eligible:
        raise RuntimeError("No eligible formal layout passed the throughput gates")

    peak_throughput = max(
        float(metrics["samples_per_second"]) for metrics in eligible.values()
    )
    near_peak = {
        layout: metrics
        for layout, metrics in eligible.items()
        if float(metrics["samples_per_second"])
        >= peak_throughput * near_peak_throughput_ratio
    }
    selected = min(
        near_peak,
        key=lambda name: (
            int(LAYOUT_SPECS[name]["global_batch_size"]),
            -float(near_peak[name]["samples_per_second"]),
        ),
    )
    for layout, metrics in eligible.items():
        throughput_ratio = float(metrics["samples_per_second"]) / peak_throughput
        if layout == selected:
            decisions[layout] = (
                f"selected: {throughput_ratio:.4f}x peak throughput; smallest "
                "global batch within near-peak band"
            )
        elif layout in near_peak:
            decisions[layout] = (
                f"not selected: {throughput_ratio:.4f}x peak throughput but "
                "larger global batch/fewer optimizer updates"
            )
        else:
            decisions[layout] = (
                f"not selected: {throughput_ratio:.4f}x peak throughput is below "
                f"the {near_peak_throughput_ratio:.4f} near-peak band"
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
    observed_throughput = float(metrics["samples_per_second"])
    padded_samples = steps_per_epoch * global_batch
    return {
        "schema_version": 2,
        "selected_layout": selected,
        **spec,
        "num_nodes": 2 if spec["gpu_count"] == 16 else 1,
        "devices_per_node": 8,
        "num_workers_per_rank": int(metrics["num_workers_per_rank"]),
        "gradient_checkpointing": bool(
            metrics.get("gradient_checkpointing", True)
        ),
        "read_only_attention_backend": str(
            metrics.get("read_only_attention_backend", "eager")
        ),
        "lr_scale_multiplier": lr_scale,
        "logical_peak_learning_rates": logical_lrs,
        "ema_actual_start_momentum": 0.996 ** (global_batch / 16.0),
        "ema_actual_end_momentum": 0.9999 ** (global_batch / 16.0),
        "dataset_length": dataset_length,
        "dataset_epochs": epochs,
        "steps_per_epoch": steps_per_epoch,
        "total_steps": steps_per_epoch * epochs,
        "sampler_padding_per_epoch": steps_per_epoch * global_batch - dataset_length,
        "observed_samples_per_second": observed_throughput,
        "estimated_epoch_hours": padded_samples / observed_throughput / 3600.0,
        "estimated_27_epoch_hours": (
            padded_samples * epochs / observed_throughput / 3600.0
        ),
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
    parser.add_argument("--near-peak-throughput-ratio", type=float, default=0.95)
    args = parser.parse_args()

    metrics_by_layout: Dict[str, Mapping[str, Any]] = {}
    hashes: Dict[str, str] = {}
    for layout in LAYOUT_SPECS:
        path = args.metrics_root / layout / "metrics.json"
        if not path.is_file():
            continue
        metrics_by_layout[layout] = json.loads(path.read_text(encoding="utf-8"))
        hashes[layout] = _sha256(path)
    if not metrics_by_layout:
        raise FileNotFoundError(
            f"No recognized benchmark metrics found under {args.metrics_root}"
        )
    selected, decisions = select_layout(
        metrics_by_layout,
        near_peak_throughput_ratio=args.near_peak_throughput_ratio,
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
