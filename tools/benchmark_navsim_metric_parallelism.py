#!/usr/bin/env python3
"""Benchmark production-like NAVSIM scene parallelism on uncached tokens."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time

for variable in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
):
    os.environ.setdefault(variable, "1")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for package_root in (PROJECT_ROOT, PROJECT_ROOT / "navsim"):
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))

import numpy as np
import torch

from navsim.common.dataloader import MetricCacheLoader
from starVLA.training.navsim_metric_supervisor import (
    DynamicMetricSupervisor,
    _OUTPUT_METRICS,
)


def _proposals(batch_size: int, candidates: int) -> torch.Tensor:
    rng = np.random.default_rng(20260824)
    time_s = np.arange(1, 9, dtype=np.float32) * 0.5
    values = np.empty((batch_size, candidates, 8, 3), dtype=np.float32)
    shape = (batch_size, candidates, 8)
    values[..., 0] = 4.0 * time_s + rng.normal(0, 0.15, shape).cumsum(-1)
    values[..., 1] = rng.normal(0, 0.06, shape).cumsum(-1)
    values[..., 2] = rng.normal(0, 0.015, shape).cumsum(-1)
    return torch.from_numpy(values)


def _build_supervisor(cache_root: Path, backend: str, workers: int):
    return DynamicMetricSupervisor(
        {
            "backend": backend,
            "workers_per_rank": workers,
            "metric_cache_root": str(cache_root),
            "score_interval": 1,
        }
    )


def _timed_score(supervisor, tokens, proposals):
    started = time.perf_counter()
    result = supervisor.score(tokens, proposals)
    return result, time.perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric-cache-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--candidates", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if min(args.workers, args.batch_size, args.candidates, args.repeats) < 1:
        raise ValueError(
            "workers, batch size, candidates, and repeats must be positive"
        )

    cache_root = args.metric_cache_root.expanduser().resolve()
    tokens = sorted(MetricCacheLoader(cache_root).metric_cache_paths)
    required_tokens = args.batch_size * (args.repeats + 3)
    if len(tokens) < required_tokens:
        raise RuntimeError(
            f"benchmark needs {required_tokens} metric-cache tokens, got {len(tokens)}"
        )
    proposals = _proposals(args.batch_size, args.candidates).to(args.device)
    specifications = (
        ("thread-1", "thread", 1),
        (f"thread-{args.workers}", "thread", args.workers),
        (f"process-{args.workers}", "process", args.workers),
    )
    supervisors = {
        name: _build_supervisor(cache_root, backend, workers)
        for name, backend, workers in specifications
    }
    timings = {name: [] for name in supervisors}
    try:
        # Start every executor and initialize its persistent evaluator outside
        # the measured region, using a distinct token batch for each backend.
        for index, (name, _, _) in enumerate(specifications):
            begin = index * args.batch_size
            _timed_score(
                supervisors[name],
                tokens[begin : begin + args.batch_size],
                proposals,
            )

        measured_offset = len(specifications) * args.batch_size
        names = [name for name, _, _ in specifications]
        for repeat in range(args.repeats):
            begin = measured_offset + repeat * args.batch_size
            batch_tokens = tokens[begin : begin + args.batch_size]
            # Rotate the order to balance filesystem page-cache advantage.
            order = names[repeat % len(names) :] + names[: repeat % len(names)]
            reference = None
            for name in order:
                result, seconds = _timed_score(
                    supervisors[name], batch_tokens, proposals
                )
                timings[name].append(seconds)
                if reference is None:
                    reference = result
                elif not all(
                    torch.equal(reference[metric], result[metric])
                    for metric in _OUTPUT_METRICS
                ):
                    raise AssertionError(f"{name} changed a NAVSIM metric value")
    finally:
        for supervisor in supervisors.values():
            supervisor.close()

    for name in names:
        values = timings[name]
        print(f"{name}.seconds=" + ",".join(f"{value:.6f}" for value in values))
        print(f"{name}.median_seconds={float(np.median(values)):.6f}")
    baseline = float(np.median(timings["thread-1"]))
    for name in names[1:]:
        print(f"{name}.speedup={baseline / float(np.median(timings[name])):.3f}x")
    print(f"result_device={proposals.device}")
    print("metrics=bitwise_equal")


if __name__ == "__main__":
    main()
