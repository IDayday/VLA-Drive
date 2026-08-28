#!/usr/bin/env python3
"""Verify exact and timed PDM scoring with a spawn-based process pool."""

from __future__ import annotations

import glob
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch

from navsim.agents.EpisodeDrive.score_module.compute_navsim_score import get_sub_score


def main() -> None:
    cache_root = os.environ.get(
        "NAVSIM_TRAIN_METRIC_CACHE",
        "/mnt/project/DriveVLA-M0-stage2/cache/metric_cache_navtrain_full",
    )
    files = glob.glob(os.path.join(cache_root, "*", "*", "*", "metric_cache.pkl"))[:8]
    if len(files) < 2:
        raise RuntimeError(f"Need at least two metric-cache files under {cache_root}")

    # Reproduce the important failure mode: create a CUDA context before the CPU
    # workers.  The spawn context keeps workers independent from that context.
    cuda_probe = torch.ones(1, device="cuda:0")
    assert cuda_probe.item() == 1

    poses = np.zeros((64, 8, 3), dtype=np.float32)
    poses[:, :, 0] = np.arange(1, 9, dtype=np.float32)[None, :]
    poses[:, :, 1] = np.linspace(-2, 2, 64, dtype=np.float32)[:, None]
    args = [(path, poses, False) for path in files]

    start = time.perf_counter()
    sequential = [get_sub_score(*item) for item in args]
    sequential_seconds = time.perf_counter() - start

    context = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=2, mp_context=context) as executor:
        # The first pass includes worker import/startup and is intentionally not
        # used for steady-state throughput.
        list(
            executor.map(
                get_sub_score,
                [item[0] for item in args[:2]],
                [item[1] for item in args[:2]],
                [item[2] for item in args[:2]],
            )
        )
        start = time.perf_counter()
        parallel = list(
            executor.map(
                get_sub_score,
                [item[0] for item in args],
                [item[1] for item in args],
                [item[2] for item in args],
            )
        )
        parallel_seconds = time.perf_counter() - start

    for sequential_item, parallel_item in zip(sequential, parallel):
        for sequential_array, parallel_array in zip(sequential_item, parallel_item):
            np.testing.assert_array_equal(sequential_array, parallel_array)

    print(f"samples={len(args)}")
    print(f"sequential_seconds={sequential_seconds:.6f}")
    print(f"parallel_seconds={parallel_seconds:.6f}")
    print(f"speedup={sequential_seconds / parallel_seconds:.6f}")
    print("exact=True")


if __name__ == "__main__":
    main()
