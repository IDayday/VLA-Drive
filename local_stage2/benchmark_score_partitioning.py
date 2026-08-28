#!/usr/bin/env python3
"""Benchmark exact proposal partitioning for the training-time PDM scorer."""

from __future__ import annotations

import glob
import math
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from navsim.agents.EpisodeDrive.score_module.compute_navsim_score import get_sub_score


def score_partition(path, poses, test, partition_index):
    return partition_index, get_sub_score(path, poses, test)


def merge_partitions(parts):
    ordered = [result for _, result in sorted(parts, key=lambda item: item[0])]
    return tuple(np.concatenate(items, axis=0) for items in zip(*ordered))


def assert_exact(left, right):
    for left_array, right_array in zip(left, right):
        np.testing.assert_array_equal(left_array, right_array)


def main() -> None:
    cache_root = os.environ.get(
        "NAVSIM_TRAIN_METRIC_CACHE",
        "/mnt/project/DriveVLA-M0-stage2/cache/metric_cache_navtrain_full",
    )
    files = glob.glob(os.path.join(cache_root, "*", "*", "*", "metric_cache.pkl"))[:8]
    if len(files) < 8:
        raise RuntimeError(f"Need eight metric caches under {cache_root}")

    poses = np.zeros((64, 8, 3), dtype=np.float32)
    poses[:, :, 0] = np.arange(1, 9, dtype=np.float32)[None, :]
    poses[:, :, 1] = np.linspace(-2, 2, 64, dtype=np.float32)[:, None]
    batches = [(files[index:index + 2], poses) for index in range(0, 8, 2)]

    sequential_results = []
    start = time.perf_counter()
    for batch_files, batch_poses in batches:
        sequential_results.append(
            [get_sub_score(path, batch_poses, False) for path in batch_files]
        )
    sequential_seconds = time.perf_counter() - start

    start_method = os.environ.get("DRIVEVLA_SCORE_START_METHOD", "spawn")
    if start_method == "forkserver":
        mp.set_forkserver_preload(
            ["navsim.agents.EpisodeDrive.score_module.compute_navsim_score"]
        )
    process_count = int(os.environ.get("DRIVEVLA_SCORE_PROCESSES", "8"))
    if process_count <= 0:
        raise ValueError("DRIVEVLA_SCORE_PROCESSES must be positive")
    partition_counts = tuple(
        int(value)
        for value in os.environ.get(
            "DRIVEVLA_SCORE_BENCH_PARTITIONS", "2,4,8,16"
        ).split(",")
    )
    if any(count <= 0 or 64 % count for count in partition_counts):
        raise ValueError("Every benchmark partition count must divide 64")

    context = mp.get_context(start_method)
    with ProcessPoolExecutor(
        max_workers=process_count, mp_context=context
    ) as executor:
        # Fully initialize all workers before measuring steady state.
        warmup_files = (files * math.ceil(process_count / len(files)))[
            :process_count
        ]
        list(
            executor.map(
                get_sub_score,
                warmup_files,
                [poses] * len(warmup_files),
                [False] * len(warmup_files),
            )
        )

        full_results = []
        full_batch_seconds = []
        for batch_files, batch_poses in batches:
            start = time.perf_counter()
            result = list(
                executor.map(
                    get_sub_score,
                    batch_files,
                    [batch_poses] * len(batch_files),
                    [False] * len(batch_files),
                )
            )
            full_batch_seconds.append(time.perf_counter() - start)
            full_results.append(result)

        all_partitioned_results = {}
        all_partitioned_batch_seconds = {}
        for partition_count in partition_counts:
            partitioned_results = []
            partitioned_batch_seconds = []
            for batch_files, batch_poses in batches:
                tasks = []
                for scene_index, path in enumerate(batch_files):
                    for partition_index, partition in enumerate(
                        np.split(batch_poses, partition_count)
                    ):
                        tasks.append((scene_index, partition_index, path, partition))

                start = time.perf_counter()
                futures = [
                    executor.submit(
                        score_partition,
                        path,
                        partition,
                        False,
                        partition_index,
                    )
                    for _, partition_index, path, partition in tasks
                ]
                task_results = [future.result() for future in futures]
                partitioned_batch_seconds.append(time.perf_counter() - start)
                partitioned_results.append(
                    [
                        merge_partitions(
                            task_results[
                                scene_index * partition_count:
                                (scene_index + 1) * partition_count
                            ]
                        )
                        for scene_index in range(2)
                    ]
                )
            all_partitioned_results[partition_count] = partitioned_results
            all_partitioned_batch_seconds[partition_count] = partitioned_batch_seconds

    for partition_count, partitioned_results in all_partitioned_results.items():
        for expected_batch, full_batch, partitioned_batch in zip(
            sequential_results, full_results, partitioned_results
        ):
            for expected, full, partitioned in zip(
                expected_batch, full_batch, partitioned_batch
            ):
                assert_exact(expected, full)
                assert_exact(expected, partitioned)

    print(f"batches={len(batches)}")
    print(f"start_method={start_method}")
    print(f"processes={process_count}")
    print(f"sequential_total_seconds={sequential_seconds:.6f}")
    print(f"full_scene_batch_seconds={full_batch_seconds}")
    print(f"full_scene_mean_seconds={np.mean(full_batch_seconds):.6f}")
    for partition_count, batch_seconds in all_partitioned_batch_seconds.items():
        print(
            f"partitions={partition_count} mean_seconds={np.mean(batch_seconds):.6f} "
            f"speedup={np.mean(full_batch_seconds) / np.mean(batch_seconds):.6f}"
        )
    print("exact=True")


if __name__ == "__main__":
    main()
