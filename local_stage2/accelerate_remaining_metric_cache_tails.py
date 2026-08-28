#!/usr/bin/env python3

"""Safely parallelize uniquely identifiable metric-cache worker tails."""

import argparse
import glob
import re
from collections import Counter
from pathlib import Path

import numpy as np
import ray
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from navsim.common.dataloader import SceneLoader
from navsim.common.dataclasses import SensorConfig
from navsim.planning.metric_caching.train_caching import cache_scenarios


PROGRESS = re.compile(r"Processing scenario (\d+) / (\d+)")


def latest_progress(path: Path):
    result = None
    with path.open("r", errors="replace") as stream:
        for line in stream:
            match = PROGRESS.search(line)
            if match:
                result = int(match.group(1)), int(match.group(2))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--worker-log-glob",
        default="/tmp/ray/session_latest/logs/worker-*-2177*.err",
    )
    parser.add_argument("--safety-gap", type=int, default=150)
    parser.add_argument("--tokens-per-task", type=int, default=16)
    parser.add_argument("--exclude-total", type=int, action="append", default=[2376])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.safety_gap <= 0 or args.tokens_per_task <= 0:
        raise ValueError("safety gap and tokens per task must be positive")

    config_dir = Path(__file__).parents[1] / "navsim/planning/script/config/metric_caching"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(
            config_name="train_metric_caching",
            overrides=[
                "train_test_split=navtrain",
                "cache.cache_path=/mnt/project/DriveVLA-M0-stage2/cache/metric_cache_navtrain_full",
                "navsim_log_path=/mnt/project/DriveDreamer-Policy/navsim_raw/navsim_logs/trainval",
                "worker=ray_distributed_no_torch",
                "worker.threads_per_node=128",
                "worker.log_to_driver=false",
            ],
        )

    full_loader = SceneLoader(
        sensor_blobs_path=None,
        data_path=Path(cfg.navsim_log_path),
        scene_filter=instantiate(cfg.train_test_split.scene_filter),
        sensor_config=SensorConfig.build_no_sensors(),
    )
    points = [
        {"cfg": cfg, "log_file": log_name, "tokens": tokens}
        for log_name, tokens in full_loader.get_tokens_list_per_log().items()
    ]
    chunks = [chunk.tolist() for chunk in np.array_split(points, 128) if len(chunk)]
    chunk_sizes = [sum(len(point["tokens"]) for point in chunk) for chunk in chunks]
    size_frequency = Counter(chunk_sizes)
    unique_chunk_by_size = {
        size: chunks[index]
        for index, size in enumerate(chunk_sizes)
        if size_frequency[size] == 1
    }

    candidates = []
    for name in glob.glob(args.worker_log_glob):
        path = Path(name)
        progress = latest_progress(path)
        if progress is None:
            continue
        completed, total = progress
        if completed >= total or total in args.exclude_total:
            continue
        if total not in unique_chunk_by_size:
            print(f"SKIP non-unique worker total {completed}/{total}: {path.name}")
            continue
        candidates.append((path, unique_chunk_by_size[total]))

    all_task_inputs = []
    for worker_log, chunk in candidates:
        log_names = [point["log_file"] for point in chunk]
        flat_tokens = [token for point in chunk for token in point["tokens"]]
        token_to_log = {
            token: point["log_file"] for point in chunk for token in point["tokens"]
        }
        scene_filter = instantiate(cfg.train_test_split.scene_filter)
        scene_filter.log_names = log_names
        scene_filter.tokens = flat_tokens
        loader = SceneLoader(
            sensor_blobs_path=None,
            data_path=Path(cfg.navsim_log_path),
            scene_filter=scene_filter,
            sensor_config=SensorConfig.build_no_sensors(),
        )
        ordered_tokens = list(loader.scene_frames_dicts)
        if len(ordered_tokens) != len(flat_tokens) or set(ordered_tokens) != set(flat_tokens):
            raise RuntimeError(f"Could not reconstruct exact worker order for {worker_log}")

        completed, total = latest_progress(worker_log)
        start = min(completed + args.safety_gap, total)
        tail = ordered_tokens[start:]
        print(
            f"{worker_log.name}: original={completed}/{total}, "
            f"tail_start={start + 1}, parallel_tokens={len(tail)}"
        )
        for offset in range(0, len(tail), args.tokens_per_task):
            group = tail[offset : offset + args.tokens_per_task]
            all_task_inputs.append(
                [
                    {"cfg": cfg, "log_file": token_to_log[token], "tokens": [token]}
                    for token in group
                ]
            )

    if not all_task_inputs:
        print("No uniquely identifiable tails require acceleration")
        return

    ray.init(address="auto", log_to_driver=False)
    remote_cache = ray.remote(num_cpus=1)(cache_scenarios)
    refs = [remote_cache.remote(group) for group in all_task_inputs]
    completed_tasks = 0
    successes = 0
    failures = 0
    while refs:
        ready, refs = ray.wait(refs, num_returns=1)
        for result in ray.get(ready[0]):
            successes += result.successes
            failures += result.failures
        completed_tasks += 1
        if completed_tasks % 10 == 0 or not refs:
            print(
                f"tail tasks {completed_tasks}/{len(all_task_inputs)}; "
                f"successes={successes}, failures={failures}"
            )
    ray.shutdown()
    if failures:
        raise SystemExit(f"Parallel tail caching had {failures} failures")


if __name__ == "__main__":
    main()
