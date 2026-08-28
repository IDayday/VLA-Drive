#!/usr/bin/env python3

"""Parallelize the untouched tail of one imbalanced metric-cache worker."""

import argparse
import re
from pathlib import Path

import ray
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from navsim.common.dataloader import SceneLoader
from navsim.common.dataclasses import SensorConfig
from navsim.planning.metric_caching.train_caching import cache_scenarios


SCENARIO_PROGRESS = re.compile(r"Processing scenario (\d+) / (\d+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("worker_log", type=Path)
    parser.add_argument("--safety-gap", type=int, default=250)
    parser.add_argument("--tokens-per-task", type=int, default=16)
    return parser.parse_args()


def latest_progress(path: Path) -> tuple[int, int]:
    progress = None
    with path.open("r", errors="replace") as stream:
        for line in stream:
            match = SCENARIO_PROGRESS.search(line)
            if match:
                progress = (int(match.group(1)), int(match.group(2)))
    if progress is None:
        raise RuntimeError(f"No progress line found in {path}")
    return progress


def main() -> None:
    args = parse_args()
    if args.safety_gap <= 0 or args.tokens_per_task <= 0:
        raise ValueError("safety gap and tokens per task must be positive")

    # This is the uniquely identified 2,376-scene static worker chunk from the
    # original one-item-per-log partition.  Preserve its exact log order.
    log_names = [
        "2021.07.09.17.06.37_veh-35_02609_05015",
        "2021.06.09.17.23.18_veh-38_02450_02515",
        "2021.09.15.14.27.22_veh-39_00038_00414",
        "2021.06.23.15.56.12_veh-16_01308_04289",
        "2021.07.16.01.22.41_veh-14_02626_04289",
        "2021.06.23.14.54.32_veh-16_00301_00410",
        "2021.06.14.11.44.56_veh-35_03389_04017",
        "2021.06.14.11.44.56_veh-35_02696_02932",
        "2021.06.09.17.37.09_veh-12_01465_01790",
    ]
    expected_counts = [714, 18, 71, 824, 447, 23, 147, 52, 80]

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

    scene_filter = instantiate(cfg.train_test_split.scene_filter)
    scene_filter.log_names = log_names
    loader = SceneLoader(
        sensor_blobs_path=None,
        data_path=Path(cfg.navsim_log_path),
        scene_filter=scene_filter,
        sensor_config=SensorConfig.build_no_sensors(),
    )
    tokens_per_log = loader.get_tokens_list_per_log()
    actual_counts = [len(tokens_per_log[name]) for name in log_names]
    if actual_counts != expected_counts:
        raise RuntimeError(
            f"Tail chunk changed: expected {expected_counts}, got {actual_counts}"
        )

    token_to_log = {
        token: log_name
        for log_name in log_names
        for token in tokens_per_log[log_name]
    }
    ordered_tokens = list(loader.scene_frames_dicts)
    if len(ordered_tokens) != sum(expected_counts):
        raise RuntimeError(
            f"Expected {sum(expected_counts)} ordered tokens, got {len(ordered_tokens)}"
        )
    if set(ordered_tokens) != set(token_to_log):
        raise RuntimeError("Combined SceneLoader token set does not match per-log token set")

    completed, total = latest_progress(args.worker_log)
    if total != len(ordered_tokens):
        raise RuntimeError(f"Worker reports {total} scenes, reconstructed {len(ordered_tokens)}")
    start = min(completed + args.safety_gap, total)
    tail_tokens = ordered_tokens[start:]
    print(
        f"original={completed}/{total}, safety_gap={args.safety_gap}, "
        f"parallel_tail_start={start + 1}, parallel_tokens={len(tail_tokens)}"
    )
    if not tail_tokens:
        return

    task_inputs = []
    for offset in range(0, len(tail_tokens), args.tokens_per_task):
        group = tail_tokens[offset : offset + args.tokens_per_task]
        task_inputs.append(
            [
                {
                    "cfg": cfg,
                    "log_file": token_to_log[token],
                    "tokens": [token],
                }
                for token in group
            ]
        )

    ray.init(address="auto", log_to_driver=False)
    remote_cache = ray.remote(num_cpus=1)(cache_scenarios)
    refs = [remote_cache.remote(group) for group in task_inputs]
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
                f"tail tasks {completed_tasks}/{len(task_inputs)}; "
                f"successes={successes}, failures={failures}"
            )
    ray.shutdown()
    if failures:
        raise SystemExit(f"Parallel tail caching had {failures} failures")


if __name__ == "__main__":
    main()
