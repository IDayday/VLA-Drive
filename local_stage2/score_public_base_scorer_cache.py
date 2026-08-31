"""Attach offline PDM factor supervision to an inference-only scorer cache.

The source cache contains only tensors available at inference.  This script
reads immutable training MetricCache files in separate CPU workers and writes a
parallel label tree.  Keeping the trees separate makes score leakage auditable:
deployment code consumes the source cache/model inputs, never this label tree.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import lzma
import multiprocessing as mp
import os
import pickle
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from navsim.agents.EpisodeDrive.score_module.compute_navsim_score import (
    config as pdm_scorer_config,
    get_sub_score_from_metric_cache,
    proposal_sampling,
)
from navsim.agents.EpisodeDrive.score_module.train_pdm_scorer import PDMScorer
from navsim.common.dataloader import MetricCacheLoader
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import (
    PDMSimulator,
)


TARGET_FACTOR_KEYS = (
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "ego_progress",
    "time_to_collision_within_bound",
    "comfort",
    "driving_direction_compliance",
    "score",
)

_METRIC_PATHS: Dict[str, str] = {}
_SIMULATOR: Optional[PDMSimulator] = None
_SCORER: Optional[PDMScorer] = None


def _atomic_torch_save(payload: Dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json_dump(payload: Dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _worker_init(metric_cache_path: str) -> None:
    global _METRIC_PATHS, _SIMULATOR, _SCORER
    torch.set_num_threads(1)
    loader = MetricCacheLoader(Path(metric_cache_path))
    _METRIC_PATHS = {
        str(token): str(path) for token, path in loader.metric_cache_paths.items()
    }
    _SIMULATOR = PDMSimulator(proposal_sampling)
    _SCORER = PDMScorer(proposal_sampling, pdm_scorer_config)


def _score_one(token: str, proposals: np.ndarray) -> np.ndarray:
    if _SIMULATOR is None or _SCORER is None:
        raise RuntimeError("PDM worker was not initialized")
    path = _METRIC_PATHS.get(token)
    if path is None:
        raise KeyError(f"Metric cache missing token {token}")
    with lzma.open(path, "rb") as file:
        metric_cache = pickle.load(file)
    factors, *_ = get_sub_score_from_metric_cache(
        metric_cache,
        proposals,
        True,
        simulator_instance=_SIMULATOR,
        scorer_instance=_SCORER,
    )
    factors = np.asarray(factors, dtype=np.float32)
    if factors.shape != (len(proposals), len(TARGET_FACTOR_KEYS)):
        raise RuntimeError(f"Unexpected PDM factor shape for {token}: {factors.shape}")
    if not np.isfinite(factors).all():
        raise RuntimeError(f"Non-finite PDM factor for {token}")
    return factors


def _score_chunk(task: Tuple[str, str, str]) -> Dict[str, object]:
    source_path = Path(task[0])
    output_path = Path(task[1])
    relative_path = task[2]
    if output_path.is_file():
        return {"relative_path": relative_path, "status": "already_complete"}
    try:
        source = torch.load(source_path, map_location="cpu")
        tokens = [str(token) for token in source["tokens"]]
        proposals = source["proposals"].float().numpy()
        if proposals.shape[:2] != (len(tokens), 64):
            raise RuntimeError(
                f"Unexpected source shape {proposals.shape} for {source_path}"
            )
        target_factors = np.stack(
            [_score_one(token, proposals[index]) for index, token in enumerate(tokens)]
        )
        payload = {
            "schema_version": 1,
            "source_relative_path": relative_path,
            "tokens": tokens,
            "log_names": [str(value) for value in source["log_names"]],
            "target_factor_keys": TARGET_FACTOR_KEYS,
            "target_factors": torch.from_numpy(target_factors),
            "valid_mask": torch.ones(len(tokens), dtype=torch.bool),
        }
        _atomic_torch_save(payload, output_path)
        return {
            "relative_path": relative_path,
            "status": "scored",
            "scene_count": len(tokens),
        }
    except Exception as error:
        return {
            "relative_path": relative_path,
            "status": "failed",
            "error": repr(error),
            "traceback": traceback.format_exc(),
        }


def _belongs_to_worker(relative_path: str, count: int, index: int) -> bool:
    digest = hashlib.sha256(relative_path.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % count == index


def _discover_tasks(
    source_root: Path,
    output_root: Path,
    worker_shard_count: int,
    worker_shard_index: int,
) -> List[Tuple[str, str, str]]:
    tasks: List[Tuple[str, str, str]] = []
    for source_path in sorted(source_root.glob("*_shard_*-of-*/chunk_*.pt")):
        relative = str(source_path.relative_to(source_root))
        if not _belongs_to_worker(relative, worker_shard_count, worker_shard_index):
            continue
        output_path = output_root / relative
        if not output_path.is_file():
            tasks.append((str(source_path), str(output_path), relative))
    return tasks


def _source_complete(source_root: Path) -> bool:
    shard_dirs = sorted(source_root.glob("*_shard_*-of-*"))
    if not shard_dirs or not all((directory / "manifest.json").is_file() for directory in shard_dirs):
        return False
    manifests = [json.loads((directory / "manifest.json").read_text()) for directory in shard_dirs]
    expected_shards = {int(value["shard_index"]) for value in manifests}
    shard_count = int(manifests[0]["shard_count"])
    return expected_shards == set(range(shard_count))


def _worker_complete(
    source_root: Path,
    output_root: Path,
    worker_shard_count: int,
    worker_shard_index: int,
) -> bool:
    for source_path in source_root.glob("*_shard_*-of-*/chunk_*.pt"):
        relative = str(source_path.relative_to(source_root))
        if _belongs_to_worker(relative, worker_shard_count, worker_shard_index):
            if not (output_root / relative).is_file():
                return False
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--metric-cache", type=Path, required=True)
    parser.add_argument("--num-workers", type=int, default=32)
    parser.add_argument("--worker-shard-count", type=int, default=1)
    parser.add_argument("--worker-shard-index", type=int, default=0)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.source_root.is_dir():
        raise FileNotFoundError(args.source_root)
    if not args.metric_cache.is_dir():
        raise FileNotFoundError(args.metric_cache)
    if args.num_workers <= 0:
        raise ValueError("num_workers must be positive")
    if not 0 <= args.worker_shard_index < args.worker_shard_count:
        raise ValueError("worker shard index is outside worker shard count")
    args.output_root.mkdir(parents=True, exist_ok=True)

    processed = 0
    failed: List[Dict[str, object]] = []
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=args.num_workers,
        mp_context=context,
        initializer=_worker_init,
        initargs=(str(args.metric_cache),),
    ) as pool:
        while True:
            tasks = _discover_tasks(
                args.source_root,
                args.output_root,
                args.worker_shard_count,
                args.worker_shard_index,
            )
            if tasks:
                futures = [pool.submit(_score_chunk, task) for task in tasks]
                for future in as_completed(futures):
                    result = future.result()
                    processed += 1
                    if result["status"] == "failed":
                        failed.append(result)
                        print(json.dumps(result, sort_keys=True), flush=True)
                    else:
                        print(
                            "PDM_LABEL "
                            f"worker_shard={args.worker_shard_index}/{args.worker_shard_count} "
                            f"chunks={processed} result={json.dumps(result, sort_keys=True)}",
                            flush=True,
                        )
                if failed:
                    break

            source_done = _source_complete(args.source_root)
            worker_done = _worker_complete(
                args.source_root,
                args.output_root,
                args.worker_shard_count,
                args.worker_shard_index,
            )
            if source_done and worker_done:
                break
            if not args.watch:
                break
            time.sleep(args.poll_seconds)

    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": str(args.source_root.resolve()),
        "metric_cache": str(args.metric_cache.resolve()),
        "target_factor_keys": TARGET_FACTOR_KEYS,
        "worker_shard_count": args.worker_shard_count,
        "worker_shard_index": args.worker_shard_index,
        "processed_chunks_this_run": processed,
        "failed_chunk_count": len(failed),
        "source_complete": _source_complete(args.source_root),
        "worker_complete": _worker_complete(
            args.source_root,
            args.output_root,
            args.worker_shard_count,
            args.worker_shard_index,
        ),
        "offline_training_labels_only": True,
    }
    _atomic_json_dump(
        manifest,
        args.output_root
        / f"worker_manifest_{args.worker_shard_index:03d}-of-{args.worker_shard_count:03d}.json",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    if failed:
        raise RuntimeError(f"{len(failed)} PDM label chunks failed")


if __name__ == "__main__":
    main()
