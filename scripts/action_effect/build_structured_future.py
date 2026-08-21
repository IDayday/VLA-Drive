#!/usr/bin/env python3
"""Build replay-grounded trajectory-aligned 1/2/4-second future tubes."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
from typing import Any

import numpy as np
import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from research.action_effect.cache_io import (  # noqa: E402
    CacheConflictError,
    CacheManifest,
    content_hash,
    file_sha256,
    finalize_manifest,
    read_manifest,
    write_json,
    write_npz,
)
from research.action_effect.metric_cache_io import load_metric_cache  # noqa: E402
from research.action_effect.probe_data import iter_jsonl  # noqa: E402
from research.action_effect.structured_future import (  # noqa: E402
    STRUCTURED_CHANNELS,
    FutureTubeConfig,
    build_future_tube,
    map_unions,
)


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise TypeError(f"configuration must be a mapping: {path}")
    return value


def _environment_cache(explicit: Path | None, suffix: str) -> Path:
    if explicit is not None:
        return explicit.resolve()
    root = os.environ.get("ACTION_EFFECT_CACHE_ROOT", "").strip()
    if not root:
        raise ValueError("pass cache paths explicitly or source load_env.sh")
    return (Path(root) / suffix).resolve()


def _process_scene(task: dict[str, Any]) -> dict[str, Any]:
    scene_id = str(task["scene_id"])
    config = FutureTubeConfig(**task["config"])
    rows = list(iter_jsonl(Path(task["consequence_jsonl"])))
    with np.load(task["consequence_npz"]) as payload:
        states = np.asarray(payload["log_replay_simulated_states"], dtype=np.float32)
    metric_cache = load_metric_cache(Path(task["metric_cache"]))
    static = map_unions(metric_cache.drivable_area_map, metric_cache.route_lane_ids)
    target = np.zeros(
        (
            len(rows),
            len(config.horizons_s),
            len(STRUCTURED_CHANNELS),
            config.resolution,
            config.resolution,
        ),
        dtype=np.float16,
    )
    valid = np.zeros(len(rows), dtype=bool)
    failures: list[dict[str, str]] = []
    for index, row in enumerate(rows):
        if not row.get("candidate_accepted") or not row["log_replay"].get("available"):
            continue
        try:
            target[index] = build_future_tube(
                simulated_states=states[index],
                future_tracks=metric_cache.future_tracked_objects,
                static_unions=static,
                config=config,
            ).astype(np.float16)
            valid[index] = True
        except Exception:  # pragma: no cover - external geometry/data errors are cached
            failures.append(
                {
                    "candidate_id": str(row["candidate_id"]),
                    "traceback": traceback.format_exc(),
                }
            )
    output = Path(task["output"])
    write_npz(output, target=target, valid=valid)
    channel_means = (
        target[valid].astype(np.float64).mean(axis=(0, 1, 3, 4)).tolist()
        if valid.any()
        else [float("nan")] * len(STRUCTURED_CHANNELS)
    )
    return {
        "scene_id": scene_id,
        "candidate_count": len(rows),
        "valid_count": int(valid.sum()),
        "failure_count": len(failures),
        "failures": failures,
        "channel_means": channel_means,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPOSITORY_ROOT / "configs/action_effect/structured_future.yaml",
    )
    parser.add_argument("--candidate-cache", type=Path)
    parser.add_argument("--consequence-cache", type=Path)
    parser.add_argument("--metric-cache", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-scenes", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    config_path = args.config.resolve()
    raw_config = _load_yaml(config_path)
    config = FutureTubeConfig.from_mapping(raw_config)
    config.validate()
    if tuple(raw_config["channels"]) != STRUCTURED_CHANNELS:
        raise ValueError("configured structured channels do not match implementation")
    if str(raw_config["provenance"]) != "log_replay":
        raise ValueError("Phase 5 structured future currently supports log_replay only")
    candidate_cache = _environment_cache(
        args.candidate_cache, "candidates/pilot_tiny/expert"
    )
    consequence_cache = _environment_cache(
        args.consequence_cache, "consequences/pilot_tiny/expert"
    )
    metric_cache = _environment_cache(
        args.metric_cache, "metric_cache/pilot_tiny/train"
    )
    output_dir = _environment_cache(
        args.cache_dir, "structured_future/pilot_tiny/expert_log_replay_32"
    )
    manifests = {
        name: read_manifest(path)
        for name, path in (
            ("candidate", candidate_cache),
            ("consequence", consequence_cache),
            ("metric", metric_cache),
        )
    }
    missing = [name for name, value in manifests.items() if value is None]
    if missing:
        raise FileNotFoundError(f"source caches are unpublished: {missing}")
    with (consequence_cache / "scene_index.json").open("r", encoding="utf-8") as stream:
        source_scene_index: dict[str, dict[str, int]] = json.load(stream)
    with (metric_cache / "metric_cache_index.json").open("r", encoding="utf-8") as stream:
        metric_index: dict[str, str] = json.load(stream)
    scene_ids = list(source_scene_index)
    if args.max_scenes is not None:
        if args.max_scenes < 1:
            raise ValueError("--max-scenes must be positive")
        scene_ids = scene_ids[: args.max_scenes]
    code_files = [
        Path(__file__),
        REPOSITORY_ROOT / "research/action_effect/structured_future.py",
        REPOSITORY_ROOT / "research/action_effect/cache_io.py",
    ]
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPOSITORY_ROOT, text=True
    ).strip()
    tree_hash = content_hash(
        {str(path.relative_to(REPOSITORY_ROOT)): file_sha256(path) for path in code_files}
    )
    manifest = CacheManifest(
        cache_kind="trajectory_aligned_structured_future",
        cache_version=str(raw_config["cache_version"]),
        dataset_version=str(raw_config["dataset_version"]),
        code_commit=f"{commit}+tree.{tree_hash[:12]}",
        config_hash=content_hash(raw_config),
        evaluator_hash=manifests["metric"].evaluator_hash,
        split=manifests["consequence"].split,
        seed=manifests["candidate"].seed,
        inputs={
            f"{name}_manifest": value.compatibility_identity()
            for name, value in manifests.items()
        }
        | {"selected_scenes_sha256": content_hash(scene_ids)},
    )
    existing = read_manifest(output_dir)
    if existing is not None:
        if existing.compatibility_identity() != manifest.compatibility_identity():
            raise CacheConflictError(f"structured-future cache identity differs: {output_dir}")
        required = ("scene_index.json", "summary.json")
        if any(not (output_dir / name).is_file() for name in required):
            raise CacheConflictError("published structured-future cache is incomplete")
        print(f"[action-effect] reusable structured-future cache: {output_dir}")
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    identity_path = output_dir / "build_identity.json"
    identity = {"compatibility_identity": manifest.compatibility_identity(), "manifest": asdict(manifest)}
    if identity_path.is_file():
        with identity_path.open("r", encoding="utf-8") as stream:
            if json.load(stream) != identity:
                raise CacheConflictError(f"unfinished structured-future cache identity differs: {output_dir}")
    else:
        write_json(identity_path, identity)
    scene_dir = output_dir / "scenes"
    scene_dir.mkdir(parents=True, exist_ok=True)
    tasks: list[dict[str, Any]] = []
    for scene_id in scene_ids:
        output = scene_dir / f"{scene_id}.npz"
        if output.is_file():
            continue
        tasks.append(
            {
                "scene_id": scene_id,
                "config": asdict(config),
                "consequence_jsonl": str(consequence_cache / "scenes" / f"{scene_id}.jsonl"),
                "consequence_npz": str(consequence_cache / "scenes" / f"{scene_id}.npz"),
                "metric_cache": str(metric_cache / metric_index[scene_id]),
                "output": str(output),
            }
        )
    results: list[dict[str, Any]] = []
    if args.workers == 1:
        for task in tasks:
            result = _process_scene(task)
            results.append(result)
            print(f"[structured-future] {len(results)}/{len(tasks)} {result['scene_id']}")
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(_process_scene, task): task["scene_id"] for task in tasks}
            for index, future in enumerate(as_completed(futures), start=1):
                result = future.result()
                results.append(result)
                if index % 25 == 0 or index == len(futures):
                    print(f"[structured-future] {index}/{len(futures)} scenes")
    # Re-open every scene so resumed and newly computed outputs receive the
    # same completeness validation and deterministic index.
    scene_index: dict[str, dict[str, Any]] = {}
    valid_count = 0
    candidate_count = 0
    target_shape: list[int] | None = None
    for scene_number, scene_id in enumerate(scene_ids):
        path = scene_dir / f"{scene_id}.npz"
        if not path.is_file():
            raise RuntimeError(f"structured-future scene is missing: {path}")
        with np.load(path) as payload:
            target = payload["target"]
            valid = payload["valid"]
            if target.shape[0] != source_scene_index[scene_id]["count"]:
                raise RuntimeError(f"candidate count mismatch in {path}")
            target_shape = list(target.shape[1:])
            valid_count += int(valid.sum())
            candidate_count += len(valid)
        scene_index[scene_id] = {
            "file": f"scenes/{scene_id}.npz",
            "scene_number": scene_number,
            "candidate_count": int(source_scene_index[scene_id]["count"]),
        }
    write_json(output_dir / "scene_index.json", scene_index)
    failure_count = sum(result["failure_count"] for result in results)
    summary = {
        "scene_count": len(scene_ids),
        "candidate_count": candidate_count,
        "valid_count": valid_count,
        "valid_rate": valid_count / max(candidate_count, 1),
        "new_failure_count": failure_count,
        "target_shape_per_candidate": target_shape,
        "horizons_s": list(config.horizons_s),
        "channels": list(STRUCTURED_CHANNELS),
        "provenance": "log_replay",
        "coordinate_frame": "candidate_rear_axle_heading_aligned",
    }
    write_json(output_dir / "summary.json", summary)
    write_json(output_dir / "worker_results.json", sorted(results, key=lambda row: row["scene_id"]))
    finalize_manifest(output_dir, manifest)
    print(json.dumps({"cache_dir": str(output_dir), **summary}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
