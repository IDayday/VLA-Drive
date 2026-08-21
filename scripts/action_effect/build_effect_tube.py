#!/usr/bin/env python3
"""Build candidate-aligned replay-grounded effect-tube targets."""

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
from research.action_effect.effect_tube import (  # noqa: E402
    EFFECT_TUBE_CHANNELS,
    EffectTubeConfig,
    build_effect_tube,
    effect_map_unions,
)
from research.action_effect.metric_cache_io import load_metric_cache  # noqa: E402
from research.action_effect.probe_data import iter_jsonl  # noqa: E402


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        result = yaml.safe_load(stream)
    if not isinstance(result, dict):
        raise TypeError(f"configuration must be a mapping: {path}")
    return result


def _resolve(explicit: Path | None, suffix: str) -> Path:
    if explicit is not None:
        return explicit.resolve()
    root = os.environ.get("ACTION_EFFECT_CACHE_ROOT", "").strip()
    if not root:
        raise ValueError("pass cache paths explicitly or source load_env.sh")
    return (Path(root) / suffix).resolve()


def _process_scene(task: dict[str, Any]) -> dict[str, Any]:
    scene_id = str(task["scene_id"])
    config = EffectTubeConfig(**task["config"])
    rows = list(iter_jsonl(Path(task["consequence_jsonl"])))
    with np.load(task["consequence_npz"]) as payload:
        states = np.asarray(payload["log_replay_simulated_states"], dtype=np.float32)
    metric_cache = load_metric_cache(Path(task["metric_cache"]))
    static = effect_map_unions(metric_cache.drivable_area_map, metric_cache.route_lane_ids)
    vehicle_parameters = metric_cache.ego_state.car_footprint.vehicle_parameters
    target = np.zeros(
        (
            len(rows),
            len(config.horizons_s),
            len(EFFECT_TUBE_CHANNELS),
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
            target[index] = build_effect_tube(
                simulated_states=states[index],
                future_tracks=metric_cache.future_tracked_objects,
                static_unions=static,
                vehicle_parameters=vehicle_parameters,
                config=config,
            ).astype(np.float16)
            valid[index] = True
        except Exception:  # pragma: no cover - external geometry/data failures are cached
            failures.append(
                {
                    "candidate_id": str(row["candidate_id"]),
                    "traceback": traceback.format_exc(),
                }
            )
    write_npz(Path(task["output"]), target=target, valid=valid)
    return {
        "scene_id": scene_id,
        "candidate_count": len(rows),
        "valid_count": int(valid.sum()),
        "failure_count": len(failures),
        "failures": failures,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPOSITORY_ROOT / "configs/action_effect/effect_tube.yaml",
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
    config = EffectTubeConfig.from_mapping(raw_config)
    config.validate()
    if tuple(raw_config["channels"]) != EFFECT_TUBE_CHANNELS:
        raise ValueError("configured effect channels do not match implementation")
    if str(raw_config["provenance"]) != "log_replay":
        raise ValueError("effect tubes currently support log_replay provenance only")
    candidate_cache = _resolve(args.candidate_cache, "candidates/pilot_tiny/expert")
    consequence_cache = _resolve(args.consequence_cache, "consequences/pilot_tiny/expert")
    metric_cache = _resolve(args.metric_cache, "metric_cache/pilot_tiny/train")
    output_dir = _resolve(args.cache_dir, "effect_tube/pilot_tiny/expert_log_replay_32")
    manifests = {
        name: read_manifest(path)
        for name, path in (
            ("candidate", candidate_cache),
            ("consequence", consequence_cache),
            ("metric", metric_cache),
        )
    }
    if any(value is None for value in manifests.values()):
        raise FileNotFoundError("candidate, consequence, and metric caches must be published")
    with (consequence_cache / "scene_index.json").open("r", encoding="utf-8") as stream:
        source_index: dict[str, dict[str, int]] = json.load(stream)
    with (metric_cache / "metric_cache_index.json").open("r", encoding="utf-8") as stream:
        metric_index: dict[str, str] = json.load(stream)
    scene_ids = list(source_index)
    if args.max_scenes is not None:
        if args.max_scenes < 1:
            raise ValueError("--max-scenes must be positive")
        scene_ids = scene_ids[: args.max_scenes]
    code_files = [
        Path(__file__),
        REPOSITORY_ROOT / "research/action_effect/effect_tube.py",
        REPOSITORY_ROOT / "research/action_effect/structured_future.py",
    ]
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPOSITORY_ROOT, text=True
    ).strip()
    tree_hash = content_hash(
        {str(path.relative_to(REPOSITORY_ROOT)): file_sha256(path) for path in code_files}
    )
    candidate_manifest = manifests["candidate"]
    consequence_manifest = manifests["consequence"]
    metric_manifest = manifests["metric"]
    assert candidate_manifest is not None and consequence_manifest is not None and metric_manifest is not None
    manifest = CacheManifest(
        cache_kind="trajectory_aligned_effect_tube",
        cache_version=str(raw_config["cache_version"]),
        dataset_version=str(raw_config["dataset_version"]),
        code_commit=f"{commit}+tree.{tree_hash[:12]}",
        config_hash=content_hash(raw_config),
        evaluator_hash=metric_manifest.evaluator_hash,
        split=consequence_manifest.split,
        seed=candidate_manifest.seed,
        inputs={
            "candidate_manifest": candidate_manifest.compatibility_identity(),
            "consequence_manifest": consequence_manifest.compatibility_identity(),
            "metric_manifest": metric_manifest.compatibility_identity(),
            "selected_scenes_sha256": content_hash(scene_ids),
        },
    )
    existing = read_manifest(output_dir)
    if existing is not None:
        if existing.compatibility_identity() != manifest.compatibility_identity():
            raise CacheConflictError(f"effect-tube cache identity differs: {output_dir}")
        for name in ("scene_index.json", "summary.json"):
            if not (output_dir / name).is_file():
                raise CacheConflictError(f"published effect-tube cache lacks {name}")
        print(f"[action-effect] reusable effect-tube cache: {output_dir}")
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    identity = {"compatibility_identity": manifest.compatibility_identity(), "manifest": asdict(manifest)}
    identity_path = output_dir / "build_identity.json"
    if identity_path.is_file():
        with identity_path.open("r", encoding="utf-8") as stream:
            if json.load(stream) != identity:
                raise CacheConflictError(f"unfinished effect-tube cache identity differs: {output_dir}")
    else:
        write_json(identity_path, identity)
    scene_dir = output_dir / "scenes"
    scene_dir.mkdir(parents=True, exist_ok=True)
    tasks = []
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
        for index, task in enumerate(tasks, start=1):
            results.append(_process_scene(task))
            print(f"[effect-tube] {index}/{len(tasks)} scenes", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(_process_scene, task) for task in tasks]
            for index, future in enumerate(as_completed(futures), start=1):
                results.append(future.result())
                if index % 25 == 0 or index == len(futures):
                    print(f"[effect-tube] {index}/{len(futures)} scenes", flush=True)
    scene_index: dict[str, dict[str, Any]] = {}
    candidate_count = valid_count = 0
    target_shape: list[int] | None = None
    for number, scene_id in enumerate(scene_ids):
        path = scene_dir / f"{scene_id}.npz"
        if not path.is_file():
            raise RuntimeError(f"effect-tube scene is missing: {path}")
        with np.load(path) as payload:
            target = payload["target"]
            valid = payload["valid"]
            if len(target) != int(source_index[scene_id]["count"]):
                raise RuntimeError(f"candidate count mismatch: {path}")
            candidate_count += len(valid)
            valid_count += int(valid.sum())
            target_shape = list(target.shape[1:])
        scene_index[scene_id] = {
            "file": f"scenes/{scene_id}.npz",
            "scene_number": number,
            "candidate_count": int(source_index[scene_id]["count"]),
        }
    failures = sum(row["failure_count"] for row in results)
    summary = {
        "scene_count": len(scene_ids),
        "candidate_count": candidate_count,
        "valid_count": valid_count,
        "valid_rate": valid_count / max(candidate_count, 1),
        "new_failure_count": failures,
        "target_shape_per_candidate": target_shape,
        "horizons_s": list(config.horizons_s),
        "channels": list(EFFECT_TUBE_CHANNELS),
        "provenance": "log_replay",
        "coordinate_frame": "candidate_rear_axle_heading_aligned",
    }
    write_json(output_dir / "scene_index.json", scene_index)
    write_json(output_dir / "summary.json", summary)
    write_json(output_dir / "worker_results.json", sorted(results, key=lambda row: row["scene_id"]))
    finalize_manifest(output_dir, manifest)
    print(json.dumps({"cache_dir": str(output_dir), **summary}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
