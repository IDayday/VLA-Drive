#!/usr/bin/env python3
"""Build official NAVSIM-v2 metric targets for a candidate-scene cache.

This wrapper deliberately calls NAVSIM's unmodified ``MetricCacheProcessor``.
It adds deterministic scene selection, resumability, a portable relative index,
and provenance checks needed by the action-effect pilots.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import traceback
from typing import Any, Iterable

from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.training.experiments.cache_metadata_entry import CacheMetadataEntry, save_cache_metadata

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from navsim.common.dataclasses import Scene, SceneFilter, SensorConfig  # noqa: E402
from navsim.common.dataloader import SceneLoader  # noqa: E402
from navsim.planning.metric_caching.metric_cache_processor import MetricCacheProcessor  # noqa: E402
from navsim.planning.scenario_builder.navsim_scenario import NavSimScenario  # noqa: E402
from research.action_effect.cache_io import (  # noqa: E402
    CacheConflictError,
    CacheManifest,
    content_hash,
    file_sha256,
    finalize_manifest,
    read_manifest,
    write_json,
)


def _code_revision(paths: Iterable[Path]) -> str:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPOSITORY_ROOT, text=True).strip()
    tree = content_hash({str(path.relative_to(REPOSITORY_ROOT)): file_sha256(path) for path in paths})
    return f"{commit}+tree.{tree[:12]}"


def _infer_log_name(processed_root: Path, split: str, token: str) -> str:
    path = processed_root / "meta" / split / f"{token}.pkl"
    with path.open("rb") as stream:
        record = pickle.load(stream)
    try:
        image_path = Path(record["glo_images"]["cam_f0"]["image_paths"][0])
    except (KeyError, IndexError, TypeError) as error:
        raise KeyError(f"cannot infer raw log from processed record: {path}") from error
    return image_path.parents[1].name


def _relative_metric_path(cache_root: Path, path: Path) -> str:
    return str(path.resolve().relative_to(cache_root.resolve()))


def _build_log_group(task: dict[str, Any]) -> dict[str, Any]:
    """Worker entry point: load one raw log once and cache requested scenes."""

    log_name = str(task["log_name"])
    tokens = list(task["tokens"])
    cache_root = Path(task["cache_root"])
    raw_log_root = Path(task["raw_log_root"])
    map_root = str(task["map_root"])
    sampling = TrajectorySampling(num_poses=40, interval_length=0.1)
    processor = MetricCacheProcessor(
        cache_path=str(cache_root),
        force_feature_computation=False,
        proposal_sampling=sampling,
    )
    scene_filter = SceneFilter(
        num_history_frames=4,
        num_future_frames=10,
        frame_interval=1,
        has_route=True,
        log_names=[log_name],
        tokens=tokens,
    )
    scene_loader = SceneLoader(
        data_path=raw_log_root,
        original_sensor_path=None,
        scene_filter=scene_filter,
        sensor_config=SensorConfig.build_no_sensors(),
    )
    successes: dict[str, str] = {}
    failures: dict[str, str] = {}
    for token in tokens:
        try:
            scene_dict = scene_loader.scene_frames_dicts[token]
            scene = Scene.from_scene_dict_list(
                scene_dict,
                None,
                num_history_frames=4,
                num_future_frames=10,
                sensor_config=SensorConfig.build_no_sensors(),
            )
            scenario = NavSimScenario(scene, map_root=map_root, map_version="nuplan-maps-v1.0")
            entry = processor.compute_and_save_metric_cache(scenario)
            if entry is None:
                raise RuntimeError("official MetricCacheProcessor returned no cache metadata")
            path = Path(entry.file_name)
            successes[token] = _relative_metric_path(cache_root, path)
        except Exception:  # pragma: no cover - failures depend on external dataset
            failures[token] = traceback.format_exc()
    missing_from_log = sorted(set(tokens) - set(scene_loader.scene_frames_dicts))
    for token in missing_from_log:
        failures.setdefault(token, f"token not found in raw log {log_name}")
    return {"log_name": log_name, "successes": successes, "failures": failures}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-cache", type=Path)
    parser.add_argument("--processed-root", type=Path)
    parser.add_argument("--raw-log-root", type=Path)
    parser.add_argument("--map-root", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--split", default="train")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-scenes", type=int)
    return parser.parse_args()


def _environment_path(explicit: Path | None, variable: str, suffix: str = "") -> Path:
    if explicit is not None:
        return explicit.resolve()
    value = os.environ.get(variable, "").strip()
    if not value:
        raise ValueError(f"pass the path explicitly or source load_env.sh to set {variable}")
    return (Path(value) / suffix).resolve()


def main() -> None:
    args = parse_args()
    candidate_cache = _environment_path(
        args.candidate_cache,
        "ACTION_EFFECT_CACHE_ROOT",
        "candidates/pilot_tiny/expert",
    )
    processed_root = _environment_path(args.processed_root, "DATA_ROOT")
    raw_log_root = _environment_path(args.raw_log_root, "NAVSIM_PUBLIC_ROOT", "navsim_logs/trainval")
    map_root = _environment_path(args.map_root, "NUPLAN_MAPS_ROOT")
    cache_root = _environment_path(
        args.cache_dir,
        "ACTION_EFFECT_CACHE_ROOT",
        "metric_cache/pilot_tiny/train",
    )
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    for path, label in ((candidate_cache, "candidate cache"), (processed_root, "processed root"),
                        (raw_log_root, "raw log root"), (map_root, "map root")):
        if not path.exists():
            raise FileNotFoundError(f"{label} is missing: {path}")

    candidate_manifest = read_manifest(candidate_cache)
    if candidate_manifest is None:
        raise FileNotFoundError(f"candidate manifest is missing: {candidate_cache}")
    with (candidate_cache / "scene_index.json").open("r", encoding="utf-8") as stream:
        scene_index = json.load(stream)
    tokens = list(scene_index)
    if args.max_scenes is not None:
        if args.max_scenes < 1:
            raise ValueError("--max-scenes must be positive")
        tokens = tokens[: args.max_scenes]

    evaluator_sources = [
        REPOSITORY_ROOT / "navsim/navsim/planning/metric_caching/metric_cache_processor.py",
        REPOSITORY_ROOT / "navsim/navsim/planning/simulation/planner/pdm_planner/scoring/pdm_scorer.py",
        REPOSITORY_ROOT / "navsim/navsim/planning/simulation/planner/pdm_planner/simulation/pdm_simulator.py",
    ]
    evaluator_hash = content_hash({str(path.relative_to(REPOSITORY_ROOT)): file_sha256(path) for path in evaluator_sources})
    code_revision = _code_revision(
        [Path(__file__), REPOSITORY_ROOT / "research/action_effect/metric_cache_io.py"]
    )
    manifest = CacheManifest(
        cache_kind="navsim_v2_metric_target",
        cache_version="action_effect_metric_cache_v1",
        dataset_version=candidate_manifest.dataset_version,
        code_commit=code_revision,
        config_hash=content_hash(
            {
                "num_history_frames": 4,
                "num_future_frames": 10,
                "proposal_num_poses": 40,
                "proposal_interval_length": 0.1,
            }
        ),
        evaluator_hash=evaluator_hash,
        split=args.split,
        seed=candidate_manifest.seed,
        inputs={
            "candidate_manifest": candidate_manifest.compatibility_identity(),
            "selected_scenes_sha256": content_hash(tokens),
            "map_version": "nuplan-maps-v1.0",
        },
    )
    required = ("metric_cache_index.json", "summary.json")
    existing = read_manifest(cache_root)
    if existing is not None:
        if existing.compatibility_identity() != manifest.compatibility_identity():
            raise CacheConflictError(f"existing metric cache has a different identity: {cache_root}")
        missing = [name for name in required if not (cache_root / name).is_file()]
        if missing:
            raise CacheConflictError(f"published metric cache is incomplete ({missing}): {cache_root}")
        print(f"[action-effect] reusable metric cache: {cache_root}")
        return

    cache_root.mkdir(parents=True, exist_ok=True)
    build_identity_path = cache_root / "build_identity.json"
    identity = {"compatibility_identity": manifest.compatibility_identity(), "manifest": asdict(manifest)}
    if build_identity_path.exists():
        with build_identity_path.open("r", encoding="utf-8") as stream:
            prior_identity = json.load(stream)
        if prior_identity != identity:
            raise CacheConflictError(f"incomplete metric cache has a different identity: {cache_root}")
    else:
        populated = [path for path in cache_root.iterdir() if path.name != "build_identity.json"]
        if populated:
            raise CacheConflictError(f"unidentified files already exist in metric cache: {cache_root}")
        write_json(build_identity_path, identity)

    log_to_tokens: dict[str, list[str]] = {}
    for token in tokens:
        log_name = _infer_log_name(processed_root, args.split, token)
        if not (raw_log_root / f"{log_name}.pkl").is_file():
            raise FileNotFoundError(f"raw NAVSIM log is missing: {raw_log_root / (log_name + '.pkl')}")
        log_to_tokens.setdefault(log_name, []).append(token)

    tasks = [
        {
            "log_name": log_name,
            "tokens": grouped_tokens,
            "cache_root": str(cache_root),
            "raw_log_root": str(raw_log_root),
            "map_root": str(map_root),
        }
        for log_name, grouped_tokens in sorted(log_to_tokens.items())
    ]
    successes: dict[str, str] = {}
    failures: dict[str, str] = {}
    completed_logs = 0
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(_build_log_group, task) for task in tasks]
        for future in as_completed(futures):
            result = future.result()
            successes.update(result["successes"])
            failures.update(result["failures"])
            completed_logs += 1
            print(
                f"[action-effect] metric logs {completed_logs}/{len(tasks)}; "
                f"scenes={len(successes)}/{len(tokens)} failures={len(failures)}",
                flush=True,
            )

    successes = dict(sorted(successes.items()))
    failures = dict(sorted(failures.items()))
    write_json(cache_root / "metric_cache_index.json", successes)
    write_json(cache_root / "failures.json", failures)
    metadata_entries = [CacheMetadataEntry(cache_root / relative) for relative in successes.values()]
    save_cache_metadata(metadata_entries, cache_root, node_id=0)
    summary = {
        "requested_scene_count": len(tokens),
        "cached_scene_count": len(successes),
        "failed_scene_count": len(failures),
        "raw_log_count": len(tasks),
        "workers": args.workers,
        "proposal_sampling": {"num_poses": 40, "interval_length": 0.1},
    }
    write_json(cache_root / "summary.json", summary)
    if failures:
        raise RuntimeError(
            f"metric caching failed for {len(failures)} scenes; inspect {cache_root / 'failures.json'}"
        )
    finalize_manifest(cache_root, manifest)
    print(json.dumps({"cache_dir": str(cache_root), **summary}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
