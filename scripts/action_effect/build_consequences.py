#!/usr/bin/env python3
"""Build exact, log-replay, and optional NAVSIM-v2 IDM consequences."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
from typing import Any, Iterable

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
    write_jsonl,
    write_npz,
)
from research.action_effect.consequence_builder import (  # noqa: E402
    ConsequenceConfig,
    build_log_replay_policy,
    build_reactive_policy,
    exact_map_consequence,
    make_scorer,
    physical_kinematics,
    score_under_assumption,
)
from research.action_effect.metric_cache_io import (  # noqa: E402
    iter_jsonl,
    load_metric_cache,
    load_relative_metric_cache_index,
)


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise TypeError(f"configuration must be a mapping: {path}")
    return value


def _code_revision(paths: Iterable[Path]) -> str:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPOSITORY_ROOT, text=True).strip()
    tree = content_hash({str(path.relative_to(REPOSITORY_ROOT)): file_sha256(path) for path in paths})
    return f"{commit}+tree.{tree[:12]}"


def _stable_reactive_subset(scene_ids: list[str], maximum: int, seed: int) -> set[str]:
    ranked = sorted(
        scene_ids,
        key=lambda token: hashlib.sha256(f"{seed}:{token}".encode("utf-8")).hexdigest(),
    )
    return set(ranked[: min(maximum, len(ranked))])


def _empty_assumption(provenance: str, reason: str) -> dict[str, Any]:
    return {"available": False, "provenance": provenance, "reason": reason}


def _process_scene(task: dict[str, Any]) -> dict[str, Any]:
    scene_id = str(task["scene_id"])
    rows = list(task["candidate_rows"])
    trajectories = np.asarray(task["trajectories"], dtype=np.float32)
    metric_cache = load_metric_cache(Path(task["metric_cache_path"]))
    output_jsonl = Path(task["output_jsonl"])
    output_npz = Path(task["output_npz"])
    config = ConsequenceConfig(**task["consequence_config"])
    require_kinematic = bool(task["require_kinematic"])
    require_route = bool(task["require_route"])
    reactive_scene = bool(task["reactive_scene"])
    map_root = str(task["map_root"])
    proposal_steps = config.proposal_num_poses + 1
    state_size = 11
    count = len(rows)

    array_names = (
        "simulated_states",
        "minimum_clearance_t",
        "minimum_dynamic_clearance_t",
        "time_indexed_overlap",
        "time_indexed_dynamic_overlap",
        "dynamic_agents_within_radius_t",
    )
    arrays: dict[str, np.ndarray] = {}
    for provenance in ("log_replay", "reactive_model"):
        arrays[f"{provenance}_simulated_states"] = np.full(
            (count, proposal_steps, state_size), np.nan, dtype=np.float32
        )
        arrays[f"{provenance}_minimum_clearance_t"] = np.full(
            (count, proposal_steps), np.nan, dtype=np.float32
        )
        arrays[f"{provenance}_minimum_dynamic_clearance_t"] = np.full(
            (count, proposal_steps), np.nan, dtype=np.float32
        )
        arrays[f"{provenance}_time_indexed_overlap"] = np.zeros(
            (count, proposal_steps), dtype=np.bool_
        )
        arrays[f"{provenance}_time_indexed_dynamic_overlap"] = np.zeros(
            (count, proposal_steps), dtype=np.bool_
        )
        arrays[f"{provenance}_dynamic_agents_within_radius_t"] = np.zeros(
            (count, proposal_steps), dtype=np.int16
        )

    simulator, scorer = make_scorer(config)
    log_policy = build_log_replay_policy(config)
    reactive_policy = build_reactive_policy(config, map_root) if reactive_scene else None
    output_rows: list[dict[str, Any]] = []
    candidate_failures = 0

    for local_index, (metadata, trajectory) in enumerate(zip(rows, trajectories)):
        accepted = bool(
            (metadata["kinematic_valid"] or not require_kinematic)
            and (metadata["route_valid"] or not require_route)
        )
        base = {
            "scene_id": scene_id,
            "candidate_id": metadata["candidate_id"],
            "candidate_index": int(metadata["trajectory"]["index"]),
            "scene_candidate_index": local_index,
            "anchor_type": metadata["anchor_type"],
            "perturbation_type": metadata["perturbation_type"],
            "perturbation_parameters": metadata["perturbation_parameters"],
            "kinematic_valid": bool(metadata["kinematic_valid"]),
            "route_proxy_valid": bool(metadata["route_valid"]),
            "candidate_accepted": accepted,
            "array_ref": {"file": f"scenes/{output_npz.name}", "index": local_index},
            "unknown": {
                "provenance": "unknown",
                "fields": ["real_interactive_agent_response", "causal_counterfactual_future"],
            },
        }
        if not accepted:
            base.update(
                {
                    "exact": {
                        "available": True,
                        "provenance": "exact",
                        "map_metrics_available": False,
                        **physical_kinematics(trajectory, config.candidate_interval_length),
                    },
                    "log_replay": _empty_assumption("log_replay", "candidate_filtered"),
                    "reactive_model": _empty_assumption("reactive_model", "candidate_filtered"),
                }
            )
            output_rows.append(base)
            continue

        try:
            replay, exact_scores, simulated_states, replay_arrays = score_under_assumption(
                metric_cache,
                trajectory,
                simulator,
                scorer,
                log_policy,
                config,
            )
            geometry_for_exact = {
                "static_object_collision": replay.pop("static_object_collision"),
                "ego_swept_footprint_area_m2": replay.pop("ego_swept_footprint_area_m2"),
            }
            exact = exact_map_consequence(
                metric_cache,
                trajectory,
                simulated_states,
                exact_scores,
                geometry_for_exact,
                config,
            )
            exact.update({"provenance": "exact", "map_metrics_available": True})
            replay["provenance"] = "log_replay"
            for name in array_names:
                arrays[f"log_replay_{name}"][local_index] = replay_arrays[name]
            base["exact"] = exact
            base["log_replay"] = replay
        except Exception:  # pragma: no cover - external evaluator failures are cached
            candidate_failures += 1
            base["exact"] = {
                "available": True,
                "provenance": "exact",
                "map_metrics_available": False,
                **physical_kinematics(trajectory, config.candidate_interval_length),
            }
            base["log_replay"] = _empty_assumption("log_replay", "evaluation_error")
            base["evaluation_error"] = traceback.format_exc()
            base["reactive_model"] = _empty_assumption("reactive_model", "evaluation_error")
            output_rows.append(base)
            continue

        if reactive_policy is None:
            base["reactive_model"] = _empty_assumption("reactive_model", "scene_not_selected")
        else:
            try:
                reactive, _, _, reactive_arrays = score_under_assumption(
                    metric_cache,
                    trajectory,
                    simulator,
                    scorer,
                    reactive_policy,
                    config,
                )
                # Static geometry and ego swept area are exact and therefore
                # intentionally never duplicated into the IDM namespace.
                reactive.pop("static_object_collision")
                reactive.pop("ego_swept_footprint_area_m2")
                reactive["provenance"] = "reactive_model"
                for name in array_names:
                    arrays[f"reactive_model_{name}"][local_index] = reactive_arrays[name]
                base["reactive_model"] = reactive
            except Exception:  # pragma: no cover - external evaluator failures are cached
                candidate_failures += 1
                base["reactive_model"] = _empty_assumption("reactive_model", "evaluation_error")
                base["reactive_evaluation_error"] = traceback.format_exc()
        output_rows.append(base)

    write_npz(output_npz, **arrays)
    write_jsonl(output_jsonl, output_rows)
    return {
        "scene_id": scene_id,
        "candidate_count": count,
        "candidate_failures": candidate_failures,
        "reactive_scene": reactive_scene,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPOSITORY_ROOT / "configs/action_effect/consequences_v2.yaml",
    )
    parser.add_argument("--candidate-cache", type=Path)
    parser.add_argument("--metric-cache", type=Path)
    parser.add_argument("--map-root", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-scenes", type=int)
    parser.add_argument(
        "--reactive-max-scenes",
        type=int,
        help="Override the configured deterministic IDM subset size (0 disables it for a dry run).",
    )
    return parser.parse_args()


def _resolve_path(explicit: Path | None, variable: str, suffix: str) -> Path:
    if explicit is not None:
        return explicit.resolve()
    root = os.environ.get(variable, "").strip()
    if not root:
        raise ValueError(f"pass a path explicitly or source load_env.sh to set {variable}")
    return (Path(root) / suffix).resolve()


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    config_path = args.config.resolve()
    config = _load_yaml(config_path)
    candidate_cache = _resolve_path(
        args.candidate_cache, "ACTION_EFFECT_CACHE_ROOT", "candidates/pilot_tiny/expert"
    )
    metric_cache_root = _resolve_path(
        args.metric_cache, "ACTION_EFFECT_CACHE_ROOT", "metric_cache/pilot_tiny/train"
    )
    map_root = _resolve_path(args.map_root, "NUPLAN_MAPS_ROOT", "")
    output_root = _resolve_path(
        args.cache_dir, "ACTION_EFFECT_CACHE_ROOT", "consequences/pilot_tiny/expert"
    )
    candidate_manifest = read_manifest(candidate_cache)
    metric_manifest = read_manifest(metric_cache_root)
    if candidate_manifest is None or metric_manifest is None:
        raise FileNotFoundError("candidate and metric caches must both have published manifests")
    with (candidate_cache / "scene_index.json").open("r", encoding="utf-8") as stream:
        candidate_scene_index = json.load(stream)
    scene_ids = list(candidate_scene_index)
    if args.max_scenes is not None:
        if args.max_scenes < 1:
            raise ValueError("--max-scenes must be positive")
        scene_ids = scene_ids[: args.max_scenes]
    metric_index = load_relative_metric_cache_index(metric_cache_root)
    missing = sorted(set(scene_ids) - set(metric_index))
    if missing:
        raise KeyError(f"metric cache is missing candidate scenes: {missing[:3]}")

    reactive_cfg = config["reactive_model"]
    reactive_max_scenes = (
        int(args.reactive_max_scenes)
        if args.reactive_max_scenes is not None
        else int(reactive_cfg["max_scenes"])
    )
    if reactive_max_scenes < 0:
        raise ValueError("--reactive-max-scenes must be non-negative")
    reactive_scene_ids = (
        _stable_reactive_subset(
            scene_ids,
            reactive_max_scenes,
            int(reactive_cfg["selection_seed"]),
        )
        if bool(reactive_cfg["enabled"])
        else set()
    )
    consequence_config = ConsequenceConfig(
        proposal_num_poses=int(config["proposal_sampling"]["num_poses"]),
        proposal_interval_length=float(config["proposal_sampling"]["interval_length"]),
        candidate_interval_length=float(config["candidate_interval_length"]),
        clearance_cap_m=float(config["clearance_cap_m"]),
        occupancy_radius_m=float(config["occupancy_radius_m"]),
        no_event_time_s=float(config["no_event_time_s"]),
    )
    evaluator_sources = [
        REPOSITORY_ROOT / "navsim/navsim/evaluate/pdm_score.py",
        REPOSITORY_ROOT / "navsim/navsim/planning/simulation/planner/pdm_planner/scoring/pdm_scorer.py",
        REPOSITORY_ROOT / "navsim/navsim/traffic_agents_policies/log_replay_traffic_agents.py",
        REPOSITORY_ROOT / "navsim/navsim/traffic_agents_policies/navsim_IDM_traffic_agents.py",
    ]
    evaluator_hash = content_hash({str(path.relative_to(REPOSITORY_ROOT)): file_sha256(path) for path in evaluator_sources})
    code_revision = _code_revision(
        [
            Path(__file__),
            REPOSITORY_ROOT / "research/action_effect/consequence_builder.py",
            REPOSITORY_ROOT / "research/action_effect/metric_cache_io.py",
        ]
    )
    manifest = CacheManifest(
        cache_kind="replay_grounded_consequence",
        cache_version=str(config["cache_version"]),
        dataset_version=str(config["dataset_version"]),
        code_commit=code_revision,
        config_hash=content_hash(config),
        evaluator_hash=evaluator_hash,
        split=candidate_manifest.split,
        seed=candidate_manifest.seed,
        inputs={
            "candidate_manifest": candidate_manifest.compatibility_identity(),
            "metric_manifest": metric_manifest.compatibility_identity(),
            "selected_scenes_sha256": content_hash(scene_ids),
            "reactive_scenes_sha256": content_hash(sorted(reactive_scene_ids)),
        },
    )
    existing = read_manifest(output_root)
    required = ("consequences.jsonl", "scene_index.json", "summary.json")
    if existing is not None:
        if existing.compatibility_identity() != manifest.compatibility_identity():
            raise CacheConflictError(f"existing consequence cache has a different identity: {output_root}")
        missing_outputs = [name for name in required if not (output_root / name).is_file()]
        if missing_outputs:
            raise CacheConflictError(f"published consequence cache is incomplete: {missing_outputs}")
        print(f"[action-effect] reusable consequence cache: {output_root}")
        return

    output_root.mkdir(parents=True, exist_ok=True)
    scenes_root = output_root / "scenes"
    scenes_root.mkdir(parents=True, exist_ok=True)
    identity_path = output_root / "build_identity.json"
    identity = {"compatibility_identity": manifest.compatibility_identity(), "manifest": asdict(manifest)}
    if identity_path.exists():
        with identity_path.open("r", encoding="utf-8") as stream:
            if json.load(stream) != identity:
                raise CacheConflictError(f"incomplete consequence cache has another identity: {output_root}")
    else:
        other_files = [path for path in output_root.iterdir() if path.name != "scenes"]
        if other_files or any(scenes_root.iterdir()):
            raise CacheConflictError(f"unidentified files already exist in consequence cache: {output_root}")
        write_json(identity_path, identity)

    all_metadata = list(iter_jsonl(candidate_cache / "metadata.jsonl"))
    trajectories = np.load(candidate_cache / "candidates.npz")["trajectories"]
    tasks: list[dict[str, Any]] = []
    completed_scenes: set[str] = set()
    for scene_id in scene_ids:
        scene_info = candidate_scene_index[scene_id]
        start, count = int(scene_info["start"]), int(scene_info["count"])
        output_jsonl = scenes_root / f"{scene_id}.jsonl"
        output_npz = scenes_root / f"{scene_id}.npz"
        if output_jsonl.is_file() and output_npz.is_file():
            completed_scenes.add(scene_id)
            continue
        if output_jsonl.exists() != output_npz.exists():
            raise CacheConflictError(f"partial scene consequence is incomplete: {scene_id}")
        tasks.append(
            {
                "scene_id": scene_id,
                "candidate_rows": all_metadata[start : start + count],
                "trajectories": trajectories[start : start + count],
                "metric_cache_path": str(metric_index[scene_id]),
                "output_jsonl": str(output_jsonl),
                "output_npz": str(output_npz),
                "consequence_config": asdict(consequence_config),
                "require_kinematic": bool(config["filter"]["require_kinematic_valid"]),
                "require_route": bool(config["filter"]["require_route_proxy_valid"]),
                "reactive_scene": scene_id in reactive_scene_ids,
                "map_root": str(map_root),
            }
        )

    candidate_failures = 0
    completed = len(completed_scenes)
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(_process_scene, task) for task in tasks]
        for future in as_completed(futures):
            result = future.result()
            candidate_failures += int(result["candidate_failures"])
            completed += 1
            print(
                f"[action-effect] consequences {completed}/{len(scene_ids)} scenes; "
                f"candidate_failures={candidate_failures}",
                flush=True,
            )

    output_rows: list[dict[str, Any]] = []
    scene_index: dict[str, dict[str, int]] = {}
    for scene_number, scene_id in enumerate(scene_ids):
        rows = list(iter_jsonl(scenes_root / f"{scene_id}.jsonl"))
        start = len(output_rows)
        output_rows.extend(rows)
        scene_index[scene_id] = {"start": start, "count": len(rows), "scene_number": scene_number}
    write_jsonl(output_root / "consequences.jsonl", output_rows)
    write_json(output_root / "scene_index.json", scene_index)
    accepted = sum(bool(row["candidate_accepted"]) for row in output_rows)
    replay_available = sum(bool(row["log_replay"]["available"]) for row in output_rows)
    reactive_available = sum(bool(row["reactive_model"]["available"]) for row in output_rows)
    cached_failures = sum(
        int("evaluation_error" in row or "reactive_evaluation_error" in row) for row in output_rows
    )
    summary = {
        "scene_count": len(scene_ids),
        "candidate_count": len(output_rows),
        "accepted_candidate_count": accepted,
        "accepted_candidate_rate": accepted / max(len(output_rows), 1),
        "log_replay_available_count": replay_available,
        "reactive_scene_count": len(reactive_scene_ids),
        "reactive_available_count": reactive_available,
        "evaluation_failure_count": cached_failures,
        "workers": args.workers,
    }
    write_json(output_root / "summary.json", summary)
    if cached_failures:
        raise RuntimeError(f"consequence evaluation produced {cached_failures} cached failures")
    finalize_manifest(output_root, manifest)
    print(json.dumps({"cache_dir": str(output_root), **summary}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
