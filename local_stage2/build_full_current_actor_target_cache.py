#!/usr/bin/env python3
"""Build full-trainval, current-frame-only actor supervision for M0 scorers.

The existing Gate-C oracle store covers only a subset of the 103,288 No-VQA
replay scenes.  This utility builds the same ``current.npy`` layout for every
scene in the immutable replay inventory, but reads only the current Scene
frame.  It never opens future sensor files, future annotations, MetricCache or
official scores.  Shards are independent and aggregation is fail-closed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import tempfile
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from tools.navsim_candidate_relative_audit.common import (
    AuditPaths,
    configure_navsim_environment,
    discover_paths,
    load_scenes_for_tokens,
)


ACTOR_SLOTS = 16
ACTOR_FIELDS = 8
CURRENT_SUMMARY_FIELDS = 6
CURRENT_WIDTH = CURRENT_SUMMARY_FIELDS + ACTOR_SLOTS * (ACTOR_FIELDS + 1)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _feature_shard_paths(feature_root: Path) -> List[Path]:
    paths = sorted(feature_root.glob("all_shard_*-of-*"))
    if not paths:
        paths = sorted(feature_root.glob("*_shard_*-of-*"))
    if not paths:
        raise RuntimeError(f"No replay shards under {feature_root}")
    return paths


def _load_inventory(paths: Sequence[Path]) -> Tuple[List[str], List[str]]:
    tokens: List[str] = []
    logs: List[str] = []
    for shard_path in paths:
        chunk_paths = sorted(shard_path.glob("chunk_*.pt"))
        if not chunk_paths:
            raise RuntimeError(f"Replay shard has no chunks: {shard_path}")
        for chunk_path in chunk_paths:
            payload = torch.load(
                chunk_path,
                map_location="cpu",
                weights_only=False,
            )
            chunk_tokens = [str(value) for value in payload["tokens"]]
            chunk_logs = [str(value) for value in payload["log_names"]]
            if len(chunk_tokens) != len(chunk_logs):
                raise RuntimeError(f"Token/log mismatch in {chunk_path}")
            tokens.extend(chunk_tokens)
            logs.extend(chunk_logs)
            del payload
    if len(tokens) != len(set(tokens)):
        raise RuntimeError("Replay inventory contains duplicate scene tokens")
    return tokens, logs


def _current_actor_slots(scene: Any, paths: AuditPaths) -> Tuple[np.ndarray, np.ndarray]:
    from navsim.planning.scenario_builder.navsim_scenario import NavSimScenario
    from nuplan.common.actor_state.tracked_objects_types import AGENT_TYPES

    scenario = NavSimScenario(scene, str(paths.map_path), "nuplan-maps-v1.0")
    current_index = int(scene.scene_metadata.num_history_frames) - 1
    current_pose = np.asarray(
        scene.frames[current_index].ego_status.ego_pose,
        dtype=np.float64,
    )
    tracks = [
        track
        for track in scenario.get_tracked_objects_at_iteration(0).tracked_objects
        if track.tracked_object_type in AGENT_TYPES
    ]
    tracks.sort(
        key=lambda track: (
            float(
                np.hypot(
                    track.center.x - current_pose[0],
                    track.center.y - current_pose[1],
                )
            ),
            str(track.track_token),
        )
    )
    values = np.zeros((ACTOR_SLOTS, ACTOR_FIELDS), dtype=np.float32)
    mask = np.zeros(ACTOR_SLOTS, dtype=bool)
    cosine, sine = math.cos(current_pose[2]), math.sin(current_pose[2])
    for slot, track in enumerate(tracks[:ACTOR_SLOTS]):
        dx = float(track.center.x - current_pose[0])
        dy = float(track.center.y - current_pose[1])
        velocity = getattr(track, "velocity", None)
        vx = float(velocity.x) if velocity is not None else 0.0
        vy = float(velocity.y) if velocity is not None else 0.0
        values[slot] = (
            int(track.tracked_object_type.value),
            cosine * dx + sine * dy,
            -sine * dx + cosine * dy,
            cosine * vx + sine * vy,
            -sine * vx + cosine * vy,
            (float(track.center.heading) - current_pose[2] + np.pi)
            % (2 * np.pi)
            - np.pi,
            float(track.box.length),
            float(track.box.width),
        )
        mask[slot] = True
    return values, mask


def _paths_from_args(args: argparse.Namespace) -> AuditPaths:
    paths = discover_paths("trainval")
    paths = replace(
        paths,
        log_path=args.log_path.resolve(),
        sensor_blobs_path=args.sensor_root.resolve(),
        map_path=args.map_path.resolve(),
    )
    configure_navsim_environment(paths)
    return paths


def build_shard(args: argparse.Namespace) -> Dict[str, Any]:
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard-index is out of range")
    feature_shards = _feature_shard_paths(args.feature_root)
    assigned = [
        path
        for index, path in enumerate(feature_shards)
        if index % args.num_shards == args.shard_index
    ]
    if not assigned:
        raise RuntimeError("current-actor shard has no assigned replay shards")
    tokens, logs = _load_inventory(assigned)
    if args.max_scenes_per_shard > 0:
        tokens = tokens[: args.max_scenes_per_shard]
        logs = logs[: args.max_scenes_per_shard]
    paths = _paths_from_args(args)
    loader = load_scenes_for_tokens(paths, tokens)
    states = np.zeros((len(tokens), ACTOR_SLOTS, ACTOR_FIELDS), dtype=np.float32)
    masks = np.zeros((len(tokens), ACTOR_SLOTS), dtype=bool)
    failures: List[Dict[str, str]] = []
    started = time.time()
    for index, token in enumerate(tokens):
        try:
            scene = loader.get_scene_from_token(token)
            states[index], masks[index] = _current_actor_slots(scene, paths)
        except Exception as exc:  # fail-closed at shard completion
            failures.append(
                {
                    "scene_token": token,
                    "log_name": logs[index],
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        if (index + 1) % 1000 == 0:
            print(
                json.dumps(
                    {
                        "shard": f"{args.shard_index}/{args.num_shards}",
                        "processed": index + 1,
                        "total": len(tokens),
                        "failures": len(failures),
                        "elapsed_seconds": time.time() - started,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    if failures:
        raise RuntimeError(
            f"Current-actor shard {args.shard_index} has {len(failures)} failures: "
            f"{failures[:3]}"
        )
    args.output_root.mkdir(parents=True, exist_ok=True)
    destination = args.output_root / f"shard_{args.shard_index:03d}-of-{args.num_shards:03d}.npz"
    if destination.exists():
        raise FileExistsError(destination)
    with tempfile.NamedTemporaryFile(
        dir=args.output_root,
        prefix=destination.name + ".",
        suffix=".tmp",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        np.savez_compressed(
            stream,
            tokens=np.asarray(tokens, dtype="U32"),
            log_names=np.asarray(logs, dtype="U96"),
            actor_states=states,
            actor_masks=masks,
        )
    temporary.replace(destination)
    result = {
        "mode": "shard",
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "scene_count": len(tokens),
        "physical_log_count": len(set(logs)),
        "actor_slots": ACTOR_SLOTS,
        "failure_count": 0,
        "elapsed_seconds": time.time() - started,
        "path": str(destination.resolve()),
        "sha256": _sha256(destination),
        "current_observation_only": True,
        "future_or_evaluator_input": False,
    }
    _atomic_json(destination.with_suffix(".json"), result)
    return result


def aggregate(args: argparse.Namespace) -> Dict[str, Any]:
    shard_paths = [
        args.output_root / f"shard_{index:03d}-of-{args.num_shards:03d}.npz"
        for index in range(args.num_shards)
    ]
    missing = [str(path) for path in shard_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing current-actor shards: {missing}")
    token_parts: List[np.ndarray] = []
    log_parts: List[np.ndarray] = []
    state_parts: List[np.ndarray] = []
    mask_parts: List[np.ndarray] = []
    for path in shard_paths:
        with np.load(path, allow_pickle=False) as payload:
            token_parts.append(payload["tokens"].copy())
            log_parts.append(payload["log_names"].copy())
            state_parts.append(payload["actor_states"].copy())
            mask_parts.append(payload["actor_masks"].copy())
    tokens = np.concatenate(token_parts).astype(str).tolist()
    logs = np.concatenate(log_parts).astype(str).tolist()
    states = np.concatenate(state_parts).astype(np.float32, copy=False)
    masks = np.concatenate(mask_parts).astype(bool, copy=False)
    if len(tokens) != args.expected_scenes:
        raise RuntimeError(
            f"Current-actor scene count {len(tokens)} != {args.expected_scenes}"
        )
    if len(set(tokens)) != len(tokens):
        raise RuntimeError("Aggregated current-actor tokens are not unique")
    if states.shape != (len(tokens), ACTOR_SLOTS, ACTOR_FIELDS):
        raise RuntimeError(f"Unexpected current-actor state shape: {states.shape}")
    if masks.shape != (len(tokens), ACTOR_SLOTS):
        raise RuntimeError(f"Unexpected current-actor mask shape: {masks.shape}")
    if not np.isfinite(states).all():
        raise RuntimeError("Current-actor states contain non-finite values")

    final_root = args.final_root
    if final_root.exists():
        raise FileExistsError(final_root)
    final_root.mkdir(parents=True)
    current = np.zeros((len(tokens), CURRENT_WIDTH), dtype=np.float32)
    current[:, CURRENT_SUMMARY_FIELDS : CURRENT_SUMMARY_FIELDS + ACTOR_SLOTS * ACTOR_FIELDS] = (
        states.reshape(len(tokens), -1)
    )
    current[:, CURRENT_SUMMARY_FIELDS + ACTOR_SLOTS * ACTOR_FIELDS :] = masks
    np.save(final_root / "current.npy", current)
    np.save(final_root / "completed.npy", np.ones(len(tokens), dtype=bool))
    metadata = pd.DataFrame(
        {
            "scene_index": np.arange(len(tokens), dtype=np.int64),
            "scene_token": tokens,
            "log_name": logs,
            "target_preflight_available": np.ones(len(tokens), dtype=bool),
        }
    )
    metadata.to_parquet(
        final_root / "scene_metadata.parquet",
        index=False,
        engine="pyarrow",
        compression="zstd",
    )
    config = {
        "schema_version": 1,
        "producer": "FullCurrentActorTargetCacheBuilder",
        "scene_count": len(tokens),
        "physical_log_count": len(set(logs)),
        "actor_slots": ACTOR_SLOTS,
        "actor_fields": (
            "object_type",
            "relative_x_m",
            "relative_y_m",
            "relative_vx_mps",
            "relative_vy_mps",
            "relative_heading_rad",
            "length_m",
            "width_m",
        ),
        "coordinate_frame": "current_ego",
        "current_observation_only": True,
        "depends_on_logged_future": False,
        "training_only_target": True,
        "available_as_model_input_at_inference": False,
        "future_or_evaluator_input": False,
        "feature_inventory_root": str(args.feature_root.resolve()),
        "source_shard_sha256": {
            path.name: _sha256(path) for path in shard_paths
        },
    }
    _atomic_json(final_root / "store_config.json", config)
    result = config | {
        "mode": "aggregate",
        "current_array_sha256": _sha256(final_root / "current.npy"),
        "completed_array_sha256": _sha256(final_root / "completed.npy"),
        "metadata_sha256": _sha256(final_root / "scene_metadata.parquet"),
        "status": "PASS",
        "root": str(final_root.resolve()),
    }
    _atomic_json(final_root / "MANIFEST.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("shard", "aggregate"), default="shard")
    parser.add_argument(
        "--feature-root",
        type=Path,
        default=Path("/root/scorer_pdms93_cache/no_vqa_e35_features_full_v1"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "/mnt/project/DriveVLA-M0-stage2/runs/scorer_pdms93/"
            "no_vqa_e35_full_current_actor_targets_v1_shards"
        ),
    )
    parser.add_argument(
        "--final-root",
        type=Path,
        default=Path(
            "/mnt/project/DriveVLA-M0-stage2/runs/scorer_pdms93/"
            "no_vqa_e35_full_current_actor_targets_v1"
        ),
    )
    parser.add_argument("--num-shards", type=int, default=8)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--expected-scenes", type=int, default=103288)
    parser.add_argument("--max-scenes-per-shard", type=int, default=0)
    parser.add_argument(
        "--log-path",
        type=Path,
        default=Path("/mnt/navsim/trainval_navsim_logs/trainval"),
    )
    parser.add_argument(
        "--sensor-root",
        type=Path,
        default=Path("/mnt/navsim/trainval_all/trainval_sensor_blobs/trainval"),
    )
    parser.add_argument(
        "--map-path",
        type=Path,
        default=Path("/mnt/navsim/maps"),
    )
    args = parser.parse_args()
    for path in (
        args.feature_root,
        args.log_path,
        args.sensor_root,
        args.map_path,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    if (
        args.num_shards <= 0
        or args.expected_scenes <= 0
        or args.max_scenes_per_shard < 0
    ):
        raise ValueError(
            "num-shards/expected-scenes must be positive and max-scenes nonnegative"
        )
    return args


def main() -> None:
    args = parse_args()
    result = aggregate(args) if args.mode == "aggregate" else build_shard(args)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
