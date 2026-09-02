#!/usr/bin/env python3
"""Build current-frame semantic-BEV supervision for scorer-private learning.

The cache is a training target, never an inference input.  It renders only the
current Scene frame and static map around the current ego pose.  It does not
open future frames, sensor blobs, MetricCache, proposals, or PDM values.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from shapely import affinity

from tools.navsim_candidate_relative_audit.common import (
    AuditPaths,
    configure_navsim_environment,
    discover_paths,
    load_scenes_for_tokens,
)


SOURCE_HEIGHT = 128
SOURCE_WIDTH = 256
BEV_HEIGHT = 16
BEV_WIDTH = 32
MAP_CHANNELS = ("road", "walkway", "centerline")
AGENT_CHANNELS = ("vehicle", "pedestrian")


class CurrentSemanticRasterizer:
    """PIL-backed equivalent of the local RetrieveTargetBuilder renderer."""

    def __init__(self) -> None:
        self.shape = (SOURCE_HEIGHT, SOURCE_WIDTH)
        self.pixel_size = 0.25
        self.radius = 32.0

    @staticmethod
    def _geometry_local_coords(geometry: Any, origin: Any) -> Any:
        cosine = np.cos(origin.heading)
        sine = np.sin(origin.heading)
        translated = affinity.affine_transform(
            geometry,
            [1, 0, 0, 1, -origin.x, -origin.y],
        )
        return affinity.affine_transform(
            translated,
            [cosine, sine, -sine, cosine, 0, 0],
        )

    def _coords_to_pixel(self, coords: np.ndarray) -> np.ndarray:
        pixel_center = np.asarray([[0.0, SOURCE_WIDTH / 2.0]])
        return np.rint(coords / self.pixel_size + pixel_center).astype(np.int32)

    @staticmethod
    def _to_bev(image: Image.Image) -> np.ndarray:
        mask = np.asarray(image, dtype=np.uint8)
        if mask.shape != (SOURCE_WIDTH, SOURCE_HEIGHT):
            raise RuntimeError(f"Unexpected PIL raster shape: {mask.shape}")
        return np.rot90(mask)[::-1] > 0

    def _blank(self) -> Image.Image:
        # PIL size is (columns, rows), matching the original OpenCV array of
        # shape (lateral_width, forward_height).
        return Image.new("L", (SOURCE_HEIGHT, SOURCE_WIDTH), color=0)

    def _map_polygon_mask(
        self,
        map_api: Any,
        ego_pose: Any,
        layers: Sequence[Any],
    ) -> np.ndarray:
        objects = map_api.get_proximal_map_objects(
            point=ego_pose.point,
            radius=self.radius,
            layers=list(layers),
        )
        image = self._blank()
        draw = ImageDraw.Draw(image)
        for layer in layers:
            for map_object in objects[layer]:
                polygon = self._geometry_local_coords(
                    map_object.polygon,
                    ego_pose,
                )
                pixels = self._coords_to_pixel(
                    np.asarray(polygon.exterior.coords, dtype=np.float64)
                )
                draw.polygon([tuple(value) for value in pixels], fill=255)
        return self._to_bev(image)

    def _map_line_mask(
        self,
        map_api: Any,
        ego_pose: Any,
        layers: Sequence[Any],
    ) -> np.ndarray:
        objects = map_api.get_proximal_map_objects(
            point=ego_pose.point,
            radius=self.radius,
            layers=list(layers),
        )
        image = self._blank()
        draw = ImageDraw.Draw(image)
        for layer in layers:
            for map_object in objects[layer]:
                line = self._geometry_local_coords(
                    map_object.baseline_path.linestring,
                    ego_pose,
                )
                pixels = self._coords_to_pixel(
                    np.asarray(line.coords, dtype=np.float64)
                )
                draw.line(
                    [tuple(value) for value in pixels],
                    fill=255,
                    width=2,
                )
        return self._to_bev(image)

    def _actor_mask(
        self,
        annotations: Any,
        included_types: Sequence[Any],
        tracked_object_types: Dict[str, Any],
    ) -> np.ndarray:
        image = self._blank()
        draw = ImageDraw.Draw(image)
        included = set(included_types)
        for name, box in zip(annotations.names, annotations.boxes):
            if name not in tracked_object_types:
                continue
            if tracked_object_types[name] not in included:
                continue
            x, y, heading = float(box[0]), float(box[1]), float(box[-1])
            half_length = 0.5 * float(box[3])
            half_width = 0.5 * float(box[4])
            longitudinal = np.asarray(
                [np.cos(heading), np.sin(heading)], dtype=np.float64
            )
            lateral = np.asarray(
                [-np.sin(heading), np.cos(heading)], dtype=np.float64
            )
            center = np.asarray([x, y], dtype=np.float64)
            corners = np.stack(
                (
                    center + half_length * longitudinal + half_width * lateral,
                    center + half_length * longitudinal - half_width * lateral,
                    center - half_length * longitudinal - half_width * lateral,
                    center - half_length * longitudinal + half_width * lateral,
                )
            )
            pixels = self._coords_to_pixel(corners)
            draw.polygon([tuple(value) for value in pixels], fill=255)
        return self._to_bev(image)

    def compute_targets(self, scene: Any) -> Dict[str, np.ndarray]:
        from nuplan.common.actor_state.state_representation import StateSE2
        from nuplan.common.actor_state.tracked_objects_types import (
            TrackedObjectType,
        )
        from nuplan.common.maps.abstract_map import SemanticMapLayer
        from navsim.planning.scenario_builder.navsim_scenario_utils import (
            tracked_object_types,
        )

        frame_index = int(scene.scene_metadata.num_history_frames) - 1
        frame = scene.frames[frame_index]
        ego_pose = StateSE2(*frame.ego_status.ego_pose)
        road = self._map_polygon_mask(
            scene.map_api,
            ego_pose,
            (SemanticMapLayer.LANE, SemanticMapLayer.INTERSECTION),
        )
        walkway = self._map_polygon_mask(
            scene.map_api,
            ego_pose,
            (SemanticMapLayer.WALKWAYS,),
        )
        centerline = self._map_line_mask(
            scene.map_api,
            ego_pose,
            (SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR),
        )
        map_target = np.zeros(self.shape, dtype=np.uint8)
        map_target[road] = 1
        map_target[walkway] = 2
        map_target[centerline] = 3

        vehicle = self._actor_mask(
            frame.annotations,
            (TrackedObjectType.VEHICLE,),
            tracked_object_types,
        )
        pedestrian = self._actor_mask(
            frame.annotations,
            (TrackedObjectType.PEDESTRIAN,),
            tracked_object_types,
        )
        agent_target = np.zeros(self.shape, dtype=np.uint8)
        agent_target[vehicle] = 1
        agent_target[pedestrian] = 2
        return {"map_target": map_target, "agent_target": agent_target}


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


def foreground_preserving_pool(
    labels: np.ndarray,
    class_masks: Sequence[np.ndarray],
    *,
    output_height: int = BEV_HEIGHT,
    output_width: int = BEV_WIDTH,
) -> np.ndarray:
    """Max-pool independent foreground channels without majority collapse."""

    if labels.shape != (SOURCE_HEIGHT, SOURCE_WIDTH):
        raise ValueError(
            f"Expected source raster {(SOURCE_HEIGHT, SOURCE_WIDTH)}, "
            f"got {labels.shape}"
        )
    masks = np.stack(class_masks).astype(np.float32, copy=False)
    tensor = torch.from_numpy(masks).unsqueeze(0)
    pooled = F.adaptive_max_pool2d(tensor, (output_height, output_width))
    return pooled.squeeze(0).bool().numpy()


def _semantic_channels(
    targets: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray]:
    map_target = np.asarray(targets["map_target"])
    agent_target = np.asarray(targets["agent_target"])
    # Centerline overwrites road in the rendered categorical raster. Include
    # it in road occupancy as well so the road-surface channel remains closed.
    map_channels = foreground_preserving_pool(
        map_target,
        (
            (map_target == 1) | (map_target == 3),
            map_target == 2,
            map_target == 3,
        ),
    )
    agent_channels = foreground_preserving_pool(
        agent_target,
        (agent_target == 1, agent_target == 2),
    )
    if map_channels.shape != (len(MAP_CHANNELS), BEV_HEIGHT, BEV_WIDTH):
        raise RuntimeError(f"Unexpected pooled map shape: {map_channels.shape}")
    if agent_channels.shape != (len(AGENT_CHANNELS), BEV_HEIGHT, BEV_WIDTH):
        raise RuntimeError(f"Unexpected pooled agent shape: {agent_channels.shape}")
    return map_channels, agent_channels


def build_shard(args: argparse.Namespace) -> Dict[str, Any]:
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard-index is out of range")
    feature_shards = _feature_shard_paths(args.feature_root)
    if args.num_shards % len(feature_shards):
        raise ValueError(
            "semantic-BEV shard count must be a multiple of replay shards"
        )
    partitions_per_feature_shard = args.num_shards // len(feature_shards)
    feature_shard_index = args.shard_index % len(feature_shards)
    partition_index = args.shard_index // len(feature_shards)
    tokens, logs = _load_inventory([feature_shards[feature_shard_index]])
    tokens = tokens[partition_index::partitions_per_feature_shard]
    logs = logs[partition_index::partitions_per_feature_shard]
    if not tokens:
        raise RuntimeError("semantic-BEV shard has no assigned replay scenes")
    if args.max_scenes_per_shard > 0:
        tokens = tokens[: args.max_scenes_per_shard]
        logs = logs[: args.max_scenes_per_shard]

    paths = _paths_from_args(args)
    loader = load_scenes_for_tokens(paths, tokens)
    renderer = CurrentSemanticRasterizer()
    map_targets = np.zeros(
        (len(tokens), len(MAP_CHANNELS), BEV_HEIGHT, BEV_WIDTH), dtype=bool
    )
    agent_targets = np.zeros(
        (len(tokens), len(AGENT_CHANNELS), BEV_HEIGHT, BEV_WIDTH), dtype=bool
    )
    failures: List[Dict[str, str]] = []
    started = time.time()
    for index, token in enumerate(tokens):
        try:
            scene = loader.get_scene_from_token(token)
            rendered = renderer.compute_targets(scene)
            map_targets[index], agent_targets[index] = _semantic_channels(rendered)
        except Exception as exc:  # fail closed when the shard completes
            failures.append(
                {
                    "scene_token": token,
                    "log_name": logs[index],
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        if (index + 1) % 500 == 0:
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
            f"Semantic-BEV shard {args.shard_index} has {len(failures)} "
            f"failures: {failures[:3]}"
        )

    args.output_root.mkdir(parents=True, exist_ok=True)
    destination = (
        args.output_root
        / f"shard_{args.shard_index:03d}-of-{args.num_shards:03d}.npz"
    )
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
            map_targets=map_targets,
            agent_targets=agent_targets,
        )
    temporary.replace(destination)
    result = {
        "mode": "shard",
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "source_feature_shard": feature_shards[feature_shard_index].name,
        "source_partition_index": partition_index,
        "source_partition_count": partitions_per_feature_shard,
        "scene_count": len(tokens),
        "physical_log_count": len(set(logs)),
        "failure_count": 0,
        "map_positive_fraction": map_targets.mean(axis=(0, 2, 3)).tolist(),
        "agent_positive_fraction": agent_targets.mean(axis=(0, 2, 3)).tolist(),
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
        raise FileNotFoundError(f"Missing semantic-BEV shards: {missing}")
    token_parts: List[np.ndarray] = []
    log_parts: List[np.ndarray] = []
    map_parts: List[np.ndarray] = []
    agent_parts: List[np.ndarray] = []
    for path in shard_paths:
        with np.load(path, allow_pickle=False) as payload:
            token_parts.append(payload["tokens"].copy())
            log_parts.append(payload["log_names"].copy())
            map_parts.append(payload["map_targets"].copy())
            agent_parts.append(payload["agent_targets"].copy())
    tokens = np.concatenate(token_parts).astype(str).tolist()
    logs = np.concatenate(log_parts).astype(str).tolist()
    map_targets = np.concatenate(map_parts).astype(bool, copy=False)
    agent_targets = np.concatenate(agent_parts).astype(bool, copy=False)
    if len(tokens) != args.expected_scenes:
        raise RuntimeError(
            f"Semantic-BEV scene count {len(tokens)} != {args.expected_scenes}"
        )
    if len(set(tokens)) != len(tokens):
        raise RuntimeError("Aggregated semantic-BEV tokens are not unique")
    expected_map = (len(tokens), len(MAP_CHANNELS), BEV_HEIGHT, BEV_WIDTH)
    expected_agent = (len(tokens), len(AGENT_CHANNELS), BEV_HEIGHT, BEV_WIDTH)
    if map_targets.shape != expected_map or agent_targets.shape != expected_agent:
        raise RuntimeError(
            f"Unexpected semantic-BEV shapes: {map_targets.shape}, "
            f"{agent_targets.shape}"
        )
    if not map_targets.any() or not agent_targets.any():
        raise RuntimeError("Semantic-BEV target cache is degenerate")

    final_root = args.final_root
    if final_root.exists():
        raise FileExistsError(final_root)
    final_root.mkdir(parents=True)
    np.save(final_root / "map_targets.npy", map_targets)
    np.save(final_root / "agent_targets.npy", agent_targets)
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
    arrays = {
        name: _sha256(final_root / name)
        for name in ("map_targets.npy", "agent_targets.npy", "completed.npy")
    }
    config: Dict[str, Any] = {
        "schema_version": 1,
        "producer": "FullCurrentSemanticBEVTargetCacheBuilder",
        "scene_count": len(tokens),
        "physical_log_count": len(set(logs)),
        "coordinate_frame": "current_ego",
        "bev_height": BEV_HEIGHT,
        "bev_width": BEV_WIDTH,
        "source_height": SOURCE_HEIGHT,
        "source_width": SOURCE_WIDTH,
        "forward_extent_m": [0.0, 32.0],
        "lateral_extent_m": [-32.0, 32.0],
        "cell_size_m": [2.0, 2.0],
        "map_channels": MAP_CHANNELS,
        "agent_channels": AGENT_CHANNELS,
        "pooling": "foreground_preserving_adaptive_max_pool",
        "map_positive_fraction": map_targets.mean(axis=(0, 2, 3)).tolist(),
        "agent_positive_fraction": agent_targets.mean(axis=(0, 2, 3)).tolist(),
        "current_observation_only": True,
        "depends_on_logged_future": False,
        "training_only_target": True,
        "available_as_model_input_at_inference": False,
        "future_or_evaluator_input": False,
        "feature_inventory_root": str(args.feature_root.resolve()),
        "array_sha256": arrays,
        "metadata_sha256": _sha256(final_root / "scene_metadata.parquet"),
        "source_shard_sha256": {
            path.name: _sha256(path) for path in shard_paths
        },
    }
    _atomic_json(final_root / "store_config.json", config)
    result = config | {
        "mode": "aggregate",
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
            "no_vqa_e35_full_current_semantic_bev_targets_v1_shards"
        ),
    )
    parser.add_argument(
        "--final-root",
        type=Path,
        default=Path(
            "/mnt/project/DriveVLA-M0-stage2/runs/scorer_pdms93/"
            "no_vqa_e35_full_current_semantic_bev_targets_v1"
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
    parser.add_argument("--map-path", type=Path, default=Path("/mnt/navsim/maps"))
    args = parser.parse_args()
    for path in (args.feature_root, args.log_path, args.sensor_root, args.map_path):
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
    feature_shard_count = len(_feature_shard_paths(args.feature_root))
    if args.num_shards % feature_shard_count:
        raise ValueError("num-shards must be a multiple of replay feature shards")
    return args


def main() -> None:
    args = parse_args()
    result = aggregate(args) if args.mode == "aggregate" else build_shard(args)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
