#!/usr/bin/env python3
"""Mine critical traffic participants from NAVSIM scenes.

Example commands:

    # Training/mini split smoke run with BEV and camera bbox visualizations.
    python tools/mine_critical_agents_navsim.py \
      --split mini \
      --data-root /mnt/data/navsim \
      --log-dir /mnt/data/navsim/mini_navsim_logs/mini \
      --max-samples 50 \
      --output-dir navsim_dataset/critical_agents \
      --visualize-dir navsim_dataset/critical_agents_viz \
      --visualize-camera-dir navsim_dataset/critical_agents_camera_viz \
      --visualize-limit 50 \
      --overwrite

    # Mine visible critical agents for zero-score test scenes.
    python tools/mine_critical_agents_navsim.py \
      --split test \
      --data-root /mnt/data/navsim \
      --log-dir /mnt/data/navsim/test_navsim_logs/test \
      --sensor-dir /mnt/data/navsim/test_sensor_blobs/test \
      --tokens-file /path/to/zero_score_tokens.txt \
      --output-dir /path/to/critical_agents_zero_score \
      --visualize-camera-dir /path/to/critical_agents_zero_score_camera_viz \
      --visualize-limit 573 \
      --overwrite

Important arguments:
    --split: NAVSIM split name, for example train, mini, or test.
    --data-root: Root containing *_navsim_logs, *_sensor_blobs, and optionally maps.
    --log-dir / --sensor-dir: Explicit log and sensor directories when the default
        <data-root>/<split>_navsim_logs/<split> layout is not used.
    --tokens-file: Optional newline-separated scene tokens to process.
    --output-dir: Destination for one JSON sidecar per scene token.
    --max-samples: Optional cap for debugging.
    --top-k: Maximum number of agents kept per scene. Use 32 with --selection-mode visible for object-slot pretraining.
    --selection-mode: critical ranks by planning-risk score; visible ranks by visual quality only.
    --agent-classes: Classes considered as critical-agent candidates. Defaults to
        vehicle, pedestrian, and bicycle.
    --min-bbox-area: Minimum projected 2D bbox area in pixels.
    --min-visible-ratio: Minimum visible 2D area ratio inside camera image bounds.
    --min-visible-corners: Minimum number of projected 3D-box corners inside image.
    --max-occlusion-ratio: Maximum 2D bbox occlusion ratio after depth sorting.
    --visualize-dir: Optional BEV debug image output directory.
    --visualize-camera-dir: Optional camera-view bbox image output directory.
    --visualize-limit: Maximum number of scenes to visualize.
    --overwrite: Regenerate existing JSON/images.

Outputs:
    Each JSON stores selected agents, current ego-frame boxes, score terms, best
    camera view, projected 2D bbox, visibility ratio, and depth-sorted occlusion
    ratio.  This stage does not extract DINO features.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
NAVSIM_PROCESS_ROOT = REPO_ROOT / "navsim_data_process"
if str(NAVSIM_PROCESS_ROOT) not in sys.path:
    sys.path.insert(0, str(NAVSIM_PROCESS_ROOT))


CURRENT_FRAME_INDEX = 3
DEFAULT_AGENT_CLASSES = {
    "vehicle",
    "pedestrian",
    "bicycle",
}

CLASS_PRIORITY = {
    "pedestrian": 1.00,
    "bicycle": 0.90,
    "vehicle": 0.65,
    "barrier": 0.45,
    "traffic_cone": 0.40,
    "czone_sign": 0.35,
    "generic_object": 0.30,
}


@dataclass
class AgentCandidate:
    track_token: Optional[str]
    instance_token: Optional[str]
    class_name: str
    box_ego: List[float]
    velocity_ego: List[float]
    distance: float
    score: float
    score_terms: Dict[str, float]
    closest_future_step: int
    view: Optional[str]
    bbox_xyxy: Optional[List[float]]
    bbox_area: float
    visible_ratio: float
    occlusion_ratio: float
    visible_box_ratio: float
    image_path: Optional[str]
    camera_projections: List[Dict[str, Any]]
    valid: bool = True


def wrap_to_pi(angle: float | np.ndarray) -> float | np.ndarray:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def rotation_matrix(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def local_to_global(points_xy: np.ndarray, ego_pose: np.ndarray) -> np.ndarray:
    return points_xy @ rotation_matrix(float(ego_pose[2])).T + ego_pose[:2]


def global_to_local(points_xy: np.ndarray, ego_pose: np.ndarray) -> np.ndarray:
    return (points_xy - ego_pose[:2]) @ rotation_matrix(float(ego_pose[2]))


def transform_local_between_ego_frames(
    points_xy: np.ndarray,
    source_ego_pose: np.ndarray,
    target_ego_pose: np.ndarray,
) -> np.ndarray:
    return global_to_local(local_to_global(points_xy, source_ego_pose), target_ego_pose)


def ann_get(annotations: Any, key: str, default: Any = None) -> Any:
    if isinstance(annotations, dict):
        return annotations.get(key, default)
    return getattr(annotations, key, default)


def ego_pose(status: Any) -> np.ndarray:
    if isinstance(status, dict):
        value = status["ego_pose"]
    else:
        value = getattr(status, "ego_pose")
    return np.asarray(value, dtype=np.float64)


def frame_timestamp(frame: Dict[str, Any]) -> float:
    timestamp = frame.get("timestamp", 0)
    return float(timestamp) * (1e-6 if abs(float(timestamp)) > 1e8 else 1.0)


def normalize_class(name: Any) -> str:
    name = str(name).lower()
    aliases = {
        "car": "vehicle",
        "truck": "vehicle",
        "bus": "vehicle",
        "trailer": "vehicle",
        "construction_vehicle": "vehicle",
        "motorcycle": "bicycle",
        "cyclist": "bicycle",
    }
    return aliases.get(name, name)


def ego_future_path_current_frame(frame_data: List[Dict[str, Any]], current_idx: int) -> np.ndarray:
    poses = np.stack([ego_pose(frame["ego_status"]) for frame in frame_data], axis=0)
    current_pose = poses[current_idx]
    future_global = poses[current_idx + 1 :, :2]
    if len(future_global) == 0:
        return np.zeros((0, 2), dtype=np.float64)
    return global_to_local(future_global, current_pose)


def ego_motion_events(frame_data: List[Dict[str, Any]], current_idx: int) -> Dict[str, Any]:
    poses = np.stack([ego_pose(frame["ego_status"]) for frame in frame_data], axis=0)
    times = np.asarray([frame_timestamp(frame) for frame in frame_data], dtype=np.float64)
    if np.any(np.diff(times) <= 0):
        times = np.arange(len(frame_data), dtype=np.float64) * 0.5

    current_pose = poses[current_idx]
    future = poses[current_idx:]
    future_local = global_to_local(future[:, :2], current_pose)

    dt = np.maximum(np.diff(times[current_idx:]), 1e-3)
    disp = np.linalg.norm(np.diff(future[:, :2], axis=0), axis=1)
    speeds = disp / dt if len(dt) else np.zeros(0, dtype=np.float64)
    accel = np.diff(speeds) / np.maximum(dt[1:], 1e-3) if len(speeds) > 1 else np.zeros(0, dtype=np.float64)
    yaw_delta = wrap_to_pi(future[-1, 2] - future[0, 2]) if len(future) > 1 else 0.0
    lateral_span = float(np.max(np.abs(future_local[:, 1]))) if len(future_local) else 0.0

    braking_strength = 0.0
    if len(accel):
        braking_strength = float(np.clip((-np.min(accel) - 0.8) / 2.2, 0.0, 1.0))
    if len(speeds) >= 4:
        braking_strength = max(
            braking_strength,
            float(np.clip((speeds[0] - np.min(speeds[:4]) - 1.0) / 3.0, 0.0, 1.0)),
        )

    turning_strength = max(
        float(np.clip((abs(yaw_delta) - 0.18) / 0.45, 0.0, 1.0)),
        float(np.clip((lateral_span - 1.0) / 3.0, 0.0, 1.0)),
    )

    return {
        "braking": braking_strength > 0.0,
        "turning": turning_strength > 0.0,
        "braking_strength": braking_strength,
        "turning_strength": turning_strength,
        "yaw_delta": float(yaw_delta),
        "lateral_span": lateral_span,
        "future_speeds": speeds.tolist(),
    }


def collect_agent_tracks_current_frame(
    frame_data: List[Dict[str, Any]],
    current_idx: int,
    allowed_classes: set[str],
    max_distance: float,
    min_distance: float,
) -> Dict[str, Dict[str, Any]]:
    current_pose = ego_pose(frame_data[current_idx]["ego_status"])
    current_ann = frame_data[current_idx]["annotations"]
    boxes = np.asarray(ann_get(current_ann, "boxes", []), dtype=np.float64)
    names = list(ann_get(current_ann, "names", []))
    velocities = np.asarray(ann_get(current_ann, "velocity_3d", np.zeros((len(boxes), 3))), dtype=np.float64)
    instance_tokens = list(ann_get(current_ann, "instance_tokens", [None] * len(boxes)))
    track_tokens = list(ann_get(current_ann, "track_tokens", [None] * len(boxes)))

    candidates: Dict[str, Dict[str, Any]] = {}
    for idx, box in enumerate(boxes):
        class_name = normalize_class(names[idx] if idx < len(names) else "generic_object")
        if class_name not in allowed_classes:
            continue
        if box.shape[0] < 7 or np.any(np.asarray(box[3:6]) <= 0):
            continue

        center = np.asarray(box[:2], dtype=np.float64)
        distance = float(np.linalg.norm(center))
        if distance < min_distance or distance > max_distance:
            continue
        if center[0] < -10.0:
            continue

        track_token = track_tokens[idx] if idx < len(track_tokens) else None
        instance_token = instance_tokens[idx] if idx < len(instance_tokens) else None
        stable_key = str(track_token or instance_token or f"idx_{idx}")
        velocity = velocities[idx] if idx < len(velocities) else np.zeros(3, dtype=np.float64)
        candidates[stable_key] = {
            "track_token": track_token,
            "instance_token": instance_token,
            "class_name": class_name,
            "box_current": np.asarray(box, dtype=np.float64),
            "velocity_current": np.asarray(velocity, dtype=np.float64),
            "distance": distance,
            "future_centers": {},
        }

    for frame_idx in range(current_idx, len(frame_data)):
        pose_i = ego_pose(frame_data[frame_idx]["ego_status"])
        ann_i = frame_data[frame_idx]["annotations"]
        boxes_i = np.asarray(ann_get(ann_i, "boxes", []), dtype=np.float64)
        track_tokens_i = list(ann_get(ann_i, "track_tokens", [None] * len(boxes_i)))
        instance_tokens_i = list(ann_get(ann_i, "instance_tokens", [None] * len(boxes_i)))
        for idx, box in enumerate(boxes_i):
            stable_key = str(
                (track_tokens_i[idx] if idx < len(track_tokens_i) else None)
                or (instance_tokens_i[idx] if idx < len(instance_tokens_i) else None)
                or f"idx_{idx}"
            )
            if stable_key not in candidates:
                continue
            center_current = transform_local_between_ego_frames(
                np.asarray(box[:2], dtype=np.float64)[None, :],
                pose_i,
                current_pose,
            )[0]
            candidates[stable_key]["future_centers"][frame_idx - current_idx] = center_current

    return candidates


def min_distance_to_path(point_xy: np.ndarray, path_xy: np.ndarray) -> tuple[float, int]:
    if len(path_xy) == 0:
        return float(np.linalg.norm(point_xy)), 0
    distances = np.linalg.norm(path_xy - point_xy[None, :], axis=1)
    idx = int(np.argmin(distances))
    return float(distances[idx]), idx + 1


def score_path_corridor(min_path_dist: float, inner_width: float, outer_width: float) -> float:
    if min_path_dist <= inner_width:
        return 1.0
    if min_path_dist >= outer_width:
        return 0.0
    return float((outer_width - min_path_dist) / max(outer_width - inner_width, 1e-6))


def score_distance(distance: float, max_distance: float = 40.0) -> float:
    return float(np.clip((max_distance - distance) / max_distance, 0.0, 1.0))


def score_ttc(center_xy: np.ndarray, velocity_xy: np.ndarray) -> float:
    # Approximate ego-relative closing time in current ego frame.
    rel_pos = center_xy
    rel_vel = velocity_xy[:2]
    closing_speed = -float(np.dot(rel_pos, rel_vel)) / max(float(np.linalg.norm(rel_pos)), 1e-6)
    if closing_speed <= 0.3:
        return 0.0
    ttc = float(np.linalg.norm(rel_pos) / closing_speed)
    if ttc >= 6.0:
        return 0.0
    return float(np.clip((6.0 - ttc) / 6.0, 0.0, 1.0))


def score_front(center_xy: np.ndarray) -> float:
    x, y = float(center_xy[0]), abs(float(center_xy[1]))
    if x <= -1.0:
        return 0.0
    forward = np.clip((x + 2.0) / 20.0, 0.0, 1.0)
    lateral = np.clip((8.0 - y) / 8.0, 0.0, 1.0)
    return float(forward * lateral)


def score_intervention(
    candidate: Dict[str, Any],
    ego_path: np.ndarray,
    events: Dict[str, Any],
    straight_reference: np.ndarray,
    inner_width: float,
    outer_width: float,
) -> tuple[float, Dict[str, float]]:
    current_center = np.asarray(candidate["box_current"][:2], dtype=np.float64)
    velocity = np.asarray(candidate["velocity_current"], dtype=np.float64)
    braking = 0.0
    turning = 0.0

    if events["braking"]:
        front_score = score_front(current_center)
        ttc = score_ttc(current_center, velocity)
        path_dist, _ = min_distance_to_path(current_center, ego_path)
        path_score = score_path_corridor(path_dist, inner_width, outer_width)
        braking = events["braking_strength"] * max(0.50 * path_score + 0.35 * ttc + 0.15 * front_score, 0.0)

    if events["turning"]:
        # Agent-caused turns are approximated by conflict with a straight-ahead
        # continuation and/or the final driven path. Route-induced turns without
        # nearby agents receive little extra score.
        straight_dist, _ = min_distance_to_path(current_center, straight_reference)
        driven_dist, _ = min_distance_to_path(current_center, ego_path)
        straight_conflict = score_path_corridor(straight_dist, inner_width + 0.5, outer_width + 1.0)
        driven_conflict = score_path_corridor(driven_dist, inner_width, outer_width)
        lateral_side = 1.0 if abs(float(current_center[1])) < outer_width + 2.0 else 0.0
        turning = events["turning_strength"] * (0.55 * straight_conflict + 0.30 * driven_conflict + 0.15 * lateral_side)

    intervention = float(np.clip(max(braking, turning), 0.0, 1.0))
    return intervention, {"braking": float(braking), "turning": float(turning)}


def build_straight_reference(ego_path: np.ndarray) -> np.ndarray:
    if len(ego_path) == 0:
        return np.zeros((0, 2), dtype=np.float64)
    distances = np.linalg.norm(ego_path, axis=1)
    return np.stack([distances, np.zeros_like(distances)], axis=1)


def score_candidates(
    candidates: Dict[str, Dict[str, Any]],
    ego_path: np.ndarray,
    events: Dict[str, Any],
    inner_width: float,
    outer_width: float,
) -> List[AgentCandidate]:
    straight_reference = build_straight_reference(ego_path)
    scored: List[AgentCandidate] = []

    for candidate in candidates.values():
        current_box = np.asarray(candidate["box_current"], dtype=np.float64)
        current_center = current_box[:2]
        distance = float(candidate["distance"])

        path_dist, closest_step = min_distance_to_path(current_center, ego_path)
        path_corridor = score_path_corridor(path_dist, inner_width, outer_width)

        # Use future agent centers when available. This catches crossing agents
        # whose current position is not yet on the ego path.
        future_path_corridor = path_corridor
        future_closest_step = closest_step
        for rel_step, center in candidate["future_centers"].items():
            if rel_step <= 0 or rel_step - 1 >= len(ego_path):
                continue
            timed_dist = float(np.linalg.norm(np.asarray(center) - ego_path[rel_step - 1]))
            timed_score = score_path_corridor(timed_dist, inner_width, outer_width)
            if timed_score > future_path_corridor:
                future_path_corridor = timed_score
                future_closest_step = int(rel_step)

        distance_term = score_distance(distance)
        ttc_term = score_ttc(current_center, np.asarray(candidate["velocity_current"]))
        front_term = score_front(current_center)
        class_term = CLASS_PRIORITY.get(candidate["class_name"], CLASS_PRIORITY["generic_object"])
        intervention_term, intervention_terms = score_intervention(
            candidate,
            ego_path,
            events,
            straight_reference,
            inner_width,
            outer_width,
        )

        base_score = (
            0.40 * future_path_corridor
            + 0.25 * distance_term
            + 0.20 * ttc_term
            + 0.10 * front_term
            + 0.05 * class_term
        )
        score = float(np.clip(0.75 * base_score + 0.25 * intervention_term, 0.0, 1.0))

        scored.append(
            AgentCandidate(
                track_token=candidate["track_token"],
                instance_token=candidate["instance_token"],
                class_name=candidate["class_name"],
                box_ego=current_box.tolist(),
                velocity_ego=np.asarray(candidate["velocity_current"]).tolist(),
                distance=distance,
                score=score,
                closest_future_step=future_closest_step,
                view=candidate["best_camera_projection"]["view"],
                bbox_xyxy=candidate["best_camera_projection"]["bbox_xyxy"],
                bbox_area=float(candidate["best_camera_projection"]["bbox_area"]),
                visible_ratio=float(candidate["best_camera_projection"]["visible_ratio"]),
                occlusion_ratio=float(candidate["best_camera_projection"].get("occlusion_ratio", 0.0)),
                visible_box_ratio=float(candidate["best_camera_projection"].get("visible_box_ratio", 1.0)),
                image_path=candidate["best_camera_projection"].get("image_path"),
                camera_projections=candidate.get("camera_projections", []),
                score_terms={
                    "path_corridor": float(future_path_corridor),
                    "distance": distance_term,
                    "ttc": ttc_term,
                    "front": front_term,
                    "class_priority": float(class_term),
                    "intervention": intervention_term,
                    "intervention_braking": intervention_terms["braking"],
                    "intervention_turning": intervention_terms["turning"],
                },
            )
        )

    scored.sort(key=lambda item: item.score, reverse=True)
    return scored


def visible_sort_score(agent: AgentCandidate) -> float:
    area_score = float(np.clip(math.log1p(max(agent.bbox_area, 0.0)) / math.log1p(20000.0), 0.0, 1.0))
    multi_view_score = float(np.clip(len(agent.camera_projections) / 3.0, 0.0, 1.0))
    distance_score = score_distance(agent.distance, max_distance=60.0)
    class_score = CLASS_PRIORITY.get(agent.class_name, CLASS_PRIORITY["generic_object"])
    return float(
        0.40 * area_score
        + 0.25 * agent.visible_box_ratio
        + 0.15 * agent.visible_ratio
        + 0.10 * multi_view_score
        + 0.05 * distance_score
        + 0.05 * class_score
    )


def order_agents(agents: List[AgentCandidate], selection_mode: str) -> List[AgentCandidate]:
    if selection_mode == "visible":
        return sorted(
            agents,
            key=lambda item: (visible_sort_score(item), item.bbox_area, -item.distance),
            reverse=True,
        )
    if selection_mode == "critical":
        return sorted(agents, key=lambda item: item.score, reverse=True)
    raise ValueError(f"Unsupported selection_mode: {selection_mode}")


def pad_agents(agents: List[AgentCandidate], top_k: int) -> List[Dict[str, Any]]:
    result = []
    for rank, agent in enumerate(agents[:top_k]):
        item = asdict(agent)
        item["rank"] = rank
        item["visible_sort_score"] = visible_sort_score(agent)
        result.append(item)

    while len(result) < top_k:
        result.append(
            {
                "rank": len(result),
                "track_token": None,
                "instance_token": None,
                "class_name": None,
                "box_ego": [0.0] * 7,
                "velocity_ego": [0.0, 0.0, 0.0],
                "distance": 0.0,
                "score": 0.0,
                "score_terms": {},
                "closest_future_step": 0,
                "view": None,
                "bbox_xyxy": None,
                "bbox_area": 0.0,
                "visible_ratio": 0.0,
                "occlusion_ratio": 1.0,
                "visible_box_ratio": 0.0,
                "image_path": None,
                "camera_projections": [],
                "valid": False,
            }
        )
    return result




def box_corners_3d(box: Iterable[float]) -> np.ndarray:
    box = np.asarray(list(box), dtype=np.float64)
    x, y, z, length, width, height, yaw = box[:7]
    local = np.array(
        [
            [length / 2.0, width / 2.0, height / 2.0],
            [length / 2.0, -width / 2.0, height / 2.0],
            [-length / 2.0, -width / 2.0, height / 2.0],
            [-length / 2.0, width / 2.0, height / 2.0],
            [length / 2.0, width / 2.0, -height / 2.0],
            [length / 2.0, -width / 2.0, -height / 2.0],
            [-length / 2.0, -width / 2.0, -height / 2.0],
            [-length / 2.0, width / 2.0, -height / 2.0],
        ],
        dtype=np.float64,
    )
    rot = np.eye(3, dtype=np.float64)
    rot[:2, :2] = rotation_matrix(float(yaw))
    return local @ rot.T + np.array([x, y, z], dtype=np.float64)


def image_hw_from_path(path: str) -> tuple[int, int]:
    from PIL import Image

    with Image.open(path) as image:
        width, height = image.size
    return int(height), int(width)


def project_box_to_camera(
    box: Iterable[float],
    camera: Dict[str, Any],
    image_hw: tuple[int, int],
    min_visible_corners: int,
) -> Optional[Dict[str, Any]]:
    corners_lidar = box_corners_3d(box)
    rotation = np.asarray(camera["sensor2lidar_rotation"], dtype=np.float64)
    translation = np.asarray(camera["sensor2lidar_translation"], dtype=np.float64)
    intrinsic = np.asarray(camera["intrinsics"], dtype=np.float64)

    # sensor2lidar maps camera coordinates into lidar/ego coordinates.
    # Invert it for lidar -> camera.
    corners_cam = (corners_lidar - translation[None, :]) @ rotation
    depth = corners_cam[:, 2]
    valid_depth = depth > 0.2
    if int(valid_depth.sum()) < min_visible_corners:
        return None

    pixels_h = corners_cam[valid_depth] @ intrinsic.T
    pixels = pixels_h[:, :2] / np.clip(pixels_h[:, 2:3], 1e-6, None)
    height, width = image_hw
    x1, y1 = pixels.min(axis=0)
    x2, y2 = pixels.max(axis=0)
    clipped_x1 = float(np.clip(x1, 0, width - 1))
    clipped_y1 = float(np.clip(y1, 0, height - 1))
    clipped_x2 = float(np.clip(x2, 0, width - 1))
    clipped_y2 = float(np.clip(y2, 0, height - 1))
    clipped_w = max(0.0, clipped_x2 - clipped_x1)
    clipped_h = max(0.0, clipped_y2 - clipped_y1)
    clipped_area = clipped_w * clipped_h
    raw_area = max(0.0, float((x2 - x1) * (y2 - y1)))
    if raw_area <= 1e-6:
        return None
    visible_ratio = clipped_area / raw_area
    center = np.array([(clipped_x1 + clipped_x2) * 0.5, (clipped_y1 + clipped_y2) * 0.5], dtype=np.float64)
    image_center = np.array([width * 0.5, height * 0.5], dtype=np.float64)
    center_offset = float(np.linalg.norm((center - image_center) / np.array([width, height], dtype=np.float64)))
    return {
        "bbox_xyxy": [clipped_x1, clipped_y1, clipped_x2, clipped_y2],
        "bbox_area": float(clipped_area),
        "raw_bbox_area": float(raw_area),
        "visible_ratio": float(visible_ratio),
        "visible_corners": int(valid_depth.sum()),
        "median_depth": float(np.median(depth[valid_depth])),
        "image_size": [height, width],
        "center_offset": center_offset,
    }



def bbox_intersection_area(a: List[float], b: List[float]) -> float:
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def estimate_occlusion_ratios(projections: List[Dict[str, Any]]) -> None:
    """Estimate 2D occlusion by nearer boxes in the same camera view.

    This is intentionally conservative and box-level: it does not claim pixel
    accurate visibility, but rejects far boxes whose 2D ROI is mostly covered by
    closer projected boxes.
    """
    by_view: Dict[str, List[Dict[str, Any]]] = {}
    for projection in projections:
        by_view.setdefault(str(projection["view"]), []).append(projection)

    for view_projections in by_view.values():
        view_projections.sort(key=lambda item: item["median_depth"])
        nearer: List[Dict[str, Any]] = []
        for projection in view_projections:
            bbox = projection["bbox_xyxy"]
            area = max(float(projection["bbox_area"]), 1e-6)
            overlap = 0.0
            # For stability, sum pairwise overlaps and clamp. This may slightly
            # overestimate occlusion when nearer boxes overlap each other, which
            # is acceptable for filtering poor DINO ROIs.
            for previous in nearer:
                overlap += bbox_intersection_area(bbox, previous["bbox_xyxy"])
            occlusion_ratio = float(np.clip(overlap / area, 0.0, 1.0))
            projection["occlusion_ratio"] = occlusion_ratio
            projection["visible_box_ratio"] = float(np.clip(1.0 - occlusion_ratio, 0.0, 1.0))
            nearer.append(projection)

def annotate_camera_visibility(
    candidates: Dict[str, Dict[str, Any]],
    frame_data: List[Dict[str, Any]],
    current_idx: int,
    views: List[str],
    min_bbox_area: float,
    min_visible_ratio: float,
    min_visible_corners: int,
    max_occlusion_ratio: float,
) -> None:
    cameras = frame_data[current_idx]["cameras"]
    image_hw_cache: Dict[str, tuple[int, int]] = {}
    all_projections: List[Dict[str, Any]] = []

    for key, candidate in candidates.items():
        for view in views:
            camera = cameras.get(view, {})
            image_path = camera.get("image_path")
            if not image_path:
                continue
            if view not in image_hw_cache:
                image_hw_cache[view] = image_hw_from_path(image_path)
            projection = project_box_to_camera(
                candidate["box_current"],
                camera,
                image_hw_cache[view],
                min_visible_corners,
            )
            if projection is None:
                continue
            projection["candidate_key"] = key
            projection["view"] = view
            projection["image_path"] = image_path
            all_projections.append(projection)

    estimate_occlusion_ratios(all_projections)

    grouped: Dict[str, List[Dict[str, Any]]] = {key: [] for key in candidates}
    for projection in all_projections:
        if projection["bbox_area"] < min_bbox_area:
            continue
        if projection["visible_ratio"] < min_visible_ratio:
            continue
        if projection.get("occlusion_ratio", 0.0) > max_occlusion_ratio:
            continue
        grouped[str(projection["candidate_key"])].append(projection)

    for key, candidate in candidates.items():
        projections = grouped.get(key, [])
        projections.sort(
            key=lambda item: (
                item.get("visible_box_ratio", 1.0) * item["bbox_area"],
                1.0 if item["view"] == "cam_f0" else 0.0,
                -item["center_offset"],
            ),
            reverse=True,
        )
        candidate["camera_projections"] = projections
        candidate["best_camera_projection"] = projections[0] if projections else None

def box_corners_xy(box: Iterable[float]) -> np.ndarray:
    box = np.asarray(list(box), dtype=np.float64)
    x, y, _, length, width, _, yaw = box[:7]
    local = np.array(
        [
            [length / 2.0, width / 2.0],
            [length / 2.0, -width / 2.0],
            [-length / 2.0, -width / 2.0],
            [-length / 2.0, width / 2.0],
        ],
        dtype=np.float64,
    )
    return local @ rotation_matrix(float(yaw)).T + np.array([x, y], dtype=np.float64)


def visualize_selection(
    container: Dict[str, Any],
    payload: Dict[str, Any],
    args: argparse.Namespace,
    output_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frame_data = container["frame_data"]
    ego_path = ego_future_path_current_frame(frame_data, payload["frame_idx"])

    fig, ax = plt.subplots(figsize=(8, 8), dpi=140)
    ax.set_title(f"{payload['token']} critical agents")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linewidth=0.4, alpha=0.4)
    ax.set_xlabel("x forward (m)")
    ax.set_ylabel("y left (m)")

    ax.plot([0], [0], marker="o", color="black", markersize=5, label="ego")
    ego_box = np.array([[2.2, 1.0], [2.2, -1.0], [-2.2, -1.0], [-2.2, 1.0], [2.2, 1.0]])
    ax.plot(ego_box[:, 0], ego_box[:, 1], color="black", linewidth=1.2)

    if len(ego_path):
        ax.plot(ego_path[:, 0], ego_path[:, 1], color="#1f77b4", linewidth=2.0, label="ego future")
        ax.scatter(ego_path[:, 0], ego_path[:, 1], color="#1f77b4", s=10)

    colors = ["#d62728", "#ff7f0e", "#9467bd", "#2ca02c", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22"]
    for agent in payload["critical_agents"]:
        if not agent.get("valid", False):
            continue
        rank = int(agent["rank"])
        color = colors[rank % len(colors)]
        corners = box_corners_xy(agent["box_ego"])
        closed = np.vstack([corners, corners[0]])
        ax.plot(closed[:, 0], closed[:, 1], color=color, linewidth=1.8)
        ax.fill(corners[:, 0], corners[:, 1], color=color, alpha=0.12)
        cx, cy = float(agent["box_ego"][0]), float(agent["box_ego"][1])
        label = f"#{rank} {agent['class_name']} {agent['score']:.2f}"
        ax.text(cx, cy, label, color=color, fontsize=8, weight="bold")

    ax.set_xlim(args.viz_x_min, args.viz_x_max)
    ax.set_ylim(args.viz_y_min, args.viz_y_max)
    ax.legend(loc="upper right", fontsize=8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def visualize_camera_bboxes(payload: Dict[str, Any], output_dir: Path) -> None:
    from PIL import Image, ImageDraw, ImageFont

    colors = [
        (214, 39, 40),
        (255, 127, 14),
        (148, 103, 189),
        (44, 160, 44),
        (140, 86, 75),
        (227, 119, 194),
        (127, 127, 127),
        (188, 189, 34),
    ]
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for agent in payload.get("critical_agents", []):
        if not agent.get("valid", False):
            continue
        image_path = agent.get("image_path")
        if not image_path or agent.get("bbox_xyxy") is None:
            continue
        grouped.setdefault(image_path, []).append(agent)

    token = payload["token"]
    for image_path, agents in grouped.items():
        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image)
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 18)
        except Exception:
            font = ImageFont.load_default()

        for agent in agents:
            rank = int(agent["rank"])
            color = colors[rank % len(colors)]
            x1, y1, x2, y2 = [float(v) for v in agent["bbox_xyxy"]]
            for offset in range(3):
                draw.rectangle([x1 - offset, y1 - offset, x2 + offset, y2 + offset], outline=color)
            label = (
                f"#{rank} {agent.get('class_name')} "
                f"s={float(agent.get('score', 0.0)):.2f} "
                f"occ={float(agent.get('occlusion_ratio', 0.0)):.2f}"
            )
            text_bbox = draw.textbbox((x1, y1), label, font=font)
            text_h = text_bbox[3] - text_bbox[1]
            text_w = text_bbox[2] - text_bbox[0]
            label_y = max(0.0, y1 - text_h - 4)
            draw.rectangle([x1, label_y, x1 + text_w + 6, label_y + text_h + 4], fill=color)
            draw.text((x1 + 3, label_y + 2), label, fill=(255, 255, 255), font=font)

        view = next((agent.get("view") for agent in agents if agent.get("view")), "camera")
        output_path = output_dir / f"{token}_{view}.jpg"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path, quality=92)

def mine_one(container: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    frame_data = container["frame_data"]
    current_idx = min(args.current_frame_index, len(frame_data) - 1)
    ego_path = ego_future_path_current_frame(frame_data, current_idx)
    events = ego_motion_events(frame_data, current_idx)
    candidates = collect_agent_tracks_current_frame(
        frame_data=frame_data,
        current_idx=current_idx,
        allowed_classes=set(args.agent_classes),
        max_distance=args.max_distance,
        min_distance=args.min_distance,
    )
    annotate_camera_visibility(
        candidates,
        frame_data=frame_data,
        current_idx=current_idx,
        views=args.views,
        min_bbox_area=args.min_bbox_area,
        min_visible_ratio=args.min_visible_ratio,
        min_visible_corners=args.min_visible_corners,
        max_occlusion_ratio=args.max_occlusion_ratio,
    )
    candidates = {
        key: value for key, value in candidates.items()
        if value.get("best_camera_projection") is not None
    }
    scored = score_candidates(
        candidates,
        ego_path=ego_path,
        events=events,
        inner_width=args.inner_corridor_width,
        outer_width=args.outer_corridor_width,
    )
    ordered = order_agents(scored, args.selection_mode)
    agents = pad_agents(ordered, args.top_k)
    return {
        "token": container["token"],
        "frame_idx": current_idx,
        "schema_version": 2,
        "agent_selection_mode": args.selection_mode,
        "ego_motion_events": events,
        "critical_agents": agents,
        "visible_agents": agents,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="train", choices=("train", "mini", "test", "navhard_two_stage"))
    parser.add_argument(
        "--data-root",
        default=os.environ.get("OPENSCENE_DATA_ROOT") or os.environ.get("NAVSIM_PUBLIC_ROOT") or "/mnt/data/navsim",
        help="NAVSIM root. The script also supports mini_navsim_logs/mini layout.",
    )
    parser.add_argument("--log-dir", default=None, help="Optional explicit NAVSIM log split dir, e.g. /mnt/data/navsim/mini_navsim_logs/mini.")
    parser.add_argument("--sensor-dir", default=None, help="Optional explicit sensor split dir, e.g. /mnt/data/navsim/sensor_blobs/mini.")
    parser.add_argument("--output-dir", default="navsim_dataset/critical_agents")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--selection-mode", choices=("critical", "visible"), default="critical")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--tokens-file", default=None, help="Optional newline-delimited scene tokens to process.")
    parser.add_argument("--current-frame-index", type=int, default=CURRENT_FRAME_INDEX)
    parser.add_argument("--min-distance", type=float, default=1.5)
    parser.add_argument("--max-distance", type=float, default=60.0)
    parser.add_argument("--inner-corridor-width", type=float, default=2.5)
    parser.add_argument("--outer-corridor-width", type=float, default=6.0)
    parser.add_argument("--agent-classes", nargs="+", default=sorted(DEFAULT_AGENT_CLASSES))
    parser.add_argument("--views", nargs="+", default=["cam_f0", "cam_l0", "cam_r0"])
    parser.add_argument("--min-bbox-area", type=float, default=256.0)
    parser.add_argument("--min-visible-ratio", type=float, default=0.25)
    parser.add_argument("--min-visible-corners", type=int, default=1)
    parser.add_argument("--max-occlusion-ratio", type=float, default=0.6)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--visualize-dir", default=None, help="Optional directory for BEV debug images.")
    parser.add_argument("--visualize-camera-dir", default=None, help="Optional directory for camera bbox debug images.")
    parser.add_argument("--visualize-limit", type=int, default=32, help="Maximum number of debug images to write.")
    parser.add_argument("--viz-x-min", type=float, default=-15.0)
    parser.add_argument("--viz-x-max", type=float, default=65.0)
    parser.add_argument("--viz-y-min", type=float, default=-35.0)
    parser.add_argument("--viz-y-max", type=float, default=35.0)
    return parser.parse_args()



def resolve_split_dirs(args: argparse.Namespace) -> tuple[Path, Path]:
    data_root = Path(args.data_root).resolve()
    split = args.split

    if args.log_dir is not None:
        log_dir = Path(args.log_dir).resolve()
    elif (data_root / "navsim_logs" / split).is_dir():
        log_dir = data_root / "navsim_logs" / split
    elif split == "mini" and (data_root / "mini_navsim_logs" / "mini").is_dir():
        log_dir = data_root / "mini_navsim_logs" / "mini"
    elif split == "train" and (data_root / "navsim_logs" / "trainval").is_dir():
        log_dir = data_root / "navsim_logs" / "trainval"
    else:
        raise FileNotFoundError(
            f"Could not resolve NAVSIM log dir for split={split!r} under {data_root}. "
            "Pass --log-dir explicitly."
        )

    if args.sensor_dir is not None:
        sensor_dir = Path(args.sensor_dir).resolve()
    elif (data_root / "sensor_blobs" / split).is_dir():
        sensor_dir = data_root / "sensor_blobs" / split
    elif split == "train" and (data_root / "sensor_blobs" / "trainval").is_dir():
        sensor_dir = data_root / "sensor_blobs" / "trainval"
    elif split == "mini" and (data_root / "mini_sensor_blobs" / "mini").is_dir():
        sensor_dir = data_root / "mini_sensor_blobs" / "mini"
    else:
        raise FileNotFoundError(
            f"Could not resolve NAVSIM sensor dir for split={split!r} under {data_root}. "
            "Pass --sensor-dir explicitly."
        )

    return log_dir, sensor_dir


def prepare_vlmnavsim_root(args: argparse.Namespace) -> Path:
    log_dir, sensor_dir = resolve_split_dirs(args)
    split = args.split
    loader_split = "trainval" if split == "train" else split
    runtime_root = Path("/tmp") / "vla_drive_navsim_mining" / split
    expected_log = runtime_root / "navsim_logs" / loader_split
    expected_sensor = runtime_root / "sensor_blobs" / loader_split

    expected_log.parent.mkdir(parents=True, exist_ok=True)
    expected_sensor.parent.mkdir(parents=True, exist_ok=True)
    for link, target in ((expected_log, log_dir), (expected_sensor, sensor_dir)):
        if link.is_symlink() or link.exists():
            if link.resolve() == target:
                continue
            if link.is_dir() and not link.is_symlink():
                shutil.rmtree(link)
            else:
                link.unlink()
        link.symlink_to(target, target_is_directory=True)

    print(f"[critical-agents] log_dir={log_dir}")
    print(f"[critical-agents] sensor_dir={sensor_dir}")
    print(f"[critical-agents] runtime_root={runtime_root}")
    return runtime_root

def main() -> None:
    args = parse_args()
    os.environ["OPENSCENE_DATA_ROOT"] = str(prepare_vlmnavsim_root(args))
    import hydra.utils

    original_instantiate = hydra.utils.instantiate

    def instantiate_with_scene_filter_defaults(*inst_args, **inst_kwargs):
        obj = original_instantiate(*inst_args, **inst_kwargs)
        if obj.__class__.__name__ == "SceneFilter":
            if not hasattr(obj, "include_synthetic_scenes"):
                setattr(obj, "include_synthetic_scenes", False)
            if not hasattr(obj, "synthetic_scene_tokens"):
                setattr(obj, "synthetic_scene_tokens", None)
        return obj

    hydra.utils.instantiate = instantiate_with_scene_filter_defaults

    from data_engine.datasets.navsim import dataset_navsim as dataset_navsim_module
    dataset_navsim_module.instantiate = instantiate_with_scene_filter_defaults
    VLMNavsim = dataset_navsim_module.VLMNavsim

    output_dir = Path(args.output_dir) / args.split
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = VLMNavsim(mode=args.split)
    if args.tokens_file is not None:
        requested_tokens = [
            line.strip() for line in Path(args.tokens_file).read_text().splitlines()
            if line.strip()
        ]
        token_to_index = {token: idx for idx, token in enumerate(dataset.navsim._scene_loader.tokens)}
        missing_tokens = [token for token in requested_tokens if token not in token_to_index]
        indices = [token_to_index[token] for token in requested_tokens if token in token_to_index]
        if args.max_samples is not None:
            indices = indices[: args.max_samples]
        print(f"[critical-agents] requested_tokens={len(requested_tokens)} matched={len(indices)} missing={len(missing_tokens)}")
        if missing_tokens:
            missing_path = output_dir / "missing_tokens.txt"
            missing_path.write_text("\n".join(missing_tokens) + "\n")
            print(f"[critical-agents] missing token list written to {missing_path}")
    else:
        total = len(dataset) if args.max_samples is None else min(len(dataset), args.max_samples)
        indices = list(range(total))

    failures = 0
    for loop_index, index in enumerate(tqdm(indices, desc=f"Mining critical agents ({args.split})")):
        container = dataset.get_container_in(index)
        if container is None:
            continue
        output_path = output_dir / f"{container['token']}.json"
        if output_path.exists() and not args.overwrite:
            continue
        try:
            payload = mine_one(container, args)
        except Exception as exc:
            failures += 1
            payload = {
                "token": container.get("token", str(index)),
                "frame_idx": args.current_frame_index,
                "schema_version": 2,
                "error": repr(exc),
                "agent_selection_mode": args.selection_mode,
                "critical_agents": pad_agents([], args.top_k),
                "visible_agents": pad_agents([], args.top_k),
            }
        with output_path.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2)
        if loop_index < args.visualize_limit and "error" not in payload:
            if args.visualize_dir is not None:
                visualize_path = Path(args.visualize_dir) / args.split / f"{container['token']}.png"
                visualize_selection(container, payload, args, visualize_path)
            if args.visualize_camera_dir is not None:
                visualize_camera_bboxes(payload, Path(args.visualize_camera_dir) / args.split)

    print(f"wrote {args.selection_mode}-agent sidecars to {output_dir}")
    if args.visualize_dir is not None:
        print(f"wrote BEV visualizations to {Path(args.visualize_dir) / args.split}")
    if args.visualize_camera_dir is not None:
        print(f"wrote camera bbox visualizations to {Path(args.visualize_camera_dir) / args.split}")
    if failures:
        print(f"warning: {failures} samples failed and were written with empty agents")


if __name__ == "__main__":
    main()
