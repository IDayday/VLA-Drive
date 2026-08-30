#!/usr/bin/env python3
"""Phase 8: small leak-audited planning-utility oracle probes.

The probe consumes the deployed trajectory-aligned effect-tube cache.  Its
channels are candidate-conditioned relabelings of a single logged future and
are therefore a dense, 500-scene-compatible counterpart of the smoke-audited
``C_environment_only`` schema.  No official score or factor is an input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from typing import Any, Sequence

import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    ndcg_score,
    roc_auc_score,
)
from sklearn.preprocessing import LabelEncoder

from .common import (
    add_common_arguments,
    consequence_rows_by_scene,
    discover_paths,
    load_candidate_source,
    load_metric_cache_index,
    metric_log_name,
    raw_log_path,
    trajectory_kinematics,
    wrap_heading,
    write_json,
    write_text,
)


EFFECT_CHANNELS = (
    "candidate_relative_dynamic_occupancy",
    "drivable_area_sdf",
    "lane_sdf",
    "route_sdf",
    "relative_longitudinal_velocity",
    "relative_lateral_velocity",
    "dynamic_clearance",
    "dynamic_collision_field",
    "ego_swept_footprint",
)
ENVIRONMENT_CHANNEL_INDICES = tuple(range(8))
INTERACTION_CHANNEL_INDICES = (0, 4, 5, 6, 7)
EFFECT_HORIZONS_S = (1.0, 2.0, 4.0)
TUBE_STATISTICS = ("mean", "std", "min", "max", "q10", "q90", "nonzero_fraction")
CURRENT_ACTOR_HORIZONS_S = EFFECT_HORIZONS_S
CURRENT_ACTOR_SLOTS = 8

TARGETS = {
    "aggregate_score": ("regression", "log_replay", "pdm_score"),
    "collision": ("binary", "log_replay", "no_at_fault_collision"),
    "ttc_violation": ("binary", "log_replay", "time_to_collision_within_bound"),
    "dac": ("binary", "exact", "drivable_area_compliance"),
    "ddc": ("binary", "exact", "driving_direction_compliance"),
    "tlc": ("binary", "log_replay", "traffic_light_compliance"),
    "progress": ("regression", "exact", "official_ego_progress_score"),
}


def finite_array(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=np.float32)
    return np.nan_to_num(result, nan=0.0, posinf=1e4, neginf=-1e4)


def trajectory_features(trajectory: np.ndarray) -> tuple[np.ndarray, list[str]]:
    poses = np.asarray(trajectory, dtype=np.float64)
    kinematics = trajectory_kinematics(poses)
    values: list[float] = []
    names: list[str] = []
    for time_index, time_s in enumerate(np.arange(1, 9) * 0.5):
        for coordinate, unit in zip(("x", "y", "heading"), ("m", "m", "rad")):
            coordinate_index = {"x": 0, "y": 1, "heading": 2}[coordinate]
            values.append(float(poses[time_index, coordinate_index]))
            names.append(f"trajectory_{coordinate}_t{time_s:g}s_{unit}")
    for key, unit in (
        ("speed", "mps"),
        ("acceleration", "mps2"),
        ("yaw_rate", "radps"),
        ("curvature", "inv_m"),
        ("jerk", "mps3"),
    ):
        for time_index, value in enumerate(np.asarray(kinematics[key])):
            values.append(float(value))
            names.append(f"trajectory_{key}_t{(time_index + 1) * 0.5:g}s_{unit}")
    values.append(float(kinematics["terminal_displacement"]))
    names.append("trajectory_terminal_displacement_m")
    return finite_array(np.asarray(values)), names


def current_scene_features(frame: dict[str, Any]) -> tuple[np.ndarray, list[str]]:
    """Extract only planning-instant fields from a raw NAVSIM train frame."""

    dynamic = np.asarray(frame.get("ego_dynamic_state", np.zeros(4)), dtype=np.float64)
    annotations = frame.get("anns", {})
    boxes = np.asarray(annotations.get("gt_boxes", np.zeros((0, 7))), dtype=np.float64)
    velocities = np.asarray(
        annotations.get("gt_velocity_3d", np.zeros((len(boxes), 3))), dtype=np.float64
    )
    names_raw = np.asarray(
        annotations.get("gt_names", np.asarray([], dtype=str))
    ).astype(str)
    distances = np.linalg.norm(boxes[:, :2], axis=1) if len(boxes) else np.asarray([])
    actor_speeds = (
        np.linalg.norm(velocities[:, :2], axis=1) if len(velocities) else np.asarray([])
    )
    names_lower = np.char.lower(names_raw) if len(names_raw) else names_raw
    traffic_lights = frame.get("traffic_lights", ())
    command = np.asarray(frame.get("driving_command", np.zeros(4)), dtype=np.float64)
    if command.shape != (4,):
        command = np.zeros(4, dtype=np.float64)
    names = [
        "current_ego_speed_mps",
        "current_ego_acceleration_mps2",
        "current_ego_lateral_speed_mps",
        "current_ego_lateral_acceleration_mps2",
        "current_actor_count",
        "current_vehicle_count",
        "current_pedestrian_count",
        "current_bicycle_count",
        "current_nearest_actor_distance_m",
        "current_mean_actor_speed_mps",
        "current_roadblock_count",
        "current_traffic_light_count",
        "current_red_light_count",
        "current_command_straight",
        "current_command_left",
        "current_command_right",
        "current_command_unknown",
    ]
    values = np.asarray(
        [
            np.linalg.norm(dynamic[:2]),
            np.linalg.norm(dynamic[2:4]),
            dynamic[1] if len(dynamic) > 1 else 0.0,
            dynamic[3] if len(dynamic) > 3 else 0.0,
            len(boxes),
            np.sum(names_lower == "vehicle"),
            np.sum(names_lower == "pedestrian"),
            np.sum((names_lower == "bicycle") | (names_lower == "cyclist")),
            float(np.min(distances)) if len(distances) else 100.0,
            float(np.mean(actor_speeds)) if len(actor_speeds) else 0.0,
            len(frame.get("roadblock_ids", ())),
            len(traffic_lights),
            sum(bool(is_red) for _, is_red in traffic_lights),
            *command.tolist(),
        ],
        dtype=np.float32,
    )
    return finite_array(values), names


def candidate_current_actor_features(
    frame: dict[str, Any],
    trajectory: np.ndarray,
    *,
    max_actors: int = CURRENT_ACTOR_SLOTS,
) -> tuple[np.ndarray, list[str]]:
    """Candidate-conditioned features using only the planning-instant frame.

    Current annotated actors are propagated with a constant-velocity prior and
    expressed in each candidate pose. This is an online-available structured
    input proxy, not a logged-future target. Fixed actor slots are sorted at
    every horizon by candidate-relative center distance.
    """

    if max_actors <= 0:
        raise ValueError("max_actors must be positive")
    poses = np.asarray(trajectory, dtype=np.float64)
    if poses.shape != (8, 3):
        raise ValueError(f"expected trajectory [8,3], got {poses.shape}")
    annotations = frame.get("anns", {})
    boxes = np.asarray(
        annotations.get("gt_boxes", np.zeros((0, 7))), dtype=np.float64
    )
    velocities = np.asarray(
        annotations.get("gt_velocity_3d", np.zeros((len(boxes), 3))),
        dtype=np.float64,
    )
    object_names = np.asarray(
        annotations.get("gt_names", np.asarray([], dtype=str))
    ).astype(str)
    if boxes.ndim != 2 or (len(boxes) and boxes.shape[1] < 7):
        raise ValueError(f"current gt_boxes must be [N,>=7], got {boxes.shape}")
    if velocities.shape[0] != len(boxes) or object_names.shape[0] != len(boxes):
        raise ValueError("current actor fields have inconsistent lengths")
    if len(boxes):
        actor_position = finite_array(boxes[:, :2]).astype(np.float64)
        actor_velocity = finite_array(velocities[:, :2]).astype(np.float64)
        actor_heading = finite_array(boxes[:, 6]).astype(np.float64)
        actor_length = np.maximum(
            finite_array(boxes[:, 3]).astype(np.float64), 0.0
        )
        actor_width = np.maximum(
            finite_array(boxes[:, 4]).astype(np.float64), 0.0
        )
    else:
        actor_position = np.zeros((0, 2), dtype=np.float64)
        actor_velocity = np.zeros((0, 2), dtype=np.float64)
        actor_heading = np.zeros(0, dtype=np.float64)
        actor_length = np.zeros(0, dtype=np.float64)
        actor_width = np.zeros(0, dtype=np.float64)

    slot_fields = (
        "valid",
        "relative_x_m",
        "relative_y_m",
        "relative_vx_mps",
        "relative_vy_mps",
        "relative_heading_rad",
        "length_m",
        "width_m",
        "center_distance_m",
        "front_ttc_s",
        "type_vehicle",
        "type_pedestrian",
        "type_bicycle",
        "type_other",
    )
    aggregate_fields = (
        "actor_count",
        "within_10m_count",
        "within_20m_count",
        "within_40m_count",
        "front_corridor_count",
        "crossing_corridor_count",
        "closing_front_count",
        "minimum_center_distance_m",
        "mean_nearest4_distance_m",
        "minimum_front_ttc_s",
        "mean_absolute_lateral_offset_m",
    )
    values: list[float] = []
    names: list[str] = []
    previous_xy = np.vstack(
        [np.zeros((1, 2), dtype=np.float64), poses[:-1, :2]]
    )
    candidate_velocity = (poses[:, :2] - previous_xy) / 0.5

    for horizon in CURRENT_ACTOR_HORIZONS_S:
        pose_index = int(round(horizon / 0.5)) - 1
        candidate_pose = poses[pose_index]
        candidate_v = candidate_velocity[pose_index]
        cosine = float(np.cos(candidate_pose[2]))
        sine = float(np.sin(candidate_pose[2]))
        forecast_position = actor_position + actor_velocity * horizon
        delta = forecast_position - candidate_pose[:2]
        relative_x = cosine * delta[:, 0] + sine * delta[:, 1]
        relative_y = -sine * delta[:, 0] + cosine * delta[:, 1]
        velocity_delta = actor_velocity - candidate_v
        relative_vx = cosine * velocity_delta[:, 0] + sine * velocity_delta[:, 1]
        relative_vy = -sine * velocity_delta[:, 0] + cosine * velocity_delta[:, 1]
        relative_heading = np.asarray(
            wrap_heading(actor_heading - candidate_pose[2]), dtype=np.float64
        )
        center_distance = np.hypot(relative_x, relative_y)
        front_ttc = np.full(len(boxes), 20.0, dtype=np.float64)
        closing = (relative_x > 0.0) & (relative_vx < -0.1)
        front_ttc[closing] = np.minimum(
            relative_x[closing] / np.maximum(-relative_vx[closing], 0.1),
            20.0,
        )
        order = np.argsort(center_distance, kind="stable")
        for slot in range(max_actors):
            prefix = f"current_cv_h{horizon:g}s_actor{slot:02d}"
            names.extend(f"{prefix}_{field}" for field in slot_fields)
            if slot >= len(order):
                values.extend([0.0] * len(slot_fields))
                continue
            index = int(order[slot])
            object_name = object_names[index].lower()
            is_vehicle = float(object_name == "vehicle")
            is_pedestrian = float(object_name == "pedestrian")
            is_bicycle = float(object_name in {"bicycle", "cyclist"})
            is_other = float(not (is_vehicle or is_pedestrian or is_bicycle))
            values.extend(
                [
                    1.0,
                    relative_x[index],
                    relative_y[index],
                    relative_vx[index],
                    relative_vy[index],
                    relative_heading[index],
                    actor_length[index],
                    actor_width[index],
                    center_distance[index],
                    front_ttc[index],
                    is_vehicle,
                    is_pedestrian,
                    is_bicycle,
                    is_other,
                ]
            )
        names.extend(
            f"current_cv_h{horizon:g}s_{field}"
            for field in aggregate_fields
        )
        nearest = np.sort(center_distance)[:4]
        values.extend(
            [
                len(boxes),
                np.sum(center_distance < 10.0),
                np.sum(center_distance < 20.0),
                np.sum(center_distance < 40.0),
                np.sum((relative_x > 0.0) & (np.abs(relative_y) < 3.0)),
                np.sum(
                    (np.abs(relative_x) < 6.0)
                    & (np.abs(relative_y) < 8.0)
                ),
                np.sum(closing & (np.abs(relative_y) < 3.0)),
                float(np.min(center_distance))
                if len(center_distance)
                else 100.0,
                float(np.mean(nearest)) if len(nearest) else 100.0,
                float(np.min(front_ttc[closing]))
                if np.any(closing)
                else 20.0,
                float(np.mean(np.abs(relative_y)))
                if len(relative_y)
                else 100.0,
            ]
        )
    return finite_array(np.asarray(values)), names


def tube_features(
    tube: np.ndarray,
    channel_indices: Sequence[int],
    prefix: str,
) -> tuple[np.ndarray, list[str]]:
    values: list[float] = []
    names: list[str] = []
    for horizon_index, horizon in enumerate(EFFECT_HORIZONS_S):
        for channel_index in channel_indices:
            grid = np.asarray(tube[horizon_index, channel_index], dtype=np.float32)
            stats = (
                np.mean(grid),
                np.std(grid),
                np.min(grid),
                np.max(grid),
                np.quantile(grid, 0.10),
                np.quantile(grid, 0.90),
                np.mean(np.abs(grid) > 1e-3),
            )
            for statistic, value in zip(TUBE_STATISTICS, stats):
                values.append(float(value))
                names.append(
                    f"{prefix}_h{horizon:g}s_{EFFECT_CHANNELS[channel_index]}_{statistic}"
                )
    return finite_array(np.asarray(values)), names


def behavior_label(metadata: dict[str, Any]) -> str:
    kind = str(metadata["perturbation_type"])
    parameters = metadata.get("perturbation_parameters", {})
    if kind == "anchor":
        return "anchor"
    if kind in {"lateral_terminal_offset", "turn_inner_outer_offset"}:
        return "left" if float(parameters.get("offset_m", 0.0)) > 0 else "right"
    if kind == "speed_scale":
        return "fast" if float(parameters.get("scale", 1.0)) > 1 else "slow"
    if kind == "brake_onset_shift":
        return "go" if float(parameters.get("shift_s", 0.0)) > 0 else "yield"
    if kind == "terminal_progress_shift":
        return "go" if float(parameters.get("shift_m", 0.0)) > 0 else "yield"
    if kind == "curvature_scale":
        return (
            "outer_curve" if float(parameters.get("scale", 1.0)) < 1 else "inner_curve"
        )
    return kind


def target_value(row: dict[str, Any], name: str) -> float:
    target_type, namespace, field = TARGETS[name]
    value = row.get(namespace, {}).get(field)
    if value is None:
        return float("nan")
    result = float(value)
    if name in {"collision", "ttc_violation"}:
        result = 1.0 - result
    if target_type == "binary":
        result = float(result >= 0.5)
    return result


def split_is_validation(log_name: str, seed: int) -> bool:
    digest = hashlib.sha256(f"{seed}:{log_name}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % 100 < 20


def choose_candidate_rows(
    rows: Sequence[dict[str, Any]],
    metadata_by_index: dict[int, dict[str, Any]],
    num_candidates: int,
) -> list[dict[str, Any]]:
    accepted = [
        row
        for row in rows
        if row.get("candidate_accepted") and row.get("log_replay", {}).get("available")
    ]
    accepted.sort(
        key=lambda row: (
            metadata_by_index[int(row["candidate_index"])].get("perturbation_type")
            != "anchor",
            metadata_by_index[int(row["candidate_index"])].get("perturbation_type", ""),
            int(row["candidate_index"]),
        )
    )
    selected = accepted[:num_candidates]
    if (
        len(selected) < num_candidates
        or metadata_by_index[int(selected[0]["candidate_index"])]["perturbation_type"]
        != "anchor"
    ):
        return []
    return selected


def build_dataset(args: argparse.Namespace) -> dict[str, Any]:
    split = "trainval" if args.split == "train" else args.split
    paths = discover_paths(args, split=split)
    if (
        paths.metric_cache is None
        or paths.effect_tube_cache is None
        or paths.logs_root is None
    ):
        raise FileNotFoundError(
            "MetricCache index, raw train logs, and the deployed effect-tube cache are required"
        )
    trajectories, metadata, _ = load_candidate_source(paths)
    metadata_by_index = {int(item["trajectory"]["index"]): item for item in metadata}
    consequences = consequence_rows_by_scene(paths)
    metric_index = load_metric_cache_index(paths.metric_cache)
    effect_index = json.loads(
        (paths.effect_tube_cache / "scene_index.json").read_text()
    )
    candidate_tensors: list[np.ndarray] = []
    scene_tensors: list[np.ndarray] = []
    environment_tensors: list[np.ndarray] = []
    interaction_tensors: list[np.ndarray] = []
    current_candidate_tensors: list[np.ndarray] = []
    targets: dict[str, list[float]] = {name: [] for name in TARGETS}
    scene_tokens: list[str] = []
    log_names: list[str] = []
    candidate_slots: list[int] = []
    behavior_labels: list[str] = []
    candidate_names: list[str] | None = None
    current_names: list[str] | None = None
    environment_names: list[str] | None = None
    interaction_names: list[str] | None = None
    current_candidate_names: list[str] | None = None
    eligible_tokens = [
        token
        for token in sorted(consequences)
        if token in effect_index and token in metric_index
    ]
    eligible_by_log: dict[str, list[str]] = {}
    for token in eligible_tokens:
        eligible_by_log.setdefault(metric_log_name(metric_index[token]), []).append(
            token
        )
    # Bound scenes from a single log so the validation set contains many complete
    # logs while each raw pickle is still read only once.
    selected_by_log: dict[str, list[tuple[str, list[dict[str, Any]]]]] = {}
    remaining = args.max_scenes
    for log_name in sorted(eligible_by_log):
        for token in eligible_by_log[log_name]:
            selected = choose_candidate_rows(
                consequences[token], metadata_by_index, args.num_candidates
            )
            if selected:
                selected_by_log.setdefault(log_name, []).append((token, selected))
                remaining -= 1
            if (
                len(selected_by_log.get(log_name, ())) >= args.max_scenes_per_log
                or remaining <= 0
            ):
                break
        if remaining <= 0:
            break
    selected_scene_count = 0
    failures: list[dict[str, str]] = []
    for log_name, log_scenes in selected_by_log.items():
        try:
            with raw_log_path(paths.logs_root, log_name).open("rb") as stream:
                raw_frames = pickle.load(stream)
            requested = {token for token, _ in log_scenes}
            frames_by_token = {
                str(frame["token"]): frame
                for frame in raw_frames
                if str(frame.get("token")) in requested
            }
        except Exception as error:
            failures.extend(
                {
                    "scene_token": token,
                    "error": f"raw log {type(error).__name__}: {error}",
                }
                for token, _ in log_scenes
            )
            continue
        for token, selected in log_scenes:
            try:
                current, current_feature_names = current_scene_features(
                    frames_by_token[token]
                )
                entry = effect_index[token]
                with np.load(paths.effect_tube_cache / entry["file"]) as payload:
                    tubes = np.asarray(payload["target"], dtype=np.float32)
                    valid = np.asarray(payload["valid"], dtype=bool)
                for candidate_rank, row in enumerate(selected):
                    source_slot = int(row["scene_candidate_index"])
                    if not valid[source_slot]:
                        raise ValueError(
                            f"effect tube invalid at source slot {source_slot}"
                        )
                    candidate, candidate_feature_names = trajectory_features(
                        trajectories[int(row["candidate_index"])]
                    )
                    environment, environment_feature_names = tube_features(
                        tubes[source_slot],
                        ENVIRONMENT_CHANNEL_INDICES,
                        "C_environment_only",
                    )
                    interaction, interaction_feature_names = tube_features(
                        tubes[source_slot],
                        INTERACTION_CHANNEL_INDICES,
                        "C_interaction_only",
                    )
                    current_candidate, current_candidate_feature_names = (
                        candidate_current_actor_features(
                            frames_by_token[token],
                            trajectories[int(row["candidate_index"])],
                        )
                    )
                    candidate_tensors.append(candidate)
                    scene_tensors.append(current)
                    environment_tensors.append(environment)
                    interaction_tensors.append(interaction)
                    current_candidate_tensors.append(current_candidate)
                    for name in TARGETS:
                        targets[name].append(target_value(row, name))
                    scene_tokens.append(token)
                    log_names.append(log_name)
                    candidate_slots.append(candidate_rank)
                    behavior_labels.append(
                        behavior_label(metadata_by_index[int(row["candidate_index"])])
                    )
                    candidate_names = candidate_feature_names
                    current_names = current_feature_names
                    environment_names = environment_feature_names
                    interaction_names = interaction_feature_names
                    current_candidate_names = current_candidate_feature_names
            except Exception as error:  # retain a falsifiable failure inventory
                failures.append(
                    {"scene_token": token, "error": f"{type(error).__name__}: {error}"}
                )
                continue
            selected_scene_count += 1
    if not selected_scene_count:
        raise RuntimeError("no oracle scenes could be assembled")
    return {
        "trajectory": np.stack(candidate_tensors),
        "current": np.stack(scene_tensors),
        "environment": np.stack(environment_tensors),
        "interaction": np.stack(interaction_tensors),
        "current_candidate": np.stack(current_candidate_tensors),
        "targets": {
            name: np.asarray(values, dtype=np.float32)
            for name, values in targets.items()
        },
        "scene_tokens": np.asarray(scene_tokens),
        "log_names": np.asarray(log_names),
        "candidate_slots": np.asarray(candidate_slots, dtype=np.int16),
        "behavior_labels": np.asarray(behavior_labels),
        "feature_names": {
            "trajectory": candidate_names or [],
            "current": current_names or [],
            "environment": environment_names or [],
            "interaction": interaction_names or [],
            "current_candidate": current_candidate_names or [],
        },
        "failures": failures,
        "selected_scene_count": selected_scene_count,
        "paths": paths.to_json(),
    }


def fit_regressor(x_train: np.ndarray, y_train: np.ndarray, seed: int) -> Any:
    model = HistGradientBoostingRegressor(
        learning_rate=0.08,
        max_iter=100,
        max_leaf_nodes=15,
        l2_regularization=0.1,
        random_state=seed,
    )
    model.fit(x_train, y_train)
    return model


def ranking_metrics(
    truth: np.ndarray,
    prediction: np.ndarray,
    scene_tokens: np.ndarray,
) -> dict[str, float | int | None]:
    pair_correct = 0
    pair_count = 0
    top1_correct = 0
    regrets: list[float] = []
    ndcgs: list[float] = []
    scene_spearman: list[float] = []
    for token in np.unique(scene_tokens):
        mask = scene_tokens == token
        y = truth[mask]
        p = prediction[mask]
        if len(y) < 2:
            continue
        difference = y[:, None] - y[None, :]
        predicted_difference = p[:, None] - p[None, :]
        upper = np.triu_indices(len(y), k=1)
        valid = np.abs(difference[upper]) > 1e-8
        pair_correct += int(
            np.sum(
                np.sign(difference[upper][valid])
                == np.sign(predicted_difference[upper][valid])
            )
        )
        pair_count += int(np.sum(valid))
        selected = int(np.argmax(p))
        best = float(np.max(y))
        top1_correct += int(abs(float(y[selected]) - best) <= 1e-8)
        regrets.append(best - float(y[selected]))
        ndcgs.append(float(ndcg_score(y[None], p[None])))
        correlation = (
            spearmanr(y, p).statistic
            if np.std(y) >= 1e-12 and np.std(p) >= 1e-12
            else float("nan")
        )
        if np.isfinite(correlation):
            scene_spearman.append(float(correlation))
    global_spearman = (
        spearmanr(truth, prediction).statistic
        if np.std(truth) >= 1e-12 and np.std(prediction) >= 1e-12
        else float("nan")
    )
    return {
        "pairwise_ranking_accuracy": pair_correct / pair_count if pair_count else None,
        "pairwise_comparison_count": pair_count,
        "ndcg_mean": float(np.mean(ndcgs)) if ndcgs else None,
        "spearman_global": float(global_spearman)
        if np.isfinite(global_spearman)
        else None,
        "spearman_per_scene_mean": float(np.mean(scene_spearman))
        if scene_spearman
        else None,
        "top1_accuracy": top1_correct / len(regrets) if regrets else None,
        "top1_score_regret_mean": float(np.mean(regrets)) if regrets else None,
        "top1_score_regret_p95": float(np.quantile(regrets, 0.95)) if regrets else None,
        "rmse": float(mean_squared_error(truth, prediction, squared=False)),
        "mae": float(mean_absolute_error(truth, prediction)),
    }


def regression_target_metrics(
    truth: np.ndarray, prediction: np.ndarray
) -> dict[str, float | None]:
    correlation = (
        spearmanr(truth, prediction).statistic
        if np.std(truth) >= 1e-12 and np.std(prediction) >= 1e-12
        else float("nan")
    )
    return {
        "rmse": float(mean_squared_error(truth, prediction, squared=False)),
        "mae": float(mean_absolute_error(truth, prediction)),
        "spearman": float(correlation) if np.isfinite(correlation) else None,
    }


def binary_model_metrics(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    seed: int,
) -> dict[str, Any]:
    train_classes = np.unique(y_train)
    if len(train_classes) < 2:
        probability = np.full(len(y_validation), float(train_classes[0]))
        mode = "constant_single_train_class"
    else:
        model = HistGradientBoostingClassifier(
            learning_rate=0.08,
            max_iter=100,
            max_leaf_nodes=15,
            l2_regularization=0.1,
            random_state=seed,
        )
        model.fit(x_train, y_train.astype(int))
        probability = model.predict_proba(x_validation)[:, 1]
        mode = "hist_gradient_boosting"
    predicted = probability >= 0.5
    validation_classes = np.unique(y_validation)
    return {
        "model": mode,
        "support": int(len(y_validation)),
        "positive_rate": float(np.mean(y_validation)),
        "auroc": float(roc_auc_score(y_validation, probability))
        if len(validation_classes) == 2
        else None,
        "f1": float(f1_score(y_validation, predicted, zero_division=0)),
        "accuracy": float(accuracy_score(y_validation, predicted)),
        "brier_calibration_error": float(brier_score_loss(y_validation, probability)),
    }


def run_probes(dataset: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    validation_mask = np.asarray(
        [split_is_validation(str(name), args.seed) for name in dataset["log_names"]],
        dtype=bool,
    )
    train_mask = ~validation_mask
    if (
        len(np.unique(dataset["log_names"][validation_mask])) == 0
        or len(np.unique(dataset["log_names"][train_mask])) == 0
    ):
        raise RuntimeError("deterministic log split produced an empty partition")
    features = {
        "Probe_A_trajectory_only": dataset["trajectory"],
        "Probe_B_current_scene_plus_trajectory": np.concatenate(
            [dataset["trajectory"], dataset["current"]], axis=1
        ),
        "Probe_C_candidate_relative_future": np.concatenate(
            [dataset["trajectory"], dataset["current"], dataset["environment"]], axis=1
        ),
    }
    feature_names = {
        "Probe_A_trajectory_only": dataset["feature_names"]["trajectory"],
        "Probe_B_current_scene_plus_trajectory": dataset["feature_names"]["trajectory"]
        + dataset["feature_names"]["current"],
        "Probe_C_candidate_relative_future": dataset["feature_names"]["trajectory"]
        + dataset["feature_names"]["current"]
        + dataset["feature_names"]["environment"],
    }
    output: dict[str, Any] = {}
    for probe_name, x in features.items():
        probe: dict[str, Any] = {
            "feature_count": int(x.shape[1]),
            "model_family": "sklearn HistGradientBoosting, fixed small configuration",
        }
        y_score = dataset["targets"]["aggregate_score"]
        valid_score = np.isfinite(y_score)
        score_train = train_mask & valid_score
        score_validation = validation_mask & valid_score
        score_model = fit_regressor(x[score_train], y_score[score_train], args.seed)
        score_prediction = score_model.predict(x[score_validation])
        probe["aggregate_score_and_ranking"] = ranking_metrics(
            y_score[score_validation],
            score_prediction,
            dataset["scene_tokens"][score_validation],
        )
        factor_results: dict[str, Any] = {}
        for target_name, (target_type, _, _) in TARGETS.items():
            if target_name == "aggregate_score":
                continue
            y = dataset["targets"][target_name]
            train = train_mask & np.isfinite(y)
            validation = validation_mask & np.isfinite(y)
            if target_type == "regression":
                model = fit_regressor(x[train], y[train], args.seed)
                prediction = model.predict(x[validation])
                factor_results[target_name] = regression_target_metrics(
                    y[validation], prediction
                )
                factor_results[target_name]["support"] = int(np.sum(validation))
            else:
                factor_results[target_name] = binary_model_metrics(
                    x[train], y[train], x[validation], y[validation], args.seed
                )
        probe["factor_prediction"] = factor_results
        output[probe_name] = probe

    added_environment_names = dataset["feature_names"]["environment"]
    forbidden_fragments = (
        "pdm",
        "score",
        "official",
        "candidate_type",
        "candidate_id",
        "waypoint",
        "ego_swept_footprint",
    )
    offenders = [
        name
        for name in added_environment_names
        if any(fragment in name.lower() for fragment in forbidden_fragments)
    ]
    return {
        "split": {
            "unit": "complete log_name",
            "train_fraction_rule": "sha256(seed:log_name) mod 100 >= 20",
            "train_logs": int(len(np.unique(dataset["log_names"][train_mask]))),
            "validation_logs": int(
                len(np.unique(dataset["log_names"][validation_mask]))
            ),
            "train_scenes": int(len(np.unique(dataset["scene_tokens"][train_mask]))),
            "validation_scenes": int(
                len(np.unique(dataset["scene_tokens"][validation_mask]))
            ),
            "log_overlap": sorted(
                set(dataset["log_names"][train_mask])
                & set(dataset["log_names"][validation_mask])
            ),
        },
        "probes": output,
        "leakage_audit": {
            "pass": not offenders,
            "environment_feature_offenders": offenders,
            "environment_added_feature_names": added_environment_names,
            "explicitly_excluded": [
                "official PDM aggregate score",
                "official PDM factor scores",
                "candidate type/id",
                "candidate waypoint copies",
                "trajectory-derived ego_swept_footprint effect-tube channel",
            ],
            "target_columns_never_concatenated_into_features": True,
        },
        "feature_names": feature_names,
        "validation_mask": validation_mask,
        "train_mask": train_mask,
    }


def inverse_probe(
    dataset: dict[str, Any],
    train_mask: np.ndarray,
    validation_mask: np.ndarray,
    seed: int,
) -> dict[str, Any]:
    encoder = LabelEncoder()
    labels = encoder.fit_transform(dataset["behavior_labels"])
    x = dataset["interaction"]
    model = HistGradientBoostingClassifier(
        learning_rate=0.08,
        max_iter=100,
        max_leaf_nodes=15,
        l2_regularization=0.1,
        random_state=seed,
    )
    model.fit(x[train_mask], labels[train_mask])
    prediction = model.predict(x[validation_mask])
    truth = labels[validation_mask]
    counts = np.bincount(labels[train_mask], minlength=len(encoder.classes_))
    chance = float(np.max(counts) / np.sum(counts))
    accuracy = float(accuracy_score(truth, prediction))
    return {
        "input": "C_interaction_only effect channels",
        "excluded": [
            "candidate x/y/heading",
            "candidate speed/curvature/acceleration",
            "all trajectory-derived features",
            "current-scene features",
            "map/lane/route SDF channels",
            "candidate type and ID as input",
        ],
        "target": "coarse candidate behavior class (label only)",
        "classes": encoder.classes_.tolist(),
        "class_count": len(encoder.classes_),
        "support": int(np.sum(validation_mask)),
        "majority_chance_accuracy": chance,
        "accuracy": accuracy,
        "macro_f1": float(
            f1_score(truth, prediction, average="macro", zero_division=0)
        ),
        "above_chance_accuracy": accuracy - chance,
        "interpretation": (
            "interaction-only consequence retains action-distinguishing inverse information"
            if accuracy >= chance + 0.10
            else "当前数据可支持候选相对风险重标注，但不足以支持强 interaction inverse dynamics。"
        ),
    }


def relative_change(c_value: float | None, baseline: float | None) -> float | None:
    if c_value is None or baseline is None:
        return None
    return float(c_value - baseline)


def render_report(result: dict[str, Any]) -> str:
    probes = result["probes"]
    rows = []
    for name, probe in probes.items():
        metrics = probe["aggregate_score_and_ranking"]
        rows.append(
            f"| {name} | {metrics['pairwise_ranking_accuracy']:.4f} | {metrics['ndcg_mean']:.4f} | "
            f"{metrics['spearman_per_scene_mean']:.4f} | {metrics['top1_accuracy']:.4f} | "
            f"{metrics['top1_score_regret_mean']:.5f} |"
        )
    inverse = result["interaction_only_inverse_probe"]
    return "\n".join(
        [
            "# Oracle Planning-Utility Probe",
            "",
            f"- Scope: **{result['scene_count']} scenes / {result['candidate_count']} candidates**",
            f"- Split: complete `log_name`; train **{result['split']['train_logs']} logs**, validation **{result['split']['validation_logs']} logs**, overlap **{len(result['split']['log_overlap'])}**",
            f"- Feature leakage audit: **{'PASS' if result['leakage_audit']['pass'] else 'FAIL'}**",
            "",
            "| Probe | Pairwise accuracy | NDCG | Per-scene Spearman | Top-1 accuracy | Top-1 regret |",
            "|---|---:|---:|---:|---:|---:|",
            *rows,
            "",
            "## Probe C gains",
            "",
            *(
                f"- {key}: `{value}`"
                for key, value in result["probe_c_delta_vs_a"].items()
            ),
            "",
            "Probe C adds only candidate-relative future effect-tube summaries: dynamic occupancy/relative velocity/clearance/collision fields and map/lane/route SDF. It excludes the trajectory-derived ego-footprint channel and every official aggregate or factor score.",
            "",
            "## Interaction-only inverse probe",
            "",
            f"- Accuracy: **{inverse['accuracy']:.4f}** (majority chance **{inverse['majority_chance_accuracy']:.4f}**)",
            f"- Macro F1: **{inverse['macro_f1']:.4f}**",
            f"- Interpretation: {inverse['interpretation']}",
            "",
            "This is an oracle sufficiency audit, not a deployable predictor: Probe C consumes logged-future-derived information that is unavailable online unless a learned future model predicts it.",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser, max_scenes=500)
    parser.add_argument("--num-candidates", type=int, default=12, choices=range(8, 17))
    parser.add_argument("--max-scenes-per-log", type=int, default=8)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset = build_dataset(args)
    result = run_probes(dataset, args)
    train_mask = result.pop("train_mask")
    validation_mask = result.pop("validation_mask")
    result["interaction_only_inverse_probe"] = inverse_probe(
        dataset, train_mask, validation_mask, args.seed
    )
    result["scene_count"] = int(dataset["selected_scene_count"])
    result["candidate_count"] = int(len(dataset["scene_tokens"]))
    result["candidates_per_scene"] = args.num_candidates
    result["source"] = {
        "candidate_bank": "existing deterministic expert-anchor heuristic perturbation bank",
        "future_features": "deployed non-reactive trajectory-aligned effect-tube cache",
        "provenance": "candidate-conditioned relabeling of one logged future; not a true counterfactual",
        "effect_channels": list(EFFECT_CHANNELS),
        "effect_horizons_s": list(EFFECT_HORIZONS_S),
    }
    result["failures"] = dataset["failures"]
    a = result["probes"]["Probe_A_trajectory_only"]["aggregate_score_and_ranking"]
    b = result["probes"]["Probe_B_current_scene_plus_trajectory"][
        "aggregate_score_and_ranking"
    ]
    c = result["probes"]["Probe_C_candidate_relative_future"][
        "aggregate_score_and_ranking"
    ]
    result["probe_c_delta_vs_a"] = {
        key: relative_change(c.get(key), a.get(key))
        for key in (
            "pairwise_ranking_accuracy",
            "ndcg_mean",
            "spearman_per_scene_mean",
            "top1_accuracy",
            "top1_score_regret_mean",
        )
    }
    result["probe_c_delta_vs_b"] = {
        key: relative_change(c.get(key), b.get(key))
        for key in (
            "pairwise_ranking_accuracy",
            "ndcg_mean",
            "spearman_per_scene_mean",
            "top1_accuracy",
            "top1_score_regret_mean",
        )
    }
    write_json(args.output_dir / "oracle_probe_results.json", result)
    write_text(args.output_dir / "ORACLE_PROBE_REPORT.md", render_report(result))
    print(
        json.dumps(
            {
                "scene_count": result["scene_count"],
                "candidate_count": result["candidate_count"],
                "probe_c_delta_vs_a": result["probe_c_delta_vs_a"],
                "leakage_audit_pass": result["leakage_audit"]["pass"],
                "inverse_probe": result["interaction_only_inverse_probe"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
