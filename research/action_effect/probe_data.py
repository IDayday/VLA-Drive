"""Load isolated factual/candidate data for action-collapse probes."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


HARD_TARGET_FIELDS = (
    "no_at_fault_collision",
    "drivable_area_compliance",
    "driving_direction_compliance",
    "traffic_light_compliance",
    "static_object_collision",
    "dynamic_collision",
)

SOFT_TARGET_FIELDS = (
    "ttc_infraction_time_s",
    "centerline_progress_m",
    "lane_keeping",
    "history_comfort",
    "minimum_dynamic_clearance_m",
    "route_deviation_max_m",
    "max_acceleration_mps2",
    "max_deceleration_mps2",
    "max_abs_jerk_mps3",
    "max_abs_curvature_inv_m",
)

EXACT_SOFT_FIELDS = frozenset(
    {
        "centerline_progress_m",
        "lane_keeping",
        "history_comfort",
        "route_deviation_max_m",
        "max_acceleration_mps2",
        "max_deceleration_mps2",
        "max_abs_jerk_mps3",
        "max_abs_curvature_inv_m",
    }
)


@dataclass(frozen=True)
class ProbeScale:
    """Training-anchor-only robust normalization for a soft target."""

    field: str
    median: float
    scale: float
    quantile_05: float
    quantile_95: float
    active: bool


@dataclass(frozen=True)
class ProbeArrays:
    """Dense arrays and identifiers used by training and candidate evaluation."""

    scene_ids: np.ndarray
    candidate_ids: np.ndarray
    scene_feature_indices: np.ndarray
    candidate_indices: np.ndarray
    trajectories: np.ndarray
    targets: np.ndarray
    raw_hard_targets: np.ndarray
    raw_soft_targets: np.ndarray
    accepted: np.ndarray
    anchor: np.ndarray


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    """Yield JSON objects without silently accepting blank/malformed records."""

    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"expected object at {path}:{line_number}")
            yield value


def deterministic_scene_split(
    scene_ids: Sequence[str], *, fraction: float, seed: int
) -> tuple[list[str], list[str]]:
    """Create a deterministic scene-level fit/held-out split."""

    if not 0.0 < fraction < 1.0:
        raise ValueError("split fraction must lie strictly between zero and one")
    unique = sorted(set(scene_ids))
    ranked = sorted(
        unique,
        key=lambda token: hashlib.sha256(f"{seed}:{token}".encode("utf-8")).hexdigest(),
    )
    fit_count = min(max(int(round(len(ranked) * fraction)), 1), len(ranked) - 1)
    return ranked[:fit_count], ranked[fit_count:]


def _soft_value(row: Mapping[str, Any], field: str, assumption: str) -> float:
    namespace = row["exact"] if field in EXACT_SOFT_FIELDS else row[assumption]
    value = namespace.get(field)
    return float(value) if value is not None else float("nan")


def raw_hard_target(row: Mapping[str, Any], assumption: str) -> np.ndarray:
    """Return raw hard labels, retaining collision fields as one means collision."""

    return np.asarray(
        [
            row[assumption]["no_at_fault_collision"],
            row["exact"]["drivable_area_compliance"],
            row["exact"]["driving_direction_compliance"],
            row[assumption]["traffic_light_compliance"],
            float(bool(row["exact"]["static_object_collision"])),
            float(bool(row[assumption]["dynamic_collision"])),
        ],
        dtype=np.float32,
    )


def fit_probe_scales(
    factual_fit_rows: Sequence[Mapping[str, Any]],
    *,
    assumption: str,
    minimum_scale: float = 1.0e-3,
) -> list[ProbeScale]:
    """Fit robust statistics using only factual anchors in the fit scenes."""

    if not factual_fit_rows:
        raise ValueError("factual fit rows are empty")
    result: list[ProbeScale] = []
    for field in SOFT_TARGET_FIELDS:
        values = np.asarray(
            [_soft_value(row, field, assumption) for row in factual_fit_rows], dtype=np.float64
        )
        finite = values[np.isfinite(values)]
        if not len(finite):
            result.append(ProbeScale(field, 0.0, 1.0, 0.0, 0.0, False))
            continue
        q05, q25, median, q75, q95 = np.quantile(finite, [0.05, 0.25, 0.5, 0.75, 0.95])
        span_scale = float((q95 - q05) / 3.29)
        scale = max(float(q75 - q25), span_scale, minimum_scale)
        result.append(
            ProbeScale(
                field=field,
                median=float(median),
                scale=scale,
                quantile_05=float(q05),
                quantile_95=float(q95),
                active=bool(float(q95 - q05) > minimum_scale),
            )
        )
    return result


def consequence_target(
    row: Mapping[str, Any],
    scales: Sequence[ProbeScale],
    *,
    assumption: str,
    clip: float = 5.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build hard labels and train-normalized soft values for one candidate."""

    hard = raw_hard_target(row, assumption)
    raw_soft = np.asarray([_soft_value(row, scale.field, assumption) for scale in scales])
    normalized = np.asarray(
        [
            np.clip((value - scale.median) / scale.scale, -clip, clip)
            if math.isfinite(float(value))
            else 0.0
            for value, scale in zip(raw_soft, scales)
        ],
        dtype=np.float32,
    )
    return np.concatenate((hard, normalized)).astype(np.float32), hard, raw_soft.astype(np.float32)


def trajectory_normalization(
    trajectories: np.ndarray, *, minimum_scale: float = 0.25
) -> dict[str, float]:
    """Fit x/y robust scales on factual fit trajectories only."""

    xy = np.asarray(trajectories, dtype=np.float64)[..., :2]
    values: dict[str, float] = {}
    for index, name in enumerate(("x", "y")):
        flattened = xy[..., index].reshape(-1)
        q25, median, q75 = np.quantile(flattened, [0.25, 0.5, 0.75])
        values[f"{name}_median"] = float(median)
        values[f"{name}_scale"] = max(float(q75 - q25), minimum_scale)
    return values


def encode_trajectories(
    trajectories: np.ndarray, normalization: Mapping[str, float]
) -> np.ndarray:
    """Encode physical ``x,y,heading`` poses without discontinuous raw yaw."""

    trajectories = np.asarray(trajectories, dtype=np.float32)
    x = (trajectories[..., 0] - normalization["x_median"]) / normalization["x_scale"]
    y = (trajectories[..., 1] - normalization["y_median"]) / normalization["y_scale"]
    heading = trajectories[..., 2]
    return np.stack((x, y, np.sin(heading), np.cos(heading)), axis=-1).astype(np.float32)


def load_probe_arrays(
    *,
    candidate_cache: Path,
    consequence_cache: Path,
    scene_feature_cache: Path,
    fit_scene_ids: Sequence[str],
    assumption: str,
) -> tuple[ProbeArrays, np.ndarray, np.ndarray, list[ProbeScale], dict[str, float], dict[str, int]]:
    """Load candidates with a strict target/features separation.

    Returns arrays plus frozen scene/action-hidden features. Consequence labels
    are never inserted into either feature tensor.
    """

    rows = list(iter_jsonl(consequence_cache / "consequences.jsonl"))
    by_candidate = {str(row["candidate_id"]): row for row in rows}
    with (candidate_cache / "metadata.jsonl").open("r", encoding="utf-8") as stream:
        metadata = [json.loads(line) for line in stream if line.strip()]
    with np.load(candidate_cache / "candidates.npz") as payload:
        all_trajectories = np.asarray(payload["trajectories"], dtype=np.float32)
    with (scene_feature_cache / "scene_index.json").open("r", encoding="utf-8") as stream:
        feature_index = {str(key): int(value) for key, value in json.load(stream).items()}
    with np.load(scene_feature_cache / "features.npz") as payload:
        scene_features = np.asarray(payload["scene_tokens"], dtype=np.float32)
        action_hidden = np.asarray(payload["action_hidden"], dtype=np.float32)

    fit_scene_set = set(str(scene_id) for scene_id in fit_scene_ids)
    factual_fit = [
        row
        for row in rows
        if str(row["scene_id"]) in fit_scene_set
        and row.get("candidate_accepted")
        and row.get("perturbation_type") == "anchor"
        and row[assumption].get("available")
    ]
    scales = fit_probe_scales(factual_fit, assumption=assumption)
    fit_trajectory_indices = [int(row["candidate_index"]) for row in factual_fit]
    trajectory_stats = trajectory_normalization(all_trajectories[fit_trajectory_indices])

    scene_ids: list[str] = []
    candidate_ids: list[str] = []
    scene_indices: list[int] = []
    candidate_indices: list[int] = []
    targets: list[np.ndarray] = []
    raw_hard: list[np.ndarray] = []
    raw_soft: list[np.ndarray] = []
    accepted: list[bool] = []
    anchors: list[bool] = []
    for item in metadata:
        candidate_id = str(item["candidate_id"])
        row = by_candidate[candidate_id]
        scene_id = str(row["scene_id"])
        if scene_id not in feature_index:
            raise KeyError(f"scene feature is missing: {scene_id}")
        is_accepted = bool(row.get("candidate_accepted") and row[assumption].get("available"))
        if is_accepted:
            target, hard, soft = consequence_target(row, scales, assumption=assumption)
        else:
            target = np.zeros(len(HARD_TARGET_FIELDS) + len(SOFT_TARGET_FIELDS), dtype=np.float32)
            hard = np.full(len(HARD_TARGET_FIELDS), np.nan, dtype=np.float32)
            soft = np.full(len(SOFT_TARGET_FIELDS), np.nan, dtype=np.float32)
        scene_ids.append(scene_id)
        candidate_ids.append(candidate_id)
        scene_indices.append(feature_index[scene_id])
        candidate_indices.append(int(row["candidate_index"]))
        targets.append(target)
        raw_hard.append(hard)
        raw_soft.append(soft)
        accepted.append(is_accepted)
        anchors.append(row.get("perturbation_type") == "anchor")
    trajectory_indices_array = np.asarray(candidate_indices, dtype=np.int64)
    encoded = encode_trajectories(all_trajectories[trajectory_indices_array], trajectory_stats)
    arrays = ProbeArrays(
        scene_ids=np.asarray(scene_ids, dtype=str),
        candidate_ids=np.asarray(candidate_ids, dtype=str),
        scene_feature_indices=np.asarray(scene_indices, dtype=np.int64),
        candidate_indices=trajectory_indices_array,
        trajectories=encoded,
        targets=np.stack(targets),
        raw_hard_targets=np.stack(raw_hard),
        raw_soft_targets=np.stack(raw_soft),
        accepted=np.asarray(accepted, dtype=bool),
        anchor=np.asarray(anchors, dtype=bool),
    )
    return arrays, scene_features, action_hidden, scales, trajectory_stats, feature_index


def scales_to_json(scales: Sequence[ProbeScale]) -> list[dict[str, Any]]:
    """Serialize scale provenance into the experiment directory."""

    return [asdict(scale) for scale in scales]


def load_structured_targets(
    cache_dir: Path, arrays: ProbeArrays
) -> tuple[np.ndarray, np.ndarray]:
    """Load per-scene future tubes in the exact global candidate-row order."""

    with (cache_dir / "scene_index.json").open("r", encoding="utf-8") as stream:
        scene_index = json.load(stream)
    target: np.ndarray | None = None
    valid = np.zeros(len(arrays.scene_ids), dtype=bool)
    assigned = np.zeros(len(arrays.scene_ids), dtype=bool)
    indices_by_scene: dict[str, list[int]] = {}
    for index, scene_id in enumerate(arrays.scene_ids):
        indices_by_scene.setdefault(str(scene_id), []).append(index)
    for scene_id, entry in scene_index.items():
        indices = np.asarray(indices_by_scene.get(str(scene_id), ()), dtype=np.int64)
        with np.load(cache_dir / entry["file"]) as payload:
            scene_target = np.asarray(payload["target"], dtype=np.float16)
            scene_valid = np.asarray(payload["valid"], dtype=bool)
        if len(indices) != len(scene_target):
            raise RuntimeError(f"structured target candidate mismatch for {scene_id}")
        if target is None:
            target = np.zeros((len(arrays.scene_ids), *scene_target.shape[1:]), dtype=np.float16)
        target[indices] = scene_target
        valid[indices] = scene_valid
        assigned[indices] = True
    if target is None or not assigned.all():
        missing = sorted(set(arrays.scene_ids[~assigned].tolist()))
        raise RuntimeError(f"structured target scenes are incomplete: {missing[:5]}")
    return target, valid
