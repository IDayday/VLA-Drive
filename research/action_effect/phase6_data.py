"""Scene-disjoint Phase-6 splits and balanced per-scene pair sampling."""

from __future__ import annotations

import hashlib
from typing import Any, Mapping, Sequence

import numpy as np


def deterministic_three_way_split(
    scene_ids: Sequence[str],
    *,
    train_count: int,
    validation_count: int,
    test_count: int,
    seed: int,
) -> dict[str, list[str]]:
    """Create exact-count, scene-disjoint partitions using a stable hash rank."""

    counts = (train_count, validation_count, test_count)
    if any(value < 1 for value in counts):
        raise ValueError("all Phase-6 split counts must be positive")
    unique = sorted(set(str(value) for value in scene_ids))
    required = sum(counts)
    if len(unique) < required:
        raise ValueError(f"Phase-6 split requires {required} scenes but only {len(unique)} are eligible")
    ranked = sorted(
        unique,
        key=lambda token: hashlib.sha256(f"{seed}:{token}".encode("utf-8")).hexdigest(),
    )
    train_stop = train_count
    validation_stop = train_stop + validation_count
    test_stop = validation_stop + test_count
    result = {
        "train": ranked[:train_stop],
        "validation": ranked[train_stop:validation_stop],
        "test": ranked[validation_stop:test_stop],
        "unused": ranked[test_stop:],
    }
    assigned = result["train"] + result["validation"] + result["test"]
    if len(assigned) != len(set(assigned)):
        raise AssertionError("Phase-6 scene split is not disjoint")
    return result


def group_indices_by_scene(scene_ids: np.ndarray, mask: np.ndarray) -> dict[str, np.ndarray]:
    """Return sorted candidate indices for every selected scene."""

    scene_ids = np.asarray(scene_ids, dtype=str)
    mask = np.asarray(mask, dtype=bool)
    grouped: dict[str, list[int]] = {}
    for index in np.flatnonzero(mask):
        grouped.setdefault(str(scene_ids[index]), []).append(int(index))
    return {
        scene: np.asarray(grouped[scene], dtype=np.int64)
        for scene in sorted(grouped)
    }


def group_pairs_by_scene(
    pair_rows: Sequence[Mapping[str, Any]],
    candidate_lookup: Mapping[str, int],
    *,
    allowed_candidate_mask: np.ndarray,
) -> dict[str, dict[str, list[tuple[int, int, Mapping[str, Any]]]]]:
    """Group valid pairs into balanced semantic categories per scene."""

    groups: dict[str, dict[str, list[tuple[int, int, Mapping[str, Any]]]]] = {}
    for row in pair_rows:
        left = int(candidate_lookup[str(row["candidate_i"])])
        right = int(candidate_lookup[str(row["candidate_j"])])
        if not allowed_candidate_mask[left] or not allowed_candidate_mask[right]:
            continue
        scene = str(row["scene_id"])
        category = str(row["pair_type"])
        scene_groups = groups.setdefault(
            scene,
            {
                "effect_equivalent": [],
                "effect_divergent": [],
                "safety_boundary": [],
                "geometrically_distinct": [],
                "confidence_effect_equivalent": [],
                "confidence_effect_divergent": [],
                "confidence_safety_boundary": [],
            },
        )
        item = (left, right, row)
        if category in {"effect_equivalent", "effect_divergent"}:
            scene_groups[category].append(item)
        if bool(row.get("safety_boundary")):
            scene_groups["safety_boundary"].append(item)
            scene_groups["confidence_safety_boundary"].append(item)
        replay_category = str(row.get("replay_pair_type", category))
        if replay_category in {"effect_equivalent", "effect_divergent"}:
            scene_groups[f"confidence_{replay_category}"].append(item)
        if float(row.get("geometric_distance", 0.0)) > 0.0:
            scene_groups["geometrically_distinct"].append(item)
    return groups


def sample_balanced_pairs(
    scene_ids: Sequence[str],
    groups: Mapping[str, Mapping[str, Sequence[tuple[int, int, Mapping[str, Any]]]]],
    *,
    rng: np.random.Generator,
    categories: Sequence[str] = (
        "effect_equivalent",
        "effect_divergent",
        "safety_boundary",
    ),
    confidence_weights: Mapping[str, float] | None = None,
) -> dict[str, np.ndarray]:
    """Sample at most one pair/category/scene and expose missing-category masks."""

    left: list[int] = []
    right: list[int] = []
    category_index: list[int] = []
    scene_index: list[int] = []
    confidence_weight: list[float] = []
    consequence_distance: list[float] = []
    availability = np.zeros((len(scene_ids), len(categories)), dtype=bool)
    confidence = dict(
        confidence_weights
        or {"high": 1.0, "medium": 0.5, "low": 0.0, "unassessed": 1.0}
    )
    for scene_position, scene in enumerate(scene_ids):
        scene_groups = groups.get(str(scene), {})
        for category_position, category in enumerate(categories):
            options = scene_groups.get(category, ())
            if not options:
                continue
            selection = options[int(rng.integers(0, len(options)))]
            left.append(int(selection[0]))
            right.append(int(selection[1]))
            category_index.append(category_position)
            scene_index.append(scene_position)
            row = selection[2]
            confidence_weight.append(confidence.get(str(row.get("pair_confidence", "unassessed")), 1.0))
            consequence_distance.append(float(row.get("consequence_distance", 1.0)))
            availability[scene_position, category_position] = True
    return {
        "left": np.asarray(left, dtype=np.int64),
        "right": np.asarray(right, dtype=np.int64),
        "category": np.asarray(category_index, dtype=np.int64),
        "scene": np.asarray(scene_index, dtype=np.int64),
        "confidence_weight": np.asarray(confidence_weight, dtype=np.float32),
        "consequence_distance": np.asarray(consequence_distance, dtype=np.float32),
        "availability": availability,
    }
