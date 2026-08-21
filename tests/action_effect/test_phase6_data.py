from __future__ import annotations

import numpy as np

from research.action_effect.phase6_data import (
    deterministic_three_way_split,
    group_pairs_by_scene,
    sample_balanced_pairs,
)


def test_three_way_split_is_exact_disjoint_and_deterministic() -> None:
    scenes = [f"scene_{index}" for index in range(20)]
    first = deterministic_three_way_split(
        scenes, train_count=10, validation_count=4, test_count=3, seed=7
    )
    second = deterministic_three_way_split(
        list(reversed(scenes)), train_count=10, validation_count=4, test_count=3, seed=7
    )
    assert first == second
    assert [len(first[name]) for name in ("train", "validation", "test", "unused")] == [10, 4, 3, 3]
    assigned = first["train"] + first["validation"] + first["test"]
    assert len(assigned) == len(set(assigned))


def test_balanced_pair_sampling_does_not_compensate_missing_category() -> None:
    rows = [
        {
            "scene_id": "a",
            "candidate_i": "a0",
            "candidate_j": "a1",
            "pair_type": "effect_equivalent",
            "safety_boundary": False,
            "geometric_distance": 1.0,
            "pair_confidence": "high",
            "consequence_distance": 0.1,
        },
        {
            "scene_id": "b",
            "candidate_i": "b0",
            "candidate_j": "b1",
            "pair_type": "effect_divergent",
            "safety_boundary": True,
            "geometric_distance": 0.5,
            "pair_confidence": "medium",
            "consequence_distance": 2.0,
        },
    ]
    lookup = {name: index for index, name in enumerate(("a0", "a1", "b0", "b1"))}
    groups = group_pairs_by_scene(rows, lookup, allowed_candidate_mask=np.ones(4, dtype=bool))
    sampled = sample_balanced_pairs(("a", "b"), groups, rng=np.random.default_rng(3))
    assert sampled["availability"].shape == (2, 3)
    assert sampled["availability"][0].tolist() == [True, False, False]
    assert sampled["availability"][1].tolist() == [False, True, True]
    assert len(sampled["left"]) == 3

    global_sample = sample_balanced_pairs(
        ("a", "b"),
        groups,
        rng=np.random.default_rng(3),
        categories=("geometrically_distinct",),
    )
    assert global_sample["availability"].tolist() == [[True], [True]]
    assert len(global_sample["left"]) == 2


def test_confidence_sampling_can_retain_replay_pair_with_weak_weight() -> None:
    rows = [
        {
            "scene_id": "a",
            "candidate_i": "a0",
            "candidate_j": "a1",
            "pair_type": "ambiguous",
            "replay_pair_type": "effect_divergent",
            "safety_boundary": True,
            "geometric_distance": 0.5,
            "pair_confidence": "low",
            "consequence_distance": 2.0,
        }
    ]
    groups = group_pairs_by_scene(
        rows, {"a0": 0, "a1": 1}, allowed_candidate_mask=np.ones(2, dtype=bool)
    )
    sampled = sample_balanced_pairs(
        ("a",),
        groups,
        rng=np.random.default_rng(3),
        categories=("confidence_effect_divergent",),
        confidence_weights={"high": 1.0, "medium": 0.5, "low": 0.1, "unassessed": 1.0},
    )
    assert np.isclose(sampled["confidence_weight"][0], 0.1)
