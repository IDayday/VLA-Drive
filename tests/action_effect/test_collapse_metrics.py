from __future__ import annotations

import json

import numpy as np

from research.action_effect.metrics import (
    compute_action_collapse_metrics,
    compute_structured_collapse_metrics,
)
from research.action_effect.probe_data import ProbeArrays, ProbeScale


def test_collapse_metrics_detect_action_independent_predictions(tmp_path) -> None:
    scene_ids = np.asarray(["s0", "s0", "s1", "s1"])
    arrays = ProbeArrays(
        scene_ids=scene_ids,
        candidate_ids=np.asarray(["a", "b", "c", "d"]),
        scene_feature_indices=np.asarray([0, 0, 1, 1]),
        candidate_indices=np.arange(4),
        trajectories=np.zeros((4, 8, 4), dtype=np.float32),
        targets=np.asarray(
            [
                [1, 1, 1, 1, 0, 0, 0],
                [0, 1, 1, 1, 0, 1, -1],
                [1, 1, 1, 1, 0, 0, 0],
                [0, 1, 1, 1, 0, 1, -1],
            ],
            dtype=np.float32,
        ),
        raw_hard_targets=np.asarray(
            [[1, 1, 1, 1, 0, 0], [0, 1, 1, 1, 0, 1]] * 2, dtype=np.float32
        ),
        raw_soft_targets=np.asarray([[5], [1], [5], [1]], dtype=np.float32),
        accepted=np.ones(4, dtype=bool),
        anchor=np.asarray([True, False, True, False]),
    )
    pair_path = tmp_path / "pairs.jsonl"
    rows = [
        {
            "scene_id": scene,
            "candidate_i": left,
            "candidate_j": right,
            "pair_type": "effect_divergent",
            "consequence_distance": 1.0,
        }
        for scene, left, right in (("s0", "a", "b"), ("s1", "c", "d"))
    ]
    pair_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    prediction = np.zeros((4, 7), dtype=np.float32)
    metrics, _ = compute_action_collapse_metrics(
        arrays=arrays,
        raw_prediction=prediction,
        heldout_scene_ids=["s0", "s1"],
        pair_path=pair_path,
        scales=[ProbeScale("ttc_infraction_time_s", 5.0, 1.0, 1.0, 5.0, True)],
        low_ttc_seconds=1.5,
        safe_threshold=0.5,
        bootstrap_samples=20,
        confidence=0.95,
        seed=3,
    )
    assert metrics["candidate_sensitivity"].point == 0.0
    assert metrics["action_gap"].point == 0.0
    assert metrics["effect_alignment"].point == 0.0


def test_structured_collapse_metrics_accept_expected_contract(tmp_path) -> None:
    scene_ids = np.asarray(["s0", "s0", "s1", "s1"])
    arrays = ProbeArrays(
        scene_ids=scene_ids,
        candidate_ids=np.asarray(["a", "b", "c", "d"]),
        scene_feature_indices=np.asarray([0, 0, 1, 1]),
        candidate_indices=np.arange(4),
        trajectories=np.zeros((4, 8, 4), dtype=np.float32),
        targets=np.zeros((4, 7), dtype=np.float32),
        raw_hard_targets=np.asarray(
            [[1, 1, 1, 1, 0, 0], [0, 1, 1, 1, 0, 1]] * 2, dtype=np.float32
        ),
        raw_soft_targets=np.zeros((4, 1), dtype=np.float32),
        accepted=np.ones(4, dtype=bool),
        anchor=np.asarray([True, False, True, False]),
    )
    pairs = [
        {"scene_id": "s0", "candidate_i": "a", "candidate_j": "b", "pair_type": "effect_divergent", "consequence_distance": 1.0},
        {"scene_id": "s1", "candidate_i": "c", "candidate_j": "d", "pair_type": "effect_equivalent", "consequence_distance": 0.0},
    ]
    pair_path = tmp_path / "pairs.jsonl"
    pair_path.write_text("".join(json.dumps(row) + "\n" for row in pairs))
    target = np.zeros((4, 3, 7, 2, 2), dtype=np.float16)
    target[:, :, 0] = 1
    prediction = np.zeros_like(target)
    metrics, details = compute_structured_collapse_metrics(
        arrays=arrays,
        structured_target=target,
        structured_valid=np.ones(4, dtype=bool),
        raw_prediction=prediction,
        heldout_scene_ids=["s0", "s1"],
        pair_path=pair_path,
        safe_threshold=0.5,
        minimum_clearance_normalized=0.1,
        ego_grid_mask=np.ones((2, 2), dtype=bool),
        bootstrap_samples=10,
        confidence=0.95,
        seed=1,
    )
    assert metrics["candidate_sensitivity"].point == 0.0
    assert details["candidate_error"].shape == (4,)
