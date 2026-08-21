from __future__ import annotations

import json

import numpy as np

from research.action_effect.probe_data import ProbeArrays
from research.action_effect.structured_audit import structured_channel_audit


def test_structured_audit_separates_invariant_and_action_channels(tmp_path) -> None:
    scene_ids = np.asarray(["a", "a", "b", "b"])
    candidate_ids = np.asarray(["a0", "a1", "b0", "b1"])
    arrays = ProbeArrays(
        scene_ids=scene_ids,
        candidate_ids=candidate_ids,
        scene_feature_indices=np.asarray([0, 0, 1, 1]),
        candidate_indices=np.arange(4),
        trajectories=np.zeros((4, 8, 4), dtype=np.float32),
        targets=np.zeros((4, 2), dtype=np.float32),
        raw_hard_targets=np.zeros((4, 1), dtype=np.float32),
        raw_soft_targets=np.zeros((4, 1), dtype=np.float32),
        accepted=np.ones(4, dtype=bool),
        anchor=np.asarray([True, False, True, False]),
    )
    target = np.zeros((4, 1, 2, 2, 2), dtype=np.float32)
    target[2:, :, 0] = 1.0  # scene-varying but action-invariant
    target[1, :, 1] = 1.0
    target[3, :, 1] = 2.0  # action-dependent in both scenes
    pairs = [
        {
            "scene_id": "a",
            "candidate_i": "a0",
            "candidate_j": "a1",
            "pair_type": "effect_divergent",
        },
        {
            "scene_id": "b",
            "candidate_i": "b0",
            "candidate_j": "b1",
            "pair_type": "effect_divergent",
        },
    ]
    pair_path = tmp_path / "pairs.jsonl"
    pair_path.write_text("".join(json.dumps(row) + "\n" for row in pairs), encoding="utf-8")
    rows, details = structured_channel_audit(
        arrays=arrays,
        target=target,
        valid=np.ones(4, dtype=bool),
        channels=("scene", "action"),
        pair_path=pair_path,
        selected_scene_ids=("a", "b"),
        minimum_action_variance_ratio=0.01,
        minimum_target_action_gap=0.01,
    )
    assert rows[0]["classification"] == "exogenous/action-invariant"
    assert rows[1]["classification"] == "action_effect/action-dependent"
    np.testing.assert_array_equal(details["action_dependent_channel_indices"], [1])
