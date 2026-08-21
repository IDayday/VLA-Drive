from __future__ import annotations

import numpy as np

from research.action_effect.reversal import build_reversal_cases, reversal_accuracy


def _candidate(scene: str, candidate: str, kind: str) -> dict:
    return {
        "scene_id": scene,
        "candidate_id": candidate,
        "perturbation_type": kind,
        "perturbation_parameters": {},
    }


def test_cross_scene_reversal_requires_scene_conditioning() -> None:
    rows = [
        _candidate("s1", "s1a", "anchor"),
        _candidate("s1", "s1b", "speed_scale"),
        _candidate("s2", "s2a", "anchor"),
        _candidate("s2", "s2b", "speed_scale"),
    ]
    pairs = [
        {
            "scene_id": "s1",
            "candidate_i": "s1a",
            "candidate_j": "s1b",
            "log_replay_order": 1,
        },
        {
            "scene_id": "s2",
            "candidate_i": "s2a",
            "candidate_j": "s2b",
            "log_replay_order": -1,
        },
    ]
    cases = build_reversal_cases(
        rows, pairs, selected_scene_ids=("s1", "s2"), maximum_cases=10, seed=3
    )
    assert len(cases) == 1
    accuracy, correct = reversal_accuracy(
        cases,
        candidate_ids=np.asarray(["s1a", "s1b", "s2a", "s2b"]),
        predicted_utility=np.asarray([0.9, 0.1, 0.1, 0.9]),
    )
    assert accuracy == 1.0
    assert correct.tolist() == [True]
