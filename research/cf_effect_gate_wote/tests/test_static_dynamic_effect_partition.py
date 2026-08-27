from __future__ import annotations

import numpy as np

from research.cf_effect_gate_wote.src.oracle_effect_verdict import (
    deterministic_effect_permutation,
    shuffle_effect_only,
)
from research.cf_effect_gate_wote.src.replay_effect_builder import (
    ReplayGroundedEffectBuilder,
)
from research.cf_effect_gate_wote.tests.test_effect_builder import (
    make_candidates,
    make_context,
)


def test_static_dynamic_are_disjoint_and_concat_to_full_primitive() -> None:
    result = ReplayGroundedEffectBuilder().build(make_candidates(), make_context())
    groups = result.flattened_primitive_groups()
    static = groups["static_effect"]
    dynamic = groups["dynamic_effect"]
    np.testing.assert_array_equal(
        groups["full_primitive_effect"], np.concatenate([static, dynamic], axis=-1)
    )
    assert static.shape[1] + dynamic.shape[1] == groups["full_primitive_effect"].shape[1]


def test_effect_shuffle_changes_only_candidate_effect() -> None:
    trajectory = make_candidates()
    labels = np.arange(len(trajectory), dtype=np.float32)
    effect = ReplayGroundedEffectBuilder().build(
        trajectory, make_context()
    ).as_primitive_dict()
    permutation = deterministic_effect_permutation("scene", candidates=len(trajectory))
    shuffled = shuffle_effect_only(effect, permutation)
    np.testing.assert_array_equal(trajectory, make_candidates())
    np.testing.assert_array_equal(labels, np.arange(len(trajectory), dtype=np.float32))
    np.testing.assert_array_equal(shuffled["ego_effect"], effect["ego_effect"][permutation])
    assert not np.array_equal(shuffled["ego_effect"], effect["ego_effect"])
