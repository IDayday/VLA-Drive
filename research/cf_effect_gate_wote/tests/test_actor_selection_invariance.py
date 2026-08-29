from __future__ import annotations

import numpy as np

from research.cf_effect_gate_wote.src.replay_effect_builder import ReplayGroundedEffectBuilder
from research.cf_effect_gate_wote.tests.test_effect_builder import make_candidates, make_context


def test_actor_selection_is_invariant_to_candidate_permutation() -> None:
    builder = ReplayGroundedEffectBuilder()
    candidates = make_candidates()
    first = builder.build(candidates, make_context())
    second = builder.build(candidates[np.array([2, 0, 1])], make_context())
    np.testing.assert_array_equal(first.selected_actor_indices, second.selected_actor_indices)
    assert first.selected_actor_tokens == second.selected_actor_tokens

