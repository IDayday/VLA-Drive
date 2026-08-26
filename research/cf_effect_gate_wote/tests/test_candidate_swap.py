from __future__ import annotations

import numpy as np

from research.cf_effect_gate_wote.src.replay_effect_builder import (
    ReplayGroundedEffectBuilder,
)
from research.cf_effect_gate_wote.tests.test_effect_builder import (
    make_candidates,
    make_context,
)


def test_candidate_swap_swaps_every_candidate_effect_axis() -> None:
    builder = ReplayGroundedEffectBuilder()
    candidates = make_candidates()
    permutation = np.array([2, 0, 1])
    original = builder.build(candidates, make_context())
    swapped = builder.build(candidates[permutation], make_context())
    for key, value in original.as_tensor_dict().items():
        np.testing.assert_array_equal(swapped.as_tensor_dict()[key], value[permutation])
    np.testing.assert_array_equal(
        swapped.selected_actor_indices, original.selected_actor_indices
    )
    assert swapped.selected_actor_tokens == original.selected_actor_tokens
