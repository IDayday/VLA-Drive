from __future__ import annotations

import numpy as np

from research.cf_effect_gate_wote.src.effect_tokenizer import intervene_effects
from research.cf_effect_gate_wote.tests.test_effect_tokenizer import make_effects


def test_full_effect_swap_only_changes_effect() -> None:
    effects = make_effects()
    trajectory = np.arange(256 * 8 * 3, dtype=np.float32).reshape(256, 8, 3)
    current_bev = np.ones((64, 256), dtype=np.float32)
    labels = np.ones((256, 6), dtype=np.float32)
    swapped = intervene_effects(effects, "full_effect_swap", "scene-a")
    assert not np.array_equal(swapped["primitive_ego_effect"], effects["primitive_ego_effect"])
    np.testing.assert_array_equal(trajectory, trajectory.copy())
    np.testing.assert_array_equal(current_bev, current_bev.copy())
    np.testing.assert_array_equal(labels, labels.copy())
    np.testing.assert_array_equal(
        swapped["shared_logged_future"], effects["shared_logged_future"]
    )


def test_actor_only_and_static_only_swaps_are_isolated() -> None:
    effects = make_effects()
    actor = intervene_effects(effects, "actor_only_swap", "scene-a")
    np.testing.assert_array_equal(actor["primitive_ego_effect"], effects["primitive_ego_effect"])
    np.testing.assert_array_equal(actor["primitive_map_effect"], effects["primitive_map_effect"])
    assert not np.array_equal(actor["primitive_actor_effect"], effects["primitive_actor_effect"])
    static = intervene_effects(effects, "static_only_swap", "scene-a")
    assert not np.array_equal(static["primitive_ego_effect"], effects["primitive_ego_effect"])
    np.testing.assert_array_equal(static["primitive_actor_effect"], effects["primitive_actor_effect"])
    np.testing.assert_array_equal(static["primitive_actor_mask"], effects["primitive_actor_mask"])


def test_scene_mean_effect_is_identical_for_all_candidates() -> None:
    mean = intervene_effects(make_effects(), "scene_mean_effect", "scene-a")
    for key in (
        "primitive_ego_effect",
        "primitive_map_effect",
        "primitive_actor_effect",
        "primitive_actor_mask",
        "primitive_interaction_mask",
    ):
        np.testing.assert_array_equal(mean[key], np.broadcast_to(mean[key][0], mean[key].shape))

