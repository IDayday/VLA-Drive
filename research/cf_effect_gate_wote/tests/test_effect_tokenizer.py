from __future__ import annotations

import numpy as np

from research.cf_effect_gate_wote.src.effect_tokenizer import EffectTokenPacker


def make_frozen() -> dict[str, np.ndarray]:
    return {
        "trajectory": np.zeros((256, 8, 3), dtype=np.float32),
        "environment_only_future": np.ones((256, 1, 64), dtype=np.float32),
    }


def make_effects() -> dict[str, np.ndarray]:
    candidate = np.arange(256, dtype=np.float32)[:, None, None]
    ego = np.broadcast_to(candidate, (256, 8, 16)).copy()
    map_effect = np.broadcast_to(candidate, (256, 8, 7)).copy()
    actor = np.broadcast_to(candidate[..., None], (256, 8, 16, 10)).copy()
    validity = np.ones((256, 8, 16), dtype=np.float32)
    interaction = np.zeros_like(validity)
    interaction[:, :, 0] = 1.0
    return {
        "primitive_ego_effect": ego,
        "primitive_map_effect": map_effect,
        "primitive_actor_effect": actor,
        "primitive_actor_mask": validity,
        "primitive_interaction_mask": interaction,
        "shared_logged_future": np.ones((8, 16, 8), dtype=np.float32),
        "shared_actor_mask": np.ones((8, 16), dtype=np.float32),
        "map_engineered_effect": np.zeros((256, 8, 1), dtype=np.float32),
        "actor_engineered_effect": np.zeros((256, 8, 16, 3), dtype=np.float32),
    }


def test_all_structured_variants_have_fixed_shape() -> None:
    packer = EffectTokenPacker()
    effects = make_effects()
    frozen = make_frozen()
    for model in (
        "trajectory_only",
        "direct_current",
        "ego_kinematic_effect",
        "static_primitive_effect",
        "shared_logged_future",
        "dynamic_replay_effect",
        "full_primitive_action_effect",
        "full_primitive_no_interaction_mask",
        "interaction_mask_only",
        "full_engineered_action_effect",
        "wote_environment_only",
    ):
        packed = packer.pack(model, effects, frozen)
        assert packed.auxiliary_tokens.shape == (256, 32, 64)
        assert np.isfinite(packed.auxiliary_tokens).all()


def test_missing_groups_are_zero_and_shared_future_is_candidate_identical() -> None:
    packer = EffectTokenPacker()
    direct = packer.pack("direct_current", make_effects(), make_frozen()).auxiliary_tokens
    assert np.count_nonzero(direct) == 0
    ego = packer.pack("ego_kinematic_effect", make_effects(), make_frozen()).auxiliary_tokens
    assert np.count_nonzero(ego[:, 8:]) == 0
    shared = packer.pack("shared_logged_future", make_effects(), make_frozen()).auxiliary_tokens
    np.testing.assert_array_equal(shared[0], shared[-1])


def test_interaction_mask_only_excludes_actor_and_map_geometry() -> None:
    packed = EffectTokenPacker().pack(
        "interaction_mask_only", make_effects(), make_frozen()
    ).auxiliary_tokens
    assert np.count_nonzero(packed[:, :24]) == 0
    assert np.count_nonzero(packed[:, 24:]) > 0

