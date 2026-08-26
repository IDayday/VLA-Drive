from __future__ import annotations

import inspect

import numpy as np
import pytest

from research.cf_effect_gate_wote.src.feature_store import stable_array_hash
from research.cf_effect_gate_wote.src.replay_effect_builder import (
    ACTOR_EFFECT_NAMES,
    EGO_EFFECT_NAMES,
    MAP_EFFECT_NAMES,
    EffectConstructionError,
    LoggedActorFutures,
    ReplayGroundedEffectBuilder,
    ReplaySceneContext,
)


def make_context() -> ReplaySceneContext:
    horizon = 8
    actor_positions = np.zeros((horizon, 2, 2), dtype=np.float32)
    actor_positions[:, 0, 0] = np.linspace(5.0, 12.0, horizon)
    actor_positions[:, 0, 1] = 1.0
    actor_positions[:, 1, 0] = 25.0
    actor_positions[:, 1, 1] = -8.0
    velocities = np.zeros_like(actor_positions)
    velocities[:, 0, 0] = 2.0
    headings = np.zeros((horizon, 2), dtype=np.float32)
    sizes = np.broadcast_to(
        np.array([[[4.5, 2.0], [0.8, 0.8]]], dtype=np.float32),
        (horizon, 2, 2),
    ).copy()
    actors = LoggedActorFutures(
        track_tokens=("vehicle-a", "pedestrian-b"),
        positions=actor_positions,
        headings=headings,
        velocities=velocities,
        sizes=sizes,
        valid=np.ones((horizon, 2), dtype=bool),
    )
    drivable = np.array(
        [[-20.0, -12.0], [60.0, -12.0], [60.0, 12.0], [-20.0, 12.0]],
        dtype=np.float32,
    )
    route = np.stack(
        [np.linspace(-20.0, 60.0, 81), np.zeros(81)], axis=-1
    ).astype(np.float32)
    return ReplaySceneContext(
        route_centerline=route,
        drivable_polygons=(drivable,),
        static_obstacles=np.array([[30.0, 3.0, 0.0, 2.0, 2.0]], dtype=np.float32),
        logged_actors=actors,
        current_speed_mps=1.0,
        current_acceleration_mps2=0.0,
        traffic_light_states=(("lane-1", True),),
    )


def make_candidates() -> np.ndarray:
    time = np.arange(1, 9, dtype=np.float32)
    candidates = np.zeros((3, 8, 3), dtype=np.float32)
    candidates[0, :, 0] = 2.0 * time
    candidates[1, :, 0] = 2.0 * time
    candidates[1, :, 1] = np.linspace(0.0, 7.0, 8)
    candidates[1, :, 2] = 0.25
    candidates[2, :, 0] = 1.0 * time
    candidates[2, :, 1] = -4.0
    return candidates


def test_effect_shapes_determinism_and_fixed_actor_set() -> None:
    builder = ReplayGroundedEffectBuilder()
    context = make_context()
    candidates = make_candidates()
    first = builder.build(candidates, context)
    second = builder.build(candidates, context)
    np.testing.assert_array_equal(first.ego_effect, second.ego_effect)
    np.testing.assert_array_equal(first.map_effect, second.map_effect)
    np.testing.assert_array_equal(first.actor_effect, second.actor_effect)
    np.testing.assert_array_equal(first.actor_mask, second.actor_mask)
    np.testing.assert_array_equal(first.interaction_mask, second.interaction_mask)
    assert first.ego_effect.shape == (3, 8, len(EGO_EFFECT_NAMES))
    assert first.map_effect.shape == (3, 8, len(MAP_EFFECT_NAMES))
    assert first.actor_effect.shape == (3, 8, 16, len(ACTOR_EFFECT_NAMES))
    assert first.actor_mask.shape == (3, 8, 16)
    assert first.interaction_mask.shape == (3, 8, 16)
    np.testing.assert_array_equal(first.actor_mask[0], first.actor_mask[1])
    np.testing.assert_array_equal(first.actor_mask[1], first.actor_mask[2])
    assert first.selected_actor_tokens == ("vehicle-a", "pedestrian-b")


def test_lateral_candidate_changes_map_and_actor_relative_effects() -> None:
    result = ReplayGroundedEffectBuilder().build(make_candidates(), make_context())
    assert not np.array_equal(result.map_effect[0], result.map_effect[1])
    assert not np.array_equal(result.actor_effect[0], result.actor_effect[1])
    lateral_index = MAP_EFFECT_NAMES.index("distance_to_route_centerline")
    assert result.map_effect[1, -1, lateral_index] > result.map_effect[0, -1, lateral_index]


def test_logged_actor_future_is_bitwise_unchanged() -> None:
    context = make_context()
    before = {
        name: stable_array_hash(getattr(context.logged_actors, name))
        for name in ("positions", "headings", "velocities", "sizes", "valid")
    }
    ReplayGroundedEffectBuilder().build(make_candidates(), context)
    after = {
        name: stable_array_hash(getattr(context.logged_actors, name))
        for name in ("positions", "headings", "velocities", "sizes", "valid")
    }
    assert before == after


def test_effect_tensor_schema_has_no_metric_or_selection_labels() -> None:
    result = ReplayGroundedEffectBuilder().build(make_candidates(), make_context())
    assert set(result.as_tensor_dict()) == {
        "ego_effect",
        "map_effect",
        "actor_effect",
        "actor_mask",
        "interaction_mask",
    }
    forbidden = {"score", "factor", "pdms", "epdms", "selected_index", "nc", "dac", "ttc", "comfort"}
    feature_names = {name.lower() for name in EGO_EFFECT_NAMES + MAP_EFFECT_NAMES + ACTOR_EFFECT_NAMES}
    assert not (forbidden & feature_names)
    signature = inspect.signature(ReplayGroundedEffectBuilder.build)
    assert tuple(signature.parameters) == ("self", "candidates", "context")


def test_interaction_mask_is_not_a_collision_label() -> None:
    result = ReplayGroundedEffectBuilder().build(make_candidates(), make_context())
    clearance_index = ACTOR_EFFECT_NAMES.index("oriented_box_clearance")
    valid_clearances = result.actor_effect[..., clearance_index][result.actor_mask]
    assert (valid_clearances >= 0).all()
    assert result.interaction_mask.any()
    assert not np.array_equal(
        result.interaction_mask[result.actor_mask],
        valid_clearances == 0.0,
    )


def test_invalid_candidate_shape_fails_closed() -> None:
    with pytest.raises(EffectConstructionError, match="candidates must be"):
        ReplayGroundedEffectBuilder().build(np.zeros((3, 7, 3)), make_context())
