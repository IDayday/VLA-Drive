"""Fast correctness tests for consequence geometry and provenance."""

from __future__ import annotations

import numpy as np
from nuplan.common.actor_state.agent import Agent
from nuplan.common.actor_state.oriented_box import OrientedBox
from nuplan.common.actor_state.scene_object import SceneObjectMetadata
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D
from nuplan.common.actor_state.static_object import StaticObject
from nuplan.common.actor_state.tracked_objects import TrackedObjects
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks

from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import StateIndex
from research.action_effect.consequence_builder import (
    geometry_against_tracks,
    physical_kinematics,
    validate_consequence_namespaces,
)


def _metadata(token: str) -> SceneObjectMetadata:
    return SceneObjectMetadata(
        timestamp_us=0,
        token=token,
        track_token=token,
        track_id=None,
    )


def _states(num_steps: int = 3) -> np.ndarray:
    states = np.zeros((num_steps, StateIndex.size()), dtype=np.float64)
    states[:, StateIndex.X] = np.arange(num_steps, dtype=np.float64)
    states[:, StateIndex.VELOCITY_X] = 1.0
    return states


def _tracks(objects: list[object], num_steps: int = 3) -> list[DetectionsTracks]:
    return [DetectionsTracks(TrackedObjects(objects)) for _ in range(num_steps)]


def test_dynamic_overlap_and_clearance_are_detected() -> None:
    agent = Agent(
        tracked_object_type=TrackedObjectType.VEHICLE,
        oriented_box=OrientedBox(StateSE2(2.0, 0.0, 0.0), 4.5, 2.0, 1.5),
        velocity=StateVector2D(0.0, 0.0),
        metadata=_metadata("agent"),
    )
    summary, arrays = geometry_against_tracks(
        _states(),
        _tracks([agent]),
        get_pacifica_parameters(),
        interval_length=0.1,
        clearance_cap_m=20.0,
        occupancy_radius_m=10.0,
    )
    assert summary["dynamic_collision"] is True
    assert summary["dynamic_collision_time_s"] == 0.0
    assert summary["minimum_dynamic_clearance_m"] == 0.0
    assert arrays["time_indexed_dynamic_overlap"].all()


def test_slowdown_does_not_move_static_map_object() -> None:
    static = StaticObject(
        tracked_object_type=TrackedObjectType.BARRIER,
        oriented_box=OrientedBox(StateSE2(30.0, 10.0, 0.0), 1.0, 1.0, 1.0),
        metadata=_metadata("barrier"),
    )
    states = _states()
    slowed = states.copy()
    slowed[:, StateIndex.X] *= 0.5
    first, _ = geometry_against_tracks(
        states,
        _tracks([static]),
        get_pacifica_parameters(),
        interval_length=0.1,
        clearance_cap_m=20.0,
        occupancy_radius_m=10.0,
    )
    second, _ = geometry_against_tracks(
        slowed,
        _tracks([static]),
        get_pacifica_parameters(),
        interval_length=0.1,
        clearance_cap_m=20.0,
        occupancy_radius_m=10.0,
    )
    assert first["static_object_collision"] is second["static_object_collision"] is False


def test_physical_consequence_is_deterministic() -> None:
    trajectory = np.stack([np.arange(1, 9), np.zeros(8), np.zeros(8)], axis=1)
    assert physical_kinematics(trajectory) == physical_kinematics(trajectory.copy())


def test_namespaces_cannot_be_mixed() -> None:
    row = {
        "exact": {"provenance": "exact", "drivable_area_compliance": 1.0},
        "log_replay": {"provenance": "log_replay", "dynamic_collision": False},
        "reactive_model": {"provenance": "reactive_model", "available": False},
        "unknown": {"provenance": "unknown"},
    }
    validate_consequence_namespaces(row)
    row["reactive_model"]["static_object_collision"] = False
    try:
        validate_consequence_namespaces(row)
    except AssertionError as error:
        assert "leaked" in str(error)
    else:  # pragma: no cover
        raise AssertionError("namespace validator accepted a mixed label")
