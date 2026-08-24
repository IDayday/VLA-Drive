from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "navsim"))

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import (
    StateSE2,
    StateVector2D,
    TimePoint,
)
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.planning.simulation.trajectory.trajectory_sampling import (
    TrajectorySampling,
)

from navsim.common.dataclasses import Trajectory
from navsim.evaluate.pdm_score import get_trajectory_as_array, transform_trajectory
from starVLA.model.modules.planning.trajectory_codec import TrajectoryCodec
from starVLA.training.navsim_metric_supervisor import (
    DynamicMetricSupervisor,
    _OUTPUT_METRICS,
    _batch_relative_trajectories_to_state_array,
)


def _initial_ego_state() -> EgoState:
    return EgoState.build_from_rear_axle(
        rear_axle_pose=StateSE2(1_000_000.0, 2_000_000.0, 0.7),
        rear_axle_velocity_2d=StateVector2D(3.0, 0.1),
        rear_axle_acceleration_2d=StateVector2D(0.2, 0.0),
        tire_steering_angle=0.05,
        time_point=TimePoint(10_000_000),
        vehicle_parameters=get_pacifica_parameters(),
        is_in_auto_mode=True,
    )


def test_vectorized_relative_pose_batch_matches_legacy_object_pipeline():
    rng = np.random.default_rng(42)
    trajectories = rng.normal(size=(7, 40, 3)).astype(np.float64)
    trajectories[..., 0] = np.cumsum(np.abs(trajectories[..., 0]) * 0.1, axis=1)
    trajectories[..., 1] *= 0.2
    trajectories[..., 2] *= 0.1
    initial = _initial_ego_state()
    sampling = TrajectorySampling(num_poses=40, interval_length=0.1)
    trajectory_sampling = TrajectorySampling(time_horizon=4.0, interval_length=0.1)
    legacy = np.stack(
        [
            get_trajectory_as_array(
                transform_trajectory(Trajectory(poses, trajectory_sampling), initial),
                sampling,
                initial.time_point,
            )
            for poses in trajectories
        ],
        axis=0,
    )
    human_states = legacy[0].copy()

    batched = _batch_relative_trajectories_to_state_array(
        trajectories, initial, human_states
    )

    np.testing.assert_array_equal(batched[0], human_states)
    np.testing.assert_allclose(batched[1:], legacy, rtol=0.0, atol=1e-9)


def test_async_metric_batch_preserves_order_shape_and_values():
    supervisor = object.__new__(DynamicMetricSupervisor)
    supervisor.codec = TrajectoryCodec()
    supervisor.backend = "thread"
    supervisor._executor = ThreadPoolExecutor(max_workers=2)

    def score_one(token, trajectories_40):
        token_value = float(int(token))
        count = trajectories_40.shape[0]
        return {
            name: np.full(count, token_value + index, dtype=np.float32)
            for index, name in enumerate(_OUTPUT_METRICS)
        }

    supervisor._score_one = score_one
    proposals = torch.zeros(3, 5, 8, 3, dtype=torch.float32)
    try:
        pending = supervisor.score_async(["3", "1", "2"], proposals)
        result = pending.result()
    finally:
        supervisor.close()

    assert list(result) == list(_OUTPUT_METRICS)
    for index, name in enumerate(_OUTPUT_METRICS):
        assert result[name].shape == (3, 5)
        torch.testing.assert_close(
            result[name][:, 0], torch.tensor([3.0, 1.0, 2.0]) + index
        )


def test_process_backend_uses_spawn_context(monkeypatch, tmp_path):
    monkeypatch.setattr(
        DynamicMetricSupervisor,
        "_new_evaluator",
        lambda _self: object(),
    )
    supervisor = DynamicMetricSupervisor(
        {
            "backend": "process",
            "workers_per_rank": 2,
            "metric_cache_root": str(tmp_path),
            "score_interval": 1,
        }
    )
    try:
        assert supervisor._executor._mp_context.get_start_method() == "spawn"
        assert supervisor._executor._max_workers == 2
    finally:
        supervisor.close()
