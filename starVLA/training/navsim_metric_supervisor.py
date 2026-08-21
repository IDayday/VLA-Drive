# SPDX-License-Identifier: Apache-2.0
# NAVSIM proposal scoring is adapted from valeoai/DrivoR commit
# f02665403df799c1b4ddd8b0d34e073f0555c13a,
# navsim/agents/drivoR/score_module/compute_navsim_score.py, and from
# William-Yao-2000/DriveSuprim commit
# 80fe792d7654a596d92e20d030d1650f6f605c02,
# navsim/evaluate/pdm_score.py.  Project adaptations: per-rank local batches,
# named metrics, detached CPU inputs, lazy imports, and optional thread workers.

"""Online NAVSIM metric labels for detached dynamic Flow proposals."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import local
from typing import Dict, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor

from starVLA.model.modules.planning.trajectory_codec import TrajectoryCodec


_OUTPUT_METRICS = (
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "time_to_collision_within_bound",
    "ego_progress",
    "driving_direction_compliance",
    "comfort",
    "lane_keeping",
    "traffic_light_compliance",
    "history_comfort",
    "aggregate_score",
)


def _score_trajectory_pool(
    metric_cache,
    trajectories: np.ndarray,
    sampling,
    simulator,
    scorer,
    traffic_policy,
) -> Dict[str, np.ndarray]:
    """Evaluate one token's physical ``[K,40,3]`` proposal pool."""

    from navsim.common.dataclasses import Trajectory
    from navsim.common.enums import SceneFrameType
    from navsim.evaluate.pdm_score import get_trajectory_as_array, transform_trajectory
    from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_comfort_metrics import (
        ego_is_comfortable,
    )
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import (
        MultiMetricIndex,
        WeightedMetricIndex,
    )

    if trajectories.ndim != 3 or tuple(trajectories.shape[1:]) != (40, 3):
        raise ValueError("NAVSIM proposal evaluator expects [K,40,3]")
    initial_ego_state = metric_cache.ego_state
    all_states = [
        get_trajectory_as_array(
            metric_cache.trajectory, sampling, initial_ego_state.time_point
        )
    ]
    trajectory_sampling = type(sampling)(time_horizon=4.0, interval_length=0.1)
    for poses in trajectories:
        transformed = transform_trajectory(
            Trajectory(poses, trajectory_sampling), initial_ego_state
        )
        all_states.append(
            get_trajectory_as_array(
                transformed, sampling, initial_ego_state.time_point
            )
        )
    states = np.stack(all_states, axis=0)
    simulated_states = simulator.simulate_proposals(states, initial_ego_state)
    traffic_tracks = traffic_policy.simulate_environment(
        simulated_states[1], metric_cache
    )
    if len(traffic_tracks) != states.shape[1]:
        raise RuntimeError("NAVSIM traffic policy returned an invalid horizon")
    frames = scorer.score_proposals(
        simulated_states,
        metric_cache.observation,
        metric_cache.centerline,
        metric_cache.route_lane_ids,
        metric_cache.drivable_area_map,
        metric_cache.map_parameters,
        traffic_tracks,
        metric_cache.past_human_trajectory,
    )
    values = {
        "no_at_fault_collisions": scorer._multi_metrics[
            MultiMetricIndex.NO_COLLISION
        ][1:],
        "drivable_area_compliance": scorer._multi_metrics[
            MultiMetricIndex.DRIVABLE_AREA
        ][1:],
        "driving_direction_compliance": scorer._multi_metrics[
            MultiMetricIndex.DRIVING_DIRECTION
        ][1:],
        "traffic_light_compliance": scorer._multi_metrics[
            MultiMetricIndex.TRAFFIC_LIGHT_COMPLIANCE
        ][1:],
        "ego_progress": scorer._weighted_metrics[WeightedMetricIndex.PROGRESS][1:],
        "time_to_collision_within_bound": scorer._weighted_metrics[
            WeightedMetricIndex.TTC
        ][1:],
        "lane_keeping": scorer._weighted_metrics[WeightedMetricIndex.LANE_KEEPING][1:],
        "history_comfort": scorer._weighted_metrics[
            WeightedMetricIndex.HISTORY_COMFORT
        ][1:],
        "aggregate_score": np.asarray(
            [float(frame["pdm_score"].iloc[0]) for frame in frames[1:]],
            dtype=np.float64,
        ),
    }
    time_points = np.arange(
        sampling.num_poses + 1, dtype=np.float64
    ) * sampling.interval_length
    values["comfort"] = ego_is_comfortable(
        simulated_states, time_points
    ).all(axis=-1)[1:]

    # Preserve donor human-penalty filtering when supported by this NAVSIM.
    if getattr(scorer._config, "human_penalty_filter", False) and (
        metric_cache.scene_type == SceneFrameType.ORIGINAL
    ):
        human = transform_trajectory(metric_cache.human_trajectory, initial_ego_state)
        human_states = get_trajectory_as_array(
            human, sampling, initial_ego_state.time_point
        )
        human_simulated = simulator.simulate_proposals(
            human_states[None], initial_ego_state
        )
        human_tracks = traffic_policy.simulate_environment(
            human_simulated[0], metric_cache
        )
        human_result = scorer.score_proposals(
            human_simulated,
            metric_cache.observation,
            metric_cache.centerline,
            metric_cache.route_lane_ids,
            metric_cache.drivable_area_map,
            metric_cache.map_parameters,
            human_tracks,
        )[0]
        for name in tuple(values):
            if name not in {"aggregate_score", "comfort"} and float(
                human_result[name].iloc[0]
            ) == 0.0:
                values[name] = np.ones_like(values[name])
        if not ego_is_comfortable(human_simulated, time_points).all(axis=-1)[0]:
            values["comfort"] = np.ones_like(values["comfort"])

    expected = trajectories.shape[0]
    for name in _OUTPUT_METRICS:
        value = np.asarray(values[name])
        if value.shape != (expected,):
            raise RuntimeError(
                f"NAVSIM metric {name} has shape {value.shape}, expected ({expected},)"
            )
        if not np.isfinite(value).all():
            raise ValueError(f"NAVSIM metric {name} contains NaN or Inf")
        values[name] = value.astype(np.float32, copy=False)
    return values


class DynamicMetricSupervisor:
    """Per-rank, non-module NAVSIM evaluator for detached dynamic proposals."""

    def __init__(self, config, rank: int = 0, world_size: int = 1) -> None:
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.backend = str(config.get("backend", "thread"))
        self.workers_per_rank = int(config.get("workers_per_rank", 1))
        cache_root = config.get("metric_cache_root")
        if cache_root is None or str(cache_root).strip().lower() in {"", "null", "none"}:
            raise FileNotFoundError(
                "dynamic_metric_supervisor.metric_cache_root is required for training"
            )
        self.metric_cache_root = Path(str(cache_root)).expanduser()
        self.score_interval = int(config.get("score_interval", 1))
        if not self.metric_cache_root.is_dir():
            raise FileNotFoundError(
                f"NAVSIM metric cache root does not exist: {self.metric_cache_root}"
            )
        if self.workers_per_rank <= 0:
            raise ValueError("workers_per_rank must be positive")
        if self.score_interval != 1:
            raise ValueError(
                "online generated proposals require dynamic score_interval=1"
            )
        if self.backend not in {"local", "thread"}:
            raise ValueError("dynamic metric backend must be 'local' or 'thread'")
        self.codec = TrajectoryCodec()
        self._thread_state = local()
        self._executor = (
            ThreadPoolExecutor(
                max_workers=self.workers_per_rank,
                thread_name_prefix=f"navsim-rank{self.rank}",
            )
            if self.backend == "thread" and self.workers_per_rank > 1
            else None
        )
        # Fail at startup with the precise optional dependency if NAVSIM is
        # absent, and retain the evaluator on the constructing thread instead
        # of loading the usually large metric-cache index twice.
        self._thread_state.evaluator = self._new_evaluator()

    def _new_evaluator(self):
        try:
            from nuplan.planning.simulation.trajectory.trajectory_sampling import (
                TrajectorySampling,
            )
            from navsim.common.dataloader import MetricCacheLoader
            from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import (
                PDMScorer,
            )
            from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import (
                PDMSimulator,
            )
            from navsim.traffic_agents_policies.log_replay_traffic_agents import (
                LogReplayTrafficAgents,
            )
        except ImportError as exc:
            raise ImportError(
                "DynamicMetricSupervisor requires the NAVSIM/nuPlan training "
                "environment; install it only for QwenPI-DrivoRSuprim training"
            ) from exc
        sampling = TrajectorySampling(num_poses=40, interval_length=0.1)
        return (
            MetricCacheLoader(self.metric_cache_root),
            sampling,
            PDMSimulator(sampling),
            PDMScorer(sampling),
            LogReplayTrafficAgents(sampling),
        )

    def _evaluator(self):
        if not hasattr(self._thread_state, "evaluator"):
            self._thread_state.evaluator = self._new_evaluator()
        return self._thread_state.evaluator

    def _score_one(self, token: str, trajectories_40: np.ndarray):
        loader, sampling, simulator, scorer, traffic_policy = self._evaluator()
        if token not in loader.metric_cache_paths:
            raise FileNotFoundError(
                f"NAVSIM metric cache {self.metric_cache_root} has no token {token!r}"
            )
        metric_cache = loader.get_from_token(token)
        return _score_trajectory_pool(
            metric_cache,
            trajectories_40,
            sampling,
            simulator,
            scorer,
            traffic_policy,
        )

    @torch.no_grad()
    def score(
        self, tokens: Sequence[str], proposals_navsim: Tensor
    ) -> Dict[str, Tensor]:
        """Return named ``[B,K]`` labels for detached physical 8-pose inputs."""

        if proposals_navsim.ndim != 4 or tuple(proposals_navsim.shape[-2:]) != (8, 3):
            raise ValueError("dynamic proposals must have shape [B,K,8,3]")
        if len(tokens) != proposals_navsim.shape[0]:
            raise ValueError("token count does not match dynamic proposal batch")
        result_device = proposals_navsim.device
        result_dtype = proposals_navsim.dtype
        proposals_40 = self.codec.upsample_8_to_40(
            proposals_navsim.detach().to(device="cpu", dtype=torch.float32)
        ).numpy()
        tasks = [(str(token), proposals_40[index]) for index, token in enumerate(tokens)]
        if self._executor is None:
            rows = [self._score_one(*task) for task in tasks]
        else:
            futures = [self._executor.submit(self._score_one, *task) for task in tasks]
            rows = [future.result() for future in futures]
        return {
            name: torch.as_tensor(
                np.stack([row[name] for row in rows], axis=0),
                device=result_device,
                dtype=result_dtype,
            )
            for name in _OUTPUT_METRICS
        }

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=False)
            self._executor = None


class StubDynamicMetricSupervisor:
    """Injectable deterministic callable for tests; never used by production config."""

    def __init__(self, scorer=None) -> None:
        self.scorer = scorer
        self.calls = 0

    @torch.no_grad()
    def score(
        self, tokens: Sequence[str], proposals_navsim: Tensor
    ) -> Dict[str, Tensor]:
        self.calls += 1
        if self.scorer is not None:
            result = self.scorer(tokens, proposals_navsim.detach())
            missing = set(_OUTPUT_METRICS).difference(result)
            if missing:
                raise KeyError(f"stub dynamic scorer is missing {sorted(missing)}")
            return result
        base = torch.sigmoid(proposals_navsim.detach()[..., 0].mean(dim=-1))
        return {name: base.clone() for name in _OUTPUT_METRICS}

    def close(self) -> None:
        return None
