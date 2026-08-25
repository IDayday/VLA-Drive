# SPDX-License-Identifier: Apache-2.0
# NAVSIM proposal scoring is adapted from valeoai/DrivoR commit
# f02665403df799c1b4ddd8b0d34e073f0555c13a,
# navsim/agents/drivoR/score_module/compute_navsim_score.py, and from
# William-Yao-2000/DriveSuprim commit
# 80fe792d7654a596d92e20d030d1650f6f605c02,
# navsim/evaluate/pdm_score.py.  Project adaptations: per-rank local batches,
# named metrics, detached CPU inputs, vectorized pose conversion, lazy imports,
# and asynchronous per-rank worker pools.

"""NAVSIM metric labels for detached dynamic proposal pools."""

from __future__ import annotations

from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from multiprocessing import get_context
from pathlib import Path
from threading import local
from typing import Dict, Mapping, Optional, Sequence, Union

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

_PROCESS_EVALUATOR = None


def _new_navsim_evaluator(metric_cache_root: Path):
    """Construct one mutable NAVSIM evaluator for a thread or process worker."""

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
            "environment; it is needed for online Flow scoring and Stage-B "
            "Register candidate-bank construction"
        ) from exc
    sampling = TrajectorySampling(num_poses=40, interval_length=0.1)
    return (
        MetricCacheLoader(metric_cache_root),
        sampling,
        PDMSimulator(sampling),
        PDMScorer(sampling),
        LogReplayTrafficAgents(sampling),
    )


def _score_with_evaluator(
    evaluator,
    metric_cache_root: Path,
    token: str,
    trajectories_40: np.ndarray,
):
    loader, sampling, simulator, scorer, traffic_policy = evaluator
    if token not in loader.metric_cache_paths:
        raise FileNotFoundError(
            f"NAVSIM metric cache {metric_cache_root} has no token {token!r}"
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


def _initialize_process_evaluator(metric_cache_root: str) -> None:
    """Initialize persistent worker-local NAVSIM state under ``spawn``."""

    global _PROCESS_EVALUATOR
    _PROCESS_EVALUATOR = _new_navsim_evaluator(Path(metric_cache_root))


def _score_one_in_process(
    metric_cache_root: str, token: str, trajectories_40: np.ndarray
):
    if _PROCESS_EVALUATOR is None:
        raise RuntimeError("NAVSIM process worker was not initialized")
    return _score_with_evaluator(
        _PROCESS_EVALUATOR,
        Path(metric_cache_root),
        token,
        trajectories_40,
    )


def _batch_relative_trajectories_to_state_array(
    trajectories: np.ndarray,
    initial_ego_state,
    reference_states: np.ndarray,
) -> np.ndarray:
    """Build PDM ``[reference + K, 41, state]`` inputs without Python poses.

    NAVSIM's object pipeline constructs 40 ``StateSE2`` and ``EgoState``
    objects for every candidate even though the generated future states only
    contain x/y/heading (future velocity and acceleration are intentionally
    zero).  The homogeneous transform below is the vectorized equivalent of
    ``relative_to_absolute_poses`` used by ``transform_trajectory``.
    """

    from nuplan.common.geometry.convert import matrix_from_pose
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_array_representation import (
        ego_state_to_state_array,
    )
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import (
        StateIndex,
    )

    trajectories = np.asarray(trajectories, dtype=np.float64)
    reference_states = np.asarray(reference_states, dtype=np.float64)
    if trajectories.ndim != 3 or tuple(trajectories.shape[1:]) != (40, 3):
        raise ValueError("NAVSIM proposal evaluator expects [K,40,3]")
    expected_reference_shape = (41, StateIndex.size())
    if tuple(reference_states.shape) != expected_reference_shape:
        raise ValueError(
            "NAVSIM reference trajectory has shape "
            f"{reference_states.shape}, expected {expected_reference_shape}"
        )
    if not np.isfinite(trajectories).all():
        raise ValueError("NAVSIM proposal trajectories contain NaN or Inf")

    candidate_count = trajectories.shape[0]
    states = np.zeros(
        (candidate_count + 1, 41, StateIndex.size()), dtype=np.float64
    )
    states[0] = reference_states
    states[1:, 0] = ego_state_to_state_array(initial_ego_state)

    headings = trajectories[..., 2]
    cosine = np.cos(headings)
    sine = np.sin(headings)
    relative_transforms = np.zeros(
        (candidate_count, 40, 3, 3), dtype=np.float64
    )
    relative_transforms[..., 0, 0] = cosine
    relative_transforms[..., 0, 1] = -sine
    relative_transforms[..., 0, 2] = trajectories[..., 0]
    relative_transforms[..., 1, 0] = sine
    relative_transforms[..., 1, 1] = cosine
    relative_transforms[..., 1, 2] = trajectories[..., 1]
    relative_transforms[..., 2, 2] = 1.0

    absolute_transforms = (
        matrix_from_pose(initial_ego_state.rear_axle) @ relative_transforms
    )
    states[1:, 1:, StateIndex.X] = absolute_transforms[..., 0, 2]
    states[1:, 1:, StateIndex.Y] = absolute_transforms[..., 1, 2]
    states[1:, 1:, StateIndex.HEADING] = np.arctan2(
        absolute_transforms[..., 1, 0], absolute_transforms[..., 0, 0]
    )
    return states


class _DynamicMetricBatch:
    """A lazy, order-preserving collection of per-scene metric results."""

    def __init__(
        self,
        items: Sequence[Union[Mapping[str, np.ndarray], Future]],
        *,
        result_device: torch.device,
        result_dtype: torch.dtype,
    ) -> None:
        self._items = tuple(items)
        self._result_device = result_device
        self._result_dtype = result_dtype
        self._result: Optional[Dict[str, Tensor]] = None

    def result(self) -> Dict[str, Tensor]:
        """Wait only when labels are consumed and materialize the ``[B,K]`` tensors."""

        if self._result is None:
            rows = [
                item.result() if isinstance(item, Future) else item
                for item in self._items
            ]
            self._result = {
                name: torch.as_tensor(
                    np.stack([row[name] for row in rows], axis=0),
                    device=self._result_device,
                    dtype=self._result_dtype,
                )
                for name in _OUTPUT_METRICS
            }
        return self._result


def _score_trajectory_pool(
    metric_cache,
    trajectories: np.ndarray,
    sampling,
    simulator,
    scorer,
    traffic_policy,
) -> Dict[str, np.ndarray]:
    """Evaluate one token's physical ``[K,40,3]`` proposal pool."""

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
    reference_states = get_trajectory_as_array(
        metric_cache.trajectory, sampling, initial_ego_state.time_point
    )
    states = _batch_relative_trajectories_to_state_array(
        trajectories, initial_ego_state, reference_states
    )
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
        if self.backend not in {"local", "thread", "process"}:
            raise ValueError(
                "dynamic metric backend must be 'local', 'thread', or 'process'"
            )
        self.codec = TrajectoryCodec()
        self._thread_state = local()
        if self.backend == "thread":
            self._executor = ThreadPoolExecutor(
                max_workers=self.workers_per_rank,
                thread_name_prefix=f"navsim-rank{self.rank}",
            )
        elif self.backend == "process":
            # Never fork a rank after PPU/CUDA initialization. Spawned workers
            # import only CPU scoring code and retain one evaluator each.
            self._executor = ProcessPoolExecutor(
                max_workers=self.workers_per_rank,
                mp_context=get_context("spawn"),
                initializer=_initialize_process_evaluator,
                initargs=(str(self.metric_cache_root),),
            )
        else:
            self._executor = None
        # Fail at startup with the precise optional dependency if NAVSIM is
        # absent. Local/thread backends retain this evaluator; process workers
        # construct isolated persistent evaluators under the safe spawn mode.
        initial_evaluator = self._new_evaluator()
        if self.backend != "process":
            self._thread_state.evaluator = initial_evaluator

    def _new_evaluator(self):
        return _new_navsim_evaluator(self.metric_cache_root)

    def _evaluator(self):
        if not hasattr(self._thread_state, "evaluator"):
            self._thread_state.evaluator = self._new_evaluator()
        return self._thread_state.evaluator

    def _score_one(self, token: str, trajectories_40: np.ndarray):
        return _score_with_evaluator(
            self._evaluator(),
            self.metric_cache_root,
            token,
            trajectories_40,
        )

    @torch.no_grad()
    def _submit_40(
        self,
        tokens: Sequence[str],
        proposals_40: np.ndarray,
        *,
        result_device: torch.device,
        result_dtype: torch.dtype,
    ) -> _DynamicMetricBatch:
        if proposals_40.ndim != 4 or tuple(proposals_40.shape[-2:]) != (40, 3):
            raise ValueError("NAVSIM proposal pools must have shape [B,K,40,3]")
        if len(tokens) != proposals_40.shape[0]:
            raise ValueError("token count does not match dynamic proposal batch")
        tasks = [
            (str(token), proposals_40[index])
            for index, token in enumerate(tokens)
        ]
        if self._executor is None:
            items = [self._score_one(*task) for task in tasks]
        elif self.backend == "process":
            items = [
                self._executor.submit(
                    _score_one_in_process,
                    str(self.metric_cache_root),
                    token,
                    trajectories,
                )
                for token, trajectories in tasks
            ]
        else:
            items = [self._executor.submit(self._score_one, *task) for task in tasks]
        return _DynamicMetricBatch(
            items,
            result_device=result_device,
            result_dtype=result_dtype,
        )

    @torch.no_grad()
    def score_async(
        self, tokens: Sequence[str], proposals_navsim: Tensor
    ) -> _DynamicMetricBatch:
        """Submit detached 8-pose proposals and return before CPU scoring completes."""

        if proposals_navsim.ndim != 4 or tuple(proposals_navsim.shape[-2:]) != (8, 3):
            raise ValueError("dynamic proposals must have shape [B,K,8,3]")
        proposals_40 = self.codec.upsample_8_to_40(
            proposals_navsim.detach().to(device="cpu", dtype=torch.float32)
        ).numpy()
        return self._submit_40(
            tokens,
            proposals_40,
            result_device=proposals_navsim.device,
            result_dtype=proposals_navsim.dtype,
        )

    @torch.no_grad()
    def score_40_async(
        self, tokens: Sequence[str], proposals_navsim_40: Tensor
    ) -> _DynamicMetricBatch:
        """Score exact native 40-pose pools without an 8-pose resampling round trip."""

        if (
            proposals_navsim_40.ndim != 4
            or tuple(proposals_navsim_40.shape[-2:]) != (40, 3)
        ):
            raise ValueError("native NAVSIM proposals must have shape [B,K,40,3]")
        values = proposals_navsim_40.detach().to(
            device="cpu", dtype=torch.float32
        ).numpy()
        return self._submit_40(
            tokens,
            values,
            result_device=proposals_navsim_40.device,
            result_dtype=proposals_navsim_40.dtype,
        )

    @torch.no_grad()
    def score(
        self, tokens: Sequence[str], proposals_navsim: Tensor
    ) -> Dict[str, Tensor]:
        """Compatibility wrapper returning named synchronous ``[B,K]`` labels."""

        return self.score_async(tokens, proposals_navsim).result()

    @torch.no_grad()
    def score_40(
        self, tokens: Sequence[str], proposals_navsim_40: Tensor
    ) -> Dict[str, Tensor]:
        """Synchronous exact-40-pose compatibility wrapper."""

        return self.score_40_async(tokens, proposals_navsim_40).result()

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
    def score_async(
        self, tokens: Sequence[str], proposals_navsim: Tensor
    ) -> _DynamicMetricBatch:
        values = self.score(tokens, proposals_navsim)
        rows = [
            {
                name: values[name][index].detach().to("cpu").numpy()
                for name in _OUTPUT_METRICS
            }
            for index in range(proposals_navsim.shape[0])
        ]
        return _DynamicMetricBatch(
            rows,
            result_device=proposals_navsim.device,
            result_dtype=proposals_navsim.dtype,
        )

    @torch.no_grad()
    def score_40_async(
        self, tokens: Sequence[str], proposals_navsim_40: Tensor
    ) -> _DynamicMetricBatch:
        if (
            proposals_navsim_40.ndim != 4
            or tuple(proposals_navsim_40.shape[-2:]) != (40, 3)
        ):
            raise ValueError("native NAVSIM proposals must have shape [B,K,40,3]")
        values = self.score(tokens, proposals_navsim_40)
        rows = [
            {
                name: values[name][index].detach().to("cpu").numpy()
                for name in _OUTPUT_METRICS
            }
            for index in range(proposals_navsim_40.shape[0])
        ]
        return _DynamicMetricBatch(
            rows,
            result_device=proposals_navsim_40.device,
            result_dtype=proposals_navsim_40.dtype,
        )

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

    @torch.no_grad()
    def score_40(
        self, tokens: Sequence[str], proposals_navsim_40: Tensor
    ) -> Dict[str, Tensor]:
        return self.score_40_async(tokens, proposals_navsim_40).result()

    def close(self) -> None:
        return None
