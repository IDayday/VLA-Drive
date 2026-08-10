"""Offline NAVSIM physical consequence labels for fixed trajectory perturbations.

This adapter deliberately uses the cached, non-reactive logged environment.  It
does not estimate aggregate EPDMS and it does not run in the model forward path.
"""

from __future__ import annotations

import lzma
import pickle
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from shapely.geometry import Point

from .teacher_protocols import PhysicalConsequenceOutput


COMPONENTS = (
    "clearance",
    "ttc",
    "collision",
    "lane_distance",
    "progress",
    "comfort",
)

# Audited against the vendored NAVSIM ``pdm_enums.py``.  Keeping these tiny
# indices local lets the pure extraction helper remain CPU-testable without
# importing NAVSIM/nuPlan at module import time.
_NO_COLLISION_INDEX = 0
_HISTORY_COMFORT_INDEX = 3
_BOUNDING_BOX_CENTER_INDEX = 4


def _discover_metric_cache_paths(root: str | Path) -> dict[str, Path]:
    """Return ``token -> metric_cache.pkl`` under ``root`` without stale CSV paths."""

    cache_root = Path(root)
    if not cache_root.is_dir():
        raise FileNotFoundError(f"NAVSIM metric cache root not found: {cache_root}")
    paths: dict[str, Path] = {}
    for path in sorted(cache_root.rglob("metric_cache.pkl")):
        token = path.parent.name
        if not token:
            raise ValueError(f"metric cache has no token parent: {path}")
        if token in paths:
            raise ValueError(
                f"duplicate NAVSIM metric cache token {token!r}: "
                f"{paths[token]} and {path}"
            )
        paths[token] = path
    if not paths:
        raise FileNotFoundError(f"no metric_cache.pkl entries under {cache_root}")
    return paths


def _minimum_clearance(scorer: Any, clearance_cap_m: float) -> np.ndarray:
    """Compute non-reactive obstacle clearance from scorer polygons ``[K,T]``."""

    ego_polygons = np.asarray(scorer._ego_polygons, dtype=object)
    candidates = int(scorer._num_proposals)
    expected = (candidates, int(scorer.proposal_sampling.num_poses) + 1)
    if ego_polygons.shape != expected:
        raise ValueError(
            f"scorer ego polygon shape must be {expected}, got {ego_polygons.shape}"
        )
    clearance = np.full(candidates, float(clearance_cap_m), dtype=np.float64)
    red_light_prefix = str(scorer._observation.red_light_token)
    for candidate in range(candidates):
        for timestep in range(expected[1]):
            ego_polygon = ego_polygons[candidate, timestep]
            occupancy = scorer._observation[timestep]
            nearby = occupancy.query(ego_polygon.buffer(clearance_cap_m))
            for geometry_index in np.asarray(nearby, dtype=np.int64).reshape(-1):
                token = occupancy.tokens[int(geometry_index)]
                if token.startswith(red_light_prefix):
                    continue
                distance = float(ego_polygon.distance(occupancy[token]))
                clearance[candidate] = min(clearance[candidate], distance)
    return clearance


def extract_physical_components(
    scorer: Any,
    *,
    clearance_cap_m: float = 50.0,
) -> PhysicalConsequenceOutput:
    """Extract ``values/valid_mask=[K,6]`` after ``PDMScorer.score_proposals``.

    The TTC value is the finite time until the first NAVSIM TTC infraction and
    is capped at the planning horizon when no infraction occurs.  Collision is
    binary at-fault collision.  Lane distance is maximum centerline deviation
    in metres; progress is the scorer's unnormalised progress in metres.
    """

    if not np.isfinite(clearance_cap_m) or clearance_cap_m <= 0:
        raise ValueError("clearance_cap_m must be finite and positive")
    candidates = int(scorer._num_proposals)
    timesteps = int(scorer.proposal_sampling.num_poses) + 1
    centers = np.asarray(scorer._ego_coords, dtype=np.float64)
    if centers.shape[:2] != (candidates, timesteps) or centers.shape[-2:] != (5, 2):
        raise ValueError(
            "scorer ego coordinate shape must be [K,T,5,2], "
            f"got {centers.shape}"
        )

    interval_s = float(scorer.proposal_sampling.interval_length)
    horizon_s = float(scorer.proposal_sampling.time_horizon)
    ttc_indices = np.asarray(scorer._ttc_time_idcs, dtype=np.float64)
    no_collision = np.asarray(
        scorer._multi_metrics[_NO_COLLISION_INDEX], dtype=np.float64
    )
    progress = np.asarray(scorer._progress_raw, dtype=np.float64)
    comfort = np.asarray(
        scorer._weighted_metrics[_HISTORY_COMFORT_INDEX], dtype=np.float64
    )
    for name, value in (
        ("ttc indices", ttc_indices),
        ("collision", no_collision),
        ("progress", progress),
        ("comfort", comfort),
    ):
        if value.shape != (candidates,):
            raise ValueError(f"scorer {name} shape must be [K], got {value.shape}")

    ttc = np.where(np.isfinite(ttc_indices), ttc_indices * interval_s, horizon_s)
    ttc = np.clip(ttc, 0.0, horizon_s)
    collision = (no_collision < 1.0).astype(np.float64)
    centerline = scorer._centerline.linestring
    lane_distance = np.zeros(candidates, dtype=np.float64)
    for candidate in range(candidates):
        lane_distance[candidate] = max(
            float(
                Point(*centers[candidate, timestep, _BOUNDING_BOX_CENTER_INDEX]).distance(
                    centerline
                )
            )
            for timestep in range(timesteps)
        )
    clearance = _minimum_clearance(scorer, float(clearance_cap_m))
    values = np.stack(
        (clearance, ttc, collision, lane_distance, progress, comfort), axis=-1
    ).astype(np.float32)
    if values.shape != (candidates, len(COMPONENTS)):
        raise ValueError("physical consequence values must have shape [K,6]")
    if not np.isfinite(values).all():
        raise ValueError("physical consequence values contain non-finite values")
    return PhysicalConsequenceOutput(
        values=values,
        valid_mask=np.ones_like(values, dtype=np.bool_),
    ).validate(candidates)


class NavsimNonReactiveConsequenceProvider:
    """Batch NAVSIM scorer adapter over logged, non-reactive future occupancy."""

    def __init__(
        self,
        metric_cache_root: str | Path,
        *,
        clearance_cap_m: float = 50.0,
    ) -> None:
        self._cache_paths = _discover_metric_cache_paths(metric_cache_root)
        self._clearance_cap_m = float(clearance_cap_m)
        if not np.isfinite(self._clearance_cap_m) or self._clearance_cap_m <= 0:
            raise ValueError("clearance_cap_m must be finite and positive")
        try:
            from nuplan.planning.simulation.trajectory.trajectory_sampling import (
                TrajectorySampling,
            )
            from navsim.common.dataclasses import Trajectory
            from navsim.evaluate.pdm_score import (
                get_trajectory_as_array,
                transform_trajectory,
            )
            from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import (
                PDMScorer,
                PDMScorerConfig,
            )
            from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import (
                PDMSimulator,
            )
        except (ImportError, ModuleNotFoundError) as error:
            raise RuntimeError(
                "NAVSIM consequence provider requires the local vendored navsim and "
                "nuplan packages; source env.sh or add <repo>/navsim to PYTHONPATH"
            ) from error

        self._trajectory_type = Trajectory
        self._transform_trajectory = transform_trajectory
        self._get_trajectory_as_array = get_trajectory_as_array
        self._model_sampling = TrajectorySampling(time_horizon=4.0, interval_length=0.5)
        self._score_sampling = TrajectorySampling(time_horizon=4.0, interval_length=0.1)
        self._simulator = PDMSimulator(self._score_sampling)
        self._scorer = PDMScorer(
            self._score_sampling,
            PDMScorerConfig(human_penalty_filter=False),
        )

    @property
    def metadata(self) -> Mapping[str, Any]:
        return {
            "name": "navsim_nonreactive_physical_components_v1",
            "returns_aggregate_epdms": False,
            "uses_logged_future_agents": True,
            "logged_future_agents_role": "non_reactive_proxy_only",
            "reactive_counterfactual": False,
            "model_sampling": {"num_poses": 8, "interval_s": 0.5},
            "score_sampling": {"num_poses": 40, "interval_s": 0.1},
            "clearance_cap_m": self._clearance_cap_m,
            "components": COMPONENTS,
        }

    def _load_metric_cache(self, token: str) -> Any:
        if token not in self._cache_paths:
            raise FileNotFoundError(f"NAVSIM metric cache token not found: {token}")
        path = self._cache_paths[token]
        try:
            with lzma.open(path, "rb") as stream:
                return pickle.load(stream)
        except (OSError, EOFError, lzma.LZMAError, pickle.UnpicklingError) as error:
            raise ValueError(f"corrupt NAVSIM metric cache: {path}") from error

    def label(
        self,
        token: str,
        metadata: Mapping[str, Any],
        physical_trajectories: np.ndarray,
    ) -> PhysicalConsequenceOutput:
        """Score local physical trajectories ``[K,8,3]`` for ``token`` offline."""

        del metadata
        trajectories = np.asarray(physical_trajectories, dtype=np.float32)
        if trajectories.ndim != 3 or trajectories.shape[1:] != (8, 3):
            raise ValueError("physical_trajectories must have shape [K,8,3]")
        if trajectories.shape[0] <= 0 or not np.isfinite(trajectories).all():
            raise ValueError("physical_trajectories must be non-empty and finite")
        metric_cache = self._load_metric_cache(str(token))
        state_arrays = []
        for candidate in trajectories:
            model_trajectory = self._trajectory_type(
                poses=candidate,
                trajectory_sampling=self._model_sampling,
            )
            interpolated = self._transform_trajectory(
                model_trajectory, metric_cache.ego_state
            )
            state_arrays.append(
                self._get_trajectory_as_array(
                    interpolated,
                    self._score_sampling,
                    metric_cache.ego_state.time_point,
                )
            )
        requested_states = np.stack(state_arrays, axis=0)
        simulated_states = self._simulator.simulate_proposals(
            requested_states, metric_cache.ego_state
        )
        self._scorer.score_proposals(
            simulated_states,
            metric_cache.observation,
            metric_cache.centerline,
            metric_cache.route_lane_ids,
            metric_cache.drivable_area_map,
            metric_cache.map_parameters,
            simulated_agent_detections_tracks=None,
            human_past_trajectory=metric_cache.past_human_trajectory,
        )
        return extract_physical_components(
            self._scorer, clearance_cap_m=self._clearance_cap_m
        )


def build_navsim_nonreactive_provider(
    metric_cache_root: str | Path,
) -> NavsimNonReactiveConsequenceProvider:
    """Factory used by ``build_consequence_labels.py --provider-factory``."""

    return NavsimNonReactiveConsequenceProvider(metric_cache_root)
