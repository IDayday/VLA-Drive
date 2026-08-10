from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from shapely.geometry import LineString, box

from starVLA.model.modules.grounded_world.navsim_consequence_provider import (
    COMPONENTS,
    _discover_metric_cache_paths,
    extract_physical_components,
)


class _FakeOccupancy:
    def __init__(self, geometries: dict[str, object]) -> None:
        self._geometries = geometries
        self.tokens = list(geometries)

    def __getitem__(self, token: str):
        return self._geometries[token]

    def query(self, geometry, predicate=None):
        del predicate
        return np.asarray(
            [
                index
                for index, token in enumerate(self.tokens)
                if geometry.intersects(self._geometries[token])
            ],
            dtype=np.int64,
        )


class _FakeObservation:
    red_light_token = "red_light"

    def __init__(self, maps: list[_FakeOccupancy]) -> None:
        self._maps = maps

    def __getitem__(self, index: int) -> _FakeOccupancy:
        return self._maps[index]


def _fake_scorer():
    candidates, timesteps = 2, 3
    centers = np.zeros((candidates, timesteps, 5, 2), dtype=np.float64)
    centers[0, :, 4] = np.asarray([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    centers[1, :, 4] = np.asarray([[0.0, 2.0], [0.5, 2.0], [1.0, 2.0]])
    polygons = np.empty((candidates, timesteps), dtype=object)
    for candidate in range(candidates):
        for timestep in range(timesteps):
            x, y = centers[candidate, timestep, 4]
            polygons[candidate, timestep] = box(x - 0.5, y - 0.5, x + 0.5, y + 0.5)
    multi = np.ones((4, candidates), dtype=np.float64)
    multi[0] = np.asarray([1.0, 0.0])
    weighted = np.ones((5, candidates), dtype=np.float64)
    weighted[3] = np.asarray([1.0, 0.0])
    obstacle = box(3.0, -0.5, 4.0, 0.5)
    red_light = box(0.0, -0.5, 0.2, 0.5)
    occupancy = _FakeObservation(
        [_FakeOccupancy({"obstacle": obstacle, "red_light_0": red_light})] * timesteps
    )
    return SimpleNamespace(
        proposal_sampling=SimpleNamespace(
            num_poses=timesteps - 1,
            interval_length=0.5,
            time_horizon=1.0,
        ),
        _num_proposals=candidates,
        _ego_coords=centers,
        _ego_polygons=polygons,
        _observation=occupancy,
        _centerline=SimpleNamespace(linestring=LineString([(0.0, 0.0), (10.0, 0.0)])),
        _collision_time_idcs=np.asarray([np.inf, 1.0]),
        _ttc_time_idcs=np.asarray([np.inf, 1.0]),
        _multi_metrics=multi,
        _weighted_metrics=weighted,
        _progress_raw=np.asarray([2.0, 1.0]),
    )


def test_extracts_only_interpretable_physical_components() -> None:
    output = extract_physical_components(_fake_scorer(), clearance_cap_m=10.0)
    assert COMPONENTS == (
        "clearance",
        "ttc",
        "collision",
        "lane_distance",
        "progress",
        "comfort",
    )
    assert output.values.shape == (2, 6)
    assert output.valid_mask.shape == output.values.shape
    assert output.valid_mask.all()
    np.testing.assert_allclose(output.values[:, 0], [0.5, np.sqrt(3.25)])
    np.testing.assert_allclose(output.values[:, 1], [1.0, 0.5])
    np.testing.assert_allclose(output.values[:, 2], [0.0, 1.0])
    np.testing.assert_allclose(output.values[:, 3], [0.0, 2.0])
    np.testing.assert_allclose(output.values[:, 4], [2.0, 1.0])
    np.testing.assert_allclose(output.values[:, 5], [1.0, 0.0])


def test_component_extraction_rejects_non_finite_or_mismatched_scorer_state() -> None:
    scorer = _fake_scorer()
    scorer._progress_raw = np.asarray([np.nan, 1.0])
    with pytest.raises(ValueError, match="non-finite"):
        extract_physical_components(scorer)

    scorer = _fake_scorer()
    scorer._ego_polygons = scorer._ego_polygons[:1]
    with pytest.raises(ValueError, match="shape"):
        extract_physical_components(scorer)


def test_metric_cache_discovery_is_root_relative_and_rejects_duplicates(
    tmp_path: Path,
) -> None:
    first = tmp_path / "log-a" / "unknown" / "token-a" / "metric_cache.pkl"
    first.parent.mkdir(parents=True)
    first.touch()
    assert _discover_metric_cache_paths(tmp_path) == {"token-a": first}

    duplicate = tmp_path / "log-b" / "unknown" / "token-a" / "metric_cache.pkl"
    duplicate.parent.mkdir(parents=True)
    duplicate.touch()
    with pytest.raises(ValueError, match="duplicate"):
        _discover_metric_cache_paths(tmp_path)
