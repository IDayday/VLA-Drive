"""Behavior tests against native v1 predicates; fixtures are not real-data evidence."""
from dataclasses import replace
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from starVLA.model.modules.st_world import consequence as c


ROOT = Path(__file__).resolve().parents[2]
NAVSIM_ROOT = ROOT / "navsim_v1.1" / "navsim"


@pytest.fixture(scope="module")
def native():
    pytest.importorskip("nuplan")
    sys.path.insert(0, str(NAVSIM_ROOT))
    from navsim.planning.simulation.planner.pdm_planner.scoring import pdm_scorer
    return pdm_scorer


def make_scorer(native, contacts=None, initial_collided=(), road_half_width=20.0, y=None, speed=2.0):
    from shapely.geometry import box
    from navsim.planning.simulation.planner.pdm_planner.observation.pdm_observation import PDMObservation
    from navsim.planning.simulation.planner.pdm_planner.observation.pdm_occupancy_map import PDMOccupancyMap, PDMDrivableMap
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_path import PDMPath
    from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
    contacts = contacts or {}
    sampling = native.TrajectorySampling(num_poses=40, interval_length=0.1)
    obs = PDMObservation(sampling, sampling, 50, observation_sample_res=1)
    obs._initialized = True
    obs._collided_track_ids = list(initial_collided)
    obs._unique_objects = {}
    obs._occupancy_maps = []
    for t in range(51):
        entries = contacts.get(t, [])
        tokens, geometries = [], []
        for token, kind, x, yy in entries:
            tokens.append(token)
            geometries.append(box(x - 0.1, yy - 0.1, x + 0.1, yy + 0.1))
            obs._unique_objects[token] = SimpleNamespace(
                tracked_object_type=getattr(TrackedObjectType, kind),
                box=SimpleNamespace(center=native.StateSE2(x, yy, 0.0)))
        obs._occupancy_maps.append(PDMOccupancyMap(tokens, np.array(geometries, dtype=object)))
    drivable = PDMDrivableMap(["road", "lane"],
        [native.SemanticMapLayer.ROADBLOCK, native.SemanticMapLayer.LANE],
        np.array([box(-100, -road_half_width, 100, road_half_width),
                  box(-100, -road_half_width, 100, road_half_width)], dtype=object))
    states = np.zeros((2, 41, 11), dtype=np.float64)
    states[:, :, 3] = speed
    if y is not None:
        states[:, :, 1] = y
    scorer = native.PDMScorer(sampling)
    center = PDMPath([native.StateSE2(-100, 0, 0), native.StateSE2(100, 0, 0)])
    scorer.score_proposals(states, obs, center, ["lane"], drivable)
    return scorer, obs


def test_right_closed_event_bins_preserve_t0():
    assert [c.event_bin(i) for i in (0, 1, 5, 6, 10, 35, 36, 40)] == [0, 0, 0, 1, 1, 6, 7, 7]
    with pytest.raises(c.ContractMismatch):
        c.event_bin(41)
    flags = np.zeros(41, dtype=bool)
    flags[[0, 5, 10, 40]] = True
    road = c.road_summary(flags)
    assert road["first_time_seconds"] == 0
    assert road["segment_any"] == [True, True, False, False, False, False, False, True]


def test_no_events_are_valid_null_not_infinity():
    assert c.road_summary(np.zeros(41, dtype=bool))["first_time_seconds"] is None
    assert c.collision_summary([]) == {"segment_nc_min": [1.] * 8, "nc_min": 1.,
        "first_at_fault_time_seconds": None, "event": False, "valid": True}
    assert c.ttc_summary([])["first_state_time_seconds"] is None


def test_full_body_corners_and_out_then_return(native):
    scorer, _ = make_scorer(native, road_half_width=1.0)
    assert np.all(scorer._states[1, :, 1] == 0)  # Center remains on road.
    observed = c.observe_canonical(scorer)
    assert observed["road"]["any"] and scorer._multi_metrics[1, 1] == 0
    y = np.zeros(41)
    y[20] = 5.0
    scorer, _ = make_scorer(native, y=y, road_half_width=3)
    observed = c.observe_canonical(scorer)
    assert not observed["road"]["native_flag"][-1]
    assert observed["road"]["first_time_seconds"] == 2.0
    assert observed["road"]["segment_any"][3]
    assert scorer._multi_metrics[1, 1] == 0


def test_static_half_then_agent_zero_and_repeated_contacts(native):
    contacts = {5: [("cone", "TRAFFIC_CONE", 2, 0)],
                6: [("cone", "TRAFFIC_CONE", 2, 0)],
                12: [("car", "VEHICLE", 2, 0)]}
    scorer, obs = make_scorer(native, contacts)
    original = list(obs.collided_track_ids)
    before = scorer._multi_metrics.copy()
    result = c.observe_canonical(scorer)
    events = result["collision_events"]
    assert [(e["state_index"], e["nc_contribution"]) for e in events] == [(5, .5), (6, .5), (12, 0.)]
    assert [e["new_minimum"] for e in events] == [True, False, True]
    assert result["collision_summary"]["first_at_fault_time_seconds"] == .5
    assert result["collision_summary"]["nc_min"] == 0.
    assert obs.collided_track_ids == original
    np.testing.assert_array_equal(before, scorer._multi_metrics)


def test_non_at_fault_initial_contacts_and_red_lights_are_suppressed(native):
    contacts = {0: [("later", "VEHICLE", 2, 0), ("ignored", "VEHICLE", 2, 0),
                    ("red_light_1", "VEHICLE", 2, 0)],
                1: [("later", "VEHICLE", 2, 0), ("ignored", "VEHICLE", 2, 0)]}
    scorer, _ = make_scorer(native, contacts, initial_collided=("ignored",), speed=0)
    scorer._states[:, 1:, 3] = 2
    # Recompute only the native NC after changing the synthetic state speeds.
    scorer._collision_time_idcs[:] = np.inf
    scorer._calculate_no_at_fault_collision()
    rows = c.collision_events(scorer)[1]
    assert len(rows) == 1 and rows[0]["object_token"] == "later"
    assert not rows[0]["at_fault"]
    assert rows[0]["collision_type"] == "STOPPED_EGO_COLLISION"
    assert c.collision_summary(rows)["nc_min"] == scorer._multi_metrics[0, 1] == 1.


def test_ttc_last_state_reads49_and_reports_state_time(native):
    scorer, _ = make_scorer(native, {49: [("late", "VEHICLE", 4.5, 0)]})
    result = c.observe_canonical(scorer)
    rows = result["ttc_events"]
    assert len(rows) == 1
    assert rows[0]["state_index"] == 40
    assert rows[0]["observation_index"] == 49
    assert rows[0]["projection_offset_steps"] == 9
    assert rows[0]["violation"]
    assert result["ttc_summary"]["first_state_time_seconds"] == 4.0
    assert result["ttc_summary"]["segment_any"][-1]


def test_ttc_stopped_threshold_and_behind_filter(native):
    scorer, _ = make_scorer(native, {4: [("front", "VEHICLE", 2, 0)]}, speed=.0049)
    assert c.observe_canonical(scorer)["ttc_events"] == []
    scorer, _ = make_scorer(native, {0: [("rear", "VEHICLE", -0.1, 0)],
                                   3: [("rear", "VEHICLE", 2, 0)]})
    rows = c.ttc_events(scorer)[1]
    assert len(rows) == 1 and not rows[0]["violation"]


def test_observer_detects_canonical_disagreement(native):
    scorer, _ = make_scorer(native)
    scorer._multi_metrics[0, 1] = 0
    with pytest.raises(c.ContractMismatch, match="disagrees"):
        c.observe_canonical(scorer)
    scorer._num_proposals = 3
    with pytest.raises(c.ContractMismatch, match="exactly"):
        c.observe_canonical(scorer)


def test_dynamic_pool_changes_ep_negative_control(native):
    scorer, _ = make_scorer(native)
    scorer._progress_raw = np.array([10., 4.])
    scorer._multi_metrics[:] = 1
    scorer._weighted_metrics[:] = 1
    scorer._aggregate_scores()
    fixed_ep = scorer._weighted_metrics[0, 1]
    scorer._progress_raw = np.array([10., 4., 20.])
    scorer._multi_metrics = np.ones((2, 3))
    scorer._weighted_metrics = np.ones((4, 3))
    scorer._aggregate_scores()
    assert fixed_ep == .4 and scorer._weighted_metrics[0, 1] == .2


def test_progress_low_distance_branch_retains_half_collision(native):
    scorer, _ = make_scorer(native)
    scorer._progress_raw[:] = [4, 2]
    scorer._multi_metrics[:, :] = [[1., .5], [1., 1.]]
    scorer._weighted_metrics[:] = 1
    scores = scorer._aggregate_scores()
    assert scorer._weighted_metrics[0, 1] == 1 and scores[1] == .5


def spec_and_request(tmp_path):
    spec = c.ConsequenceSpec(str(NAVSIM_ROOT), str(tmp_path), (str(tmp_path),), "campaign", "a" * 64)
    tr = np.zeros((8, 3))
    request = c.ConsequenceRequest("campaign", "scene", "log", "candidate", "p1", "b" * 64,
        "c" * 64, c.trajectory_sha256(tr), "d" * 64, "a" * 64,
        str(tmp_path / "log/unknown/scene/metric_cache.pkl"), tr)
    return spec, request


@pytest.mark.parametrize("change,status", [
    ({"trajectory": np.zeros((7, 3))}, "INVALID_PREDICTION"),
    ({"trajectory": np.full((8, 3), np.nan)}, "INVALID_PREDICTION"),
    ({"trajectory": np.full((8, 3), 180.)}, "INVALID_PREDICTION"),
    ({"policy_version": "p0"}, "CONTRACT_MISMATCH"),
    ({"score_protocol_hash": "e" * 64}, "CONTRACT_MISMATCH"),
    ({"trajectory_sha256": "f" * 64}, "CONTRACT_MISMATCH"),
    ({"generator_checkpoint_sha256": None}, "CONTRACT_MISMATCH"),
    ({"generator_kind": "engineering_control"}, "CONTRACT_MISMATCH"),
])
def test_rejected_requests_never_reward_zero(tmp_path, change, status):
    spec, request = spec_and_request(tmp_path)
    engine = object.__new__(c.ConsequenceEngine)
    engine.spec, engine._pid = spec, os.getpid()
    result = engine.score(replace(request, **change), "p1")
    assert result["status"] == status and not result["valid"]
    assert result["metrics"] is None and not any(result["metrics_valid"].values())
    assert result["cost"]["scoring_calls"] == 0


def test_missing_corrupt_and_wrong_scene_context_rejected(tmp_path):
    spec, request = spec_and_request(tmp_path)
    engine = object.__new__(c.ConsequenceEngine)
    engine.spec, engine._pid, engine._cache_key = spec, os.getpid(), None
    assert engine.score(request, "p1")["status"] == "MISSING_OR_INCOMPATIBLE_CONTEXT"
    path = Path(request.metric_cache_path)
    path.parent.mkdir(parents=True)
    path.write_bytes(b"not a metric cache")
    assert engine.score(request, "p1")["status"] == "CONTRACT_MISMATCH"
    assert engine.score(replace(request, source_log="other"), "p1")["status"] == "CONTRACT_MISMATCH"
    assert engine.score(replace(request, metric_cache_path="/not/approved/cache.pkl"), "p1")["status"] == "CONTRACT_MISMATCH"


def test_transient_timeout_is_recoverable_not_valid_zero(tmp_path, monkeypatch):
    spec, request = spec_and_request(tmp_path)
    engine = object.__new__(c.ConsequenceEngine)
    engine.spec, engine._pid = spec, os.getpid()
    def failed(_):
        raise TimeoutError("fixture temporary IO")
    monkeypatch.setattr(engine, "load_context", failed)
    result = engine.score(request, "p1")
    assert result["status"] == "TRANSIENT_SCORING_FAILURE"
    assert result["metrics"] is None and result["error"]["retryable"]


def test_changed_pinned_source_rejected(tmp_path):
    path = tmp_path / "navsim/evaluate/pdm_score.py"
    path.parent.mkdir(parents=True)
    path.write_text("# wrong source\n")
    with pytest.raises(c.ContractMismatch, match="Pinned canonical source"):
        c.protocol_identity(str(tmp_path))


def test_cache_rejects_missing_future_observation(native):
    from navsim.planning.metric_caching.metric_cache import MetricCache
    scorer, obs = make_scorer(native)
    cache = object.__new__(MetricCache)
    cache.file_path, cache.trajectory, cache.ego_state = None, None, None
    cache.observation, cache.centerline = obs, scorer._centerline
    cache.route_lane_ids, cache.drivable_area_map = ["lane"], scorer._drivable_area_map
    obs._occupancy_maps = obs._occupancy_maps[:49]
    with pytest.raises(c.IncompatibleContext, match="index49"):
        c.validate_cache(cache)


def test_worker_failures_preserve_denominator_and_bounded_budget(tmp_path, native):
    spec, request = spec_and_request(tmp_path)
    protocol = c.protocol_identity(str(NAVSIM_ROOT))
    spec = replace(spec, score_protocol_hash=protocol["sha256"])
    request = replace(request, score_protocol_hash=protocol["sha256"], trajectory=np.zeros((1, 3)))
    with c.ConsequenceService(spec, workers=2, max_calls=3) as service:
        rows = service.score([request, request], expected_policy_version="p1", require_valid=False)
        assert len(rows) == 2 and all(r["status"] == "INVALID_PREDICTION" for r in rows)
        with pytest.raises(c.ConsequenceBatchError) as error:
            service.score([request], expected_policy_version="p1")
        assert len(error.value.records) == 1
        with pytest.raises(RuntimeError, match="budget exhausted"):
            service.score([request], expected_policy_version="p1")


def test_worker_count_and_queue_bounds(tmp_path):
    spec, _ = spec_and_request(tmp_path)
    with pytest.raises(ValueError):
        c.ConsequenceService(spec, workers=5, max_calls=1)
    with pytest.raises(ValueError):
        c.ConsequenceService(spec, workers=2, max_calls=1, max_inflight=100)
