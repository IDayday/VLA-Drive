"""G0 fault tests; synthetic feedback fixtures, never real repair evidence."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np
import pytest

from tools.action_effect.common import artifact, digest, read_json, write_json
from tools.action_effect import simulate_interventions as g0
from starVLA.model.modules.st_world.consequence import _reference_evidence, trajectory_sha256


@pytest.fixture
def factory(tmp_path):
    path = Path(__file__).with_name("test_feedback_schema.py")
    spec = importlib.util.spec_from_file_location("_g0_marked_fixture", path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module.record_factory.__wrapped__(tmp_path)


def test_ledger_reserves_before_work_and_reuses_no_slot(tmp_path):
    ledger = g0.CallLedger(tmp_path / "ledger", identity="run", max_calls=2, retry_budget=1)
    attempt = tmp_path / "candidate" / "attempt_00"
    assert ledger.admit(attempt, request_identity={"id": 1}, retry=False)
    assert not ledger.admit(attempt, request_identity={"id": 1}, retry=False)
    assert read_json(tmp_path / "ledger" / "COUNTER.json")["reserved_calls"] == 1
    with pytest.raises(ValueError, match="Foreign"):
        ledger.admit(attempt, request_identity={"id": 2}, retry=False)
    assert ledger.admit(attempt.parent / "attempt_01", request_identity={"id": 1}, retry=True)
    with pytest.raises(g0.BudgetExhausted):
        ledger.admit(tmp_path / "third", request_identity={}, retry=False)


def test_crash_after_counter_before_admission_never_undercounts(tmp_path):
    ledger = g0.CallLedger(tmp_path / "ledger", identity="run", max_calls=2, retry_budget=1)
    with pytest.raises(RuntimeError, match="Injected"):
        ledger.admit(tmp_path / "unused", request_identity={"id": 1}, retry=False, _fault_after_reservation=True)
    assert not (tmp_path / "unused" / "ADMITTED.json").exists()
    fresh = g0.CallLedger(tmp_path / "ledger", identity="run", max_calls=2, retry_budget=1)
    assert fresh.admit(tmp_path / "actual", request_identity={"id": 1}, retry=False)
    assert read_json(tmp_path / "ledger" / "COUNTER.json")["reserved_calls"] == 2
    with pytest.raises(g0.BudgetExhausted): fresh.admit(tmp_path / "extra", request_identity={}, retry=False)


def test_concurrent_admissions_respect_global_limit(tmp_path):
    ledger = g0.CallLedger(tmp_path / "ledger", identity="run", max_calls=12, retry_budget=0)
    def admit(i):
        try: return ledger.admit(tmp_path / str(i), request_identity={"id": i}, retry=False)
        except g0.BudgetExhausted: return False
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(admit, range(24)))
    assert sum(results) == 12
    assert read_json(tmp_path / "ledger" / "COUNTER.json") == {"reserved_calls": 12, "reserved_retries": 0}


def prepared_record(factory, request):
    record, _, _ = factory(candidate=request.candidate_id, lateral=0., progress=4. if request.candidate_id == "intervention_00" else 9.)
    poses = np.asarray(request.trajectory, dtype=np.float64)
    states = record["executed_states"].copy()
    for coordinate in range(3):
        states[:, coordinate] = np.interp(np.arange(41)*.1, np.arange(9)*.5, np.r_[0., poses[:, coordinate]])
    reference = record["reference_interpolated_states"]
    pair = np.stack((record["reference_executed_states"], states))
    record.update(identity=request.identity(), requested_trajectory=poses.copy(), executed_states=states,
        **_reference_evidence(request, record["context_hash"], reference, pair,
                              {"dependencies": record["reference_identity"]["software_identity"]}))
    return record


def test_completed_content_cache_reused_across_population_without_score(factory, tmp_path):
    _, request, _ = factory()
    class Engine:
        calls = 0
        def score(self, req, policy):
            self.calls += 1
            return prepared_record(factory, req)
    engine = Engine(); cache = tmp_path / "global_cache" / "key"
    first = g0.CallLedger(tmp_path / "pilot_calls", identity="pilot", max_calls=1, retry_budget=0)
    result, wrapped, path = g0.score_candidate(engine, first, cache, request,
        observation={"observation_hash": "9"*64}, deadline=time.monotonic()+60)
    assert engine.calls == 1 and result["valid"] and wrapped["valid"]
    second = g0.CallLedger(tmp_path / "expanded_calls", identity="expanded", max_calls=1, retry_budget=0)
    again, _, _ = g0.score_candidate(engine, second, cache, request,
        observation={"observation_hash": "9"*64}, deadline=time.monotonic()+60)
    assert engine.calls == 1 and again["valid"]
    assert read_json(tmp_path / "expanded_calls" / "COUNTER.json")["reserved_calls"] == 0
    (path / "CANONICAL.json.gz").write_bytes(b"broken")
    with pytest.raises(ValueError, match="stale"):
        g0.score_candidate(engine, second, cache, request, observation={"observation_hash": "9"*64}, deadline=time.monotonic()+60)
    assert engine.calls == 1


def test_incomplete_admission_requires_distinct_budgeted_retry(factory, tmp_path):
    _, request, _ = factory()
    ledger = g0.CallLedger(tmp_path / "calls", identity="run", max_calls=2, retry_budget=1)
    cache = tmp_path / "candidate"
    ledger.admit(cache / "attempt_00", request_identity=request.identity(), retry=False)
    class Engine:
        calls = 0
        def score(self, req, policy): self.calls += 1; return prepared_record(factory, req)
    engine = Engine()
    _, _, path = g0.score_candidate(engine, ledger, cache, request, observation={"observation_hash": "9"*64}, deadline=time.monotonic()+60)
    assert path.name == "attempt_01" and engine.calls == 1
    assert not (cache / "attempt_00" / "COMPLETE.json").exists()
    assert read_json(tmp_path / "calls" / "COUNTER.json") == {"reserved_calls": 2, "reserved_retries": 1}


def test_transient_retry_reuses_prediction_and_never_fills_zero(factory, tmp_path):
    _, request, _ = factory()
    ledger = g0.CallLedger(tmp_path / "calls", identity="run", max_calls=2, retry_budget=1)
    class Engine:
        calls = 0
        def score(self, req, policy):
            self.calls += 1
            if self.calls == 1:
                return {"schema": "st_world_canonical_consequence_v1",
                    "identity": req.identity(), "valid": False, "status": "TRANSIENT_SCORING_FAILURE", "metrics": None,
                    "metrics_valid": {}, "error": {"retryable": True, "type": "TimeoutError", "message": "fixture timeout"}}
            return prepared_record(factory, req)
    engine = Engine()
    result, _, _ = g0.score_candidate(engine, ledger, tmp_path / "cache", request,
        observation={"observation_hash": "9"*64}, deadline=time.monotonic()+60)
    assert result["valid"] and engine.calls == 2
    first = g0.from_jsonable(g0.read_gzip(artifact(tmp_path / "cache" / "attempt_00" / "CANONICAL.json.gz")))
    assert first["metrics"] is None and not first["valid"]


def test_repair_selection_lexicographic_safety_progress_comfort_not_score_only():
    base = {"status": "VALID_SCORED", "metrics": {"score": .2, "no_at_fault_collisions": 0.,
        "drivable_area_compliance": 1., "time_to_collision_within_bound": 1., "comfort": 1.},
        "progress": {"raw_meters": 10.}}
    candidate = {"valid": True, "metrics": {"score": 1., "no_at_fault_collisions": 1.,
        "drivable_area_compliance": 1., "time_to_collision_within_bound": 1., "comfort": 1.},
        "progress": {"raw_meters": 8.}}
    assert not g0.meets_repair_constraints(base, candidate, {"controller_tracking_pass": True}, {"geometrically_admissible": True})
    candidate["progress"]["raw_meters"] = 9.
    assert g0.meets_repair_constraints(base, candidate, {"controller_tracking_pass": True}, {"geometrically_admissible": True})
    candidate["metrics"]["comfort"] = 0.
    assert not g0.meets_repair_constraints(base, candidate, {"controller_tracking_pass": True}, {"geometrically_admissible": True})


def test_support_candidate_is_not_a_local_repair():
    base = {"candidate_id": "base", "status": "VALID_SCORED", "metrics": {"score": .5,
        "no_at_fault_collisions": 1., "drivable_area_compliance": 1., "time_to_collision_within_bound": 1.}}
    support = {"candidate_id": "extra", "kind": "policy_resample", "eligible_repair": True,
               "distance_m": .1, "metrics": {"score": .99}, "trajectory_sha256": "a"*64}
    assert g0.select_repair([base, support])["status"] == "REQUIRES_LARGER_SUPPORT_SEARCH"
    local = {**support, "candidate_id": "local", "kind": "lateral", "distance_m": .2}
    assert g0.select_repair([base, support, local])["selected"] == "local"


def test_scene_transaction_compact_datasets_and_reuse_no_new_calls(factory, tmp_path, monkeypatch):
    from tools.action_effect.build_effect_pairs import build
    from tools.action_effect.analyze_repair_opportunities import analyze
    root = tmp_path / "population_000001"
    physical = np.column_stack((np.arange(1, 9)*.5, np.zeros(8), np.zeros(8)))
    cached = {"physical": physical, "extra_policy_physical": [physical.copy(), physical.copy()]}
    entry = {"scene_id": "scene_1", "source_log": "log_1", "role": "fit", "observation_hash": "9"*64,
             "shard": {"path": "/explicitly_synthetic_context.pt", "sha256": "a"*64}, "shard_offset": 0}
    observation = {"observation_hash": "9"*64, "metric_cache": {"path": "/fixture/log_1/unknown/scene_1/metric_cache.pkl", "sha256": "6"*64}}
    spec = {"navsim_root": "/fixture", "maps_root": "/fixture", "cache_roots": ["/fixture"],
            "campaign_id": "test_campaign", "score_protocol_hash": "7"*64, "allow_engineering_controls": True}
    reg = {"schema": "action_effect_g0_registration_v1", "spec": spec, "split": "train", "seed": 42,
           "base_checkpoint": {"sha256": "4"*64}, "numeric_runtime_hash": "5"*64,
           "max_calls": 26, "retry_budget": 0, "max_samples": 1, "population": [{"context": entry, "observation": observation}]}
    reg["identity"] = digest(reg); write_json(root / "REGISTRATION.json", reg)
    class Engine:
        calls = 0
        def load_context(self, request):
            return SimpleNamespace(ego_state=SimpleNamespace(dynamic_car_state=SimpleNamespace(
                rear_axle_velocity_2d=SimpleNamespace(array=np.array([1., 0.])))))
        def score(self, request, policy):
            self.calls += 1
            return prepared_record(factory, request)
    engine = Engine()
    monkeypatch.setattr(g0, "worker_engine", lambda spec: engine)
    monkeypatch.setattr(g0, "load_context_record", lambda entry: cached)
    result = g0.process_scene(root, reg, reg["population"][0], time.monotonic()+60)
    assert result["status"] == "COMPLETE", result
    calls = engine.calls
    again = g0.process_scene(root, reg, reg["population"][0], time.monotonic()+60)
    assert again["reused"] and engine.calls == calls
    write_json(root / "COMPLETE.json", {"registration_identity": reg["identity"],
        "scenes": [artifact(root / "scenes" / "scene_1" / "COMPLETE.json")]})
    built = build(root)
    assert built["pairs"] > 0
    pairs = read_json(root / "PAIR_DATASET.json")
    assert pairs["sampling_contract"].startswith("uniform scene")
    assert not any(r["kind"] == "policy_resample" for r in pairs["entries"])
    assert all(len(r["base_labels"]["values"]) == 11 for r in pairs["entries"])
    assert all(len(r["temporal_event_delta"]) == 41 for r in pairs["entries"])
    assert len(read_json(root / "REPAIR_DATASET.json")["entries"]) == 1
    assert analyze(root)["requested_scenes"] == 1
    assert build(root) == built
    first = read_json(root / "scenes" / "scene_1" / "SCENE.json")["candidates"][0]
    receipt = read_json(first["canonical_receipt"]["path"])
    Path(receipt["canonical"]["path"]).write_bytes(b"broken original native record")
    with pytest.raises(ValueError, match="stale"):
        build(root)


def test_partial_population_never_becomes_training_dataset(tmp_path):
    from tools.action_effect.build_effect_pairs import read_population
    root = tmp_path / "partial"
    reg = {"split": "train", "max_samples": 1}; reg["identity"] = digest(reg)
    write_json(root / "REGISTRATION.json", reg)
    with pytest.raises(FileNotFoundError): read_population(root)


def test_invalid_prediction_is_retained_not_retry_or_valid_zero(factory, tmp_path):
    from starVLA.model.modules.st_world.consequence import METRICS
    _, request, _ = factory()
    ledger = g0.CallLedger(tmp_path / "calls", identity="run", max_calls=2, retry_budget=1)
    class Engine:
        calls = 0
        def score(self, req, policy):
            self.calls += 1
            return {"schema": "st_world_canonical_consequence_v1", "identity": req.identity(),
                "valid": False, "status": "INVALID_PREDICTION", "metrics": None,
                "metrics_valid": {key: False for key in METRICS}, "executed_states": None, "consequences": None,
                "error": {"retryable": False, "type": "InvalidPrediction", "message": "fixture contract failure"}}
    engine = Engine()
    result, wrapped, _ = g0.score_candidate(engine, ledger, tmp_path / "candidate", request,
        observation={"observation_hash": "9"*64}, deadline=time.monotonic()+60)
    assert engine.calls == 1 and result["metrics"] is None
    assert wrapped["valid"] is False and wrapped["status"] == "INVALID_PREDICTION"


def test_uncertain_attempt_without_retry_budget_stays_partial(factory, tmp_path):
    _, request, _ = factory()
    ledger = g0.CallLedger(tmp_path / "calls", identity="run", max_calls=1, retry_budget=0)
    candidate = tmp_path / "candidate"
    ledger.admit(candidate / "attempt_00", request_identity=request.identity(), retry=False)
    class Engine:
        def score(self, *_): raise AssertionError("Must not score without an admitted retry")
    with pytest.raises(g0.BudgetExhausted):
        g0.score_candidate(Engine(), ledger, candidate, request, observation={"observation_hash": "9"*64}, deadline=time.monotonic()+60)
    assert not (candidate / "attempt_00" / "COMPLETE.json").exists()


def test_actual_context_cpu_tensor_boundary_preserves_base_float_values():
    import torch
    physical = torch.zeros((8, 3), dtype=torch.float32)
    physical[:, 0] = torch.arange(1, 9)*2.5
    proposals = g0.candidate_proposals({"physical": physical, "extra_policy_physical": [physical.clone(), physical.clone()]}, seed=42)
    assert len(proposals) == 26
    assert proposals[0]["physical"].dtype == np.float64
    assert proposals[0]["physical"].tobytes() == physical.numpy().astype(np.float64).tobytes()


def test_old_complete_markers_do_not_override_current_validation_failure(tmp_path, monkeypatch):
    # This test exercises receipt rejection, under the real CPU worker launch
    # contract, independently of the invoking shell's CUDA/BLAS environment.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        monkeypatch.setenv(name, "1")
    root = tmp_path / "population"
    reg = {"identity": "fixture_run", "population": [{"context": {"scene_id": "one"}}, {"context": {"scene_id": "two"}}]}
    for token in ("one", "two"):
        write_json(root / "scenes" / token / "COMPLETE.json", {"old": "marker_is_not_current_validation"})
    g0.CallLedger(root / "calls", identity=reg["identity"], max_calls=2, retry_budget=0)
    monkeypatch.setattr(g0, "prepare", lambda *args, **kwargs: (root, reg))
    monkeypatch.setattr(g0, "ProcessPoolExecutor", lambda max_workers, **kwargs: ThreadPoolExecutor(max_workers=max_workers))
    monkeypatch.setattr(g0, "process_scene", lambda *args: {"scene_id": args[2]["context"]["scene_id"],
        "status": "PARTIAL", "error": "fixture checksum mismatch"})
    result = g0.run({}, max_samples=2, max_simulation_budget=2, workers=2, resume=True)
    assert result["status"] == "PARTIAL"
    assert not (root / "COMPLETE.json").exists()
    assert result["cost"]["invocation_wall_seconds"] >= 0
