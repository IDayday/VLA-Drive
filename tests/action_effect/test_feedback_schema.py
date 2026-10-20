"""CPU contract tests use marked synthetic records, not real integration claims."""
from dataclasses import replace
import json

import numpy as np
import pytest

from starVLA.model.modules.st_world.consequence import (
    METRICS, SCHEMA, STATE_COLUMNS, ConsequenceRequest, ContractMismatch,
    _reference_evidence, collision_summary, file_sha256, road_summary, trajectory_sha256, ttc_summary,
)
from tools.action_effect.feedback_schema import (
    canonical_record, content_sha256, from_jsonable, label_arrays, to_jsonable,
    validate_feedback, validate_pair,
)
from tools.action_effect.navsim_feedback_adapter import from_canonical_record, make_effect_pair, summary
from tools.action_effect.validate_feedback import main


@pytest.fixture
def record_factory(tmp_path):
    receipt = tmp_path / "FIXTURE_ONLY_NOT_REAL_SIMULATION.json"
    receipt.write_text(json.dumps({"real_simulation": False, "status": "TEST_FIXTURE"}))
    pin = {"path": str(receipt.resolve()), "sha256": file_sha256(receipt)}
    def make(*, candidate="base", lateral=0., progress=4., collision=False, road=False, ttc=False,
             observation_sha256="9" * 64):
        poses = np.zeros((8, 3), dtype=np.float64)
        poses[:, 0] = np.arange(1, 9) * .5
        poses[:, 1] = lateral
        request = ConsequenceRequest("test_campaign", "scene_1", "log_1", candidate, "base_version",
            "4" * 64, "5" * 64, trajectory_sha256(poses), "6" * 64, "7" * 64,
            "/test_only/log_1/unknown/scene_1/metric_cache.pkl", poses.tolist())
        states = np.zeros((2, 41, 11), dtype=np.float64)
        states[:, :, 0] = np.arange(41) * .1
        states[1, :, 1] = lateral
        states[:, :, 3] = 1.
        reference = np.zeros((41, 11), dtype=np.float64)
        flags = np.zeros(41, dtype=np.bool_)
        flags[10] = road
        nc_events = ([{"state_index": 5, "state_time_seconds": .5, "at_fault": True,
                       "nc_contribution": 0., "object_token": "fixture_car"}] if collision else [])
        ttc_events = ([{"state_index": 8, "state_time_seconds": .8, "violation": True,
                        "projection_offset_steps": 3, "observation_index": 11,
                        "observation_time_seconds": 1.1, "object_token": "fixture_car"}] if ttc else [])
        consequences = {"road": road_summary(flags), "collision_events": nc_events,
            "collision_summary": collision_summary(nc_events), "ttc_events": ttc_events,
            "ttc_summary": ttc_summary(ttc_events)}
        multi = np.array([[1., not collision], [1., not road]], dtype=np.float64)
        raw = np.array([10., progress], dtype=np.float64)
        effective = raw * multi.prod(axis=0)
        denominator = float(effective.max())
        ep = effective / denominator
        weighted = np.array([ep, [1., not ttc], [1., 1.], [1., 1.]], dtype=np.float64)
        weights = np.array([5., 5., 2., 0.])
        scores = multi.prod(axis=0) * ((weighted * weights[..., None]).sum(axis=0) / weights.sum())
        metrics = dict(zip(METRICS, [float(scores[1]), float(multi[0, 1]), float(multi[1, 1]),
            float(ep[1]), float(weighted[1, 1]), 1., 1.]))
        record = {"schema": SCHEMA, "status": "VALID_SCORED", "valid": True, "identity": request.identity(),
            "metrics": metrics, "metrics_valid": {k: True for k in METRICS}, "score_units": "0..1",
            "requested_trajectory": poses.copy(), "executed_states": states[1].copy(),
            "state_columns": STATE_COLUMNS, "executed_state_times_seconds": np.arange(41) * .1,
            "executed_body_coordinates": np.zeros((41, 5, 2), dtype=np.float64),
            "canonical_arrays": {"multi_metrics": multi, "weighted_metrics": weighted, "progress_raw": raw},
            "consequences": consequences,
            "progress": {"raw_meters": float(raw[1]), "reference_raw_meters": float(raw[0]),
                "effective_candidate_meters": float(effective[1]), "effective_reference_meters": float(effective[0]),
                "ep_denominator_meters": denominator, "ep_branch": "normalize"},
            "official_first_at_fault_time_seconds": consequences["collision_summary"]["first_at_fault_time_seconds"],
            "official_first_ttc_state_time_seconds": consequences["ttc_summary"]["first_state_time_seconds"],
            "context_hash": "3" * 64, "context_unchanged": True,
            **_reference_evidence(request, "3" * 64, reference, states, {"dependencies": {"fixture": "explicitly_synthetic"}})}
        wrapped = from_canonical_record(record, request, observation_id="observation_1",
                                        observation_sha256=observation_sha256, canonical_receipt=pin)
        return record, request, wrapped
    return make


def reidentify(feedback):
    feedback["identity"] = content_sha256({k: v for k, v in feedback.items() if k != "identity"})
    return feedback


def test_lossless_json_roundtrip_all_array_bits(record_factory):
    _, _, feedback = record_factory()
    restored = json.loads(json.dumps(feedback, allow_nan=False))
    validate_feedback(restored)
    original = canonical_record(feedback)
    after = canonical_record(restored)
    for key in ("executed_states", "reference_interpolated_states", "reference_executed_states", "requested_trajectory"):
        assert after[key].dtype == original[key].dtype
        assert after[key].shape == original[key].shape
        assert after[key].tobytes() == original[key].tobytes()


def test_first_events_absent_are_masked_not_zero_seconds(record_factory):
    _, _, feedback = record_factory()
    labels = label_arrays(feedback)
    assert labels["valid"][:8].all()
    assert not labels["valid"][8:].any()
    assert summary(feedback)["labels"]["first_at_fault_time_seconds"]["value"] is None
    assert labels["temporal"]["continuous_ttc_seconds"] is None
    assert labels["temporal"]["continuous_risk_valid"] is False


def test_native_event_times_and_motion_keep_provenance(record_factory):
    _, _, feedback = record_factory(collision=True, road=True, ttc=True)
    labels = label_arrays(feedback)
    assert labels["values"][-3:].tolist() == [.5, .8, 1.]
    assert labels["valid"].all()
    assert labels["temporal"]["at_fault_contact"][5]
    assert labels["temporal"]["road_violation"][10]
    assert labels["temporal"]["ttc_violation"][8]
    assert "not_credit" in labels["temporal"]["event_provenance"]
    assert labels["temporal"]["motion_provenance"] == "derived_from_official_executed_states"


def test_pair_delta_sign_masks_event_repair_and_real_distance(record_factory):
    _, _, before = record_factory(collision=True, road=True, ttc=True)
    _, _, after = record_factory(candidate="repaired", progress=9., lateral=.25)
    pair = make_effect_pair(before, after, basis_id="smooth_lateral_v1", coefficients=[.25],
        active_interval_seconds=(0., 4.), geometric_distance_m=.25, executable=True)
    validate_pair(json.loads(json.dumps(pair)))
    delta = from_jsonable(pair["delta_labels"])
    assert delta["values"][0] > 0
    assert delta["values"][1:3].tolist() == [1., 1.]
    assert not delta["valid"][-3:].any()
    assert all(v["repaired"] for v in pair["event_transition"].values())
    assert from_jsonable(pair["delta_temporal"])["event_delta"].sum() == -3


@pytest.mark.parametrize("field,value", [("observation_sha256", "a" * 64), ("scene_id", "foreign"),
    ("numeric_runtime_hash", "b" * 64), ("score_protocol_hash", "c" * 64),
    ("reference_input_sha256", "d" * 64), ("context_hash", "e" * 64)])
def test_changed_identity_rejected(record_factory, field, value):
    _, _, feedback = record_factory()
    _, _, other = record_factory(candidate="other", lateral=.1, observation_sha256=value if field == "observation_sha256" else "9" * 64)
    if field != "observation_sha256":
        other[field] = value
        reidentify(other)
    with pytest.raises(ContractMismatch):
        make_effect_pair(feedback, other, basis_id="test", coefficients=[.1],
                         active_interval_seconds=(0., 4.), geometric_distance_m=.1, executable=True)


def test_v2_explicitly_rejected(record_factory):
    _, _, feedback = record_factory()
    feedback["navsim_version"] = "v2"
    reidentify(feedback)
    with pytest.raises(ContractMismatch, match="separate verified adapter"):
        validate_feedback(feedback)


def test_mutated_original_receipt_rejected(record_factory):
    _, _, feedback = record_factory()
    from pathlib import Path
    Path(feedback["canonical_receipt"]["path"]).write_text("changed")
    with pytest.raises(ContractMismatch, match="receipt missing or checksum changed"):
        validate_feedback(feedback)


def test_mutated_record_cannot_be_relabelled_by_wrapper_hash(record_factory):
    _, _, feedback = record_factory()
    feedback["canonical_record"]["metrics"]["score"] = .99
    feedback["canonical_record_sha256"] = content_sha256(feedback["canonical_record"])
    reidentify(feedback)
    with pytest.raises(ContractMismatch, match="reconstructed"):
        validate_feedback(feedback)


@pytest.mark.parametrize("field,value", [("score_units", "0..100"),
    ("state_columns", ["wrong"] * 11), ("executed_body_coordinates", np.zeros((41, 4, 2), dtype=np.float64))])
def test_units_state_schema_and_full_body_required(record_factory, field, value):
    _, _, feedback = record_factory()
    record = canonical_record(feedback)
    record[field] = value
    feedback["canonical_record"] = to_jsonable(record)
    feedback["canonical_record_sha256"] = content_sha256(record)
    reidentify(feedback)
    with pytest.raises(ContractMismatch):
        validate_feedback(feedback)


def test_event_time_not_trajectory_point_credit(record_factory):
    _, _, feedback = record_factory()
    feedback["event_credit_semantics"] = "credit[t]=-collision[t]"
    reidentify(feedback)
    with pytest.raises(ContractMismatch, match="causal credit"):
        validate_feedback(feedback)


def test_wrong_event_time_cannot_be_rehashed_away(record_factory):
    _, _, feedback = record_factory(collision=True)
    feedback["canonical_record"]["consequences"]["collision_events"][0]["state_time_seconds"] = .6
    feedback["canonical_record_sha256"] = content_sha256(feedback["canonical_record"])
    reidentify(feedback)
    with pytest.raises(ContractMismatch, match="timestamp"):
        validate_feedback(feedback)


def test_invalid_prediction_preserves_failure_and_nonfinite_evidence(record_factory):
    _, request, _ = record_factory()
    request = replace(request, trajectory=[[float("nan"), 0., 0.]] * 8)
    failure = {"schema": SCHEMA, "status": "INVALID_PREDICTION", "valid": False,
        "identity": request.identity(), "metrics": None, "metrics_valid": {k: False for k in METRICS},
        "executed_states": None, "consequences": None,
        "error": {"type": "InvalidPrediction", "message": "nonfinite", "retryable": False}}
    _, _, valid = record_factory()
    feedback = from_canonical_record(failure, request, observation_id="observation_1", observation_sha256="9" * 64,
                                     canonical_receipt=valid["canonical_receipt"])
    encoded = json.loads(json.dumps(feedback, allow_nan=False))
    assert encoded["status"] == "INVALID_PREDICTION"
    assert np.isnan(from_jsonable(encoded["request"])["trajectory"][0][0])
    assert not label_arrays(encoded)["valid"].any()
    with pytest.raises(ContractMismatch, match="successfully simulated"):
        make_effect_pair(feedback, valid, basis_id="invalid", coefficients=[0.],
                         active_interval_seconds=(0., 4.), geometric_distance_m=0., executable=False)


def test_zero_intervention_pair_is_exact_no_effect(record_factory):
    _, _, before = record_factory()
    pair = make_effect_pair(before, before, basis_id="identity", coefficients=[0.],
        active_interval_seconds=(0., 0.), geometric_distance_m=0., executable=True)
    assert not from_jsonable(pair["delta_labels"])["values"].any()
    assert not from_jsonable(pair["delta_temporal"])["event_delta"].any()


def test_bad_distance_and_effect_sign_rejected(record_factory):
    _, _, before = record_factory()
    _, _, after = record_factory(candidate="after", lateral=.25, progress=9.)
    with pytest.raises(ContractMismatch, match="physical intervention distance"):
        make_effect_pair(before, after, basis_id="test", coefficients=[.25],
                         active_interval_seconds=(0., 4.), geometric_distance_m=25., executable=True)
    pair = make_effect_pair(before, after, basis_id="test", coefficients=[.25],
                           active_interval_seconds=(0., 4.), geometric_distance_m=.25, executable=True)
    delta = from_jsonable(pair["delta_labels"])
    delta["values"] *= -1
    pair["delta_labels"] = to_jsonable(delta)
    reidentify(pair)
    with pytest.raises(ContractMismatch, match="delta/sign"):
        validate_pair(pair)


def test_json_validator_exit_codes_and_budget(record_factory, tmp_path, capsys):
    _, _, feedback = record_factory()
    artifact = tmp_path / "feedback.json"
    artifact.write_text(json.dumps(feedback))
    assert main(["--input", str(artifact)]) == 0
    assert json.loads(capsys.readouterr().out)["simulation_calls"] == 0
    assert main(["--input", str(artifact), "--max-bytes", "1"]) == 1
    assert json.loads(capsys.readouterr().out)["status"] == "FAIL"


def test_wrapper_does_not_mutate_native_record(record_factory):
    record, _, feedback = record_factory()
    digest = content_sha256(record)
    restored = canonical_record(feedback)
    restored["executed_states"][0, 0] = 999.
    assert content_sha256(record) == digest
    assert canonical_record(feedback)["executed_states"][0, 0] != 999.


def test_typed_array_roundtrip_preserves_negative_zero_and_nan_evidence():
    array = np.array([-0., 0., np.nan, np.inf], dtype=np.float64)
    restored = from_jsonable(json.loads(json.dumps(to_jsonable(array), allow_nan=False)))
    assert restored.tobytes() == array.tobytes()
