"""Thin wrapping of checksum-verified canonical feedback; no scorer mutation."""
from __future__ import annotations

from dataclasses import asdict
from typing import Any, Mapping

import numpy as np

from starVLA.model.modules.st_world.consequence import (
    PROTOCOL, STATE_COLUMNS, ConsequenceRequest,
)
from tools.action_effect.feedback_schema import (
    FEEDBACK_SCHEMA, NAVSIM_VERSION, PAIR_SCHEMA, canonical_record,
    content_sha256, label_arrays, temporal_differences, to_jsonable, validate_feedback, validate_pair,
)


def from_canonical_record(record: Mapping[str, Any], request: ConsequenceRequest, *,
                          observation_id: str, observation_sha256: str,
                          canonical_receipt: Mapping[str, str]) -> dict:
    """Wrap an original ledger-verified record and pin its independent receipt.

    Callers load records through OfficialFeedbackLedger or equivalently verify
    its COMPLETE -> result-file checksum -> request identity chain. This thin
    adapter neither unpickles that file again nor calls the simulator. Valid
    records require observe=True; a legacy terminal-only record is not silently
    upgraded. Failed records remain failed and cannot enter an EffectPair.
    """
    valid = record.get("valid") is True
    feedback = {"schema": FEEDBACK_SCHEMA, "navsim_version": NAVSIM_VERSION,
        "protocol": PROTOCOL, "scene_id": request.scene_id, "source_log": request.source_log,
        "observation_id": observation_id, "observation_sha256": observation_sha256,
        "request": to_jsonable(asdict(request)), "valid": valid, "status": record.get("status"),
        "error": record.get("error"), "canonical_record": to_jsonable(record),
        "canonical_record_sha256": content_sha256(record), "canonical_receipt": dict(canonical_receipt),
        "reference_input_sha256": record["reference_contract"]["input_sha256"] if valid else None,
        "context_hash": record.get("context_hash") if valid else None,
        "numeric_runtime_hash": request.inference_runtime_hash, "score_protocol_hash": request.score_protocol_hash,
        "submitted_trajectory": None, "executed_trajectory": None,
        "field_provenance": {
            "terminal_metrics": "official_pdm_score_unmodified",
            "executed_states": "official_LQR_plus_kinematic_bicycle_global_rear_axle",
            "collision_road_ttc_events": "read_only_exact_replay_of_native_predicates_asserted_against_official",
            "motion": "derived_from_native_executed_states",
            "continuous_risk": "unavailable_not_fabricated",
            "canonical_receipt": "external_file_checksum_verified; record association pinned by wrapper; caller verifies ledger result chain",
            "causal_credit": "not_assigned_by_event_timestamp"},
        "event_credit_semantics": "observed_event_time_not_causal_credit"}
    if valid:
        feedback["submitted_trajectory"] = to_jsonable({
            "poses": np.asarray(record["requested_trajectory"]).copy(),
            "times_seconds": np.arange(1, 9, dtype=np.float64) * .5,
            "coordinate_frame": "initial_rear_axle_local", "point_reference": "rear_axle", "yaw_units": "radian"})
        feedback["executed_trajectory"] = to_jsonable({
            "states": np.asarray(record["executed_states"]).copy(),
            "times_seconds": np.asarray(record["executed_state_times_seconds"]).copy(),
            "coordinate_frame": "global_map", "point_reference": "rear_axle", "state_columns": list(STATE_COLUMNS)})
    feedback["identity"] = content_sha256(feedback)
    validate_feedback(feedback)
    return feedback


def make_effect_pair(base_feedback: Mapping[str, Any], modified_feedback: Mapping[str, Any], *,
                     basis_id: str, coefficients, active_interval_seconds,
                     geometric_distance_m: float, executable: bool,
                     intervention_metadata: Mapping[str, Any] | None = None) -> dict:
    """Construct signed modified-minus-base labels for a real paired intervention."""
    left, right = label_arrays(base_feedback), label_arrays(modified_feedback)
    common = left["valid"] & right["valid"]
    if not base_feedback["valid"] or not modified_feedback["valid"]:
        from starVLA.model.modules.st_world.consequence import ContractMismatch
        raise ContractMismatch("Effect differences require two successfully simulated trajectories")
    a, b = canonical_record(base_feedback)["consequences"], canonical_record(modified_feedback)["consequences"]
    def transitions(before, after):
        return {"repaired": bool(before and not after), "introduced": bool(not before and after),
                "unchanged": bool(before == after), "base_event": bool(before), "modified_event": bool(after)}
    pair = {"schema": PAIR_SCHEMA, "base_feedback": dict(base_feedback), "modified_feedback": dict(modified_feedback),
        "intervention": to_jsonable({"basis_id": basis_id, "coefficients": np.asarray(coefficients, dtype=np.float64),
            "active_interval_seconds": list(active_interval_seconds), "geometric_distance_m": geometric_distance_m,
            "executable": executable, "metadata": {**dict(intervention_metadata or {}),
                "distance_definition": "mean_pointwise_xy_l2_m"}}),
        "delta_labels": to_jsonable({"names": left["names"], "units": left["units"], "valid": common,
            "values": np.where(common, right["values"] - left["values"], 0.),
            "definition": "modified_minus_base; event times valid only when present in both"}),
        "delta_temporal": to_jsonable(temporal_differences(left["temporal"], right["temporal"])),
        "event_transition": {"collision": transitions(a["collision_summary"]["event"], b["collision_summary"]["event"]),
            "road": transitions(a["road"]["event"], b["road"]["event"]),
            "ttc": transitions(a["ttc_summary"]["event"], b["ttc_summary"]["event"])},
        "causal_scope": "controlled_same_context_intervention_local_effect_not_global_gradient"}
    pair["identity"] = content_sha256(pair)
    validate_pair(pair)
    return pair


def summary(feedback: Mapping[str, Any]) -> dict:
    labels = label_arrays(feedback)
    return {"schema": feedback["schema"], "identity": feedback["identity"], "valid": feedback["valid"],
        "scene_id": feedback["scene_id"], "source_log": feedback["source_log"],
        "labels": {name: {"value": float(value) if valid else None, "valid": bool(valid), "unit": unit,
                          "provenance": provenance}
                   for name, value, valid, unit, provenance in zip(labels["names"], labels["values"], labels["valid"],
                                                                  labels["units"], labels["provenance"])},
        "continuous_risk": "NOT_AVAILABLE", "causal_credit": "NOT_ASSIGNED"}
