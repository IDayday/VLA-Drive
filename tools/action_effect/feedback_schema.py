"""Versioned, lossless feedback and same-scene intervention-pair contracts.

Only the already audited NAVSIM v1.1 canonical observer is supported. Event
times are observations, not causal credit. A pair records an actual controlled
intervention; it does not turn an event time into responsibility attribution.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from starVLA.model.modules.st_world.consequence import (
    ContractMismatch, METRICS, PROTOCOL, REFERENCE_CONTRACT, SCHEMA,
    STATE_COLUMNS, ConsequenceRequest, array_sha256, file_sha256,
    fingerprint, trajectory_sha256, validated_reference_input,
)

FEEDBACK_SCHEMA = "action_effect_feedback_v1"
PAIR_SCHEMA = "action_effect_pair_v1"
NAVSIM_VERSION = "v1.1"
_ARRAY_TAG = "__action_effect_ndarray_v1__"
_FLOAT_TAG = "__action_effect_nonfinite_float_v1__"


def require_sha256(value: str, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ContractMismatch(f"{name} must be a lowercase SHA256")
    return value


def to_jsonable(value: Any) -> Any:
    """Losslessly preserve numerical ndarray dtype/shape in a JSON artifact."""
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject or value.dtype.kind not in "biuf":
            raise ContractMismatch("Only finite numerical/bool arrays may enter feedback")
        return {_ARRAY_TAG: True, "dtype": value.dtype.str,
                "shape": list(value.shape), "bytes_hex": np.ascontiguousarray(value).tobytes().hex()}
    if isinstance(value, np.generic):
        return to_jsonable(value.item())
    if isinstance(value, Mapping):
        if any(not isinstance(k, str) for k in value):
            raise ContractMismatch("Feedback mappings require string keys")
        return {k: to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        return {_FLOAT_TAG: "nan" if math.isnan(value) else ("inf" if value > 0 else "-inf")}
    raise ContractMismatch(f"Unsupported/nonfinite feedback value: {type(value).__name__}")


def from_jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        if _ARRAY_TAG in value:
            if set(value) != {_ARRAY_TAG, "dtype", "shape", "bytes_hex"} or value[_ARRAY_TAG] is not True:
                raise ContractMismatch("Invalid typed ndarray encoding")
            dtype = np.dtype(value["dtype"])
            if dtype.hasobject or dtype.kind not in "biuf":
                raise ContractMismatch("Invalid ndarray dtype")
            shape = value["shape"]
            if not isinstance(shape, list) or any(type(v) is not int or v < 0 for v in shape):
                raise ContractMismatch("Invalid ndarray shape")
            raw = bytes.fromhex(value["bytes_hex"])
            if math.prod(shape) * dtype.itemsize != len(raw):
                raise ContractMismatch("Invalid ndarray byte inventory")
            return np.frombuffer(raw, dtype=dtype).reshape(shape).copy()
        if _FLOAT_TAG in value:
            if set(value) != {_FLOAT_TAG} or value[_FLOAT_TAG] not in ("nan", "inf", "-inf"):
                raise ContractMismatch("Invalid nonfinite prediction evidence")
            return float(value[_FLOAT_TAG])
        return {k: from_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [from_jsonable(v) for v in value]
    return value


def content_sha256(value: Any) -> str:
    """Digest full typed content; ndarray bits, shape and dtype are all bound."""
    def identity(v):
        if isinstance(v, np.ndarray):
            return {"ndarray_sha256": array_sha256(v), "shape": list(v.shape), "dtype": v.dtype.str}
        if isinstance(v, Mapping):
            return {str(k): identity(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [identity(x) for x in v]
        if isinstance(v, np.generic):
            return identity(v.item())
        if isinstance(v, float) and not math.isfinite(v):
            return to_jsonable(v)
        return v
    return fingerprint(identity(from_jsonable(value)))


def canonical_record(feedback: Mapping[str, Any]) -> dict:
    """Return a new native representation, without loading a scorer or torch."""
    return from_jsonable(feedback["canonical_record"])


def _receipt(receipt: Mapping[str, Any], verify_file: bool) -> None:
    if not isinstance(receipt, Mapping) or set(receipt) != {"path", "sha256"}:
        raise ContractMismatch("Explicit independent canonical receipt path/sha256 required")
    require_sha256(receipt["sha256"], "canonical_receipt.sha256")
    path = Path(receipt["path"])
    if not path.is_absolute():
        raise ContractMismatch("Canonical receipt must use an absolute path")
    if verify_file and (not path.is_file() or file_sha256(path) != receipt["sha256"]):
        raise ContractMismatch("Canonical receipt missing or checksum changed")


def _finite_array(value: Any, shape: tuple, name: str, dtype="<f8") -> np.ndarray:
    array = np.asarray(value)
    if array.shape != shape or array.dtype.str != dtype or not np.isfinite(array).all():
        raise ContractMismatch(f"{name} requires finite {dtype}{shape}")
    return array


def validate_feedback(feedback: Mapping[str, Any], *, verify_receipt: bool = True) -> Mapping[str, Any]:
    """Validate the wrapper, complete canonical record and original input hashes.

    The external receipt checksum is independently checked; this function does
    not deserialize that receipt's potentially large RESULTS.pt. The caller
    must have obtained ``record`` using the existing checksum-verifying ledger.
    The exact record and request association is pinned by this wrapper.
    """
    required = {"schema", "navsim_version", "protocol", "scene_id", "source_log",
                "observation_id", "observation_sha256", "request", "valid", "status",
                "error", "canonical_record", "canonical_record_sha256", "canonical_receipt",
                "reference_input_sha256", "context_hash", "numeric_runtime_hash",
                "score_protocol_hash", "submitted_trajectory", "executed_trajectory",
                "field_provenance", "event_credit_semantics", "identity"}
    if not isinstance(feedback, Mapping) or set(feedback) != required:
        raise ContractMismatch("Invalid feedback wrapper fields")
    if (feedback["schema"] != FEEDBACK_SCHEMA or feedback["navsim_version"] != NAVSIM_VERSION
            or feedback["protocol"] != PROTOCOL):
        raise ContractMismatch("Unsupported feedback/NAVSIM version; v2 requires a separate verified adapter")
    if feedback["event_credit_semantics"] != "observed_event_time_not_causal_credit":
        raise ContractMismatch("Event timestamps must not masquerade as causal credit")
    if not isinstance(feedback["observation_id"], str) or not feedback["observation_id"]:
        raise ContractMismatch("Explicit legal observation ID required")
    for key in ("observation_sha256", "canonical_record_sha256", "numeric_runtime_hash", "score_protocol_hash"):
        require_sha256(feedback[key], key)
    if feedback["identity"] != content_sha256({k: v for k, v in feedback.items() if k != "identity"}):
        raise ContractMismatch("Feedback content/identity mismatch")
    _receipt(feedback["canonical_receipt"], verify_receipt)
    request = ConsequenceRequest(**from_jsonable(feedback["request"]))
    record = canonical_record(feedback)
    if (record.get("schema") != SCHEMA or content_sha256(record) != feedback["canonical_record_sha256"]
            or record.get("identity") != request.identity()
            or feedback["scene_id"] != request.scene_id or feedback["source_log"] != request.source_log
            or feedback["numeric_runtime_hash"] != request.inference_runtime_hash
            or feedback["score_protocol_hash"] != request.score_protocol_hash
            or record.get("valid") is not feedback["valid"] or record.get("status") != feedback["status"]
            or record.get("error") != feedback["error"]):
        raise ContractMismatch("Canonical feedback/request identity mismatch")
    if not feedback["valid"]:
        if record.get("status") not in {"INVALID_PREDICTION", "MISSING_OR_INCOMPATIBLE_CONTEXT",
                "CONTRACT_MISMATCH", "TRANSIENT_SCORING_FAILURE"} or not isinstance(record.get("error"), Mapping):
            raise ContractMismatch("Unknown failure type cannot be silently converted to a quality label")
        if record.get("metrics") is not None or any(record.get("metrics_valid", {}).values()):
            raise ContractMismatch("Invalid simulation cannot supply valid/zero metrics")
        if (feedback["submitted_trajectory"] is not None or feedback["executed_trajectory"] is not None
                or feedback["reference_input_sha256"] is not None or feedback["context_hash"] is not None):
            raise ContractMismatch("Invalid feedback must not fabricate executed or reference data")
        return feedback
    validated_reference_input(record, request)
    if record.get("score_units") != "0..1" or tuple(record.get("state_columns", ())) != STATE_COLUMNS:
        raise ContractMismatch("Canonical score units/executed state columns changed")
    _finite_array(record["executed_body_coordinates"], (41, 5, 2), "executed full body corners and center")
    if (feedback["reference_input_sha256"] != record["reference_contract"]["input_sha256"]
            or feedback["context_hash"] != record["context_hash"]):
        raise ContractMismatch("Pairable context/reference input identity mismatch")
    if set(record["metrics"]) != set(METRICS) or set(record["metrics_valid"]) != set(METRICS):
        raise ContractMismatch("Incomplete canonical metric schema")
    for name, value in record["metrics"].items():
        if record["metrics_valid"][name] is not True or not math.isfinite(value) or not 0 <= value <= 1:
            raise ContractMismatch(f"Illegal canonical metric/mask {name}")
    multi = _finite_array(record["canonical_arrays"]["multi_metrics"], (2, 2), "native multi metrics")
    weighted = _finite_array(record["canonical_arrays"]["weighted_metrics"], (4, 2), "native weighted metrics")
    raw = _finite_array(record["canonical_arrays"]["progress_raw"], (2,), "native raw progress")
    weights = np.asarray([5., 5., 2., 0.], dtype=np.float64)
    score = multi.prod(axis=0) * ((weighted * weights[..., None]).sum(axis=0) / weights.sum())
    metric_views = {"score": score[1], "no_at_fault_collisions": multi[0, 1],
                    "drivable_area_compliance": multi[1, 1], "ego_progress": weighted[0, 1],
                    "time_to_collision_within_bound": weighted[1, 1], "comfort": weighted[2, 1],
                    "driving_direction_compliance": weighted[3, 1]}
    if any(float(value) != record["metrics"][name] for name, value in metric_views.items()):
        raise ContractMismatch("Official metric/score cannot be reconstructed from saved native arrays")
    effective = raw * multi.prod(axis=0)
    denominator = float(effective.max())
    ep = effective / denominator if denominator > 5. else np.where(multi.prod(axis=0) == 0., 0., 1.)
    progress = {"raw_meters": float(raw[1]), "reference_raw_meters": float(raw[0]),
                "effective_candidate_meters": float(effective[1]), "effective_reference_meters": float(effective[0]),
                "ep_denominator_meters": denominator,
                "ep_branch": "normalize" if denominator > 5. else "unit_with_zero_multiplier_mask"}
    if not np.array_equal(ep, weighted[0]) or record["progress"] != progress:
        raise ContractMismatch("Saved fixed-reference progress/gate/normalization evidence mismatch")
    submitted = from_jsonable(feedback["submitted_trajectory"])
    executed = from_jsonable(feedback["executed_trajectory"])
    if (submitted["coordinate_frame"] != "initial_rear_axle_local" or submitted["yaw_units"] != "radian"
            or submitted["point_reference"] != "rear_axle"
            or not np.array_equal(submitted["times_seconds"], np.arange(1, 9, dtype=np.float64) * .5)
            or trajectory_sha256(_finite_array(submitted["poses"], (8, 3), "submitted poses")) != request.trajectory_sha256
            or not np.array_equal(submitted["poses"], record["requested_trajectory"])):
        raise ContractMismatch("Submitted trajectory contract/content mismatch")
    if (executed["coordinate_frame"] != "global_map" or executed["point_reference"] != "rear_axle"
            or executed["state_columns"] != list(STATE_COLUMNS)
            or not np.array_equal(executed["states"], record["executed_states"])
            or not np.array_equal(executed["times_seconds"], np.arange(41, dtype=np.float64) * .1)
            or not np.array_equal(executed["times_seconds"], record["executed_state_times_seconds"])):
        raise ContractMismatch("Executed trajectory contract/content mismatch")
    _finite_array(executed["states"], (41, 11), "executed states")
    consequences = record.get("consequences")
    if not isinstance(consequences, Mapping) or set(consequences) != {
            "road", "collision_events", "collision_summary", "ttc_events", "ttc_summary"}:
        raise ContractMismatch("Complete observe=True native events are required")
    for family in ("collision_events", "ttc_events"):
        for event in consequences[family]:
            index = event["state_index"]
            if type(index) is not int or not 0 <= index <= 40 or event["state_time_seconds"] != index * .1:
                raise ContractMismatch("Invalid native event timestamp/index")
            if family == "ttc_events":
                offset = event["projection_offset_steps"]
                if (offset not in (0, 3, 6, 9) or event["observation_index"] != index + offset
                        or event["observation_time_seconds"] != (index + offset) * .1):
                    raise ContractMismatch("TTC projected-observation time differs from native v1 predicate")
    from starVLA.model.modules.st_world.consequence import collision_summary, road_summary, ttc_summary
    expected_road = road_summary(np.asarray(consequences["road"]["native_flag"]))
    expected_collision = collision_summary(consequences["collision_events"])
    expected_ttc = ttc_summary(consequences["ttc_events"])
    if (content_sha256(expected_road) != content_sha256(consequences["road"])
            or expected_collision != consequences["collision_summary"] or expected_ttc != consequences["ttc_summary"]):
        raise ContractMismatch("Canonical event summaries/content mismatch")
    if (expected_collision["nc_min"] != record["metrics"]["no_at_fault_collisions"]
            or 1. - float(expected_road["any"]) != record["metrics"]["drivable_area_compliance"]
            or 1. - float(expected_ttc["any"]) != record["metrics"]["time_to_collision_within_bound"]
            or expected_collision["first_at_fault_time_seconds"] != record["official_first_at_fault_time_seconds"]
            or expected_ttc["first_state_time_seconds"] != record["official_first_ttc_state_time_seconds"]):
        raise ContractMismatch("Event replay disagrees with canonical metrics/first times")
    return feedback


def label_arrays(feedback: Mapping[str, Any], *, verify_receipt: bool = True) -> dict:
    """Return training LABELS only, with independent masks and physical units.

    Missing first events have time-valid=false, never a fabricated zero-second
    label. Temporal event flags are native-predicate observations; motion and
    incremental progress are separately named derived quantities.
    """
    validate_feedback(feedback, verify_receipt=verify_receipt)
    names = list(METRICS) + ["progress_raw_meters", "first_at_fault_time_seconds",
                            "first_ttc_state_time_seconds", "first_road_time_seconds"]
    units = ["0..1"] * len(METRICS) + ["m", "s", "s", "s"]
    values, valid = np.zeros(len(names), dtype=np.float64), np.zeros(len(names), dtype=np.bool_)
    if not feedback["valid"]:
        return {"names": names, "units": units, "values": values, "valid": valid,
                "provenance": ["official_terminal"] * len(METRICS) + ["official_internal", "official_native_event_replay", "official_native_event_replay", "official_native_event_replay"],
                "temporal": None}
    record = canonical_record(feedback)
    con = record["consequences"]
    labels = [record["metrics"][name] for name in METRICS] + [record["progress"]["raw_meters"],
        con["collision_summary"]["first_at_fault_time_seconds"],
        con["ttc_summary"]["first_state_time_seconds"], con["road"]["first_time_seconds"]]
    for i, value in enumerate(labels):
        if value is not None:
            if not math.isfinite(float(value)):
                raise ContractMismatch("Nonfinite valid label")
            values[i], valid[i] = value, True
    nc, ttc = np.zeros(41, dtype=np.bool_), np.zeros(41, dtype=np.bool_)
    for event in con["collision_events"]:
        nc[event["state_index"]] |= event["at_fault"]
    for event in con["ttc_events"]:
        ttc[event["state_index"]] |= event["violation"]
    states = record["executed_states"]
    temporal = {"times_seconds": record["executed_state_times_seconds"].copy(),
        "at_fault_contact": nc, "road_violation": con["road"]["native_flag"].copy(),
        "ttc_violation": ttc, "valid": np.ones((41, 3), dtype=np.bool_),
        "event_provenance": "official_native_predicate_replay_not_credit",
        "speed_mps": np.hypot(states[:, 3], states[:, 4]),
        "acceleration_mps2": states[:, 5:7].copy(),
        "yaw_rate_radps": states[:, 9].copy(),
        "motion_provenance": "derived_from_official_executed_states",
        "continuous_collision_clearance": None, "continuous_road_clearance": None,
        "continuous_ttc_seconds": None,
        "continuous_risk_valid": False,
        "missing_reason": "Native v1 TTC is a binary swept-polygon test; continuous clearance/TTC requires a separately audited geometry computation"}
    return {"names": names, "units": units, "values": values, "valid": valid,
        "provenance": ["official_terminal"] * len(METRICS) + ["official_internal", "official_native_event_replay", "official_native_event_replay", "official_native_event_replay"],
        "temporal": temporal}


def validate_pair(pair: Mapping[str, Any], *, verify_receipts: bool = True) -> Mapping[str, Any]:
    required = {"schema", "base_feedback", "modified_feedback", "intervention", "delta_labels", "delta_temporal",
                "event_transition", "causal_scope", "identity"}
    if not isinstance(pair, Mapping) or set(pair) != required or pair["schema"] != PAIR_SCHEMA:
        raise ContractMismatch("Invalid/unsupported effect pair schema")
    if pair["identity"] != content_sha256({k: v for k, v in pair.items() if k != "identity"}):
        raise ContractMismatch("Effect-pair content identity mismatch")
    base, modified = pair["base_feedback"], pair["modified_feedback"]
    for feedback in (base, modified):
        validate_feedback(feedback, verify_receipt=verify_receipts)
        if not feedback["valid"]:
            raise ContractMismatch("Effect differences require two successfully simulated trajectories")
    for key in ("scene_id", "source_log", "observation_id", "observation_sha256", "context_hash",
                "reference_input_sha256", "numeric_runtime_hash", "score_protocol_hash", "navsim_version", "protocol"):
        if base[key] != modified[key]:
            raise ContractMismatch(f"Cross-scene/context/runtime pair rejected: {key}")
    intervention = from_jsonable(pair["intervention"])
    if set(intervention) != {"basis_id", "coefficients", "active_interval_seconds", "geometric_distance_m",
                              "executable", "metadata"}:
        raise ContractMismatch("Incomplete explicit intervention identity")
    if not isinstance(intervention["basis_id"], str) or not intervention["basis_id"]:
        raise ContractMismatch("Intervention basis identity required")
    coefficients = np.asarray(intervention["coefficients"], dtype=np.float64)
    if coefficients.ndim != 1 or not coefficients.size or not np.isfinite(coefficients).all():
        raise ContractMismatch("Finite 1D intervention coefficients required")
    interval = np.asarray(intervention["active_interval_seconds"], dtype=np.float64)
    if interval.shape != (2,) or not np.isfinite(interval).all() or not 0 <= interval[0] <= interval[1] <= 4:
        raise ContractMismatch("Intervention interval must lie within actual 0..4 seconds")
    distance = intervention["geometric_distance_m"]
    if isinstance(distance, bool) or not isinstance(distance, (int, float)) or not math.isfinite(distance) or distance < 0:
        raise ContractMismatch("Finite nonnegative physical trajectory distance required")
    if type(intervention["executable"]) is not bool:
        raise ContractMismatch("Executable must record a real bool check outcome")
    if intervention["metadata"].get("distance_definition") != "mean_pointwise_xy_l2_m":
        raise ContractMismatch("Explicit physical-distance definition required")
    before_poses = from_jsonable(base["submitted_trajectory"])["poses"]
    after_poses = from_jsonable(modified["submitted_trajectory"])["poses"]
    actual_distance = float(np.linalg.norm(after_poses[:, :2] - before_poses[:, :2], axis=-1).mean())
    if distance != actual_distance:
        raise ContractMismatch("Recorded physical intervention distance differs from submitted trajectories")
    if pair["causal_scope"] != "controlled_same_context_intervention_local_effect_not_global_gradient":
        raise ContractMismatch("Unsupported causal attribution claim")
    left = label_arrays(base, verify_receipt=False)
    right = label_arrays(modified, verify_receipt=False)
    common = left["valid"] & right["valid"]
    expected = {"names": left["names"], "units": left["units"], "valid": common,
                "values": np.where(common, right["values"] - left["values"], 0.),
                "definition": "modified_minus_base; event times valid only when present in both"}
    if content_sha256(expected) != content_sha256(pair["delta_labels"]):
        raise ContractMismatch("Effect delta/sign/units/masks mismatch")
    temporal_delta = temporal_differences(left["temporal"], right["temporal"])
    if content_sha256(temporal_delta) != content_sha256(pair["delta_temporal"]):
        raise ContractMismatch("Temporal intervention effect labels/masks mismatch")
    def transitions(a, b):
        return {"repaired": bool(a and not b), "introduced": bool(not a and b),
                "unchanged": bool(a == b), "base_event": bool(a), "modified_event": bool(b)}
    a, b = canonical_record(base)["consequences"], canonical_record(modified)["consequences"]
    expected_events = {"collision": transitions(a["collision_summary"]["event"], b["collision_summary"]["event"]),
                       "road": transitions(a["road"]["event"], b["road"]["event"]),
                       "ttc": transitions(a["ttc_summary"]["event"], b["ttc_summary"]["event"])}
    if pair["event_transition"] != expected_events:
        raise ContractMismatch("Effect event-transition labels mismatch")
    return pair


def temporal_differences(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict:
    """Signed observed effect; these finite differences are not global gradients."""
    if not np.array_equal(before["times_seconds"], after["times_seconds"]):
        raise ContractMismatch("Temporal labels require the same physical time grid")
    return {"times_seconds": before["times_seconds"].copy(),
        "event_names": ["at_fault_contact", "road_violation", "ttc_violation"],
        "event_delta": np.stack([after[k].astype(np.int8) - before[k].astype(np.int8)
                                  for k in ("at_fault_contact", "road_violation", "ttc_violation")], axis=-1),
        "event_valid": before["valid"] & after["valid"],
        "speed_delta_mps": after["speed_mps"] - before["speed_mps"],
        "acceleration_delta_mps2": after["acceleration_mps2"] - before["acceleration_mps2"],
        "yaw_rate_delta_radps": after["yaw_rate_radps"] - before["yaw_rate_radps"],
        "definition": "modified_minus_base; observed native events and derived motion, not responsibility or a global gradient"}
