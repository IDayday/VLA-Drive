"""CPU-only, versioned observation of the unmodified NAVSIM v1.1 scorer.

One call always means [the cache's official PDM reference, one candidate].
Events are a read-only replay of the *native* predicates after official scoring;
they are never used to modify trajectories or replace canonical metrics.

Narrow reference-worker pattern adapted from VLA-Drive commit
6edb05d576b40a9718421e86c06d560f184a11c6, safe_flow_improvement/v1_reward.py
SHA256 33a8c9e4742a9b5158f9a7fc01a18bdca2c067f3fe410f8769147d8f65517fd1.
Event loops follow the pinned original pdm_scorer.py, with provenance below.
No torch, VLM, GT, MIN/refiner, or deployment dependency is introduced here.
"""
from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict, dataclass
import hashlib
import importlib.metadata
import importlib.util
import json
import lzma
import multiprocessing
import os
from pathlib import Path
import pickle
import resource
import sys
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


SCHEMA = "st_world_canonical_consequence_v1"
REFERENCE_CONTRACT = "st_world_reference_input_and_execution_v2"
PROTOCOL = "navsim_v1.1_fixed_reference_single_candidate"
SOURCE_COMMIT = "1493deda247107efe076b9294863a6753391298a"
SOURCE_SHA256 = {
    "evaluate/pdm_score.py": "07c3fdf737388f4ee219bba4de5187a2140d8e6411151d686302666be5f063c8",
    "planning/simulation/planner/pdm_planner/scoring/pdm_scorer.py": "02ab075017d4edb1c8bba6476ef1674b84d43dbca0f599df87cfe052b2dd5211",
    "planning/simulation/planner/pdm_planner/scoring/pdm_scorer_utils.py": "110b51aab2005318c0b262ce06a80926d5bdf9588f526ed47294ef8f5d6f30c6",
    "planning/simulation/planner/pdm_planner/simulation/pdm_simulator.py": "ae07b533ada10bc37e973e5685a46a7cafc51de8d922ed02add558452ff04807",
    "planning/simulation/planner/pdm_planner/simulation/batch_lqr.py": "03c4d1b781edfba61dd2eee4f9308cc197e8db264de3df3936b6ff11e30238e7",
    "planning/simulation/planner/pdm_planner/simulation/batch_kinematic_bicycle.py": "253beb3c1e7005a8c18a2aa33f43cbf6181829bc6aeda57f74d7bbc91f081ee0",
    "planning/simulation/planner/pdm_planner/observation/pdm_observation.py": "f0cd40d7394490ed89ecc04f033e2e479dcc08bff292db7c92a4647ad30e64f6",
    "planning/metric_caching/metric_cache_processor.py": "164c6be26ce4b1286d1260886f2124afddc5aae8476692fb43b0120b78ff4895",
    "planning/simulation/planner/pdm_planner/utils/pdm_array_representation.py": "e5102697bf941d2f01ff5fc172416c435f639004884b6959741eea5524614dd3",
}
OFFICIAL_CONFIG = {
    "progress_weight": 5.0, "ttc_weight": 5.0, "comfortable_weight": 2.0,
    "driving_direction_weight": 0.0, "driving_direction_horizon": 1.0,
    "driving_direction_compliance_threshold": 2.0,
    "driving_direction_violation_threshold": 6.0,
    "stopped_speed_threshold": 0.005, "progress_distance_threshold": 5.0,
}
STATE_COLUMNS = (
    "x_m", "y_m", "heading_rad", "velocity_x_mps", "velocity_y_mps",
    "acceleration_x_mps2", "acceleration_y_mps2", "steering_angle_rad",
    "steering_rate_radps", "angular_velocity_radps", "angular_acceleration_radps2",
)
METRICS = ("score", "no_at_fault_collisions", "drivable_area_compliance",
           "ego_progress", "time_to_collision_within_bound", "comfort",
           "driving_direction_compliance")


class ContractMismatch(ValueError):
    """A version, identity, or canonical replay invariant was violated."""


class InvalidPrediction(ValueError):
    """A genuine invalid policy output, not a zero reward or an IO error."""


class IncompatibleContext(ValueError):
    """The locked v1 reference/environment is unavailable or incompatible."""


class ConsequenceBatchError(RuntimeError):
    """All returned rows (including failures) remain accessible to the caller."""

    def __init__(self, records):
        self.records = records
        super().__init__(f"{sum(not r['valid'] for r in records)}/{len(records)} consequence requests failed")


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def fingerprint(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def array_sha256(value: np.ndarray) -> str:
    a = np.ascontiguousarray(value)
    h = hashlib.sha256()
    h.update(json.dumps({"dtype": a.dtype.str, "shape": a.shape}, sort_keys=True).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def trajectory_sha256(trajectory: Any) -> str:
    """Hash the exact canonical float64 pose array, including dtype and shape."""
    return array_sha256(np.asarray(trajectory, dtype=np.float64))


def _sha(value: Any, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ContractMismatch(f"{field} must be a lowercase SHA256")
    return value


def _reference_evidence(request, context_hash, reference, executed_pair, protocol):
    """Keep immutable reference inputs separate from native numerical outputs.

    The official [reference, candidate] simulation is unchanged. Its executed
    reference is an OUTPUT, not an environment identity: native NumPy SIMD vs
    scalar dispatch can depend on temporary-array layout. We retain exact hashes
    and full arrays for both; no rounding or numerical tolerance is introduced.
    """
    legacy = {"full_cache_sha256": request.metric_cache_sha256,
        "reference_interpolated_states_sha256": array_sha256(reference),
        "reference_executed_states_sha256": array_sha256(executed_pair[0]),
        "reference_dtype": reference.dtype.str, "reference_shape": list(reference.shape),
        "score_protocol_hash": request.score_protocol_hash,
        "configuration_hash": fingerprint(OFFICIAL_CONFIG),
        "canonical_source_hashes": dict(SOURCE_SHA256),
        "software_identity": dict(protocol["dependencies"])}
    inputs = {k: v for k, v in legacy.items() if k != "reference_executed_states_sha256"}
    inputs.update(campaign_id=request.campaign_id, scene_id=request.scene_id,
                  source_log=request.source_log, context_hash=context_hash)
    outputs = {"input_sha256": fingerprint(inputs),
        "candidate_trajectory_sha256": request.trajectory_sha256,
        "reference_executed_states_sha256": legacy["reference_executed_states_sha256"],
        "candidate_executed_states_sha256": array_sha256(executed_pair[1])}
    return {"reference_identity": legacy,
        "reference_interpolated_states": reference.copy(),
        "reference_executed_states": executed_pair[0].copy(),
        "reference_contract": {"schema": REFERENCE_CONTRACT, "input": inputs,
            "input_sha256": fingerprint(inputs), "execution": outputs,
            "execution_sha256": fingerprint(outputs)}}


def validated_reference_input(record: Dict[str, Any], request: ConsequenceRequest) -> Dict[str, Any]:
    """Validate exact per-call evidence; return the identity shared by candidates.

    V1 records without full reference arrays are deliberately rejected. They
    remain historical evidence, but are not silently promoted to this contract.
    The caller must still validate canonical metrics/events and complete receipts.
    """
    if (record.get("identity") != request.identity() or record.get("status") != "VALID_SCORED"
            or record.get("valid") is not True or record.get("context_unchanged") is not True):
        raise ContractMismatch("Reference evidence belongs to an invalid/different request or mutated context")
    contract = record.get("reference_contract")
    if (not isinstance(contract, dict) or contract.get("schema") != REFERENCE_CONTRACT
            or set(contract) != {"schema", "input", "input_sha256", "execution", "execution_sha256"}):
        raise ContractMismatch("Missing v2 reference evidence; legacy records require explicit re-scoring")
    inputs, outputs = contract["input"], contract["execution"]
    if not isinstance(inputs, dict) or not isinstance(outputs, dict):
        raise ContractMismatch("Invalid reference evidence schema")
    legacy = record.get("reference_identity", {})
    legacy_keys = {"full_cache_sha256", "reference_interpolated_states_sha256",
        "reference_executed_states_sha256", "reference_dtype", "reference_shape",
        "score_protocol_hash", "configuration_hash", "canonical_source_hashes", "software_identity"}
    if not isinstance(legacy, dict) or set(legacy) != legacy_keys:
        raise ContractMismatch("Incomplete official reference identity")
    expected_input = {k: v for k, v in legacy.items() if k != "reference_executed_states_sha256"}
    expected_input.update(campaign_id=request.campaign_id, scene_id=request.scene_id,
                          source_log=request.source_log, context_hash=record.get("context_hash"))
    if inputs != expected_input or contract["input_sha256"] != fingerprint(inputs):
        raise ContractMismatch("Reference input identity/checksum mismatch")
    for name in ("full_cache_sha256", "reference_interpolated_states_sha256", "score_protocol_hash",
                 "configuration_hash", "context_hash"):
        _sha(inputs[name], name)
    if (inputs["full_cache_sha256"] != request.metric_cache_sha256
            or inputs["score_protocol_hash"] != request.score_protocol_hash
            or inputs["configuration_hash"] != fingerprint(OFFICIAL_CONFIG)
            or inputs["canonical_source_hashes"] != SOURCE_SHA256
            or inputs["reference_shape"] != [41, 11] or inputs["reference_dtype"] != "<f8"):
        raise ContractMismatch("Official reference input/cache/protocol/configuration changed")
    if (request.reference_sha256 is not None
            and request.reference_sha256 != inputs["reference_interpolated_states_sha256"]):
        raise ContractMismatch("Reference differs from the request's bound input")
    arrays = {}
    for name in ("reference_interpolated_states", "reference_executed_states", "executed_states"):
        value = record.get(name)
        if (not isinstance(value, np.ndarray) or value.shape != (41, 11)
                or value.dtype.str != "<f8" or not np.isfinite(value).all()):
            raise ContractMismatch(f"Invalid/missing exact native array: {name}")
        arrays[name] = array_sha256(value)
    if arrays["reference_interpolated_states"] != inputs["reference_interpolated_states_sha256"]:
        raise ContractMismatch("Reference input array checksum mismatch")
    if arrays["reference_executed_states"] != legacy["reference_executed_states_sha256"]:
        raise ContractMismatch("Native reference execution checksum mismatch")
    expected_output = {"input_sha256": contract["input_sha256"],
        "candidate_trajectory_sha256": request.trajectory_sha256,
        "reference_executed_states_sha256": arrays["reference_executed_states"],
        "candidate_executed_states_sha256": arrays["executed_states"]}
    if outputs != expected_output or contract["execution_sha256"] != fingerprint(outputs):
        raise ContractMismatch("Native reference/candidate execution provenance mismatch")
    return inputs


def validate_reference_group(records: Sequence[Dict[str, Any]],
                             requests: Sequence[ConsequenceRequest]) -> Dict[str, Any]:
    """Exact common-input check, while independently validating every execution."""
    if not records or len(records) != len(requests):
        raise ContractMismatch("Incomplete official reference group")
    first = validated_reference_input(records[0], requests[0])
    for row, request in zip(records[1:], requests[1:]):
        if validated_reference_input(row, request) != first:
            raise ContractMismatch("Candidates did not use one identical official reference INPUT/context")
    return first


def protocol_identity(navsim_root: str) -> Dict[str, Any]:
    """Bind installed dependencies and all v1 sources; never guess a v2 adapter."""
    root = Path(navsim_root).resolve()
    for relative, expected in SOURCE_SHA256.items():
        path = root / "navsim" / relative
        if not path.is_file() or file_sha256(path) != expected:
            raise ContractMismatch(f"Pinned canonical source differs: {path}")
    sources = {str(p.relative_to(root)): file_sha256(p)
               for p in sorted((root / "navsim").rglob("*.py"))}
    nuplan_spec = importlib.util.find_spec("nuplan")
    if nuplan_spec is None or nuplan_spec.origin is None:
        raise IncompatibleContext("nuplan is not installed in this interpreter")
    nuplan_root = Path(nuplan_spec.origin).parent
    nuplan_sources = {str(p.relative_to(nuplan_root)): file_sha256(p)
                      for p in sorted(nuplan_root.rglob("*.py"))}
    result = {
        "name": PROTOCOL, "canonical_commit": SOURCE_COMMIT, "sources": sources,
        "nuplan_python_sources_sha256": fingerprint(nuplan_sources),
        "proposal_pool": ["fixed_cache_PDM_reference", "one_requested_candidate"],
        "request_sampling": {"num_poses": 8, "interval_length": 0.5},
        "simulation_sampling": {"num_poses": 40, "interval_length": 0.1},
        "configuration": OFFICIAL_CONFIG, "score_units": "0..1",
        "traffic": "locked_logged_replay_not_reactive", "random_evaluation": False,
        "dependencies": {k: importlib.metadata.version(k) for k in ("numpy", "scipy", "shapely")},
        "python": list(sys.version_info[:3]),
    }
    result["sha256"] = fingerprint(result)
    return result


@dataclass(frozen=True)
class ConsequenceSpec:
    navsim_root: str
    maps_root: str
    cache_roots: Tuple[str, ...]
    campaign_id: str
    score_protocol_hash: str
    allow_engineering_controls: bool = False


@dataclass(frozen=True)
class ConsequenceRequest:
    campaign_id: str
    scene_id: str
    source_log: str
    candidate_id: str
    policy_version: str
    generator_checkpoint_sha256: Optional[str]
    inference_runtime_hash: str
    trajectory_sha256: str
    metric_cache_sha256: str
    score_protocol_hash: str
    metric_cache_path: str
    trajectory: Any
    generator_kind: str = "policy"
    generator_source_hash: Optional[str] = None
    reference_sha256: Optional[str] = None
    repetition: int = 0
    observe: bool = True
    generator_identity: Optional[Dict[str, Any]] = None

    def identity(self):
        # Preserve the identity of existing file-backed requests. New in-memory
        # optimizer-boundary snapshots have an explicit typed provenance.
        return {k: v for k, v in vars(self).items() if k not in ("trajectory", "observe")
                and not (k == "generator_identity" and v is None)}


def _validate_request(request: ConsequenceRequest, spec: ConsequenceSpec, policy_version: str) -> np.ndarray:
    if request.campaign_id != spec.campaign_id or request.policy_version != policy_version:
        raise ContractMismatch("Campaign or policy_version mismatch (stale rollout rejected)")
    if request.score_protocol_hash != spec.score_protocol_hash:
        raise ContractMismatch("Scoring protocol changed")
    for key in ("scene_id", "source_log", "candidate_id", "policy_version"):
        value = getattr(request, key)
        if not isinstance(value, str) or not value or any(x in value for x in ("/", "\\", "..")):
            raise ContractMismatch(f"Invalid {key}")
    for key in ("inference_runtime_hash", "trajectory_sha256", "metric_cache_sha256", "score_protocol_hash"):
        _sha(getattr(request, key), key)
    if request.generator_kind == "policy":
        _sha(request.generator_checkpoint_sha256, "generator_checkpoint_sha256")
        if request.generator_identity is not None:
            raise ContractMismatch("File-backed policy identity cannot also be an in-memory snapshot")
    elif request.generator_kind == "policy_snapshot":
        if request.generator_checkpoint_sha256 is not None:
            raise ContractMismatch("In-memory snapshots must not pretend to be checkpoint files")
        origin = request.generator_identity
        required = {"kind", "state_sha256", "optimizer_step", "policy_version",
                    "parent_checkpoint_sha256", "parent_receipt_sha256", "numeric_execution_hash"}
        if not isinstance(origin, dict) or set(origin) != required:
            raise ContractMismatch("Complete typed optimizer-boundary provenance is required")
        if (origin["kind"] != "in_memory_parameter_snapshot"
                or type(origin["optimizer_step"]) is not int or origin["optimizer_step"] < 0
                or origin["policy_version"] != request.policy_version
                or origin["numeric_execution_hash"] != request.inference_runtime_hash):
            raise ContractMismatch("In-memory policy version/step/numeric execution mismatch")
        for key in ("state_sha256", "parent_checkpoint_sha256", "parent_receipt_sha256", "numeric_execution_hash"):
            _sha(origin[key], key)
        if _sha(request.generator_source_hash, "generator_source_hash") != fingerprint(origin):
            raise ContractMismatch("In-memory generator provenance checksum mismatch")
    elif request.generator_kind == "engineering_control" and spec.allow_engineering_controls:
        if request.generator_checkpoint_sha256 is not None or request.generator_identity is not None:
            raise ContractMismatch("Engineering controls must not pretend to originate from policy checkpoints")
        _sha(request.generator_source_hash, "generator_source_hash")
    else:
        raise ContractMismatch("Unsupported generator source")
    if request.reference_sha256 is not None:
        _sha(request.reference_sha256, "reference_sha256")
    if request.repetition != 0:
        raise ContractMismatch("Locked deterministic v1 protocol uses repetition=0; repeats are consistency audits")
    try:
        trajectory = np.asarray(request.trajectory, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise InvalidPrediction("Trajectory is not a numeric array") from exc
    if trajectory.shape != (8, 3) or not np.isfinite(trajectory).all():
        raise InvalidPrediction("Expected finite [8,3] rear-axle-local poses at .5..4 seconds")
    if (np.abs(trajectory[:, 2]) > np.pi + 1e-8).any():
        raise InvalidPrediction("Yaw is not a principal angle in radians; no implicit wrapping")
    if trajectory_sha256(trajectory) != request.trajectory_sha256:
        raise ContractMismatch("Trajectory content differs from the request identity")
    return trajectory.copy()


def event_bin(state_index: int) -> int:
    """Native 10Hz state -> [0,.5],(.5,1],...,(3.5,4] physical-time bin."""
    if not isinstance(state_index, (int, np.integer)) or not 0 <= state_index <= 40:
        raise ContractMismatch("Event state_index must be an integer in [0,40]")
    return max(0, (int(state_index) - 1) // 5)


def road_summary(flags: Sequence[bool]) -> Dict[str, Any]:
    flags = np.asarray(flags)
    if flags.shape != (41,) or flags.dtype != np.bool_:
        raise ContractMismatch("Native road flags must be bool[41]")
    bins = [False] * 8
    indices = np.flatnonzero(flags)
    for i in indices:
        bins[event_bin(i)] = True
    return {"native_flag": flags.copy(), "segment_any": bins, "any": bool(len(indices)),
            "first_time_seconds": float(indices[0] * 0.1) if len(indices) else None,
            "event": bool(len(indices)), "valid": True,
            "definition": "canonical_four_body_corners_not_center_only_not_polygon_interior_proof"}


def collision_summary(events: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    bins = [1.0] * 8
    first = None
    for e in events:
        if e["at_fault"]:
            i = e["state_index"]
            bins[event_bin(i)] = min(bins[event_bin(i)], e["nc_contribution"])
            first = i if first is None else min(i, first)
    return {"segment_nc_min": bins, "nc_min": min(bins),
            "first_at_fault_time_seconds": first * 0.1 if first is not None else None,
            "event": first is not None, "valid": True}


def ttc_summary(events: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    indices = [e["state_index"] for e in events if e["violation"]]
    bins = [False] * 8
    for i in indices:
        bins[event_bin(i)] = True
    return {"segment_any": bins, "any": bool(indices), "event": bool(indices), "valid": True,
            "first_state_time_seconds": min(indices) * 0.1 if indices else None}


def _native():
    # Lazy imports keep this module usable without torch/CUDA and avoid v1/v2
    # pollution until the explicitly selected runtime has been verified.
    from navsim.planning.simulation.planner.pdm_planner.scoring import pdm_scorer
    return pdm_scorer


def collision_events(scorer) -> List[List[Dict[str, Any]]]:
    """Replay canonical contact order, including repeated at-fault contacts.

    Native non-at-fault contacts join a local ignore list. Native at-fault
    contacts do NOT, so a later worse contact and repeated events must survive.
    """
    n = _native()
    observation = scorer._observation
    collided = [list(observation.collided_track_ids) for _ in range(scorer._num_proposals)]
    output = [[] for _ in range(scorer._num_proposals)]
    scores = np.ones(scorer._num_proposals)
    ordinal = 0
    for t in range(scorer.proposal_sampling.num_poses + 1):
        hits = observation[t].query(scorer._ego_polygons[:, t], predicate="intersects")
        if len(hits) == 0:
            continue
        for p, g in zip(hits[0], hits[1]):
            p = int(p)
            token = observation[t].tokens[g]
            ordinal += 1
            if observation.red_light_token in token or token in collided[p]:
                continue
            off = bool(scorer._ego_areas[p, t, n.EgoAreaIndex.MULTIPLE_LANES]
                       or scorer._ego_areas[p, t, n.EgoAreaIndex.NON_DRIVABLE_AREA])
            obj = observation.unique_objects[token]
            kind = n.get_collision_type(scorer._states[p, t], scorer._ego_polygons[p, t], obj,
                                        observation[t][token])
            fault = (kind in (n.CollisionType.ACTIVE_FRONT_COLLISION, n.CollisionType.STOPPED_TRACK_COLLISION)
                     or (off and kind == n.CollisionType.ACTIVE_LATERAL_COLLISION))
            contribution = (0.0 if obj.tracked_object_type in n.AGENT_TYPES else 0.5) if fault else 1.0
            new_minimum = contribution < scores[p]
            scores[p] = min(scores[p], contribution)
            if not fault:
                collided[p].append(token)
            output[p].append({
                "state_index": t, "state_time_seconds": t * 0.1, "object_token": str(token),
                "object_type": obj.tracked_object_type.name, "collision_type": kind.name,
                "at_fault": bool(fault), "nc_contribution": contribution,
                "loop_ordinal": ordinal, "new_minimum": bool(new_minimum),
                "multiple_lanes_or_non_drivable": off,
            })
    return output


def ttc_events(scorer) -> List[List[Dict[str, Any]]]:
    """Replay native four projection offsets; first time refers to the ego state.

    This is the canonical TTC binary test, not distance/speed TTC. The last
    state at4s reads logged occupancy at4.9s. No observation clamping/filling.
    """
    n = _native()
    observation = scorer._observation
    collided = [list(observation.collided_track_ids) for _ in range(scorer._num_proposals)]
    output = [[] for _ in range(scorer._num_proposals)]
    offsets = np.arange(0, 10, 3)
    exterior = scorer._ego_coords.copy()
    exterior[:, :, n.BBCoordsIndex.CENTER] = exterior[:, :, n.BBCoordsIndex.FRONT_LEFT]
    projected = np.repeat(exterior[:, :, None], len(offsets), axis=2)
    speeds = np.hypot(scorer._states[..., n.StateIndex.VELOCITY_X], scorer._states[..., n.StateIndex.VELOCITY_Y])
    dxy = np.stack([np.cos(scorer._states[..., n.StateIndex.HEADING]) * speeds,
                    np.sin(scorer._states[..., n.StateIndex.HEADING]) * speeds], axis=-1)
    for j, offset in enumerate(offsets):
        projected[:, :, j] = projected[:, :, j] + dxy[:, :, None] * (float(offset) * 0.1)
    polygons = n.creation.polygons(projected)
    ordinal = 0
    for t in range(scorer.proposal_sampling.num_poses + 1):
        for j, offset in enumerate(offsets):
            u = t + int(offset)
            hits = observation[u].query(polygons[:, t, j], predicate="intersects")
            if len(hits) == 0:
                continue
            for p, g in zip(hits[0], hits[1]):
                p = int(p)
                token = observation[u].tokens[g]
                ordinal += 1
                if (observation.red_light_token in token or token in collided[p]
                        or speeds[p, t] < scorer._config.stopped_speed_threshold):
                    continue
                off = bool(scorer._ego_areas[p, t, n.EgoAreaIndex.MULTIPLE_LANES]
                           or scorer._ego_areas[p, t, n.EgoAreaIndex.NON_DRIVABLE_AREA])
                ego = n.StateSE2(*scorer._states[p, t, n.StateIndex.STATE_SE2])
                centroid = observation[u][token].centroid
                heading = observation.unique_objects[token].box.center.heading
                other = n.StateSE2(centroid.x, centroid.y, heading)
                # Preserve the native short-circuit predicates exactly.
                violation = n.is_agent_ahead(ego, other) or (
                    (off or scorer._drivable_area_map.is_in_layer(ego.point, layer=n.SemanticMapLayer.INTERSECTION))
                    and not n.is_agent_behind(ego, other))
                if not violation:
                    collided[p].append(token)
                output[p].append({
                    "state_index": t, "state_time_seconds": t * 0.1,
                    "projection_offset_steps": int(offset), "observation_index": u,
                    "observation_time_seconds": u * 0.1, "object_token": str(token),
                    "violation": bool(violation), "loop_ordinal": ordinal,
                    "ego_speed_mps": float(speeds[p, t]), "multiple_lanes_or_non_drivable": off,
                })
    return output


def _finite_time(value):
    return float(value) if np.isfinite(value) else None


def observe_canonical(scorer, proposal_index: int = 1) -> Dict[str, Any]:
    """Read only a completed fixed-reference scorer; assert every replay label."""
    if scorer._num_proposals != 2 or scorer._states.shape != (2, 41, 11):
        raise ContractMismatch("Observer requires exactly [official_reference, one_candidate]")
    if scorer.proposal_sampling.interval_length != 0.1 or scorer.proposal_sampling.num_poses != 40:
        raise ContractMismatch("Observer sampling differs from pinned v1 contract")
    if proposal_index not in (0, 1):
        raise ContractMismatch("Invalid fixed-pair proposal index")
    n = _native()
    nc_events = collision_events(scorer)
    tc_events = ttc_events(scorer)
    results = []
    for p in range(2):
        road = road_summary(scorer._ego_areas[p, :, n.EgoAreaIndex.NON_DRIVABLE_AREA])
        nc = collision_summary(nc_events[p])
        tc = ttc_summary(tc_events[p])
        checks = (
            (nc["nc_min"], float(scorer._multi_metrics[n.MultiMetricIndex.NO_COLLISION, p])),
            (1.0 - float(road["any"]), float(scorer._multi_metrics[n.MultiMetricIndex.DRIVABLE_AREA, p])),
            (1.0 - float(tc["any"]), float(scorer._weighted_metrics[n.WeightedMetricIndex.TTC, p])),
            (nc["first_at_fault_time_seconds"], _finite_time(scorer.time_to_at_fault_collision(p))),
            (tc["first_state_time_seconds"], _finite_time(scorer.time_to_ttc_infraction(p))),
        )
        if any(a != b for a, b in checks):
            raise ContractMismatch(f"Read-only observer disagrees with canonical consequences: {checks}")
        results.append({"road": road, "collision_events": nc_events[p], "collision_summary": nc,
                        "ttc_events": tc_events[p], "ttc_summary": tc})
    return results[proposal_index]


def _context_state_hash(cache) -> str:
    """Immutable semantic environment fingerprint, unaffected by lazy geometry caches."""
    o = cache.observation
    h = hashlib.sha256()
    h.update(json.dumps({"collided": o.collided_track_ids, "red_light": o.red_light_token,
                         "global_to_local": o._global_to_local_idcs,
                         "route_lane_ids": cache.route_lane_ids}, sort_keys=True).encode())
    for occupancy in o._occupancy_maps:
        for token, geometry in zip(occupancy.tokens, occupancy._geometries):
            h.update(str(token).encode())
            h.update(geometry.wkb)
    for token, obj in sorted(o.unique_objects.items()):
        # These are exactly the object attributes used by native collision/TTC.
        h.update(str(token).encode())
        h.update(str(obj.tracked_object_type).encode())
        h.update(np.asarray(obj.box.center.serialize(), dtype=np.float64).tobytes())
        if hasattr(obj, "velocity"):
            h.update(np.asarray([obj.velocity.x, obj.velocity.y], dtype=np.float64).tobytes())
    for token, kind, geometry in zip(cache.drivable_area_map.tokens, cache.drivable_area_map.map_types,
                                     cache.drivable_area_map._geometries):
        h.update(str((token, int(kind))).encode())
        h.update(geometry.wkb)
    # Hash numerical/geometry inputs, not implementation-local lazy-property
    # caches that nuPlan may legally materialize on first access.
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_array_representation import ego_state_to_state_array
    h.update(cache.centerline.linestring.wkb)
    h.update(np.asarray(cache.centerline._states_se2_array, dtype=np.float64).tobytes())
    h.update(ego_state_to_state_array(cache.ego_state).tobytes())
    h.update(str(cache.ego_state.time_point.time_us).encode())
    h.update(pickle.dumps(cache.ego_state.car_footprint.vehicle_parameters, protocol=4))
    return h.hexdigest()


def validate_cache(cache):
    from navsim.planning.metric_caching.metric_cache import MetricCache
    expected = {"file_path", "trajectory", "ego_state", "observation", "centerline", "route_lane_ids", "drivable_area_map"}
    if type(cache) is not MetricCache or set(vars(cache)) != expected:
        raise IncompatibleContext("Cache is not the exact v1 MetricCache schema; v2 conversion forbidden")
    o = cache.observation
    if not o._initialized or o._sample_interval != 0.1 or o._observation_sample_res != 1:
        raise IncompatibleContext("Official logged replay must be initialized at10Hz with resolution1")
    if len(o._global_to_local_idcs) < 50 or len(o._occupancy_maps) < 50:
        raise IncompatibleContext("TTC at4s requires native observation index49 (4.9s)")
    for t in range(50):
        if o._global_to_local_idcs[t] != t:
            raise IncompatibleContext("Unexpected temporal aliasing in official logged replay")
        o[t]  # Fail before scoring for truncated or uninitialized observations.
    if cache.trajectory.start_time.time_us > cache.ego_state.time_point.time_us:
        raise IncompatibleContext("Official reference starts after the requested initial state")
    if cache.trajectory.end_time.time_us < cache.ego_state.time_point.time_us + 4000000:
        raise IncompatibleContext("Official reference does not cover the four-second evaluation horizon")


class ConsequenceEngine:
    """A sequential CPU worker with private simulator/scorer and one bounded cache.

    Instantiate inside a spawned worker, never share between threads. The class
    is also usable directly in an isolated CPU process for engineering audits.
    """

    def __init__(self, spec: ConsequenceSpec):
        self.spec = spec
        root = Path(spec.navsim_root).resolve()
        if not spec.campaign_id or not spec.cache_roots:
            raise ContractMismatch("Campaign and allowed cache roots must be explicit")
        self.protocol = protocol_identity(str(root))
        if self.protocol["sha256"] != spec.score_protocol_hash:
            raise ContractMismatch("Canonical source/software/config identity differs from registration")
        sys.path.insert(0, str(root))
        import navsim
        if not Path(navsim.__file__).resolve().is_relative_to(root):
            raise ContractMismatch("Foreign navsim already imported: refusing a mixed v1/v2 interpreter")
        # Check loaded modules too: a canonical __init__ must not hide foreign submodules.
        for name, module in list(sys.modules.items()):
            path = getattr(module, "__file__", None)
            if name.startswith("navsim.") and path and not Path(path).resolve().is_relative_to(root):
                raise ContractMismatch(f"Foreign NAVSIM submodule: {name}")
        from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer, PDMScorerConfig
        from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
        from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
        self.sampling = TrajectorySampling(num_poses=40, interval_length=0.1)
        self.action_sampling = TrajectorySampling(num_poses=8, interval_length=0.5)
        self.scorer = PDMScorer(self.sampling, PDMScorerConfig())
        self.simulator = PDMSimulator(self.sampling)
        if asdict(self.scorer._config) != OFFICIAL_CONFIG:
            raise ContractMismatch("Scorer defaults differ from the pinned official configuration")
        self._cache_key = None
        self._cache = None
        self._cache_stat = None
        self._pid = os.getpid()

    def load_context(self, request: ConsequenceRequest):
        path = Path(request.metric_cache_path).resolve()
        roots = [Path(p).resolve() for p in self.spec.cache_roots]
        if not any(path.is_relative_to(root) for root in roots):
            raise ContractMismatch("Metric cache is outside registered cache roots")
        if list(path.parts[-4:]) != [request.source_log, "unknown", request.scene_id, "metric_cache.pkl"]:
            raise ContractMismatch("Cache path does not match scene/log identity")
        stat = path.stat()
        stamp = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        key = (str(path), request.metric_cache_sha256)
        if key == self._cache_key:
            if stamp != self._cache_stat:
                raise ContractMismatch("An already verified immutable metric cache changed")
            return self._cache
        if file_sha256(path) != request.metric_cache_sha256:
            raise ContractMismatch("Metric cache checksum mismatch")
        try:
            with lzma.open(path, "rb") as stream:
                cache = pickle.load(stream)
        except (pickle.UnpicklingError, lzma.LZMAError, EOFError, AttributeError, ModuleNotFoundError) as exc:
            raise IncompatibleContext(f"Unreadable v1 metric cache: {type(exc).__name__}: {exc}") from exc
        validate_cache(cache)
        self._cache_key, self._cache_stat, self._cache = key, stamp, cache
        return cache

    def score(self, request: ConsequenceRequest, expected_policy_version: str) -> Dict[str, Any]:
        started = time.monotonic()
        cpu = resource.getrusage(resource.RUSAGE_SELF)
        calls = 0
        identity = request.identity()
        try:
            if os.getpid() != self._pid:
                raise ContractMismatch("A scorer/simulator instance crossed a process boundary")
            trajectory = _validate_request(request, self.spec, expected_policy_version)
            cache = self.load_context(request)
            from navsim.common.dataclasses import Trajectory
            from navsim.evaluate.pdm_score import get_trajectory_as_array, pdm_score
            reference = get_trajectory_as_array(cache.trajectory, self.sampling, cache.ego_state.time_point)
            reference_hash = array_sha256(reference)
            if request.reference_sha256 is not None and request.reference_sha256 != reference_hash:
                raise ContractMismatch("Official reference content differs from the bound reference")
            before = _context_state_hash(cache)
            calls = 1
            values = asdict(pdm_score(cache, Trajectory(trajectory, self.action_sampling), self.sampling,
                                      self.simulator, self.scorer))
            if set(values) != set(METRICS) or not all(np.isfinite(v) and 0 <= v <= 1 for v in values.values()):
                raise ContractMismatch("Canonical scorer returned invalid metric schema or values")
            observed = observe_canonical(self.scorer) if request.observe else None
            if before != _context_state_hash(cache):
                raise ContractMismatch("Official context was mutated by scoring or observation")
            s = self.scorer
            if s._states.shape != (2, 41, 11) or not np.isfinite(s._states).all():
                raise ContractMismatch("Canonical simulation produced invalid states")
            multipliers = s._multi_metrics.prod(axis=0)
            effective = s._progress_raw * multipliers
            denom = float(effective.max())
            progress = {
                "raw_meters": float(s._progress_raw[1]), "reference_raw_meters": float(s._progress_raw[0]),
                "effective_candidate_meters": float(effective[1]), "effective_reference_meters": float(effective[0]),
                "ep_denominator_meters": denom,
                "ep_branch": "normalize" if denom > 5.0 else "unit_with_zero_multiplier_mask",
            }
            result = {
                "schema": SCHEMA, "status": "VALID_SCORED", "valid": True, "identity": identity,
                "metrics": {k: float(v) for k, v in values.items()}, "metrics_valid": {k: True for k in METRICS},
                "score_units": "0..1", "requested_trajectory": trajectory,
                "executed_states": s._states[1].copy(), "state_columns": STATE_COLUMNS,
                "executed_state_times_seconds": np.arange(41, dtype=np.float64) * 0.1,
                "executed_body_coordinates": s._ego_coords[1].copy(),
                "canonical_arrays": {"multi_metrics": s._multi_metrics.copy(),
                                     "weighted_metrics": s._weighted_metrics.copy(),
                                     "progress_raw": s._progress_raw.copy()},
                "consequences": observed, "progress": progress,
                "official_first_at_fault_time_seconds": _finite_time(s.time_to_at_fault_collision(1)),
                "official_first_ttc_state_time_seconds": _finite_time(s.time_to_ttc_infraction(1)),
                "context_hash": before, "context_unchanged": True,
                **_reference_evidence(request, before, reference, s._states, self.protocol),
            }
            validated_reference_input(result, request)
        except InvalidPrediction as exc:
            result = self._failure(identity, "INVALID_PREDICTION", exc, False)
        except (IncompatibleContext, FileNotFoundError) as exc:
            result = self._failure(identity, "MISSING_OR_INCOMPATIBLE_CONTEXT", exc, False)
        except ContractMismatch as exc:
            result = self._failure(identity, "CONTRACT_MISMATCH", exc, False)
        except (TimeoutError, OSError) as exc:
            result = self._failure(identity, "TRANSIENT_SCORING_FAILURE", exc, True)
        # Unexpected programming/native errors deliberately propagate. They must
        # never silently become a valid low-reward sample or a discarded row.
        after_cpu = resource.getrusage(resource.RUSAGE_SELF)
        result["cost"] = {"worker_wall_seconds": time.monotonic() - started,
                          "worker_cpu_user_seconds": after_cpu.ru_utime - cpu.ru_utime,
                          "worker_cpu_system_seconds": after_cpu.ru_stime - cpu.ru_stime,
                          "scoring_calls": calls, "worker_pid": self._pid}
        return result

    @staticmethod
    def _failure(identity, status, exc, retryable):
        return {"schema": SCHEMA, "status": status, "valid": False, "identity": identity,
                "metrics": None, "metrics_valid": {k: False for k in METRICS},
                "executed_states": None, "consequences": None,
                "error": {"type": type(exc).__name__, "message": str(exc), "retryable": retryable}}


_WORKER_ENGINE = None


def _initialize_worker(spec: ConsequenceSpec):
    global _WORKER_ENGINE
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "1"
    os.environ["NUPLAN_MAPS_ROOT"] = spec.maps_root
    os.environ["NUPLAN_MAP_VERSION"] = "nuplan-maps-v1.0"
    _WORKER_ENGINE = ConsequenceEngine(spec)


def _worker_score(job):
    request, policy_version = job
    if _WORKER_ENGINE is None:
        raise RuntimeError("Spawned consequence worker was not initialized")
    return _WORKER_ENGINE.score(request, policy_version)


class ConsequenceService:
    """Spawn-only bounded submission, order-preserving results, explicit budget.

    A worker owns one scorer, simulator and cache and handles requests serially.
    One request is one official fixed pair, irrespective of chunk/pool size.
    require_valid=True rejects the update while retaining all failure records.
    """

    def __init__(self, spec: ConsequenceSpec, *, workers: int, max_calls: int,
                 max_inflight: Optional[int] = None):
        if not 1 <= workers <= 4 or max_calls < 1:
            raise ValueError("Explicit CPU budget required: workers in[1,4], max_calls>0")
        # Spawn imports this module (and numpy) before its initializer. Require
        # inherited limits, rather than setting them too late or changing a
        # training parent's process-global environment behind its back.
        if any(os.environ.get(k) != "1" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")):
            raise ContractMismatch("Set OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1 before starting workers")
        self.max_inflight = workers if max_inflight is None else max_inflight
        if not workers <= self.max_inflight <= 2 * workers:
            raise ValueError("Submission queue must be bounded between workers and2*workers")
        self.max_calls, self.submitted = max_calls, 0
        self.spec = spec
        self.pool = ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                                        initializer=_initialize_worker, initargs=(spec,))

    def score(self, requests: Iterable[ConsequenceRequest], *, expected_policy_version: str,
              require_valid: bool = True) -> List[Dict[str, Any]]:
        if not expected_policy_version:
            raise ValueError("Explicit current policy version required")
        iterator = iter(requests)
        pending, results = {}, {}
        count, exhausted = 0, False
        while pending or not exhausted:
            while not exhausted and len(pending) < self.max_inflight:
                try:
                    request = next(iterator)
                except StopIteration:
                    exhausted = True
                    break
                if self.submitted >= self.max_calls:
                    raise RuntimeError("Consequence call admission budget exhausted")
                self.submitted += 1
                pending[self.pool.submit(_worker_score, (request, expected_policy_version))] = count
                count += 1
            if pending:
                completed, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in completed:
                    i = pending.pop(future)
                    results[i] = future.result()  # propagate worker death/programming errors
        ordered = [results[i] for i in range(count)]
        if require_valid and any(not r["valid"] for r in ordered):
            raise ConsequenceBatchError(ordered)
        return ordered

    def close(self):
        self.pool.shutdown(wait=True, cancel_futures=True)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
