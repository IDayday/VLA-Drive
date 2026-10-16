"""Registered, CPU-only engineering parity test; never a policy evaluation.

Plan first: fixed hash-order train scenes and input hashes before any scoring.
Run at most6 canonical calls/scene (plain, observed, reordered duplicate for
each of2 controls). No GPU, no checkpoint/model loading, no dev/Navtest.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import gzip
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

import numpy as np

from starVLA.model.modules.st_world.consequence import (
    ConsequenceEngine, ConsequenceRequest, ConsequenceService, ConsequenceSpec,
    ContractMismatch, SOURCE_COMMIT, array_sha256, file_sha256, fingerprint,
    protocol_identity, trajectory_sha256,
)


ROOT = Path(__file__).resolve().parents[2]
FILES = ("starVLA/model/modules/st_world/consequence.py",
         "tools/st_world_policy/consequence_smoke.py", "tests/st_world/test_consequence.py")
SELECTION_SEED = "st-consequence-smoke-train-20261016-v1"
POLICY_VERSION = "engineering_controls_v1"


def now():
    return datetime.now(timezone.utc).isoformat()


def json_ready(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: json_ready(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_ready(v) for v in value]
    return value


def atomic_json(path, value):
    path = Path(path)
    content = (json.dumps(json_ready(value), indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    if path.exists():
        if path.read_bytes() != content:
            raise ContractMismatch(f"Immutable artifact conflict: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("xb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    # link+unlink, not replace: never race-overwrite an existing completed file.
    os.link(temporary, path)
    temporary.unlink()


def write_records(path, rows):
    payload = json.dumps(json_ready(rows), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temp.open("xb") as stream:
        stream.write(gzip.compress(payload, mtime=0))
        stream.flush()
        os.fsync(stream.fileno())
    os.link(temp, path)
    temp.unlink()


def load_records(path):
    return json.loads(gzip.decompress(path.read_bytes()))


def source_hashes():
    return {rel: file_sha256(ROOT / rel) for rel in FILES}


def check_host():
    if socket.gethostname() not in ("training-vla-zt", "training-vla-zt-worker-0"):
        raise ContractMismatch("This bounded engineering smoke lease is only for local training-vla-zt")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ContractMismatch("Set CUDA_VISIBLE_DEVICES='' for this CPU-only smoke")
    if any(os.environ.get(k) != "1" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")):
        raise ContractMismatch("Set OMP/OPENBLAS/MKL_NUM_THREADS=1 before launching")


def plan(args):
    check_host()
    root = Path(args.output).resolve()
    if (root / "REGISTRATION.json").exists():
        old = json.loads((root / "REGISTRATION.json").read_text())
        if old["source_hashes"] != source_hashes():
            raise ContractMismatch("Source changed after registration; use a new explicitly budgeted attempt")
        for field, value in (("max_scenes", args.max_scenes), ("workers", args.workers)):
            if old[field] != value:
                raise ContractMismatch(f"Cannot silently change registered {field}")
        return {"status": "REGISTERED_REUSED", "registration": str(root / "REGISTRATION.json")}
    receipt_path, train_path = Path(args.cache_receipt).resolve(), Path(args.train_manifest).resolve()
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("status") != "COMPLETE" or receipt.get("cohort") != "diagnostic_random_tokens":
        raise ContractMismatch("Only the registered diagnostic_random_v1 complete train cohort is authorized")
    legal = {(r["log"], r["token"]) for r in json.loads(train_path.read_text())}
    entries = [e for log in receipt["logs"] for e in log["entries"]]
    if len({(e["log"], e["token"]) for e in entries}) != len(entries):
        raise ContractMismatch("Duplicated scene identities in cache receipt")
    if any((e["log"], e["token"]) not in legal for e in entries):
        raise ContractMismatch("Cache cohort contains scenes outside the locked train manifest")
    entries.sort(key=lambda e: fingerprint([SELECTION_SEED, e["log"], e["token"]]))
    entries = entries[:args.max_scenes]
    protocol = protocol_identity(args.navsim_root)
    cache_root = str(Path(receipt["spec"]["build_cache_root"]).resolve())
    spec = ConsequenceSpec(str(Path(args.navsim_root).resolve()), str(Path(args.maps_root).resolve()),
                           (cache_root,), args.campaign_id, protocol["sha256"], True)
    os.environ["NUPLAN_MAPS_ROOT"] = spec.maps_root
    os.environ["NUPLAN_MAP_VERSION"] = "nuplan-maps-v1.0"
    engine = ConsequenceEngine(spec)  # Construction/context reads only, zero simulation calls.
    from navsim.common.dataloader import SceneLoader
    from navsim.common.dataclasses import SceneFilter, SensorConfig
    from navsim.evaluate.pdm_score import get_trajectory_as_array
    code = source_hashes()
    runtime = fingerprint({"generator": POLICY_VERSION, "code": code,
                           "future_gt_usage": "engineering_control_only_not_policy_input",
                           "current_cv": "current_rear_axle_speed_and_heading_only"})
    requests, scenes = [], []
    raw_logs = Path(args.raw_logs).resolve()
    for entry in entries:
        token, log = entry["token"], entry["log"]
        raw_path = raw_logs / (log + ".pkl")
        raw_hash = file_sha256(raw_path)
        loader = SceneLoader(raw_logs, None,
            SceneFilter(num_history_frames=4, num_future_frames=10, frame_interval=1,
                        log_names=[log], tokens=[token]), SensorConfig.build_no_sensors())
        if loader.tokens != [token]:
            raise ContractMismatch(f"Raw log scene missing: {log}/{token}")
        scene = loader.get_scene_from_token(token)
        gt = np.asarray(scene.get_future_trajectory(8).poses, dtype=np.float64)
        provisional = ConsequenceRequest(spec.campaign_id, token, log, "original_gt", POLICY_VERSION,
            None, runtime, trajectory_sha256(gt), entry["sha256"], protocol["sha256"], entry["path"],
            gt.tolist(), "engineering_control", fingerprint(code))
        cache = engine.load_context(provisional)
        reference = get_trajectory_as_array(cache.trajectory, engine.sampling, cache.ego_state.time_point)
        reference_hash = array_sha256(reference)
        gt_request = replace(provisional, reference_sha256=reference_hash)
        speed = float(cache.ego_state.dynamic_car_state.rear_axle_velocity_2d.magnitude())
        cv = np.zeros((8, 3), dtype=np.float64)
        cv[:, 0] = speed * np.arange(1, 9) * 0.5
        cv_request = replace(gt_request, candidate_id="current_constant_velocity", trajectory=cv.tolist(),
                             trajectory_sha256=trajectory_sha256(cv))
        # Shape/yaw/identity contract checked now, before registering any call.
        from starVLA.model.modules.st_world.consequence import _validate_request
        for request in (gt_request, cv_request):
            _validate_request(request, spec, POLICY_VERSION)
            requests.append(asdict(request))
        scenes.append({"scene_id": token, "source_log": log, "role": "train_engineering_smoke",
            "cache_path": entry["path"], "cache_sha256": entry["sha256"],
            "raw_log_path": str(raw_path), "raw_log_sha256": raw_hash,
            "official_reference_sha256": reference_hash,
            "controls": {"original_gt": trajectory_sha256(gt), "current_constant_velocity": trajectory_sha256(cv)}})
    registration = {
        "schema": "st_consequence_smoke_registration_v1", "registered_at_utc": now(),
        "status": "REGISTERED_BEFORE_SCORING", "scientific_result": False,
        "host": socket.gethostname(), "python": sys.executable,
        "code_base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "canonical_base_commit": SOURCE_COMMIT, "source_hashes": code,
        "spec": asdict(spec), "protocol": protocol,
        "scene_selection": "fixed_sha256_order_no_score_access", "selection_seed": SELECTION_SEED,
        "cache_receipt": {"path": str(receipt_path), "sha256": file_sha256(receipt_path)},
        "train_manifest": {"path": str(train_path), "sha256": file_sha256(train_path)},
        "max_scenes": args.max_scenes, "actual_scenes": len(scenes), "workers": args.workers,
        "blas_threads": 1, "gpus": 0, "max_calls_per_scene": 6, "max_calls_total": 6 * len(scenes),
        "phases": ["plain_natural_order", "observer_reverse_order", "observer_duplicate_shuffled"],
        "scenes": scenes, "requests": requests,
        "controls_not_scientific_candidates": True,
        "recovery": "Completed phases are reused; incomplete admitted phase requires explicit remaining-budget audit, never automatic replay",
    }
    atomic_json(root / "REGISTRATION.json", registration)
    return {"status": "REGISTERED", "scenes": len(scenes), "max_calls": 6 * len(scenes),
            "registration": str(root / "REGISTRATION.json"), "sha256": file_sha256(root / "REGISTRATION.json")}


def read_registration(root):
    path = root / "REGISTRATION.json"
    registration = json.loads(path.read_text())
    if registration["source_hashes"] != source_hashes():
        raise ContractMismatch("Source content changed after pre-registration")
    if file_sha256(Path(registration["cache_receipt"]["path"])) != registration["cache_receipt"]["sha256"]:
        raise ContractMismatch("Original cache receipt changed")
    if file_sha256(Path(registration["train_manifest"]["path"])) != registration["train_manifest"]["sha256"]:
        raise ContractMismatch("Original train manifest changed")
    if registration["actual_scenes"] > 32 or registration["max_calls_per_scene"] != 6 or not 1 <= registration["workers"] <= 4:
        raise ContractMismatch("Registration exceeds bounded CPU smoke authorization")
    return registration


def run(args):
    check_host()
    root = Path(args.output).resolve()
    registration = read_registration(root)
    spec = ConsequenceSpec(**registration["spec"])
    if protocol_identity(spec.navsim_root)["sha256"] != spec.score_protocol_hash:
        raise ContractMismatch("Runtime changed after registration")
    requests = [ConsequenceRequest(**r) for r in registration["requests"]]
    for phase in range(3):
        path = root / f"phase_{phase}.json.gz"
        receipt_path = root / f"phase_{phase}_receipt.json"
        admission_path = root / f"phase_{phase}_ADMITTED.json"
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text())
            if file_sha256(path) != receipt["result_sha256"]:
                raise ContractMismatch("Completed phase checksum mismatch")
            continue
        if admission_path.exists():
            raise ContractMismatch(f"Incomplete admitted phase{phase}; audit consumed budget before authorizing a retry")
        if phase == 0:
            phase_requests = [replace(r, observe=False) for r in requests]
        elif phase == 1:
            phase_requests = list(reversed(requests))
        else:
            phase_requests = sorted(requests, key=lambda r: fingerprint(["duplicate_order", r.scene_id, r.candidate_id]))
        atomic_json(admission_path, {"at_utc": now(), "pid": os.getpid(), "phase": phase,
            "registration_sha256": file_sha256(root / "REGISTRATION.json"),
            "max_calls": len(phase_requests), "request_identities_hash": fingerprint([r.identity() for r in phase_requests])})
        start = time.monotonic()
        rows = []
        # Deliberately change scheduler chunk/inflight sizes while every scorer
        # call continues to contain exactly the fixed reference and one control.
        chunk = len(phase_requests) if phase == 0 else (7 if phase == 1 else 3)
        workers = registration["workers"]
        with ConsequenceService(spec, workers=workers, max_calls=len(phase_requests),
                                max_inflight=workers * (2 if phase == 1 else 1)) as service:
            for begin in range(0, len(phase_requests), chunk):
                rows.extend(service.score(phase_requests[begin:begin + chunk],
                                           expected_policy_version=POLICY_VERSION, require_valid=False))
        write_records(path, rows)
        receipt = {"phase": phase, "at_utc": now(), "wall_seconds": time.monotonic() - start,
            "requests": len(rows), "valid": sum(r["valid"] for r in rows),
            "native_calls": sum(r["cost"]["scoring_calls"] for r in rows),
            "result_sha256": file_sha256(path), "bytes": path.stat().st_size}
        atomic_json(receipt_path, receipt)
        print(json.dumps(receipt), flush=True)
        if not all(r["valid"] for r in rows):
            return {"status": "FAIL", "phase": phase, "receipt": str(receipt_path)}
    return verify(args)


def verify(args):
    root = Path(args.output).resolve()
    registration = read_registration(root)
    if (root / "RESULT.json").exists():
        # A former PASS is not proof that its underlying evidence is intact.
        # Legacy results remain preserved, but require explicit re-auditing if
        # they predate this complete evidence receipt.
        receipt = json.loads((root / "EVIDENCE_COMPLETE.json").read_text())
        names = {"REGISTRATION.json", "RESULT.json", "PER_SCENE_PARITY.json", "NEGATIVE_REQUESTS.json"}
        names |= {f"phase_{i}{suffix}" for i in range(3) for suffix in (".json.gz", "_receipt.json")}
        if receipt.get("schema") != "st_consequence_evidence_complete_v1" or set(receipt["files"]) != names:
            raise ContractMismatch("Incomplete or legacy consequence verification evidence")
        for name, expected in receipt["files"].items():
            if file_sha256(root / name) != expected:
                raise ContractMismatch("Changed consequence verification evidence: " + name)
        return json.loads((root / "RESULT.json").read_text())
    phases = []
    receipts = []
    for phase in range(3):
        path = root / f"phase_{phase}.json.gz"
        receipt = json.loads((root / f"phase_{phase}_receipt.json").read_text())
        if file_sha256(path) != receipt["result_sha256"]:
            raise ContractMismatch("Phase result was changed")
        receipts.append(receipt)
        rows = load_records(path)
        indexed = {(r["identity"]["scene_id"], r["identity"]["candidate_id"]): r for r in rows}
        if len(indexed) != len(rows):
            raise ContractMismatch("Duplicate scene/candidate record in a scoring phase")
        phases.append(indexed)
    expected = {(r["scene_id"], r["candidate_id"]) for r in registration["requests"]}
    if any(set(rows) != expected or not all(r["valid"] for r in rows.values()) for rows in phases):
        raise ContractMismatch("Requested denominator is incomplete or contains failed scoring")
    detailed = []
    for key in sorted(expected):
        a, b, d = [phase[key] for phase in phases]
        fields = ("metrics", "executed_states", "executed_body_coordinates", "canonical_arrays",
                  "context_hash", "reference_identity", "progress",
                  "official_first_at_fault_time_seconds", "official_first_ttc_state_time_seconds")
        matches = {name: a[name] == b[name] == d[name] for name in fields}
        matches["duplicate_events"] = b["consequences"] == d["consequences"]
        matches["context_unchanged"] = all(r["context_unchanged"] for r in (a, b, d))
        detailed.append({"scene_id": key[0], "candidate_id": key[1], "exact_matches": matches,
                         "metrics": a["metrics"],
                         "nc_events": len(b["consequences"]["collision_events"]),
                         "ttc_events": len(b["consequences"]["ttc_events"]),
                         "road_event": b["consequences"]["road"]["any"]})
    if not all(all(row["exact_matches"].values()) for row in detailed):
        atomic_json(root / "PARITY_FAILURES.json", detailed)
        raise ContractMismatch("Plain/observer/reorder/duplicate parity failed")
    # Negative requests must fail *before* any official scorer call. They do not
    # consume additional scene candidate calls from the registered budget.
    engine = ConsequenceEngine(ConsequenceSpec(**registration["spec"]))
    first = ConsequenceRequest(**registration["requests"][0])
    negatives = [replace(first, trajectory=np.full((8, 3), np.nan)),
                 replace(first, metric_cache_sha256="0" * 64),
                 replace(first, reference_sha256="0" * 64),
                 replace(first, policy_version="stale")]
    bad = [engine.score(request, POLICY_VERSION) for request in negatives]
    if any(r["valid"] or r["metrics"] is not None or r["cost"]["scoring_calls"] != 0 for r in bad):
        raise ContractMismatch("Invalid request was scored or became a valid zero reward")
    native_calls = sum(receipt["native_calls"] for receipt in receipts)
    if native_calls > registration["max_calls_total"]:
        raise ContractMismatch("Registered native-call budget exceeded")
    summary = {
        "status": "PASS", "review_status": "NEEDS_INDEPENDENT_REVIEW", "scientific_gain": "UNTESTED",
        "actual_scenes": registration["actual_scenes"], "distinct_controls": len(expected),
        "native_calls": native_calls, "max_calls_total": registration["max_calls_total"],
        "all_metrics_and_states_exact": True, "reorder_duplicate_and_chunk_exact": True,
        "zero_score_valid_controls": sum(row["metrics"]["score"] == 0.0 for row in detailed),
        "controls_with_nc_events": sum(row["nc_events"] > 0 for row in detailed),
        "controls_with_ttc_events": sum(row["ttc_events"] > 0 for row in detailed),
        "controls_with_road_events": sum(row["road_event"] for row in detailed),
        "failure_statuses": [r["status"] for r in bad], "failure_calls": 0,
        "workers": registration["workers"], "gpus": 0, "blas_threads": 1,
        "phase_wall_seconds": [r["wall_seconds"] for r in receipts],
        "worker_user_cpu_seconds": [sum(r["cost"]["worker_cpu_user_seconds"] for r in p.values()) for p in phases],
        "worker_system_cpu_seconds": [sum(r["cost"]["worker_cpu_system_seconds"] for r in p.values()) for p in phases],
        "phase_artifact_bytes": [r["bytes"] for r in receipts],
        "registration_sha256": file_sha256(root / "REGISTRATION.json"),
        "source_hashes": registration["source_hashes"],
        "score_protocol_hash": registration["spec"]["score_protocol_hash"],
        "boundaries": ["Engineering GT/current-velocity controls; not a trained policy result",
            "No dev/Navtest, no GPU/model inference, no scientific reward or gradient claim",
            "Full read-only event replay adds CPU queries; it is not an optimized scorer backend"],
    }
    atomic_json(root / "PER_SCENE_PARITY.json", detailed)
    atomic_json(root / "NEGATIVE_REQUESTS.json", bad)
    atomic_json(root / "RESULT.json", summary)
    names = {"REGISTRATION.json", "RESULT.json", "PER_SCENE_PARITY.json", "NEGATIVE_REQUESTS.json"}
    names |= {f"phase_{i}{suffix}" for i in range(3) for suffix in (".json.gz", "_receipt.json")}
    atomic_json(root / "EVIDENCE_COMPLETE.json", {"schema": "st_consequence_evidence_complete_v1",
        "files": {name: file_sha256(root / name) for name in sorted(names)}})
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("plan", "run", "verify"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--campaign-id", default="st_consequence_cpu_smoke_20261016")
    parser.add_argument("--cache-receipt")
    parser.add_argument("--train-manifest")
    parser.add_argument("--raw-logs")
    parser.add_argument("--maps-root")
    parser.add_argument("--navsim-root")
    parser.add_argument("--max-scenes", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    if not 1 <= args.max_scenes <= 32 or not 1 <= args.workers <= 4:
        parser.error("Budget: 1..32 scenes,1..4 CPU workers")
    if args.phase == "plan" and any(getattr(args, key) is None for key in
            ("cache_receipt", "train_manifest", "raw_logs", "maps_root", "navsim_root")):
        parser.error("Plan requires explicit cache receipt/train manifest/raw logs/maps root/NAVSIM root")
    try:
        result = {"plan": plan, "run": run, "verify": verify}[args.phase](args)
        print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
        return 0 if result["status"] in ("PASS", "REGISTERED", "REGISTERED_REUSED") else 1
    except Exception as exc:
        print(json.dumps({"status": "FAIL", "error_type": type(exc).__name__, "error": str(exc)}), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
