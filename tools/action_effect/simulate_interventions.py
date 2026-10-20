"""Bounded TRAIN-only G0 simulation with durable per-call and scene receipts."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict
import gzip
import fcntl
import json
import multiprocessing
import os
from pathlib import Path
import resource
import time

import numpy as np

from .common import (artifact, atomic_bytes, digest, load_config, lock, read_json,
                     sha256, verify_artifact, write_json)
from .feedback_schema import (canonical_record, content_sha256, from_jsonable,
                              to_jsonable, validate_feedback)
from .navsim_feedback_adapter import from_canonical_record
from starVLA.model.modules.action_effect.intervention_basis import BasisSpec, intervention_grid, reconstruct
from starVLA.model.modules.action_effect.trajectory_constraints import ConstraintSpec, geometric_screen, tracking_screen
from starVLA.model.modules.st_world.consequence import (
    ConsequenceEngine, ConsequenceRequest, ConsequenceSpec, protocol_identity, trajectory_sha256,
)


class BudgetExhausted(RuntimeError):
    pass


@contextmanager
def serial_lock(path):
    """Short blocking critical section; not a nonblocking scene claim."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield


def write_gzip(path, value):
    raw = json.dumps(to_jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    atomic_bytes(path, gzip.compress(raw, mtime=0))


def read_gzip(pin):
    path = verify_artifact(pin)
    if path.stat().st_size > 32 * 1024 * 1024:
        raise ValueError("Oversized per-candidate artifact")
    with gzip.open(path, "rt") as stream:
        return json.load(stream)


class CallLedger:
    """Reserve durably before scoring; an uncertain reservation is never free.

    Counter is atomically replaced BEFORE an ADMITTED file. A crash in that
    small window can conservatively charge an unused slot; it cannot undercount
    a simulation. An existing ADMITTED with no COMPLETE is never scored again
    as the same attempt. A separate explicitly budgeted retry is required.
    """
    def __init__(self, root, *, identity, max_calls, retry_budget):
        self.root = Path(root)
        write_json(self.root / "IDENTITY.json", {"identity": identity, "max_calls": max_calls,
                                               "retry_budget": retry_budget})
        self.max_calls, self.retry_budget = max_calls, retry_budget
        self.identity = identity
        with serial_lock(self.root / "COUNTER.lock"):
            if not (self.root / "COUNTER.json").exists():
                write_json(self.root / "COUNTER.json", {"reserved_calls": 0, "reserved_retries": 0})

    def admit(self, attempt, *, request_identity, retry, _fault_after_reservation=False):
        path = Path(attempt) / "ADMITTED.json"
        with serial_lock(self.root / "COUNTER.lock"):
            if path.exists():
                old = read_json(path)
                if old["request_identity"] != request_identity or old["retry"] != retry:
                    raise ValueError("Foreign existing call admission")
                return False
            counter = read_json(self.root / "COUNTER.json")
            if counter["reserved_calls"] >= self.max_calls or (retry and counter["reserved_retries"] >= self.retry_budget):
                raise BudgetExhausted("Explicit global simulation/retry budget exhausted")
            counter["reserved_calls"] += 1
            counter["reserved_retries"] += int(retry)
            write_json(self.root / "COUNTER.json", counter, immutable=False)
            if _fault_after_reservation:
                raise RuntimeError("Injected after durable reservation, before admission")
            write_json(path, {"request_identity": request_identity, "retry": retry,
                             "ledger_identity": self.identity, "reservation_number": counter["reserved_calls"],
                             "admitted_unix": time.time()})
            return True


def _validate_identity(value):
    if value.get("identity") != digest({k: v for k, v in value.items() if k != "identity"}):
        raise ValueError("Immutable input identity mismatch")


def prepare(cfg, *, max_samples, max_simulation_budget, workers, context_manifest=None,
            navsim_root=None, maps_root=None):
    if not 1 <= workers <= min(4, cfg["g0"]["workers"]) or not 1 <= max_samples <= cfg["cohort_size"]:
        raise ValueError("Explicit valid CPU/sample budget required")
    if not max_samples <= max_simulation_budget <= cfg["g0"]["max_simulation_budget"]:
        raise ValueError("CLI call budget exceeds declared stage ceiling")
    if cfg["g0"]["candidate_count"] != 26 or cfg["g0"]["safe_score"] != .8 or cfg["g0"]["minimum_progress_fraction"] != .9:
        raise ValueError("G0 candidate count and preregistered repair thresholds changed")
    root = Path(cfg["artifact_root"])
    from .collect_base_trajectories import prepare as validate_binding, validate_context_manifest
    binding = validate_binding(cfg)
    if binding["config_hash"] != digest(cfg):
        raise ValueError("Campaign config changed after Base binding")
    cohort = read_json(verify_artifact(binding["cohort"])); _validate_identity(cohort)
    if cohort.get("split") != "train":
        raise ValueError("G0 requires an immutable TRAIN cohort")
    base = read_json(verify_artifact(binding["base_binding"]))
    if cohort["base_binding_hash"] != base["binding_sha256"]:
        raise ValueError("Foreign Base/context population")
    context_path = Path(context_manifest) if context_manifest else root / "contexts" / f"MANIFEST_{max_samples:06d}.json"
    manifest = read_json(context_path); _validate_identity(manifest)
    if (manifest.get("schema") != "action_effect_context_manifest_v1" or manifest.get("status") != "COMPLETE"
            or manifest["binding_identity"] != binding["identity"] or manifest["max_samples"] < max_samples):
        raise ValueError("Complete matching real S0 context manifest required")
    validate_context_manifest(manifest, binding, cohort["entries"][:manifest["max_samples"]])
    selected = manifest["entries"][:max_samples]
    expected = cohort["entries"][:max_samples]
    if (len(selected) != max_samples or [r["scene_id"] for r in selected] != [r["scene_id"] for r in expected]
            or len({r["scene_id"] for r in selected}) != max_samples):
        raise ValueError("Context population/order differs from pre-score cohort")
    for entry, original in zip(selected, expected):
        if (entry["source_log"] != original["source_log"] or entry["observation_hash"] != original["observation_hash"]
                or entry["role"] != original["effect_role"] or entry["role"] not in ("fit", "validation")):
            raise ValueError("Context scene/log/observation/role join mismatch")
    navsim_root = navsim_root or cfg["g0"].get("navsim_root")
    maps_root = maps_root or cfg["g0"].get("maps_root")
    if not navsim_root or not maps_root or not Path(navsim_root).is_dir() or not Path(maps_root).is_dir():
        raise ValueError("Actual existing NAVSIM/maps preflight paths must be explicit")
    protocol = protocol_identity(str(Path(navsim_root).resolve()))
    roots = tuple(sorted({str(Path(r["metric_cache"]["path"]).resolve().parents[3]) for r in expected}))
    spec = ConsequenceSpec(str(Path(navsim_root).resolve()), str(Path(maps_root).resolve()), roots,
                           cfg["campaign_id"], protocol["sha256"], True)
    source_paths = [Path(__file__), Path(__file__).with_name("feedback_schema.py"),
                    Path(__file__).with_name("navsim_feedback_adapter.py")]
    import inspect
    for function in (reconstruct, geometric_screen, ConsequenceEngine):
        source_paths.append(Path(inspect.getsourcefile(function)))
    reg = {"schema": "action_effect_g0_registration_v1", "binding_identity": binding["identity"],
        "context_manifest": artifact(context_path), "cohort": binding["cohort"],
        "config_hash": digest(cfg), "spec": asdict(spec), "base_checkpoint": binding["base_checkpoint"],
        "numeric_runtime_hash": digest({"base": base["binding_sha256"], "inference": base["inference_contract"],
                                       "extraction_contract": binding["extraction_contract"]}),
        "sources": {str(p.resolve()): sha256(p) for p in source_paths},
        "population": [{"context": r, "observation": o} for r, o in zip(selected, expected)],
        "max_samples": max_samples, "max_calls": max_simulation_budget, "retry_budget": cfg["g0"]["retry_budget"],
        "workers": workers, "seed": cfg["seed"], "basis": asdict(BasisSpec()), "constraints": asdict(ConstraintSpec()),
        "repair_rule": {"safety_full_metrics": ["no_at_fault_collisions", "drivable_area_compliance", "time_to_collision_within_bound"],
                        "zero_epsilon": 1e-8, "minimum_score": .8, "minimum_progress_fraction": .9,
                        "comfort_must_not_regress": True, "minimize": "mean_pointwise_xy_l2_m"},
        "split": "train", "actual_candidates_per_scene": 26}
    reg["identity"] = digest(reg)
    output = root / "g0" / f"population_{max_samples:06d}"
    write_json(output / "REGISTRATION.json", reg)
    write_json(output / "RESOLVED_CONFIG.json", cfg)
    CallLedger(output / "calls", identity=reg["identity"], max_calls=max_simulation_budget,
               retry_budget=reg["retry_budget"])
    return output, reg


_SHARD_CACHE = None
_ENGINE = None
_ENGINE_SPEC = None


def load_context_record(entry):
    global _SHARD_CACHE
    import torch
    path = Path(entry["shard"]["path"])
    stat = path.stat()
    key = (str(path), entry["shard"]["sha256"], stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    if path.stat().st_size > 512 * 1024 * 1024:
        raise ValueError("Context shard exceeds bounded CPU loader")
    if _SHARD_CACHE is None or _SHARD_CACHE[0] != key:
        verify_artifact(entry["shard"])
        _SHARD_CACHE = (key, torch.load(path, map_location="cpu", weights_only=False))
    values = _SHARD_CACHE[1]
    record = values[entry["shard_offset"]]
    for key in ("scene_id", "source_log", "role", "observation_hash"):
        if record[key] != entry[key]:
            raise ValueError("Context shard/offset identity mismatch")
    return record


def worker_engine(spec):
    global _ENGINE, _ENGINE_SPEC
    identity = digest(asdict(spec))
    if _ENGINE is None or _ENGINE_SPEC != identity:
        _ENGINE = ConsequenceEngine(spec)
        _ENGINE_SPEC = identity
    return _ENGINE


def candidate_proposals(record, *, seed):
    base = np.asarray(record["physical"], dtype=np.float64)
    result = []
    for row in intervention_grid(BasisSpec(), seed=seed):
        poses = reconstruct(base, row["coefficients"], BasisSpec())
        result.append({**row, "physical": poses, "trajectory_sha256": trajectory_sha256(poses)})
    for i, value in enumerate(record["extra_policy_physical"]):
        poses = np.asarray(value, dtype=np.float64)
        result.append({"kind": "policy_resample", "intervention_id": f"policy_resample_{i+1}",
            "coefficients": None, "seed": seed+i+1, "basis_sha256": None,
            "ramp_intervals_seconds": [], "active_intervals_seconds": [[0., 4.]],
            "physical": poses, "trajectory_sha256": trajectory_sha256(poses)})
    if len(result) != 26:
        raise ValueError("Exactly24 controlled grid +2 original policy samples required")
    return result


def meets_repair_constraints(base, candidate, tracking, geometry, *, eps=1e-8):
    if not candidate["valid"] or not tracking["controller_tracking_pass"] or not geometry["geometrically_admissible"]:
        return False
    before, after = base["metrics"], candidate["metrics"]
    return (all(after[k] >= 1.-eps for k in ("no_at_fault_collisions", "drivable_area_compliance", "time_to_collision_within_bound"))
        and after["score"] >= .8 and after["comfort"] >= before["comfort"]-eps
        and candidate["progress"]["raw_meters"] >= .9 * base["progress"]["raw_meters"]-eps)


def select_repair(rows):
    base = rows[0]
    if base["status"] != "VALID_SCORED":
        return {"status": "BLOCKED_BASE_FEEDBACK", "selected": None}
    m = base["metrics"]
    failure = m["score"] < .8 or any(m[k] < 1.-1e-8 for k in
        ("no_at_fault_collisions", "drivable_area_compliance", "time_to_collision_within_bound"))
    if not failure:
        return {"status": "NO_CHANGE_REQUIRED", "selected": base["candidate_id"]}
    eligible = [r for r in rows[1:] if r.get("eligible_repair")]
    local = [r for r in eligible if r["kind"] not in ("larger_support", "policy_resample")]
    support = [r for r in eligible if r["kind"] in ("larger_support", "policy_resample")]
    key = lambda r: (r["distance_m"], -r["metrics"]["score"], r["trajectory_sha256"])
    if local:
        return {"status": "VERIFIED_LOCAL_REPAIR", "selected": min(local, key=key)["candidate_id"]}
    return {"status": "REQUIRES_LARGER_SUPPORT_SEARCH" if support else "NO_VERIFIED_REPAIR_WITHIN_BUDGET",
            "selected": None, "verified_support_candidate": min(support, key=key)["candidate_id"] if support else None}


def load_completed_attempt(folder, request):
    receipt = read_json(Path(folder) / "COMPLETE.json")
    if (receipt["request_identity"] != request.identity()
            or receipt["admission_sha256"] != sha256(Path(folder) / "ADMITTED.json")):
        raise ValueError("Call receipt/request/admission mismatch")
    record = from_jsonable(read_gzip(receipt["canonical"]))
    if record["identity"] != request.identity() or content_sha256(record) != receipt["canonical_content_sha256"]:
        raise ValueError("Wrong/corrupt canonical record")
    return record


def verified_feedback(row):
    """Follow the independent receipt all the way to its canonical bytes."""
    wrapped = read_gzip(row["feedback"])
    validate_feedback(wrapped)
    receipt_path = verify_artifact(row["canonical_receipt"])
    if wrapped["canonical_receipt"] != {"path": str(receipt_path), "sha256": row["canonical_receipt"]["sha256"]}:
        raise ValueError("Wrapper and candidate name different canonical receipts")
    request = ConsequenceRequest(**from_jsonable(wrapped["request"]))
    native = load_completed_attempt(receipt_path.parent, request)
    if content_sha256(native) != wrapped["canonical_record_sha256"]:
        raise ValueError("Wrapped record differs from independent canonical bytes")
    return wrapped


def score_candidate(engine, ledger, folder, request, *, observation, deadline):
    # A content-addressed candidate may be shared by separately registered
    # populations. One immutable result remains authoritative; no copy/rescore.
    with lock(Path(folder) / "CACHE_CLAIM.lock"):
        return _score_candidate_locked(engine, ledger, folder, request, observation=observation, deadline=deadline)


def _score_candidate_locked(engine, ledger, folder, request, *, observation, deadline):
    for attempt in range(2):
        path = Path(folder) / f"attempt_{attempt:02d}"
        if (path / "COMPLETE.json").exists():
            result = load_completed_attempt(path, request)
        elif (path / "ERROR.json").exists():
            raise RuntimeError("Prior unexpected native failure requires explicit diagnosis")
        elif (path / "ADMITTED.json").exists():
            # Unknown whether scoring happened: charge it and use a distinct
            # explicitly-budgeted retry, never rerun the same attempt.
            continue
        else:
            if time.monotonic() >= deadline:
                raise BudgetExhausted("Invocation time budget reached before new call admission")
            if not ledger.admit(path, request_identity=request.identity(), retry=attempt > 0):
                raise RuntimeError("Scene claim failed to prevent duplicate call admission")
            try:
                result = engine.score(request, request.policy_version)
            except Exception as exc:
                write_json(path / "ERROR.json", {"error_type": type(exc).__name__, "message": str(exc), "retryable": False})
                raise
            write_gzip(path / "CANONICAL.json.gz", result)
            write_json(path / "COMPLETE.json", {"request_identity": request.identity(),
                "admission_sha256": sha256(path / "ADMITTED.json"), "canonical": artifact(path / "CANONICAL.json.gz"),
                "canonical_content_sha256": content_sha256(result)})
        if result["valid"] or result["status"] != "TRANSIENT_SCORING_FAILURE" or not result.get("error", {}).get("retryable"):
            wrapped = from_canonical_record(result, request, observation_id=request.scene_id,
                observation_sha256=observation["observation_hash"], canonical_receipt={
                    "path": str((path / "COMPLETE.json").resolve()), "sha256": sha256(path / "COMPLETE.json")})
            return result, wrapped, path
    raise BudgetExhausted("Unresolved scoring after one explicit retry; evidence retained")


def process_scene(output, reg, item, deadline):
    root = Path(output); entry, observation = item["context"], item["observation"]
    scene = root / "scenes" / entry["scene_id"]
    try:
        with lock(scene / "CLAIM.lock"):
            if (scene / "COMPLETE.json").exists():
                receipt = read_json(scene / "COMPLETE.json")
                if receipt["registration_identity"] != reg["identity"]:
                    raise ValueError("Stale scene completion")
                verify_artifact(receipt["scene_record"])
                rows = read_json(scene / "SCENE.json")
                for row in rows["candidates"]:
                    if "feedback" in row:
                        verified_feedback(row)
                return {"scene_id": entry["scene_id"], "status": "COMPLETE", "reused": True}
            spec = ConsequenceSpec(**reg["spec"])
            os.environ["NUPLAN_MAPS_ROOT"] = spec.maps_root
            os.environ["NUPLAN_MAP_VERSION"] = "nuplan-maps-v1.0"
            engine = worker_engine(spec)
            ledger = CallLedger(root / "calls", identity=reg["identity"], max_calls=reg["max_calls"], retry_budget=reg["retry_budget"])
            cached = load_context_record(entry)
            proposals = candidate_proposals(cached, seed=reg["seed"])
            runtime = reg["numeric_runtime_hash"]
            version = "S0_" + reg["base_checkpoint"]["sha256"][:24]
            rows, base_result, seen = [], None, {}
            for proposal in proposals:
                candidate_id = proposal["intervention_id"]
                poses = proposal["physical"]
                control = proposal["kind"] not in ("identity", "policy_resample")
                request = ConsequenceRequest(spec.campaign_id, entry["scene_id"], entry["source_log"], candidate_id,
                    version, None if control else reg["base_checkpoint"]["sha256"], runtime,
                    proposal["trajectory_sha256"], observation["metric_cache"]["sha256"], spec.score_protocol_hash,
                    observation["metric_cache"]["path"], poses.tolist(),
                    generator_kind="engineering_control" if control else "policy",
                    generator_source_hash=digest({k: v for k, v in proposal.items() if k != "physical"}) if control else None)
                context = engine.load_context(request)
                # The official native bicycle controller interprets these cache
                # velocity components in the initial ego/rear-axle frame.
                initial_v = context.ego_state.dynamic_car_state.rear_axle_velocity_2d.array
                geometry = geometric_screen(poses, BasisSpec(), ConstraintSpec(), initial_velocity_xy=initial_v)
                distance = float(np.linalg.norm(poses[:, :2]-np.asarray(cached["physical"])[:, :2], axis=-1).mean())
                row = {k: v for k, v in proposal.items() if k != "physical"}
                row.update(candidate_id=candidate_id, distance_m=distance, geometry=geometry,
                           physical=poses.tolist(), eligible_repair=False)
                if proposal["trajectory_sha256"] in seen:
                    row.update(status="DUPLICATE_TRAJECTORY", metrics=None,
                               duplicate_of=seen[proposal["trajectory_sha256"]])
                    rows.append(row); continue
                seen[proposal["trajectory_sha256"]] = candidate_id
                if not geometry["geometrically_admissible"] and proposal["kind"] != "identity":
                    row.update(status="INVALID_GEOMETRIC_PROPOSAL", metrics=None)
                    rows.append(row); continue
                candidate_cache = root.parent / "candidate_cache" / digest(request.identity())
                result, wrapped, attempt = score_candidate(engine, ledger, candidate_cache, request,
                                                           observation=observation, deadline=deadline)
                wrapper_path = scene / candidate_id / "FEEDBACK.json.gz"
                write_gzip(wrapper_path, wrapped)
                row.update(status=result["status"], metrics=result["metrics"], feedback=artifact(wrapper_path),
                           canonical_receipt=artifact(attempt / "COMPLETE.json"))
                if result["valid"]:
                    tracking = tracking_screen(poses, result["executed_states"], BasisSpec(), ConstraintSpec())
                    row.update(tracking=tracking, progress_raw_meters=result["progress"]["raw_meters"],
                               executable=bool(tracking["controller_tracking_pass"] and geometry["geometrically_admissible"]))
                    if proposal["kind"] == "identity":
                        base_result = result
                    elif base_result is not None:
                        row["eligible_repair"] = meets_repair_constraints(base_result, result, tracking, geometry)
                elif result["status"] not in ("INVALID_PREDICTION",):
                    raise RuntimeError("Unresolved/foreign official context: " + result["status"])
                rows.append(row)
            value = {"schema": "action_effect_g0_scene_v1", "registration_identity": reg["identity"],
                "scene_id": entry["scene_id"], "source_log": entry["source_log"], "role": entry["role"],
                "context_entry": entry, "observation_hash": observation["observation_hash"],
                "candidates": rows, "repair": select_repair(rows)}
            write_json(scene / "SCENE.json", value)
            write_json(scene / "COMPLETE.json", {"registration_identity": reg["identity"], "scene_record": artifact(scene / "SCENE.json")})
            return {"scene_id": entry["scene_id"], "status": "COMPLETE", "reused": False}
    except Exception as exc:
        return {"scene_id": entry["scene_id"], "status": "PARTIAL", "error_type": type(exc).__name__, "error": str(exc)}


def run(cfg, *, max_samples, max_simulation_budget, workers, resume=False, context_manifest=None,
        navsim_root=None, maps_root=None, max_seconds=3600.):
    started = time.monotonic()
    cpu_before, child_before = resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or any(os.environ.get(k) != "1" for k in
            ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")):
        raise ValueError("CPU worker launch requires CUDA hidden and all three BLAS limits1")
    if not np.isfinite(max_seconds) or not 0 < max_seconds <= 14400:
        raise ValueError("Explicit bounded invocation time required")
    output, reg = prepare(cfg, max_samples=max_samples, max_simulation_budget=max_simulation_budget,
        workers=workers, context_manifest=context_manifest, navsim_root=navsim_root, maps_root=maps_root)
    if not resume and any((output / "scenes").glob("*/CLAIM.lock")):
        raise ValueError("Prior stage work exists: use explicit --resume")
    deadline = time.monotonic()+max_seconds
    with lock(output / "RUN.lock"):
        results = []
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            pending, at = {}, 0
            while at < len(reg["population"]) or pending:
                while at < len(reg["population"]) and len(pending) < workers and time.monotonic() < deadline:
                    item = reg["population"][at]; at += 1
                    future = pool.submit(process_scene, str(output), reg, item, deadline)
                    pending[future] = item["context"]["scene_id"]
                if not pending:
                    break
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    scene_id = pending.pop(future)
                    try: result = future.result()
                    except Exception as exc: result = {"scene_id": scene_id, "status": "PARTIAL", "error": str(exc)}
                    results.append(result); print(json.dumps(result), flush=True)
        # Existing markers do not bypass the validation performed by each
        # worker, including after a previous COMPLETE becomes stale/corrupt.
        complete = (len(results) == len(reg["population"]) and
                    all(row["status"] == "COMPLETE" for row in results))
        value = {"schema": "action_effect_g0_population_result_v1", "status": "COMPLETE" if complete else "PARTIAL",
                 "registration_identity": reg["identity"], "requested_scenes": len(reg["population"]),
                 "scene_results_this_invocation": results, "calls": read_json(output / "calls" / "COUNTER.json")}
        cpu_after, child_after = resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN)
        value["cost"] = {"invocation_wall_seconds": time.monotonic()-started,
            "parent_cpu_seconds": cpu_after.ru_utime+cpu_after.ru_stime-cpu_before.ru_utime-cpu_before.ru_stime,
            "waited_child_cpu_seconds": child_after.ru_utime+child_after.ru_stime-child_before.ru_utime-child_before.ru_stime,
            "parent_peak_rss_bytes": cpu_after.ru_maxrss*1024,
            "waited_child_max_rss_bytes": child_after.ru_maxrss*1024,
            "rss_scope": "Linux waited-child high-water mark, not summed process-tree memory",
            "gpu_allocation": "none_CPU_only"}
        write_json(output / f"INVOCATION_{time.time_ns()}.json", value)
        if complete:
            write_json(output / "COMPLETE.json", {"registration_identity": reg["identity"],
                "scenes": [artifact(output / "scenes" / r["context"]["scene_id"] / "COMPLETE.json") for r in reg["population"]]})
        return value


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True); p.add_argument("--max-samples", required=True, type=int)
    p.add_argument("--max-simulation-budget", required=True, type=int); p.add_argument("--workers", required=True, type=int)
    p.add_argument("--context-manifest"); p.add_argument("--navsim-root"); p.add_argument("--maps-root")
    p.add_argument("--max-seconds", type=float, default=3600.); p.add_argument("--resume", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)
    try:
        cfg = load_config(args.config)
        if args.dry_run:
            print(json.dumps({"status": "NOT_RUN", "config_hash": digest(cfg), "simulation_calls": 0})); return 0
        result = run(cfg, **{k: v for k, v in vars(args).items() if k not in ("config", "dry_run")})
        print(json.dumps(result)); return 0 if result["status"] == "COMPLETE" else 1
    except Exception as exc:
        print(json.dumps({"status": "FAIL", "error_type": type(exc).__name__, "error": str(exc)})); return 1


if __name__ == "__main__":
    raise SystemExit(main())
