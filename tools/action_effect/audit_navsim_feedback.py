"""Bounded independent native replays, never reused as new scientific samples.

Replay slots are separate call-ledger admissions with the SAME native request
(repetition stays zero). Re-running this audit resumes its own immutable slots.
The command never launches a model, GPU or training job.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import time

import numpy as np

from .common import artifact, digest, load_config, lock, read_json, sha256, verify_artifact, verify_identity, write_json
from .feedback_schema import canonical_record, content_sha256, from_jsonable, to_jsonable
from .simulate_interventions import CallLedger, score_candidate, verified_feedback, write_gzip
from starVLA.model.modules.st_world import consequence as canonical_backend
from starVLA.model.modules.st_world.consequence import ConsequenceEngine, ConsequenceRequest, ConsequenceSpec


def _array_comparison(left, right):
    left, right = np.asarray(left), np.asarray(right)
    same_shape = left.shape == right.shape
    return {"equal": bool(same_shape and left.dtype == right.dtype and np.array_equal(left, right)),
        "shape_equal": same_shape, "dtype_equal": left.dtype == right.dtype,
        "max_absolute_difference": float(np.max(np.abs(left.astype(np.float64)-right.astype(np.float64))))
            if same_shape and left.size else None}


def compare_records(left, right):
    """Exact primary comparison; state-only roundoff stays INCONCLUSIVE.

    Cost/PID/timing fields are excluded. No tolerance is selected after seeing
    the native results. Event timestamps are observations, not causal credit.
    """
    if not left.get("valid") or not right.get("valid"):
        raise ValueError("Two successful native feedback records required")
    checks = {
        "same_request": left["identity"] == right["identity"],
        "same_reference_input": left["reference_contract"]["input_sha256"] == right["reference_contract"]["input_sha256"],
        "terminal_metrics": left["metrics"] == right["metrics"],
        "native_events": content_sha256(left["consequences"]) == content_sha256(right["consequences"]),
        "progress": left["progress"] == right["progress"],
        "canonical_arrays": content_sha256(left["canonical_arrays"]) == content_sha256(right["canonical_arrays"]),
        "first_at_fault": left["official_first_at_fault_time_seconds"] == right["official_first_at_fault_time_seconds"],
        "first_ttc": left["official_first_ttc_state_time_seconds"] == right["official_first_ttc_state_time_seconds"],
    }
    arrays = {name: _array_comparison(left[name], right[name]) for name in (
        "requested_trajectory", "executed_states", "executed_state_times_seconds",
        "executed_body_coordinates", "reference_interpolated_states", "reference_executed_states")}
    critical = ("same_request", "same_reference_input", "terminal_metrics", "native_events", "first_at_fault", "first_ttc")
    exact = all(checks.values()) and all(x["equal"] for x in arrays.values())
    critical_pass = all(checks[k] for k in critical) and arrays["requested_trajectory"]["equal"]
    return {"status": "PASS" if exact else "INCONCLUSIVE" if critical_pass else "FAIL",
            "exact_checks": checks, "array_checks": arrays,
            "tolerance_policy": "exact; state-only numerical differences retained as INCONCLUSIVE, never silently rounded"}


def select_inputs(g0_root, *, max_samples):
    if type(max_samples) is not int or not 1 <= max_samples <= 4:
        raise ValueError("Feedback replay audit is limited to four fixed first TRAIN scenes")
    root = Path(g0_root); reg = verify_identity(read_json(root/"REGISTRATION.json"))
    complete = read_json(root/"COMPLETE.json")
    if (reg.get("split") != "train" or complete["registration_identity"] != reg["identity"]
            or len(complete["scenes"]) != reg["max_samples"]):
        raise ValueError("Complete immutable TRAIN source population required")
    selected = []
    for item, pin in zip(reg["population"][:max_samples], complete["scenes"][:max_samples]):
        receipt = read_json(verify_artifact(pin)); scene = read_json(verify_artifact(receipt["scene_record"]))
        context = item["context"]
        if (receipt["registration_identity"] != reg["identity"] or scene["registration_identity"] != reg["identity"]
                or scene["context_entry"] != context
                or any(scene[k] != context[k] for k in ("scene_id", "source_log", "role"))):
            raise ValueError("Source scene/context identity mismatch")
        base = scene["candidates"][0]
        if base["kind"] != "identity" or base["status"] != "VALID_SCORED":
            raise ValueError("Audit requires a real valid baseline record, not a selected repair")
        feedback = verified_feedback(base)
        request = ConsequenceRequest(**from_jsonable(feedback["request"]))
        if (request.scene_id != context["scene_id"] or request.source_log != context["source_log"]
                or feedback["observation_sha256"] != context["observation_hash"]):
            raise ValueError("Native source request and observation differ")
        selected.append({"context": context, "observation": item["observation"],
            "source_feedback": base["feedback"], "source_canonical_receipt": base["canonical_receipt"],
            "source_scene_receipt": pin, "request": request})
    if len(selected) != max_samples or len({x["request"].scene_id for x in selected}) != max_samples:
        raise ValueError("Insufficient/duplicate fixed source scenes")
    return reg, selected


def run(cfg, *, g0_root, output, max_samples=4, repeats=2, max_simulation_budget=16,
        retry_budget=8, max_seconds=600., resume=False):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or any(os.environ.get(k) != "1" for k in
            ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")):
        raise ValueError("CPU audit requires CUDA hidden and explicit BLAS limits=1")
    if type(repeats) is not int or repeats != 2 or not 0 < max_seconds <= 1800:
        raise ValueError("Exactly two native replays and a bounded wall time are required")
    if not max_samples*repeats <= max_simulation_budget <= 16 or not 0 <= retry_budget <= max_simulation_budget-max_samples*repeats:
        raise ValueError("Audit is limited to sixteen durably charged calls including retries")
    g0_root, output = Path(g0_root).resolve(), Path(output).resolve()
    if (not g0_root.is_relative_to(Path(cfg["artifact_root"]).resolve()/"g0") or output.is_relative_to(g0_root)
            or not output.is_relative_to(Path(cfg["artifact_root"]).resolve())):
        raise ValueError("Use the actual campaign TRAIN population and a separate audit output")
    reg, selected = select_inputs(g0_root, max_samples=max_samples)
    if reg["config_hash"] != digest(cfg): raise ValueError("Audit configuration differs from the bound campaign")
    spec = ConsequenceSpec(**reg["spec"])
    sources = {str(Path(__file__).resolve()): sha256(__file__),
               str(Path(canonical_backend.__file__).resolve()): sha256(canonical_backend.__file__)}
    for name in ("simulate_interventions.py", "feedback_schema.py", "navsim_feedback_adapter.py", "common.py"):
        p = Path(__file__).with_name(name); sources[str(p.resolve())] = sha256(p)
    registration = {"schema": "action_effect_native_replay_audit_v1", "source_registration": artifact(g0_root/"REGISTRATION.json"),
        "source_completion": artifact(g0_root/"COMPLETE.json"), "source_identity": reg["identity"],
        "score_protocol_hash": spec.score_protocol_hash, "numeric_runtime_hash": reg["numeric_runtime_hash"],
        "max_samples": max_samples, "repeats": repeats, "max_calls": max_simulation_budget, "retry_budget": retry_budget,
        "selected": [{k: to_jsonable(asdict(v)) if k == "request" else v for k,v in x.items()} for x in selected],
        "source_hashes": sources, "comparison": "exact_metrics_events_input_and_arrays; no posthoc tolerance"}
    registration["identity"] = digest(registration)
    write_json(output/"REGISTRATION.json", registration)
    if not resume and (output/"calls"/"COUNTER.json").exists(): raise ValueError("Existing audit requires --resume")
    started = time.monotonic(); deadline = started+max_seconds
    ledger = CallLedger(output/"calls", identity=registration["identity"], max_calls=max_simulation_budget, retry_budget=retry_budget)
    with lock(output/"RUN.lock"):
        if (output/"COMPLETE.json").exists():
            r = read_json(output/"COMPLETE.json")
            if r["registration_identity"] != registration["identity"]: raise ValueError("Foreign audit completion")
            report = read_json(verify_artifact(r["report"]))
            for scene in report["scenes"]:
                for row in scene["replays"]: verified_feedback(row)
            return {**report, "reused": True, "new_calls": 0}
        os.environ["NUPLAN_MAPS_ROOT"] = spec.maps_root
        os.environ["NUPLAN_MAP_VERSION"] = "nuplan-maps-v1.0"
        engine = ConsequenceEngine(spec); scenes = []; errors = []
        before = read_json(output/"calls"/"COUNTER.json")["reserved_calls"]
        for item in selected:
            req = item["request"]; rows = []; records = []
            for repeat in range(repeats):
                try:
                    folder = output/"scenes"/req.scene_id/f"replay_{repeat:02d}"
                    record, wrapped, attempt = score_candidate(engine, ledger, folder, req,
                        observation=item["observation"], deadline=deadline)
                    write_gzip(folder/"FEEDBACK.json.gz", wrapped)
                    row = {"repeat": repeat, "feedback": artifact(folder/"FEEDBACK.json.gz"),
                           "canonical_receipt": artifact(attempt/"COMPLETE.json")}
                    verified_feedback(row)
                    rows.append(row)
                    if not record["valid"]: raise ValueError("Invalid replay: "+record["status"])
                    records.append(record)
                except Exception as exc:
                    errors.append({"scene_id": req.scene_id, "repeat": repeat,
                                   "error_type": type(exc).__name__, "error": str(exc)})
                    break
            source = canonical_record(verified_feedback({"feedback":item["source_feedback"],
                                                         "canonical_receipt":item["source_canonical_receipt"]}))
            scenes.append({"scene_id": req.scene_id, "source_log": req.source_log, "role": item["context"]["role"],
                "replays": rows, "successful_replays":len(records),
                "pair_comparison": compare_records(records[0],records[1]) if len(records)==2 else None,
                "comparisons_to_source": [compare_records(source,record) for record in records]})
        after = read_json(output/"calls"/"COUNTER.json")
        complete = not errors and all(len(x["replays"]) == repeats for x in scenes)
        comparisons = [x["pair_comparison"] for x in scenes if x["pair_comparison"]]
        comparisons += [p for x in scenes for p in x["comparisons_to_source"]]
        status = "PARTIAL" if not complete else "FAIL" if any(c["status"]=="FAIL" for c in comparisons) else (
            "INCONCLUSIVE" if any(c["status"]=="INCONCLUSIVE" for c in comparisons) else "PASS")
        report = {"schema":"action_effect_native_replay_audit_report_v1","status":status,
            "registration_identity":registration["identity"],"requested_scenes":max_samples,"requested_replays":max_samples*repeats,
            "completed_replays":sum(s["successful_replays"] for s in scenes),
            "recorded_replay_slots":sum(len(s["replays"]) for s in scenes),"scenes":scenes,"errors":errors,
            "reserved_calls":after["reserved_calls"],"reserved_retries":after["reserved_retries"],
            "new_calls":after["reserved_calls"]-before,"wall_seconds":time.monotonic()-started,
            "simulation_batch":"official fixed reference + one candidate each call; audit slots repeat that native protocol",
            "event_credit_semantics":"observations only, no causal credit assigned"}
        write_json(output/"LATEST_REPORT.json",report,immutable=False)
        if complete:
            write_json(output/"AUDIT_REPORT.json",report)
            write_json(output/"COMPLETE.json",{"registration_identity":registration["identity"],"status":status,
                                               "report":artifact(output/"AUDIT_REPORT.json")})
        return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",required=True); p.add_argument("--g0-root",required=True); p.add_argument("--output",required=True)
    p.add_argument("--max-samples",type=int,default=4); p.add_argument("--repeats",type=int,default=2)
    p.add_argument("--max-simulation-budget",type=int,default=16); p.add_argument("--retry-budget",type=int,default=8)
    p.add_argument("--max-seconds",type=float,default=600.); p.add_argument("--resume",action="store_true")
    p.add_argument("--dry-run",action="store_true")
    args = p.parse_args(argv)
    try:
        cfg = load_config(args.config)
        if args.dry_run:
            if (args.repeats != 2 or not args.max_samples*2 <= args.max_simulation_budget <= 16
                    or not 0 <= args.retry_budget <= args.max_simulation_budget-args.max_samples*2
                    or not 0 < args.max_seconds <= 1800):
                raise ValueError("Dry-run uses the same bounded replay/call/retry/time contract")
            _, items = select_inputs(args.g0_root,max_samples=args.max_samples)
            print(json.dumps({"status":"NOT_RUN","scenes":[x["request"].scene_id for x in items],
                              "planned_native_calls":len(items)*args.repeats,"simulation_calls":0}));return 0
        result=run(cfg,**{k:v for k,v in vars(args).items() if k not in ("config","dry_run")})
        print(json.dumps(result));return 0 if result["status"]=="PASS" else 2
    except Exception as exc:
        print(json.dumps({"status":"FAIL","error_type":type(exc).__name__,"error":str(exc)}));return 1


if __name__=="__main__":raise SystemExit(main())
