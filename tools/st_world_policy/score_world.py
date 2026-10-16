"""Bounded canonical train/dev scoring for immutable single-trajectory exports.

All metrics come from the existing ConsequenceService. No metric formula or
candidate selection is implemented here. The shared admitted-call ledger keeps
completed feedback and failure attempts across interruptions.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import fcntl
import io
import json
import math
import os
from pathlib import Path
import time

import numpy as np

from starVLA.model.modules.st_world.consequence import (
    ConsequenceRequest, ConsequenceService, ConsequenceSpec, METRICS,
    file_sha256, fingerprint, protocol_identity, trajectory_sha256,
)
from tools.st_world_policy.base_binding import immutable_json
from tools.st_world_policy.export_world import validate_cached_prediction


def join_fixed_contexts(scenes, entries):
    if not scenes or len({r["token"] for r in scenes}) != len(scenes):
        raise ValueError("Fixed evaluation population must be nonempty and unique")
    lookup = {row["token"]: row for row in entries}
    if len(lookup) != len(entries):
        raise ValueError("Duplicate metric-context scene identity")
    result = []
    for row in scenes:
        context = lookup.get(row["token"])
        if context is None or context["log"] != row["log"]:
            raise ValueError("Missing or foreign log/scene metric context")
        result.append(context)
    return result


def read_complete_export(root):
    root = Path(root)
    export = json.loads((root / "IDENTITY.json").read_text())
    if (export.get("schema") != "st_single_trajectory_export_v1"
            or fingerprint({k: v for k, v in export.items() if k != "identity"}) != export["identity"]
            or export["observations"]["split"] not in ("train", "dev")
            or export["candidates_per_scene"] != 1 or export["scorer"] is not None
            or export["targets_read"] or export["teacher_cache_read"]):
        raise ValueError("Only complete current/past-only train/dev policy exports are admitted")
    selected = json.loads((root / "OBSERVATIONS.json").read_text())
    scenes = selected["scenes"]
    if (selected["identity"] != export["observations"]["identity"]
            or fingerprint(scenes) != export["selected_scenes_sha256"]
            or len(scenes) != export["requested_scenes"] or len({r["token"] for r in scenes}) != len(scenes)):
        raise ValueError("Export observation manifest changed")
    for rank in range(export["world_size"]):
        receipt = json.loads((root / f"COMPLETE_rank{rank}.json").read_text())
        assigned = scenes[rank::export["world_size"]]
        if (receipt["identity"] != export["identity"] or receipt["rank"] != rank
                or receipt["requested"] != len(assigned)
                or sorted(r["token"] for r in receipt["receipts"]) != sorted(r["token"] for r in assigned)):
            raise ValueError("Incomplete or wrong inference shard population")
        for row in receipt["receipts"]:
            meta = root / "predictions" / (row["token"] + ".json")
            if file_sha256(meta) != row["receipt_sha256"]:
                raise ValueError("Prediction receipt changed after shard completion")
    return export, scenes


def make_requests(root, export, scenes, entries, spec):
    root = Path(root)
    requests = []
    contexts = join_fixed_contexts(scenes, entries)
    runtime = fingerprint({k: export[k] for k in ("environment", "seed", "solver", "precision", "scene_noise")})
    for row, context in zip(scenes, contexts):
        path = root / "predictions" / (row["token"] + ".npz")
        meta = json.loads(path.with_suffix(".json").read_text())
        validate_cached_prediction(path, meta, identity=export["identity"], token=row["token"], log=row["log"])
        with np.load(path, allow_pickle=False) as payload:
            physical = payload["trajectory"].astype(np.float64)
        cache_path = Path(context["cache_path"]).resolve()
        if not any(cache_path.is_relative_to(Path(p).resolve()) for p in spec.cache_roots):
            raise ValueError("Metric context is outside the explicitly registered roots")
        cache_hash = file_sha256(cache_path)
        if context.get("sha256", cache_hash) != cache_hash:
            raise ValueError("Metric context changed after its original cache receipt")
        requests.append(ConsequenceRequest(spec.campaign_id, row["token"], row["log"], "single_policy",
            export["checkpoint"]["sha256"], export["checkpoint"]["sha256"], runtime,
            trajectory_sha256(physical), cache_hash, spec.score_protocol_hash, str(cache_path),
            physical.tolist(), "policy", fingerprint(export["checkpoint"]["deployment_source_contract"]),
            observe=True))
    return requests


def summarize(records, requests):
    if len(records) != len(requests) or not requests:
        raise ValueError("Cannot report a subset as the complete fixed evaluation")
    for record, request in zip(records, requests):
        if not record["valid"] or record["identity"] != request.identity():
            raise ValueError("Invalid score is not a valid zero or an ignorable sample")
        if set(record["metrics"]) != set(METRICS) or not all(record["metrics_valid"].get(k) for k in METRICS):
            raise ValueError("Incomplete canonical metric schema")
        if any(not math.isfinite(record["metrics"][k]) or not 0 <= record["metrics"][k] <= 1 for k in METRICS):
            raise ValueError("Nonfinite or out-of-range canonical score")
    def group(indices):
        return {"scenes": len(indices), "metrics": {
            key: float(np.mean([records[i]["metrics"][key] for i in indices])) for key in METRICS},
            "zero_count": sum(records[i]["metrics"]["score"] <= 1e-8 for i in indices)}
    logs = sorted({r.source_log for r in requests})
    return {"status": "COMPLETE", "score_units": "0..1", "requested_scenes": len(requests),
        "valid_scored": len(records), "invalid": 0, "unresolved": 0, "aggregate": group(range(len(records))),
        "by_log": {log: group([i for i, r in enumerate(requests) if r.source_log == log]) for log in logs},
        "official_single_candidate_reference": True, "deployment_candidates_per_scene": 1}


def run(args):
    if (not 1 <= args.workers <= 4 or args.max_calls < 1 or not 0 <= args.retry_attempts <= 2
            or not math.isfinite(args.max_seconds) or args.max_seconds <= 0):
        raise ValueError("Explicit bounded workers/calls/retries/time required")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("Run this reference scoring job with CUDA_VISIBLE_DEVICES='' ")
    if any(os.environ.get(k) != "1" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")):
        raise ValueError("Reference CPU workers require all three BLAS thread limits=1")
    export, scenes = read_complete_export(args.predictions)
    protocol = protocol_identity(args.navsim_root)
    spec = ConsequenceSpec(str(Path(args.navsim_root).resolve()), str(Path(args.maps_root).resolve()),
        tuple(str(Path(p).resolve()) for p in args.cache_root), args.run_id, protocol["sha256"], False)
    entries = json.loads(Path(args.metric_index).read_text())
    requests = make_requests(args.predictions, export, scenes, entries, spec)
    if len(requests) > args.max_calls:
        raise ValueError("Budget cannot cover the fixed requested scene population")
    identity = {"schema": "st_world_official_evaluation_v1", "export_identity": export["identity"],
        "export_file_sha256": file_sha256(Path(args.predictions) / "IDENTITY.json"),
        "metric_index": {"path": str(Path(args.metric_index).resolve()), "sha256": file_sha256(args.metric_index)},
        "spec": asdict(spec), "protocol": protocol, "request_identities": [r.identity() for r in requests],
        "max_calls": args.max_calls, "retry_attempts": args.retry_attempts, "workers": args.workers,
        "evaluation_purpose": "registered_train_or_development_only", "checkpoint": export["checkpoint"]}
    identity["identity"] = fingerprint(identity)
    root = Path(args.output); root.mkdir(parents=True, exist_ok=True)
    immutable_json(root / "IDENTITY.json", identity)
    with (root / "EVALUATION.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        from tools.st_world_policy.train_flow_policy import OfficialFeedbackLedger
        ledger = OfficialFeedbackLedger(root / "ledger", identity=identity["identity"],
                                        max_calls=args.max_calls, retry_attempts=args.retry_attempts)
        started = time.perf_counter(); records = []
        with ConsequenceService(spec, workers=args.workers, max_calls=args.max_calls) as service:
            # Fixed chunks survive retries and do not alter canonical reference context.
            for at in range(0, len(requests), 4):
                if time.perf_counter() - started >= args.max_seconds:
                    result = {"status": "PARTIAL", "requested_scenes": len(requests),
                              "processed_this_pass": len(records), "identity": identity["identity"]}
                    immutable_json(root / "attempts" / (str(time.time_ns()) + ".json"), result)
                    return result
                records.extend(ledger.score(requests[at:at + 4], policy_version=export["checkpoint"]["sha256"],
                                            service=service))
        result = summarize(records, requests)
        rows = [{"scene_id": request.scene_id, "source_log": request.source_log,
                 **record["metrics"]} for record, request in zip(records, requests)]
        csv_path = root / "SCENES.csv"
        buffer = io.StringIO(newline="")
        writer = csv.DictWriter(buffer, fieldnames=["scene_id", "source_log", *METRICS])
        writer.writeheader(); writer.writerows(rows)
        content = buffer.getvalue().encode()
        if csv_path.exists() and csv_path.read_bytes() != content:
            raise ValueError("Existing CSV is not the complete verified canonical population")
        if not csv_path.exists():
            temp = root / ("SCENES." + str(os.getpid()) + ".tmp")
            with temp.open("xb") as stream:
                stream.write(content); stream.flush(); os.fsync(stream.fileno())
            os.link(temp, csv_path); temp.unlink()
        result.update(identity=identity["identity"], scenes_csv_sha256=file_sha256(csv_path),
                      checkpoint_scope=export["checkpoint"]["scope"],
                      diagnostic_weights=not export["checkpoint"]["weights_usable_for_formal_init"])
        immutable_json(root / "COMPLETE.json", result)
        return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("predictions", "metric-index", "navsim-root", "maps-root", "run-id", "output"):
        p.add_argument("--" + key, required=True)
    p.add_argument("--cache-root", required=True, action="append")
    p.add_argument("--workers", type=int, required=True)
    p.add_argument("--max-calls", type=int, required=True)
    p.add_argument("--retry-attempts", type=int, default=1)
    p.add_argument("--max-seconds", type=float, required=True)
    try:
        result = run(p.parse_args())
    except Exception as exc:
        print(json.dumps({"status": "FAILED", "error": repr(exc)}), flush=True)
        return 2
    print(json.dumps({k: v for k, v in result.items() if k != "by_log"}), flush=True)
    return 0 if result["status"] == "COMPLETE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
