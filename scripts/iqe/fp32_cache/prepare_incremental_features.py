"""Consume committed per-log contexts using the locked IQE feature-cache path.

This scheduling helper overlaps official context preparation with frozen 22k
feature extraction. It never publishes OFFICIAL_SCENES, changes the scene set,
trains a model, or treats a partially built official bank as complete.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def check_gpu_owners(device, *, gpu_rows=None, process_rows=None):
    """Recheck after slow shared checkpoint I/O; never compete with a late job."""
    if not device.startswith("cuda"):
        return
    from iqe.contracts import require
    if gpu_rows is None:
        gpu_rows = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid",
                                            "--format=csv,noheader"], text=True)
    if process_rows is None:
        process_rows = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid",
                                                "--format=csv,noheader"], text=True)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0].strip()
    gpu_map = {i.strip(): uuid.strip() for i, uuid in (line.split(",") for line in gpu_rows.splitlines() if line.strip())}
    uuid = gpu_map.get(visible, visible)
    foreign = [int(pid.strip()) for pid, gpu in (line.split(",") for line in process_rows.splitlines() if line.strip())
               if gpu.strip() == uuid and int(pid.strip()) != os.getpid()]
    require(not foreign, f"BLOCKED_GPU_RESOURCES: late unrelated GPU processes {foreign}; cache receipts retained")


def committed_contexts(version, scenes, *, receipt_hashes=None):
    """Read only atomically committed, complete per-log receipts.

    Records for an unfinished/failed log remain pending. Invalid identities or
    mutation of an already consumed receipt fail closed, including on resume.
    Individual cache bytes are checked immediately before their use below.
    """
    from iqe.contracts import require
    from iqe.io import digest, file_hash, read_json

    version = Path(version).resolve()
    identity = read_json(version / "IDENTITY.json")
    require(digest(identity) == version.name, "official context version identity mismatch")
    by_id = {s.scene_id: s for s in scenes}
    require(len(by_id) == len(scenes), "duplicate requested scene")
    require(set(identity["scene_ids"]) == set(by_id), "context/scene manifest mismatch")
    require(all(s.split_role in identity["roles"] for s in scenes), "context role mismatch")
    require(set(identity["roles"]) <= {"incremental_fit", "stage_val", "selector_cal"},
            "training-domain contexts only")
    logs = {}
    for scene in scenes:
        logs.setdefault(scene.source_log_id, set()).add(scene.scene_id)
    known = receipt_hashes if receipt_hashes is not None else {}
    for log in known:
        require((version / "records" / (log + ".json")).is_file(), "consumed context receipt removed")
    ready = []
    for log in sorted(logs):
        path = version / "records" / (log + ".json")
        if not path.exists():
            continue
        checksum = file_hash(path)
        require(log not in known or known[log] == checksum, "consumed context receipt changed")
        rows = read_json(path)
        require(isinstance(rows, list) and rows, "invalid per-log receipt")
        require(len(rows) == len(logs[log]) and {r["token"] for r in rows} == logs[log],
                "per-log receipt missing/duplicate/unexpected scene")
        require(all(r["log"] == log for r in rows), "cross-log context injection")
        if any(r["status"] != "ok" for r in rows):
            continue
        for row in rows:
            require(row.get("official_reference_hash") and row.get("sha256"),
                    "official reference/hash absent")
            cache = Path(row["cache_path"]).resolve()
            require(cache.is_relative_to(version / "cache"), "context outside version")
            ready.append(replace(by_id[row["token"]], metric_context_ref=str(cache),
                                 metric_context_hash=row["sha256"]))
        known[log] = checksum
    return sorted(ready, key=lambda s: s.scene_id), known


def main(argv=None):
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--source-repo", required=True, help="unchanged qualified 22k source checkout")
    p.add_argument("--source-commit", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--metric-version", required=True)
    p.add_argument("--progress-root", required=True)
    p.add_argument("--shard-index", type=int, required=True)
    p.add_argument("--num-shards", type=int, required=True)
    p.add_argument("--max-samples", type=int, required=True, help="same explicit per-role cap as formal CLI")
    p.add_argument("--max-new-scenes", type=int, required=True)
    p.add_argument("--reserve-gib", type=float, required=True)
    p.add_argument("--max-seconds", type=int, required=True)
    p.add_argument("--device", choices=("cpu", "cuda"), required=True)
    p.add_argument("--wait-seconds", type=int, default=0, help="exit and release GPU after draining available logs by default")
    a = p.parse_args(argv)
    if not (0 <= a.shard_index < a.num_shards and min(a.max_samples, a.max_new_scenes, a.max_seconds) > 0
            and a.reserve_gib > 0 and a.wait_seconds >= 0):
        p.error("positive bounded budgets and valid shard required")
    source = Path(a.source_repo).resolve()
    if subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip() != a.source_commit:
        p.error("locked source checkout moved")
    sys.path.insert(0, str(source))
    from iqe.config import load_config
    from iqe.contracts import require
    from iqe.io import atomic_json, digest, file_hash, read_json
    from iqe.resources import qualify
    from iqe.pipeline import Pipeline
    import torch

    require(Path(sys.modules["iqe"].__file__).resolve().parent == source / "iqe", "wrong imported IQE source")
    root = Path(a.progress_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock = open(root / (f"shard_{a.shard_index}.lock"), "a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    resources = qualify(a.device)
    torch.set_num_threads(2)
    config = load_config(a.config)
    pipeline = Pipeline(config, mode="full", max_samples=a.max_samples, device=a.device)
    pipeline.use_locked_base()
    scenes = pipeline.scenes(["incremental_fit", "stage_val", "selector_cal"])
    version = Path(a.metric_version).resolve()
    official = read_json(version / "IDENTITY.json")
    require(official["navsim_source_hash"] == pipeline.contract["metric"]["python_source_hash"],
            "scoring context source differs from locked Base")
    require(official["source_commit"] == pipeline.contract["source_commit"], "context framework source mismatch")
    identity = {"source_commit": a.source_commit, "config_file_hash": file_hash(a.config),
                "lock_hash": file_hash(pipeline.root / "LOCKED_S0.json"),
                "original_manifest_hash": file_hash(config["data"]["scene_manifest"]),
                "context_identity": digest(official), "roles": official["roles"],
                "scene_ids_hash": digest(sorted(s.scene_id for s in scenes)),
                "num_shards": a.num_shards, "shard_index": a.shard_index,
                "feature_function": "original Pipeline.cache_scene_list, unchanged batch=1"}
    state_path = root / (f"shard_{a.shard_index}.json")
    prior = read_json(state_path) if state_path.exists() else {}
    require(not prior or prior["identity"] == identity, "resume feature worker identity changed")
    receipts = dict(prior.get("context_receipt_hashes", {}))
    owned = {s.scene_id for i, s in enumerate(sorted(scenes, key=lambda s: s.scene_id)) if i % a.num_shards == a.shard_index}
    done, written, reused = set(), 0, 0
    started = time.monotonic()
    attempt = root / "attempts" / f"{time.time_ns()}_shard_{a.shard_index}.json"
    model = None
    status = "RUNNING"

    def save():
        result = {"status": status, "identity": identity, "resources": resources,
                  "dispatcher_source_hash": file_hash(__file__),
                  "pid": os.getpid(), "context_receipt_hashes": receipts,
                  "requested_shard_scenes": len(owned), "verified_features": len(done),
                  "new_features": written, "reused_features": reused,
                  "elapsed_seconds": time.monotonic() - started, "updated_unix": time.time(),
                  "feature_root": str(pipeline.features.root), "official_pointer_modified": False,
                  "new_training_started": False}
        atomic_json(state_path, result)
        atomic_json(attempt, result)
        return result

    save()
    try:
        while True:
            ready, receipts = committed_contexts(version, scenes, receipt_hashes=receipts)
            pending = [s for s in ready if s.scene_id in owned and s.scene_id not in done]
            for scene in pending:
                if time.monotonic() - started >= a.max_seconds or written >= a.max_new_scenes:
                    status = "BUDGET_PAUSED"
                    print(json.dumps(save()))
                    return 3
                require(file_hash(scene.metric_context_ref) == scene.metric_context_hash, "official context checksum changed")
                record = pipeline.feature_record(scene)
                try:
                    pipeline.features.get(record)
                    reused += 1
                except FileNotFoundError:
                    require(shutil.disk_usage(pipeline.root).free > (a.reserve_gib + 1) * 2**30,
                            "BLOCKED_SHARED_STORAGE_RESERVE")
                    if model is None:
                        check_gpu_owners(a.device)
                        status = "LOADING_FROZEN_BASE"
                        save()
                        model = pipeline.model()
                        require(not any(p.requires_grad for p in model.parameters()), "frozen cache model is trainable")
                        check_gpu_owners(a.device)
                        status = "RUNNING"
                        save()
                    if written % 16 == 0:
                        check_gpu_owners(a.device)
                    pipeline.cache_scene_list(model, [scene])
                    # Read through the exact downstream API, including checksum.
                    pipeline.features.get(record)
                    written += 1
                done.add(scene.scene_id)
                if len(done) % 16 == 0:
                    save()
            if (version / "COMPLETE.json").exists():
                require(len(done) == len(owned), "official COMPLETE does not cover requested feature shard")
                final = read_json(pipeline.root / "OFFICIAL_SCENES.json")
                require(Path(final["path"]).resolve() == version / "scenes.json", "official final pointer/version mismatch")
                final_scenes = pipeline.scenes(["incremental_fit", "stage_val", "selector_cal"])
                current = {s.scene_id: pipeline.feature_record(s).cache_key for s in ready}
                require(all(current[s.scene_id] == pipeline.feature_record(s).cache_key for s in final_scenes),
                        "staged feature identity differs from final official scene manifest")
                status = "COMPLETE"
                print(json.dumps(save()))
                return 0
            if not a.wait_seconds or time.monotonic() - started >= a.max_seconds:
                status = "WAITING_OFFICIAL_INPUTS"
                print(json.dumps(save()))
                return 3
            status = "WAITING_OFFICIAL_INPUTS"
            save()
            time.sleep(min(a.wait_seconds, 60))
            status = "RUNNING"
    except Exception as error:
        status = "FAILED"
        result = save() | {"error": repr(error)}
        atomic_json(attempt, result)
        print(json.dumps(result), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
