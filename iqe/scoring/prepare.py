"""Bounded, resumable per-log official cache construction; existing caches stay intact."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
import os
import subprocess
import time
from ..data.manifests import save_scenes, load_scenes
from ..io import atomic_json, digest, file_hash, read_json, BlockedError
from ..contracts import require


def prepare_contexts(pipeline, roles):
    require("dev_report" not in roles and set(roles) <= {"incremental_fit","stage_val","selector_cal"}, "prepare only allowed train-domain roles; independent dev already has official caches")
    selected = pipeline.scenes(roles)
    d = pipeline.config["data"]
    require(Path(d["raw_log_root"]).is_dir() and Path(d["map_root"]).is_dir(), "real logs/maps unavailable for official reference generation")
    folder = pipeline.root / "official_train_metric"
    identity = {"scene_ids": sorted(s.scene_id for s in selected), "roles": roles,
        "navsim_source_hash": pipeline.contract["metric"]["python_source_hash"], "source_commit": pipeline.contract["source_commit"],
        "raw_log_root": d["raw_log_root"], "map_root": d["map_root"], "original_caches_modified": False}
    groups = {}
    for s in selected: groups.setdefault(s.source_log_id, []).append(s.scene_id)
    identity["raw_log_hashes"] = {log: file_hash(Path(d["raw_log_root"]) / (log + ".pkl")) for log in groups}
    version = folder / digest(identity)
    atomic_json(version / "IDENTITY.json", identity, immutable=True)
    if (version / "COMPLETE.json").exists():
        for path in (version / "records").glob("*.json"):
            require(all(r["status"] == "ok" and file_hash(r["cache_path"]) == r["sha256"] for r in read_json(path)), "completed official cache changed")
        atomic_json(pipeline.root / "OFFICIAL_SCENES.json", {"path":str(version / "scenes.json"), "checksum":file_hash(version / "scenes.json")})
        return read_json(version / "COMPLETE.json") | {"reused":True}
    started = time.perf_counter()
    def one(item):
        log, tokens = item
        request, response = version / "requests" / (log + ".json"), version / "records" / (log + ".json")
        if response.exists():
            rows = read_json(response)
            if all(r["status"] == "ok" and Path(r["cache_path"]).is_file() and file_hash(r["cache_path"]) == r["sha256"] for r in rows):
                return rows
        payload = {"navsim_root": pipeline.config["metric"]["navsim_root"], "source_root": pipeline.contract["source_root"], "log": log, "tokens": tokens,
                   "args": {"raw_log_root":d["raw_log_root"], "map_root":d["map_root"], "output":str(version)}}
        atomic_json(request,payload)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
        repo = str(Path(__file__).resolve().parents[2])
        env["PYTHONPATH"] = repo + os.pathsep + str(Path(repo) / "nuplan-devkit")
        job = subprocess.run([pipeline.config["metric"]["python"], "-m", "iqe.scoring.prepare_worker", str(request), str(response)],
            env=env,capture_output=True,text=True,timeout=pipeline.config["metric"]["timeout_seconds"] * max(1,len(tokens)))
        if job.returncode or not response.exists():
            return [{"token":t,"log":log,"status":"failed","error":job.stderr[-2000:]} for t in tokens]
        return read_json(response)
    rows = []
    with ThreadPoolExecutor(max_workers=pipeline.config["execution"]["metric_cache_workers"]) as pool:
        for group in pool.map(one, groups.items()):
            rows.extend(group)
            atomic_json(version / "PROGRESS.json", {"completed":len(rows), "expected":len(selected), "failed":sum(r["status"] != "ok" for r in rows)})
    errors = [r for r in rows if r["status"] != "ok"]
    if errors:
        atomic_json(version / "FAILED.json", errors)
        raise BlockedError(f"official reference cache construction failed for {len(errors)} scenes; completed logs retained for resume")
    lookup = {r["token"]:r for r in rows}
    all_scenes = pipeline.scenes()
    fixed = [replace(s, metric_context_ref=lookup[s.scene_id]["cache_path"], metric_context_hash=lookup[s.scene_id]["sha256"]) if s.scene_id in lookup else s for s in all_scenes]
    save_scenes(version / "scenes.json",fixed)
    # A versioned pointer, never overwrite the original source manifest.
    atomic_json(pipeline.root / "OFFICIAL_SCENES.json", {"path":str(version / "scenes.json"), "checksum":file_hash(version / "scenes.json")})
    result = {"status":"COMPLETE", "scenes":len(rows), "roles":roles, "output":str(version), "wall_seconds":time.perf_counter()-started,
              "CPU_worker_limit":pipeline.config["execution"]["metric_cache_workers"], "original_caches_modified":False}
    atomic_json(version / "COMPLETE.json",result,immutable=True)
    return result
