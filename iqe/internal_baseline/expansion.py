"""Expand the mining population without changing its Base, roles, or thresholds.

Only immutable joint contexts are imported. Official score receipts include
run-local request identities and are rebuilt through the reference backend.
"""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
import argparse
import json
import os
import time

from iqe.contracts import require
from iqe.data.manifests import load_scenes
from iqe.io import atomic_json, digest, file_hash, lock, read_json
from .config import load_config
from .features import JointCache, cache_key
from .partition import subrole_ids


def reuse_context(source, destination, workers=2):
    """Verify every shared scene and payload before hard-linking immutable data.

    The full SceneRecord matters: joint files also contain supervised targets,
    although these never enter the deployment context whitelist.
    """
    require(1 <= workers <= 4, "bounded verification concurrency required")
    require(source.root.resolve() != destination.root.resolve(), "distinct experiment roots required")
    from .binding import binding
    old, new = binding(source), binding(destination)
    for field in ("checkpoint_hash", "framework_contract_hash", "contract_hash", "inference_runtime"):
        require(old[field] == new[field], "context reuse Base/runtime mismatch: " + field)
    for cfg, bound in ((source, old), (destination, new)):
        require(file_hash(cfg.root/"scenes.json") == bound["scene_manifest_hash"], "scene manifest changed")
    old_partition = read_json(source.root/"DATA_PARTITION.json")
    new_partition = read_json(destination.root/"DATA_PARTITION.json")
    require(old_partition["all_assignments"] == new_partition["all_assignments"], "source group assignments changed")
    old_rows = {s.scene_id: s for s in load_scenes(source.root/"scenes.json")}
    new_rows = {s.scene_id: s for s in load_scenes(destination.root/"scenes.json")}
    shared = sorted(old_rows.keys() & new_rows.keys())
    require(shared, "no reusable observations")
    # Validate the entire overlap before writing anything to the destination.
    for scene_id in shared:
        require(asdict(old_rows[scene_id]) == asdict(new_rows[scene_id]), "scene/target/provenance changed: " + scene_id)
        require(cache_key(source, old_rows[scene_id]) == cache_key(destination, new_rows[scene_id]),
                "joint feature contract changed: " + scene_id)
    src, dst = JointCache(source), JointCache(destination)
    inputs = {"source_binding": file_hash(source.root/"BASE_BINDING.json"),
              "destination_binding": file_hash(destination.root/"BASE_BINDING.json"),
              "source_manifest": old["scene_manifest_hash"], "destination_manifest": new["scene_manifest_hash"],
              "scene_ids_hash": digest(shared), "feature_code": file_hash(Path(__file__).with_name("features.py"))}
    progress = destination.root/"CONTEXT_REUSE_PROGRESS.json"

    def import_one(scene_id):
        s, d = old_rows[scene_id], new_rows[scene_id]
        require(src.valid(s), "source context missing: " + scene_id)
        receipt = read_json(str(src.path(s))+".COMPLETE.json")
        path = dst.path(d)
        path.parent.mkdir(parents=True, exist_ok=True)
        with lock(str(path)+".lock"):
            if path.exists():
                require(file_hash(path) == receipt["checksum"], "destination context conflict: " + scene_id)
            else:
                # Same project filesystem: link creation is atomic, and neither
                # writer ever mutates the contents of an accepted payload.
                os.link(src.path(s), path)
            atomic_json(str(path)+".COMPLETE.json", receipt, immutable=True)
        require(dst.valid(d), "destination context verification failed")
        return {"scene_id": scene_id, "scene_hash": digest(asdict(d)), "key": receipt["key"],
                "checksum": receipt["checksum"], "bytes": path.stat().st_size}

    with lock(destination.root/"context_reuse.lock"):
        rows = []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for result in executor.map(import_one, shared):
                rows.append(result)
                if len(rows) % 256 == 0:
                    atomic_json(progress, {"status": "RUNNING", "completed": len(rows), "total": len(shared), "inputs": inputs})
        report = {"schema": "verified_joint_context_reuse_v1", "status": "COMPLETE", "inputs": inputs,
                  "source_root": str(source.root), "destination_root": str(destination.root),
                  "records": rows, "count": len(rows), "logical_bytes": sum(r["bytes"] for r in rows),
                  "storage": "same_filesystem_immutable_hardlinks", "scores_imported": False}
        atomic_json(destination.root/"CONTEXT_REUSE.json", report, immutable=True)
        atomic_json(progress, {"status": "COMPLETE", "completed": len(rows), "total": len(shared), "inputs": inputs})
        return {k: v for k, v in report.items() if k != "records"}


def score_ready(config, *, max_seconds=43200, chunk_size=512):
    """Pipeline reference scoring behind GPU extraction; all fit is covered.

    D_S labels are computed but never determine the expert's manifest. GT audit
    here only uses D_E/stage_val. Final hard/ordinary construction is unchanged.
    Each chunk uses verified score receipts, so a genuine restart only supplies
    missing labels instead of replacing the sampler or an experiment identity.
    """
    require(0 < max_seconds <= 86400 and 1 <= chunk_size <= 1024, "bounded scoring/wait budget required")
    from .bank import cache_candidates, score_candidates, failure_and_target_audit
    from .binding import binding
    binding(config)
    scenes = load_scenes(config.root/"scenes.json")
    de, ds = subrole_ids(config, "expert_fit"), subrole_ids(config, "scorer_fit")
    require(de.isdisjoint(ds), "expert/scorer leakage")
    # Expert preparation first; independently score the entire D_S afterwards.
    rows = sorted(scenes, key=lambda s: (0 if s.scene_id in de else 1 if s.split_role == "stage_val" else 2, s.scene_id))
    done, failures = set(), {"expert_fit": 0, "scorer_fit": 0, "stage_val": 0, "selector_cal": 0, "dev_report": 0}
    started, batches = time.monotonic(), 0
    status_path = config.root/"runtime/BASE_PREFIX_STATUS.json"
    with lock(config.root/"reference_prefix.lock"):
        while len(done) < len(rows):
            require(time.monotonic()-started < max_seconds, "bounded reference preparation deadline; partial receipts retained")
            ready = [s for s in rows if s.scene_id not in done and (config.root/"features"/(s.scene_id+".pt.COMPLETE.json")).is_file()]
            status = {"status": "RUNNING", "completed": len(done), "total": len(rows), "ready": len(ready),
                      "base_failures": failures, "elapsed_seconds": time.monotonic()-started, "batches": batches}
            if len(ready) < chunk_size and len(done)+len(ready) < len(rows):
                atomic_json(status_path, {**status, "status": "WAITING_FEATURES"})
                time.sleep(10)
                continue
            subset = ready[:chunk_size]
            labels = score_candidates(config, subset, cache_candidates(config, subset, "base"), f"all_fit_base_{batches:05d}")
            audit = []
            for scene, label in zip(subset, labels):
                role = "expert_fit" if scene.scene_id in de else "scorer_fit" if scene.scene_id in ds else scene.split_role
                if failure_and_target_audit(asdict(label))["failure"]:
                    failures[role] += 1
                    if role in {"expert_fit", "stage_val"}:
                        audit.append(scene)
            if audit:
                score_candidates(config, audit, cache_candidates(config, audit, "gt"), f"all_fit_gt_{batches:05d}")
            done.update(s.scene_id for s in subset)
            batches += 1
            atomic_json(status_path, {**status, "completed": len(done), "batches": batches,
                                      "base_failures": failures, "elapsed_seconds": time.monotonic()-started})
        result = {"status": "COMPLETE", "completed": len(done), "total": len(rows), "base_failures": failures,
                  "elapsed_seconds": time.monotonic()-started, "Base_hash": read_json(config.root/"BASE_BINDING.json")["checkpoint_hash"],
                  "partition_hash": file_hash(config.root/"DATA_PARTITION.json"), "D_S_guides_expert": False}
        atomic_json(status_path, result)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    reuse = sub.add_parser("reuse-context", help="strict verified same-Base context import")
    reuse.add_argument("--source-config", required=True)
    reuse.add_argument("--config", required=True)
    reuse.add_argument("--workers", type=int, default=2)
    score = sub.add_parser("score-stream", help="bounded official scoring of the full bound manifest")
    score.add_argument("--config", required=True)
    score.add_argument("--max-seconds", type=int, default=43200)
    score.add_argument("--chunk-size", type=int, default=512)
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.command == "reuse-context":
        result = reuse_context(load_config(args.source_config), cfg, args.workers)
    else:
        result = score_ready(cfg, max_seconds=args.max_seconds, chunk_size=args.chunk_size)
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
