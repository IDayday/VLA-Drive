"""Persistent imports; fresh official cache, observation, simulator and scorer per call.

Run in the existing NAVSIM Python environment, never in the GPU loss process.
"""
from pathlib import Path
import argparse
import sys
import time
from iqe.io import read_json, atomic_json, file_hash, digest, lock
from iqe.contracts import require


def score_item(job, item, official_single):
    response = Path(item["response"])
    with lock(str(response) + ".SCORING.lock"):
        payload = item["request"]
        require(file_hash(payload["scene"]["metric_context_ref"]) == payload["scene"]["metric_context_hash"], "context changed")
        require(file_hash(payload["candidate"]["trajectory_ref"]) == payload["candidate"]["trajectory_hash"], "trajectory changed")
        key = digest(payload)
        receipt = response.with_suffix(".COMPLETE.json")
        if receipt.exists():
            r = read_json(receipt)
            require(r["request_hash"] == key and r["checksum"] == file_hash(response), "stale score receipt")
            require(read_json(response)["score_valid"], "invalid score must not have complete receipt")
            return True
        attempts = response.parent / "attempts" / response.stem
        request = attempts / "REQUEST.json"
        atomic_json(request, payload, immutable=True)
        success = False
        start_index = len(list(attempts.glob("attempt_*.json")))
        for attempt in range(start_index, start_index + job["attempts_per_invocation"]):
            record = attempts / f"attempt_{attempt:04d}.json"
            t = time.time()
            try:
                official_single(str(request), str(response))
                value = read_json(response)
                require(value["score_valid"], value.get("error_reason", "invalid official label"))
                atomic_json(record, {"status": "VALID_SCORED", "at": t, "response_hash": file_hash(response)}, immutable=True)
                atomic_json(receipt, {"status": "COMPLETE", "request_hash": key, "checksum": file_hash(response)}, immutable=True)
                success = True
                break
            except Exception as e:
                atomic_json(record, {"status": "SCORING_FAILURE", "at": t, "error": repr(e)}, immutable=True)
        return success


def run(job_path, shard, shards, worker_id=None):
    job = read_json(job_path)
    require(0 <= shard < shards <= 64, "bounded official worker shards")
    status_name = str(shard) if worker_id is None else worker_id
    require(status_name.replace("_", "").replace("-", "").isalnum(), "safe score worker identity")
    atomic_json(Path(job_path).parent / "COOPERATIVE_PROTOCOL.json", {
        "schema": "record_locked_official_score_v1", "job_hash": file_hash(job_path),
        "worker_code_hash": file_hash(__file__), "lock_suffix": ".SCORING.lock",
        "official_numeric_function_changed": False}, immutable=True)
    sys.path.insert(0, job["nuplan_root"])
    sys.path.insert(0, job["navsim_root"])
    import nuplan
    require(Path(nuplan.__file__).resolve().is_relative_to(Path(job["nuplan_root"]).resolve()), "wrong NuPlan runtime")
    from iqe.scoring.worker import main as official_single
    done, failed = 0, []
    started = time.perf_counter()
    for item in job["items"][shard::shards]:
        success = score_item(job, item, official_single)
        if success:
            done += 1
        else:
            failed.append(item["request"]["scene"]["scene_id"])
        if (done+len(failed)) % 25 == 0:
            atomic_json(Path(job_path).parent / f"worker_{status_name}.json", {"completed": done, "failed": failed,
                        "seconds": time.perf_counter()-started, "status": "RUNNING"})
    atomic_json(Path(job_path).parent / f"worker_{status_name}.json", {"completed": done, "failed": failed,
                "seconds": time.perf_counter()-started, "status": "COMPLETE" if not failed else "PARTIAL"})
    require(not failed, f"unresolved official scoring failures: {failed[:8]}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("job"); p.add_argument("--shard", type=int, required=True); p.add_argument("--shards", type=int, required=True)
    p.add_argument("--worker-id")
    a = p.parse_args(); run(a.job, a.shard, a.shards, a.worker_id)
