"""One bounded, resumable experiment; no Base training, navtest or ACTIVE writes."""
from pathlib import Path
from datetime import datetime, timezone
import os
import time
from iqe.io import read_json, atomic_json, file_hash, lock
from iqe.contracts import require
from iqe.data.manifests import load_scenes


def run(config, device="cuda:0", wait_cache_seconds=0):
    from .bank import build_initial
    from .training import train_expert, build_pair_bank, train_scorer
    from .reporting import evaluate, diagnostics, export_bundle
    require(0 <= wait_cache_seconds <= 10800, "bounded cache wait required")
    # One workflow owner for this lineage; independent stage commands are for
    # recovery after the workflow has exited, not concurrent mutation.
    import fcntl
    path = config.root/"RUN.lock"
    with open(path, "a+") as guard:
        try:
            fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as e:
            raise ValueError("same experiment already has an active workflow") from e
        policy = read_json("/mnt/project/server_dispatch_policy.json")
        hostname = os.uname().nodename.removesuffix("-worker-0")
        require(hostname in policy["task_authorizations"]["iqe_v1"]["allowed_hosts"], "host outside task allowlist")
        atomic_json(config.root/"WORKFLOW_LIVE.json", {"pid": os.getpid(), "host": hostname,
                    "started_utc": datetime.now(timezone.utc).isoformat(), "experiment": config.experiment_id,
                    "scope": "one expert1000 and scratch scorer2000, no navtest/ACTIVE", "status": "RUNNING"})
        rows = [s for s in load_scenes(config.root/"scenes.json") if s.split_role in {"incremental_fit", "stage_val"}]
        deadline = time.monotonic()+wait_cache_seconds
        while True:
            missing = [s.scene_id for s in rows if not (config.root/"features"/(s.scene_id+".pt.COMPLETE.json")).exists()]
            if not missing: break
            atomic_json(config.root/"WORKFLOW_STAGE.json", {"stage": "WAIT_FEATURES", "missing_fit_val": len(missing)})
            require(time.monotonic() < deadline, "WAITING_INPUTS: joint fit/val cache incomplete")
            time.sleep(min(30, max(0, deadline-time.monotonic())))
        # Receipts are only a readiness hint above. Each consumer verifies the
        # complete content/contract hashes before any record can be used.
        stages = [("build_hard", lambda: build_initial(config)),
                  ("expert_pause_250", lambda: train_expert(config, device, _resume(config.root/"expert"), 250)
                   if _step(config.root/"expert") < 250 else {"reused": True}),
                  ("expert_finish", lambda: train_expert(config, device, _resume(config.root/"expert"))),
                  ("pair_bank", lambda: build_pair_bank(config, device)),
                  ("scorer_pause_500", lambda: train_scorer(config, device, _resume(config.root/"scorer"), 500)
                   if _step(config.root/"scorer") < 500 else {"reused": True}),
                  ("scorer_finish", lambda: train_scorer(config, device, _resume(config.root/"scorer"))),
                  ("evaluate", lambda: evaluate(config, device)),
                  ("diagnostics", lambda: diagnostics(config, device)),
                  ("export_diagnostic_bundle", lambda: export_bundle(config, device))]
        for name, operation in stages:
            atomic_json(config.root/"WORKFLOW_STAGE.json", {"stage": name, "status": "RUNNING"})
            start = time.perf_counter()
            try:
                output = operation()
            except Exception as error:
                atomic_json(config.root/"WORKFLOW_STAGE.json", {"stage": name, "status": "FAILED", "error": repr(error)})
                raise
            atomic_json(config.root/"workflow_stages"/(name+".json"), {"stage": name, "status": "COMPLETE",
                        "wall_seconds": time.perf_counter()-start, "result": output})
        atomic_json(config.root/"WORKFLOW_STAGE.json", {"stage": "ALL", "status": "COMPLETE"})
        return read_json(config.root/"VERDICT.json")


def _resume(folder):
    return "auto" if list(folder.glob("step_*.pt.COMPLETE.json")) else None


def _step(folder):
    return max([read_json(p)["optimizer_step"] for p in folder.glob("step_*.pt.COMPLETE.json")], default=0)
