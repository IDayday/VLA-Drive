"""One bounded, resumable experiment; no Base training, navtest or ACTIVE writes."""
from pathlib import Path
from datetime import datetime, timezone
import os
import subprocess
import time
from iqe.io import read_json, atomic_json, file_hash, lock
from iqe.contracts import require
from iqe.data.manifests import load_scenes


def run(config, device="cuda:0", wait_cache_seconds=0, from_stage="beginning"):
    # An operator may revise a not-yet-started research training budget while
    # immutable context/official scoring preparation continues independently.
    # This gate is outside every optimizer and never changes Base or sampler.
    hold_path = config.root/"runtime/HOLD_HEAD_TRAINING.json"
    hold_deadline = time.monotonic()+10800
    while hold_path.exists() and read_json(hold_path).get("active", False):
        atomic_json(config.root/"WORKFLOW_STAGE.json", {"stage": "WAIT_EXPLICIT_TRAINING_BUDGET", "status": "HELD",
                    "reason": read_json(hold_path).get("reason"), "optimizer_updates_started": False})
        require(time.monotonic() < hold_deadline, "bounded head-training hold expired")
        time.sleep(20)
    from .bank import build_initial
    from .training import train_expert, build_pair_bank, train_scorer
    from .reporting import evaluate, diagnostics, export_bundle
    from .budget import training_budget
    budget = training_budget(config)
    require(0 <= wait_cache_seconds <= 10800, "bounded cache wait required")
    require(from_stage in {"beginning", "scorer"}, "unknown recovery stage")
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
        require(policy.get("global_dispatch_prohibitions", {}).get(hostname, {}).get("new_task_dispatch_allowed", True),
                "host explicitly withdrawn")
        if from_stage == "scorer":
            # Revalidate complete scene/candidate/checkpoint/protocol identities;
            # a later-stage restart never fabricates skipped-stage completion.
            from .training import load_bank
            load_bank(config)
        live = {"pid": os.getpid(), "host": hostname,
                    "started_utc": datetime.now(timezone.utc).isoformat(), "experiment": config.experiment_id,
                    "scope": f"one expert{budget['expert_updates']} and scratch scorer{budget['scorer_updates']}, no navtest/ACTIVE", "status": "RUNNING",
                    "effective_training_budget": budget,
                    "from_stage": from_stage}
        atomic_json(config.root/"WORKFLOW_LIVE.json", live)
        if config.research_protocol == "two_candidate_v1":
            # Record the implementation actually imported after the offline
            # input wait, rather than only the queue launch's earlier commit.
            repo = Path(__file__).resolve().parents[2]
            atomic_json(config.root/"WORKFLOW_CODE_IDENTITY.json", {
                "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
                "config_hash": config.identity,
                "source_hashes": {name: file_hash(Path(__file__).with_name(name))
                                  for name in ("run.py", "training.py", "reporting.py")},
                "planned_training_interruptions": False,
                "expert_updates": budget["expert_updates"],
                "scorer_updates": budget["scorer_updates"],
                "pointwise_updates": budget["pointwise_updates"],
                "training_budget_amendment_hash": budget["amendment_hash"],
                "phase_switch_is_internal": True,
            }, immutable=True)
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
        if config.research_protocol == "two_candidate_v1":
            # This experiment runs each trainer continuously. Checkpoints and
            # recovery remain available for genuine interruptions; the internal
            # scoring phase switch does not stop or restart its optimizer.
            stages = [(name, operation) for name, operation in stages
                      if name not in {"expert_pause_250", "scorer_pause_500"}]
            # Keep the already owned GPU for the requested final online evidence
            # immediately after training. These are read-only evaluations; no
            # artificial pause/resume or additional model fitting is involved.
            from .reporting import audit_online
            from .latency import benchmark
            from .audit_training import audit as audit_training
            stages.extend((
                ("audit_actual_training", lambda: audit_training(config, device)),
                ("audit_current_input_labels", lambda: audit_online(config, device)),
                ("benchmark_online_bundle", lambda: benchmark(config, device)),
            ))
        if from_stage == "scorer":
            first_scorer = next(i for i, (name, _) in enumerate(stages)
                                if name in {"scorer_pause_500", "scorer_finish"})
            stages = stages[first_scorer:]
        for name, operation in stages:
            atomic_json(config.root/"WORKFLOW_STAGE.json", {"stage": name, "status": "RUNNING"})
            start = time.perf_counter()
            try:
                output = operation()
            except Exception as error:
                atomic_json(config.root/"WORKFLOW_STAGE.json", {"stage": name, "status": "FAILED", "error": repr(error)})
                atomic_json(config.root/"WORKFLOW_LIVE.json", {**live, "status": "FAILED", "stage": name,
                            "ended_utc": datetime.now(timezone.utc).isoformat(), "error": repr(error)})
                raise
            atomic_json(config.root/"workflow_stages"/(name+".json"), {"stage": name, "status": "COMPLETE",
                        "wall_seconds": time.perf_counter()-start, "result": output})
            if operator_paused(name, output):
                status = {"stage": name, "status": "STOPPED_AT_BOUNDARY", "result": output}
                atomic_json(config.root/"WORKFLOW_STAGE.json", status)
                atomic_json(config.root/"WORKFLOW_LIVE.json", {**live, **status,
                            "ended_utc": datetime.now(timezone.utc).isoformat()})
                return status
        atomic_json(config.root/"WORKFLOW_STAGE.json", {"stage": "ALL", "status": "COMPLETE"})
        atomic_json(config.root/"WORKFLOW_LIVE.json", {**live, "status": "COMPLETE",
                    "ended_utc": datetime.now(timezone.utc).isoformat()})
        return read_json(config.root/"VERDICT.json")


def operator_paused(stage, output):
    if output.get("termination", {}).get("termination_reason") != "OPERATOR_PAUSED":
        return False
    planned = {"expert_pause_250": 250, "scorer_pause_500": 500}.get(stage)
    return planned is None or output["optimizer_steps"] < planned


def _resume(folder):
    return "auto" if list(folder.glob("step_*.pt.COMPLETE.json")) else None


def _step(folder):
    return max([read_json(p)["optimizer_step"] for p in folder.glob("step_*.pt.COMPLETE.json")], default=0)
