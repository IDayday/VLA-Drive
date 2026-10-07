"""Finite A/V 80k/90k/100k queue, using the unchanged native scoring adapters.

Preservation runs independently of the slow checkpoint hashing/export worker.
Only complete exact checkpoints are linked into external evaluation artifacts;
no trainer files, processes or save configuration are changed.
"""
import argparse
import copy
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

from starVLA.model.modules.vehicle_joint.initialization import identity_hash
from tools.full_foresight.navtest_schedule import atomic, busy, lease, preserve, read, sha, source_identity
from tools.planning_interface_transfer import evaluate_fixed_navtest as fixed

MODULE = "tools.planning_interface_transfer.watch_navtest"
UPDATES = [80000, 90000, 100000]
TERMINAL = {"COMPLETE", "FAILED", "MISSED_CHECKPOINT"}


def register(base_registration, output):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Never overwrite an observer registration")
    base = fixed.load(base_registration)
    c = base["config"]
    if c["update"] != 75000 or {m["arm"] for m in c["models"]} != fixed.ARMS:
        raise ValueError("Use the registered exact75k three-arm evaluation as the protocol reference")
    models = []
    for m in c["models"]:
        run = Path(m["training_run"])
        training = read(run/"identity.json")
        original = run.parent.parent/"registrations"/(run.name+".json")
        if (training["updates"] != 100000 or training["world_size"] != 8
                or training["global_batch"] != 32 or training["scope"] != "formal"
                or sha(original) != training["registration_sha256"]
                or read(original)["arm"] != m["arm"]):
            raise ValueError("Wrong formal training population/protocol")
        model = {k:v for k,v in m.items() if k not in ("checkpoint", "tag")}
        model["run_identity"] = training["identity"]
        model["training_source_sha"] = training["source_sha"]
        models.append(model)
    r = dict(schema="planning_interface_navtest_queue_v1", updates=UPDATES,
        models=models, root=str(output.parent), source_worktree=str(Path.cwd()),
        source_sha=source_identity(Path.cwd()), canonical_hostname=socket.gethostname(),
        base_registration=str(Path(base_registration).resolve()), base_identity=base["identity"],
        protocol=copy.deepcopy(c), poll_seconds=5, worker_python=sys.executable,
        deadline_unix=time.time()+48*3600, evaluation_gpu_hours_limit=144,
        per_task_gpu_hours_limit=16, total_tasks=9,
        authorization="User explicitly requested all three A/V arms at exact80k/90k/100k after75k; no test-based recipe selection.",
        asset_hashes={**base["asset_hashes"], str(Path(base_registration).resolve()):sha(base_registration)})
    r["identity"] = identity_hash(r)
    atomic(output, r)
    return r


def load(path):
    r = read(path)
    if (r["schema"] != "planning_interface_navtest_queue_v1" or r["updates"] != UPDATES
            or {m["arm"] for m in r["models"]} != fixed.ARMS
            or r["identity"] != identity_hash({k:v for k,v in r.items() if k != "identity"})
            or source_identity(r["source_worktree"]) != r["source_sha"]):
        raise ValueError("Queue registration/source changed")
    for p, expected in r["asset_hashes"].items():
        if sha(p) != expected:
            raise ValueError("Registered asset changed: "+p)
    if socket.gethostname() != r["canonical_hostname"]:
        raise ValueError("Wrong observer/CPU host")
    return r


def capture(r):
    """Quickly link every requested checkpoint even while evaluation is busy."""
    states = {}
    for m in r["models"]:
        progress = read(Path(m["training_run"])/"status.json")
        if progress["identity"] != m["run_identity"]:
            raise ValueError("Training status identity changed")
        for update in UPDATES:
            key = f"{m['arm']}_{update:06d}"; job = Path(r["root"])/"jobs"/key
            state = read(job/"status.json") if (job/"status.json").exists() else dict(status="WAITING")
            if state["status"] not in TERMINAL and not (job/"snapshot.json").exists():
                try:
                    snapshot = preserve(m, update, job)
                    if snapshot is not None:
                        state.update(status="QUEUED", preserved_unix=snapshot["preserved_unix"])
                    elif progress["completed"] >= update+200:
                        state.update(status="MISSED_CHECKPOINT", error="Exact checkpoint missing; never substitute another step")
                except FileNotFoundError:
                    # GC/save race: retry on the next poll, never use a partial snapshot.
                    state.update(status="WAITING", last_error="Checkpoint publication/retention race; retry")
                except Exception as error:
                    state.update(status="FAILED", error=repr(error))
                atomic(job/"status.json", state)
            states[key] = state["status"]
    return states


def task(path, key):
    r = load(path); root = Path(r["root"])
    pairs = {f"{m['arm']}_{u:06d}":(m,u) for m in r["models"] for u in UPDATES}
    if key not in pairs:
        raise ValueError("Task not registered")
    m, update = pairs[key]; job = root/"jobs"/key
    with lease(root/"executor.lock"):
        with lease(job/"worker.lock"):
            if (job/"result.json").exists():
                return
            atomic(job/"status.json", dict(status="RUNNING", phase="VALIDATE_EXACT_CHECKPOINT", pid=os.getpid()))
            try:
                snapshot = read(job/"snapshot.json")
                config = copy.deepcopy(r["protocol"])
                config.update(update=update, gpu_hours_limit=r["per_task_gpu_hours_limit"],
                              authorization=r["authorization"])
                config["models"] = [{k:v for k,v in m.items() if k not in ("run_identity", "training_source_sha")}]
                config["models"][0].update(training_run=snapshot["training_run"], tag=snapshot["tag"],
                                             source_training_run=m["training_run"])
                request = job/"request.json"; output = job/"evaluation/registration.json"
                if not output.exists():
                    atomic(request, config)
                    fixed.prepare(request, output)
                atomic(job/"status.json", dict(status="RUNNING", phase="EXPORT_AND_OFFICIAL_SCORE", pid=os.getpid()))
                fixed.run(output)
                result = read(job/"evaluation"/m["arm"]/"result.json")
                atomic(job/"result.json", result)
                atomic(job/"status.json", dict(status="COMPLETE", completed_unix=time.time(), optimizer_updates=0))
            except BaseException as error:
                atomic(job/"status.json", dict(status="FAILED", error=repr(error), optimizer_updates=0))
                raise


def watch(path, once=False):
    path = Path(path).resolve(); r = load(path); root = Path(r["root"])
    with lease(root/"observer.lock"):
        children = []
        while True:
            states = capture(r)
            stopped = (root/"STOP_SCHEDULING").exists() or time.time() >= r["deadline_unix"]
            prior = Path(r["protocol"]["root"])/"status.json"
            # Preserve while75k runs, but avoid simultaneous model-load/scoring waves.
            previous_complete = prior.exists() and read(prior)["status"] == "COMPLETE"
            orphan = any(v == "RUNNING" for v in states.values()) and not busy(root/"executor.lock")
            if previous_complete and not stopped and not orphan and not busy(root/"executor.lock") and not children:
                for key, status in sorted(states.items(), key=lambda kv:(int(kv[0].rsplit("_",1)[1]), kv[0])):
                    job = root/"jobs"/key
                    if status != "QUEUED" or busy(job/"worker.lock"):
                        continue
                    command = [r["worker_python"], "-u", "-m", MODULE, "task", "--registration", str(path), "--task", key]
                    with (job/f"worker_{time.time_ns()}.log").open("x") as stream:
                        child = subprocess.Popen(command, cwd=r["source_worktree"], env=fixed.environment(),
                            stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                    children.append((child, job))
                    atomic(job/"launch.json", dict(pid=child.pid, command=command, started_unix=time.time()))
                    break
            for child, job in list(children):
                code = child.poll()
                if code is None:
                    continue
                children.remove((child,job))
                if code and read(job/"status.json")["status"] not in TERMINAL:
                    atomic(job/"status.json", dict(status="FAILED", exit_code=code))
            states = {key:read(root/"jobs"/key/"status.json")["status"] for key in states}
            results = {p.parent.name:read(p) for p in (root/"jobs").glob("*/result.json")}
            finished = all(v in TERMINAL for v in states.values())
            atomic(root/"SUMMARY.json", results)
            atomic(root/"status.json", dict(status=("COMPLETE" if all(v=="COMPLETE" for v in states.values())
                else "FINISHED_WITH_FAILURES") if finished else "PAUSED" if stopped or orphan else "RUNNING",
                registration=r["identity"], pid=os.getpid(), tasks=states, completed_tasks=len(results),
                awaiting75k=not previous_complete, orphan_requires_inspection=orphan,
                optimizer_updates=0, updated_unix=time.time()))
            if once or finished or (stopped and not children):
                return
            time.sleep(r["poll_seconds"])


def main():
    p = argparse.ArgumentParser(__doc__); sub = p.add_subparsers(dest="mode", required=True)
    q = sub.add_parser("register"); q.add_argument("--base-registration", required=True); q.add_argument("--output", required=True)
    for mode in ("watch", "task"):
        q = sub.add_parser(mode); q.add_argument("--registration", required=True)
        if mode == "watch": q.add_argument("--once", action="store_true")
        else: q.add_argument("--task", required=True)
    a = p.parse_args()
    if a.mode == "register": register(a.base_registration, a.output)
    elif a.mode == "watch": watch(a.registration, a.once)
    else: task(a.registration, a.task)


if __name__ == "__main__":
    main()
