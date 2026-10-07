"""Finite user-requested A/V checkpoint evaluation using unchanged native adapters.

The GPU queue runs on a separate authorized host. Every native rank has an
immutable scene partition and waits for an idle, UUID-checked GPU lease. CPU
scoring runs locally and can consume atomic predictions while exports continue.
No training source, controller, checkpoint or earlier result is modified.
"""
import argparse
import csv
import fcntl
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import sys
import time

from starVLA.model.modules.vehicle_joint.initialization import identity_hash
from tools.foresight.checkpoints import checkpoint_identity
from tools.full_foresight.navtest_schedule import atomic, read, sha, source_identity, validate_scores


ARMS = {"A_ACTION", "A_NO_MAE", "V_QUERY"}
METRICS = ("score", "no_at_fault_collisions", "drivable_area_compliance", "ego_progress",
           "time_to_collision_within_bound", "comfort", "driving_direction_compliance")


def prepare(request, output):
    c = read(request)
    if Path(output).exists():
        raise FileExistsError("Never overwrite a fixed-checkpoint registration")
    if {m["arm"] for m in c["models"]} != ARMS or len(c["models"]) != 3:
        raise ValueError("Exactly the three requested interface experiments are required")
    if c["update"] != 50000 or c["sampling_seed"] != 42:
        raise ValueError("This request is the exact common 50k checkpoint, sampling seed42")
    c["controller_source"] = source_identity(Path.cwd())
    c["controller_worktree"] = str(Path.cwd())
    c["canonical_hostname"] = socket.gethostname()
    c["root"] = str(Path(output).resolve().parent)
    current = read(Path(c["current_root"])/"identity.json")
    index = read(Path(c["current_root"])/"index.json")
    metric = read(c["metric_index"])
    audit = read(c["cache_audit"])
    if (current["split"] != "navtest" or len(index) != 12146 or len(metric) != 12146
            or len({r["token"] for r in index}) != 12146 or len({r["log"] for r in index}) != 136
            or {(r["token"], r["log"]) for r in index} != {(r["token"], r["log"]) for r in metric}
            or identity_hash(index) != current["index_sha256"] or not audit["passed"]
            or sha(c["metric_index"]) != audit["metric_index_sha256"]
            or sha(c["cache_snapshot"]) != audit["cache_snapshot_sha256"]):
        raise ValueError("Canonical original full-precision population/cache changed")
    from tools.local_interaction_mask_v2.score_async import python_tree_digest
    if python_tree_digest(Path(c["devkit"])/"navsim") != audit["navsim_tree_sha256"]:
        raise ValueError("Official scoring source changed")
    files = [request, c["metric_index"], c["cache_snapshot"], c["cache_audit"],
             c["gpu_policy"], c["qualified_cpu_parity"],
             str(Path(c["current_root"])/"identity.json"), str(Path(c["current_root"])/"index.json")]
    if not read(c["qualified_cpu_parity"])["passed"]:
        raise ValueError("Missing applicable official CPU qualification")
    for m in c["models"]:
        if source_identity(m["evaluation_worktree"]) != m["evaluation_source"]:
            raise ValueError("Immutable native inference source changed")
        training, checkpoint = checkpoint_identity(m["training_run"], m["tag"])
        run = Path(m["training_run"])
        upstream = run.parent.parent/"registrations"/(run.name+".json")
        original_registration = read(upstream)
        if (training["scope"] != "formal" or checkpoint["completed"] != c["update"]
                or checkpoint["training_seed"] != 42 or training["source_sha"] != m["evaluation_source"]
                or original_registration["arm"] != m["arm"]
                or sha(upstream) != training["registration_sha256"]
                or checkpoint["model_class"] != "starVLA.model.framework.ddp_action_video_foresight.DDPActionVideoForesight"
                or training["config"]["framework"]["action_model"]["num_inference_timesteps"] != 10):
            raise ValueError("Wrong formal model, source, exact update or solver")
        m["checkpoint"] = checkpoint
        job = Path(c["root"])/m["arm"]
        job.mkdir(parents=True, exist_ok=False)
        lock = dict(schema="foresight_navtest_checkpoint_probe_lock_v1",
            evaluation_purpose="user_requested_fixed_checkpoint", requested_update=c["update"],
            authorization=c["authorization"], experimental_arm=m["arm"],
            evaluation_source_sha=m["evaluation_source"], observer_source_sha=c["controller_source"],
            checkpoints=[checkpoint["sha256"]], checkpoint_records={checkpoint["sha256"]: checkpoint},
            current_data_identity=current["identity"], current_index_identity=current["index_sha256"],
            scene_count=12146, log_count=136, official_metric_index_sha256=sha(c["metric_index"]),
            sampling_seeds=[42], inference_steps=10, candidates_per_scene=1, learned_scorer=None,
            precision="FP32", final_endpoint_comparison=False,
            checkpoint_selection="exact user-requested 50k; no best-Navtest selection")
        lock["identity"] = identity_hash(lock)
        atomic(job/"lock.json", lock)
        atomic(job/"status.json", dict(status="QUEUED", optimizer_updates=0))
        files += [str(run/"identity.json"), str(upstream), str(job/"lock.json")]
    for p in Path(c["skill_deployment"]).glob("*.py"):
        files.append(str(p))
    reg = dict(schema="planning_interface_fixed_navtest_v1", config=c,
               asset_hashes={str(Path(p).resolve()): sha(p) for p in files}, created_unix=time.time())
    reg["identity"] = identity_hash(reg)
    atomic(output, reg)


def load(path):
    r = read(path)
    if r["identity"] != identity_hash({k:v for k,v in r.items() if k != "identity"}):
        raise ValueError("Registration changed")
    c = r["config"]
    if source_identity(c["controller_worktree"]) != c["controller_source"]:
        raise ValueError("Controller source changed")
    for p, expected in r["asset_hashes"].items():
        if sha(p) != expected:
            raise ValueError("Registered asset changed: "+p)
    return r


def environment():
    return dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1", TOKENIZERS_PARALLELISM="false",
                CUBLAS_WORKSPACE_CONFIG=":4096:8")


def launch(command, cwd, log):
    with Path(log).open("x") as stream:
        return subprocess.Popen(command, cwd=cwd, env=environment(), stdout=stream,
                                stderr=subprocess.STDOUT, start_new_session=True)


def finish(c, m, job):
    summary, rows = validate_scores(job/"scores", c["cache_snapshot"], c["update"])
    export = read(job/"predictions/identity.json")
    if summary["export_identity"] != export or export["checkpoint"] != m["checkpoint"]:
        raise ValueError("Scored model differs from the fixed registration")
    for rank in range(len(c["gpus"])):
        shard = read(job/"predictions"/f"shard_{rank}.json")
        if (shard["status"] != "complete" or shard["failed"]
                or shard["identity_sha256"] != identity_hash(export)
                or shard["completed"] != len(range(rank, 12146, len(c["gpus"])))):
            raise ValueError("Missing/failed/mismatched export partition")
    # An independently qualified backend is reused; every bank also receives
    # a deterministic distinct-log check through the unmodified official runner.
    lookup = {r["token"]:r for r in read(c["metric_index"])}
    spots, seen = [], set()
    for row in rows:
        if row["log"] not in seen:
            seen.add(row["log"]); spots.append(lookup[row["token"]])
        if len(spots) == 8:
            break
    atomic(job/"official_spot_index.json", spots)
    reference = job/"official_spots"
    if not (reference/"official.csv").exists():
        command = [c["scoring_python"], str(Path(c["skill_deployment"])/"score_official.py"), "reference",
                   "--index", str(job/"official_spot_index.json"), "--predictions", str(job/"predictions"),
                   "--devkit", c["devkit"], "--output", str(reference)]
        if launch(command, m["evaluation_worktree"], job/f"official_spots_{time.time_ns()}.log").wait():
            raise RuntimeError("Independent official spot replay failed")
    actual = {r["token"]:r for r in rows}
    errors = [abs(float(r[k])-float(actual[r["token"]][k]))
              for r in csv.DictReader((reference/"official.csv").open()) for k in METRICS]
    if len(errors) != 8*len(METRICS) or max(errors) > 1e-8:
        raise ValueError("Official spot replay mismatch")
    ego = job/"ego"
    if not (ego/"summary.json").exists():
        command = [c["inference_python"], "-m", "tools.foresight.evaluate_ego",
                   "--predictions", str(job/"predictions"), "--current-root", c["current_root"],
                   "--processed-root", c["ego_labels"], "--output", str(ego)]
        if launch(command, m["evaluation_worktree"], job/f"ego_{time.time_ns()}.log").wait():
            raise RuntimeError("Ego evaluation failed")
    fit = read(ego/"summary.json")
    if not fit["valid"] or fit["failed"] or fit["scenes"] != 12146:
        raise ValueError("Invalid ego-fit population")
    result = dict(status="COMPLETE", arm=m["arm"], update=c["update"], scenes=12146, logs=136,
        failed=0, zero_scenes=sum(float(r["score"]) == 0 for r in rows),
        PDMS_points=100*summary["PDMS"], metrics_points={k:100*v for k,v in summary["metrics"].items()},
        ego=fit["groups"]["all"], checkpoint=m["checkpoint"], evaluation_source=m["evaluation_source"],
        controller_source=c["controller_source"], sampling_seed=42,
        precision="FP32 optimizer masters/FP32 compute/TF32off", inference_steps=10, candidates=1,
        cpu_qualification="reused full official replay plus new8 distinct-log official spots",
        reused_cpu_parity_sha256=sha(c["qualified_cpu_parity"]), official_spot_max_error=max(errors),
        complete_csv_sha256=sha(job/"scores/scenes.csv"), optimizer_updates=0, completed_unix=time.time())
    atomic(job/"result.json", result)
    atomic(job/"status.json", result)
    return result


def _run(path, children):
    r = load(path); c = r["config"]; root = Path(c["root"])
    if socket.gethostname() != c["canonical_hostname"]:
        raise ValueError("Run the controller/CPU score on the registered canonical host")
    guard = (root/"controller.lock").open("a+")
    fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
    queue, active, scorers, retry, results = [], {}, {}, {}, {}
    for m in c["models"]:
        job = root/m["arm"]
        if (job/"result.json").exists():
            results[m["arm"]] = read(job/"result.json"); continue
        for rank in range(len(c["gpus"])):
            shard = job/"predictions"/f"shard_{rank}.json"
            if shard.exists() and read(shard)["status"] == "complete" and not read(shard)["failed"]:
                continue
            queue.append((m, rank))
    start = time.time()
    while len(results) < len(c["models"]):
        if time.time()-start > c["controller_timeout_seconds"]:
            raise TimeoutError("Finite evaluation window exhausted; all progress is retained")
        for key, record in list(active.items()):
            child, m, rank = record
            code = child.poll()
            if code is None:
                continue
            del active[key]
            if code == 75:
                queue.append((m, rank)); retry[key] = time.time()+30
            elif code:
                atomic(root/m["arm"]/"status.json", dict(status="FAILED", rank=rank, exit_code=code))
                raise RuntimeError(f"Native export failed for {m['arm']} rank{rank}: {code}")
        for m, rank in list(queue):
            card = c["gpus"][rank]; key = (m["arm"], rank)
            if any(c["gpus"][v[2]] == card for v in active.values()) or time.time() < retry.get(key, 0):
                continue
            job = root/m["arm"]; stamp = time.time_ns()
            native = [c["inference_python"], "-u", "-m", "tools.foresight.export_predictions",
                "--training-run", m["training_run"], "--checkpoint-tag", m["tag"],
                "--current-root", c["current_root"], "--output", str(job/"predictions"),
                "--campaign-root", str(root), "--run-id", f"{m['arm']}_50k_rank{rank}_{stamp}",
                "--sampling-seed", "42", "--rank", str(rank), "--world-size", str(len(c["gpus"])),
                "--max-seconds", str(c["export_timeout_seconds"]), "--campaign-gpu-hours", str(c["gpu_hours_limit"]),
                "--gpu-memory-fraction", "0.45", "--final-lock", str(job/"lock.json")]
            remote = ["env", "OMP_NUM_THREADS=1", "MKL_NUM_THREADS=1", "OPENBLAS_NUM_THREADS=1",
                "TOKENIZERS_PARALLELISM=false", "CUBLAS_WORKSPACE_CONFIG=:4096:8",
                "python3", str(Path(c["skill_deployment"])/"with_gpu_lease.py"),
                "--gpus", str(card), "--expect-uuid", c["gpu_uuids"][str(card)],
                "--policy", c["gpu_policy"], "--minimum-free-mib", "20000",
                "--lease-file", str(root/"gpu_leases"/f"gpu{card}.lock"),
                "--cwd", m["evaluation_worktree"], "--", *native]
            command = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", c["gpu_host"],
                       "exec "+shlex.join(remote)]
            child = launch(command, c["controller_worktree"], job/f"export_rank{rank}_{stamp}.log")
            children.append(child)
            active[key] = (child, m, rank); queue.remove((m, rank))
            atomic(job/f"export_rank{rank}_launch.json", dict(pid=child.pid, command=command,
                   physical_gpu=card, uuid=c["gpu_uuids"][str(card)], rank=rank,
                   world_size=len(c["gpus"]), started_unix=time.time(), optimizer_updates=0))
        for m in c["models"]:
            arm = m["arm"]; job = root/arm
            if arm in results:
                continue
            if arm not in scorers and (job/"predictions/identity.json").exists() and len(scorers) < c["cpu_slots"]:
                command = [c["scoring_python"], "-u", "-m", "tools.foresight.score_pdms",
                    "--devkit", c["devkit"], "--metric-index", c["metric_index"],
                    "--current-index", str(Path(c["current_root"])/"index.json"),
                    "--predictions", str(job/"predictions"), "--output", str(job/"scores"),
                    "--campaign-root", str(root), "--run-id", f"{arm}_50k_cpu_{time.time_ns()}",
                    "--workers", str(c["cpu_workers"]), "--timeout-seconds", str(c["controller_timeout_seconds"])]
                if (job/"scores/identity.json").exists():
                    command.append("--resume")
                scorers[arm] = launch(command, m["evaluation_worktree"], job/f"score_{time.time_ns()}.log")
                children.append(scorers[arm])
            if arm in scorers and scorers[arm].poll() is not None:
                if scorers[arm].returncode:
                    raise RuntimeError(f"Official CPU scoring failed for {arm}; artifacts retained")
                del scorers[arm]
                if (job/"scores/summary.json").exists():
                    results[arm] = finish(c, m, job)
        atomic(root/"status.json", dict(status="COMPLETE" if len(results)==3 else "RUNNING",
            pid=os.getpid(), registration=r["identity"], completed=list(results),
            active_gpu_ranks=[list(k) for k in active], queued_gpu_ranks=[[m["arm"],rank] for m,rank in queue],
            cpu_scoring=list(scorers), updated_unix=time.time(), optimizer_updates=0))
        if len(results) < 3:
            time.sleep(5)
    atomic(root/"SUMMARY.json", results)


def run(path):
    children = []
    try:
        _run(path, children)
    except BaseException as error:
        root = Path(read(path)["config"]["root"])
        atomic(root/"status.json", dict(status="FAILED", error=repr(error), pid=os.getpid(),
                                       updated_unix=time.time(), optimizer_updates=0))
        # Wait only for our own bounded children; never send a signal to a
        # trainer, pressure process or another evaluation/controller.
        for child in children:
            if child.poll() is None:
                child.wait()
        raise


def main():
    p = argparse.ArgumentParser(__doc__)
    s = p.add_subparsers(dest="mode", required=True)
    q = s.add_parser("prepare"); q.add_argument("--request", required=True); q.add_argument("--output", required=True)
    q = s.add_parser("run"); q.add_argument("--registration", required=True)
    a = p.parse_args()
    if a.mode == "prepare":
        prepare(a.request, a.output)
    else:
        run(a.registration)


if __name__ == "__main__":
    main()
