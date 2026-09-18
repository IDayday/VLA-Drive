"""Fixed paired recipe on explicit multi-node resources, with log-sharded eval.

Reuses the production checkpoint planner, semantic acceptance gate and original
evaluation CLI. A new world size starts from the two original SFT weights in a
new run: this command does not claim cross-world exact optimizer restoration.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import signal
import threading
import time
import numpy as np
import pandas as pd
from scripts.cluster_flow_grpo.cluster import run as cluster_run, PYTHON, ROOT, write_json, exclusive_controller
from scripts.cluster_flow_grpo.parallel_evaluation import evaluate_parallel, run_evaluator
from scripts.cluster_flow_grpo.identity import configure_release, validate_binding
from starVLA.rl.flow_grpo.acceptance import acceptance_context, enforce_training_budget, executable_identity
from starVLA.rl.flow_grpo.config import resolve_config
from starVLA.rl.flow_grpo.checkpoint import validate_checkpoint, validate_export
from starVLA.rl.flow_grpo.orchestration import advance_target
from starVLA.rl.flow_grpo.reproducibility import configure_numerics, resume_assets, training_provenance
from starVLA.rl.flow_grpo.loading import file_sha
from starVLA.rl.flow_grpo.evaluation import EVALUATION_SEEDS, paired_scores
from starVLA.rl.flow_grpo.evaluation_transaction import completed_evaluation


def validate_plan(plan):
    all_slots = []
    worlds = []
    for variant in ("frozen_visual", "unfrozen_visual"):
        group = plan["groups"][variant]
        sizes = {len(n["devices"]) for n in group["nodes"]}
        if len(sizes) != 1 or 0 in sizes:
            raise ValueError("each group requires equal nonempty node sizes")
        slots = [(n["host"], gpu) for n in group["nodes"] for gpu in n["devices"]]
        all_slots.extend(slots); worlds.append(len(slots))
    if worlds[0] != worlds[1] or 16 % worlds[0]:
        raise ValueError("equal world sizes dividing fixed global16 required")
    if len(all_slots) != len(set(all_slots)):
        raise ValueError("paired GPU allocations overlap")
    return worlds[0]


def evaluate_seeds(evaluate, checkpoint, label, split, slots, seeds=EVALUATION_SEEDS):
    """Parallel dev seeds on disjoint GPU groups; keep at least4 GPUs per seed.

    The dev split's largest complete log already limits one evaluation to556
    scenes on its slowest worker. Independent seeds can use the remaining GPUs
    without splitting a log or changing token-keyed inference randomness.
    Navtest keeps the whole allocation for its much larger scene set.
    """
    workers=max(1,min(4,len(slots)//4)) if split=="rl_dev" else 1
    assignments=[slots[i::workers] for i in range(workers)]
    def worker(index):
        return [(seed,evaluate(checkpoint,label,split,seed,eval_slots=assignments[index]))
                for seed in seeds[index::workers]]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results=dict(row for batch in pool.map(worker,range(workers)) for row in batch)
    return [results[seed] for seed in seeds]


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--plan", required=True)
    p.add_argument("--resume", action="store_true")
    a = p.parse_args()
    plan = json.loads(Path(a.plan).read_text())
    world = validate_plan(plan)
    cluster_identity,_=configure_release()
    os.environ.update(WORLD_SIZE=str(world), FLASH_ATTENTION_DETERMINISTIC="1",
                      CUBLAS_WORKSPACE_CONFIG=":4096:8")
    configure_numerics()
    root = Path(plan["output"]).resolve()
    descriptor = {"plan": plan, "training_executable_sha256": executable_identity(),
                  "orchestration_sources": {str(f.relative_to(ROOT)): file_sha(f)
                  for f in Path(__file__).parent.glob("*.py")}}
    if root.exists():
        if not a.resume:
            raise FileExistsError("existing run requires explicit --resume")
        if json.loads((root/"cluster_identity.json").read_text()) != descriptor:
            raise ValueError("cluster plan/code changed; explicit new validated run required")
    else:
        root.mkdir(parents=True)
        write_json(root/"cluster_identity.json", descriptor)
    cancelled = threading.Event()
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, lambda *args: cancelled.set())

    def variant_run(variant):
        try:
            group = plan["groups"][variant]
            cfg, sft = resolve_config(group["config"])
            if cfg["runtime"]["accumulation_steps"] * world != 16:
                raise ValueError("configured global scene batch differs from16")
            enforce_training_budget(cfg)
            validate_binding(cfg,cluster_identity)
            assets = resume_assets(cfg, sft)
            enforce_training_budget(cfg, acceptance_context(cfg, assets))
            provenance = training_provenance(cfg, sft, assets)
            run = root/variant
            eval_root = root/(variant+"_evaluation")
            eval_root.mkdir(exist_ok=True)
            slots = [{"host": node["host"], "gpu": gpu} for node in group["nodes"] for gpu in node["devices"]]
            dev = str(Path(cfg["paths"]["split_manifest"]).parent/"dev_tokens.json")
            locations = {}
            checkpoint_locations = {}
            locations_lock = threading.Lock()

            def evaluate(checkpoint, label, split, seed, eval_slots=None):
                if cancelled.is_set():
                    raise RuntimeError("paired experiment cancelled")
                tokens = dev if split == "rl_dev" else cfg["paths"]["test_list"]
                cache = cfg["paths"]["metric_cache"] if split == "rl_dev" else plan["navtest_cache"]
                dest = eval_root/f"{label}_{split}_seed{seed}"
                reused = group.get("reusable_evaluations", {}).get(f"{label}/{split}/{seed}")
                if reused:
                    dest = Path(reused)
                key=(str(Path(checkpoint).resolve()),split,seed)
                with locations_lock:
                    dest=checkpoint_locations.get(key,dest)
                report = evaluate_parallel(group["config"], checkpoint, dest, split, tokens,
                    cfg["paths"]["data_root"], cache, seed, eval_slots or slots)
                with locations_lock:
                    locations[(label,split,seed)] = dest
                    checkpoint_locations[key]=dest
                    write_json(eval_root/"locations.json", {"/".join(map(str,k)):str(v) for k,v in locations.items()})
                return report

            evaluate(cfg["sft_checkpoint"], "sft", "rl_dev", 42)
            best = None
            def train(resume, target):
                if cancelled.is_set():
                    raise RuntimeError("paired experiment cancelled")
                validate_binding(cfg,cluster_identity)
                suffix = f"{variant}_to{target}_{time.time_ns()}"
                spec = {"job_id": suffix, "nodes": group["nodes"],
                    "master_addr": group["master_addr"], "master_port": group["master_port"],
                    "control_dir": str(root/(suffix+".control")),
                    "require_idle_gpus": True,
                    "environment": group.get("environment", {}), "timeout_seconds": 28800,
                    "warm_files": group.get("warm_files", []),
                    "entry": ["-m", "starVLA.rl.flow_grpo.cli", "train", "--config", group["config"],
                        "--output-dir", str(run), "--max-updates", str(target)] +
                        (["--resume", str(resume)] if resume else []) +
                        ["--set", "runtime.run_mode="+("paired_short" if target==100 else "formal")]}
                path = root/(suffix+".json");write_json(path,spec)
                cluster_run(path,cancel_event=cancelled)

            def export(checkpoint, target):
                dest = run/f"export_update{target}"
                if not validate_export(checkpoint,dest):
                    run_evaluator([PYTHON,"-m","starVLA.rl.flow_grpo.cli","export",
                        "--checkpoint",str(checkpoint),"--output-dir",str(dest)],slots[0],
                        root/f"{variant}_export{target}.log")
                if not validate_export(checkpoint,dest):
                    raise RuntimeError("export did not complete")
                return dest

            for target in [100]+list(range(200,2001,200)):
                cp, exported, result = advance_target(run,target,
                    validate=lambda path:validate_checkpoint(path,cfg,provenance,world),
                    train=train,export=export,
                    evaluate=lambda export,update:evaluate(export,f"step{update}","rl_dev",42))
                if not result["complete_split"]:
                    raise ValueError("incomplete dev evaluation cannot select best")
                if target % 200 == 0 and (best is None or result["epdms"]>best["epdms"]):
                    best={"update":target,"epdms":result["epdms"],"checkpoint":str(cp),"export":str(exported)}
                write_json(run/"selection.json",{"best":best,"last_evaluated_update":target,
                    "rule":"max full dev seed42 EPDMS every200; earliest tie; no navtest"})
            for split in ("rl_dev","navtest"):
                evaluate_seeds(evaluate,cfg["sft_checkpoint"],"sft",split,slots)
                for label, checkpoint in [("last",run/"export_update2000"),("best",Path(best["export"]))]:
                    comparisons=[]
                    results=evaluate_seeds(evaluate,checkpoint,label,split,slots)
                    for seed,result in zip(EVALUATION_SEEDS,results):
                        if not result["complete_split"]:
                            raise ValueError("partial final evaluation")
                        left=locations[("sft",split,seed)];right=locations[(label,split,seed)]
                        rows,report=paired_scores(pd.read_csv(left/"original_protocol_scores.csv"),
                                                 pd.read_csv(right/"original_protocol_scores.csv"))
                        rows.to_csv(eval_root/f"paired_{label}_{split}_seed{seed}.csv",index=False)
                        comparisons.append({"seed":seed,**report})
                    values=[r["mean_paired_delta"] for r in comparisons]
                    write_json(eval_root/f"paired_{label}_{split}.json",{
                        "seeds":comparisons,"mean_delta":float(np.mean(values)),"seed_std":float(np.std(values,ddof=1))})
            write_json(run/"EXPERIMENT_COMPLETE.json",{"status":"COMPLETE","optimizer_updates":2000})
        except BaseException as exc:
            cancelled.set()
            write_json(root/(variant+"_failure.json"),{"time":time.time(),"error":str(exc)})
            raise

    with exclusive_controller(root), ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(variant_run, variant) for variant in plan["groups"]]
        for future in futures:
            future.result()


if __name__ == "__main__":
    main()
