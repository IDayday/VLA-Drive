"""Bounded production-layout checks; never starts a formal training budget."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import time
from scripts.cluster_flow_grpo.cluster import run, write_json
from starVLA.rl.flow_grpo.config import resolve_config


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument("--base-spec",required=True)
    p.add_argument("--steps",nargs="+",default=["repeat","resume","scale","rl_only","sft_only","ref_only"])
    args=p.parse_args()
    base=json.loads(Path(args.base_spec).read_text())
    base_cfg,_=resolve_config(base["entry"][base["entry"].index("--config")+1])
    process_timeout=base_cfg["runtime"].get("process_group_timeout",120)
    root=Path(base["control_dir"]).parent
    prefix=base["job_id"].split("_full_cont")[0]
    variant="unfrozen_visual" if prefix.startswith("u") else "frozen_visual"
    current=Path(base["control_dir"])/"result.json"
    start=time.monotonic()
    while not current.exists():
        if time.monotonic()-start>base["timeout_seconds"]+60:
            raise TimeoutError("initial bounded job did not finish")
        time.sleep(5)
    if json.loads(current.read_text())["status"]!="PASS":
        raise RuntimeError("initial bounded job failed")
    for step in args.steps:
        if step not in {"repeat","resume","scale","rl_only","sft_only","ref_only"}:
            raise ValueError(step)
        name=prefix+"_"+step
        spec=deepcopy(base)
        spec.update(job_id=name,control_dir=str(root/(name+"_control")),timeout_seconds=2400)
        output=root/name
        ancillary=step in {"scale","rl_only","sft_only","ref_only"}
        config=(f"configs/flow_grpo/paired_fp32_partition_{variant}.yaml" if ancillary
                else f"configs/flow_grpo/paired_world16_{variant}.yaml")
        spec["entry"]=["-m","starVLA.rl.flow_grpo.cli","train","--config",config,
                       "--output-dir",str(output),"--max-updates",
                       "1" if step in {"scale","rl_only","sft_only"} else "2"]
        if step=="resume":
            continuous=Path(base["entry"][base["entry"].index("--output-dir")+1])
            spec["entry"] += ["--resume",str(continuous/"checkpoints/update_000001")]
        spec["entry"] += ["--set","runtime.run_mode=diagnostic","runtime.save_every=1",
                           "runtime.accumulation_steps=1",
                           f"runtime.process_group_timeout={process_timeout}"]
        if step=="scale":
            spec["entry"] += ["optimizer.max_grad_norm=1000000"]
        if step in {"rl_only","sft_only","ref_only"}:
            spec["entry"] += ["runtime.diagnostic_gradient_statistics=true",
                "runtime.diagnostic_loss_scope="+("reference_after_joint" if step=="ref_only" else step)]
        path=root/(name+"_spec.json")
        if path.exists():
            raise FileExistsError(path)
        write_json(path,spec)
        print(name,"START",flush=True)
        result=run(path)
        print(name,result,flush=True)


if __name__=="__main__":
    main()
