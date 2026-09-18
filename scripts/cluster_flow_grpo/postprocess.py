"""Read-only, bounded checks after actual distributed diagnostic stages."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
from scripts.cluster_flow_grpo.cluster import PYTHON, ROOT, base_env, write_json
from scripts.cluster_flow_grpo.parallel_evaluation import run_evaluator


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument("--variant",choices=["f","u"],required=True)
    p.add_argument("--root",default="runs/resource_reallocation_v2");p.add_argument("--gpu",type=int,required=True)
    a=p.parse_args();root=Path(a.root).resolve();prefix=a.variant+"16"
    variant="frozen_visual" if a.variant=="f" else "unfrozen_visual"
    pointer=root/(prefix+"_continuous_attempt.json")
    continuous=Path(json.loads(pointer.read_text())["run"]) if pointer.exists() else root/(prefix+"_full_cont")
    def wait(name):
        path=root/(name+"_control/result.json");started=time.monotonic()
        while not path.exists():
            if time.monotonic()-started>5400:raise TimeoutError(name)
            time.sleep(5)
        r=json.loads(path.read_text());assert r["status"]=="PASS" and all(c==0 for c in r["exit_codes"]),r
    def execute(name,command):
        path=root/(prefix+"_"+name+".execution.json")
        if path.exists():raise FileExistsError(path)
        started=time.time();env=base_env();env["CUDA_VISIBLE_DEVICES"]=str(a.gpu)
        env["TRITON_CACHE_DIR"]=str(root/(prefix+"_postprocess_triton"))
        with (root/(prefix+"_"+name+".log")).open("x") as log:
            code=subprocess.run([PYTHON,*command],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=1800).returncode
        write_json(path,{"command":[PYTHON,*command],"exit_code":code,"started":started,"finished":time.time()})
        if code:raise RuntimeError(f"{name} exited{code}")
        print(name,"PASS",flush=True)
    wait(continuous.name)
    execute("adam_oracle_final",["scripts/cluster_flow_grpo/optimizer_evidence.py","adam","--run",str(continuous),"--output",str(root/(prefix+"_adam_oracle_final.json"))])
    for kind in ("repeat","resume"):
        wait(prefix+"_"+kind)
        execute(kind+"_comparison",["scripts/cluster_flow_grpo/boundary_evidence.py","--continuous",str(continuous/"checkpoints/update_000002"),
                "--resumed",str(root/(prefix+"_"+kind)/"checkpoints/update_000002"),"--output",str(root/(prefix+"_"+kind+"_comparison.json"))])
    for kind in ("full","resume"):
        boundary=(continuous if kind=="full" else root/(prefix+"_resume"))/"checkpoints/update_000002"
        exported=root/(prefix+"_"+kind+"_export")
        execute(kind+"_export",["-m","starVLA.rl.flow_grpo.cli","export","--checkpoint",str(boundary),"--output-dir",str(exported)])
    cfg=f"configs/flow_grpo/paired_world16_{variant}.yaml"
    execute("export_check",["-m","starVLA.rl.flow_grpo.cli","verify-export","--config",cfg,
        "--checkpoint",str(continuous/"checkpoints/update_000002"),"--export-dir",str(root/(prefix+"_full_export")),"--output-dir",str(root/(prefix+"_export_check"))])
    execute("resume_ode",["-m","starVLA.rl.flow_grpo.cli","verify-export","--config",cfg,
        "--checkpoint",str(continuous/"checkpoints/update_000002"),"--export-dir",str(root/(prefix+"_resume_export")),"--output-dir",str(root/(prefix+"_resume_ode"))])
    wait(prefix+"_scale")
    execute("scaling_comparison",["scripts/cluster_flow_grpo/optimizer_evidence.py","scaling",
        "--left",f"runs/production_acceptance_v3/{a.variant}_scale1/checkpoints/update_000001",
        "--right",str(root/(prefix+"_scale")/"checkpoints/update_000001"),"--output",str(root/(prefix+"_scaling_comparison.json"))])
    for kind in ("rl_only","sft_only","ref_only"):wait(prefix+"_"+kind)
    write_json(root/(prefix+"_postprocess_complete.json"),{"status":"PASS","scope":"completed bound diagnostic postprocessing; no production release issued"})


if __name__=="__main__":main()
