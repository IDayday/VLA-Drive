"""Synchronous, finite, identity-bound phase execution (no background scheduler)."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from starVLA.teachers.decodable_st.source_registry import atomic_json,object_hash,sha256

PHASES=("audit","teacher","tiny","representation","planning","export")
STATES={"NOT_RUN","RUNNING","PASS","FAIL","BLOCKED","INCONCLUSIVE"}


def load_config(path):
    from omegaconf import OmegaConf
    cfg=OmegaConf.to_container(OmegaConf.load(path),resolve=True)
    required=("schema_version","baseline_identity","teacher_identities","dataset_manifests","observation_contract",
        "spatial_contract","temporal_contract","query_counts","trainable_selection","loss_weights","phase_budgets",
        "evaluation_population","seeds","resource_limits","acceptance_criteria","jobs","artifact_root")
    missing=[k for k in required if k not in cfg]
    if missing:raise ValueError("Missing resolved campaign fields: "+", ".join(missing))
    if cfg["schema_version"]!="decodable_st_campaign_v1":raise ValueError("Unsupported campaign schema")
    if cfg["resource_limits"]["gpu_hours_cap"]<=0:raise ValueError("Finite positive budget required")
    return cfg


def allowed_host():
    p=Path("/mnt/project/server_dispatch_policy.json");policy=json.loads(p.read_text());host=socket.gethostname()
    matches=[h for h,r in policy["allowed_hosts"].items() if host==h or host in r["aliases"]]
    if len(matches)!=1:raise PermissionError("Current host not in live dispatch allowlist: "+host)
    return {"host":host,"canonical":matches[0],"policy_sha256":sha256(p)}


def job_result(job,root,*,execute,resume,device,seed,max_updates,max_exposures):
    receipt=Path(job["receipt"])
    if receipt.is_file():
        result=json.loads(receipt.read_text())
        if job.get("reuse_sha256") and sha256(receipt)!=job["reuse_sha256"]:
            raise ValueError("Frozen receipt changed: "+str(receipt))
        if result.get("status") in STATES and not job.get("force_execute",False):
            # A numerical/ability FAIL is preserved, never retried by relaunching.
            return {"status":result["status"],"receipt":str(receipt),"sha256":sha256(receipt),"reused":True}
    if not execute:return {"status":"NOT_RUN","reason":"dry-run"}
    if job.get("blocked_reason"):return {"status":"BLOCKED","reason":job["blocked_reason"]}
    module=job["module"]
    if not module.startswith("tools.decodable_st."):raise ValueError("Foreign campaign command prohibited")
    python=Path(job["python"])
    if not python.is_file():return {"status":"BLOCKED","reason":"Missing isolated environment: "+str(python)}
    args=list(map(str,job["args"]))
    if "--device" not in args and job.get("uses_device",False):args += ["--device",device]
    if job.get("train",False):
        cap=int(job["update_cap"]);requested=cap if max_updates is None else int(max_updates)
        if requested>cap:raise ValueError("Requested updates exceed the registered finite phase budget")
        if max_exposures is not None:
            import math
            if max_exposures<=0:raise ValueError("Exposure cap must be positive")
            requested=min(requested,math.ceil(max_exposures*job["scene_count"]/job["global_batch"]))
        if "--max-updates" in args:
            at=args.index("--max-updates");args[at+1]=str(requested)
        else:args += ["--max-updates",str(requested)]
        if seed is not None:
            if job.get("accept_seed",False):args += ["--seed",str(seed)]
            elif seed!=job.get("seed",42):raise ValueError("This bounded operator test has one fixed registered seed")
        if resume:args += ["--resume"]
    command=[str(python),"-u","-m",module,*args]
    start=time.monotonic();routing=allowed_host()
    logs=root/"command_logs";logs.mkdir(exist_ok=True,parents=True)
    path=logs/(job["id"]+".log")
    with path.open("a") as f:
        code=subprocess.run(command,cwd=Path(__file__).resolve().parents[2],stdout=f,stderr=subprocess.STDOUT).returncode
    result={"status":"FAIL" if code else "INCONCLUSIVE","command":command,"returncode":code,
            "seconds":time.monotonic()-start,"log":str(path),"routing":routing}
    if receipt.is_file():result.update(status=json.loads(receipt.read_text()).get("status","INCONCLUSIVE"),receipt=str(receipt),sha256=sha256(receipt))
    if code:result["status"]="FAIL"
    return result


def gates(cfg):
    result={"engineering_verified":"INCONCLUSIVE","teacher_spatial_verified":"INCONCLUSIVE",
         "teacher_temporal_native_verified":"INCONCLUSIVE","teacher_temporal_driving_verified":"BLOCKED",
         "spatial_representation_verified":"NOT_RUN","temporal_representation_verified":"NOT_RUN",
         "planning_verified":"NOT_RUN","teacher_free_export_verified":"NOT_RUN"}
    root=Path(cfg["artifact_root"])
    parity=root/"audits/S0_PREFIX_CACHED_PARITY.json"
    if parity.exists():result["baseline_parity"]=json.loads(parity.read_text())["status"]
    capacity={}
    for name in ("compression16_v1","compression32_v1","compression32_patch_v1"):
        path=root/"runs"/name/"metrics_001000.json"
        if path.exists():
            m=json.loads(path.read_text())["teacher_agreement"]["reconstructed"]
            capacity[name]={"occupied_iou":m["occupied_iou"],"false_free":m["false_free"],
              "status":"PASS" if m["occupied_iou"]>=cfg["acceptance_criteria"]["compression_min_teacher_iou"] and
                 m["false_free"]<=cfg["acceptance_criteria"]["compression_max_false_free"] else "FAIL"}
    result["compression_acceptance"]=capacity
    export=root/"engineering_export_v1/DEPLOYMENT_REPORT.json"
    if export.exists():result["teacher_free_export_verified"]=json.loads(export.read_text())["status"]
    # Capability records must bind the baseline, teacher lock and raw evidence.
    # Missing records remain unresolved; a YAML switch cannot synthesize a PASS.
    acceptance=root/"CAPABILITY_ACCEPTANCE.json"
    if acceptance.exists():
        accepted=json.loads(acceptance.read_text())
        if accepted.get("baseline_identity")!=cfg["baseline_identity"] or accepted.get("teacher_identities")!=cfg["teacher_identities"]:
            raise ValueError("Capability acceptance has foreign baseline/teacher identity")
        for field,row in accepted.get("capabilities",{}).items():
            if field not in result or row["status"] not in STATES:raise ValueError("Unknown capability record")
            if row["status"]=="PASS" and not row.get("evidence"):raise ValueError("PASS lacks raw evidence")
            for evidence in row.get("evidence",[]):
                if sha256(evidence["path"])!=evidence["sha256"]:raise ValueError("Capability evidence changed")
            result[field]=row["status"]
    spatial=(result["teacher_spatial_verified"]=="PASS" and any(v["status"]=="PASS" for v in capacity.values()))
    temporal=result["teacher_temporal_driving_verified"]=="PASS"
    result["admitted_arms"]={"RS":spatial,"RT":temporal,"RST":spatial and temporal,
                             "R0":spatial or temporal,"Rcap":spatial or temporal,"Rdecode":spatial,"Rfrozen":spatial,
                             "Rjoint":False,"Rgt":False}
    result["formal_representation_admitted"]=any(result["admitted_arms"].values())
    result["formal_blockers"]=[] if result["formal_representation_admitted"] else ["Independent teacher occupancy quality cannot be accepted from hit-only labels",
                              "SGDrive native physical horizon and ego compensation are unverified",
                              "Tested compression mappings have not met the fixed native decode protection"]
    return result


def run(args):
    cfg=load_config(args.config);root=Path(args.artifact_root or cfg["artifact_root"]);root.mkdir(parents=True,exist_ok=True)
    identity=object_hash(cfg);state_path=root/"CAMPAIGN_STATE.json"
    lock=open(root/"CAMPAIGN.lock","a");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state=json.loads(state_path.read_text()) if state_path.exists() else {"identity":identity,"phases":{p:{"status":"NOT_RUN"} for p in PHASES}}
    if state["identity"]!=identity:raise ValueError("Campaign configuration changed; register an explicit new version")
    chosen=PHASES if args.phase=="all" else (args.phase,)
    execute=not args.dry_run and (args.phase!="all" or args.execute_ready_stages)
    state["gates"]=gates(cfg)
    for phase in chosen:
        if phase in ("representation","planning") and not state["gates"]["formal_representation_admitted"]:
            state["phases"][phase]={"status":"BLOCKED","reasons":state["gates"]["formal_blockers"]};atomic_json(state_path,state)
            if not args.continue_independent_on_failure and args.phase=="all":break
            continue
        rows=[]
        for job in cfg["jobs"].get(phase,[]):
            if args.arm and job.get("arm") not in (None,args.arm):continue
            if phase=="representation" and not state["gates"]["admitted_arms"].get(job.get("arm"),False):
                rows.append({"id":job["id"],"status":"BLOCKED","reason":"This arm's teacher/capacity prerequisites are not accepted"});continue
            if phase=="planning" and state["gates"]["spatial_representation_verified"]!="PASS" and state["gates"]["temporal_representation_verified"]!="PASS":
                rows.append({"id":job["id"],"status":"BLOCKED","reason":"Representation capability not accepted"});continue
            state["phases"][phase]={"status":"RUNNING","job":job["id"]};atomic_json(state_path,state)
            try:result=job_result(job,root,execute=execute,resume=args.resume,device=args.device,seed=args.seed,
                           max_updates=args.max_updates,max_exposures=args.max_exposures)
            except Exception as exc:result={"status":"FAIL","error":repr(exc)}
            rows.append({"id":job["id"],**result})
            state["phases"][phase]={"status":"RUNNING","jobs":rows};atomic_json(state_path,state)
            if result["status"] in ("FAIL","BLOCKED") and not args.continue_independent_on_failure:break
        statuses=[r["status"] for r in rows]
        status=("NOT_RUN" if not rows else "FAIL" if "FAIL" in statuses else "BLOCKED" if "BLOCKED" in statuses else
                "INCONCLUSIVE" if any(s!="PASS" for s in statuses) else "PASS")
        # A teacher interface PASS cannot become teacher capability PASS.
        if phase in ("teacher","tiny") and status=="PASS":status="INCONCLUSIVE"
        state["phases"][phase]={"status":status,"jobs":rows};state["gates"]=gates(cfg);atomic_json(state_path,state)
    print(json.dumps({"identity":identity,"phases":{k:v["status"] for k,v in state["phases"].items()},"gates":state["gates"]},indent=2))
    return state


def main():
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--phase",choices=(*PHASES,"all"),required=True)
    p.add_argument("--dry-run",action="store_true");p.add_argument("--resume",action="store_true");p.add_argument("--arm")
    p.add_argument("--seed",type=int);p.add_argument("--max-updates",type=int);p.add_argument("--max-exposures",type=float)
    p.add_argument("--device",default="cuda");p.add_argument("--artifact-root");p.add_argument("--execute-ready-stages",action="store_true")
    p.add_argument("--continue-independent-on-failure",action="store_true");run(p.parse_args())


if __name__=="__main__":main()
