"""Publish bounded numerical evidence, never weights, source images or cache."""
import argparse
import csv
import json
from pathlib import Path
import statistics
from starVLA.teachers.decodable_st.source_registry import atomic_json,sha256


def run(root,output):
    root=Path(root);out=Path(output);out.mkdir(parents=True,exist_ok=True)
    evidence={};cost=[]
    def read(path,category=None,gpu=True):
        file=root/path
        if not file.is_file():return None
        x=json.loads(file.read_text());evidence[path]=sha256(file)
        if category and ("gpu_hours" in x or "seconds" in x):
            cost.append({"artifact":path,"category":category,"GPU_hours":x.get("gpu_hours",x.get("seconds",0)/3600) if gpu else 0,
                         "wall_seconds":x.get("seconds"),"measurement":"receipt timer; pre-timer initialization may be excluded"})
        return x
    baseline=read("audits/baseline_manifest.json");sources=read("audits/external_sources.v2.lock.json")
    native={k:read(v,"teacher") for k,v in [("ViGT","vigt_train64_native_v1/ROUNDTRIP.json"),("SGDrive","sgdrive_train64_native_v1/ROUNDTRIP.json")]}
    read("sgdrive_train2_native_v1/ROUNDTRIP.json","teacher")
    data=read("data_train64_v2/DATA_COORDINATE_AUDIT.json","labels",False)
    capacity={}
    for name in ("compression16_v1","compression32_v1","compression32_patch_v1","compression32_patch_frozen_v2"):
        m=read(f"runs/{name}/metrics_001000.json");r=read(f"runs/{name}/RESULT.json","compression");ident=read(f"runs/{name}/identity.json")
        if m:capacity[name]={"metrics":m,"run":r,"source_sha":ident["source_sha"],"run_identity":ident["identity"]}
    learners={}
    for arm,name in [("RS","RS_gradient_v1"),("Rfrozen","Rfrozen_tiny64_v1")]:
        r=read(f"runs/{name}/RESULT.json","student_tiny");identity=read(f"runs/{name}/identity.json")
        evaluation=read(f"evaluations/{name}/DECODING_REPORT.json","student_evaluation")
        if evaluation:
            # Per-scene metrics are numbers and public opaque identifiers, not RGB.
            atomic_json(out/(arm+"_DECODING.json"),evaluation)
        learners[arm]={"run":r,"source_sha":identity["source_sha"],"identity":identity["identity"],
                       "trainable_parameters":sum(p["elements"] for p in identity["trainable_inventory"]),
                       "LoRA_modules":identity["lora_modules"],"evaluation":None if evaluation is None else {k:v for k,v in evaluation.items() if k!="per_scene"}}
    read("runs/RS_gradient_v1/segments/update203_RESULT.json","student_tiny")
    # This checkpoint is the preserved first two-update segment; its historical
    # wall timer was observed before RESULT was replaced by the next segment.
    segment=read("runs/RS_gradient_v1/segments/update2_observed_RESULT.json","student_tiny")
    gradient=read("runs/RS_gradient_v1/segments/update203_gradient_calibration.json")
    probes={}
    for name,path in [("prefix","audits/S0_PREFIX_CACHED_PARITY.json"),("initial_appended_failure","audits/S0_PARITY.json"),
                      ("batching","audits/REAL_BATCHING.json"),("decoder_precision","audits/DECODER_PRECISION.json"),
                      ("DDP","audits/REAL_DDP.json")]:
        x=read(path,"engineering")
        if x:probes[name]=x
    recovery={}
    for name,path in [("AA","audits/FULL_MODEL_REPEAT_AA.json"),("AB_resume","audits/FULL_MODEL_RECOVERY.json")]:
        x=read(path)
        if x:
            rows=x["differing_float_tensors"]
            recovery[name]={"strict_equal":x["strict_bitwise_equal"],"discrete_errors":x["discrete_state_errors"],"different_tensors":len(rows),
                 "max_abs":max((r["max_abs"] for r in rows),default=0),"max_normalized_l2":max((r["normalized_l2"] for r in rows),default=0),
                 "minimum_cosine":min((r["cosine"] for r in rows),default=1)}
    for name in ("continuous","resumed","repeated_continuous"):
        read(f"runs/resume_boundary_v1/{name}/RESULT.json","recovery")
    deploy=read("engineering_export_v2/DEPLOYMENT_REPORT.json")
    export=read("engineering_export_v2/manifest.json")
    if deploy:
        deploy={**deploy,"median_latency_seconds":statistics.median(deploy["latency_seconds"]),"export_weight_sha256":export["weights_sha256"]}
    read("runs/planner_engineering_v1/RESULT.json","planner_engineering")
    read("planning_engineering_dev8_v1/EXPORT.json","planning_export")
    scores={}
    for kind in ("q0","q_final"):
        s=read(f"planning_engineering_dev8_v1/{kind}_scores/summary.json")
        if s:scores[kind]={k:s[k] for k in ("scenes","logs","PDMS","zero_fraction","metrics","failed","valid")}
    transitions={}
    if len(scores)==2:
        def rows(kind):
            with (root/f"planning_engineering_dev8_v1/{kind}_scores/scenes.csv").open() as f:return {r["token"]:r for r in csv.DictReader(f)}
        a,b=rows("q0"),rows("q_final")
        if a.keys()!=b.keys():raise ValueError("Planning population changed")
        transitions={"zero_to_nonzero":sum(float(a[k]["score"])==0 and float(b[k]["score"])>0 for k in a),
                     "nonzero_to_zero":sum(float(a[k]["score"])>0 and float(b[k]["score"])==0 for k in a)}
    cache=read("sparse_cache64_v1/CACHE_BUILD.json","cache",False)
    visual=read("private_student_visualizations_v1/INDEX.json")
    if visual:atomic_json(out/"PRIVATE_VISUALIZATION_INDEX.json",visual)
    result={"scope":"P0/P1 and execution plumbing; formal scientific matrix incomplete","baseline":{k:baseline[k] for k in ("baseline","identity","source_commit","actual_hidden_dim","shared_training_gpu_hours")},
            "teacher_native_interface":{k:{"status":v["status"],"scenes":len(v.get("records",v.get("rows",[]))),"seconds":v["seconds"]} for k,v in native.items()},
            "data":{k:data[k] for k in ("train_scenes","train_logs","dev_scenes","dev_logs","log_overlap","token_overlap","independent_occupancy_quality")},
            "capacity":capacity,"learners":learners,"operator_checks":probes,"recovery":recovery,"deployment":deploy,
            "engineering_dev8_only":scores,"engineering_failure_transitions":transitions,
            "gradient_calibration":{k:v for k,v in (gradient or {}).items() if k!="parameter_groups"},
            "states":{"engineering_verified":"INCONCLUSIVE","teacher_spatial_verified":"INCONCLUSIVE","teacher_temporal_native_verified":"INCONCLUSIVE",
                "teacher_temporal_driving_verified":"BLOCKED","spatial_representation_verified":"INCONCLUSIVE","temporal_representation_verified":"NOT_RUN",
                "planning_verified":"NOT_RUN","teacher_free_export_verified":deploy["status"] if deploy else "NOT_RUN"},
            "cost":{"recorded_GPU_hours_lower_bound":sum(r["GPU_hours"] for r in cost),"items":cost,
                "unmetered":"Some import/hash/load, downloads, failed preflight and CPU rendering have no reliable complete wall timer. No total cost exactness claim.","budget_GPU_hours":16},
            "missing":["independent occupancy/free/unknown and native height-frame audit","SGDrive physical horizon/target-domain adaptation/persistence capability",
                "accepted compression/representation and training-scale loss calibration","formal P2/P3 full-population implementation and runs",
                "R0/RT/RST/Rcap/Rdecode/Rjoint/Rgt matched formal experiments","three independent new training seeds",
                "full model multirank checkpoint/resume and post-LoRA resume characterization","full development/Navtest benefit, cluster bootstrap and safety transitions"]}
    atomic_json(out/"RESULTS.json",result);atomic_json(out/"EVIDENCE_SHA256.json",evidence);atomic_json(out/"external_sources.lock.json",sources)
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--artifact-root",required=True);p.add_argument("--output",required=True);a=p.parse_args();run(a.artifact_root,a.output)
