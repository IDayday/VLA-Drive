"""Resolve actual local assets into a campaign, retaining blocked dependencies.

No empty weight paths or generic pretrained fallback. This command does not
admit formal training or fabricate acceptance from a successful operator test.
"""
import argparse
import json
from pathlib import Path
from starVLA.teachers.decodable_st.source_registry import atomic_json,sha256


def run(root,output,python):
    root=Path(root).resolve();baseline=json.loads((root/"audits/baseline_manifest.json").read_text())
    lock=json.loads((root/"audits/external_sources.v2.lock.json").read_text())
    job_config=json.loads((root/"audits/tiny_job.json").read_text());jobs={k:[] for k in ("audit","teacher","tiny","representation","planning","export")}
    def job(phase,id,module,receipt,args,**kwargs):
        path=root/receipt
        row={"id":id,"module":"tools.decodable_st."+module,"python":str(Path(python).resolve()),"receipt":str(path),"args":args,**kwargs}
        if path.is_file():row["reuse_sha256"]=sha256(path)
        jobs[phase].append(row)
    job("audit","baseline_identity","audit_baseline","audits/baseline_manifest.json",[],
        blocked_reason="Baseline must be produced from the independently preserved anchor by audit_baseline; no replacement is allowed")
    job("audit","data_coordinates","audit_data","data_train64_v2/DATA_COORDINATE_AUDIT.json",
        ["--config",str(root/"audits/data_job.json"),"--output",str(root/"data_train64_v2")])
    for name,module,receipt in [("native_batching","verify_batching","REAL_BATCHING.json"),
                                ("native_DDP","verify_distributed","REAL_DDP.json"),
                                ("decoder_precision","verify_decoder_precision","DECODER_PRECISION.json")]:
        job("audit",name,module,"audits/"+receipt,[],
            blocked_reason="Use the independently registered finite GPU verification command if its receipt is absent")
    for name,receipt,config,mode in [("ViGT","vigt_train64_native_v1/ROUNDTRIP.json","vigt_job.json","vigt-real"),
                                    ("SGDrive","sgdrive_train64_native_v1/ROUNDTRIP.json","sgdrive_job.json",None)]:
        job("teacher",name,"audit_teacher" if name=="ViGT" else "audit_sgdrive_native",receipt,[],
            blocked_reason="Frozen native audit receipt missing; use recorded isolated native command, not a generic fallback")
    for name in ("compression16_v1","compression32_v1","compression32_patch_v1","compression32_patch_frozen_v2"):
        job("tiny",name,"train_compression",f"runs/{name}/RESULT.json",[],blocked_reason="Capacity receipt missing; exact variant must be independently registered")
    for arm,name in [("RS","RS_gradient_v1"),("Rfrozen","Rfrozen_tiny64_v1")]:
        job("tiny",arm,"train_representation",f"runs/{name}/RESULT.json",
             ["--config",str(root/"audits/tiny_job.json"),"--output",str(root/"runs"/name),"--arm",arm],
             arm=arm,train=True,update_cap=1000,scene_count=64,global_batch=1,seed=42,accept_seed=True,
             blocked_reason="Do not recreate a frozen-source run from the moving reporting worktree; use its exact resume command")
        for suffix in (name,arm+"_dev16_v1"):
            job("tiny",suffix+"_decoding","evaluate_decoding",f"evaluations/{suffix}/DECODING_REPORT.json",[],arm=arm,
                blocked_reason="Run the recorded fixed-endpoint decoding command; never substitute an earlier checkpoint")
    job("export","independent_teacher_free","verify_export","engineering_export_v2/DEPLOYMENT_REPORT.json",
        ["--root",str(root/"engineering_export_v2")],uses_device=True)
    job("export","trained_representation_teacher_free","verify_export","representation_export1000_v1/DEPLOYMENT_REPORT.json",
        ["--root",str(root/"representation_export1000_v1")],uses_device=True,
        blocked_reason="The immutable 1000-update representation export must exist and be verified first")
    for arm in ("R0","Rcap","RS","RT","RST","Rjoint","Rfrozen","Rdecode","Rgt"):
        job("representation",arm,"train_representation",f"formal/{arm}/RESULT.json",[],arm=arm,
            blocked_reason="Formal teacher quality, compression, loss calibration and full-population trainer acceptance are not established")
    job("planning","formal_reader_DiT","train_planner","formal/planner/RESULT.json",[],
        blocked_reason="No accepted representation; engineering two-update FM probe is not formal P3")
    cfg={"schema_version":"decodable_st_campaign_v1","artifact_root":str(root),"baseline_identity":baseline["identity"],
      "teacher_identities":{r["name"]:{"commit":r["commit"],"revision":r["weight_revision"],"weight_hashes":r["weight_hashes"]} for r in lock["sources"]},
      "dataset_manifests":{k:{"root":v["root"],"sha256":v["file_sha256"],"scenes":v["scenes"],"logs":v["logs"]} for k,v in baseline["data"].items()},
      "observation_contract":{"cameras":["CAM_F0","CAM_L0","CAM_R0"],"history_images":0,"state_shape":[1,4],"calibration_added":False},
      "spatial_contract":{"context":[256,256,256],"axes":"CXY","roi":[[-40,40],[-40,40],[-1,5.4]],"native_to_NAVSIM_height_origin":"UNVERIFIED"},
      "temporal_contract":{"hidden":[625,1536],"logits":[2,100,100,16],"native_horizon_seconds":None,"driving_status":"BLOCKED", "task":"native local binary future OCC"},
      "query_counts":{"DINO":144,"spatial":256,"temporal":625,"planning_memory_per_source":64},
      "trainable_selection":{"Qwen":"last min(8,depth) attention LoRA rank32; exact names in each run identity","vision":"frozen","DiT":"frozen in representation"},
      "loss_weights":{"tiny":job_config["loss_weights"],"formal":"BLOCKED_PENDING_TRAIN_ONLY_GRADIENT_CALIBRATION"},
      "phase_budgets":{"tiny_updates":1000,"warmup_updates":200,"formal_exposures_proposed":[2,2,1,1,5],"formal_admitted":False},
      "evaluation_population":{"tiny":"64 train-only scenes","capacity":"48 train /16 held-out train logs",
                               "diagnostic_dev":"one fixed scene from each of 16 independent dev logs","full_dev":1696,"navtest_selection":False},
      "seeds":[42],"resource_limits":{"gpu_hours_cap":16,"parallel_GPU_cap":6,"source":"current live dispatch policy; no withdrawn host use"},
      "acceptance_criteria":{"compression_min_teacher_iou":.99,"compression_max_false_free":.005,"unknown_is_negative":False},"jobs":jobs}
    atomic_json(output,cfg);return cfg


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--artifact-root",required=True);p.add_argument("--output",required=True);p.add_argument("--python",required=True);a=p.parse_args()
    run(a.artifact_root,a.output,a.python)
