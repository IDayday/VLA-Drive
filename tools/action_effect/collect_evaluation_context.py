"""Register and shard the fixed original-S0 DEV1696 context for final evaluation.

Only current images/history pose/navigation are encoded. GT and metric payloads
never enter the model; metric references remain offline registration metadata.
No training, simulation, checkpoint selection, or navtest access is performed.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import time

import numpy as np

from .common import (artifact, atomic_bytes, check_space, digest, load_config, lock,
                     read_json, sha256, verify_artifact, verify_identity, write_json)
from .evaluation_context_data import (FEATURE_SCHEMA, FULL_LOGS, FULL_SCENES, MANIFEST_SCHEMA,
    REGISTRATION_SCHEMA, SHARD_SCHEMA, INDEX_FIELDS, shard_population, validate_feature_record,
    validate_manifest, validate_registration, validate_shard)


def _pin(path, read_mib_per_second):
    path=Path(path).resolve();before=path.stat();start=time.perf_counter();total=0;h=hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(1<<20),b""):
            h.update(block);total+=len(block)
            wait=total/(read_mib_per_second*1024*1024)-(time.perf_counter()-start)
            if wait>0:time.sleep(wait)
    after=path.stat()
    if ((before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)
            !=(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns) or total!=after.st_size):
        raise ValueError("DEV dependency changed while hashing")
    return {"path":str(path),"bytes":total,"sha256":h.hexdigest()}


def register(cfg, *, current_root, metric_index, historical_export, historical_score_identity,
             output, shard_scenes=16, read_mib_per_second=128):
    if type(shard_scenes) is not int or not 1<=shard_scenes<=32 or not 1<=read_mib_per_second<=2048:
        raise ValueError("Explicit bounded DEV shard/registration read budget required")
    from .collect_base_trajectories import prepare as prepare_campaign
    from tools.st_world_policy.base_binding import read_binding
    from starVLA.model.modules.st_world.consequence import protocol_identity, ConsequenceSpec
    campaign=prepare_campaign(cfg);base=read_binding(cfg["base_binding"])
    root=Path(current_root).resolve(); export=Path(historical_export).resolve(); output=Path(output).resolve()
    if not output.is_relative_to(Path(cfg["artifact_root"]).resolve()):
        raise ValueError("DEV cache must stay inside the campaign's bounded artifact namespace")
    current=read_json(root/"identity.json"); index=read_json(root/"index.json"); complete=read_json(root/"COMPLETE.json")
    old=read_json(export/"identity.json"); scorer=read_json(historical_score_identity); metric=read_json(metric_index)
    expected_protocol={"precision":"FP32","tf32":False,"scorer":None,"candidates_per_scene":1,
        "sampling_seed":cfg["seed"],"steps":base["inference_contract"]["flow_steps"],
        "current_views":["CAM_F0","CAM_L0","CAM_R0"],"history_images":0,"future_conditioning":False}
    if (current.get("schema")!="ddpolicy_current_cameras_v1" or current.get("split")!="dev"
            or current.get("current_only") is not True or current.get("cameras")!=expected_protocol["current_views"]
            or digest(index)!=current.get("index_sha256") or old.get("checkpoint")!=base["checkpoint"]
            or old.get("current_identity")!=current or old.get("limit")!=0
            or any(old.get("protocol",{}).get(k)!=v for k,v in expected_protocol.items())
            or scorer.get("export")!=old or scorer.get("metric_index_sha256")!=sha256(metric_index)
            or scorer.get("current_index_sha256")!=sha256(root/"index.json")
            or scorer.get("export_file_sha256")!=sha256(export/"identity.json")
            or complete.get("failures")!=0 or complete.get("scenes")!=FULL_SCENES
            or complete.get("logs")!=FULL_LOGS or complete.get("current_identity")!=current["identity"]):
        raise ValueError("DEV/current/historical Base/protocol/metric-index identity differs")
    if (len(index)!=FULL_SCENES or len({r["token"] for r in index})!=FULL_SCENES
            or len({r["log"] for r in index})!=FULL_LOGS or len(metric)!=FULL_SCENES
            or len({r["token"] for r in metric})!=FULL_SCENES):
        raise ValueError("Only the complete pre-existing DEV1696/16log population is supported")
    train_cohort=verify_identity(read_json(verify_artifact(campaign["cohort"])))
    train_root=Path(train_cohort["data_root"]);train_index=read_json(train_root/"index.json")
    if {r["log"] for r in train_index}&{r["log"] for r in index}:
        raise ValueError("DEV logs overlap Base training inputs")
    lookup={r["token"]:r for r in metric}; signature=digest(old); population=[]
    for dataset_index,row in enumerate(index):
        sid,log=row["token"],row["log"]
        if any(x in sid for x in ("/","\\","..")) or sid not in lookup or lookup[sid]["log"]!=log:
            raise ValueError("DEV scene/log/metric join mismatch")
        current_path=root/"current"/(sid+".json"); observation=read_json(current_path)
        if (observation.get("identity")!=current["identity"] or observation.get("token")!=sid
                or observation.get("log")!=log or len(observation.get("image_paths",[]))!=3):
            raise ValueError("Current-only DEV observation identity mismatch")
        cache=Path(lookup[sid]["cache_path"])
        if cache.parent.name!=sid or cache.parents[2].name!=log:raise ValueError("Metric path belongs to another scene/log")
        meta_path=export/"predictions"/(sid+".json"); meta=read_json(meta_path)
        pred_path=export/"predictions"/(sid+".npz")
        pred=_pin(pred_path,read_mib_per_second)
        if (meta.get("identity_sha256")!=signature or meta.get("status")!="ok" or meta.get("token")!=sid
                or meta.get("log")!=log or meta.get("proposal_sha256")!=pred["sha256"]):
            raise ValueError("Original S0 DEV prediction receipt changed")
        with np.load(pred_path,allow_pickle=False) as saved: physical=saved["trajectory"]
        if physical.shape!=(8,3) or physical.dtype!=np.float32 or not np.isfinite(physical).all():
            raise ValueError("Original DEV reference trajectory is not finite FP32 8x3")
        current_pin=_pin(current_path,read_mib_per_second)
        images=[_pin(p,read_mib_per_second) for p in observation["image_paths"]]
        item={"scene_id":sid,"source_log":log,"role":"dev_report","split":"dev","dataset_index":dataset_index,
              "current":current_pin,"images":images,"metric_cache":_pin(cache,read_mib_per_second),
              "historical_prediction":pred,"historical_meta":_pin(meta_path,read_mib_per_second)}
        item["observation_hash"]=digest({"current":current_pin,"images":images})
        population.append(item)
    head_receipt=Path(cfg["artifact_root"])/"contexts"/"S0_ACTION_HEAD.COMPLETE.json"
    head=read_json(head_receipt)
    if (head.get("schema")!="action_effect_original_action_head_v1" or head.get("strict_original_master") is not True
            or head.get("base_binding_hash")!=base["binding_sha256"]):
        raise ValueError("Existing original S0 action-head receipt required; no duplicate head is written")
    if _pin(head["artifact"]["path"],read_mib_per_second)!=head["artifact"]:
        raise ValueError("Existing S0 action-head bytes changed")
    protocol=protocol_identity(cfg["g0"]["navsim_root"])
    roots=tuple(sorted({str(Path(r["metric_cache"]["path"]).resolve().parents[3]) for r in population}))
    spec=ConsequenceSpec(cfg["g0"]["navsim_root"],cfg["g0"]["maps_root"],roots,cfg["campaign_id"],protocol["sha256"],True)
    from dataclasses import asdict
    repo=Path(__file__).resolve().parents[2]
    source_paths=[Path(__file__),Path(__file__).with_name("evaluation_context_data.py")]
    reg={"schema":REGISTRATION_SCHEMA,"split":"dev","role":"dev_report","for_training":False,
        "requested_scenes":FULL_SCENES,"requested_logs":FULL_LOGS,"train_dev_log_overlap":0,
        "campaign_binding":artifact(Path(cfg["artifact_root"])/"CAMPAIGN_BINDING.json"),
        "campaign_binding_identity":campaign["identity"],"base_binding":campaign["base_binding"],
        "base_checkpoint_binding_hash":base["binding_sha256"],"base_checkpoint":base["checkpoint"],
        "base_inference_contract":base["inference_contract"],"extraction_contract":campaign["extraction_contract"],
        "numeric_runtime_hash":digest({"base":base["binding_sha256"],"inference":base["inference_contract"],
                                       "extraction_contract":campaign["extraction_contract"]}),
        "current_root":str(root),"current_identity":artifact(root/"identity.json"),"current_index":artifact(root/"index.json"),
        "current_complete":artifact(root/"COMPLETE.json"),"train_index":artifact(train_root/"index.json"),
        "historical_export":str(export),"historical_export_identity":artifact(export/"identity.json"),
        "historical_scoring_identity":artifact(historical_score_identity),"metric_index":artifact(metric_index),
        "population":population,"spec":asdict(spec),"score_protocol_hash":protocol["sha256"],
        "head_receipt":artifact(head_receipt),"seed":cfg["seed"],"shard_scenes":shard_scenes,"dtype":"float32",
        "single_encoder_per_scene":True,"historical_comparison":"exact FP32 full trajectory, fixed same-noise Euler10",
        "sources":{str(p.relative_to(repo)):sha256(p) for p in source_paths},
        "source_commit":subprocess.check_output(["git","-C",str(Path(__file__).resolve().parents[2]),"rev-parse","HEAD"],text=True).strip()}
    reg["identity"]=digest(reg);validate_registration(reg)
    output.mkdir(parents=True,exist_ok=True);write_json(output/"EVAL_REGISTRATION.json",reg)
    write_json(output/"RESOLVED_CONFIG.json",cfg)
    return {"status":"REGISTERED","registration":artifact(output/"EVAL_REGISTRATION.json"),"scenes":FULL_SCENES,"logs":FULL_LOGS,
            "optimizer_updates":0,"simulation_calls":0,"GPU_calls":0}


def verify_execution(reg,cfg):
    from .collect_base_trajectories import prepare as prepare_campaign
    campaign=prepare_campaign(cfg)
    if (campaign["identity"]!=reg["campaign_binding_identity"] or campaign["base_checkpoint"]!=reg["base_checkpoint"]
            or campaign["extraction_contract"]!=reg["extraction_contract"]):
        raise ValueError("DEV extraction Base/runtime differs from the frozen TRAIN contract")
    repo=Path(__file__).resolve().parents[2]
    for path,expected in reg["sources"].items():
        if sha256(repo/path)!=expected:raise ValueError("DEV collector/reader source changed")
    for key in ("campaign_binding","base_binding","current_identity","current_index","current_complete",
                "train_index","historical_export_identity","historical_scoring_identity","metric_index","head_receipt"):
        verify_artifact(reg[key])
    return campaign


def collect(cfg, *, output, rank, world_size, device, max_seconds, resume=False):
    import torch
    from starVLA.dataloader.foresight_dataset import ForesightCurrentDataset,decode_ego
    from tools.foresight.checkpoints import load_student,scene_noise
    from tools.st_world_policy.base_binding import read_binding
    from starVLA.model.modules.flow_posttraining.policy_snapshot import tensor_hash
    output=Path(output);reg=validate_registration(read_json(output/"EVAL_REGISTRATION.json"));verify_execution(reg,cfg)
    if (type(rank) is not int or type(world_size) is not int or not 0<=rank<world_size<=8
            or not np.isfinite(max_seconds) or not 0<max_seconds<=14400):
        raise ValueError("Explicit bounded DEV shard rank/world/time required")
    if not str(device).startswith("cuda") or not torch.cuda.is_available():raise ValueError("Actual allocated CUDA device required")
    assigned=[(i,chunk) for i,chunk in enumerate(shard_population(reg)) if i%world_size==rank]
    if not resume and any((output/"shards").glob("*.COMPLETE.json")):
        # Other disjoint ranks may already be publishing. Only own completed
        # work requires resume; a different worker must not block this worker.
        if any((output/"shards"/(digest([reg["identity"],[r["scene_id"] for r in chunk]])+".COMPLETE.json")).exists() for _,chunk in assigned):
            raise ValueError("Own prior DEV shards exist; use --resume")
    pending=[]
    for i,chunk in assigned:
        tag=digest([reg["identity"],[r["scene_id"] for r in chunk]])
        receipt=output/"shards"/(tag+".COMPLETE.json")
        if receipt.exists():validate_shard(read_json(receipt),reg,chunk)
        else:pending.append((i,tag,chunk))
    torch.set_num_threads(2);torch.cuda.set_device(device);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    base=read_binding(cfg["base_binding"]);data=ForesightCurrentDataset(reg["current_root"])
    start=time.perf_counter();deadline=time.monotonic()+max_seconds;done=len(assigned)-len(pending);busy=0;model=None
    hooks={"visual":0,"language":0};handles=[]
    try:
        if pending:
            model=load_student(base["run"],base["tag"],base["parent_identity"],device=device,precision="fp32")
            model.requires_grad_(False).eval()
            if any(p.dtype!=torch.float32 for p in model.parameters()):raise ValueError("S0 canonical DEV extraction requires FP32 parameters")
            def hook(name):
                def count(*_):hooks[name]+=1
                return count
            handles=[model.qwen_vl_interface.model.model.visual.register_forward_hook(hook("visual")),
                     model.qwen_vl_interface.model.model.language_model.register_forward_hook(hook("language"))]
        with torch.no_grad():
            for i,tag,chunk in pending:
                if time.monotonic()>=deadline:break
                check_space(cfg);receipt=output/"shards"/(tag+".COMPLETE.json");payload=output/"shards"/(tag+".pt")
                try:
                    with lock(output/"shards"/(tag+".CLAIM.lock")):
                        if receipt.exists():validate_shard(read_json(receipt),reg,chunk);done+=1;continue
                        records=[];entries=[]
                        for row in chunk:
                            for pin in [row["current"],*row["images"],row["historical_prediction"],row["historical_meta"]]:verify_artifact(pin)
                            obs=data[row["dataset_index"]]
                            if set(obs)!={"image","state","lang","token"} or obs["token"]!=row["scene_id"]:
                                raise ValueError("DEV encoder takes current-only legal observations")
                            before=hooks.copy();encoded=model.encode_current([obs]);condition=model.build_planner_condition(encoded)
                            if {k:hooks[k]-before[k] for k in hooks}!={"visual":1,"language":1}:
                                raise ValueError("Exactly one shared visual and language call per DEV scene required")
                            noise=scene_noise(obs["token"],reg["seed"],device)
                            raw=model.action_model.predict_action(condition,initial_noise=noise);physical=decode_ego(raw.float())
                            record={"schema":FEATURE_SCHEMA,**{k:row[k] for k in INDEX_FIELDS},
                                "base_binding_hash":reg["base_checkpoint_binding_hash"],"evaluation_binding":reg["identity"],
                                "memory":torch.cat((encoded["W"],encoded["action_queries"]),1)[0].cpu(),
                                "ego":torch.as_tensor(obs["state"]).float().cpu(),"condition":condition[0].cpu(),
                                "W_count":encoded["W"].shape[1],"raw":raw[0].cpu(),"physical":physical[0].cpu(),"noise":noise[0].cpu(),
                                "raw_hash":tensor_hash(raw),"physical_hash":tensor_hash(physical),"historical_physical_exact":True}
                            validate_feature_record(record,row,reg,compare_historical=True);records.append(record)
                            entries.append({**{k:row[k] for k in INDEX_FIELDS},"shard_offset":len(entries),
                                            "raw_hash":record["raw_hash"],"physical_hash":record["physical_hash"]})
                        buffer=io.BytesIO();torch.save(records,buffer);atomic_bytes(payload,buffer.getvalue());pin=artifact(payload)
                        for entry in entries:entry["shard"]=pin
                        done_receipt={"schema":SHARD_SCHEMA,"evaluation_binding":reg["identity"],"split":"dev","for_training":False,
                            "single_encoder_per_scene":True,"historical_physical_exact":True,"shard":pin,"entries":entries}
                        done_receipt["identity"]=digest(done_receipt);write_json(receipt,done_receipt);done+=1
                        print(json.dumps({"rank":rank,"world_size":world_size,"shard":i,"scenes":len(records),"status":"COMPLETE"}),flush=True)
                except BlockingIOError:busy+=1
    finally:
        for handle in handles:handle.remove()
    result={"schema":"action_effect_evaluation_context_worker_v1","evaluation_binding":reg["identity"],
            "status":"WORKER_COMPLETE" if done==len(assigned) else "PARTIAL","rank":rank,"world_size":world_size,
            "assigned_shards":len(assigned),"completed_shards":done,"busy_shards":busy,"seconds":time.perf_counter()-start,
            "encoder_calls":hooks,"optimizer_updates":0,"simulation_calls":0,
            "peak_allocated_bytes":torch.cuda.max_memory_allocated(device),"peak_reserved_bytes":torch.cuda.max_memory_reserved(device)}
    write_json(output/f"WORKER_{rank}_OF_{world_size}_{time.time_ns()}.json",result)
    return result


def finalize(output):
    import torch
    output=Path(output);reg=validate_registration(read_json(output/"EVAL_REGISTRATION.json"));entries=[];receipts=[]
    with lock(output/"FINALIZE.lock"):
        for chunk in shard_population(reg):
            tag=digest([reg["identity"],[r["scene_id"] for r in chunk]])
            receipt=output/"shards"/(tag+".COMPLETE.json")
            done=validate_shard(read_json(receipt),reg,chunk)
            records=torch.load(verify_artifact(done["shard"]),map_location="cpu",weights_only=False)
            if len(records)!=len(chunk):raise ValueError("DEV shard payload coverage mismatch")
            for record,row in zip(records,chunk):validate_feature_record(record,row,reg,compare_historical=True)
            entries.extend(done["entries"]);receipts.append(artifact(receipt))
        manifest={"schema":MANIFEST_SCHEMA,"status":"COMPLETE","split":"dev","role":"dev_report","for_training":False,
            "requested_scenes":FULL_SCENES,"requested_logs":FULL_LOGS,"dtype":"float32","evaluation_binding":reg["identity"],
            "registration":artifact(output/"EVAL_REGISTRATION.json"),"base_checkpoint_binding_hash":reg["base_checkpoint_binding_hash"],
            "numeric_runtime_hash":reg["numeric_runtime_hash"],"head_receipt":reg["head_receipt"],
            "shard_receipts":receipts,"entries":entries,"historical_physical_exact":True,"single_encoder_per_scene":True}
        manifest["identity"]=digest(manifest);validate_manifest(manifest,reg)
        write_json(output/"EVAL_CONTEXT_MANIFEST.json",manifest)
        result={"status":"COMPLETE","scenes":FULL_SCENES,"logs":FULL_LOGS,"manifest":artifact(output/"EVAL_CONTEXT_MANIFEST.json"),
                "payload_bytes":sum(read_json(p["path"])["shard"]["bytes"] for p in receipts),"new_action_head_bytes":0}
        write_json(output/"COMPLETE.json",result)
        return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",required=True);p.add_argument("--phase",choices=("register","collect","finalize","verify"),required=True)
    p.add_argument("--output",required=True);p.add_argument("--current-root");p.add_argument("--metric-index")
    p.add_argument("--historical-export");p.add_argument("--historical-score-identity")
    p.add_argument("--shard-scenes",type=int,default=16);p.add_argument("--hash-read-mib-per-second",type=float,default=128.)
    p.add_argument("--rank",type=int);p.add_argument("--world-size",type=int);p.add_argument("--device",default="cuda:0")
    p.add_argument("--max-seconds",type=float);p.add_argument("--resume",action="store_true");p.add_argument("--dry-run",action="store_true")
    a=p.parse_args(argv)
    try:
        cfg=load_config(a.config)
        if a.dry_run:print(json.dumps({"status":"NOT_RUN","phase":a.phase,"simulation_calls":0,"optimizer_updates":0}));return 0
        if a.phase=="register":
            if not all((a.current_root,a.metric_index,a.historical_export,a.historical_score_identity)):
                raise ValueError("DEV registration needs explicit existing current/metric/historical identities")
            result=register(cfg,current_root=a.current_root,metric_index=a.metric_index,historical_export=a.historical_export,
                historical_score_identity=a.historical_score_identity,output=a.output,shard_scenes=a.shard_scenes,
                read_mib_per_second=a.hash_read_mib_per_second)
        elif a.phase=="collect":
            if a.rank is None or a.world_size is None or a.max_seconds is None:
                raise ValueError("GPU extraction needs explicit rank/world/time bounds")
            result=collect(cfg,output=a.output,rank=a.rank,world_size=a.world_size,device=a.device,max_seconds=a.max_seconds,resume=a.resume)
        else:
            result=finalize(a.output)
        print(json.dumps(result));return 0 if result["status"] in ("COMPLETE","REGISTERED","WORKER_COMPLETE") else 1
    except Exception as exc:
        print(json.dumps({"status":"FAIL","error_type":type(exc).__name__,"error":str(exc)}));return 1


if __name__=="__main__":raise SystemExit(main())
