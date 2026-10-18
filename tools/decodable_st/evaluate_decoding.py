"""Fixed train-only tiny population: native decoder agreement and DINO replay."""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import time
import torch
from safetensors.torch import load_file,save_file
from .train_representation import load_model,load_data,observation
from .runtime import numerical_policy,source_sha
from starVLA.model.modules.decodable_st.training_state import TrainableDelta
from starVLA.model.modules.decodable_st.metrics import occupancy_counts,summarize_counts
from starVLA.teachers.decodable_st.vigt_adapter import load_vigt_decoder
from starVLA.teachers.decodable_st.source_registry import atomic_json,sha256


@torch.no_grad()
def run(run_root,output,device="cuda",evaluation_config=None,required_update=None,wait_seconds=0):
    numerical_policy(42,device,.60);start=time.monotonic();run_root=Path(run_root);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    identity=json.loads((run_root/"identity.json").read_text());config=identity["config"]
    model,_,_=load_model(config,device,identity["arm"],identity["seed"])
    # A finite evaluation worker can overlap baseline loading with the final
    # training segment. It never evaluates an earlier/best checkpoint and never
    # dispatches or resumes training. Only the atomic endpoint receipt is read.
    deadline=time.monotonic()+wait_seconds
    if required_update is not None:
        while True:
            receipt=run_root/"RESULT.json"
            completed=json.loads(receipt.read_text()) if receipt.exists() else {}
            if completed.get("update")==required_update:break
            if time.monotonic()>=deadline:raise TimeoutError("Registered common endpoint not available")
            time.sleep(min(10,max(0,deadline-time.monotonic())))
    checkpoint=torch.load(run_root/"latest.pt",map_location="cpu",weights_only=False)
    if checkpoint["identity"]!=identity["identity"]:raise ValueError("Evaluation checkpoint identity mismatch")
    if required_update is not None and checkpoint["progress"]["update"]!=required_update:raise ValueError("Endpoint changed after receipt")
    TrainableDelta(model,[r["name"] for r in identity["trainable_inventory"]]).load_state_dict(checkpoint["model"],strict=True)
    decoder,_=load_vigt_decoder(config["vigt_weights"],json.loads(Path(config["vigt_config"]).read_text()));decoder.to(device)
    data_config=dict(config)
    if evaluation_config:
        if set(evaluation_config)!={"current_root","selection","native_context_root"}:raise ValueError("Held-out override may change only population and targets")
        baseline=json.loads(Path(config["baseline_manifest"]).read_text())
        if evaluation_config["current_root"]!=baseline["data"]["dev"]["root"]:raise ValueError("Only registered development data admitted")
        training=json.loads(Path(config["selection"]).read_text());held=json.loads(Path(evaluation_config["selection"]).read_text())
        if {r["log"] for r in training}&{r["log"] for r in held}:raise ValueError("Tiny training/development log overlap")
        data_config.update(evaluation_config)
    data,rows,lookup=load_data(data_config);model.eval();outputs=[];counts=torch.zeros(5,dtype=torch.long)
    pairs=[];mean_context=None;fixed_xyz=None;action_drift=[]
    regions={name:torch.zeros(5,dtype=torch.long) for name in
             ("near_0_15m","far_15_40m","teacher_boundary")}
    x=torch.linspace(.5,39.5,64,device=device);y=torch.linspace(-20,20,64,device=device)
    xx,yy=torch.meshgrid(x,y,indexing="ij")
    slices=torch.stack([torch.stack((xx,yy,torch.full_like(xx,z)),-1) for z in (.5,1.5)]).reshape(1,-1,3)
    def decode_chunks(context,xyz):
        return torch.cat([decoder(context,q).cpu() for q in xyz.split(1024,dim=1)],1)
    from starVLA.model.modules.foresight.losses import masked_regression
    for row in rows:
        example,labels=data[lookup[row["token"]]];representation=model(observation(example))
        teacher=load_file(str(Path(data_config["native_context_root"])/(row["token"]+".safetensors")))
        logits=decoder(representation.predicted_spatial_context,teacher["xyz"].to(device)).cpu()
        if fixed_xyz is None:fixed_xyz=teacher["xyz"]
        elif not torch.equal(fixed_xyz,teacher["xyz"]):raise ValueError("Negative-control query coordinates differ")
        pairs.append((logits,teacher["logits"]))
        if not evaluation_config:
            mean_context=teacher["context"].clone()/len(rows) if mean_context is None else mean_context+teacher["context"]/len(rows)
        c=occupancy_counts(logits>=0,teacher["logits"]>=0,torch.ones_like(logits,dtype=torch.bool));counts+=c
        masks={"near_0_15m":teacher["xyz"][...,0:1]<15,
               "far_15_40m":teacher["xyz"][...,0:1]>=15,
               "teacher_boundary":teacher["logits"].abs()<2.}
        for name,mask in masks.items():
            regions[name]+=occupancy_counts(logits>=0,teacher["logits"]>=0,mask)
        dino,_=masked_regression(model.dino_prediction(representation),labels["current_dino"][None].to(device),labels["current_dino_valid"][None,:,None].to(device))
        outputs.append({**row,"native_teacher_agreement":summarize_counts(c),"DINO_MSE":dino.item(),
             "feature_rmse":(representation.predicted_spatial_context.cpu()-teacher["context"]).square().mean().sqrt().item()})
        # Only bounded low-resolution samples are retained, not dense full-dataset BEV.
        save_file({"xyz":teacher["xyz"],"student_logits":logits,"teacher_logits":teacher["logits"]},str(output/(row["token"]+".safetensors")))
        if len(outputs)<=32:
            save_file({"xyz":slices.cpu(),"student_logits":decode_chunks(representation.predicted_spatial_context,slices),
                       "teacher_logits":decode_chunks(teacher["context"].to(device),slices)},str(output/(row["token"]+".slices.safetensors")))
        if len(outputs)<=8:
            from tools.foresight.checkpoints import scene_noise
            from starVLA.dataloader.foresight_dataset import decode_ego
            noise=scene_noise(row["token"],42,device)
            new_condition=model.action_condition(representation) # reader remains disabled
            new=decode_ego(model.baseline.action_model.predict_action(new_condition,initial_noise=noise.clone()).float())
            lm=model.baseline.qwen_vl_interface.model.model.language_model
            with lm.disable_adapter() if hasattr(lm,"disable_adapter") else nullcontext():
                old=model.baseline.predict_action([example],initial_noise=noise.clone())
            action_drift.append({"token":row["token"],"mean_xy_displacement_m":(new[...,:2]-old[...,:2]).norm(dim=-1).mean().item(),
                                 "max_abs_component_change":(new-old).abs().max().item()})
    if evaluation_config:
        # Negative-control mean is fitted exclusively on the original train64.
        train_rows=json.loads(Path(config["selection"]).read_text())
        for row in train_rows:
            c=load_file(str(Path(config["native_context_root"])/(row["token"]+".safetensors")))["context"]
            mean_context=c.clone()/len(train_rows) if mean_context is None else mean_context+c/len(train_rows)
    mean_logits=decoder(mean_context.to(device),fixed_xyz.to(device)).cpu()
    constant_logits=decoder(torch.zeros_like(mean_context,device=device),fixed_xyz.to(device)).cpu()
    controls={k:torch.zeros(5,dtype=torch.long) for k in ("scene_shuffled_student","mean_training_teacher_context","zero_context")}
    for i,(_,target) in enumerate(pairs):
        for key,value in [("scene_shuffled_student",pairs[(i+1)%len(pairs)][0]),("mean_training_teacher_context",mean_logits),("zero_context",constant_logits)]:
            controls[key]+=occupancy_counts(value>=0,target>=0,torch.ones_like(target,dtype=torch.bool))
    result={"status":"INCONCLUSIVE","arm":identity["arm"],"run_identity":identity["identity"],"progress":checkpoint["progress"],
        "evaluation_source_sha":source_sha(),"training_source_sha":identity["source_sha"],
        "selection_sha256":sha256(data_config["selection"]),"baseline_identity":identity["baseline_identity"],
        "population":"fixed independent development logs; teacher agreement only" if evaluation_config else "same 64 training scenes used for tiny fit; NOT a held-out ability claim","scenes":len(rows),
        "native_teacher_agreement":summarize_counts(counts),"mean_DINO_MSE":sum(x["DINO_MSE"] for x in outputs)/len(outputs),
        "regions_native_teacher_agreement":{k:summarize_counts(v) for k,v in regions.items()},
        "region_definition":"Same fixed xyz as native compression audit; boundary means abs(native teacher logit)<2, not an independent physical boundary label",
        "negative_controls":{k:summarize_counts(v) for k,v in controls.items()},
        "same_noise_no_reader_action_drift_train8":action_drift,
        "per_scene":outputs,"seconds":time.monotonic()-start,"gpu_hours":(time.monotonic()-start)/3600,
        "independent_GT_occupancy":"BLOCKED_OBSERVED_FREE_PROVENANCE","planning":"NOT_RUN"}
    atomic_json(output/"DECODING_REPORT.json",result);return result


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--run-root",required=True);p.add_argument("--output",required=True);p.add_argument("--device",default="cuda")
    p.add_argument("--evaluation-config")
    p.add_argument("--required-update",type=int);p.add_argument("--wait-seconds",type=float,default=0)
    a=p.parse_args()
    if not 0<=a.wait_seconds<=1800:raise ValueError("Evaluation wait is finite and capped at 30 minutes")
    run(a.run_root,a.output,a.device,json.loads(Path(a.evaluation_config).read_text()) if a.evaluation_config else None,a.required_update,a.wait_seconds)
