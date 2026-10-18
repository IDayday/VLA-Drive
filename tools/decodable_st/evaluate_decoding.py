"""Fixed train-only tiny population: native decoder agreement and DINO replay."""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import time
import torch
from safetensors.torch import load_file,save_file
from .train_representation import load_model,load_data,observation
from .runtime import numerical_policy
from starVLA.model.modules.decodable_st.training_state import TrainableDelta
from starVLA.model.modules.decodable_st.metrics import occupancy_counts,summarize_counts
from starVLA.teachers.decodable_st.vigt_adapter import load_vigt_decoder
from starVLA.teachers.decodable_st.source_registry import atomic_json


@torch.no_grad()
def run(run_root,output,device="cuda"):
    numerical_policy(42,device,.60);start=time.monotonic();run_root=Path(run_root);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    identity=json.loads((run_root/"identity.json").read_text());config=identity["config"]
    model,_,_=load_model(config,device,identity["arm"],identity["seed"])
    checkpoint=torch.load(run_root/"latest.pt",map_location="cpu",weights_only=False)
    if checkpoint["identity"]!=identity["identity"]:raise ValueError("Evaluation checkpoint identity mismatch")
    TrainableDelta(model,[r["name"] for r in identity["trainable_inventory"]]).load_state_dict(checkpoint["model"],strict=True)
    decoder,_=load_vigt_decoder(config["vigt_weights"],json.loads(Path(config["vigt_config"]).read_text()));decoder.to(device)
    data,rows,lookup=load_data(config);model.eval();outputs=[];counts=torch.zeros(5,dtype=torch.long)
    pairs=[];mean_context=None;fixed_xyz=None;action_drift=[]
    x=torch.linspace(.5,39.5,64,device=device);y=torch.linspace(-20,20,64,device=device)
    xx,yy=torch.meshgrid(x,y,indexing="ij")
    slices=torch.stack([torch.stack((xx,yy,torch.full_like(xx,z)),-1) for z in (.5,1.5)]).reshape(1,-1,3)
    def decode_chunks(context,xyz):
        return torch.cat([decoder(context,q).cpu() for q in xyz.split(1024,dim=1)],1)
    from starVLA.model.modules.foresight.losses import masked_regression
    for row in rows:
        example,labels=data[lookup[row["token"]]];representation=model(observation(example))
        teacher=load_file(str(Path(config["native_context_root"])/(row["token"]+".safetensors")))
        logits=decoder(representation.predicted_spatial_context,teacher["xyz"].to(device)).cpu()
        if fixed_xyz is None:fixed_xyz=teacher["xyz"]
        elif not torch.equal(fixed_xyz,teacher["xyz"]):raise ValueError("Negative-control query coordinates differ")
        pairs.append((logits,teacher["logits"]))
        mean_context=teacher["context"].clone()/len(rows) if mean_context is None else mean_context+teacher["context"]/len(rows)
        c=occupancy_counts(logits>=0,teacher["logits"]>=0,torch.ones_like(logits,dtype=torch.bool));counts+=c
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
    mean_logits=decoder(mean_context.to(device),fixed_xyz.to(device)).cpu()
    constant_logits=decoder(torch.zeros_like(mean_context,device=device),fixed_xyz.to(device)).cpu()
    controls={k:torch.zeros(5,dtype=torch.long) for k in ("scene_shuffled_student","mean_training_teacher_context","zero_context")}
    for i,(_,target) in enumerate(pairs):
        for key,value in [("scene_shuffled_student",pairs[(i+1)%len(pairs)][0]),("mean_training_teacher_context",mean_logits),("zero_context",constant_logits)]:
            controls[key]+=occupancy_counts(value>=0,target>=0,torch.ones_like(target,dtype=torch.bool))
    result={"status":"INCONCLUSIVE","arm":identity["arm"],"run_identity":identity["identity"],"progress":checkpoint["progress"],
        "population":"same 64 training scenes used for tiny fit; NOT a held-out ability claim","scenes":len(rows),
        "native_teacher_agreement":summarize_counts(counts),"mean_DINO_MSE":sum(x["DINO_MSE"] for x in outputs)/len(outputs),
        "negative_controls":{k:summarize_counts(v) for k,v in controls.items()},
        "same_noise_no_reader_action_drift_train8":action_drift,
        "per_scene":outputs,"seconds":time.monotonic()-start,"gpu_hours":(time.monotonic()-start)/3600,
        "independent_GT_occupancy":"BLOCKED_OBSERVED_FREE_PROVENANCE","planning":"NOT_RUN"}
    atomic_json(output/"DECODING_REPORT.json",result);return result


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--run-root",required=True);p.add_argument("--output",required=True);p.add_argument("--device",default="cuda")
    a=p.parse_args();run(a.run_root,a.output,a.device)
