"""Fixed train-only tiny population: native decoder agreement and DINO replay."""
import argparse
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
    from starVLA.model.modules.foresight.losses import masked_regression
    for row in rows:
        example,labels=data[lookup[row["token"]]];representation=model(observation(example))
        teacher=load_file(str(Path(config["native_context_root"])/(row["token"]+".safetensors")))
        logits=decoder(representation.predicted_spatial_context,teacher["xyz"].to(device)).cpu()
        c=occupancy_counts(logits>=0,teacher["logits"]>=0,torch.ones_like(logits,dtype=torch.bool));counts+=c
        dino,_=masked_regression(model.dino_prediction(representation),labels["current_dino"][None].to(device),labels["current_dino_valid"][None,:,None].to(device))
        outputs.append({**row,"native_teacher_agreement":summarize_counts(c),"DINO_MSE":dino.item(),
             "feature_rmse":(representation.predicted_spatial_context.cpu()-teacher["context"]).square().mean().sqrt().item()})
        # Only bounded low-resolution samples are retained, not dense full-dataset BEV.
        save_file({"xyz":teacher["xyz"],"student_logits":logits,"teacher_logits":teacher["logits"]},str(output/(row["token"]+".safetensors")))
    result={"status":"INCONCLUSIVE","arm":identity["arm"],"run_identity":identity["identity"],"progress":checkpoint["progress"],
        "population":"same 64 training scenes used for tiny fit; NOT a held-out ability claim","scenes":len(rows),
        "native_teacher_agreement":summarize_counts(counts),"mean_DINO_MSE":sum(x["DINO_MSE"] for x in outputs)/len(outputs),
        "per_scene":outputs,"seconds":time.monotonic()-start,"gpu_hours":(time.monotonic()-start)/3600,
        "independent_GT_occupancy":"BLOCKED_OBSERVED_FREE_PROVENANCE","planning":"NOT_RUN"}
    atomic_json(output/"DECODING_REPORT.json",result);return result


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--run-root",required=True);p.add_argument("--output",required=True);p.add_argument("--device",default="cuda")
    a=p.parse_args();run(a.run_root,a.output,a.device)
