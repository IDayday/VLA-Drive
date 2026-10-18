"""One train-only gradient-scale calibration before the matched patch P1 runs."""
import argparse
import json
from pathlib import Path
import statistics
import time
import torch
from safetensors.torch import load_file
from torch.nn import functional as F
from .runtime import numerical_policy,source_sha
from .train_representation import load_model,load_data,observation
from starVLA.teachers.decodable_st.source_registry import atomic_json,sha256
from starVLA.teachers.decodable_st.vigt_adapter import load_vigt_decoder
from starVLA.model.modules.foresight.losses import masked_regression


def run(config,output,device="cuda"):
    numerical_policy(42,device,.8);start=time.monotonic();output=Path(output)
    if output.exists():raise FileExistsError("Calibration is finite and versioned; no overwriting")
    output.mkdir(parents=True)
    model,baseline,_=load_model(config,device,"RS",42);model.eval()
    shared=[p for n,p in model.named_parameters() if p.requires_grad and "lora_B" in n]
    if not shared:raise ValueError("Actual shared LoRA_B parameters required")
    decoder,_=load_vigt_decoder(config["vigt_weights"],json.loads(Path(config["vigt_config"]).read_text()));decoder.to(device)
    data,rows,lookup=load_data(config)
    means=[];squares=[]
    for row in rows:
        c=load_file(str(Path(config["native_context_root"])/(row["token"]+".safetensors")))["context"]
        means.append(c.mean((0,2,3)));squares.append(c.square().mean((0,2,3)))
    std=(torch.stack(squares).mean(0)-torch.stack(means).mean(0).square()).clamp_min(1e-6).sqrt()[None,:,None,None].to(device)
    results=[]
    for row in rows[:8]:
        example,targets=data[lookup[row["token"]]]
        native=load_file(str(Path(config["native_context_root"])/(row["token"]+".safetensors")))
        rep=model(observation(example));dino,_=masked_regression(model.dino_prediction(rep),targets["current_dino"][None].to(device),targets["current_dino_valid"][None,:,None].to(device))
        values={"DINO":dino,"spatial_feature":((rep.predicted_spatial_context-native["context"].to(device))/std).square().mean(),
            "spatial_decode":F.binary_cross_entropy_with_logits(decoder(rep.predicted_spatial_context,native["xyz"].to(device)),native["logits"].to(device).sigmoid())}
        vectors={}
        for key,value in values.items():
            gradients=torch.autograd.grad(value,shared,retain_graph=True,allow_unused=True)
            vectors[key]=torch.cat([torch.zeros_like(p).flatten() if g is None else g.detach().float().flatten() for p,g in zip(shared,gradients)])
        norms={k:float(v.norm()) for k,v in vectors.items()}
        if not all(torch.isfinite(v).all() for v in vectors.values()):raise FloatingPointError("Nonfinite raw calibration gradients")
        cos={k:float(F.cosine_similarity(vectors["DINO"],v,dim=0)) for k,v in vectors.items() if k!="DINO"}
        results.append({"token":row["token"],"raw_losses":{k:float(v.detach()) for k,v in values.items()},"shared_raw_gradient_norms":norms,"cosine_to_DINO":cos})
        atomic_json(output/"progress.json",{"completed":len(results),"requested":8})
        del rep,values,vectors,gradients
    median={k:statistics.median(r["shared_raw_gradient_norms"][k] for r in results) for k in ("DINO","spatial_feature","spatial_decode")}
    if median["DINO"]<=0:raise ValueError("Zero shared DINO gradient; calibration cannot synthesize a scale")
    weights={"DINO":1.,**{k:min(.1,.5*median["DINO"]/max(median[k],1e-30)) for k in ("spatial_feature","spatial_decode")}}
    report={"status":"PASS","scope":"gradient calibration only, no capability acceptance","source_sha":source_sha(),
        "baseline_identity":baseline["identity"],"rows":results,"median_shared_raw_gradient_norms":median,"loss_weights":weights,
        "selection_rule":"Each new task median weighted shared gradient <= 0.5 DINO, maximum weight .1; one fixed train8 calibration",
        "peak_allocated_bytes":torch.cuda.max_memory_allocated(),"seconds":time.monotonic()-start,"gpu_hours":(time.monotonic()-start)/3600}
    atomic_json(output/"CALIBRATION.json",report)
    resolved={**config,"loss_weights":{**config["loss_weights"],**weights},"loss_calibration":{"report_sha256":sha256(output/"CALIBRATION.json"),"train_scene_count":8,"source_sha":source_sha()}}
    atomic_json(output/"resolved_tiny_job.json",resolved);return report


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--output",required=True)
    p.add_argument("--device",default="cuda");a=p.parse_args();run(json.loads(Path(a.config).read_text()),a.output,a.device)
