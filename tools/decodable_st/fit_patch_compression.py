"""Train-only PCA initialization of the existing 32x32 linear patch bottleneck.

This changes the optimizer/initialization, not the previously registered capacity.
Native decoder agreement remains distinct from independent occupancy quality.
"""
import argparse
import json
from pathlib import Path
import time
import torch
from torch.nn import functional as F
from safetensors.torch import load_file,save_file
from starVLA.teachers.decodable_st.source_registry import atomic_json,sha256
from starVLA.teachers.decodable_st.vigt_adapter import load_vigt_decoder
from starVLA.model.modules.decodable_st.vigt_readout import ContextPatchBottleneck
from .runtime import numerical_policy,source_sha,write_identity,parameter_digest
from .train_compression import evaluate


def run(config,output,device="cuda"):
    numerical_policy(42,device,.75);start=time.monotonic();output=Path(output)
    if output.exists():raise FileExistsError("PCA experiment requires a new output identity")
    root=Path(config["native_context_root"]);receipt=json.loads((root/"ROUNDTRIP.json").read_text())
    if len(receipt["records"])!=64 or receipt["status"]!="PASS":raise ValueError("Registered native64 required")
    rows=[]
    for row in receipt["records"]:
        p=root/(row["token"]+".safetensors")
        if sha256(p)!=row["sha256"]:raise ValueError("Modified native context")
        rows.append(load_file(str(p)))
    cfg=json.loads(Path(config["config"]).read_text());decoder,_=load_vigt_decoder(config["weights"],cfg)
    decoder=decoder.to(device);fixed=parameter_digest(decoder)
    model=ContextPatchBottleneck(2048,decoder.contract,(32,32)).to(device)
    identity=write_identity(output,{"schema":"existing_patch_capacity_PCA_v1","source_sha":source_sha(),"config":config,
        "source_receipt_sha256":sha256(root/"ROUNDTRIP.json"),"grid":[32,32],"hidden":2048,"seed":42,
        "fit_tokens":[r["token"] for r in receipt["records"][:48]],"held_tokens":[r["token"] for r in receipt["records"]][48:],
        "algorithm":"torch.pca_lowrank centered rank2048 niter2 FP32","same_architecture_as":"compression32_patch_frozen_v2",
        "teacher_GT_quality":"NOT_ESTABLISHED"})
    atomic_json(output/"state.json",{"status":"RUNNING","stage":"building_training_patch_matrix"})
    patches=torch.cat([F.unfold(r["context"],kernel_size=(8,8),stride=(8,8)).transpose(1,2).reshape(-1,16384) for r in rows[:48]],0).to(device)
    mu=patches.mean(0);patches.sub_(mu)
    torch.cuda.synchronize();pca_start=time.monotonic()
    atomic_json(output/"state.json",{"status":"RUNNING","stage":"PCA","matrix_shape":list(patches.shape)})
    with torch.no_grad():
        _,s,v=torch.pca_lowrank(patches,q=2048,center=False,niter=2)
        model.encode.weight.copy_(v.T);model.encode.bias.copy_(-mu@v)
        model.readout.project.weight.copy_(v);model.readout.project.bias.copy_(mu)
    torch.cuda.synchronize();pca_seconds=time.monotonic()-pca_start
    total_energy=patches.square().sum().item();explained=s.square().sum().item()/total_energy
    del patches,v,s;torch.cuda.empty_cache()
    train_mean=sum(r["context"] for r in rows[:48])/48
    metrics=evaluate(model,decoder,rows[48:],train_mean,device)
    atomic_json(output/"METRICS.json",metrics)
    if parameter_digest(decoder)!=fixed:raise AssertionError("Native decoder changed")
    save_file({k:v.cpu().contiguous() for k,v in model.state_dict().items()},str(output/"bottleneck.safetensors.tmp"))
    (output/"bottleneck.safetensors.tmp").replace(output/"bottleneck.safetensors")
    elapsed=time.monotonic()-start
    result={"status":"INCONCLUSIVE","fit_complete":True,"identity":identity,"pca_seconds":pca_seconds,
        "train_patch_explained_energy":explained,"seconds":elapsed,"gpu_hours":elapsed/3600,
        "peak_allocated_bytes":torch.cuda.max_memory_allocated(),"weights_sha256":sha256(output/"bottleneck.safetensors"),
        "decoder_unchanged":True,"native_agreement":metrics["teacher_agreement"]["reconstructed"],
        "scope":"same-capacity compression learning test; no target-domain or student capability acceptance"}
    atomic_json(output/"RESULT.json",result);atomic_json(output/"state.json",{"status":"INCONCLUSIVE","fit_complete":True})
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--output",required=True)
    p.add_argument("--device",default="cuda");a=p.parse_args();run(json.loads(Path(a.config).read_text()),a.output,a.device)
