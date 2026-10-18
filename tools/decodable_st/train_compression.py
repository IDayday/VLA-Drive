"""Real native ViGT bottleneck training; teacher agreement is NOT GT accuracy."""
import argparse
import json
import math
from pathlib import Path
import time
import torch
from torch.nn import functional as F
from safetensors.torch import load_file
from starVLA.teachers.decodable_st.vigt_adapter import load_vigt_decoder
from starVLA.teachers.decodable_st.source_registry import atomic_json,sha256
from starVLA.model.modules.decodable_st.vigt_readout import ContextBottleneck,ContextPatchBottleneck
from starVLA.model.modules.decodable_st.metrics import occupancy_counts,summarize_counts
from .runtime import (source_sha,parameter_digest,numerical_policy,write_identity,
                      save_checkpoint,load_checkpoint)


@torch.no_grad()
def evaluate(model,decoder,rows,mean_context,device):
    metrics={name:torch.zeros(5,dtype=torch.long) for name in ("reconstructed","pooled_only","mean","scene_shuffle")}
    groups={name:torch.zeros(5,dtype=torch.long) for name in ("near_0_15m","far_15_40m","teacher_boundary")}
    errors=[]
    model.eval()
    for i,row in enumerate(rows):
        c=row["context"].to(device);xyz=row["xyz"].to(device);target=row["logits"].to(device)
        rebuilt=model(c);pred=decoder(rebuilt,xyz)
        pooled=F.interpolate(F.adaptive_avg_pool2d(c,model.grid_hw),c.shape[-2:],mode="bilinear",align_corners=False)
        values={"reconstructed":pred,"pooled_only":decoder(pooled,xyz),"mean":decoder(mean_context.to(device),xyz),
                "scene_shuffle":decoder(rows[(i+1)%len(rows)]["context"].to(device),xyz)}
        # Native logit zero = p=.5. This compares A/B only, no safety calibration.
        occupied=target>=0
        for name,value in values.items():metrics[name]+=occupancy_counts(value>=0,occupied,torch.ones_like(occupied,dtype=torch.bool))
        masks={"near_0_15m":xyz[...,0:1]<15,"far_15_40m":xyz[...,0:1]>=15,
               "teacher_boundary":target.abs()<2.}
        for name,mask in masks.items():groups[name]+=occupancy_counts(pred>=0,occupied,mask)
        errors.append({"feature_rmse":(rebuilt-c).square().mean().sqrt().item(),
                       "logit_mae":(pred-target).abs().mean().item()})
    return {"teacher_agreement":{k:summarize_counts(v) for k,v in metrics.items()},
            "regions":{k:summarize_counts(v) for k,v in groups.items()},
            "mean_feature_rmse":sum(x["feature_rmse"] for x in errors)/len(errors),
            "mean_logit_mae":sum(x["logit_mae"] for x in errors)/len(errors),
            "population":len(rows),"teacher_GT_quality":"NOT_ESTABLISHED"}


def run(config,output,*,device="cuda",max_updates=1000,grid=16,resume=False,stop_after=None,readout="bilinear"):
    if max_updates>1000 or grid not in (16,32):raise ValueError("Bounded registered capacity pilot only")
    numerical_policy(42,device,.50);start=time.monotonic();output=Path(output)
    cache=Path(config["native_context_root"])
    receipt=json.loads((cache/"ROUNDTRIP.json").read_text())
    if receipt["status"]!="PASS":raise ValueError("Native context not verified")
    cfg=json.loads(Path(config["config"]).read_text())
    decoder,load_receipt=load_vigt_decoder(config["weights"],cfg);decoder.to(device)
    frozen_digest=parameter_digest(decoder)
    rows=[]
    for record in receipt["records"]:
        path=cache/(record["token"]+".safetensors")
        if sha256(path)!=record["sha256"]:raise ValueError("Native context payload changed")
        rows.append(load_file(str(path)))
    # Both sets are train-split logs. The held-out 16 are not development/test data.
    train,held=rows[:48],rows[48:64]
    if len(train)!=48 or len(held)!=16:raise ValueError("Registered 64-scene population required")
    mean_context=torch.stack([x["context"] for x in train]).mean(0)
    channel_mean=torch.stack([x["context"].mean((0,2,3)) for x in train]).mean(0)
    channel_sq=torch.stack([x["context"].square().mean((0,2,3)) for x in train]).mean(0)
    std=(channel_sq-channel_mean.square()).clamp_min(1e-6).sqrt()[None,:,None,None].to(device)
    if readout not in ("bilinear","patch"):raise ValueError("Unknown capacity mapping")
    if readout=="patch" and grid!=32:raise ValueError("Second expansion is fixed at 1024 queries")
    model=(ContextBottleneck if readout=="bilinear" else ContextPatchBottleneck)(2048,decoder.contract,(grid,grid)).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=.001)
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda u:min((u+1)/200,1.)*(.1+.9*.5*(1+math.cos(math.pi*min(u,max_updates)/max_updates))))
    identity=write_identity(output,{"schema":"vigt_compression_pilot_v1","source_sha":source_sha(),"config":config,
       "source_receipt_hash":sha256(cache/"ROUNDTRIP.json"),"max_updates":max_updates,"grid":grid,"seed":42,"readout":readout,
       "feature_standardization":"train48 channel std","loss":{"feature":1.,"teacher_logit_soft_bce":1.},
       "batch":1,"precision":"FP32 TF32 disabled","load_receipt":load_receipt,
       "train_tokens":[r["token"] for r in receipt["records"][:48]],"held_tokens":[r["token"] for r in receipt["records"]][48:64],
       "scope":"compression upper bound; decoder teacher agreement, not student or GT capability"})
    sampler=torch.Generator().manual_seed(4242);order=torch.randperm(len(train),generator=sampler).tolist();offset=0;step=0;exposure=0
    if resume:
        state=load_checkpoint(output/"latest.pt",model=model,optimizer=optimizer,scheduler=scheduler,identity=identity)
        step=state["progress"]["update"];exposure=state["progress"]["scene_exposures"]
        order=state["sampler"]["order"];offset=state["sampler"]["offset"];sampler.set_state(state["sampler"]["rng"])
    else:
        atomic_json(output/"initial_metrics.json",evaluate(model,decoder,held,mean_context,device))
    atomic_json(output/"state.json",{"status":"RUNNING","update":step})
    logs=open(output/"train.jsonl","a",buffering=1);window=time.monotonic()
    try:
        while step<max_updates:
            if offset==len(order):order=torch.randperm(len(train),generator=sampler).tolist();offset=0
            row=train[order[offset]];offset+=1
            c=row["context"].to(device);xyz=row["xyz"].to(device);target=row["logits"].to(device)
            model.train();optimizer.zero_grad(set_to_none=True)
            pred=model(c);logits=decoder(pred,xyz)
            feature=((pred-c)/std).square().mean()
            decode=F.binary_cross_entropy_with_logits(logits,target.sigmoid())
            loss=feature+decode
            if not torch.isfinite(loss):raise FloatingPointError("Compression loss nonfinite")
            loss.backward();grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            optimizer.step();scheduler.step();step+=1;exposure+=1
            if step%20==0 or step==1:
                logs.write(json.dumps({"update":step,"scene_exposures":exposure,"feature":feature.item(),"decode":decode.item(),
                            "gradient_norm":grad.item(),"lr":optimizer.param_groups[0]["lr"],"window_seconds":time.monotonic()-window})+"\n")
                window=time.monotonic()
            stop=(output/"STOP_REQUESTED").exists() or (stop_after is not None and step>=stop_after)
            if step%200==0 or step==max_updates or stop:
                save_checkpoint(output/"latest.pt",model=model,optimizer=optimizer,scheduler=scheduler,identity=identity,
                    progress={"update":step,"scene_exposures":exposure,"phase":"tiny_compression"},
                    sampler={"order":order,"offset":offset,"rng":sampler.get_state()})
                metric=evaluate(model,decoder,held,mean_context,device)
                atomic_json(output/f"metrics_{step:06d}.json",metric)
                atomic_json(output/"state.json",{"status":"RUNNING" if step<max_updates else "INCONCLUSIVE","update":step})
            if stop:break
        if parameter_digest(decoder)!=frozen_digest:raise AssertionError("Fixed decoder changed")
        result={"status":"INCONCLUSIVE","training_complete":step==max_updates,"update":step,"scene_exposures":exposure,
            "seconds":time.monotonic()-start,"gpu_hours":(time.monotonic()-start)/3600,
            "peak_allocated_bytes":torch.cuda.max_memory_allocated() if device.startswith("cuda") else 0,
            "decoder_unchanged":True,"scope":"teacher-agreement capacity test; independent target-domain GT still required"}
        atomic_json(output/"RESULT.json",result);return result
    finally:logs.flush();logs.close()


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--output",required=True)
    p.add_argument("--device",default="cuda");p.add_argument("--max-updates",type=int,default=1000)
    p.add_argument("--grid",type=int,default=16);p.add_argument("--readout",choices=("bilinear","patch"),default="bilinear")
    p.add_argument("--resume",action="store_true");p.add_argument("--stop-after",type=int)
    a=p.parse_args();run(json.loads(Path(a.config).read_text()),a.output,device=a.device,max_updates=a.max_updates,
                         grid=a.grid,resume=a.resume,stop_after=a.stop_after,readout=a.readout)
