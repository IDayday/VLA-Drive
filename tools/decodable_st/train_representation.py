"""Real current-image, fixed-native-decoder representation training.

The bounded pilot is teacher-agreement exploration, never a teacher-quality PASS.
Formal entry requires independent teacher/label and compression acceptance.
"""
import argparse
import json
import math
from pathlib import Path
import time
import torch
from torch.nn import functional as F
from safetensors.torch import load_file
from starVLA.teachers.decodable_st.source_registry import atomic_json,sha256
from .runtime import numerical_policy,source_sha,write_identity,save_checkpoint,load_checkpoint,parameter_digest


def load_model(config,device,arm,seed):
    from tools.foresight.checkpoints import load_student
    from starVLA.model.framework.ddp_decodable_st import DDPDecodableST
    from starVLA.model.modules.decodable_st.contracts import SpatialContract,TemporalContract
    from starVLA.model.modules.decodable_st.student_queries import select_tail_lora
    m=json.loads(Path(config["baseline_manifest"]).read_text())
    if m["status"]!="IDENTITY_VERIFIED":raise ValueError("BLOCKED_BASELINE_IDENTITY")
    identity=json.loads((Path(m["run"])/"identity.json").read_text())
    base=load_student(m["run"],m["tag"],identity,device=device,precision="fp32",strip=False).requires_grad_(False).eval()
    spatial=SpatialContract(256,256,256,(-40.,-40.,-1.),(40.,40.,5.4)) if arm not in ("R0","RT") else None
    temporal=None
    if arm in ("RT","RST","Rjoint"):
        t=config.get("temporal_contract")
        if not t:raise ValueError("BLOCKED_TEMPORAL_CONTRACT")
        temporal=TemporalContract(**t)
        if t["task_version"].startswith("T-driving"):temporal.require_driving()
    model=DDPDecodableST(base,spatial_contract=spatial,temporal_contract=temporal,
                         spatial_grid=(config.get("spatial_grid",16),)*2,seed=seed).to(device)
    model.bridge.requires_grad_(False)
    names=[]
    if arm!="Rfrozen":
        lm=base.qwen_vl_interface.model.model.language_model
        wrapped,names=select_tail_lora(lm,rank=32,last_layers=8,seed=seed+4201)
        base.qwen_vl_interface.model.model.language_model=wrapped
    # KV is local and differentiable; recomputing a cached layer would mutate it.
    base.qwen_vl_interface.model.gradient_checkpointing_disable()
    model.eval()
    return model,m,names


def load_data(config):
    from starVLA.dataloader.full_foresight_dataset import FullForesightDataset
    root=config["current_root"]
    d=FullForesightDataset(root,candidate="C1",current=True,future=False,dino_root=config["dino_root"],
        dino_index=config["dino_index"],expected_dino=config["dino_identity"])
    lookup={r["token"]:i for i,r in enumerate(d.index)}
    selected=json.loads(Path(config["selection"]).read_text())
    if not selected or any(x["token"] not in lookup for x in selected):raise ValueError("Illegal training population")
    return d,selected,lookup


def observation(example):
    from starVLA.model.modules.decodable_st.contracts import AllowedObservation
    return AllowedObservation((tuple(example["image"]),),(example["lang"],),
                torch.as_tensor(example["state"],dtype=torch.float32)[None],("CAM_F0","CAM_L0","CAM_R0"))


def run(config,output,*,arm="RS",seed=42,device="cuda",max_updates=1000,resume=False,stop_after=None):
    if arm not in ("R0","RS","Rfrozen","Rcap","Rdecode","RT","RST","Rjoint","Rgt"):
        raise ValueError("Unknown experiment arm")
    if config.get("scope")!="tiny" and not config.get("formal_acceptance"):
        raise ValueError("Formal representation training needs independent signed identity-bound acceptance")
    if arm in ("Rgt","Rjoint"):
        raise ValueError("NOT_IMPLEMENTED_FORMAL_ARM: GT-only/joint not silently approximated by RS")
    if max_updates>1000 and config.get("scope")=="tiny":raise ValueError("Tiny pilot update cap")
    numerical_policy(seed,device,.80);start=time.monotonic();output=Path(output)
    model,manifest,lora_modules=load_model(config,device,arm,seed)
    from starVLA.teachers.decodable_st.vigt_adapter import load_vigt_decoder
    from starVLA.teachers.decodable_st.sgdrive_adapter import load_sgdrive_decoder
    from starVLA.model.modules.decodable_st.training_state import TrainableDelta
    from starVLA.model.modules.foresight.losses import masked_regression
    decoder=None;temporal_decoder=None
    if model.spatial_contract is not None and arm!="Rcap":
        decoder,_=load_vigt_decoder(config["vigt_weights"],json.loads(Path(config["vigt_config"]).read_text()));decoder.to(device)
    if model.temporal_contract is not None:
        temporal_decoder,_=load_sgdrive_decoder(config["sgdrive_weights"],json.loads(Path(config["sgdrive_config"]).read_text()),future=True)
        temporal_decoder.to(device)
    fixed_digests=[parameter_digest(d) for d in (decoder,temporal_decoder) if d is not None]
    data,selected,lookup=load_data(config)
    target_rows={}
    for row in selected:
        tensors={}
        if decoder is not None:tensors.update(load_file(str(Path(config["native_context_root"])/(row["token"]+".safetensors"))))
        if temporal_decoder is not None:
            tensors.update(load_file(str(Path(config["native_temporal_root"])/(row["token"]+".safetensors"))))
        target_rows[row["token"]]=tensors
    std=None
    if decoder is not None:
        mean=torch.stack([t["context"].mean((0,2,3)) for t in target_rows.values()]).mean(0)
        square=torch.stack([t["context"].square().mean((0,2,3)) for t in target_rows.values()]).mean(0)
        std=(square-mean.square()).clamp_min(1e-6).sqrt()[None,:,None,None].to(device)
    parameters=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
    delta=TrainableDelta(model,[n for n,p in parameters])
    optimizer=torch.optim.AdamW([p for _,p in parameters],lr=1e-4,weight_decay=.001)
    # Tail LoRA receives the same fixed optimizer group rate; pilot calibration
    # does not establish a full-scale prescription.
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda s:min((s+1)/200,1.)*(.1+.9*.5*(1+math.cos(math.pi*min(s,max_updates)/max_updates))))
    identity=write_identity(output,{"schema":"decodable_representation_v1","source_sha":source_sha(),"config":config,
        "arm":arm,"seed":seed,"baseline_identity":manifest["identity"],"max_updates":max_updates,
        "trainable_inventory":[{"name":n,"shape":list(p.shape),"elements":p.numel()} for n,p in parameters],
        "lora_modules":lora_modules,"query_execution":"prefix_cached_v1","language_checkpointing":False,
        "global_scene_batch":1,"precision":"FP32 TF32 disabled","scope":config["scope"],
        "loss_weights":config["loss_weights"],"current_DINO_recipe":config["dino_identity"],
        "frozen_Qwen_control_queries_trainable":arm=="Rfrozen"})
    generator=torch.Generator().manual_seed(seed+101);order=torch.randperm(len(selected),generator=generator).tolist()
    cursor=0;step=0
    if resume:
        state=load_checkpoint(output/"latest.pt",model=delta,optimizer=optimizer,scheduler=scheduler,identity=identity)
        step=state["progress"]["update"];cursor=state["sampler"]["cursor"];order=state["sampler"]["order"];generator.set_state(state["sampler"]["rng"])
    atomic_json(output/"state.json",{"status":"RUNNING","update":step})
    log=open(output/"train.jsonl","a",buffering=1)
    windows=time.monotonic();gradient_evidence={};first_saved=False
    try:
        while step<max_updates:
            if cursor==len(order):order=torch.randperm(len(selected),generator=generator).tolist();cursor=0
            row=selected[order[cursor]];cursor+=1
            example,original_targets=data[lookup[row["token"]]]
            targets=target_rows[row["token"]]
            # Dropout is fixed eval in this bounded probe. LoRA/new queries remain
            # trainable; the baseline visual/DiT/ego/DINO parameters never update.
            model.eval()
            for name,p in parameters:
                if "lora_" in name:p.requires_grad_(step>=config.get("warmup_updates",200) or arm=="R0")
            optimizer.zero_grad(set_to_none=True)
            rep=model(observation(example))
            dino,_=masked_regression(model.dino_prediction(rep),original_targets["current_dino"][None].to(device),
                                     original_targets["current_dino_valid"][None,:,None].to(device))
            losses={"DINO":dino};weights=config["loss_weights"]
            if decoder is not None:
                context=targets["context"].to(device);xyz=targets["xyz"].to(device);teacher=targets["logits"].to(device)
                logits=decoder(rep.predicted_spatial_context,xyz)
                losses["spatial_feature"]=((rep.predicted_spatial_context-context)/std).square().mean()
                losses["spatial_decode"]=F.binary_cross_entropy_with_logits(logits,teacher.sigmoid())
                if arm=="Rdecode":losses.pop("spatial_feature")
            if temporal_decoder is not None:
                hidden=targets["future_hidden"].to(device);target_logits=targets["future_logits"].to(device)
                losses["temporal_feature"]=(F.layer_norm(rep.predicted_future_latent,(hidden.shape[-1],))-F.layer_norm(hidden,(hidden.shape[-1],))).square().mean()
                prediction=temporal_decoder(rep.predicted_future_latent)
                losses["temporal_decode"]=-(target_logits.softmax(1)*prediction.log_softmax(1)).sum(1).mean()
            loss=sum(weights[k]*v for k,v in losses.items())
            if not loss.requires_grad:raise ValueError("No active supervised gradient; unsupported capacity-only P2 schedule")
            if not torch.isfinite(loss):raise FloatingPointError("Nonfinite representation loss")
            if step in (0,config.get("warmup_updates",200),config.get("warmup_updates",200)+2):
                chosen=[p for n,p in parameters if p.requires_grad and ("lora_B" in n or ".queries" in n)]
                terms={}
                for key,value in losses.items():
                    grads=torch.autograd.grad(weights[key]*value,chosen,retain_graph=True,allow_unused=True) if value.requires_grad else [None]*len(chosen)
                    terms[key]=sum(float(g.detach().float().square().sum()) for g in grads if g is not None)**.5
                gradient_evidence[str(step+1)]=terms;atomic_json(output/"gradient_calibration.json",gradient_evidence)
            loss.backward()
            for n,p in parameters:
                if p.grad is not None and (step==0 or step==config.get("warmup_updates",200)+2):
                    gradient_evidence.setdefault("parameter_groups",{})[n]=float(p.grad.norm())
            norm=torch.nn.utils.clip_grad_norm_([p for _,p in parameters],1.,error_if_nonfinite=True)
            optimizer.step();scheduler.step();step+=1
            if step%20==0 or step==1:
                log.write(json.dumps({"update":step,"scene_exposures":step,"losses":{k:v.item() for k,v in losses.items()},
                    "gradient_norm":float(norm),"lr":optimizer.param_groups[0]["lr"],"window_seconds":time.monotonic()-windows})+"\n")
                windows=time.monotonic()
            stop=(output/"STOP_REQUESTED").exists() or (stop_after is not None and step>=stop_after)
            if step%200==0 or step==max_updates or stop:
                save_checkpoint(output/"latest.pt",model=delta,optimizer=optimizer,scheduler=scheduler,identity=identity,
                    progress={"update":step,"scene_exposures":step,"phase":"representation_tiny"},
                    sampler={"order":order,"cursor":cursor,"rng":generator.get_state()},
                    extra={"loss_calibration":gradient_evidence,"baseline_identity":manifest["identity"]})
                atomic_json(output/"state.json",{"status":"RUNNING","update":step})
            if stop:break
        if [parameter_digest(d) for d in (decoder,temporal_decoder) if d is not None]!=fixed_digests:raise AssertionError("Fixed decoder updated")
        atomic_json(output/"gradient_calibration.json",gradient_evidence)
        result={"status":"INCONCLUSIVE","training_complete":step==max_updates,"update":step,"scene_exposures":step,
           "seconds":time.monotonic()-start,"gpu_hours":(time.monotonic()-start)/3600,
           "peak_allocated_bytes":torch.cuda.max_memory_allocated() if device.startswith("cuda") else 0,
           "frozen_decoders_unchanged":True,"scientific_capability":"Requires full decoding and independent labels; loss decrease is insufficient"}
        atomic_json(output/"RESULT.json",result);return result
    except Exception as exc:
        import traceback
        atomic_json(output/"state.json",{"status":"FAIL","update":step,"error":repr(exc),"traceback":traceback.format_exc()});raise
    finally:log.flush();log.close()


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--output",required=True)
    p.add_argument("--arm",default="RS");p.add_argument("--seed",type=int,default=42);p.add_argument("--device",default="cuda")
    p.add_argument("--max-updates",type=int,default=1000);p.add_argument("--resume",action="store_true");p.add_argument("--stop-after",type=int)
    a=p.parse_args();run(json.loads(Path(a.config).read_text()),a.output,arm=a.arm,seed=a.seed,device=a.device,
                         max_updates=a.max_updates,resume=a.resume,stop_after=a.stop_after)
