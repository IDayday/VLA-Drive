"""Two real CUDA ranks: S0 images, original DINO, native ViGT loss and AdamW.

This is a bounded one-to-four-update execution/resume test, not a scientific experiment.
The separate loss-reduction unit test checks unequal/empty rank equivalence.
"""
import argparse
import json
import os
from pathlib import Path
import time
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from safetensors.torch import load_file
from .runtime import numerical_policy,parameter_digest,source_sha,write_identity,save_checkpoint,load_checkpoint
from .train_representation import load_model,load_data,observation
from starVLA.model.modules.decodable_st.training_state import TrainableDelta
from starVLA.teachers.decodable_st.vigt_adapter import load_vigt_decoder
from starVLA.teachers.decodable_st.source_registry import atomic_json


def run(config,output,updates=1,stop_after=None,resume=False):
    if not 1<=updates<=4 or (stop_after is not None and not 1<=stop_after<=updates):
        raise ValueError("Finite distributed execution probe: at most four updates")
    rank=int(os.environ["RANK"]);local=int(os.environ["LOCAL_RANK"]);world=int(os.environ["WORLD_SIZE"])
    if world!=2:raise ValueError("Registered operator probe uses exactly two ranks")
    torch.cuda.set_device(local);device=f"cuda:{local}";numerical_policy(42,device,.65)
    dist.init_process_group("nccl");start=time.monotonic()
    try:
        model,baseline,_=load_model(config,device,"RS",42)
        decoder,_=load_vigt_decoder(config["vigt_weights"],json.loads(Path(config["vigt_config"]).read_text()));decoder.to(device)
        fixed=parameter_digest(decoder);data,selected,lookup=load_data(config)
        wrapper=DistributedDataParallel(model,device_ids=[local],broadcast_buffers=False,find_unused_parameters=True)
        parameters=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
        optimizer=torch.optim.AdamW([p for _,p in parameters],lr=1e-5,weight_decay=.001)
        scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda s:1-.1*min(s,updates)/updates)
        output=Path(output);states=output.with_suffix("")
        ident={"source_sha":source_sha(),"baseline_identity":baseline["identity"],"config":config,
               "scope":"real_two_rank_execution_only","max_updates":updates,"world":world,"seed":42,
               "LoRA":"active from first step, unlike scientific warmup","feature_scale":"unit, execution test only",
               "inventory":[n for n,_ in parameters],"lr":1e-5,"scheduler":"linear to .9 initial LR"}
        signature=[write_identity(states,ident) if rank==0 else None];dist.broadcast_object_list(signature,src=0)
        delta=TrainableDelta(model,[n for n,_ in parameters]);generator=torch.Generator().manual_seed(4243)
        order=torch.randperm(len(selected),generator=generator).tolist();step=0;start_step=0
        if resume:
            previous=load_checkpoint(states/f"rank{rank}.pt",model=delta,optimizer=optimizer,scheduler=scheduler,identity=signature[0])
            step=previous["progress"]["update"];start_step=step
            order=previous["sampler"]["order"];generator.set_state(previous["sampler"]["rng"])
            if previous["sampler"]["offset"]!=step*world:raise ValueError("Resume sampler boundary changed")
        steps=[None]*world;dist.all_gather_object(steps,step)
        if len(set(steps))!=1:raise ValueError("Ranks resumed different updates")
        wrapper.eval();traces=[]
        from starVLA.model.modules.foresight.losses import masked_regression
        while step<updates and (stop_after is None or step<stop_after):
            row=selected[order[step*world+rank]];example,targets=data[lookup[row["token"]]]
            teacher=load_file(str(Path(config["native_context_root"])/(row["token"]+".safetensors")))
            optimizer.zero_grad(set_to_none=True);rep=wrapper(observation(example))
            dino,_=masked_regression(model.dino_prediction(rep),targets["current_dino"][None].to(device),targets["current_dino_valid"][None,:,None].to(device))
            feature=(rep.predicted_spatial_context-teacher["context"].to(device)).square().mean()
            decode=torch.nn.functional.binary_cross_entropy_with_logits(decoder(rep.predicted_spatial_context,teacher["xyz"].to(device)),teacher["logits"].to(device).sigmoid())
            loss=dino+.1*feature+.1*decode;loss.backward()
            norm=torch.nn.utils.clip_grad_norm_([p for _,p in parameters],1.,error_if_nonfinite=True);optimizer.step();scheduler.step();step+=1
            traces.append({"update":step,"token":row["token"],"loss":loss.item(),"gradient_norm":norm.item()})
        save_checkpoint(states/f"rank{rank}.pt",model=delta,optimizer=optimizer,scheduler=scheduler,identity=signature[0],
            progress={"update":step,"scene_exposures":step*world,"phase":"distributed_execution_probe"},
            sampler={"order":order,"offset":step*world,"rank":rank,"rng":generator.get_state()})
        dist.barrier()
        if parameter_digest(decoder)!=fixed:raise AssertionError("Frozen decoder changed")
        digest=parameter_digest(delta)
        receipt={"rank":rank,"trainable_parameter_sha256":digest,"trace":traces,
                 "peak_allocated_bytes":torch.cuda.max_memory_allocated(),"decoder_unchanged":True}
        rows=[None]*world;dist.all_gather_object(rows,receipt)
        if rank==0:
            atomic_json(output,{"status":"PASS" if len({r["trainable_parameter_sha256"] for r in rows})==1 else "FAIL",
                "rows":rows,"source_sha":source_sha(),"baseline_identity":baseline["identity"],"seconds":time.monotonic()-start,
                "gpu_hours":2*(time.monotonic()-start)/3600,"precision":"FP32 TF32 disabled",
                "scope":"genuine two-rank representation/optimizer execution; unit-scale feature loss, not a scientific P1 result",
                "update":step,"segment_start_update":start_step,"training_complete":step==updates,"identity":signature[0],
                "checkpoint_comparison":"separate continuous/resumed comparison required; successful load is not numerical equality"})
    finally:dist.destroy_process_group()


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--output",required=True)
    p.add_argument("--updates",type=int,default=1);p.add_argument("--stop-after",type=int);p.add_argument("--resume",action="store_true")
    a=p.parse_args();run(json.loads(Path(a.config).read_text()),a.output,a.updates,a.stop_after,a.resume)
