"""Two real CUDA ranks: S0 images, original DINO, native ViGT loss and AdamW.

This is a single-update execution test, not a distributed scientific experiment.
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
from .runtime import numerical_policy,parameter_digest,source_sha
from .train_representation import load_model,load_data,observation
from starVLA.model.modules.decodable_st.training_state import TrainableDelta
from starVLA.teachers.decodable_st.vigt_adapter import load_vigt_decoder
from starVLA.teachers.decodable_st.source_registry import atomic_json


def run(config,output):
    rank=int(os.environ["RANK"]);local=int(os.environ["LOCAL_RANK"]);world=int(os.environ["WORLD_SIZE"])
    if world!=2:raise ValueError("Registered operator probe uses exactly two ranks")
    torch.cuda.set_device(local);device=f"cuda:{local}";numerical_policy(42,device,.65)
    dist.init_process_group("nccl");start=time.monotonic()
    try:
        model,baseline,_=load_model(config,device,"RS",42)
        decoder,_=load_vigt_decoder(config["vigt_weights"],json.loads(Path(config["vigt_config"]).read_text()));decoder.to(device)
        fixed=parameter_digest(decoder);data,selected,lookup=load_data(config);row=selected[rank]
        example,targets=data[lookup[row["token"]]]
        teacher=load_file(str(Path(config["native_context_root"])/(row["token"]+".safetensors")))
        wrapper=DistributedDataParallel(model,device_ids=[local],broadcast_buffers=False,find_unused_parameters=True)
        parameters=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
        optimizer=torch.optim.AdamW([p for _,p in parameters],lr=1e-5,weight_decay=.001)
        wrapper.eval();optimizer.zero_grad(set_to_none=True);rep=wrapper(observation(example))
        from starVLA.model.modules.foresight.losses import masked_regression
        dino,_=masked_regression(model.dino_prediction(rep),targets["current_dino"][None].to(device),targets["current_dino_valid"][None,:,None].to(device))
        feature=(rep.predicted_spatial_context-teacher["context"].to(device)).square().mean()
        decode=torch.nn.functional.binary_cross_entropy_with_logits(decoder(rep.predicted_spatial_context,teacher["xyz"].to(device)),teacher["logits"].to(device).sigmoid())
        loss=dino+.1*feature+.1*decode;loss.backward()
        norm=torch.nn.utils.clip_grad_norm_([p for _,p in parameters],1.,error_if_nonfinite=True);optimizer.step()
        if parameter_digest(decoder)!=fixed:raise AssertionError("Frozen decoder changed")
        delta=TrainableDelta(model,[n for n,_ in parameters]);digest=parameter_digest(delta)
        receipt={"rank":rank,"trainable_parameter_sha256":digest,"loss":loss.item(),"gradient_norm":norm.item(),
                 "peak_allocated_bytes":torch.cuda.max_memory_allocated(),"decoder_unchanged":True}
        rows=[None]*world;dist.all_gather_object(rows,receipt)
        if rank==0:
            atomic_json(output,{"status":"PASS" if len({r["trainable_parameter_sha256"] for r in rows})==1 else "FAIL",
                "rows":rows,"source_sha":source_sha(),"baseline_identity":baseline["identity"],"seconds":time.monotonic()-start,
                "gpu_hours":2*(time.monotonic()-start)/3600,"precision":"FP32 TF32 disabled",
                "scope":"one genuine two-rank representation/optimizer update; unit-scale feature loss for execution only, not a formal P1 result",
                "full_multirank_checkpoint_resume":"NOT_RUN"})
    finally:dist.destroy_process_group()


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--output",required=True);a=p.parse_args();run(json.loads(Path(a.config).read_text()),a.output)
