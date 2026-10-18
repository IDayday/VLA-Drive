"""Real Qwen3 multimodal padding/grid/batch parity; no teacher dependency."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import time
import torch
from starVLA.teachers.decodable_st.source_registry import atomic_json


def run(manifest_path,output,device="cuda"):
    from tools.foresight.checkpoints import load_student
    from starVLA.dataloader.decodable_st_dataset import DecodableDataset,collate_observations
    from starVLA.model.framework.ddp_decodable_st import DDPDecodableST
    from starVLA.model.modules.decodable_st.contracts import SpatialContract,TemporalContract
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(.75)
    manifest=json.loads(Path(manifest_path).read_text());start=time.monotonic()
    identity=json.loads((Path(manifest["run"])/"identity.json").read_text())
    base=load_student(manifest["run"],manifest["tag"],identity,device=device,precision="fp32",strip=False).eval().requires_grad_(False)
    base.qwen_vl_interface.model.gradient_checkpointing_disable()
    cpu_rng=torch.get_rng_state();cuda_rng=torch.cuda.get_rng_state_all()
    model=DDPDecodableST(base,spatial_contract=SpatialContract(256,256,256,(-40,-40,-1),(40,40,5.4)),
        temporal_contract=TemporalContract(1536,625,(100,100),16,None,"native_lidar_frame_unverified","local_point_hit","T-native-v1")).to(device).eval()
    rng_exact=torch.equal(cpu_rng,torch.get_rng_state()) and all(torch.equal(a,b) for a,b in zip(cuda_rng,torch.cuda.get_rng_state_all()))
    dataset=DecodableDataset(manifest["data"]["train"]["root"],indices=[0,1,2,3]);results=[]
    tokenizer=base.qwen_vl_interface.processor.tokenizer;old_side=tokenizer.padding_side
    try:
        for side,batch,mixed in [("left",1,False),("left",2,False),("left",4,False),("right",2,False),("left",2,True),("right",2,True)]:
            tokenizer.padding_side=side
            obs,_,_=collate_observations([dataset[i] for i in range(batch)])
            # Deliberately unequal text lengths and actual image grids exercise
            # native position/DeepStack handling. Resized inputs are engineering
            # tests only; they never enter the frozen scientific population.
            obs=replace(obs,navigation=tuple(s+(" Keep following the road."*i) for i,s in enumerate(obs.navigation)))
            if mixed:
                obs=replace(obs,images=tuple(tuple(im.resize((672,378)) if i else im for im in ims) for i,ims in enumerate(obs.images)))
            examples=[{"image":list(ims),"lang":nav,"state":state.numpy()} for ims,nav,state in zip(obs.images,obs.navigation,obs.ego_state)]
            with torch.no_grad():
                reference=base.encode_current(examples)["action_queries"]
                representation=model(obs)
            actual=representation.original_action_hidden
            error=(reference-actual).abs().max().item()
            row={"padding":side,"batch":batch,"mixed_image_grid":mixed,"prefix_max_abs":error,"prefix_exact":torch.equal(actual,reference),
                 "spatial_shape":list(representation.predicted_spatial_context.shape),"temporal_shape":list(representation.predicted_future_latent.shape)}
            # Same-sample comparison detects cross-sample leakage, with disclosed
            # native kernel shape error instead of declaring bitwise batch parity.
            errors=[]
            for i in range(batch):
                single=replace(obs,images=(obs.images[i],),navigation=(obs.navigation[i],),ego_state=obs.ego_state[i:i+1])
                with torch.no_grad():one=model(single)
                errors.append((one.original_action_hidden-actual[i:i+1]).abs().max().item())
            row["batch_vs_single_action_max_abs"]=errors
            results.append(row);atomic_json(output,{"status":"RUNNING","cases":results})
            del representation,actual,reference,one
    finally:tokenizer.padding_side=old_side
    result={"status":"PASS" if rng_exact and all(r["prefix_exact"] for r in results) else "FAIL",
        "cases":results,"new_module_initialization_preserves_CPU_CUDA_RNG":rng_exact,
        "seconds":time.monotonic()-start,"gpu_hours":(time.monotonic()-start)/3600,
        "peak_allocated_bytes":torch.cuda.max_memory_allocated(),
        "scope":"native Qwen3 DeepStack/mRoPE forward-local KV; inference cache across separate requests is not implemented"}
    atomic_json(output,result);return result


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--manifest",required=True);p.add_argument("--output",required=True);p.add_argument("--device",default="cuda")
    a=p.parse_args();run(a.manifest,a.output,a.device)
