"""Run real current-only stage1 native inference and re-decode captured hidden."""
import argparse
import ast
import json
import pickle
from pathlib import Path
import time
import torch
import numpy as np
from starVLA.teachers.decodable_st.source_registry import atomic_json,sha256


def current_ego_inputs(current_root,raw_log_root,row):
    record=json.loads((Path(current_root)/"current"/(row["token"]+".json")).read_text())
    with (Path(raw_log_root)/(row["log"]+".pkl")).open("rb") as f:frames=pickle.load(f)
    current=next(f for f in frames if f["token"]==row["token"])
    if int(current["timestamp"])!=int(record["timestamp"]):raise ValueError("Current ego timestamp mismatch")
    history=np.asarray(record["global_pose_history"],dtype=np.float64)
    c,s=np.cos(history[-1,2]),np.sin(history[-1,2])
    xy=(history[:,:2]-history[-1,:2])@np.array([[c,-s],[s,c]])
    yaw=(history[:,2]-history[-1,2]+np.pi)%(2*np.pi)-np.pi
    local=np.column_stack((xy,yaw)).astype(np.float32)
    state=np.concatenate((np.asarray(current["driving_command"]),np.asarray(current["ego_dynamic_state"])[2:],
                          local[-1],np.asarray(current["ego_dynamic_state"])[:2])).astype(np.float32)
    if state.shape!=(11,) or local.shape!=(4,3):raise ValueError("Native current-state contract")
    # Same two-decimal history quantization as the native agent interface.
    return torch.from_numpy(state)[None],torch.from_numpy(np.round(local,2))[None]


def source_image_functions(source_root):
    # Extract unchanged, dependency-light upstream preprocessing functions.
    # Never import the agent module: it imports lmdeploy and the full simulator.
    source=Path(source_root)/"navsim/agents/intervl_agent_wm.py"
    tree=ast.parse(source.read_text());names={"build_transform","find_closest_aspect_ratio","dynamic_preprocess","load_image"}
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    if len(nodes)!=4:raise ValueError("Upstream preprocessing definitions changed")
    from PIL import Image
    import torchvision.transforms as T
    from torchvision.transforms import InterpolationMode
    scope={"torch":torch,"np":np,"Image":Image,"T":T,"InterpolationMode":InterpolationMode,
           "IMAGENET_MEAN":(.485,.456,.406),"IMAGENET_STD":(.229,.224,.225)}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),"exec"),scope)
    return scope["load_image"],sha256(source)


def run(config,output,device="cuda",limit=2):
    from starVLA.teachers.decodable_st.sgdrive_adapter import load_native_stage1,current_prefix
    from starVLA.model.modules.decodable_st.frozen_decoders import sgdrive_logits_layout
    from starVLA.dataloader.foresight_dataset import ForesightCurrentDataset
    from safetensors.torch import save_file
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=torch.backends.cudnn.allow_tf32=False
    if device.startswith("cuda"):torch.cuda.set_per_process_memory_fraction(.55)
    output=Path(output);output.mkdir(parents=True,exist_ok=True);start=time.monotonic()
    model,tokenizer,receipt=load_native_stage1(config["source_root"],Path(config["weights"]).parent,device)
    preprocess,preprocess_hash=source_image_functions(config["source_root"])
    data=ForesightCurrentDataset(config["current_root"]);lookup={r["token"]:i for i,r in enumerate(data.index)}
    selected=json.loads(Path(config["selection"]).read_text())[:limit];rows=[]
    for row in selected:
        obs=data[lookup[row["token"]]]
        pixels=[preprocess(np.asarray(im),input_size=448,max_num=12) for im in obs["image"]]
        inputs,text=current_prefix(model,tokenizer,navigation=obs["lang"],num_patches=[len(p) for p in pixels])
        ego_status,hist_traj=current_ego_inputs(config["current_root"],config["raw_log_root"],row)
        ids=inputs["input_ids"].to(device)
        if (ids==model.world_token_id).sum()!=model.NUM_WORLD_TOKEN or (ids==model.dream_token_id).sum()!=675:
            raise AssertionError("Native prompt token counts mismatch")
        captured={}
        def hook(key):
            def capture(module,args,result):captured[key]=(args[0].detach().clone(),result.detach().clone())
            return capture
        handles=[model._occ_head.register_forward_hook(hook("current")),model._occ_head_dream.register_forward_hook(hook("future"))]
        try:
            # Current CAN state and four permitted historical ego poses only.
            # The native model requires these; None would not be a valid probe.
            native=model.inference(pixel_values=torch.cat(pixels).to(device),input_ids=ids,
                    attention_mask=inputs["attention_mask"].to(device),ego_status=ego_status.to(device),
                    hist_traj=hist_traj.to(device),use_cache=False)
        finally:
            for h in handles:h.remove()
        checks={};tensors={}
        for key,head in (("current",model._occ_head),("future",model._occ_head_dream)):
            hidden,raw=captured[key]
            with torch.no_grad():split=head(hidden,[[100,100],[50,50]])
            torch.testing.assert_close(raw,split,rtol=1e-4,atol=1e-5)
            labels=sgdrive_logits_layout(split).argmax(1)
            torch.testing.assert_close(labels,native["occ_out" if key=="current" else "occ_out_dream"],rtol=0,atol=0)
            checks[key]={"max_abs":(raw-split).abs().max().item(),"hidden_shape":list(hidden.shape),"logits_shape":list(sgdrive_logits_layout(split).shape)}
            tensors[key+"_hidden"]=hidden.cpu();tensors[key+"_logits"]=sgdrive_logits_layout(split).cpu()
        save_file(tensors,str(output/(row["token"]+".safetensors")))
        rows.append({**row,"checks":checks,"input_tokens":ids.numel(),"image_tiles":[len(p) for p in pixels],
                     "qa_answers_in_prefix":False,"current_state_source":"current log CAN and permitted four historical ego poses",
                     "teacher_extra_current_velocity_acceleration":True,"prefix_sha256":__import__('hashlib').sha256(text.encode()).hexdigest()})
        atomic_json(output/"progress.json",{"completed":len(rows),"requested":len(selected)})
    result={"status":"PASS","scope":"native stage1 current-only inference versus extracted hidden/native head",
            "load_receipt":receipt,"preprocess_sha256":preprocess_hash,"rows":rows,"seconds":time.monotonic()-start,
            "native_horizon_seconds":None,"teacher_temporal_driving_verified":"BLOCKED",
            "reason":"native training future_idx is not a timestamp; raw LiDAR is not ego(t0)-compensated and GT trajectory gates label corridor"}
    atomic_json(output/"ROUNDTRIP.json",result);return result


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--output",required=True)
    p.add_argument("--device",default="cuda");p.add_argument("--limit",type=int,default=2);a=p.parse_args()
    run(json.loads(Path(a.config).read_text()),a.output,a.device,a.limit)
