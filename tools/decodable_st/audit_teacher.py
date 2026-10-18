"""Real native decoder loading/roundtrip/gradient probes, with separate evidence levels."""
import argparse
import json
import os
from pathlib import Path
import time
import traceback
import torch
from starVLA.teachers.decodable_st.source_registry import atomic_json,sha256


def finite_input_gradient(decoder,features,*args):
    decoder.eval().requires_grad_(False)
    before={k:v.detach().clone() for k,v in decoder.state_dict().items()}
    value=features.detach().float().requires_grad_(True)
    logits=decoder(value,*args)
    logits.float().square().mean().backward()
    grad=value.grad
    if grad is None or not torch.isfinite(grad).all() or not grad.abs().sum()>0:
        raise AssertionError("No finite nonzero native decoder input gradient")
    if any(p.grad is not None for p in decoder.parameters()):raise AssertionError("Teacher parameter received gradient")
    for k,v in decoder.state_dict().items():torch.testing.assert_close(v,before[k],rtol=0,atol=0)
    return {"gradient_norm":grad.norm().item(),"gradient_finite":True,"parameters_and_buffers_unchanged":True}


def native_decoder_probe(config,output,device):
    from starVLA.teachers.decodable_st.vigt_adapter import load_vigt_decoder
    from starVLA.teachers.decodable_st.sgdrive_adapter import load_sgdrive_decoder
    result={"status":"RUNNING","scope":"native_head_operator_test; synthetic hidden is not teacher capability evidence"}
    start=time.monotonic();atomic_json(output,result)
    if device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32=torch.backends.cudnn.allow_tf32=False
        torch.cuda.set_per_process_memory_fraction(.45)
    try:
        if config["kind"]=="vigt":
            c=json.loads(Path(config["config"]).read_text())
            decoder,receipt=load_vigt_decoder(config["weights"],c);decoder.to(device)
            feature=torch.randn(1,decoder.contract.channels,decoder.contract.height,decoder.contract.width,device=device)
            xyz=torch.rand(1,1024,3,device=device)
            lo=xyz.new_tensor(decoder.contract.roi_min);hi=xyz.new_tensor(decoder.contract.roi_max)
            xyz=lo+xyz*(hi-lo)
            gradient=finite_input_gradient(decoder,feature,xyz)
            result.update(load_receipt=receipt,gradient=gradient,contract=vars(decoder.contract))
        else:
            c=json.loads(Path(config["config"]).read_text());records=[]
            for future in (False,True):
                decoder,receipt=load_sgdrive_decoder(config["weights"],c,future=future);decoder.to(device)
                feature=torch.randn(1,decoder.contract.num_queries,decoder.contract.hidden_size,device=device)
                with torch.no_grad():
                    a=decoder(feature);b=decoder(feature)
                torch.testing.assert_close(a,b,rtol=0,atol=0)
                records.append({"future":future,"load_receipt":receipt,"gradient":finite_input_gradient(decoder,feature),
                                "input_shape":list(feature.shape),"output_shape":list(a.shape),"fresh_shapes_repeat_exact":True})
                del decoder,feature,a,b
            result["heads"]=records
        result["status"]="PASS"
    except Exception as exc:
        result.update(status="FAIL",error=repr(exc),traceback=traceback.format_exc());raise
    finally:
        result["seconds"]=time.monotonic()-start
        result["gpu_hours"]=result["seconds"]/3600 if device.startswith("cuda") else 0
        result["teacher_capability_status"]="NOT_RUN"
        if device.startswith("cuda"):
            result["peak_allocated_bytes"]=torch.cuda.max_memory_allocated();result["peak_reserved_bytes"]=torch.cuda.max_memory_reserved()
        atomic_json(output,result)
    return result


def vigt_real_roundtrip(config,output,device):
    from starVLA.teachers.decodable_st.vigt_adapter import ViGTTeacherAdapter
    from starVLA.dataloader.foresight_dataset import ForesightCurrentDataset
    from torchvision.transforms import v2
    from safetensors.torch import save_file
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=torch.backends.cudnn.allow_tf32=False
    if device.startswith("cuda"):torch.cuda.set_per_process_memory_fraction(.45)
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    start=time.monotonic();cfg=json.loads(Path(config["config"]).read_text())
    adapter=ViGTTeacherAdapter(config["source_root"],config["weights"],cfg,device)
    data=ForesightCurrentDataset(config["current_root"])
    selection=json.loads(Path(config["selection"]).read_text())
    lookup={x["token"]:i for i,x in enumerate(data.index)}
    transform=v2.Compose([v2.ToImage(),v2.Resize(size=192),v2.ToDtype(torch.float32,scale=True),
                          v2.Normalize(mean=[.485,.456,.406],std=[.229,.224,.225]),v2.ToPureTensor()])
    generator=torch.Generator().manual_seed(42)
    xyz=torch.rand(1,4096,3,generator=generator)*torch.tensor([39.,40.,5.4])+torch.tensor([.5,-20.,-.5])
    xyz=xyz.to(device);records=[]
    for row in selection:
        obs=data[lookup[row["token"]]]
        sample={"images":[transform(im).to(device) for im in obs["image"]],"front_camera_idx":0}
        t=time.monotonic()
        context,logits,check=adapter.roundtrip(sample,xyz)
        if not torch.isfinite(context).all() or not torch.isfinite(logits).all():raise FloatingPointError("Native teacher nonfinite")
        # Preserve full context only for the bounded audit/capacity subset.
        path=output/(row["token"]+".safetensors")
        save_file({"context":context.cpu(),"xyz":xyz.cpu(),"logits":logits.cpu()},str(path)+".tmp")
        os.replace(str(path)+".tmp",path)
        records.append({"token":row["token"],"log":row["log"],**check,"seconds":time.monotonic()-t,
                        "bytes":path.stat().st_size,"sha256":sha256(path)})
        atomic_json(output/"progress.json",{"completed":len(records),"requested":len(selection)})
    result={"status":"PASS","scope":"native_forward_vs_split_on_real_current_three_views",
         "load_receipt":adapter.load_receipt,"records":records,"seconds":time.monotonic()-start,
         "context_bytes_per_scene":context.numel()*context.element_size(),
         "teacher_capability_status":"INCONCLUSIVE","reason":"native equivalence does not establish target-domain occupancy accuracy"}
    result["gpu_hours"]=result["seconds"]/3600 if device.startswith("cuda") else 0
    result["projected_full_train_context_bytes"]=result["context_bytes_per_scene"]*len(data)
    atomic_json(output/"ROUNDTRIP.json",result)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--output",required=True)
    p.add_argument("--mode",choices=("decoder","vigt-real"),default="decoder");p.add_argument("--device",default="cuda")
    a=p.parse_args();cfg=json.loads(Path(a.config).read_text())
    (native_decoder_probe if a.mode=="decoder" else vigt_real_roundtrip)(cfg,a.output,a.device)

if __name__=="__main__":main()
