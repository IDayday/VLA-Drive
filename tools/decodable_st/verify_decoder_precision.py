"""Native heads with BF16 student storage and autocast, retaining failures."""
import argparse
import json
from pathlib import Path
import torch
from starVLA.teachers.decodable_st.source_registry import atomic_json
from starVLA.teachers.decodable_st.vigt_adapter import load_vigt_decoder
from starVLA.teachers.decodable_st.sgdrive_adapter import load_sgdrive_decoder
from .runtime import parameter_digest


def run(config,output):
    torch.set_num_threads(2);torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32=torch.backends.cudnn.allow_tf32=False
    rows=[]
    for kind in ("vigt","sgdrive_current","sgdrive_future"):
        if kind=="vigt":
            decoder,_=load_vigt_decoder(config["vigt_weights"],json.loads(Path(config["vigt_config"]).read_text()))
            shape=(1,256,256,256);args=(torch.rand(1,256,3,device="cuda")*torch.tensor([70.,70.,5.],device="cuda")+torch.tensor([-35.,-35.,-.5],device="cuda"),)
        else:
            decoder,_=load_sgdrive_decoder(config["sgdrive_weights"],json.loads(Path(config["sgdrive_config"]).read_text()),future=kind.endswith("future"))
            shape=(1,625,1536);args=()
        decoder.to("cuda");digest=parameter_digest(decoder)
        for policy in ("bf16_storage_fp32_native_decode","bf16_autocast_native_decode"):
            value=torch.randn(shape,device="cuda",dtype=torch.bfloat16,requires_grad=True)
            try:
                with torch.autocast("cuda",dtype=torch.bfloat16,enabled=policy.endswith("autocast_native_decode")):
                    logits=decoder(value.float(),*args)
                logits.float().square().mean().backward()
                if value.grad is None or not torch.isfinite(logits).all() or not torch.isfinite(value.grad).all():raise FloatingPointError("Nonfinite/missing input gradient")
                if parameter_digest(decoder)!=digest or any(p.grad is not None for p in decoder.parameters()):raise AssertionError("Fixed decoder changed")
                rows.append({"kind":kind,"policy":policy,"status":"PASS","input_gradient_norm":value.grad.float().norm().item(),"output_dtype":str(logits.dtype)})
            except Exception as exc:rows.append({"kind":kind,"policy":policy,"status":"FAIL","error":repr(exc)})
            atomic_json(output,{"status":"RUNNING","rows":rows})
        del decoder
    atomic_json(output,{"status":"PASS" if all(r["status"]=="PASS" for r in rows) else "INCONCLUSIVE","rows":rows,
                       "scope":"operator finite checks only; formal training/inference remains FP32"})


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--output",required=True);a=p.parse_args()
    run(json.loads(Path(a.config).read_text()),a.output)
