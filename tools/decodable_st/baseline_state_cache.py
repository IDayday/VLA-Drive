"""One identity-bound FP32 master replay asset, never observation features.

Repeated tiny jobs otherwise reconstruct the same eight DeepSpeed optimizer
shards. The official reconstruction is performed once and every subsequent
model load remains strict. The historical checkpoint is never modified.
"""
import argparse
import json
from pathlib import Path
import time
import torch
from safetensors.torch import load_file,save_file
from starVLA.teachers.decodable_st.source_registry import atomic_json,sha256
from .runtime import source_sha


def build(baseline_manifest,output):
    torch.set_num_threads(2);start=time.monotonic();output=Path(output)
    if output.exists():raise FileExistsError("FP32 master replay requires a fresh atomic asset")
    output.mkdir(parents=True)
    manifest=json.loads(Path(baseline_manifest).read_text())
    if manifest["status"]!="IDENTITY_VERIFIED":raise ValueError("Unverified baseline")
    from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
    state=get_fp32_state_dict_from_zero_checkpoint(str(Path(manifest["run"])/"checkpoints"),tag=manifest["tag"])
    # Separate aliased embedding/lm_head tensors while preserving their values.
    values={k:v.detach().cpu().contiguous().clone() for k,v in state.items()};del state
    save_file(values,str(output/"master.safetensors.tmp"));(output/"master.safetensors.tmp").replace(output/"master.safetensors")
    record={"schema":"verified_S0_FP32_master_replay_v1","baseline_identity":manifest["identity"],
        "original_manifest_sha256":sha256(baseline_manifest),"source_sha":source_sha(),"source_tag":manifest["tag"],
        "weights_sha256":sha256(output/"master.safetensors"),"bytes":(output/"master.safetensors").stat().st_size,
        "inventory":{k:{"shape":list(v.shape),"dtype":str(v.dtype)} for k,v in values.items()},
        "official_reconstruction":"DeepSpeed0.16.9 get_fp32_state_dict_from_zero_checkpoint; no BF16 save-value shortcut",
        "seconds":time.monotonic()-start,"GPU_hours":0,"baseline_output_parity":"NOT_RUN"}
    atomic_json(output/"manifest.json",record);return record


def load(config,baseline,identity,device):
    cache=config["baseline_state_cache"];root=Path(cache["root"])
    if sha256(root/"manifest.json")!=cache["manifest_sha256"]:raise ValueError("FP32 replay identity changed")
    record=json.loads((root/"manifest.json").read_text())
    if record["baseline_identity"]!=baseline["identity"]:raise ValueError("Wrong S0 replay ancestry")
    if sha256(root/"master.safetensors")!=record["weights_sha256"]:raise ValueError("Corrupt FP32 replay payload")
    from omegaconf import OmegaConf
    from starVLA.model.framework import build_framework
    cfg=OmegaConf.create(identity["config"]);cfg.framework.qwenvl.device_map="cpu"
    model=build_framework(cfg).float();state=load_file(str(root/"master.safetensors"))
    actual={k:{"shape":list(v.shape),"dtype":str(v.dtype)} for k,v in state.items()}
    if actual!=record["inventory"]:raise ValueError("Replay parameter inventory differs")
    model.load_state_dict(state,strict=True);del state
    model.to(device).eval();model.inference_fp32=True
    return model


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--baseline-manifest",required=True);p.add_argument("--output",required=True)
    a=p.parse_args();build(a.baseline_manifest,a.output)
