"""Trusted, version-locked offline PyTorch research export, with no teacher modules.

The artifact includes the complete executable student, processor/tokenizer and
weights. It is not a safe loader for untrusted pickle files. Never publish weights.
"""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import torch
from starVLA.teachers.decodable_st.source_registry import atomic_json,sha256
from .runtime import numerical_policy,source_sha


def export_loaded(model,output,provenance):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    model.eval();model.baseline.strip_auxiliary_heads()
    if any(any(t in type(m).__module__ for t in ("teachers.decodable_st","third_party.decodable_st")) for m in model.modules()):
        raise ValueError("Teacher or fixed decoder present in student export")
    # Explicitly retain student queries/readouts/reader, even when untrained.
    model.baseline.qwen_vl_interface.processor.save_pretrained(output/"processor")
    model.cpu()
    with (output/"student.pt.tmp").open("wb") as f:torch.save(model,f);f.flush();os.fsync(f.fileno())
    os.replace(output/"student.pt.tmp",output/"student.pt")
    record={"schema":"decodable_st_trusted_offline_module_v1","source_sha":source_sha(),"provenance":provenance,
          "weights_sha256":sha256(output/"student.pt"),"weights_bytes":(output/"student.pt").stat().st_size,
          "precision":"FP32 TF32 disabled","torch":torch.__version__,"transformers":__import__('transformers').__version__,
          "spatial_contract":asdict(model.spatial_contract) if model.spatial_contract else None,
          "temporal_contract":asdict(model.temporal_contract) if model.temporal_contract else None,
          "reader_enabled":model.reader_enabled,"teacher_decoder_modules":0,
          "retained_DINO_deployment":["foresight_queries","query_geometry","original Qwen/ego/DiT"],
          "export_load_policy":"trusted local artifact only, sha256 checked, version-locked repository required"}
    atomic_json(output/"manifest.json",record);return record


def run(config,output,device="cuda",run_root=None):
    from .train_representation import load_model,load_data,observation
    from starVLA.model.modules.decodable_st.training_state import TrainableDelta
    from tools.foresight.checkpoints import scene_noise
    numerical_policy(42,device,.75)
    arm="RS";seed=42
    if run_root:
        ident=json.loads((Path(run_root)/"identity.json").read_text());arm=ident["arm"];seed=ident["seed"]
    model,baseline,_=load_model(config,device,arm,seed)
    provenance={"baseline_identity":baseline["identity"],"arm":arm,"training_status":"untrained engineering export"}
    if run_root:
        state=torch.load(Path(run_root)/"latest.pt",map_location="cpu",weights_only=False)
        if state["identity"]!=ident["identity"]:raise ValueError("Export checkpoint identity mismatch")
        delta=TrainableDelta(model,[x["name"] for x in ident["trainable_inventory"]]);delta.load_state_dict(state["model"],strict=True)
        provenance.update(run_identity=ident["identity"],progress=state["progress"],training_status="engineering/pilot only; no planning acceptance")
    data,rows,lookup=load_data(config);row=rows[0];example,_=data[lookup[row["token"]]]
    obs=observation(example);noise=scene_noise(row["token"],42,device)
    model.eval()
    with torch.no_grad():reference=model.predict_action(obs,initial_noise=noise)
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    # Private probe contains current images only; no labels or teacher tensors.
    torch.save({"observation":obs,"noise":noise.cpu(),"trajectory":reference.cpu()},output/"current_only_probe.pt")
    return export_loaded(model,output,provenance)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--output",required=True)
    p.add_argument("--run-root");p.add_argument("--device",default="cuda");a=p.parse_args()
    run(json.loads(Path(a.config).read_text()),a.output,a.device,a.run_root)
