"""Cached master -> original S0 trajectories from the prior canonical export."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from .runtime import numerical_policy,source_sha
from .train_representation import load_model,observation
from starVLA.dataloader.foresight_dataset import ForesightCurrentDataset
from starVLA.teachers.decodable_st.source_registry import atomic_json,sha256
from tools.foresight.checkpoints import scene_noise


def run(config,output,reference_root,device="cuda"):
    numerical_policy(42,device,.6);start=time.monotonic();cfg=dict(config)
    for key in ("spatial_readout_initialization","spatial_readout_kind","spatial_grid"):cfg.pop(key,None)
    if not cfg.get("baseline_state_cache"):raise ValueError("No explicit verified replay cache")
    model,baseline,_=load_model(cfg,device,"R0",42);model.eval()
    data=ForesightCurrentDataset(baseline["data"]["dev"]["root"]);lookup={r["token"]:i for i,r in enumerate(data.index)}
    paths=sorted((Path(reference_root)/"q0/predictions").glob("*.npz"))
    if len(paths)!=8:raise ValueError("Existing frozen eight-scene S0 reference required")
    rows=[]
    with torch.no_grad():
        for path in paths:
            example=data[lookup[path.stem]];prediction=model.predict_action(observation(example),initial_noise=scene_noise(path.stem,42,device)).cpu()[0]
            reference=torch.from_numpy(np.load(path)["trajectory"])
            torch.testing.assert_close(prediction,reference,rtol=1e-5,atol=1e-5)
            rows.append({"token":path.stem,"max_abs_trajectory_error":float((prediction-reference).abs().max()),"reference_sha256":sha256(path)})
    result={"status":"PASS","source_sha":source_sha(),"baseline_identity":baseline["identity"],"rows":rows,
        "max_abs_trajectory_error":max(r["max_abs_trajectory_error"] for r in rows),
        "precision":"FP32 TF32 off, original single-scene Euler10/noise42","seconds":time.monotonic()-start,
        "gpu_hours":(time.monotonic()-start)/3600,"scope":"baseline master replay parity; not representation/planning acceptance"}
    atomic_json(output,result);return result


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--output",required=True)
    p.add_argument("--reference-root",required=True);p.add_argument("--device",default="cuda")
    a=p.parse_args();run(json.loads(Path(a.config).read_text()),a.output,a.reference_root,a.device)
