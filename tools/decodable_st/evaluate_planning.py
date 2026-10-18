"""Current-only exported student -> unchanged canonical single-trajectory scorer.

The bounded development smoke establishes plumbing, not planning improvement.
No Navtest dispatch is allowed without an identity-bound final method lock.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import numpy as np
import torch
from starVLA.teachers.decodable_st.source_registry import atomic_json,sha256,object_hash
from .runtime import numerical_policy,source_sha


def run(config,output,device="cuda"):
    from starVLA.dataloader.decodable_st_dataset import DecodableDataset
    from starVLA.dataloader.foresight_dataset import decode_ego
    from tools.foresight.checkpoints import scene_noise
    numerical_policy(config["seed"],device,.70)
    exported=Path(config["export_root"]);manifest=json.loads((exported/"manifest.json").read_text())
    if sha256(exported/"student.pt")!=manifest["weights_sha256"]:raise ValueError("Export weight identity mismatch")
    data=DecodableDataset(config["current_root"]);limit=int(config.get("limit",0));root=Path(output);root.mkdir(parents=True,exist_ok=True)
    population=data.current.identity
    if population["split"]=="navtest":
        lock=json.loads(Path(config["final_method_lock"]).read_text())
        expected={"weights_sha256":manifest["weights_sha256"],"population":population,"sampling_seed":config["seed"],"evaluation_source":source_sha()}
        if limit or lock.get("identity")!=expected or lock.get("representation_and_planner_accepted") is not True:
            raise ValueError("Navtest requires a frozen accepted full-population final method")
    elif config.get("scope")=="engineering" and not 0<limit<=8:
        raise ValueError("Development engineering smoke is bounded to eight scenes")
    model=torch.load(exported/"student.pt",map_location="cpu",weights_only=False).to(device).eval()
    if any(p.dtype!=torch.float32 for p in model.parameters()):raise ValueError("Canonical export requires FP32 parameters")
    count=min(limit or len(data),len(data));banks={};start=time.monotonic()
    for kind in ("q0","q_final"):
        bank=root/kind;(bank/"predictions").mkdir(parents=True,exist_ok=True)
        identity={"checkpoint":{"sha256":manifest["weights_sha256"],"scope":"small_fit" if config["scope"]=="engineering" else "formal"},
          "current_identity":population,"evaluation_source":source_sha(),"world_size":1,"limit":limit,
          "protocol":{"precision":"FP32","tf32":False,"scorer":None,"candidates_per_scene":1,"future_conditioning":False,
            "sampling_seed":config["seed"],"solver":"original Euler","steps":10,"branch":kind,
            "evaluation_purpose":"engineering development smoke" if config["scope"]=="engineering" else "frozen final comparison"}}
        old=bank/"identity.json"
        if old.exists() and json.loads(old.read_text())!=identity:raise ValueError("Refuse foreign export overwrite")
        atomic_json(old,identity);banks[kind]=(bank,identity,object_hash(identity))
    for i in range(count):
        observation,_,scene=data[i];noise=scene_noise(scene.scene_token,config["seed"],device)
        with torch.no_grad():
            rep=model(observation)
            conditions={"q0":model.baseline.build_planner_condition({"action_queries":rep.original_action_hidden,"W":rep.dino_hidden}),
                        "q_final":model.action_condition(rep)}
            with model.baseline.amp():
                values={k:decode_ego(model.baseline.action_model.predict_action(c,initial_noise=noise.clone()).float())[0] for k,c in conditions.items()}
        for kind,trajectory in values.items():
            if trajectory.shape!=(8,3) or not torch.isfinite(trajectory).all():raise FloatingPointError("Invalid trajectory; population may not be shortened")
            bank,_,signature=banks[kind];dest=bank/"predictions"/(scene.scene_token+".npz");temp=dest.with_suffix(".tmp")
            with temp.open("wb") as f:np.savez_compressed(f,trajectory=trajectory.cpu().numpy())
            os.replace(temp,dest)
            atomic_json(dest.with_suffix(".json"),{"token":scene.scene_token,"log":scene.log_id,"identity_sha256":signature,"status":"ok","proposal_sha256":sha256(dest)})
    for bank,identity,signature in banks.values():
        atomic_json(bank/"shard_0.json",{"status":"complete","requested":count,"completed":count,"failed":0,"identity_sha256":signature})
    result={"status":"INCONCLUSIVE","export_complete":True,"scenes":count,"seconds":time.monotonic()-start,
         "gpu_hours":(time.monotonic()-start)/3600,"planning_gain":"NOT_ESTABLISHED","scope":config["scope"],"banks":[str(v[0]) for v in banks.values()]}
    atomic_json(root/"EXPORT.json",result)
    # CPU execution uses the existing evaluator in its own unchanged environment.
    if config.get("score"):
        score=config["score"]
        for kind,(bank,_,_) in banks.items():
            cmd=[score["python"],"-m","tools.foresight.score_pdms","--devkit",score["devkit"],"--metric-index",score["metric_index"],
                "--current-index",str(Path(config["current_root"])/"index.json"),"--predictions",str(bank),"--output",str(root/(kind+"_scores")),
                "--campaign-root",str(root/"cpu_meter"),"--run-id",kind,"--workers",str(score.get("workers",4)),"--timeout-seconds","600"]
            with (root/(kind+"_score.log")).open("a") as f:
                subprocess.run(cmd,check=True,stdout=f,stderr=subprocess.STDOUT)
        result["canonical_scoring_complete"]=True;atomic_json(root/"EXPORT.json",result)
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--output",required=True);p.add_argument("--device",default="cuda");a=p.parse_args()
    run(json.loads(Path(a.config).read_text()),a.output,a.device)
