"""Small explicit native checkpoint utilities shared by bounded experiments."""
import hashlib
import json
import os
import random
import subprocess
from pathlib import Path
import numpy as np
import torch
from starVLA.teachers.decodable_st.source_registry import atomic_json, object_hash


def source_sha():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def parameter_digest(module):
    h=hashlib.sha256()
    for k,v in module.state_dict().items():
        h.update(k.encode());h.update(v.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def rng_state():
    return {"python":random.getstate(),"numpy":np.random.get_state(),"torch":torch.get_rng_state(),
            "cuda":torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"]);np.random.set_state(state["numpy"]);torch.set_rng_state(state["torch"])
    if state["cuda"]:torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(path, *, model, optimizer, scheduler, identity, progress, sampler, extra=None):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    state={"schema":"decodable_st_native_checkpoint_v1","identity":identity,
           "model":model.state_dict(),"optimizer":optimizer.state_dict(),"scheduler":scheduler.state_dict(),
           "scaler":None,"rng":rng_state(),"progress":progress,"sampler":sampler,"extra":extra or {}}
    tmp=path.with_suffix(path.suffix+".tmp")
    with tmp.open("wb") as f:torch.save(state,f);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)


def load_checkpoint(path, *, model, optimizer, scheduler, identity):
    state=torch.load(path,map_location="cpu",weights_only=False)
    if state.get("schema")!="decodable_st_native_checkpoint_v1" or state.get("identity")!=identity:
        raise ValueError("Checkpoint schema/campaign identity mismatch; warm-start is not resume")
    model.load_state_dict(state["model"],strict=True)
    optimizer.load_state_dict(state["optimizer"]);scheduler.load_state_dict(state["scheduler"])
    restore_rng(state["rng"])
    return state


def write_identity(root,config):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    identity=object_hash(config);path=root/"identity.json"
    if path.exists():
        if json.loads(path.read_text())["identity"]!=identity:raise ValueError("Refuse to overwrite a different run")
    else:atomic_json(path,{"identity":identity,**config})
    return identity


def numerical_policy(seed,device,memory_fraction=.65):
    torch.set_num_threads(2)
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32=torch.backends.cudnn.allow_tf32=False
    if str(device).startswith("cuda"):torch.cuda.set_per_process_memory_fraction(memory_fraction)
