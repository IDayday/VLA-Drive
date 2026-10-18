import hashlib
import json
import os
from pathlib import Path
import subprocess


def sha256(path):
    h = hashlib.sha256()
    with open(path,"rb") as f:
        for data in iter(lambda:f.read(8<<20), b""): h.update(data)
    return h.hexdigest()


def object_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary = path.with_name(path.name+f".{os.getpid()}.tmp")
    with temporary.open("w") as f:
        json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write("\n");f.flush();os.fsync(f.fileno())
    os.replace(temporary,path)


def verify_source(path, expected_commit):
    actual = subprocess.check_output(["git","-C",str(path),"rev-parse","HEAD"],text=True).strip()
    if actual != expected_commit:
        raise ValueError(f"External source mismatch: {actual} != {expected_commit}")
    if subprocess.check_output(["git","-C",str(path),"status","--porcelain"],text=True).strip():
        raise ValueError("External source has unregistered modifications")


def load_exact_subset(module, weights, prefix):
    from safetensors import safe_open
    with safe_open(str(weights),framework="pt",device="cpu") as f:
        selected = {k[len(prefix):]:f.get_tensor(k) for k in f.keys() if k.startswith(prefix)}
    if not selected:
        raise ValueError(f"No checkpoint parameters for {prefix}")
    expected = module.state_dict()
    if set(selected) != set(expected):
        raise ValueError(f"Strict head inventory mismatch: missing={sorted(set(expected)-set(selected))}, extra={sorted(set(selected)-set(expected))}")
    for k,v in selected.items():
        if v.shape != expected[k].shape:
            raise ValueError(f"Checkpoint shape mismatch: {prefix}{k}")
    module.load_state_dict(selected,strict=True)
    return {"prefix":prefix,"parameters_and_buffers":len(selected),"missing":[],"unexpected":[],
            "checkpoint_dtypes":sorted({str(x.dtype) for x in selected.values()})}
