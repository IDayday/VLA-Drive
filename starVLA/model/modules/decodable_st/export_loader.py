"""Teacher-free loader for explicitly trusted, version-locked research exports."""
import hashlib
import json
from pathlib import Path
import torch


def checksum(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(8<<20),b""):h.update(block)
    return h.hexdigest()


def load_trusted_export(root,device="cuda"):
    root=Path(root);record=json.loads((root/"manifest.json").read_text())
    if record["schema"]!="decodable_st_trusted_offline_module_v1":raise ValueError("Unknown export schema")
    import transformers
    if record["torch"]!=torch.__version__ or record["transformers"]!=transformers.__version__:
        raise ValueError("Export software versions differ; revalidate explicitly before use")
    if checksum(root/"student.pt")!=record["weights_sha256"]:raise ValueError("Export checksum mismatch")
    model=torch.load(root/"student.pt",map_location="cpu",weights_only=False).to(device).eval()
    if record.get("overlay"):
        from safetensors.torch import load_file
        from .training_state import TrainableDelta
        overlay=record["overlay"]
        if Path(overlay["file"]).name!=overlay["file"]:raise ValueError("Overlay must be contained in export")
        payload=root/overlay["file"]
        if checksum(payload)!=overlay["sha256"]:raise ValueError("Overlay checksum mismatch")
        TrainableDelta(model,overlay["inventory"]).load_state_dict(load_file(str(payload)),strict=True)
    model.reader_enabled=record["reader_enabled"]
    return model,record
