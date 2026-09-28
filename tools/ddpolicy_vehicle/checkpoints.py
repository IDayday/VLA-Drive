"""Only this campaign's complete, identity-bound ZeRO checkpoints are accepted."""
import json
from pathlib import Path
from starVLA.model.modules.vehicle_joint.initialization import file_sha256, identity_hash


def checkpoint_identity(run, tag):
    run = Path(run)
    identity = json.loads((run/"identity.json").read_text())
    if identity_hash({k:v for k,v in identity.items() if k != "sha256"}) != identity["sha256"]:
        raise ValueError("Training identity changed")
    checkpoint = run/"checkpoints"/tag
    if checkpoint.parent.resolve() != (run/"checkpoints").resolve():
        raise ValueError("Checkpoint tag must be one directory name")
    complete = json.loads((checkpoint/"COMPLETE.json").read_text())
    if complete["identity"] != identity["sha256"] or complete["tag"] != tag:
        raise ValueError("Incomplete or mismatched campaign checkpoint")
    files = {p.name:file_sha256(p) for p in sorted(checkpoint.glob("*.pt"))}
    if not any("model_states" in x for x in files) or not any("optim_states" in x for x in files):
        raise ValueError("Checkpoint lacks model or FP32 optimizer masters")
    record = {"run_identity": identity["sha256"], "training_source_sha": identity["source_sha"],
              "tag": tag, "completed": complete["completed"], "arm": identity["arm"],
              "startup": identity["startup"], "files": files}
    return identity, {"sha256": identity_hash(record), **record}


def scene_noise(token, sampling_seed, actors, device):
    """Partition-independent per-scene noise; slot0 is identical in A/B/C."""
    import hashlib
    import torch
    seed = int.from_bytes(hashlib.sha256(f"ddpolicy-v1:{sampling_seed}:{token}".encode()).digest()[:8], "little") % (2**63-1)
    # Always generate the same full shape, including Base's unused actor slots.
    value = torch.randn((1, 9, 8, 4), generator=torch.Generator().manual_seed(seed), dtype=torch.float32)
    return value[:, :actors].to(device)
