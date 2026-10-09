from __future__ import annotations
import hashlib
import torch
from ..contracts import require, periodic_error


def module_hash(module):
    h = hashlib.sha256()
    for name, t in sorted(module.state_dict().items()):
        h.update(name.encode())
        h.update(str(t.dtype).encode())
        h.update(str(tuple(t.shape)).encode())
        h.update(t.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def frozen_snapshot(model, active_id=None):
    modules = {"shared": model.adapter}
    modules.update({k: v for k, v in model.experts.items() if k != active_id})
    return {k: {"parameters": module_hash_parameters(m), "buffers": module_hash_buffers(m)} for k, m in modules.items()}


def _tensor_map_hash(tensors):
    h = hashlib.sha256()
    for name, t in sorted(tensors):
        h.update(name.encode()); h.update(str(t.dtype).encode()); h.update(str(tuple(t.shape)).encode())
        h.update(t.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def module_hash_parameters(module):
    return _tensor_map_hash(module.named_parameters())


def module_hash_buffers(module):
    return _tensor_map_hash(module.named_buffers())


def audit_frozen(model, before, references, features, tolerances):
    current = frozen_snapshot(model, model.active_expert_id if model.stage == "expert_train" else None)
    require(before == current, "frozen parameter/buffer hashes changed")
    reports = {}
    model.eval()
    with torch.no_grad():
        for eid, reference in references.items():
            output = model.adapter.expert_forward(model.experts[eid], features)
            raw_error = (output.raw - reference["raw"].to(output.raw)).abs().max().item()
            physical = reference["physical"].to(output.physical)
            delta = (output.physical - physical).abs().max().item()
            xy = (output.physical[..., :2] - physical[..., :2]).norm(dim=-1)
            yaw = periodic_error(output.physical[..., 2], physical[..., 2]).abs()
            require(raw_error <= tolerances["raw_max_abs"] and delta <= tolerances["physical_max_abs"], "old output drift exceeds preregistered tolerance")
            reports[eid] = {"raw_max_abs": raw_error, "physical_max_abs": delta, "ADE_m": xy.mean().item(),
                            "FDE_m": xy[:, -1].mean().item(), "yaw_error_rad": yaw.mean().item()}
    return {"frozen_hashes": current, "output_drift": reports, "passed": True}
