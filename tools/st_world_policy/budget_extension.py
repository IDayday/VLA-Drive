"""Explicit phase-budget extension with unchanged mathematics and native state.

A screened 5k run is immutable. Its approved 20k continuation has a new run
identity binding the exact complete parent checkpoint and registration. This
never resets the 20k cosine horizon, initialization, optimizer or data cursor.
The normal independent FORMAL approval applies to the new registration too.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

from starVLA.model.modules.vehicle_joint.initialization import file_sha256, identity_hash
from tools.st_world_policy.training_state import verify_checkpoint


def validate_extension_config(value):
    if not isinstance(value, dict) or set(value) != {
            "parent_registration", "parent_tag", "parent_receipt_sha256"}:
        raise ValueError("Budget extension requires exact parent registration, tag and receipt")
    ref = value["parent_registration"]
    if not isinstance(ref, dict) or set(ref) != {"path", "sha256"} or not Path(ref["path"]).is_absolute():
        raise ValueError("Explicit absolute parent registration and checksum required")
    for digest in (ref["sha256"], value["parent_receipt_sha256"]):
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("SHA256 parent evidence required")
    tag = value["parent_tag"]
    if not isinstance(tag, str) or Path(tag).name != tag or tag in ("", ".", ".."):
        raise ValueError("Safe explicit complete parent tag required")


def _recipe(cfg):
    result = copy.deepcopy(cfg)
    result.pop("continuation", None)
    result["execution"].pop("stop_after")
    result["runtime"].pop("output_root")
    return result


def resolve_extension(cfg, registration, *, repo, data_identity, base_binding):
    """Validate the parent before constructing/advancing the new run state."""
    from tools.st_world_policy.check_formal_approval import verify_registration
    request = cfg["continuation"]
    validate_extension_config(request)
    path = Path(request["parent_registration"]["path"])
    if file_sha256(path) != request["parent_registration"]["sha256"]:
        raise ValueError("Parent registration checksum changed")
    parent_reg = json.loads(path.read_text())
    verify_registration(path, repo=repo)
    parent_cfg = json.loads(Path(parent_reg["artifacts"]["config"]["path"]).read_text())
    if (parent_reg["source_fingerprint"] != registration["source_fingerprint"] or
            _recipe(parent_cfg) != _recipe(cfg) or
            not parent_cfg["execution"]["stop_after"] < cfg["execution"]["stop_after"]):
        raise ValueError("Extension may only increase stop_after within the original fixed recipe")
    if cfg["runtime"].get("profile_mode", "none") != "none":
        raise ValueError("Measured profile windows cannot be extended or reused as formal weights")
    parent_run = Path(parent_cfg["runtime"]["output_root"]) / parent_reg["run_id"]
    new_run = Path(cfg["runtime"]["output_root"]) / registration["run_id"]
    if parent_run.resolve() == new_run.resolve():
        raise ValueError("A budget extension requires a new immutable run identity")
    identity_path = parent_run / "identity.json"
    identity = json.loads(identity_path.read_text())
    if (identity.get("schema") != "st_world_training_v1" or
            identity_hash({k: v for k, v in identity.items() if k != "identity"}) != identity["identity"] or
            identity["registration_sha256"] != file_sha256(path) or identity["config"] != parent_cfg or
            identity["data"] != data_identity or identity["base_binding"] != base_binding or
            identity["weights_usable_for_formal_init"] != registration["weights_usable_for_formal_init"]):
        raise ValueError("Parent source/data/Base/scope does not match continuation")
    folder = parent_run / "checkpoints" / request["parent_tag"]
    if file_sha256(folder / "COMPLETE.json") != request["parent_receipt_sha256"]:
        raise ValueError("Parent complete checkpoint receipt changed")
    receipt = verify_checkpoint(folder, identity=identity["identity"],
                                world_size=cfg["execution"]["world_size"])
    if (receipt["state"]["completed"] != parent_cfg["execution"]["stop_after"] or
            receipt["state"]["termination_reason"] not in {"RUNNING_CHECKPOINT", "BUDGET_PAUSED"} or
            receipt["state"]["completed"] >= cfg["execution"]["max_updates"]):
        raise ValueError("Extension requires the exact completed earlier phase, not an arbitrary restart")
    return {"schema": "st_budget_extension_v1", "selected": {"folder": str(folder), "receipt": receipt},
            "parent_run": str(parent_run), "parent_identity": identity["identity"],
            "parent_identity_sha256": file_sha256(identity_path), "request": request,
            "old_stop_after": parent_cfg["execution"]["stop_after"],
            "new_stop_after": cfg["execution"]["stop_after"],
            "original_schedule_max_updates": cfg["execution"]["max_updates"],
            "optimizer_rng_sampler_preserved": True, "formal_approval_is_separate": True}
