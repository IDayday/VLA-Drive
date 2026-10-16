"""Small native-checkpoint helpers for ST runs, reusing original task RNG state.

Checkpoint contents use DeepSpeed's native schema. The receipt is written only
after every rank's optimizer/model and consumed-cursor RNG files are durable.
Incomplete directories remain diagnostic evidence and never become resume points.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re

from starVLA.dataloader.st_world_targets import atomic_json
from starVLA.model.modules.vehicle_joint.initialization import file_sha256, identity_hash


TERMINATIONS = {"RUNNING_CHECKPOINT", "OPERATOR_PAUSED", "BUDGET_PAUSED", "MAX_STEPS"}


def require_native_inventory(names, world_size):
    """Exact installed ZeRO2 layout, not 'some optimizer file exists'."""
    if type(world_size) is not int or world_size < 1:
        raise ValueError("A positive native data-parallel world size is required")
    names = set(names)
    optimizer_names = {name for name in names if name.endswith("_optim_states.pt")}
    expected_optimizer = {f"bf16_zero_pp_rank_{rank}_mp_rank_00_optim_states.pt"
                          for rank in range(world_size)}
    if optimizer_names != expected_optimizer:
        raise ValueError("Missing or foreign native optimizer rank shards: " +
                         str(sorted(optimizer_names.symmetric_difference(expected_optimizer))))
    model_names = {name for name in names if name.endswith("_model_states.pt")}
    if model_names != {"mp_rank_00_model_states.pt"}:
        raise ValueError("Missing or foreign native ZeRO2 model shard")
    rng_names = {name for name in names if name.startswith("rng_rank") and name.endswith(".pt")}
    if rng_names != {f"rng_rank{rank}.pt" for rank in range(world_size)}:
        raise ValueError("Missing or foreign consumed-cursor rank RNG inventory")


def checkpoint_receipt(folder, state, *, world_size):
    folder = Path(folder)
    if state["termination_reason"] not in TERMINATIONS or state["completed"] < 0:
        raise ValueError("Explicit native optimizer-boundary progress is required")
    files = sorted(folder.glob("*.pt"))
    names = {p.name for p in files}
    require_native_inventory(names, world_size)
    result = {"schema": "st_native_checkpoint_v1", "state": state, "world_size": world_size,
              "files": {p.name: {"sha256": file_sha256(p), "bytes": p.stat().st_size} for p in files}}
    result["receipt_identity"] = identity_hash(result)
    atomic_json(folder / "COMPLETE.json", result, immutable=True)
    return result


def verify_checkpoint(folder, *, identity, world_size):
    folder = Path(folder)
    receipt = json.loads((folder / "COMPLETE.json").read_text())
    body = {k: v for k, v in receipt.items() if k != "receipt_identity"}
    if (receipt.get("schema") != "st_native_checkpoint_v1" or identity_hash(body) != receipt["receipt_identity"]
            or receipt["state"]["identity"] != identity or receipt["world_size"] != world_size
            or receipt["state"]["tag"] != folder.name):
        raise ValueError("Foreign checkpoint/receipt/source or world size")
    require_native_inventory(receipt["files"], world_size)
    for name, info in receipt["files"].items():
        if Path(name).name != name:
            raise ValueError("Unsafe checkpoint record")
        p = folder / name
        if p.stat().st_size != info["bytes"] or file_sha256(p) != info["sha256"]:
            raise ValueError("Incomplete or corrupt checkpoint: " + name)
    if receipt["state"]["termination_reason"] not in TERMINATIONS:
        raise ValueError("Unknown saved terminal status")
    return receipt


def resolve_latest(root, *, identity, world_size):
    """Choose the latest complete compatible state, not a stale latest pointer."""
    root = Path(root)
    candidates = []
    for folder in root.iterdir() if root.exists() else []:
        if not folder.is_dir():
            continue
        try:
            metadata = json.loads((folder / "COMPLETE.json").read_text())
            step = metadata["state"]["completed"]
            if type(step) is not int or step < 0:
                raise ValueError("Invalid progress")
            candidates.append((step, folder, None))
        except (OSError, ValueError, KeyError, TypeError) as error:
            candidates.append((-1, folder, str(error)))
    rejected = [{"path": str(folder), "declared_step": step, "reason": error}
                for step, folder, error in candidates if error is not None]
    selected = None
    # Same-step snapshots differ only in saved stop intent. Tag order is stable;
    # MAX_STEPS is additionally enforced from completed >= the fixed horizon.
    for step, folder, error in sorted(candidates, key=lambda x: (x[0], x[1].name), reverse=True):
        if error is not None:
            continue
        if error is None:
            try:
                receipt = verify_checkpoint(folder, identity=identity, world_size=world_size)
                selected = {"folder": str(folder), "receipt": receipt}
                break
            except (OSError, ValueError, KeyError, TypeError) as exc:
                error = str(exc)
        rejected.append({"path": str(folder), "declared_step": step, "reason": error})
    report = {"selected": selected, "rejected": rejected,
              "enumerated_directories": [str(folder) for _, folder, _ in candidates],
              "policy": "latest_complete_compatible_with_explicit_corruption_report"}
    if selected is None:
        raise ValueError("No valid native resume state: " + json.dumps(report))
    return report


def update_latest(root, tag):
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", tag):
        raise ValueError("Unsafe native checkpoint tag")
    root = Path(root)
    temporary = root / ("latest.tmp." + str(os.getpid()))
    with temporary.open("w") as stream:
        stream.write(tag + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(root / "latest")


def terminal(state, max_updates):
    """Read before consuming sampler or RNG; an operator pause can resume."""
    if state["completed"] > max_updates:
        raise ValueError("Cannot shrink an already-consumed fixed training budget")
    return state["termination_reason"] == "MAX_STEPS" or state["completed"] == max_updates
