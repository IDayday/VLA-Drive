"""Explicit same-world placement migration; never alters model/optimizer math.

Native checkpoints store CUDA RNG for every visible local GPU. Splitting an
eight-GPU node requires projecting those lists onto the new local GPU list.
Global ranks, data cursors, pending chains and each active RNG remain identical.
The original checkpoint stays immutable; migrated snapshots have separate seals.
Only splitting existing rank groups is supported, never merging or reordering.
"""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import time
import torch
from starVLA.rl.flow_grpo.checkpoint import directory_seal, validate_checkpoint
from starVLA.rl.flow_grpo.loading import file_sha
from starVLA.rl.flow_grpo.transactions import atomic_json, publication_lock, preserve_attempt


def rank_layout(nodes):
    rows = []
    slots = set()
    for node_index, node in enumerate(nodes):
        if not node["devices"]:
            raise ValueError("empty node")
        for local_rank, gpu in enumerate(node["devices"]):
            slot = (node["host"], gpu)
            if slot in slots:
                raise ValueError("duplicate GPU slot")
            slots.add(slot)
            rows.append({"node": node_index, "local_rank": local_rank,
                         "local_world_size": len(node["devices"]), "host": node["host"], "gpu": gpu})
    return rows


def rng_projection(source_nodes, target_nodes):
    source, target = rank_layout(source_nodes), rank_layout(target_nodes)
    if not source or len(source) != len(target):
        raise ValueError("same nonempty world size required")
    result = []
    for rank, row in enumerate(target):
        peers = [r for r, peer in enumerate(target) if peer["node"] == row["node"]]
        if len({source[r]["node"] for r in peers}) != 1:
            raise ValueError("only splitting old rank groups is supported")
        indices = [source[r]["local_rank"] for r in peers]
        assert indices[row["local_rank"]] == source[rank]["local_rank"]
        result.append({"rank": rank, "source": source[rank], "target": row, "indices": indices})
    return result


def project_checkpoint(source, destination, source_nodes, target_nodes, cfg):
    source, destination = Path(source).resolve(), Path(destination).absolute()
    mapping = rng_projection(source_nodes, target_nodes)
    state = json.loads((source/"trainer_state.json").read_text())
    validate_checkpoint(source, cfg, state["provenance"], len(mapping))
    identity = {"schema_version": 1, "source": str(source),
                "source_seal_sha256": file_sha(source/"checkpoint_files.json"), "mapping": mapping}
    with publication_lock(destination.with_name(destination.name+".lock")):
        if destination.exists():
            receipt = destination/"deployment_rng.json"
            if receipt.is_file() and json.loads(receipt.read_text()) != identity:
                raise ValueError("checkpoint deployment identity conflict")
            if (destination/"COMPLETE").is_file():
                if not receipt.is_file():
                    raise ValueError("completed migration missing receipt")
                validate_checkpoint(destination, cfg, state["provenance"], len(mapping))
                return destination
            preserve_attempt(destination)
        temporary = destination.with_name("."+destination.name+".incomplete")
        if temporary.exists():
            preserve_attempt(temporary)
        temporary.mkdir(parents=True)
        # Hardlink immutable large tensors when possible; RNG files are rewritten
        # using new inodes. No write may follow a link back into the source.
        rewritten = {f"rank_{i}.pt" for i in range(len(mapping))} | {
            f"random_states_{i}.pkl" for i in range(len(mapping))}
        for path in source.rglob("*"):
            if not path.is_file():
                continue
            relative = path.relative_to(source)
            if str(relative) in rewritten | {"COMPLETE", "checkpoint_files.json", "deployment_rng.json"}:
                continue
            target = temporary/relative
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(path.resolve(), target)
            except OSError:
                shutil.copy2(path, target)
        for row in mapping:
            rank = row["rank"]
            for name, container, key in [(f"rank_{rank}.pt", "rng", "cuda"),
                                         (f"random_states_{rank}.pkl", None, "torch_cuda_manual_seed")]:
                data = torch.load(source/name, map_location="cpu", weights_only=False)
                parent = data[container] if container else data
                original = parent[key]
                if len(original) != row["source"]["local_world_size"]:
                    raise ValueError(f"RNG inventory does not match source topology: {name}")
                parent[key] = [original[i].clone() for i in row["indices"]]
                if not torch.equal(parent[key][row["target"]["local_rank"]],
                                   original[row["source"]["local_rank"]]):
                    raise AssertionError("active GPU RNG changed")
                torch.save(data, temporary/name)
        atomic_json(temporary/"deployment_rng.json", identity)
        atomic_json(temporary/"checkpoint_files.json", directory_seal(temporary))
        (temporary/"COMPLETE").write_text("same global ranks; explicitly projected local CUDA RNG inventory\n")
        # Native name validation uses update_NNNNNN, so validate after publication.
        os.replace(temporary, destination)
        validate_checkpoint(destination, cfg, state["provenance"], len(mapping))
        return destination


def migrate_descriptor(root, descriptor, cluster_identity):
    """Caller holds exclusive_controller. Gate GPU proof before publishing."""
    from scripts.cluster_flow_grpo.identity import validate_binding
    from starVLA.rl.flow_grpo.config import resolve_config
    from starVLA.rl.flow_grpo.acceptance import enforce_training_budget
    root = Path(root)
    old = json.loads((root/"cluster_identity.json").read_text())
    if old["training_executable_sha256"] != descriptor["training_executable_sha256"]:
        raise ValueError("deployment migration cannot change actor executable")
    before, after = deepcopy(old["plan"]), deepcopy(descriptor["plan"])
    after.pop("resource_limits", None); before.pop("resource_limits", None)
    # Scheduling-only migration: fixed save/eval targets, recipes and native
    # checkpoint semantics stay unchanged. The new source still needs release.
    before.pop("async_evaluation", None); after.pop("async_evaluation", None)
    from scripts.cluster_flow_grpo.paired import validate_plan
    if "async_evaluation" in descriptor["plan"]:
        validate_plan(descriptor["plan"])
    reuse = after.pop("asset_verification_receipt", None)
    before.pop("asset_verification_receipt", None)
    if reuse:
        from scripts.cluster_flow_grpo.assets import install_receipt
        install_receipt(reuse["path"], reuse["sha256"])
    for variant, group in after["groups"].items():
        original = before["groups"][variant]
        layout = group.pop("resume_layout", None)
        original.pop("resume_layout", None)
        if group["nodes"] != original["nodes"]:
            if not layout or layout["source_nodes"] != original["nodes"]:
                raise ValueError("migration lacks matching source topology")
            rng_projection(original["nodes"], group["nodes"])
            checkpoints = list((root/variant/"checkpoints").glob("update_*/COMPLETE"))
            if not checkpoints or max(int(p.parent.name.split("_")[-1]) for p in checkpoints) != layout["through_update"]:
                raise ValueError("migration cutoff must equal latest completed update")
        group["nodes"] = original["nodes"]
        group.pop("placement_validation", None); original.pop("placement_validation", None)
        cfg, _ = resolve_config(group["config"])
        enforce_training_budget(cfg)
        validate_binding(cfg, cluster_identity)
    if before != after:
        raise ValueError("only explicit resource placement may change")
    archive = root/"deployment_history"/str(time.time_ns())
    archive.mkdir(parents=True)
    atomic_json(archive/"previous_identity.json", old)
    atomic_json(archive/"migration.json", {"new_identity": descriptor,
        "scope": "placement/scheduling only; native actor/config/world unchanged; original checkpoints preserved"})
    atomic_json(root/"cluster_identity.json", descriptor)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--source", required=True); p.add_argument("--destination", required=True)
    p.add_argument("--source-spec", required=True); p.add_argument("--target-spec", required=True)
    p.add_argument("--config", required=True)
    a = p.parse_args()
    from starVLA.rl.flow_grpo.config import resolve_config
    cfg, _ = resolve_config(a.config)
    nodes = [json.loads(Path(path).read_text())["nodes"] for path in (a.source_spec, a.target_spec)]
    print(project_checkpoint(a.source, a.destination, *nodes, cfg))


if __name__ == "__main__":
    main()
