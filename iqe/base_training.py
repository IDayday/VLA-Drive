"""Portable online-image Query Base training, with trajectory IL as its only loss.

No persistent feature-pool implementation or service is used by this entrypoint.
Run from the repository root. Data/model paths in the original immutable
contract can be supplied on another server through explicit read-only mounts.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import fcntl
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import torch
from .config import load_config
from .contracts import require
from .io import atomic_json, digest, file_hash, read_json
from .training.il_only import derive_contract, validate_contract
from .training.online_base import OnlineBasePolicy

REPO = Path(__file__).resolve().parents[1]


def inputs(path):
    path = Path(path).resolve()
    value = read_json(path)
    require(value.get("schema") == "query_base_online_inputs_v1", "unsupported Base inputs schema")
    require(value["content_hash"] == digest({k: v for k, v in value.items() if k != "content_hash"}), "inputs checksum mismatch")
    require(value["objective"] == "trajectory_il_only_v1", "this entrypoint trains trajectory IL only")
    contract = read_json(path.parent / "IL_ONLY_CONTRACT.json")
    require(file_hash(path.parent / "IL_ONLY_CONTRACT.json") == value["contract_checksum"], "IL-only contract mismatch")
    require(file_hash(path.parent / "SOURCE_CONFIG_IL_ONLY.json") == contract["source_config_hash"], "source config checksum mismatch")
    require(value["fit_scene_ids"] == sorted(set(value["fit_scene_ids"])) and value["fit_scene_ids"], "invalid fixed fit population")
    return value, contract


def configuration(bundle, contract):
    config = load_config(REPO / "configs/iqe/main.yaml")
    config["query_base"] = deepcopy(bundle["query_base"])
    config["data"] = deepcopy(bundle["data"])
    config["seed"] = bundle["seed"]
    config["s0"]["retain_existing_auxiliary_objectives"] = False
    config["s0"].update(source_root=contract["source_root"], source_commit=contract["source_commit"])
    config["base_runtime"].update(prefetch_depth=1, frozen_vision_ahead=True, deterministic_backward=True,
        allow_legacy_validation_metadata=True, allow_deterministic_backend_migration=True)
    config["resume"]["allow_fallback"] = False
    config["execution"]["loader_workers"] = 4
    return config


def prepare(args):
    contract = read_json(args.contract)
    validate_contract(contract, REPO)
    config = load_config(args.config)
    if args.scene_manifest:
        from .data.manifests import load_scenes
        from .data.splits import validate_isolation
        scenes = load_scenes(args.scene_manifest)
        validate_isolation(scenes)
        fit = [s for s in scenes if s.split_role == "incremental_fit"]
        require(all(s.source_kind == "original" and s.target_provenance == "gt" for s in fit), "Base uses original GT scenes")
        fit_ids = sorted(s.scene_id for s in fit)
        population = {"scene_manifest_hash": file_hash(args.scene_manifest)}
        data, cfg, seed, world = config["data"], config["query_base"], config["seed"], args.world_size
    else:
        # Optional export of only the established population/recipe. No weights
        # from this checkpoint initialize the new experiment.
        from .training.resume import inspect_checkpoint
        parent = inspect_checkpoint(args.checkpoint)
        state = torch.load(args.checkpoint, map_location="cpu", mmap=True, weights_only=False)
        require(state["kind"] == "iqe_query_base" and state["framework_contract_hash"] == contract["framework_contract_hash"], "checkpoint/contract mismatch")
        require(state["fit_scene_ids"] == sorted(set(state["fit_scene_ids"])) and
                digest(state["fit_scene_ids"]) == state["dependencies"]["fit_manifest"], "checkpoint fit population mismatch")
        require(state["mode"] == "full" and state["optimizer_step"] > 0, "use a real complete training checkpoint")
        fit_ids = state["fit_scene_ids"]
        population = {"recipe_source_checkpoint_hash": parent["checksum"], "driving_weights_imported": False}
        data, cfg = state["dependencies"]["original_targets"], state["dependencies"]["config"]
        seed, world = state["seed"], state["dependencies"]["world_size"]
    require(fit_ids and world > 0, "empty fit population or invalid world size")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "PARENT_CONTRACT.json", contract, immutable=True)
    derived = derive_contract(contract, output)
    data = deepcopy(data)
    # Neither the loader nor preflight reads any of these former auxiliary or
    # official scoring dependencies. They are not necessary on the new server.
    for key in data:
        if key != "train_root":
            data[key] = None
    cfg = deepcopy(cfg)
    cfg["validation_every_steps"] = 0
    bundle = {"schema": "query_base_online_inputs_v1", "objective": "trajectory_il_only_v1",
              "population_source": population, "parent_contract_checksum": file_hash(output / "PARENT_CONTRACT.json"),
              "contract_checksum": file_hash(output / "IL_ONLY_CONTRACT.json"),
              "seed": seed, "fit_scene_ids": fit_ids,
              "query_base": cfg, "world_size": world, "data": data,
              "framework_contract_hash": derived["framework_contract_hash"],
              "current_dataset_files": {name: file_hash(Path(data["train_root"]) / name)
                  for name in ("identity.json", "index.json", "ego_identity.json")}}
    bundle["content_hash"] = digest(bundle)
    atomic_json(output / "TRAINING_INPUTS.json", bundle, immutable=True)
    return {"status": "PREPARED", "inputs": str(output / "TRAINING_INPUTS.json"),
            "fit_scenes": len(bundle["fit_scene_ids"]), "losses": ["ego_il"], "training_started": False}


def check(args):
    bundle, contract = inputs(args.inputs)
    validate_contract(contract, REPO)
    for name, checksum in bundle["current_dataset_files"].items():
        require(file_hash(Path(bundle["data"]["train_root"]) / name) == checksum, "current dataset changed: " + name)
    from .training.base_trainer import original_dataset
    dataset = original_dataset(configuration(bundle, contract), contract)
    by_id = {r["token"]: i for i, r in enumerate(dataset.index)}
    require(all(token in by_id for token in bundle["fit_scene_ids"]), "fit scene missing in original dataset")
    rows = []
    for token in bundle["fit_scene_ids"][:args.samples]:
        observation, target = dataset[by_id[token]]
        require(set(target) == {"ego"}, "IL-only dataset unexpectedly read auxiliary labels")
        require(len(observation["image"]) == 3 and target["ego"].shape == (8, 4) and torch.isfinite(target["ego"]).all(), "invalid real IL training sample")
        rows.append({"scene_id": token, "image_sizes": [list(im.size) for im in observation["image"]],
                     "target_fields": sorted(target), "target_shape": list(target["ego"].shape)})
    import importlib.metadata
    versions = {name: importlib.metadata.version(name) for name in
                ("torch", "torchvision", "transformers", "diffusers", "timm", "omegaconf", "Pillow", "numpy")}
    return {"status": "PASS", "scope": "source_contract_and_real_online_loader", "versions": versions,
            "fit_scenes": len(bundle["fit_scene_ids"]), "real_samples": rows,
            "GPU_training_checked": False, "auxiliary_targets_required": False}


def train(args):
    require(args.from_generic != bool(args.resume), "choose --from-generic or --resume, never both")
    require(args.mode == "full" or (args.updates is not None and 1 <= args.updates <= 32), "bounded mode requires 1..32 updates")
    require(args.updates is None or args.updates > 0, "positive additional update budget required")
    bundle, contract = inputs(args.inputs)
    validate_contract(contract, REPO)
    # The host must explicitly appear in a user-supplied resource authorization.
    # No local lab hostname is embedded in this reusable entrypoint.
    from .resources import qualify_distributed
    qualify_distributed("cuda", args.resource_policy)
    from .training.runtime_identity import configure_runtime
    configure_runtime({"deterministic_backward": True})
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    torch.set_num_threads(2)
    from .training.trainer import init_distributed
    from .training.execution import collective_check
    rank, world = init_distributed("cuda")
    require(world == bundle["world_size"], "fixed training topology changed")
    config = configuration(bundle, contract)
    policy = OnlineBasePolicy(checkpoint_policy=args.checkpoint_policy)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    handle = None
    error = None
    if rank == 0:
        try:
            handle = (output / "run.lock").open("a+")
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            atomic_json(output / "RUN_INPUTS.json", {"inputs_hash": bundle["content_hash"], "contract_hash": contract["framework_contract_hash"]}, immutable=True)
            atomic_json(output / "resolved_config.json", {"config": config, "online_execution": policy.identity(contract, world)})
        except Exception as exc:
            error = repr(exc)
    collective_check(error, device=f"cuda:{os.environ.get('LOCAL_RANK', '0')}")
    try:
        stop = None
        if args.updates is not None:
            step = 0
            if args.resume:
                # The trainer resolves/checks the same receipt set. No filenames
                # without complete receipts are guessed as a resume point.
                from .training.resume import resolve_resume
                path = [None]
                error = None
                if rank == 0:
                    try:
                        path[0] = resolve_resume(output, args.resume)
                    except Exception as exc:
                        error = repr(exc)
                collective_check(error, device=f"cuda:{os.environ.get('LOCAL_RANK', '0')}")
                if world > 1:
                    torch.distributed.broadcast_object_list(path, src=0)
                args.resume = path[0]
                step = read_json(str(args.resume) + ".COMPLETE.json")["optimizer_step"]
            stop = step + args.updates
        scenes = [SimpleNamespace(scene_id=t, split_role="incremental_fit", source_kind="original", target_provenance="gt")
                  for t in bundle["fit_scene_ids"]]
        from .training.base_trainer import train_base
        result = train_base(config, contract, scenes, output, mode="full" if args.mode == "full" else "profile",
                            stop_after=stop, resume=args.resume, execution_policy=policy)
        return {"status": "RETURNED", "checkpoint": result, "output": str(output)}
    finally:
        if handle is not None:
            handle.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare", help="bind the fixed training population/recipe, not previous driving weights")
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", help="export only its fixed scene IDs and recipe; do not load driving weights")
    source.add_argument("--scene-manifest", help="verified IQE SceneRecord manifest; use incremental_fit only")
    p.add_argument("--config", default="configs/iqe/main.yaml")
    p.add_argument("--world-size", type=int, default=8)
    p.add_argument("--contract", default="reports/iqe/S0_CONTRACT.json")
    p.add_argument("--output", required=True)
    p.set_defaults(function=prepare)
    p = sub.add_parser("check", help="verify protected code and read actual images/ego labels without auxiliary data")
    p.add_argument("--inputs", required=True)
    p.add_argument("--samples", type=int, choices=range(1, 33), default=4)
    p.set_defaults(function=check)
    p = sub.add_parser("train", help="train directly from images; preserve the original full learning-rate horizon")
    p.add_argument("--inputs", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--resource-policy", required=True)
    p.add_argument("--mode", choices=["bounded", "full"], required=True)
    p.add_argument("--updates", type=int, help="additional updates, not a replacement cosine horizon")
    p.add_argument("--resume", help="complete checkpoint path or auto in this output directory")
    p.add_argument("--from-generic", action="store_true", help="explicit new training; never inferred from a missing checkpoint")
    p.add_argument("--checkpoint-policy", choices=["none", "all"], default="none")
    p.set_defaults(function=train)
    args = parser.parse_args()
    try:
        result = args.function(args)
        if int(os.environ.get("RANK", "0")) == 0:
            print(json.dumps(result, allow_nan=False), flush=True)
    except Exception as exc:
        print(json.dumps({"status": "FAILED", "command": args.command, "error": str(exc)}), file=sys.stderr, flush=True)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
