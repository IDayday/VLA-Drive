"""Train the Query-migrated base using the actual original S0 dataset and objectives.

FP32 parameters/AdamW states, BF16 autocast, original gradient checkpointing,
fixed global batch, true global auxiliary denominators and optimizer-boundary
resume. No learned driving weight is loaded. Full execution requires free GPUs.
"""
from __future__ import annotations
from contextlib import nullcontext
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import os
import random
import signal
import time
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from ..contracts import require
from ..io import atomic_json, atomic_torch, read_json, digest, file_hash
from ..query_base import build_query_framework
from .trainer import init_distributed, scheduler_factory
from .checkpoint import capture_rng, restore_rng, environment, configure_language_checkpointing
from ..losses import collective_error


def original_dataset(config, contract, *, allow_partial=False):
    from starVLA.dataloader.action_video_foresight_dataset import ActionVideoForesightDataset
    s = contract["source_config"]["foresight"]
    d = config["data"]
    return ActionVideoForesightDataset(d["train_root"], dino_root=d["dino_root"], dino_index=d["dino_index"],
        expected_dino=d["dino_identity"], candidate=s["candidate"], current=s["enable_current_dino"], future=s["enable_future_dino"],
        interaction_root=d["interaction_root"], expected_interaction=d["interaction_identity"],
        clip_root=d["clip_root"], expected_clip=d["clip_identity"], future_type=s["future_target_type"],
        expected_teacher=s["video_teacher_identity"], allow_partial=allow_partial)


def train_base(config, contract, scenes, output, *, mode, steps=None, stop_after=None, resume=None, device="cuda"):
    require(mode in {"smoke", "full", "profile"}, "explicit base execution mode")
    if mode == "full":
        require(device.startswith("cuda"), "formal Query base training needs qualified GPU resources")
    rank, world = init_distributed(device)
    device = f"cuda:{os.environ.get('LOCAL_RANK', '0')}" if device.startswith("cuda") else device
    from omegaconf import OmegaConf
    cfg = dict(config["query_base"])
    if steps is not None:
        require(0 < steps <= cfg["max_optimizer_steps"], "base optimizer-step budget")
        cfg["max_optimizer_steps"] = steps
        cfg["warmup_steps"] = min(cfg["warmup_steps"], steps - 1)
    require(mode == "full" or cfg["max_optimizer_steps"] <= 32, "bounded smoke/profile base steps <=32")
    require(cfg["global_batch_size"] % world == 0, "base global batch/world size")
    seed = config["seed"]
    fit = sorted(s.scene_id for s in scenes if s.split_role == "incremental_fit")
    dependencies = {"framework_contract_hash": contract["framework_contract_hash"], "fit_manifest": digest(fit),
                    "original_targets": config["data"], "world_size": world, "config": cfg, "environment": environment()}
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    finished = read_json(output / "result.json") if (output / "result.json").exists() else None
    if resume is None and finished is not None and finished["status"] == "COMPLETE":
        receipt = read_json(finished["checkpoint"] + ".COMPLETE.json")
        require(receipt["dependencies"] == dependencies and finished["mode"] == mode and
                file_hash(finished["checkpoint"]) == receipt["checksum"], "completed base dependencies changed")
        return finished["checkpoint"]
    require(resume is not None or (finished is None and not list(output.glob("step_*.pt.COMPLETE.json"))),
            "existing unfinished Query base requires explicit resume")
    random.seed(seed + rank); np.random.seed(seed + rank); torch.manual_seed(seed)
    model = build_query_framework(OmegaConf.create(contract["source_config"]), contract["query_architecture"],
                                  contract["source_root"], contract["source_commit"]).float().to(device)
    if world > 1:
        configure_language_checkpointing(model)
    # The original model freezes its visual tower and preserves its mode.
    model.train()
    dataset = original_dataset(config, contract, allow_partial=mode != "full")
    by_id = {x["token"]: i for i, x in enumerate(dataset.index)}
    fit = sorted(s.scene_id for s in scenes if s.split_role == "incremental_fit")
    require(fit and all(t in by_id for t in fit), "Query base fit records must join original training data")
    fit_ids = set(fit)
    require(all(s.source_kind == "original" and s.target_provenance == "gt" for s in scenes if s.scene_id in fit_ids), "base trains only original GT")
    from tools.ddpolicy_vehicle.training_state import epoch_batches
    from tools.foresight.student_state import optimizer_batch_counts
    from starVLA.dataloader.foresight_dataset import collate_training
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg["weight_decay"],
                                 betas=tuple(contract["source_config"]["trainer"]["optimizer"]["betas"]))
    scheduler = scheduler_factory(optimizer, cfg)
    wrapped = DistributedDataParallel(model, device_ids=[int(os.environ.get("LOCAL_RANK", "0"))] if device.startswith("cuda") else None,
                                       find_unused_parameters=True) if world > 1 else model
    step = epoch = offset = 0
    if resume:
        require(file_hash(resume) == read_json(str(resume) + ".COMPLETE.json")["checksum"], "base checkpoint partial/corrupt")
        state = torch.load(resume, map_location="cpu", weights_only=False)
        require(state["dependencies"] == dependencies, "exact base resume config/data/world/environment changed")
        model.load_state_dict(state["model"], strict=True); optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        step, epoch, offset = state["optimizer_step"], state["epoch"], state["offset"]
        restore_rng(state["rng_by_rank"][rank])
    start_step = step; start = time.perf_counter(); stop = [False]
    old_handler = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__(0, True))
    def checkpoint():
        rngs = [None] * world
        if world > 1:
            dist.all_gather_object(rngs, capture_rng())
        else:
            rngs[0] = capture_rng()
        if rank == 0:
            path = output / f"step_{step:06d}.pt"
            atomic_torch(path, {"kind": "iqe_query_base", "framework_contract_hash": contract["framework_contract_hash"],
                "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "optimizer_step": step, "epoch": epoch, "offset": offset, "dependencies": dependencies,
                "rng_by_rank": rngs, "resume_boundary": "optimizer_step", "learned_driving_weights_loaded": False,
                "fit_scene_ids": fit, "seed":seed, "mode":mode})
            atomic_json(str(path) + ".COMPLETE.json", {"checksum": file_hash(path), "optimizer_step": step, "dependencies": dependencies})
        if world > 1:
            dist.barrier()
    try:
        with ThreadPoolExecutor(max_workers=config["execution"]["loader_workers"] or 1) as workers:
            while step < cfg["max_optimizer_steps"]:
                update_start = time.perf_counter()
                batches = epoch_batches(len(fit), cfg["global_batch_size"], seed, epoch)
                if offset == len(batches):
                    epoch += 1; offset = 0; continue
                entries = batches[offset]
                mine = entries[rank::world]
                valid_count = len(mine)
                while len(mine) < int(np.ceil(len(entries) / world)):
                    mine.append(entries[0])
                calls = int(np.ceil(len(mine) / cfg["micro_batch_size"]))
                calls_by_rank = [None] * world
                if world > 1:
                    dist.all_gather_object(calls_by_rank, calls)
                    require(len(set(calls_by_rank)) == 1, "base rank microbatch count differs")
                error = None
                try:
                    samples = list(workers.map(dataset.__getitem__, [by_id[fit[i]] for i in mine]))
                    observations, targets = collate_training(samples)
                    sample_valid = torch.arange(len(mine)) < valid_count
                    for k, v in targets.items():
                        if k.endswith("_valid") or k == "future_valid":
                            v[~sample_valid] = False
                    targets["iqe_sample_valid"] = sample_valid.to(device)
                except Exception as e:
                    error = repr(e)
                collective_error(error)
                generator = torch.Generator().manual_seed(seed + step * world + rank)
                counts = optimizer_batch_counts(targets, generator, device)
                counts["ego_scenes"] = len(entries)
                optimizer.zero_grad(set_to_none=True)
                logs = {}
                for at in range(0, len(mine), cfg["micro_batch_size"]):
                    end = min(len(mine), at + cfg["micro_batch_size"])
                    sync = end == len(mine)
                    with wrapped.no_sync() if world > 1 and not sync else nullcontext():
                        result = wrapped(observations[at:end], {k: v[at:end] for k, v in targets.items()},
                                         completed_updates=step, global_counts=counts)
                        loss = result["loss"]
                        collective_error(None if torch.isfinite(loss) else "nonfinite Query base loss")
                        loss.backward()
                    for k, v in result["losses"].items():
                        logs[k] = logs.get(k, 0.) + float(v.detach())
                collective_error("nonfinite base gradient" if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in params) else None)
                norm = torch.nn.utils.clip_grad_norm_(params, cfg["grad_clip_norm"])
                optimizer.step(); scheduler.step(); step += 1; offset += 1
                stopping = torch.tensor(int(stop[0] or step == stop_after), device=device)
                if world > 1:
                    dist.all_reduce(stopping, op=dist.ReduceOp.MAX)
                if rank == 0:
                    row = {"optimizer_step": step, "losses": logs, "gradient_norm": float(norm), "actual_global_batch": len(entries),
                           "sample_ids": [fit[i] for i in entries], "seconds_per_update": (time.perf_counter() - start) / (step - start_step),
                           "compute_update_seconds":time.perf_counter()-update_start,
                           "retained_objectives": ["ego_il", "current_dino", "future_clip", "interaction"],
                           "learned_driving_weights_loaded": False, "status": "RUNNING"}
                    atomic_json(output / "status.json", row)
                    import json
                    with open(output / "steps.jsonl", "a") as f:
                        f.write(json.dumps(row) + "\n")
                if step % config["execution"]["checkpoint_every_steps"] == 0 or step == cfg["max_optimizer_steps"] or stopping:
                    checkpoint()
                if stopping:
                    break
        if rank == 0:
            atomic_json(output / "result.json", {"status": "COMPLETE" if step == cfg["max_optimizer_steps"] else "STOPPED_AT_BOUNDARY",
                "optimizer_steps": step, "checkpoint": str(output / f"step_{step:06d}.pt"),
                "mode": mode, "science": "UNTESTED", "new_optimizer_steps": step - start_step,
                "GPU_seconds": (time.perf_counter() - start) * world if device.startswith("cuda") else 0})
        if world > 1:
            dist.barrier()
        return str(output / f"step_{step:06d}.pt")
    finally:
        signal.signal(signal.SIGTERM, old_handler)
