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
from .execution import EpochPlan, TrainingPrefetch, FrozenVisionAhead, collective_check, finite_gradients
from .runtime_identity import resume_dependencies, validate_runtime, configure_runtime


def original_dataset(config, contract, *, allow_partial=False):
    s = contract["source_config"]["foresight"]
    d = config["data"]
    if not any(s[k] for k in ("enable_current_dino", "enable_future_dino", "enable_interaction")):
        require(all(d[k] is None for k in ("dino_root", "dino_index", "interaction_root", "clip_root")),
                "IL-only Base must not retain auxiliary label paths")
        from starVLA.dataloader.foresight_dataset import ForesightTrainingDataset
        return ForesightTrainingDataset(d["train_root"])
    from starVLA.dataloader.action_video_foresight_dataset import ActionVideoForesightDataset
    return ActionVideoForesightDataset(d["train_root"], dino_root=d["dino_root"], dino_index=d["dino_index"],
        expected_dino=d["dino_identity"], candidate=s["candidate"], current=s["enable_current_dino"], future=s["enable_future_dino"],
        interaction_root=d["interaction_root"], expected_interaction=d["interaction_identity"],
        clip_root=d["clip_root"], expected_clip=d["clip_identity"], future_type=s["future_target_type"],
        expected_teacher=s["video_teacher_identity"], allow_partial=allow_partial)


def train_base(config, contract, scenes, output, *, mode, steps=None, stop_after=None, resume=None, device="cuda", execution_policy=None):
    require(mode in {"smoke", "full", "profile"}, "explicit base execution mode")
    if mode == "full":
        require(device.startswith("cuda"), "formal Query base training needs qualified GPU resources")
    configure_runtime(config["base_runtime"])
    rank, world = init_distributed(device)
    device = f"cuda:{os.environ.get('LOCAL_RANK', '0')}" if device.startswith("cuda") else device
    from omegaconf import OmegaConf
    cfg = dict(config["query_base"])
    require(cfg.get("validation_every_steps", 0) == 0, "inline Base validation unsupported; use evaluate-base with a fixed manifest, validation_every_steps must be0")
    if steps is not None:
        require(0 < steps <= cfg["max_optimizer_steps"], "base optimizer-step budget")
        cfg["max_optimizer_steps"] = steps
        cfg["warmup_steps"] = min(cfg["warmup_steps"], steps - 1)
    require(mode == "full" or cfg["max_optimizer_steps"] <= 32 or (mode == "profile" and stop_after is not None),
            "bounded smoke/profile requires <=32 updates or an explicit profile stop boundary")
    require(cfg["global_batch_size"] % world == 0, "base global batch/world size")
    seed = config["seed"]
    runtime = config["base_runtime"]
    runtime_identity = (execution_policy.identity(contract, world) if execution_policy else
                        validate_runtime(runtime, contract, world, mode=mode, training_config=cfg))
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
    resume_info = None
    if resume:
        # Rank0 performs the expensive checksum scan once, not eight times.
        error = None
        if rank == 0:
            try:
                from .resume import resolve_resume
                resume = resolve_resume(output, resume, expected_dependencies=dependencies,
                    allow_fallback=config["resume"]["allow_fallback"], dependency_validator=lambda old, new:
                    resume_dependencies(old, new, allow_legacy_validation_metadata=runtime["allow_legacy_validation_metadata"],
                    allow_deterministic_backend_migration=runtime.get("allow_deterministic_backend_migration",False)))
                receipt = read_json(str(resume) + ".COMPLETE.json")
                migration = resume_dependencies(receipt["dependencies"], dependencies,
                    allow_legacy_validation_metadata=runtime["allow_legacy_validation_metadata"],
                    allow_deterministic_backend_migration=runtime.get("allow_deterministic_backend_migration",False))
                from .local_checkpoint import stage_checkpoint
                require(runtime["local_checkpoint_root"] is None or world == int(os.environ.get("LOCAL_WORLD_SIZE", "1")),
                        "local checkpoint staging currently requires a single training host")
                staged = stage_checkpoint(resume, receipt["checksum"], runtime["local_checkpoint_root"])
                resume_info = {"path": str(resume), "load_path": str(staged), "receipt": receipt, "migration": migration}
            except Exception as exception:
                error = repr(exception)
        collective_check(error, device=device)
        if world > 1:
            payload = [resume_info]; dist.broadcast_object_list(payload, src=0); resume_info = payload[0]
        resume = resume_info["path"]
    if mode == "full":
        from ..storage import capacity_plan, require_capacity
        error = None
        if rank == 0:
            try:
                sizes = [p.stat().st_size for p in output.glob("step_*.pt")]
                if resume:
                    sizes.append(Path(resume).stat().st_size)
                generic = Path(contract["generic_vlm"])
                measured = max(sizes) if sizes else 4 * sum(p.stat().st_size for p in generic.glob("*.safetensors"))
                capacity = capacity_plan(output, measured, config["execution"]["checkpoint_every_steps"],
                    cfg["max_optimizer_steps"], resume_info["receipt"]["optimizer_step"] if resume_info else 0)
                atomic_json(output / "CAPACITY_PLAN.json", capacity)
                require_capacity(capacity)
            except Exception as exception:
                error = repr(exception)
        collective_check(error, device=device)
    if rank == 0:
        atomic_json(output / "RUNTIME_IDENTITY.json", runtime_identity)
        if resume_info:
            atomic_json(output / "EXECUTION_MIGRATION.json", {**resume_info, "runtime": runtime_identity,
                "source_checkpoint_rewritten": False})
    random.seed(seed + rank); np.random.seed(seed + rank); torch.manual_seed(seed)
    model = build_query_framework(OmegaConf.create(contract["source_config"]), contract["query_architecture"],
                                  contract["source_root"], contract["source_commit"]).float().to(device)
    if world > 1:
        configure_language_checkpointing(model)
    # The original model freezes its visual tower and preserves its mode.
    model.train()
    if execution_policy:
        execution_policy.configure_model(model)
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
    optimizer = (execution_policy.make_optimizer(model, cfg, contract) if execution_policy else
                 torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg["weight_decay"],
                                   betas=tuple(contract["source_config"]["trainer"]["optimizer"]["betas"])))
    scheduler = scheduler_factory(optimizer, cfg)
    ddp_options = ({"gradient_as_bucket_view": execution_policy.gradient_as_bucket_view,
                    "bucket_cap_mb": execution_policy.bucket_cap_mb} if execution_policy else {})
    wrapped = DistributedDataParallel(model, device_ids=[int(os.environ.get("LOCAL_RANK", "0"))] if device.startswith("cuda") else None,
                                       find_unused_parameters=True, **ddp_options) if world > 1 else model
    step = epoch = offset = 0
    from .online_base import checkpoint_origin
    provenance = checkpoint_origin(None, None, None)
    execution_migration = None
    if resume:
        state = torch.load(resume_info["load_path"], map_location="cpu", weights_only=False, mmap=True)
        resume_dependencies(state["dependencies"], dependencies,
            allow_legacy_validation_metadata=runtime["allow_legacy_validation_metadata"],
                    allow_deterministic_backend_migration=runtime.get("allow_deterministic_backend_migration",False))
        require(state["optimizer_step"] == resume_info["receipt"]["optimizer_step"], "Base receipt/step mismatch")
        require(state["seed"] == seed and state["fit_scene_ids"] == fit,
                "exact Base resume seed or ordered scene population changed")
        if execution_policy:
            execution_migration = execution_policy.validate_state(state, runtime_identity)
        elif "runtime_identity" in state:
            require(state["runtime_identity"]["backend_flags"] == runtime_identity["backend_flags"],
                    "exact Base resume numerical backend flags changed")
        model.load_state_dict(state["model"], strict=True)
        if execution_policy:
            execution_policy.load_optimizer(optimizer, state["optimizer"])
        else:
            optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        step, epoch, offset = state["optimizer_step"], state["epoch"], state["offset"]
        provenance = checkpoint_origin(state, resume, resume_info["receipt"]["checksum"])
        restore_rng(state["rng_by_rank"][rank])
        del state
        if execution_policy and rank == 0:
            atomic_json(output / "ONLINE_EXECUTION_MIGRATION.json", execution_migration)
    if step == cfg["max_optimizer_steps"]:
        if rank == 0:
            atomic_json(output / "result.json", {"status": "COMPLETE", "optimizer_steps": step,
                "checkpoint": resume, "new_optimizer_steps": 0, "mode": mode, "termination_reason": "MAX_STEPS"})
        if world > 1:
            dist.barrier()
        return resume
    require(stop_after is None or step < stop_after <= cfg["max_optimizer_steps"], "stop_after must exceed restored step and fit schedule")
    if mode != "full":
        require(0 < (stop_after or cfg["max_optimizer_steps"]) - step <= 32,
                "bounded smoke/profile may execute at most32 additional updates; schedule stays unchanged")
    vision = FrozenVisionAhead(model) if runtime["frozen_vision_ahead"] else None
    start_step = step; start = time.perf_counter(); stop = [False]
    launch_id = f"{os.getpid()}-{time.time_ns()}"
    old_handler = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__(0, True))
    def checkpoint():
        rngs = [None] * world
        if world > 1:
            dist.all_gather_object(rngs, capture_rng())
        else:
            rngs[0] = capture_rng()
        error = None
        if rank == 0:
            try:
                path = output / f"step_{step:06d}.pt"
                require(not path.exists(), "refusing to overwrite an existing Base checkpoint")
                atomic_torch(path, {"kind": "iqe_query_base", "framework_contract_hash": contract["framework_contract_hash"],
                    "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                    "optimizer_step": step, "epoch": epoch, "offset": offset, "dependencies": dependencies,
                    "rng_by_rank": rngs, "resume_boundary": "optimizer_step", **provenance,
                    "fit_scene_ids": fit, "seed":seed, "mode":mode, "runtime_identity": runtime_identity,
                    **({"online_base_identity": runtime_identity} if execution_policy else {}),
                    "terminal": step == cfg["max_optimizer_steps"],
                    "termination_reason": "MAX_STEPS" if step == cfg["max_optimizer_steps"] else
                        ("OPERATOR_PAUSED" if stop[0] or step == stop_after else "RUNNING_CHECKPOINT")})
                atomic_json(str(path) + ".COMPLETE.json", {"checksum": file_hash(path), "optimizer_step": step, "dependencies": dependencies})
            except Exception as exception:
                error = repr(exception)
        collective_check(error, device=device)
    checkpoint_seconds = 0.
    last_boundary = time.perf_counter()
    try:
        with ThreadPoolExecutor(max_workers=config["execution"]["loader_workers"] or 1) as workers:
            plan = EpochPlan(len(fit), cfg["global_batch_size"], seed, rank, world, epoch_batches)
            def prepare(address):
                began = time.perf_counter()
                samples = list(workers.map(dataset.__getitem__, [by_id[fit[i]] for i in address.mine]))
                observations, targets = collate_training(samples)
                sample_valid = torch.arange(len(address.mine)) < address.valid_count
                require(bool(torch.isfinite(targets["ego"][sample_valid]).all()), "nonfinite valid Base GT before forward")
                for k, v in targets.items():
                    if k.endswith("_valid") or k == "future_valid":
                        v[~sample_valid] = False
                targets["iqe_sample_valid"] = sample_valid
                data_seconds = time.perf_counter() - began
                return observations, targets, None, data_seconds
            def prepare_vision(batch):
                observations, targets, _, data_seconds = batch
                return observations, targets, vision.prepare(observations, cfg["micro_batch_size"]), data_seconds
            remaining = min(cfg["max_optimizer_steps"], stop_after or cfg["max_optimizer_steps"]) - step
            with TrainingPrefetch(plan.iter_from(epoch, offset, remaining), remaining, prepare,
                                  prepare_vision if vision else None, runtime["prefetch_depth"]) as ahead:
                while step < cfg["max_optimizer_steps"]:
                    update_start = time.perf_counter()
                    error = None
                    try:
                        address, batch = ahead.pop()
                    except Exception as e:
                        error = repr(e)
                    collective_check(error, device=device)
                    observations, targets, prepared, data_seconds = batch
                    entries, mine = address.entries, address.mine
                    if resume and step == start_step and rank == 0:
                        from .resume import record_next_batch
                        record_next_batch(output, [fit[i] for i in entries],
                                          exact=not execution_migration or execution_migration["exact_resume"])
                    targets["iqe_sample_valid"] = targets["iqe_sample_valid"].to(device)
                    data_wait = time.perf_counter() - update_start
                    generator = torch.Generator().manual_seed(seed + step * world + rank)
                    counts = optimizer_batch_counts(targets, generator, device)
                    counts["ego_scenes"] = len(entries)
                    optimizer.zero_grad(set_to_none=True)
                    logs = {}
                    with vision.activate(prepared) if vision else nullcontext():
                        for at in range(0, len(mine), cfg["micro_batch_size"]):
                            end = min(len(mine), at + cfg["micro_batch_size"])
                            sync = end == len(mine)
                            with wrapped.no_sync() if world > 1 and not sync else nullcontext():
                                result = wrapped(observations[at:end], {k: v[at:end] for k, v in targets.items()},
                                                 completed_updates=step, global_counts=counts)
                                loss = result["loss"]
                                collective_check(None if bool(torch.isfinite(loss)) else "nonfinite Query base loss", device=device)
                                loss.backward()
                            for k, v in result["losses"].items():
                                logs[k] = logs.get(k, 0.) + float(v.detach())
                    good_gradients = finite_gradients(params) if runtime["batched_gradient_checks"] else not any(
                        p.grad is not None and not torch.isfinite(p.grad).all() for p in params)
                    collective_check(None if good_gradients else "nonfinite base gradient", device=device)
                    norm = torch.nn.utils.clip_grad_norm_(params, cfg["grad_clip_norm"])
                    optimizer.step(); scheduler.step(); step += 1
                    # Only consumed addresses advance the saved checkpoint cursor.
                    epoch, offset = address.epoch, address.offset + 1
                    stopping = torch.tensor(int(stop[0] or step == stop_after), device=device)
                    if world > 1:
                        dist.all_reduce(stopping, op=dist.ReduceOp.MAX)
                    stop_now = bool(stopping)
                    now = time.perf_counter()
                    compute_seconds = now - update_start
                    if rank == 0:
                        row = {"optimizer_step": step, "losses": logs, "gradient_norm": float(norm), "actual_global_batch": len(entries),
                               "sample_ids": [fit[i] for i in entries], "seconds_per_update": (now - start) / (step - start_step),
                               "seconds_per_update_semantics": "cumulative_since_launch_including_previous_saves",
                               "compute_update_seconds": compute_seconds, "step_wall_seconds": now - last_boundary,
                               "data_wait_seconds": data_wait, "data_load_seconds": data_seconds,
                               "previous_checkpoint_seconds": checkpoint_seconds,
                               "runtime_identity_hash": runtime_identity["identity_hash"],
                               "launch_id": launch_id,
                               "retained_objectives": sorted(logs),
                               "learned_driving_weights_loaded": provenance["learned_driving_weights_loaded"], "status": "RUNNING"}
                        atomic_json(output / "status.json", row)
                        import json
                        with open(output / "steps.jsonl", "a") as f:
                            f.write(json.dumps(row, allow_nan=False) + "\n")
                    last_boundary = now
                    checkpoint_seconds = 0.
                    if step % config["execution"]["checkpoint_every_steps"] == 0 or step == cfg["max_optimizer_steps"] or stop_now:
                        saving = time.perf_counter()
                        checkpoint()
                        checkpoint_seconds = time.perf_counter() - saving
                    if stop_now:
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
        if vision:
            vision.close()
        signal.signal(signal.SIGTERM, old_handler)
