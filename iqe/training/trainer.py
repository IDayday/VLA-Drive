"""Reusable bounded trainer for private experts, Scorer and Router.

Frozen cached features eliminate VLM recomputation. Full-batch denominators are
computed before per-microbatch backward; no prefetch cursor is used for resume.
Validation callbacks execute outside the optimizer step, never in batch loss.
"""
from __future__ import annotations
from contextlib import nullcontext
from dataclasses import dataclass
import math
import os
from pathlib import Path
import random
import signal
import time
from datetime import timedelta
import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from ..contracts import require
from ..io import atomic_json, atomic_torch, digest, read_json, file_hash
from ..losses import LossTerm, reduce_terms, collective_error
from .checkpoint import save_checkpoint, load_checkpoint
from ..evaluation.retention import module_hash_parameters
from ..contracts import FeatureBundle


def slice_batch(value, start, end):
    if isinstance(value, FeatureBundle):
        return value.subset(torch.arange(start, end, device=value.scene.device))
    if isinstance(value, torch.Tensor):
        return value[start:end]
    if isinstance(value, dict):
        return {k: slice_batch(v, start, end) for k, v in value.items()}
    if isinstance(value, list):
        return value[start:end]
    raise TypeError(f"unsupported microbatch value: {type(value)}")


def scheduler_factory(optimizer, config):
    steps, warmup, minimum = config["max_optimizer_steps"], config["warmup_steps"], config["lr_min_ratio"]
    def factor(step):
        if step < warmup:
            return (step + 1) / max(1, warmup)
        return minimum + (1 - minimum) * .5 * (1 + math.cos(math.pi * min(1, (step - warmup) / max(1, steps - warmup))))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def init_distributed(device):
    if int(os.environ.get("WORLD_SIZE", "1")) > 1 and not dist.is_initialized():
        if device.startswith("cuda"):
            torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        # Rank0 performs bounded offline official stage_val evaluation between
        # updates. Its duration is unrelated to CUDA collective throughput.
        dist.init_process_group("nccl" if device.startswith("cuda") else "gloo", timeout=timedelta(hours=6))
    return dist.get_rank() if dist.is_initialized() else 0, dist.get_world_size() if dist.is_initialized() else 1


class TrainingStage(nn.Module):
    def __init__(self, module, loss_function):
        super().__init__()
        self.module = module
        self.loss_function = loss_function

    def forward(self, batch):
        return self.loss_function(self.module, batch)


def train(module, loss_function, fetch, sampler, config, dependencies, output, *, device="cpu", weights=None,
          resume=None, non_exact_finetune=False, validate=None, stop_after=None, audit=None, seed=42, denominator_function=None):
    rank, world = init_distributed(device)
    device = f"cuda:{int(os.environ.get('LOCAL_RANK', '0'))}" if device.startswith("cuda") else device
    global_batch = config.get("global_batch_size", config.get("global_scene_batch_size"))
    micro = config["micro_batch_size"]
    require(global_batch > 0 and global_batch % world == 0, "global batch must divide world size")
    local_batch = global_batch // world
    accumulation = math.ceil(local_batch / micro)
    require(accumulation == 1 or denominator_function is not None, "microbatch accumulation requires explicit full-step label denominators")
    require(stop_after is None or 0 < stop_after <= config["max_optimizer_steps"], "stop_after budget")
    random.seed(seed + rank); np.random.seed(seed + rank); torch.manual_seed(seed + rank)
    module.to(device)
    params = [p for p in module.parameters() if p.requires_grad]
    require(params, "empty trainable stage")
    optimizer = torch.optim.AdamW(params, lr=config["lr"], weight_decay=config["weight_decay"])
    scheduler = scheduler_factory(optimizer, config)
    stage = TrainingStage(module, loss_function)
    wrapped = DistributedDataParallel(stage, device_ids=[int(os.environ.get("LOCAL_RANK", "0"))] if device.startswith("cuda") else None,
                                      find_unused_parameters=config.get("find_unused_parameters", False)) if world > 1 else stage
    start_step, consumed, restored_trainer = 0, [], {}
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    finished = read_json(output / "result.json") if (output / "result.json").exists() else None
    reuse_complete = resume is None and finished is not None and finished["status"] == "COMPLETE"
    if reuse_complete:
        resume = finished["checkpoint"]
    require(resume is not None or (finished is None and not list(output.glob("step_*.pt.COMPLETE.json"))),
            "existing unfinished training requires explicit resume; refusing restart/overwrite")
    if resume:
        restored = load_checkpoint(resume, module, optimizer, scheduler, None, sampler, dependencies, config,
                                   global_batch=global_batch, accumulation=accumulation, non_exact_finetune=non_exact_finetune)
        start_step, consumed = restored["step"], restored["consumed_ids"]
        restored_trainer = restored.get("trainer_state", {})
    completed_resume = (not non_exact_finetune and finished is not None and finished["status"] == "COMPLETE"
                        and resume is not None and Path(resume).resolve() == Path(finished["checkpoint"]).resolve()
                        and start_step == finished["optimizer_steps"])
    if reuse_complete or completed_resume:
        require(file_hash(output / "best.pt") == finished["best_checkpoint_hash"], "completed best checkpoint changed")
        return []
    before = module_hash_parameters(module)
    def amp():
        return torch.autocast("cuda", dtype=torch.bfloat16) if device.startswith("cuda") and config.get("precision") == "fp32_master_bf16_autocast" else nullcontext()
    stop = [False]
    previous_handler = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__(0, True))
    start = time.perf_counter(); best = restored_trainer.get("best"); bad_cycles = restored_trainer.get("bad_cycles", 0); history = []
    best = tuple(best) if best is not None else None
    step = start_step
    try:
        while step < config["max_optimizer_steps"]:
            update_start = time.perf_counter()
            entries = sampler.peek(global_batch)
            require(entries, "sampling plan exhausted before optimizer budget")
            # True tail counts, including empty ranks, are handled by fetch's padded/masked sentinel.
            mine = entries[rank::world]
            target_local_count = math.ceil(len(entries) / world)
            while len(mine) < target_local_count:
                mine.append(dict(entries[0], _padding=True))
            batch = None; error = None
            try:
                batch = fetch(mine, device)
            except Exception as e:
                error = repr(e)
            collective_error(error)
            module.train()
            if config.get("diagnostic_disable_dropout"):
                for m in module.modules():
                    if isinstance(m, nn.Dropout) or m.__class__.__name__ == "DropPath":
                        m.eval()
            if audit:
                audit(optimizer)
            optimizer.zero_grad(set_to_none=True)
            logs = {}; loss_value = 0.0
            if denominator_function is None:
                with amp():
                    terms = wrapped(batch)
                loss, logs = reduce_terms(terms, weights)
                collective_error(None if bool(torch.isfinite(loss)) else "nonfinite training loss")
                loss.backward(); loss_value = sum(v["mean"] * (weights or {}).get(k, 1.) for k, v in logs.items())
            else:
                denominators = denominator_function(batch)
                global_denominators = {k: v.detach().float().clone() for k, v in denominators.items()}
                if world > 1:
                    for v in global_denominators.values():
                        dist.all_reduce(v)
                sums = {k: torch.zeros((), device=device) for k in denominators}
                counts = {k: torch.zeros((), device=device) for k in denominators}
                for start_index in range(0, len(mine), micro):
                    end_index = min(len(mine), start_index + micro)
                    context = wrapped.no_sync() if world > 1 and end_index < len(mine) else nullcontext()
                    with context:
                        with amp():
                            terms = wrapped(slice_batch(batch, start_index, end_index))
                        require(set(terms) == set(denominators), "microbatch loss schema changed")
                        loss = sum(term.numerator * world / global_denominators[k].clamp_min(1) * (weights or {}).get(k, 1.) for k, term in terms.items())
                        collective_error(None if bool(torch.isfinite(loss)) else "nonfinite training loss")
                        loss.backward()
                    for k, term in terms.items():
                        sums[k] += term.numerator.detach().float()
                        counts[k] += term.denominator.detach().float()
                for k in sums:
                    require(bool(torch.allclose(counts[k], denominators[k].float())), f"label denominator changed in forward: {k}")
                    if world > 1:
                        dist.all_reduce(sums[k])
                    logs[k] = {"numerator": float(sums[k]), "denominator": float(global_denominators[k]), "mean": float(sums[k] / global_denominators[k].clamp_min(1))}
                    loss_value += logs[k]["mean"] * (weights or {}).get(k, 1.)
            bad = any(p.grad is not None and not torch.isfinite(p.grad).all() for p in params)
            collective_error("nonfinite gradient" if bad else None)
            norm = torch.nn.utils.clip_grad_norm_(params, config["grad_clip_norm"])
            gradient_evidence = {name: {"has_grad": p.grad is not None, "finite": bool(torch.isfinite(p.grad).all()) if p.grad is not None else True,
                                      "norm": p.grad.detach().float().norm().item() if p.grad is not None else None}
                                 for name, p in module.named_parameters() if p.requires_grad and (step == start_step or stop_after == step + 1)}
            optimizer.step(); scheduler.step(); sampler.consume(len(entries)); step += 1
            consumed.extend(e["scene_id"] for e in entries)
            row = {"optimizer_step": step, "losses": logs, "loss": loss_value, "gradient_norm": float(norm),
                   "sample_ids": [e["scene_id"] for e in entries], "global_batch_actual": len(entries),
                   "elapsed_seconds": time.perf_counter() - start, "compute_update_seconds":time.perf_counter()-update_start, "gradients": gradient_evidence}
            history.append(row)
            if rank == 0:
                atomic_json(output / "status.json", {"status": "RUNNING", "step": step, "global_batch": global_batch,
                    "seconds_per_update": (time.perf_counter() - start) / max(1, step - start_step), "world_size": world})
                with open(output / "steps.jsonl", "a") as f:
                    import json
                    f.write(json.dumps(row, allow_nan=False) + "\n")
            if validate and (step % config["validation_every_steps"] == 0 or step == config["max_optimizer_steps"]):
                module.eval()
                validation, error = None, None
                if rank == 0:
                    try:
                        with torch.no_grad():
                            validation = validate(module, step)
                    except Exception as e:
                        error = repr(e)
                collective_error(error)
                if world > 1:
                    payload = [validation]; dist.broadcast_object_list(payload, src=0); validation = payload[0]
                key = tuple(validation["selection_key"])
                if rank == 0:
                    atomic_json(output / f"validation_{step:06d}.json", validation, immutable=True)
                if best is None or key > best:
                    best, bad_cycles = key, 0
                    if rank == 0:
                        atomic_torch(output / "best.pt", module.state_dict())
                else:
                    bad_cycles += 1
                if bad_cycles >= config.get("early_stop_patience", 10**9):
                    stop[0] = True
                module.train()
            stopping = torch.tensor(int(stop[0] or step == stop_after), device=device)
            if world > 1:
                dist.all_reduce(stopping, op=dist.ReduceOp.MAX)
            stop[0] = bool(stopping)
            if step % config.get("checkpoint_every_steps", config["validation_every_steps"]) == 0 or step == config["max_optimizer_steps"] or stop[0] or step == stop_after:
                save_checkpoint(output / f"step_{step:06d}.pt", module, optimizer, scheduler, None, sampler, step,
                                dependencies, config, global_batch=global_batch, accumulation=accumulation, consumed_ids=consumed,
                                trainer_state={"best": best, "bad_cycles": bad_cycles})
            if stop[0]:
                break
        require(step > start_step or start_step == config["max_optimizer_steps"], "trainer made no progress")
        after = module_hash_parameters(module)
        if rank == 0:
            if best is None:
                atomic_torch(output / "best.pt", module.state_dict())
            atomic_json(output / "result.json", {"status": "COMPLETE" if step == config["max_optimizer_steps"] or bad_cycles >= config.get("early_stop_patience", 10**9) else "STOPPED_AT_BOUNDARY",
                       "optimizer_steps": step, "new_optimizer_steps": step - start_step, "parameters_changed": before != after,
                       "parameter_hash_before": before, "parameter_hash_after": after,
                       "elapsed_seconds": time.perf_counter() - start, "GPU_seconds": (time.perf_counter() - start) * world if device.startswith("cuda") else 0,
                       "actual_global_batch": global_batch, "resume_boundary": "optimizer_step",
                       "checkpoint": str(output / f"step_{step:06d}.pt"), "best_selection_key": best,
                       "best_checkpoint_hash":file_hash(output / "best.pt")})
        if world > 1:
            dist.barrier()
        return history
    finally:
        signal.signal(signal.SIGTERM, previous_handler)
