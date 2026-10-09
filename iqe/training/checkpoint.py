"""Exact optimizer-boundary resume; active-module states and strict frozen dependencies."""
from __future__ import annotations
import os
import platform
import random
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from ..contracts import require
from ..io import atomic_torch, atomic_json, file_hash, read_json


def configure_language_checkpointing(framework):
    """Keep activation recomputation with DDP-compatible non-reentrant autograd.

    Only the already-checkpointed language stack changes its recomputation
    implementation. Forward values, losses and RNG preservation are unchanged;
    the frozen visual tower and input sequence retain their original execution.
    """
    language = framework.qwen_vl_interface.model.model.language_model
    if language.is_gradient_checkpointing:
        language.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})


def capture_rng():
    return {"python": random.getstate(), "numpy": np.random.get_state(), "cpu": torch.get_rng_state(),
            "cuda": [torch.cuda.get_rng_state()] if torch.cuda.is_initialized() else [],
            "cuda_device_index": torch.cuda.current_device() if torch.cuda.is_initialized() else None}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["cpu"].cpu())
    if state["cuda"]:
        require(torch.cuda.current_device() == state["cuda_device_index"], "exact resume CUDA device index changed")
        torch.cuda.set_rng_state(state["cuda"][0].cpu())


def environment():
    return {"torch": torch.__version__, "python": platform.python_version(), "cuda": torch.version.cuda,
            "deterministic": torch.are_deterministic_algorithms_enabled(),
            "tf32": torch.backends.cuda.matmul.allow_tf32}


def save_checkpoint(path, module, optimizer, scheduler, scaler, sampler, step, dependencies, config, *, global_batch, accumulation, consumed_ids, trainer_state=None):
    rank = dist.get_rank() if dist.is_initialized() else 0
    world = dist.get_world_size() if dist.is_initialized() else 1
    rngs = [None] * world
    if dist.is_initialized():
        dist.all_gather_object(rngs, capture_rng())
    else:
        rngs[0] = capture_rng()
    if rank == 0:
        state = {"schema_version": 1, "module": module.state_dict(), "optimizer": optimizer.state_dict(),
                 "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict() if scaler else None,
                 "optimizer_step": step, "sampler": sampler.state_dict(), "consumed_sample_ids": consumed_ids,
                 "rng_by_rank": rngs, "world_size": world, "global_batch": global_batch,
                 "accumulation": accumulation, "dependencies": dependencies, "resolved_config": config,
                 "environment": environment(), "resume_boundary": "optimizer_step", "trainer_state": trainer_state or {}}
        atomic_torch(path, state)
        atomic_json(str(path) + ".COMPLETE.json", {"checksum": file_hash(path), "optimizer_step": step,
                                                   "world_size": world, "dependencies": dependencies})
    if dist.is_initialized():
        dist.barrier()


def load_checkpoint(path, module, optimizer, scheduler, scaler, sampler, dependencies, config, *, global_batch, accumulation, non_exact_finetune=False):
    receipt = read_json(str(path) + ".COMPLETE.json")
    require(file_hash(path) == receipt["checksum"], "partial/corrupt training checkpoint")
    state = torch.load(path, map_location="cpu", weights_only=False)
    world = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0
    require(state["dependencies"] == dependencies, "checkpoint frozen/data dependencies changed")
    module.load_state_dict(state["module"], strict=True)
    if non_exact_finetune:
        return {"step": 0, "consumed_ids": [], "resume_mode": "NON_EXACT_FINETUNE"}
    require(state["world_size"] == world and state["global_batch"] == global_batch and state["accumulation"] == accumulation,
            "exact resume topology/global batch changed; explicitly use non_exact_finetune")
    require(state["resolved_config"] == config and state["environment"] == environment(), "exact resume execution environment/config changed")
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    if scaler is not None:
        require(state["scaler"] is not None, "checkpoint scaler mismatch")
        scaler.load_state_dict(state["scaler"])
    sampler.load_state_dict(state["sampler"])
    restore_rng(state["rng_by_rank"][rank])
    return {"step": state["optimizer_step"], "consumed_ids": state["consumed_sample_ids"], "resume_mode": "EXACT_OPTIMIZER_BOUNDARY", "trainer_state": state.get("trainer_state", {})}
