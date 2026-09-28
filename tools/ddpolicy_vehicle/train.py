"""Full-recipe camera training using original DeepSpeed ZeRO-2 and strict resume.

Launch with torchrun on allocated devices. Formal runs require an explicit NEW
campaign cap; --startup is bounded to <=4 real optimizer updates for debugging.
"""
import argparse
import datetime
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import time


def main():
    entry_start = time.time()
    p = argparse.ArgumentParser(__doc__)
    for key in ("config", "tokens", "processed-root", "vehicle-root", "depth-root", "campaign-root", "run-id"):
        p.add_argument("--"+key, required=True)
    p.add_argument("--global-batch", type=int, default=32)
    p.add_argument("--micro-batch", type=int, default=1)
    p.add_argument("--updates", type=int, default=100000)
    p.add_argument("--save-every", type=int, default=1000)
    p.add_argument("--milestones", default="0,1000,5000,10000,25000,50000,75000,90000,100000")
    p.add_argument("--campaign-gpu-hours", type=float)
    p.add_argument("--max-seconds", type=int, default=1800)
    p.add_argument("--startup", action="store_true")
    p.add_argument("--offload-optimizer", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--resume-tag", help="Explicit complete checkpoint when an interrupted write left latest incomplete")
    p.add_argument("--acknowledge-stop", action="store_true")
    p.add_argument("--stop-after", type=int, default=0)
    a = p.parse_args()
    if min(a.global_batch, a.micro_batch, a.updates, a.save_every, a.max_seconds) < 1:
        raise ValueError("Positive training/ledger bounds required")
    if a.stop_after < 0 or a.stop_after > a.updates: raise ValueError("Invalid planned pause boundary")
    if a.resume_tag and not a.resume: raise ValueError("--resume-tag requires --resume")
    milestones = set(map(int, a.milestones.split(",")))
    if any(x < 0 or x > a.updates for x in milestones): raise ValueError("Milestones must be within this run")
    if a.startup and (a.updates > 4 or a.max_seconds > 1800):
        raise ValueError("Startup is <=4 real updates and <=1800 seconds; it is not a formal run")
    if not a.startup and (a.campaign_gpu_hours is None or a.campaign_gpu_hours <= 0):
        raise ValueError("Formal training requires the NEW explicitly registered campaign GPU-hour cap")
    import numpy as np
    import torch
    import torch.distributed as dist
    import deepspeed
    from omegaconf import OmegaConf
    from types import SimpleNamespace
    from starVLA.model.modules.vehicle_joint.initialization import identity_hash, file_sha256
    from starVLA.model.modules.vehicle_joint.masks import VehicleRoleScheduler
    from starVLA.dataloader.ddpolicy_vehicle_dataset import DDPVehicleDataset
    from starVLA.model.framework.DDPVehicle import DDPVehicle
    from .prepare_data import atomic_json
    from .training_state import epoch_batches, capture_rng, restore_rng
    import random
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    torch.cuda.set_device(local_rank)
    os.environ.setdefault("LOCAL_RANK", str(local_rank))
    if dist.is_initialized():
        if dist.get_rank() != rank or dist.get_world_size() != world:
            raise ValueError("Existing distributed group does not match launcher identity")
    elif world > 1:
        dist.init_process_group("nccl")
    else:
        # DeepSpeed requires a process group even for single-GPU CPU offload.
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29783")
        os.environ.setdefault("RANK", "0"); os.environ.setdefault("WORLD_SIZE", "1")
        dist.init_process_group("nccl", rank=0, world_size=1)
    if a.global_batch % world:
        raise ValueError("Global batch must divide across allocated GPUs")
    cfg = OmegaConf.load(a.config)
    if int(cfg.trainer.max_train_steps) < a.updates:
        raise ValueError("Requested updates exceed the preregistered scheduler horizon")
    if cfg.from_scratch.auxiliary_cadence != 4 or cfg.from_scratch.auxiliary_weight != .1 or cfg.from_scratch.vehicle_xy_scale != 20.:
        raise ValueError("Unsupported research recipe override")
    if cfg.from_scratch.graph_training != "predicted_only_from_start":
        raise ValueError("This implementation explicitly trains camera-predicted graphs from the start")
    source_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if subprocess.check_output(["git", "status", "--porcelain"]):
        raise ValueError("Lock a clean training source commit before launching")
    root, run_dir = Path(a.campaign_root), Path(a.campaign_root)/"training"/a.run_id
    identity = {"source_sha": source_sha, "config": OmegaConf.to_container(cfg, resolve=True),
                "tokens_sha256": file_sha256(a.tokens), "vehicle_identity": json.loads((Path(a.vehicle_root)/"identity.json").read_text()),
                "depth_identity": json.loads((Path(a.depth_root)/"identity.json").read_text()),
                "world_size": world, "global_batch": a.global_batch, "micro_batch": a.micro_batch,
                "updates": a.updates, "startup": a.startup, "offload_optimizer": a.offload_optimizer,
                "precision": "DeepSpeed BF16 with FP32 optimizer masters", "arm": cfg.from_scratch.arm}
    identity_sha = identity_hash(identity)
    lock = None
    if rank == 0:
        if run_dir.exists() and not a.resume: raise FileExistsError("Run exists; no implicit restart/overwrite")
        run_dir.mkdir(parents=True, exist_ok=True)
        lock = (run_dir/"RUN.lock").open("a+")
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if a.resume:
            if json.loads((run_dir/"identity.json").read_text())["sha256"] != identity_sha:
                raise ValueError("Resume code/config/data/precision/device-count identity mismatch")
            status = json.loads((run_dir/"status.json").read_text())["status"]
            if status == "COMPLETE": raise ValueError("Completed runs cannot be silently extended")
            if not a.acknowledge_stop: raise ValueError("Resume requires explicit --acknowledge-stop")
            if (run_dir/"STOP_REQUESTED").exists():
                os.replace(run_dir/"STOP_REQUESTED", run_dir/f"STOP_ACKNOWLEDGED_{time.time_ns()}")
        else:
            atomic_json(run_dir/"identity.json", {"sha256": identity_sha, **identity})
    dist.barrier()
    attempt = f"attempt_{time.time_ns()}" if rank == 0 else None
    shared = [attempt]; dist.broadcast_object_list(shared, 0); attempt = shared[0]
    start = entry_start
    status = {"status": "RUNNING", "identity": identity_sha, "source_sha": source_sha,
              "gpu_count": world, "host": socket.gethostname(), "pid": os.getpid(), "start_unix": start,
              "real_optimizer_updates": 0, "sample_presentations": 0, "kind": "startup_training" if a.startup else "formal_training"}
    engine = None; completed = epoch = offset = exposures = 0
    attempt_initial_updates = attempt_initial_exposures = 0
    signal_stop = [False]
    def stop_handler(*_): signal_stop[0] = True
    signal.signal(signal.SIGTERM, stop_handler); signal.signal(signal.SIGINT, stop_handler)
    def save_status():
        if rank != 0: return
        status.update(real_optimizer_updates=completed, sample_presentations=exposures,
                      attempt_optimizer_updates=completed-attempt_initial_updates,
                      attempt_sample_presentations=exposures-attempt_initial_exposures,
                      epoch=epoch, batch_offset=offset, end_unix=time.time(),
                      gpu_hours=(time.time()-start)*world/3600)
        atomic_json(run_dir/"status.json", status)
        atomic_json(run_dir/(attempt+".json"), status)
    def campaign_used():
        used = 0.
        for path in list((root/"runs").glob("*/status.json"))+list((root/"training").glob("*/attempt_*.json")):
            entry = json.loads(path.read_text())
            end = time.time() if entry["status"] == "RUNNING" else entry.get("end_unix", entry["start_unix"])
            used += (end-entry["start_unix"])*entry["gpu_count"]/3600
        return used
    save_status()
    try:
        if not a.startup and campaign_used() >= a.campaign_gpu_hours:
            raise RuntimeError("Campaign budget exhausted before model loading")
        random.seed(cfg.seed+rank); np.random.seed(cfg.seed+rank); torch.manual_seed(cfg.seed+rank)
        sources = json.loads(Path(cfg.from_scratch.source_manifest).read_text())
        os.environ["DEPTH_MODEL_CKPTS"] = sources["ppd"]["root"]
        ds = DDPVehicleDataset(a.tokens, a.processed_root, a.vehicle_root, a.depth_root, cfg,
                              sensor_roots=json.loads((Path(a.vehicle_root)/"identity.json").read_text()).get("fallback_sensor_root", []))
        if len(ds) < 1: raise ValueError("Empty training manifest")
        if len(ds) % world: raise ValueError("This fixed split requires an exactly divisible final per-rank tail")
        model = DDPVehicle(cfg, accelerator=SimpleNamespace(process_index=rank, device=torch.device("cuda", local_rank)))
        if rank == 0 and not a.resume:
            from starVLA.model.modules.vehicle_joint.initialization import driving_initialization_manifest
            atomic_json(run_dir/"driving_initialization.json", driving_initialization_manifest(model))
            atomic_json(run_dir/"parameters.json", {
                "total": sum(p.numel() for p in model.parameters()),
                "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
                "generic_sources": {k:{f:v for f,v in source.items() if f != "root"} for k,source in sources.items()},
                "pretrained_driving_weights_loaded": False,
                "policy_initialization": "generic public modules plus random driving modules"})
        dist.barrier()
        ds_config = {"train_micro_batch_size_per_gpu": a.micro_batch, "gradient_accumulation_steps": 1,
                     "train_batch_size": a.micro_batch*world, "bf16": {"enabled": True},
                     "gradient_clipping": 1., "steps_per_print": 1000000,
                     "zero_optimization": {"stage": 2, "overlap_comm": True, "contiguous_gradients": True,
                        "reduce_bucket_size": 50000000, "allgather_bucket_size": 50000000},
                     "optimizer": {"type": "AdamW", "params": {"lr": float(cfg.trainer.learning_rate.base),
                        "betas": list(cfg.trainer.optimizer.betas), "eps": float(cfg.trainer.optimizer.eps),
                        "weight_decay": float(cfg.trainer.optimizer.weight_decay)}}}
        if a.offload_optimizer:
            ds_config["zero_optimization"]["offload_optimizer"] = {"device": "cpu", "pin_memory": True}
            ds_config["zero_force_ds_cpu_optimizer"] = True
        engine, _, _, _ = deepspeed.initialize(model=model,
            model_parameters=[p for p in model.parameters() if p.requires_grad], config=ds_config)
        roles = VehicleRoleScheduler(int(cfg.seed)+7000+rank)
        noise = torch.Generator(device=torch.device("cuda", local_rank)).manual_seed(int(cfg.seed)+9000+rank)
        if a.resume:
            tag = a.resume_tag or (run_dir/"checkpoints/latest").read_text().strip()
            if Path(tag).name != tag: raise ValueError("Resume tag must be a single directory name")
            complete = json.loads((run_dir/"checkpoints"/tag/"COMPLETE.json").read_text())
            if complete["identity"] != identity_sha or complete["tag"] != tag:
                raise ValueError("Latest checkpoint is incomplete or has changed identity")
            load_path, state = engine.load_checkpoint(str(run_dir/"checkpoints"), tag=tag, load_module_strict=True,
                load_optimizer_states=True, load_lr_scheduler_states=True)
            if not load_path or state["identity"] != identity_sha: raise ValueError("Invalid resume checkpoint identity")
            completed, epoch, offset, exposures = (state[k] for k in ("completed", "epoch", "offset", "exposures"))
            if any(state[k] != complete[k] for k in ("completed", "epoch", "offset", "exposures")):
                raise ValueError("Checkpoint completion record does not match saved progress")
            attempt_initial_updates, attempt_initial_exposures = completed, exposures
            rank_state = torch.load(run_dir/"checkpoints"/state["tag"]/f"rng_rank{rank}.pt", weights_only=False, map_location="cpu")
            restore_rng(rank_state, model, noise, roles)
        def checkpoint(tag):
            destination = run_dir/"checkpoints"/tag
            exists = [destination.exists() if rank == 0 else None]
            dist.broadcast_object_list(exists, 0)
            if exists[0]: raise FileExistsError("Immutable checkpoint tag already exists: "+tag)
            state = {"identity": identity_sha, "completed": completed, "epoch": epoch, "offset": offset,
                     "exposures": exposures, "tag": tag, "scheduler": {
                         "type": "fixed_formula", "completed": completed,
                         "warmup": int(cfg.trainer.num_warmup_steps), "horizon": int(cfg.trainer.max_train_steps),
                         "base_lr": float(cfg.trainer.learning_rate.base),
                         "minimum_lr": float(cfg.trainer.scheduler_specific_kwargs.min_lr)}}
            engine.save_checkpoint(str(run_dir/"checkpoints"), tag=tag, client_state=state, save_latest=False)
            torch.save(capture_rng(model, noise, roles), run_dir/"checkpoints"/tag/f"rng_rank{rank}.pt")
            dist.barrier()
            if rank == 0:
                atomic_json(run_dir/"checkpoints"/tag/"COMPLETE.json", state)
                pointer = run_dir/"checkpoints"/f"latest.{os.getpid()}.tmp"
                pointer.write_text(tag+"\n")
                os.replace(pointer, run_dir/"checkpoints/latest")
                if tag.startswith("periodic_"):
                    # Only this run's obsolete rolling saves are replaceable.
                    # Milestones, pauses, endpoints and foreign artifacts stay.
                    import shutil
                    for old in (run_dir/"checkpoints").glob("periodic_*"):
                        if old == destination or not (old/"COMPLETE.json").exists(): continue
                        previous = json.loads((old/"COMPLETE.json").read_text())
                        if previous["identity"] != identity_sha: raise ValueError("Foreign rolling checkpoint")
                        if previous["completed"] < completed: shutil.rmtree(old)
            dist.barrier()
        if 0 in milestones and not a.resume: checkpoint("milestone_000000")
        model.train()
        while completed < a.updates:
            global_batches = epoch_batches(len(ds), a.global_batch, int(cfg.seed), epoch)
            if offset == len(global_batches): epoch += 1; offset = 0; continue
            stop = signal_stop[0] or (run_dir/"STOP_REQUESTED").exists() or time.time()-start >= a.max_seconds
            stop = stop or bool(a.stop_after and completed >= a.stop_after)
            if not a.startup and campaign_used() >= a.campaign_gpu_hours: stop = True
            flag = torch.tensor(int(stop), device="cuda"); dist.all_reduce(flag, op=dist.ReduceOp.MAX)
            if flag:
                checkpoint(f"paused_{completed:06d}_{attempt}"); status["status"] = "PAUSED"; break
            ids = global_batches[offset][rank::world]
            warmup, horizon = int(cfg.trainer.num_warmup_steps), int(cfg.trainer.max_train_steps)
            base, minimum = float(cfg.trainer.learning_rate.base), float(cfg.trainer.scheduler_specific_kwargs.min_lr)
            lr = base*(completed+1)/warmup if completed < warmup else minimum+(base-minimum)*.5*(1+math.cos(math.pi*(completed-warmup)/max(1, horizon-warmup)))
            for group in engine.optimizer.param_groups: group["lr"] = lr
            step_start = time.monotonic(); loss_log = {}; coordinate_log = {"ego": 0, "vehicle": 0}
            graph_log = {"selected_vehicles": 0, "context_vehicles": 0, "ego_only": 0}
            for j in range(0, len(ids), a.micro_batch):
                part = ids[j:j+a.micro_batch]
                engine.set_gradient_accumulation_boundary(j+len(part) == len(ids))
                batch = [ds[index] for index in part]
                output = engine(batch, completed_updates=completed, role_scheduler=roles, noise_generator=noise)
                if not torch.isfinite(output["loss"]): raise FloatingPointError("Nonfinite full camera training loss")
                # Equal scene weighting, including the actual final tail size.
                weight = len(part)/len(ids)
                engine.backward(output["loss"]*weight)
                for name, value in output["losses"].items(): loss_log[name] = loss_log.get(name, 0.)+float(value.detach())*weight
                coordinate_log["ego"] += output["metrics"].get("ego_coordinates", 0)
                coordinate_log["vehicle"] += output["metrics"].get("vehicle_coordinates", 0)
                for audit in output["metrics"].get("graphs", []):
                    graph_log["selected_vehicles"] += audit["selected_vehicles"]
                    graph_log["context_vehicles"] += audit["context_vehicles"]
                    graph_log["ego_only"] += audit["ego_only"]
            engine.step()
            completed += 1; offset += 1; exposures += len(global_batches[offset-1])
            row = {"update": completed, "epoch": epoch, "offset": offset, "global_scene_exposure": exposures,
                   "lr": lr, "seconds": time.monotonic()-step_start, "rank": rank, "losses": loss_log,
                   "coordinates": coordinate_log, "graphs": graph_log, "roles": dict(roles.counts),
                   "peak_allocated_bytes": torch.cuda.max_memory_allocated()}
            with (run_dir/f"train_rank{rank}.jsonl").open("a") as f: f.write(json.dumps(row)+"\n")
            if completed in milestones: checkpoint(f"milestone_{completed:06d}")
            elif completed % a.save_every == 0: checkpoint(f"periodic_{completed:06d}")
            save_status()
        if completed == a.updates:
            # A milestone at the endpoint is already a complete, immutable save.
            if completed not in milestones: checkpoint(f"endpoint_{completed:06d}")
            status["status"] = "COMPLETE"
    except BaseException as error:
        status["status"] = "FAILED"; status["error"] = repr(error)
        raise
    finally:
        save_status()
        dist.destroy_process_group()
        if lock is not None: lock.close()


if __name__ == "__main__": main()
