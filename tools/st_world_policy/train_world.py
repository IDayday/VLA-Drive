"""Registered original-DiT ST training using the existing DeepSpeed optimizer path.

Invoke with torchrun. New experiments strictly load the common S0, then start
fresh matched optimizers. Resume restores this run's native complete state.
No teacher model, scoring worker or evaluation data is loaded by this trainer.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
import fcntl
import json
import os
from pathlib import Path
import random
import signal
import socket
import subprocess
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.profiler import record_function

from starVLA.dataloader.st_world_targets import atomic_json
from starVLA.model.modules.st_world.losses import optimizer_batch_counts
from starVLA.model.modules.st_world.requests import sample_requests, gather_current, gather_future
from starVLA.model.modules.vehicle_joint.initialization import identity_hash, file_sha256
from tools.ddpolicy_vehicle.optimizer_safety import bounded_parameter_groups, capture_master_samples, master_update_evidence
from tools.ddpolicy_vehicle.training_state import epoch_batches
from tools.foresight.student_state import capture_rng, restore_rng, learning_rate, validate_rank_batches
from tools.st_world_policy.base_binding import read_binding
from tools.st_world_policy.check_formal_approval import check_formal_approval, read_json
from tools.st_world_policy.training_config import validate_config, environment_record
from tools.st_world_policy.training_state import checkpoint_receipt, resolve_latest, update_latest, terminal


def collective_call(function, description):
    """Coordinate data/forward exceptions before a rank enters backward."""
    result, error = None, None
    try:
        result = function()
    except Exception as exc:
        error = repr(exc)
    errors = [None] * dist.get_world_size()
    dist.all_gather_object(errors, error)
    if any(errors):
        raise RuntimeError(description + " failed on one or more ranks: " + str(errors))
    return result


def verified_runtime(registration, cfg):
    expected = read_json(registration["artifacts"]["environment"]["path"])
    if environment_record() != expected:
        raise ValueError("Actual Python/CUDA/training libraries differ from registration")
    lease = read_json(registration["artifacts"]["resource_lease"]["path"])
    if lease.get("schema") != "st_training_resource_lease_v1":
        raise ValueError("A training-specific bounded resource lease is required")
    host = socket.gethostname()
    if host not in lease["hostnames"] or lease["world_size"] != cfg["execution"]["world_size"]:
        raise ValueError("Actual host/rank topology differs from resource lease")
    if cfg["runtime"]["max_seconds"] > lease["max_invocation_seconds"]:
        raise ValueError("Invocation exceeds registered resource budget")
    if not lease["valid_from_unix"] <= time.time() < lease["valid_until_unix"]:
        raise ValueError("Resource lease is not currently valid")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible != lease["cuda_visible_devices_by_host"][host]:
        raise ValueError("Actual visible GPU set differs from the approved lease")
    uuids = subprocess.check_output(["nvidia-smi", "--id=" + visible,
        "--query-gpu=uuid", "--format=csv,noheader"], text=True).splitlines()
    if uuids != lease["gpu_uuids_by_host"][host]:
        raise ValueError("Physical GPU identities differ from the approved lease")
    return lease


def run(args):
    repo = Path(__file__).resolve().parents[2]
    registration = read_json(args.registration)
    cfg = read_json(registration["artifacts"]["config"]["path"])
    st = validate_config(cfg)
    execution = cfg["execution"]
    rank, world, local = (int(os.environ[k]) for k in ("RANK", "WORLD_SIZE", "LOCAL_RANK"))
    if world != execution["world_size"]:
        raise ValueError("torchrun process count differs from registered effective batch")
    approval = check_formal_approval(args.registration, args.approval, repo=repo,
                                     effective_execution=execution)
    lease = verified_runtime(registration, cfg)
    torch.set_num_threads(2)
    torch.cuda.set_device(local)
    dist.init_process_group("nccl", timeout=timedelta(minutes=10))
    lock, pool, profiler = None, None, None
    out = Path(cfg["runtime"]["output_root"]) / registration["run_id"]
    started = time.perf_counter()
    status = {"status": "STARTING"}
    try:
        from starVLA.dataloader.st_world_dataset import STWorldTrainingDataset, collate_st_world
        teacher = read_json(registration["artifacts"]["teacher_manifest"]["path"])
        contract = read_json(registration["artifacts"]["target_contract"]["path"])
        if contract != st.teacher.to_dict():
            raise ValueError("Registered target contract differs from active loss")
        if Path(registration["artifacts"]["teacher_manifest"]["path"]).resolve() != (
                Path(cfg["data"]["teacher_root"]) / "identity.json").resolve():
            raise ValueError("Actual teacher reader root is not the registered dependency")
        data = collective_call(lambda: STWorldTrainingDataset(
            cfg["data"]["root"], registration["artifacts"]["data_manifest"]["path"],
            cfg["data"]["teacher_root"], teacher["identity"], st.teacher,
            history_root=cfg["data"]["history_root"], pca_path=cfg["data"]["pca_path"],
            require_split="train", need_future=st.future), "training manifest/cache")
        if execution["sample_limit"] and len(data) != execution["sample_limit"]:
            raise ValueError("Diagnostic population must be the exact registered manifest, not a prefix")
        validate_rank_batches(len(data), execution["global_batch"], world, execution["micro_batch"])
        binding = collective_call(lambda: read_binding(
            registration["artifacts"]["base_binding"]["path"], verify_files=rank == 0), "bound S0 byte identity")
        identity = {"schema": "st_world_training_v1", "config": cfg, "data": data.identity,
                    "base_binding": binding["binding_sha256"], "approval": approval,
                    "source_sha": registration["source_sha"],
                    "registration_sha256": file_sha256(args.registration),
                    "new_optimizer_from_common_S0": "continuation" not in cfg,
                    "weights_usable_for_formal_init": registration["weights_usable_for_formal_init"]}
        signature = identity_hash(identity)

        def own_run():
            nonlocal lock
            if rank != 0:
                return
            if out.exists() and not args.resume:
                raise FileExistsError("Output already exists; explicitly resume this identity")
            if args.resume and not out.is_dir():
                raise FileNotFoundError("Resume cannot silently start a new run")
            out.mkdir(parents=True, exist_ok=True)
            lock = (out / "RUN.lock").open("a+")
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if args.resume:
                if read_json(out / "identity.json") != {"identity": signature, **identity}:
                    raise ValueError("Resume source/config/data/precision identity changed")
                if (out / "STOP_REQUESTED").exists():
                    if not args.acknowledge_stop:
                        raise ValueError("Explicit acknowledgment required for the prior operator stop")
                    (out / "STOP_REQUESTED").rename(out / ("STOP_ACKNOWLEDGED_" + str(time.time_ns())))
            else:
                atomic_json(out / "identity.json", {"identity": signature, **identity}, immutable=True)
                atomic_json(out / "resolved_config.json", cfg, immutable=True)
        collective_call(own_run, "run ownership")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(cfg["runtime"]["deterministic"])
        random.seed(execution["seed"] + rank)
        np.random.seed(execution["seed"] + rank)
        torch.manual_seed(execution["seed"] + rank)
        from tools.foresight.checkpoints import load_student
        from starVLA.model.framework.st_world_model import SpaceTimeStudent
        parent = load_student(binding["run"], binding["tag"], binding["parent_identity"],
                              device="cpu", precision="fp32", strip=False)
        parent.inference_fp32 = False  # Restore original training AMP, not canonical evaluation mode.
        checkpointing = cfg["runtime"]["gradient_checkpointing"]
        language = parent.qwen_vl_interface.model.model.language_model
        if checkpointing:
            language.gradient_checkpointing_enable()
        else:
            language.gradient_checkpointing_disable()
        parent.action_model.model.gradient_checkpointing = checkpointing
        # Temporal history implementation is injected explicitly; current-only
        # controls cannot silently replace it with repeated-current inputs.
        history_encoder = None
        if st.history:
            from starVLA.model.modules.st_world.history_adapter import build_history_encoder
            history_encoder = build_history_encoder(parent, initialization_seed=execution["seed"] + 3700)
        model = SpaceTimeStudent(parent, st, initialization_seed_value=execution["seed"],
                                  history_encoder=history_encoder)
        groups, group_records = bounded_parameter_groups(model.named_parameters())
        opt = cfg["optimizer"]
        ds = {"train_micro_batch_size_per_gpu": execution["micro_batch"],
              "gradient_accumulation_steps": 1, "train_batch_size": execution["micro_batch"] * world,
              "bf16": {"enabled": True}, "gradient_clipping": opt["clip"], "steps_per_print": 1000000,
              "zero_optimization": {"stage": 2, "overlap_comm": True, "contiguous_gradients": True,
                                    "reduce_bucket_size": 50000000, "allgather_bucket_size": 50000000},
              "optimizer": {"type": "AdamW", "params": {"lr": opt["lr"], "betas": opt["betas"],
                             "eps": opt["eps"], "weight_decay": opt["weight_decay"]}}}
        import deepspeed
        engine, _, _, _ = deepspeed.initialize(model=model, model_parameters=groups, config=ds)
        profile_mode = cfg["runtime"].get("profile_mode", "none")
        if profile_mode != "none":
            if args.resume:
                raise ValueError("A timing/trace run requires its complete separately registered measurement window")
            from tools.st_world_policy.profile_pipeline import PipelineProfiler, dtype_inventory
            scientific = {key: cfg[key] for key in ("space_time", "optimizer")}
            scientific.update(precision_policy=execution["precision_policy"], seed=execution["seed"],
                max_updates=execution["max_updates"], global_batch=execution["global_batch"],
                deterministic=cfg["runtime"]["deterministic"],
                gradient_checkpointing=cfg["runtime"]["gradient_checkpointing"])
            profile_identity = {"run_identity": signature, "source_sha": registration["source_sha"],
                "scientific_config_sha256": identity_hash(scientific),
                "data_manifest_sha256": registration["artifacts"]["data_manifest"]["sha256"],
                "teacher_identity": teacher["identity"], "precision_policy": execution["precision_policy"],
                "global_batch": execution["global_batch"],
                "sample_order_sha256": identity_hash({"manifest": data.identity["data_manifest_sha256"],
                    "seed": execution["seed"], "global_batch": execution["global_batch"],
                    "algorithm": "tools.ddpolicy_vehicle.training_state.epoch_batches"}),
                "registration_sha256": file_sha256(args.registration),
                "data_working_set_scenes": len(data), "setup_in_window": False}
            profiler = PipelineProfiler(out / "profile", identity=profile_identity, rank=rank,
                world_size=world, gpu_uuids=lease["gpu_uuids_by_host"][socket.gethostname()], mode=profile_mode)
            profiler.attach_modules({"frozen_visual": model.base.qwen_vl_interface.model.model.visual,
                "language": model.base.qwen_vl_interface.model.model.language_model,
                "action_projection": model.base.action_model.qwen_proj,
                "DiT": model.base.action_model.model, "sparse_readout": model.sparse_readout})
            atomic_json(out / "profile" / f"initial_dtypes_rank{rank}.json", dtype_inventory(model, engine.optimizer))
        if rank == 0 and not args.resume:
            atomic_json(out / "initialization.json", {"base": binding["binding_sha256"],
                "world_migration": model.migration, "groups": group_records,
                "model_parameters": sum(p.numel() for p in model.parameters()),
                "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                "teacher_instantiated": False,
                "optimizer_reset_for_common_control": "continuation" not in cfg,
                "native_parent_state_restored_next": "continuation" in cfg}, immutable=True)
        generators = {name: torch.Generator(device="cpu" if name in ("patch", "horizon") else "cuda").manual_seed(
            execution["seed"] + offset + rank) for name, offset in
            (("noise", 9000), ("time", 10000), ("horizon", 11000), ("patch", 12000))}
        state = {"identity": signature, "completed": 0, "epoch": 0, "offset": 0, "exposure": 0,
                 "termination_reason": "RUNNING_CHECKPOINT", "counts": {}}
        if args.resume or "continuation" in cfg:
            if args.resume:
                resolution = collective_call(lambda: resolve_latest(out / "checkpoints", identity=signature,
                                                                 world_size=world) if rank == 0 else None,
                                         "latest native checkpoint")
            else:
                from tools.st_world_policy.budget_extension import resolve_extension
                resolution = collective_call(lambda: resolve_extension(cfg, registration, repo=repo,
                    data_identity=data.identity, base_binding=binding["binding_sha256"])
                    if rank == 0 else None, "explicit reviewed phase-budget parent")
            resolutions = [resolution]
            dist.broadcast_object_list(resolutions, 0)
            resolution = resolutions[0]
            receipt = resolution["selected"]["receipt"]
            folder = Path(resolution["selected"]["folder"])
            location, restored = engine.load_checkpoint(str(folder.parent), tag=folder.name,
                load_module_strict=True, load_optimizer_states=True, load_lr_scheduler_states=True)
            if not location or any(restored.get(k) != v for k, v in receipt["state"].items()):
                raise ValueError("Native model/optimizer progress differs from complete receipt")
            # Installed DS0.16.9 returns additional engine metadata alongside
            # client_state. Preserve our exact registered client schema.
            state = dict(receipt["state"])
            if engine.global_steps != state["completed"]:
                raise ValueError("DeepSpeed optimizer step and consumed cursor disagree")
            if not args.resume:
                state.update(identity=signature, termination_reason="RUNNING_CHECKPOINT",
                             budget_extension_parent=resolution["parent_identity"])
                state.pop("tag", None)
            restore_rng(torch.load(folder / f"rng_rank{rank}.pt", map_location="cpu", weights_only=False), generators)
            if rank == 0:
                atomic_json(out / "RESUME_RESOLUTION.json", resolution)
        pause = [False]
        def signal_handler(*_):
            pause[0] = True
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
        pool = ThreadPoolExecutor(max_workers=cfg["runtime"]["loader_workers"]) if cfg["runtime"]["loader_workers"] else None
        status = {"identity": signature, "status": "RUNNING", "host": socket.gethostname(),
                  "pid": os.getpid(), "started_unix": time.time(), "source_sha": registration["source_sha"]}

        def write_status():
            if rank == 0:
                atomic_json(out / "status.json", {**status, **state, "updated_unix": time.time()})

        def save(reason):
            with record_function("st.checkpoint"):
                checkpoint_start = time.perf_counter()
                # A periodic checkpoint at this exact consumed boundary already
                # contains the needed native state. Do not duplicate tens of GB
                # merely to annotate an immediately following operator pause.
                if reason in ("OPERATOR_PAUSED", "BUDGET_PAUSED") and state.get("tag"):
                    previous = out / "checkpoints" / state["tag"] / "COMPLETE.json"
                    if previous.exists():
                        saved_state = read_json(previous)["state"]
                        keys = ("identity", "completed", "epoch", "offset", "exposure", "counts")
                        if all(saved_state[k] == state[k] for k in keys):
                            status["pause_reason"] = reason
                            write_status()
                            return 0.
                stamp = [time.time_ns() if rank == 0 else None]
                dist.broadcast_object_list(stamp, 0)
                tag = f"step_{state['completed']:06d}_{stamp[0]}"
                state.update(tag=tag, termination_reason=reason)
                saved_rng = capture_rng(generators)
                destination = out / "checkpoints" / tag
                engine.save_checkpoint(str(out / "checkpoints"), tag=tag, client_state=state, save_latest=False)
                temporary = destination / f"rng_rank{rank}.pt.tmp"
                torch.save(saved_rng, temporary)
                temporary.replace(destination / f"rng_rank{rank}.pt")
                dist.barrier()
                def commit():
                    if rank == 0:
                        checkpoint_receipt(destination, state, world_size=world)
                        update_latest(out / "checkpoints", tag)
                collective_call(commit, "checkpoint commit")
                restore_rng(saved_rng, generators)
                if rank == 0:
                    with (out / "checkpoint_costs.jsonl").open("a") as stream:
                        stream.write(json.dumps({"step": state["completed"], "tag": tag,
                            "seconds": time.perf_counter() - checkpoint_start}) + "\n")
                write_status()
                return time.perf_counter() - checkpoint_start

        model.train()
        write_status()
        if terminal(state, execution["max_updates"]):
            status["status"] = "COMPLETE"
            if rank == 0:
                atomic_json(out / "RESULT.json", {**status, **state, "new_optimizer_updates": 0})
            write_status()
            return
        initial_step = state["completed"]
        while state["completed"] < execution["max_updates"]:
            batches = epoch_batches(len(data), execution["global_batch"], execution["seed"], state["epoch"])
            if state["offset"] == len(batches):
                state["epoch"] += 1
                state["offset"] = 0
                continue
            stop = (pause[0] or (out / "STOP_REQUESTED").exists()
                    or time.perf_counter() - started >= cfg["runtime"]["max_seconds"]
                    or time.time() >= lease["valid_until_unix"]
                    or state["completed"] >= execution["stop_after"])
            stopped = torch.tensor(int(stop), device="cuda")
            dist.all_reduce(stopped, op=dist.ReduceOp.MAX)
            if stopped.item():
                operator = torch.tensor(int(pause[0] or (out / "STOP_REQUESTED").exists()), device="cuda")
                dist.all_reduce(operator, op=dist.ReduceOp.MAX)
                status["status"] = "PAUSED"
                save("OPERATOR_PAUSED" if operator.item() else "BUDGET_PAUSED")
                break
            torch.cuda.synchronize()
            if profiler:
                profiler.begin_update(state["completed"] + 1)
            tick = time.perf_counter()
            ids = batches[state["offset"]][rank::world]
            with record_function("st.data"):
                def fetch():
                    rows = list(pool.map(data.__getitem__, ids)) if pool else [data[i] for i in ids]
                    return collate_st_world(rows)
                observations, targets = collective_call(fetch, "current data and teacher labels")
            data_seconds = time.perf_counter() - tick
            request = sample_requests(len(ids), len(st.teacher.view_names), st.teacher.output_grid_hw,
                st.sampled_positions, patch_generator=generators["patch"], future=st.future,
                horizon_generator=generators["horizon"])
            _, current_mask = gather_current(targets["current_dense_features"], targets["current_valid"], request)
            future_mask = None
            if st.future:
                _, future_mask = gather_future(targets["future_dense_features"], targets["future_valid"], request)
                future_mask = future_mask.cuda()
            counts = optimizer_batch_counts(st.teacher.channels, current_valid=current_mask.cuda(),
                                             future_valid=future_mask, delta=st.delta)
            counts = {key: int(value.item()) for key, value in counts.items()}
            counts["ego_scenes"] = len(batches[state["offset"]])
            lr = learning_rate(state["completed"], opt["lr"], opt["minimum_lr"], opt["warmup"], execution["max_updates"])
            for group in engine.optimizer.param_groups:
                group["lr"] = lr
            logs = {}
            torch.cuda.reset_peak_memory_stats()
            for at in range(0, len(ids), execution["micro_batch"]):
                end = min(len(ids), at + execution["micro_batch"])
                boundary = end == len(ids)
                engine.set_gradient_accumulation_boundary(boundary)
                output = collective_call(lambda: engine(observations[at:end],
                    {k: v[at:end] for k, v in targets.items()}, requests=request.slice(at, end),
                    global_counts=counts, noise_generator=generators["noise"],
                    time_generator=generators["time"], completed_updates=state["completed"]), "ST forward")
                with record_function("st.backward"):
                    engine.backward(output["loss"])
                for name, value in output["losses"].items():
                    logs[name] = logs.get(name, torch.zeros((), device="cuda")) + value.detach().float()
                if boundary:
                    before = capture_master_samples(engine.optimizer)
                    if profiler and profile_mode == "trace" and state["completed"] == 1:
                        atomic_json(out / "profile" / f"after_backward_dtypes_rank{rank}.json",
                                    dtype_inventory(model, engine.optimizer))
                with record_function("st.optimizer"):
                    engine.step()
                if boundary:
                    update_evidence = master_update_evidence(engine.optimizer, before, torch.device("cuda", local))
            values = torch.stack([logs[key] for key in sorted(logs)])
            dist.all_reduce(values)
            logs = dict(zip(sorted(logs), (values / world).cpu().tolist()))
            state["completed"] += 1
            state["offset"] += 1
            state["exposure"] += counts["ego_scenes"]
            state["termination_reason"] = "RUNNING_CHECKPOINT"
            for name, value in counts.items():
                state["counts"][name] = state["counts"].get(name, 0) + value
            torch.cuda.synchronize()
            perf = torch.tensor([time.perf_counter() - tick, data_seconds,
                torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved(),
                float(model.last_sequence_lengths.max())], device="cuda", dtype=torch.float64)
            all_perf = [torch.empty_like(perf) for _ in range(world)]
            dist.all_gather(all_perf, perf)
            all_ids = [None] * world
            dist.all_gather_object(all_ids, [x.current["token"] for x in observations])
            if rank == 0:
                rows = [p.cpu().tolist() for p in all_perf]
                with (out / "steps.jsonl").open("a") as stream:
                    stream.write(json.dumps({"step": state["completed"], "lr": lr, "losses": logs,
                        "counts": counts, "sample_ids_by_rank": all_ids, "epoch": state["epoch"],
                        "offset": state["offset"], "exposure": state["exposure"],
                        "slowest_rank_seconds": max(p[0] for p in rows), "per_rank_profile": rows,
                        "profile_fields": ["update_seconds", "data_seconds", "peak_allocated", "peak_reserved", "max_sequence"],
                        "gradient_norm": float(engine.get_global_grad_norm()), **update_evidence}) + "\n")
            write_status()
            checkpoint_seconds = 0.
            if state["completed"] == execution["max_updates"]:
                status["status"] = "COMPLETE"
                checkpoint_seconds = save("MAX_STEPS")
            elif state["completed"] % cfg["runtime"]["save_every"] == 0:
                checkpoint_seconds = save("RUNNING_CHECKPOINT")
            # The complete-window boundary follows reductions, ID collection,
            # log/status I/O, and native checkpoint work. The old internal
            # step timer remains a diagnostic, never the throughput denominator.
            if profiler:
                profiler.end_update(local_scenes=len(ids), checkpoint_seconds=checkpoint_seconds,
                    metadata={"counts": counts, "losses": logs, "sample_ids": all_ids[rank]})
            if status["status"] == "COMPLETE":
                break
        if profiler:
            atomic_json(out / "profile" / f"final_dtypes_rank{rank}.json", dtype_inventory(model, engine.optimizer))
            profiler.finish()
            profiler = None
        if rank == 0:
            atomic_json(out / "RESULT.json", {**status, **state,
                "new_optimizer_updates": state["completed"] - initial_step,
                "allocation_wall_seconds": time.perf_counter() - started,
                "scientific_result": "NOT_EVALUATED", "formal_launch_approval": approval["formal_launch_allowed"]})
        write_status()
    except BaseException as error:
        if rank == 0 and out.exists():
            atomic_json(out / "FAILURE.json", {"status": "FAILED", "error": repr(error), "time_unix": time.time()})
        raise
    finally:
        if pool:
            pool.shutdown(wait=True)
        if lock:
            lock.close()
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", required=True)
    parser.add_argument("--approval")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--acknowledge-stop", action="store_true")
    args = parser.parse_args()
    if args.acknowledge_stop and not args.resume:
        parser.error("--acknowledge-stop requires --resume")
    try:
        run(args)
    except Exception as error:
        print(json.dumps({"status": "FAILED", "error": repr(error)}), flush=True)
        raise


if __name__ == "__main__":
    main()
