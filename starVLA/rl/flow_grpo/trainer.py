"""Synchronous full-parameter Accelerate/DeepSpeed Flow-GRPO training."""

from pathlib import Path
import json
import os
import time
from datetime import timedelta
import torch
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.utils import (
    DistributedDataParallelKwargs,
    GradientAccumulationPlugin,
    set_seed,
    InitProcessGroupKwargs,
)
from .checkpoint import save_boundary, resume_boundary
from .config import split_tokens, config_hash
from .contracts import (
    optimizer_groups,
    parameter_manifest,
    check_manifest,
    check_optimizer,
    digest,
    apply_rl_freezes,
)
from .data import KeyedDataset, SceneStream, to_device
from .distributed import (
    synchronized_call,
    rank0_call,
    rank0_result,
    initialize_control_group,
)
from .loading import load_policy, enable_checkpointing
from .model import FlowGRPOActor, make_reference
from .observation import prepare_policy_observation
from .rollout import SamplingSpec, evaluate_transitions
from .advantages import assign_behavior_advantages
from .reward import RewardService
from .monitor import (
    GradientMonitor,
    CheckedDeepSpeedBackward,
    ParameterProbe,
    optimizer_norm_metrics,
    dtype_inventory,
    ActivationDtypeMonitor,
    CommunicationDtypeMonitor,
)
from .metrics import Metrics
from .reproducibility import configure_numerics, resume_assets
from .acceptance import enforce_training_budget, acceptance_context
from .math import reduce_dimensions


def make_accelerator(cfg):
    runtime = cfg["runtime"]
    accum = runtime["accumulation_steps"]
    plugin = None
    if runtime["deepspeed_stage"]:
        ds = {
            "bf16": {"enabled": True},
            "fp16": {"enabled": False},
            "data_types": {"grad_accum_dtype": "fp32"},
            "communication_data_type": "fp32",
            "train_micro_batch_size_per_gpu": 1,
            "gradient_accumulation_steps": accum,
            "train_batch_size": "auto",
            "zero_optimization": {
                "stage": runtime["deepspeed_stage"],
                "overlap_comm": False,
                "contiguous_gradients": True,
                "reduce_bucket_size": 50000000,
                "allgather_bucket_size": 50000000,
            },
            "gradient_clipping": cfg["optimizer"]["max_grad_norm"],
            "steps_per_print": 1000,
            "zero_allow_untested_optimizer": True,
        }
        if runtime["optimizer_offload"]:
            ds["zero_optimization"]["offload_optimizer"] = {
                "device": "cpu",
                "pin_memory": True,
            }
        plugin = DeepSpeedPlugin(hf_ds_config=ds)
    return Accelerator(
        mixed_precision="bf16",
        deepspeed_plugin=plugin,
        gradient_accumulation_plugin=GradientAccumulationPlugin(
            num_steps=accum, sync_with_dataloader=False, sync_each_batch=True
        ),
        kwargs_handlers=[
            DistributedDataParallelKwargs(find_unused_parameters=True),
            InitProcessGroupKwargs(
                timeout=timedelta(seconds=runtime.get("process_group_timeout", 120))
            ),
        ],
    )


def _run(cfg, sft, resume=None):
    from .diagnostic_loss import validate_scope, selected_loss

    runtime = cfg["runtime"]
    validate_scope(runtime)
    maximum = runtime["max_updates"]
    if maximum is None or maximum <= 0 or not runtime["output_dir"]:
        raise ValueError("MAX_UPDATES and OUTPUT_DIR must be explicit")
    enforce_training_budget(cfg)
    accelerator = make_accelerator(cfg)
    initialize_control_group(runtime.get("asset_verification_timeout", 1800))
    configure_numerics()
    assets = rank0_result(lambda: resume_assets(cfg, sft), accelerator.device)
    initialize_control_group(runtime.get("process_group_timeout", 120))
    rank0_call(
        lambda: enforce_training_budget(cfg, acceptance_context(cfg, assets)),
        accelerator.device,
    )
    set_seed(runtime["seed"], device_specific=True)
    output = Path(runtime["output_dir"])
    rank0_call(
        lambda: initialize_run_directory(output, cfg, resume), accelerator.device
    )
    from .transactions import atomic_json

    def write_execution_context():
        path = output / "execution_context.json"
        context = acceptance_context(cfg, assets)
        if path.exists():
            if json.loads(path.read_text()) != context:
                raise ValueError("existing execution context differs; refusing to overwrite evidence")
        else:
            atomic_json(path, context)

    rank0_call(
        write_execution_context,
        accelerator.device,
    )
    train, val = split_tokens(cfg)
    dataset = KeyedDataset(sft)
    train_stream = SceneStream(
        dataset,
        train,
        runtime["seed"],
        accelerator.process_index,
        accelerator.num_processes,
        workers=runtime["data_workers"],
        prefetch=runtime["prefetch_factor"],
    )
    replay_stream = SceneStream(
        dataset,
        train,
        runtime["seed"] + 104729,
        accelerator.process_index,
        accelerator.num_processes,
        workers=runtime["data_workers"],
        prefetch=runtime["prefetch_factor"],
    )
    policy = load_policy(cfg, sft, accelerator).to(accelerator.device)
    # Deterministic policy forward for rollout AND gradient recomputation, including
    # activation checkpoint recomputation. eval does not turn off autograd.
    policy.eval()
    groups = optimizer_groups(policy, sft, cfg["optimizer"]["initial_lr_multiplier"])
    source_manifest = parameter_manifest(policy, groups)
    allowed_freezes = apply_rl_freezes(policy, cfg)
    groups = optimizer_groups(policy, sft, cfg["optimizer"]["initial_lr_multiplier"])
    manifest = parameter_manifest(policy, groups)
    check_manifest(source_manifest, manifest, allowed_freezes)

    def write_manifests():
        (output / "sft_parameter_manifest.json").write_text(
            json.dumps(manifest, indent=2)
        )
        (output / "source_sft_parameter_manifest.json").write_text(
            json.dumps(source_manifest, indent=2)
        )
        (output / "actor_parameter_manifest.json").write_text(
            json.dumps(manifest, indent=2)
        )
        (output / "parameter_contract_exceptions.json").write_text(
            json.dumps(
                {
                    "user_authorized_rl_freezes": cfg.get("rl_freeze_modules", []),
                    "authorized_frozen_parameter_names": allowed_freezes,
                    "sft_trainable_numel": source_manifest["trainable_numel"],
                    "actor_trainable_numel": manifest["trainable_numel"],
                },
                indent=2,
            )
        )

    rank0_call(write_manifests, accelerator.device)
    dtype_before = dtype_inventory(policy)
    # Clone BEFORE wrapping checkpoint methods or DeepSpeed conversion.
    reference = make_reference(policy).to("cpu")
    enable_checkpointing(policy, runtime["activation_checkpointing"])
    monitor = GradientMonitor(policy)
    actor = FlowGRPOActor(policy, cfg)
    check_manifest(manifest, parameter_manifest(actor.policy, groups))
    check_optimizer(actor, groups)
    optcfg = sft.trainer.optimizer
    optimizer = torch.optim.AdamW(
        groups,
        betas=tuple(optcfg.betas),
        eps=optcfg.eps,
        weight_decay=optcfg.weight_decay,
    )
    # Fixed LR preserves max_updates-independent exact resume; SFT group LR ratios
    # and AdamW hyperparameters are inherited. Scheduler state remains explicit.
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
    actor, optimizer, scheduler = accelerator.prepare(actor, optimizer, scheduler)
    from .zero2_precision import PROFILE, install_fp32_partitions

    if runtime.get("numerical_profile") == PROFILE:
        if runtime["deepspeed_stage"] != 2:
            raise ValueError("FP32 partition profile requires ZeRO-2")
        correction = install_fp32_partitions(actor)
        rank0_call(
            lambda: (output / "zero2_precision.json").write_text(json.dumps(correction, indent=2)),
            accelerator.device,
        )
    if runtime["deepspeed_stage"]:
        from .monitor import save_full_optimizer_gradients

        probe = None
        if runtime.get("diagnostic_optimizer_gradients", False) or runtime.get("diagnostic_gradient_statistics", False):
            if runtime.get("run_mode", "diagnostic") != "diagnostic":
                raise ValueError(
                    "full gradient dump is only available in bounded diagnostics"
                )

            def probe(engine):
                return save_full_optimizer_gradients(
                    engine, output / "optimizer_gradients",
                    save_tensors=runtime.get("diagnostic_optimizer_gradients", False),
                )

        accelerator.deepspeed_engine_wrapped = CheckedDeepSpeedBackward(
            actor, monitor, accelerator.device, boundary_probe=probe
        )
    actor.eval()
    policy = accelerator.unwrap_model(actor).policy
    activation_dtypes = ActivationDtypeMonitor(policy)
    communication_monitor = None
    if runtime["deepspeed_stage"] and runtime.get("run_mode") == "diagnostic":
        communication_monitor = CommunicationDtypeMonitor(actor)
    if runtime["deepspeed_stage"]:
        reference.to(dtype=next(policy.action_model.parameters()).dtype)
    if not runtime["reference_offload"]:
        reference.to(accelerator.device)
    synchronized_call(
        lambda: (
            output / f"dtype_prepared_rank{accelerator.process_index}.json"
        ).write_text(
            json.dumps(
                {
                    "before_prepare": dtype_before,
                    "after_prepare": dtype_inventory(
                        policy, actor if runtime["deepspeed_stage"] else None
                    ),
                },
                indent=2,
            )
        ),
        accelerator.device,
    )
    sftsha = policy._flow_source_sha256
    if sftsha != cfg["checkpoint_contract"]["sha256"]:
        raise ValueError("SFT checkpoint SHA does not match audited contract")
    from .reproducibility import training_provenance

    provenance = training_provenance(cfg, sft, assets)
    service = RewardService(
        Path("navsim").resolve(),
        cfg["paths"]["metric_cache"],
        "train",
        train,
        workers=runtime["reward_workers"],
        timeout=runtime["reward_timeout"],
        cache_dir=output / "reward_cache",
    )
    assert provenance["reward_config_hash"] == digest(service.metadata)
    spec = SamplingSpec(
        group_size=cfg["sampling"]["group_size"],
        num_steps=cfg["sampling"]["num_steps"],
        noise_level=cfg["sampling"]["noise_level"],
        reduction=cfg["algorithm"]["logprob_reduction"],
        candidate_chunk_size=cfg["sampling"]["candidate_chunk_size"],
        transition_chunk_size=cfg["sampling"]["transition_chunk_size"],
    )
    update = version = 0
    pending = None
    if resume:
        synchronized_call(
            lambda: check_manifest(
                json.loads((Path(resume) / "sft_parameter_manifest.json").read_text()),
                manifest,
            ),
            accelerator.device,
        )
        update, version, pending = resume_boundary(
            resume, accelerator, actor, cfg, provenance, [train_stream, replay_stream]
        )
    from infer import deal_action_1225

    accelerator.print(
        f"Flow-GRPO full SFT: {manifest['trainable_numel']:,} trainable parameters; start={update}, stop={maximum}"
    )
    optimizer.zero_grad()
    parameter_probe = ParameterProbe(policy)
    from .contracts import tensor_hashes

    def verify_own_visual():
        if cfg.get("rl_freeze_modules"):
            actor_visual = policy.get_submodule("qwen_vl_interface.model.visual")
            source_visual = reference.get_submodule("qwen_vl_interface.model.visual")
            if tensor_hashes(actor_visual) != tensor_hashes(source_visual):
                raise ValueError(
                    "actor visual no longer equals its OWN original SFT reference visual"
                )
            if any(
                p.requires_grad or p.grad is not None for p in actor_visual.parameters()
            ):
                raise ValueError("visual must remain frozen with no gradient")

    synchronized_call(verify_own_visual, accelerator.device)

    immutable_before = synchronized_call(
        lambda: {
            "reference": tensor_hashes(reference),
            "frozen": tensor_hashes(policy, lambda n, p: not p.requires_grad),
        },
        accelerator.device,
    )

    def check_immutable():
        after = {
            "reference": tensor_hashes(reference),
            "frozen": tensor_hashes(policy, lambda n, p: not p.requires_grad),
        }
        changed = {
            scope: [
                name
                for name, value in hashes.items()
                if after[scope].get(name) != value
            ]
            for scope, hashes in immutable_before.items()
        }
        evidence = {
            "status": "FAILED" if any(changed.values()) else "TESTED",
            "scope": "all complete tensors before and after actual optimizer updates",
            "end_update": update,
            "changed": changed,
            "before": immutable_before,
            "after": after,
        }
        (
            output
            / f"immutable_update{update:06d}_rank{accelerator.process_index}.json"
        ).write_text(json.dumps(evidence, indent=2))
        if any(changed.values()):
            raise RuntimeError("reference/frozen parameters changed during training")

    failed = True
    try:
        while update < maximum:
            t0 = time.monotonic()
            buffers = []
            start_inner = 0
            if pending is not None:
                buffers = to_device(pending["buffers"], accelerator.device)
                replay = pending["replay"]
                start_inner = pending["next_inner"]
                pending = None
            else:
                scenes = synchronized_call(
                    lambda: train_stream.take(runtime["accumulation_steps"]),
                    accelerator.device,
                )
                replay = synchronized_call(
                    lambda: replay_stream.take(runtime["accumulation_steps"]),
                    accelerator.device,
                )
                # Entire behavior batch is sampled before ANY optimizer update.
                for i, scene in enumerate(scenes):
                    observation = prepare_policy_observation([scene])
                    seed = (
                        runtime["seed"]
                        + version * 1000003
                        + accelerator.process_index * 1009
                        + i
                    )
                    if runtime.get("noise_seed_schedule") == "global_scene_v1":
                        global_position = (
                            train_stream.cursor - len(scenes) + i
                        ) * accelerator.num_processes + accelerator.process_index
                        seed = runtime["seed"] + global_position * 1000003
                    with torch.no_grad(), accelerator.autocast():
                        rollout = actor(
                            mode="rollout",
                            observation=observation,
                            spec=spec,
                            policy_version=version,
                            seed=seed,
                            provenance=provenance,
                        )
                    physical = deal_action_1225(
                        rollout.raw_final_action.cpu().numpy(),
                        act_norm=int(sft.datasets.vla_data.act_norm),
                    )
                    rollout.physical_trajectories = physical
                    records = synchronized_call(
                        lambda: service.score(observation.tokens, physical),
                        accelerator.device,
                    )
                    rollout.score_records = records
                    rewards = torch.tensor(
                        [[r.score for r in row] for row in records],
                        device=accelerator.device,
                    )
                    rollout.rewards = rewards
                    if runtime["reference_offload"]:
                        reference.to(accelerator.device)
                    with torch.no_grad(), accelerator.autocast():
                        stats = synchronized_call(
                            lambda: evaluate_transitions(
                                reference, observation, rollout
                            ),
                            accelerator.device,
                        )
                    rollout.reference_mean = stats["mean"].detach()
                    rollout.reference_std = stats["std"].detach()
                    if runtime["reference_offload"]:
                        reference.to("cpu")
                    buffers.append(rollout)
                assign_behavior_advantages(buffers, cfg["algorithm"], accelerator.device)
            rollout_seconds = time.monotonic() - t0
            old_fingerprints = [behavior_digest(r) for r in buffers]
            for inner in range(start_inner, cfg["algorithm"]["inner_epochs"]):
                metrics = Metrics()
                pre_norm = None
                t1 = time.monotonic()
                for i, rollout in enumerate(buffers):
                    rollout.validate(version)
                    if rollout.provenance != provenance:
                        raise ValueError(
                            "behavior provenance differs from this training job; diagnostic buffers are forbidden"
                        )
                    with accelerator.accumulate(actor):
                        batch = to_device([replay[i]], accelerator.device)
                        with accelerator.autocast():
                            result = actor(mode="update", rollout=rollout, replay=batch)
                        # One equally weighted scene per microbatch; Accelerate/DS
                        # owns the 1/accum scaling, never G*K copies of replay.
                        accelerator.backward(selected_loss(result, runtime, update))
                        if not runtime["deepspeed_stage"]:
                            monitor.assert_finite_collective(accelerator.device)
                        if (
                            accelerator.sync_gradients
                            and not runtime["deepspeed_stage"]
                        ):
                            pre_norm = accelerator.clip_grad_norm_(
                                actor.parameters(), cfg["optimizer"]["max_grad_norm"]
                            )
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()
                        metrics.update_scene(
                            result, rollout, cfg["algorithm"]["ppo_clip_range"]
                        )
                        del result
                update += 1
                if runtime["deepspeed_stage"] and actor.global_steps != update:
                    raise RuntimeError(
                        f"optimizer update mismatch: engine={actor.global_steps}, trainer={update}"
                    )
                # Probe exactly the behavior chain after this optimizer update.
                with torch.no_grad(), accelerator.autocast():
                    for rollout in buffers:
                        stats = actor(
                            mode="transitions",
                            observation=rollout.observation,
                            rollout=rollout,
                        )
                        current = reduce_dimensions(
                            stats["elementwise_logprob"],
                            rollout.dimension_mask,
                            rollout.spec.reduction,
                        )
                        metrics.ratios(
                            "post_update_probe_ratio",
                            (current - rollout.old_logprob).exp()[
                                rollout.transition_mask
                            ],
                            cfg["algorithm"]["ppo_clip_range"],
                        )
                if old_fingerprints != [behavior_digest(r) for r in buffers]:
                    raise RuntimeError(
                        "behavior chain / old log-prob / advantages mutated"
                    )
                row = metrics.result()
                row.update(monitor.consume())
                row.update(parameter_probe.consume())
                row.update(
                    optimizer_norm_metrics(
                        actor,
                        runtime["deepspeed_stage"],
                        pre_norm,
                        cfg["optimizer"]["max_grad_norm"],
                    )
                )
                row.update(
                    update=update,
                    policy_version=version,
                    inner_epoch=inner,
                    rank=accelerator.process_index,
                    device={
                        "type": accelerator.device.type,
                        "index": accelerator.device.index,
                        "name": torch.cuda.get_device_name(accelerator.device),
                    },
                    behavior_sha256=old_fingerprints,
                    scene_tokens=[r.observation.tokens for r in buffers],
                    replay_tokens=[s["token"] for s in replay],
                    rollout_score_reference_seconds=rollout_seconds,
                    update_seconds=time.monotonic() - t1,
                    peak_gpu_gib=torch.cuda.max_memory_allocated() / 2**30,
                    loss_coefficients={
                        "grpo": 1.0,
                        "reference": cfg["algorithm"]["reference_kl_coefficient"],
                        "sft": cfg["retention"]["original_sft_coefficient"],
                    },
                    diagnostic_loss_scope=runtime.get("diagnostic_loss_scope"),
                    lr=[g["lr"] for g in optimizer.param_groups],
                    reward_errors=service.errors,
                    dtype_before_step=getattr(actor, "flow_pre_step_dtype", None),
                    activation_dtypes=dict(activation_dtypes.values),
                    dtype=dtype_inventory(
                        policy, actor if runtime["deepspeed_stage"] else None
                    ),
                )
                synchronized_call(
                    lambda: append_json(
                        output / f"training_rank{accelerator.process_index}.jsonl", row
                    ),
                    accelerator.device,
                )
                rank_rows = [None] * accelerator.num_processes
                if torch.distributed.is_initialized():
                    torch.distributed.all_gather_object(
                        rank_rows, {"row": row, "metrics": metrics.state}
                    )
                else:
                    rank_rows = [{"row": row, "metrics": metrics.state}]

                def write_global_log():
                    merged = Metrics()
                    for item in rank_rows:
                        merged.merge(Metrics(item["metrics"]))
                    global_row = merged.result()
                    global_row.update(
                        {
                            k: row[k]
                            for k in (
                                "update",
                                "policy_version",
                                "inner_epoch",
                                "loss_coefficients",
                                "lr",
                            )
                        }
                    )
                    for key in (
                        "pre_clip_grad_norm",
                        "clip_scale",
                        "post_clip_grad_norm",
                    ):
                        values = [r["row"][key] for r in rank_rows]
                        global_row[key] = (
                            values[0] if all(v == values[0] for v in values) else None
                        )
                    global_row["grad_norm_status"] = row["grad_norm_status"]
                    global_row["ranks"] = [r["row"] for r in rank_rows]
                    global_row["peak_gpu_gib_by_rank"] = [
                        r["row"]["peak_gpu_gib"] for r in rank_rows
                    ]
                    global_row["peak_gpu_gib_max"] = max(
                        global_row["peak_gpu_gib_by_rank"]
                    )
                    global_row["reward_errors"] = sum(
                        r["row"]["reward_errors"] for r in rank_rows
                    )
                    append_json(output / "training.jsonl", global_row)
                    accelerator.print(json.dumps(global_row))

                rank0_call(write_global_log, accelerator.device)
                complete_batch = inner + 1 == cfg["algorithm"]["inner_epochs"]
                if update % runtime["save_every"] == 0 or update >= maximum:
                    synchronized_call(check_immutable, accelerator.device)
                    saved_pending = (
                        None
                        if complete_batch
                        else {
                            "buffers": buffers,
                            "replay": replay,
                            "next_inner": inner + 1,
                        }
                    )
                    save_boundary(
                        accelerator,
                        actor,
                        cfg,
                        sft,
                        manifest,
                        provenance,
                        update,
                        version + int(complete_batch),
                        [train_stream, replay_stream],
                        pending=saved_pending,
                    )
                if update >= maximum:
                    break
            # Save diagnostic raw chains separately; never change the candidate set.
            if version < 2:
                synchronized_call(
                    lambda: torch.save(
                        buffers,
                        output
                        / f"rollout_rank{accelerator.process_index}_v{version}.pt",
                    ),
                    accelerator.device,
                )
            if complete_batch:
                version += 1

        synchronized_call(check_immutable, accelerator.device)
        failed = False
    except BaseException:
        try:
            if "buffers" in locals():
                torch.save(
                    buffers,
                    output
                    / f"failed_rollout_rank{accelerator.process_index}_v{version}.pt",
                )
        except Exception:
            pass
        raise
    finally:
        service.close(abort=failed)
        monitor.close()
        activation_dtypes.close()
        if communication_monitor is not None:
            communication_monitor.close()
    accelerator.end_training()


def append_json(path, value):
    with Path(path).open("a") as stream:
        stream.write(json.dumps(value) + "\n")


def behavior_digest(rollout):
    import hashlib

    h = hashlib.sha256()
    for tensor in (rollout.chain, rollout.old_elementwise_logprob, rollout.advantages):
        h.update(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def run(cfg, sft, resume=None):
    try:
        return _run(cfg, sft, resume)
    except BaseException as exc:
        import traceback

        # No collective/barrier during fatal cleanup. torchrun kills peers.
        path = Path(cfg["runtime"]["output_dir"] or ".")
        try:
            path.mkdir(parents=True, exist_ok=True)
            (
                path / f"failure_rank{os.getenv('RANK', '0')}_{time.time_ns()}.json"
            ).write_text(
                json.dumps(
                    {
                        "error": repr(exc),
                        "traceback": traceback.format_exc(),
                        "status": "FAIL",
                    },
                    indent=2,
                )
            )
        except OSError:
            pass
        raise


def initialize_run_directory(output, cfg, resume=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "training.jsonl").exists() and not resume:
        raise FileExistsError(
            "existing run: use explicit resume or a new output directory"
        )
    if resume:
        saved = output / "rl_config.json"
        if saved.exists() and config_hash(json.loads(saved.read_text())) != config_hash(
            cfg
        ):
            raise ValueError("output directory belongs to a different configuration")
    (output / "rl_config.json").write_text(json.dumps(cfg, indent=2))
    from .audit import capture_source_environment

    capture_source_environment(output, cfg)
