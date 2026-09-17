"""Synchronous full-parameter Accelerate/DeepSpeed Flow-GRPO training."""
from pathlib import Path
import csv
import json
import os
import time
import torch
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.utils import (
    DistributedDataParallelKwargs,
    GradientAccumulationPlugin,
    set_seed,
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
from .distributed import synchronized_call
from .loading import load_policy, enable_checkpointing
from .model import FlowGRPOActor, make_reference
from .audit import source_fingerprints
from .observation import prepare_policy_observation
from .rollout import SamplingSpec, evaluate_transitions
from .math import group_advantages
from .reward import RewardService, VERSION
from .monitor import GradientMonitor, CheckedDeepSpeedBackward, ParameterProbe


def make_accelerator(cfg):
    runtime = cfg["runtime"]
    accum = runtime["accumulation_steps"]
    plugin = None
    if runtime["deepspeed_stage"]:
        ds = {
            "bf16": {"enabled": True},
            "fp16": {"enabled": False},
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
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
    )


def run(cfg, sft, resume=None):
    runtime = cfg["runtime"]
    maximum = runtime["max_updates"]
    if maximum is None or maximum <= 0 or not runtime["output_dir"]:
        raise ValueError("MAX_UPDATES and OUTPUT_DIR must be explicit")
    if maximum % cfg["algorithm"]["inner_epochs"]:
        raise ValueError("max_updates must end at a complete inner-epoch boundary")
    accelerator = make_accelerator(cfg)
    if os.getenv("FLASH_ATTENTION_DETERMINISTIC", "0") == "1":
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    set_seed(runtime["seed"], device_specific=True)
    output = Path(runtime["output_dir"])
    if accelerator.is_main_process:
        output.mkdir(parents=True, exist_ok=True)
        if (output / "training.jsonl").exists() and not resume:
            raise FileExistsError(
                "existing run: use explicit resume or a new output directory"
            )
        (output / "rl_config.json").write_text(json.dumps(cfg, indent=2))
        from .audit import capture_source_environment

        capture_source_environment(output, cfg)
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
    if accelerator.is_main_process:
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
    if runtime["deepspeed_stage"]:
        accelerator.deepspeed_engine_wrapped = CheckedDeepSpeedBackward(
            actor, monitor, accelerator.device
        )
    actor.eval()
    policy = accelerator.unwrap_model(actor).policy
    if runtime["deepspeed_stage"]:
        reference.to(dtype=next(policy.action_model.parameters()).dtype)
    if not runtime["reference_offload"]:
        reference.to(accelerator.device)
    sftsha = policy._flow_source_sha256
    if sftsha != cfg["checkpoint_contract"]["sha256"]:
        raise ValueError("SFT checkpoint SHA does not match audited contract")
    provenance = dict(
        implementation_sha256=digest(source_fingerprints()),
        sft_sha256=sftsha,
        reference_sha256=sftsha,
        normalization=digest(
            {
                "ver_1225": int(sft.ver_1225),
                "act_norm": int(sft.datasets.vla_data.act_norm),
            }
        ),
        reward_version=VERSION,
        config_hash=config_hash(cfg),
        reference_lock=digest(json.loads(Path("reference_lock.json").read_text())),
    )
    service = RewardService(
        Path("navsim").resolve(),
        cfg["paths"]["metric_cache"],
        "train",
        train,
        workers=runtime["reward_workers"],
        timeout=runtime["reward_timeout"],
        cache_dir=output / "reward_cache",
    )
    provenance["reward_config_hash"] = digest(service.metadata)
    spec = SamplingSpec(
        group_size=cfg["sampling"]["group_size"],
        num_steps=cfg["sampling"]["num_steps"],
        noise_level=cfg["sampling"]["noise_level"],
        reduction=cfg["algorithm"]["logprob_reduction"],
        candidate_chunk_size=cfg["sampling"]["candidate_chunk_size"],
        transition_chunk_size=cfg["sampling"]["transition_chunk_size"],
    )
    update = version = 0
    if resume:
        check_manifest(
            json.loads((Path(resume) / "sft_parameter_manifest.json").read_text()),
            manifest,
        )
        update, version = resume_boundary(
            resume, accelerator, actor, cfg, provenance, [train_stream, replay_stream]
        )
    from infer import deal_action_1225

    accelerator.print(
        f'Flow-GRPO full SFT: {manifest["trainable_numel"]:,} trainable parameters; start={update}, stop={maximum}'
    )
    optimizer.zero_grad()
    parameter_probe = ParameterProbe(policy)
    from .contracts import tensor_hashes

    immutable_before = synchronized_call(
        lambda: {
            "reference": tensor_hashes(reference),
            "frozen": tensor_hashes(policy, lambda n, p: not p.requires_grad),
        },
        accelerator.device,
    )
    try:
        while update < maximum:
            t0 = time.monotonic()
            buffers = []
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
                    rollout = synchronized_call(
                        lambda: actor(
                            mode="rollout",
                            observation=observation,
                            spec=spec,
                            policy_version=version,
                            seed=seed,
                            provenance=provenance,
                        ),
                        accelerator.device,
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
                rollout.advantages = group_advantages(
                    rewards,
                    epsilon=cfg["algorithm"]["advantage_epsilon"],
                    clip=cfg["algorithm"]["advantage_clip"],
                )
                if runtime["reference_offload"]:
                    reference.to(accelerator.device)
                with torch.no_grad(), accelerator.autocast():
                    stats = synchronized_call(
                        lambda: evaluate_transitions(reference, observation, rollout),
                        accelerator.device,
                    )
                rollout.reference_mean = stats["mean"].detach()
                rollout.reference_std = stats["std"].detach()
                if runtime["reference_offload"]:
                    reference.to("cpu")
                buffers.append(rollout)
            rollout_seconds = time.monotonic() - t0
            for inner in range(cfg["algorithm"]["inner_epochs"]):
                rows = []
                t1 = time.monotonic()
                for i, rollout in enumerate(buffers):
                    rollout.validate(version)
                    with accelerator.accumulate(actor):
                        batch = to_device([replay[i]], accelerator.device)
                        with accelerator.autocast():
                            result = synchronized_call(
                                lambda: actor(
                                    mode="update", rollout=rollout, replay=batch
                                ),
                                accelerator.device,
                            )
                        # One equally weighted scene per microbatch; Accelerate/DS
                        # owns the 1/accum scaling, never G*K copies of replay.
                        accelerator.backward(result["loss"])
                        if not runtime["deepspeed_stage"]:
                            monitor.assert_finite_collective(accelerator.device)
                        if (
                            accelerator.sync_gradients
                            and not runtime["deepspeed_stage"]
                        ):
                            accelerator.clip_grad_norm_(
                                actor.parameters(), cfg["optimizer"]["max_grad_norm"]
                            )
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()
                        row = {
                            key: float(result[key].detach())
                            for key in ("loss", "grpo", "reference", "sft")
                        }
                        row.update(
                            ratio_mean=float(result["ratio"].mean()),
                            ratio_min=float(result["ratio"].min()),
                            ratio_max=float(result["ratio"].max()),
                            clip_fraction=float(
                                (
                                    (result["ratio"] - 1).abs()
                                    > cfg["algorithm"]["ppo_clip_range"]
                                )
                                .float()
                                .mean()
                            ),
                            old_policy_drift=float(
                                result["logratio"].square().mean() / 2
                            ),
                            reward=float(rollout.rewards.mean()),
                            reward_zero_fraction=float(
                                (rollout.rewards == 0).float().mean()
                            ),
                            all_zero_groups=float(
                                (rollout.rewards == 0).all(1).float().mean()
                            ),
                            valid_group_fraction=1.0,
                            all_equal_groups=float(
                                (rollout.rewards.std(1, unbiased=False) == 0)
                                .float()
                                .mean()
                            ),
                            advantage_std=float(rollout.advantages.std(unbiased=False)),
                            components={
                                k: float(v.detach())
                                for k, v in result["components"].items()
                            },
                        )
                        rows.append(row)
                update += 1
                if runtime["deepspeed_stage"] and actor.global_steps != update:
                    raise RuntimeError(
                        f"optimizer update mismatch: engine={actor.global_steps}, trainer={update}"
                    )
                row = {
                    key: sum(r[key] for r in rows) / len(rows)
                    for key in rows[0]
                    if key != "components"
                }
                row.update(monitor.consume())
                row.update(parameter_probe.consume())
                row.update(
                    update=update,
                    policy_version=version,
                    rank=accelerator.process_index,
                    inner_epoch=inner,
                    scene_tokens=[r.observation.tokens for r in buffers],
                    replay_tokens=[s["token"] for s in replay],
                    original_sft_components={
                        k: sum(r["components"][k] for r in rows) / len(rows)
                        for k in rows[0]["components"]
                    },
                    rollout_score_reference_seconds=rollout_seconds,
                    update_seconds=time.monotonic() - t1,
                    peak_gpu_gib=torch.cuda.max_memory_allocated() / 2**30,
                    lr=[g["lr"] for g in optimizer.param_groups],
                    reward_errors=service.errors,
                    reward_worker_count=runtime["reward_workers"],
                    reward_pending_jobs=0,
                )
                metric_records = [
                    record
                    for buffer in buffers
                    for scene_records in buffer.score_records
                    for record in scene_records
                ]
                row["reward_components"] = {
                    key: sum(
                        record.metrics[key]
                        for record in metric_records
                        if record.metrics.get(key) is not None
                    )
                    / sum(
                        record.metrics.get(key) is not None for record in metric_records
                    )
                    if any(
                        record.metrics.get(key) is not None for record in metric_records
                    )
                    else None
                    for key in metric_records[0].metrics
                }
                with (output / f"training_rank{accelerator.process_index}.jsonl").open(
                    "a"
                ) as stream:
                    stream.write(json.dumps(row) + "\n")
                if torch.distributed.is_initialized():
                    rank_rows = [None] * accelerator.num_processes
                    torch.distributed.all_gather_object(rank_rows, row)
                else:
                    rank_rows = [row]
                if accelerator.is_main_process:
                    row = dict(row)
                    for key, value in row.items():
                        if isinstance(value, float):
                            row[key] = sum(r[key] for r in rank_rows) / len(rank_rows)
                    row["global_scene_tokens"] = [r["scene_tokens"] for r in rank_rows]
                    for field in ("reward_components", "original_sft_components"):
                        row[field] = {
                            key: sum(
                                r[field][key]
                                for r in rank_rows
                                if r[field][key] is not None
                            )
                            / sum(r[field][key] is not None for r in rank_rows)
                            if any(r[field][key] is not None for r in rank_rows)
                            else None
                            for key in row[field]
                        }
                    row["peak_gpu_gib"] = max(r["peak_gpu_gib"] for r in rank_rows)
                    with (output / "training.jsonl").open("a") as stream:
                        stream.write(json.dumps(row) + "\n")
                    scalar = {
                        k: v for k, v in row.items() if isinstance(v, (int, float, str))
                    }
                    csv_path = output / "training.csv"
                    if csv_path.exists():
                        with csv_path.open() as previous:
                            if next(csv.reader(previous)) != list(scalar):
                                csv_path = (
                                    output / f"training_{digest(list(scalar))[:8]}.csv"
                                )
                    with csv_path.open("a") as stream:
                        writer = csv.DictWriter(stream, fieldnames=scalar.keys())
                        if stream.tell() == 0:
                            writer.writeheader()
                        writer.writerow(scalar)
                    accelerator.print(json.dumps(row))
            # Save diagnostic raw chains separately; never change the candidate set.
            if version < 2:
                torch.save(
                    buffers,
                    output / f"rollout_rank{accelerator.process_index}_v{version}.pt",
                )
            version += 1
            buffers.clear()
            if update % runtime["save_every"] == 0 or update == maximum:
                save_boundary(
                    accelerator,
                    actor,
                    cfg,
                    sft,
                    manifest,
                    provenance,
                    update,
                    version,
                    [train_stream, replay_stream],
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
                raise RuntimeError(
                    "reference/frozen parameters changed during training"
                )

        synchronized_call(check_immutable, accelerator.device)
    except BaseException as exc:
        if "buffers" in locals():
            torch.save(
                buffers,
                output
                / f"failed_rollout_rank{accelerator.process_index}_v{version}.pt",
            )
        (output / f"failure_rank{accelerator.process_index}.json").write_text(
            json.dumps(dict(update=update, version=version, error=repr(exc)))
        )
        raise
    finally:
        service.close()
        monitor.close()
    accelerator.end_training()
