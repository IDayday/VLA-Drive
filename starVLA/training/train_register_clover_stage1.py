#!/usr/bin/env python3
"""CLOVER Stage 1: pseudo-expert Register64 plus online PDMS value learning."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from omegaconf import OmegaConf

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from starVLA.dataloader import build_dataloader
from starVLA.model.framework import build_framework
from starVLA.model.modules.planning.trajectory_codec import TrajectoryCodec
from starVLA.model.modules.register_planner.checkpoint import (
    save_register_generator_checkpoint,
    save_stage_component_checkpoint,
    sha256_file,
)
from starVLA.training.clover_pseudo_experts import CloverPseudoExpertStore
from starVLA.training.config_loader import load_training_config
from starVLA.training.navsim_metric_supervisor import DynamicMetricSupervisor
from starVLA.training.register_stage_utils import (
    EarlyStopping,
    TrainingProgress,
    atomic_json,
    cosine_schedule,
    optimizer_steps_per_epoch,
)
from starVLA.training.train_register_generator import (
    FirstBackwardGradientGate,
    _component_metadata,
    _fixed_validation_loader,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--profile-steps",
        type=int,
        default=0,
        help="Run exactly this many optimizer steps, report performance, and exit.",
    )
    return parser.parse_args()


def _optimizer(model, config) -> torch.optim.AdamW:
    learning_rates = config.optimizer.learning_rates
    modules = []
    if "qwen_visual" in learning_rates:
        modules.append(("qwen_visual", model.qwen_visual))
    modules.extend(
        [
            ("qwen_vl_interface", model.qwen_vl_interface),
            ("scene_encoder", model.scene_encoder),
            ("register_generator", model.register_generator),
            ("action_input_model", model.action_input_model),
            ("drivor_scorer", model.drivor_scorer),
        ]
    )
    groups = []
    seen: set[int] = set()
    for name, module in modules:
        parameters = [
            parameter
            for parameter in module.parameters()
            if parameter.requires_grad and id(parameter) not in seen
        ]
        for parameter in parameters:
            seen.add(id(parameter))
        if parameters:
            if name not in learning_rates:
                raise KeyError(f"optimizer.learning_rates has no {name}")
            groups.append(
                {
                    "name": name,
                    "params": parameters,
                    "lr": float(learning_rates[name]),
                }
            )
    expected = {
        id(parameter) for parameter in model.parameters() if parameter.requires_grad
    }
    if seen != expected:
        raise RuntimeError("CLOVER Stage-1 optimizer parameter coverage differs")
    return torch.optim.AdamW(
        groups,
        betas=tuple(float(value) for value in config.optimizer.betas),
        weight_decay=float(config.optimizer.weight_decay),
        eps=float(config.optimizer.get("eps", 1.0e-8)),
        fused=bool(config.optimizer.get("fused", True)) and torch.cuda.is_available(),
    )


def _optimizer_group_grad_norms(optimizer) -> dict[str, torch.Tensor]:
    """Compute un-clipped L2 gradient norms for named optimizer groups."""

    result: dict[str, torch.Tensor] = {}
    for index, group in enumerate(optimizer.param_groups):
        name = str(group.get("name", f"group_{index}"))
        total = None
        for parameter in group["params"]:
            if parameter.grad is None:
                continue
            value = parameter.grad.detach().float().square().sum()
            total = value if total is None else total + value
        if total is None:
            reference = group["params"][0]
            total = torch.zeros((), device=reference.device, dtype=torch.float32)
        result[name] = total.sqrt()
    return result


def _assert_proposal_sanitization(metrics: dict[str, torch.Tensor]) -> None:
    """Stop numerical collapse while allowing rare bounded XY clipping."""

    nonfinite = float(metrics.get("proposal_nonfinite_rate", torch.tensor(0.0)))
    xy_clamped = float(metrics.get("proposal_xy_clamped_rate", torch.tensor(0.0)))
    if nonfinite > 1.0e-6:
        raise RuntimeError(
            f"Register proposal non-finite rate is too high: {nonfinite:.8f}"
        )
    if xy_clamped > 0.05:
        raise RuntimeError(
            f"Register proposal XY clamp rate is too high: {xy_clamped:.8f}"
        )


def _ground_truth(examples, device: torch.device) -> torch.Tensor:
    actions = torch.as_tensor(
        np.asarray([example["action"] for example in examples]),
        device=device,
        dtype=torch.float32,
    )
    return TrajectoryCodec().flow_to_navsim(actions)


@torch.no_grad()
def _evaluate(model, loader, supervisor, accelerator: Accelerator) -> dict[str, float]:
    model.eval()
    names = ("oracle_pdms_64", "min_ade_64", "proposal0_ade")
    totals = torch.zeros(len(names) + 1, device=accelerator.device)
    unwrapped = accelerator.unwrap_model(model)
    scorer = unwrapped.drivor_scorer
    calibrate = scorer.selection_mode == "calibrated_hybrid"
    alphas = torch.linspace(0.0, 1.0, 21, device=accelerator.device)
    selector_totals = torch.zeros(
        (len(alphas) if calibrate else 1) + 1,
        device=accelerator.device,
    )
    for examples in loader:
        with accelerator.autocast():
            output = model(
                examples,
                clover_supervisor=supervisor,
                compute_loss=False,
            )
        proposals = output["generator_output"].proposals.float()
        ground_truth = output["ground_truth"].float()
        ade = torch.linalg.vector_norm(
            proposals[..., :2] - ground_truth[:, None, :, :2],
            ord=2,
            dim=-1,
        ).mean(dim=-1)
        values = {
            "oracle_pdms_64": output["metrics"]["oracle_pdms_64"],
            "min_ade_64": ade.amin(dim=1).mean(),
            "proposal0_ade": ade[:, 0].mean(),
        }
        count = float(len(examples))
        totals[:-1] += torch.stack([values[name].float() for name in names]) * count
        totals[-1] += count
        true_score = output["metric_targets"]["aggregate_score"].float()
        scorer_output = output["scorer_output"]
        if calibrate:
            if scorer_output.aggregate_logit is None or scorer_output.formula_score is None:
                raise RuntimeError("CLOVER hybrid calibration lacks scorer branches")
            direct = scorer._scene_standardize(
                scorer_output.aggregate_logit.float()
            )
            structured = scorer._scene_standardize(
                scorer_output.formula_score.float()
            )
            blended = (
                (1.0 - alphas[:, None, None]) * direct[None]
                + alphas[:, None, None] * structured[None]
            )
            selected = blended.argmax(dim=-1)
            selected_true = torch.gather(
                true_score[None].expand(len(alphas), -1, -1),
                2,
                selected[..., None],
            ).squeeze(-1)
            selector_totals[:-1] += selected_true.sum(dim=1)
        else:
            selector_totals[0] += (
                output["metrics"]["selected_true_pdms"].float() * count
            )
        selector_totals[-1] += count
    totals = accelerator.reduce(totals, reduction="sum")
    selector_totals = accelerator.reduce(selector_totals, reduction="sum")
    count = totals[-1].clamp_min(1.0)
    selector_count = selector_totals[-1].clamp_min(1.0)
    selector_means = selector_totals[:-1] / selector_count
    if calibrate:
        best_index = int(selector_means.argmax().item())
        best_alpha = float(alphas[best_index].item())
        scorer.set_selection_alpha(best_alpha)
    else:
        best_index = 0
        best_alpha = float(scorer.selection_alpha.item())
    selected = float(selector_means[best_index])
    oracle = float(totals[0] / count)
    model.train()
    result = {
        name: float(totals[index] / count) for index, name in enumerate(names)
    }
    result.update(
        selected_true_pdms=selected,
        scorer_regret=oracle - selected,
        selector_alpha=best_alpha,
    )
    if calibrate:
        result.update(
            selector_direct_pdms=float(selector_means[0]),
            selector_structured_pdms=float(selector_means[-1]),
        )
    return result


def _scorer_state(full_state: dict[str, torch.Tensor], module) -> dict[str, torch.Tensor]:
    prefix = "drivor_scorer."
    expected = set(module.state_dict())
    result = {
        name[len(prefix) :]: value
        for name, value in full_state.items()
        if name.startswith(prefix) and name[len(prefix) :] in expected
    }
    if set(result) != expected:
        raise RuntimeError("gathered CLOVER scorer state is incomplete")
    return result


def _save_components(
    *,
    output_dir: Path,
    stem: str,
    model,
    config,
    full_state: dict[str, torch.Tensor],
    validation: dict[str, float],
    epoch: int,
    completed_steps: int,
    pseudo_store: CloverPseudoExpertStore,
) -> tuple[Path, Path]:
    generator_path = output_dir / f"{stem}_generator.pt"
    metadata = _component_metadata(model, config)
    metadata.update(
        epoch=epoch,
        completed_steps=completed_steps,
        validation=validation,
        training_stage="clover_stage1",
        label_protocol=str(config.metric_supervisor.protocol),
        pseudo_expert_sha256=pseudo_store.sha256,
    )
    save_register_generator_checkpoint(
        generator_path,
        qwen_vl_interface=model.qwen_vl_interface,
        action_input_model=model.action_input_model,
        scene_encoder=model.scene_encoder,
        register_generator=model.register_generator,
        metadata=metadata,
        full_model_state_dict=full_state,
    )
    scorer_path = output_dir / f"{stem}_scorer.pt"
    scorer_config = config.framework.drivor_scorer
    save_stage_component_checkpoint(
        scorer_path,
        stage="clover_stage1_scorer",
        module=model.drivor_scorer,
        state_dict=_scorer_state(full_state, model.drivor_scorer),
        metadata={
            "proposal_num": 64,
            "scene_dim": int(scorer_config.scene_dim),
            "model_dim": int(scorer_config.model_dim),
            "decoder_layers": int(scorer_config.num_layers),
            "decoder_heads": int(scorer_config.num_heads),
            "aggregate_weights": dict(model.drivor_scorer.aggregate_weights),
            "aggregate_head": bool(model.drivor_scorer.aggregate_head_enabled),
            "selection_mode": str(model.drivor_scorer.selection_mode),
            "selection_alpha": float(model.drivor_scorer.selection_alpha.item()),
            "aggregate_temperature": float(
                model.drivor_scorer.aggregate_temperature
            ),
            "label_protocol": str(config.metric_supervisor.protocol),
            "loss_type": "pdms_value",
            "training_profile": "clover_stage1_joint_pdms_v1",
            "generator_checkpoint_sha256": sha256_file(generator_path),
            "pseudo_expert_sha256": pseudo_store.sha256,
            "epoch": epoch,
            "completed_steps": completed_steps,
            "validation": validation,
        },
    )
    return generator_path, scorer_path


def main() -> None:
    args = _parse_args()
    if args.profile_steps < 0:
        raise ValueError("profile steps must be non-negative")
    config = load_training_config(args.config)
    if str(config.framework.name) != "QwenRegisterClover":
        raise ValueError("CLOVER Stage-1 requires framework=QwenRegisterClover")
    if str(config.metric_supervisor.protocol) != "navsim_v1_1_pdms_two_way":
        raise ValueError("the PDMS route requires NAVSIM-v1.1 training labels")
    trainer = config.trainer
    epochs = int(trainer.get("max_epochs", 25))
    if not 1 <= epochs <= 25:
        raise ValueError("CLOVER Stage-1 epochs must lie in [1,25]")
    accumulation = int(trainer.get("gradient_accumulation_steps", 1))
    accelerator = Accelerator(
        gradient_accumulation_steps=accumulation,
        mixed_precision=str(config.get("precision", "bf16")),
    )
    set_seed(int(config.get("seed", 2)), device_specific=True)
    global_batch = int(trainer.get("global_batch_size", 32))
    denominator = accelerator.num_processes * accumulation
    if global_batch % denominator:
        raise ValueError("global batch must divide world size * accumulation")
    per_device_batch = global_batch // denominator
    config.datasets.vla_data.per_device_batch_size = per_device_batch
    output_dir = Path(str(config.run_root_dir)) / str(config.run_id)
    config.output_dir = str(output_dir)
    train_loader = build_dataloader(
        cfg=config, dataset_py=config.datasets.vla_data.dataset_py
    )
    val_loader = _fixed_validation_loader(config, per_device_batch)
    model = build_framework(config)
    if model.__class__.__name__ != "QwenRegisterClover":
        raise RuntimeError("framework registry returned the wrong CLOVER class")
    model.assert_qwen_trainability()
    pseudo_config = config.pseudo_experts
    pseudo_store = CloverPseudoExpertStore(
        str(pseudo_config.path),
        top_k=int(pseudo_config.get("top_k", 8)),
        score_threshold=float(pseudo_config.get("score_threshold", 0.8)),
        fps_min_distance=float(pseudo_config.get("fps_min_distance", 0.05)),
        gt_coverage_distance=float(
            pseudo_config.get("gt_coverage_distance", 0.5)
        ),
    )
    optimizer = _optimizer(model, config)
    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )
    supervisor = DynamicMetricSupervisor(
        config.metric_supervisor,
        rank=accelerator.process_index,
        world_size=accelerator.num_processes,
    )
    gradient_gate = (
        FirstBackwardGradientGate(model)
        if bool(trainer.get("gradient_gate", {}).get("enabled", True))
        else None
    )
    steps_per_epoch = optimizer_steps_per_epoch(len(train_loader), accumulation)
    total_steps = steps_per_epoch * epochs
    scheduler = cosine_schedule(
        optimizer,
        total_steps=total_steps,
        warmup_ratio=float(config.scheduler.get("warmup_ratio", 0.05)),
    )
    progress = TrainingProgress()
    accelerator.register_for_checkpointing(scheduler)
    accelerator.register_for_checkpointing(progress)
    resume_from = trainer.get("resume_from")
    if args.profile_steps and resume_from:
        raise ValueError("Stage-1 profiling cannot resume a training checkpoint")
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "checkpoints").mkdir(exist_ok=True)
        OmegaConf.save(config, output_dir / "config.yaml")
        metrics_path = output_dir / "metrics.jsonl"
        if not resume_from or not metrics_path.is_file():
            metrics_path.write_text("", encoding="utf-8")
        accelerator.unwrap_model(model).log_architecture_summary(
            type("PrintLogger", (), {"info": staticmethod(lambda msg, *a: print(msg % a))})
        )
        print(
            f"CLOVER Stage 1: pseudo_scenes={len(pseudo_store)} "
            f"global_batch={global_batch} steps/epoch={steps_per_epoch}"
        )
    early_config = trainer.get("early_stopping", {})
    early = EarlyStopping(
        patience=int(early_config.get("patience", 5)), mode="max"
    )
    if resume_from:
        accelerator.load_state(str(resume_from))
        early.best = progress.early_best
        early.bad_epochs = progress.early_bad_epochs
    save_epochs = {
        int(value)
        for value in trainer.get("save_epochs", [5, 10, 15, 20, 25])
    }
    completed_steps = progress.completed_steps
    log_every_steps = max(1, int(trainer.get("log_every_steps", 20)))
    diagnostics_steps = max(0, int(trainer.get("gradient_diagnostics_steps", 200)))
    running_names = (
        "loss",
        "trajectory_gt",
        "pseudo_expert_coverage",
        "scorer",
        "scorer_submetric",
        "scorer_aggregate",
        "scorer_listwise",
        "scorer_pairwise",
        "selected_true_pdms",
        "oracle_pdms_64",
        "scorer_regret",
        "proposal_nonfinite_rate",
        "proposal_xy_clamped_rate",
        "proposal_heading_wrapped_rate",
    )
    should_stop = False
    last_epoch = progress.epoch
    optimizer.zero_grad(set_to_none=True)
    profile_reached = False
    profile_samples = 0
    profile_start = time.perf_counter()
    if args.profile_steps and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    try:
        for epoch in range(progress.epoch + 1, epochs + 1):
            last_epoch = epoch
            model.train()
            start = time.perf_counter()
            samples = 0
            running = torch.zeros(len(running_names) + 1, device=accelerator.device)
            latest_grad_norms: dict[str, torch.Tensor] = {}
            for examples in train_loader:
                gt = _ground_truth(examples, accelerator.device)
                pseudo, pseudo_mask = pseudo_store.batch(
                    [str(example["token"]) for example in examples],
                    gt,
                    device=accelerator.device,
                    dtype=torch.float32,
                    require_all_tokens=bool(
                        pseudo_config.get("require_all_tokens", True)
                    ),
                )
                with accelerator.accumulate(model):
                    with accelerator.autocast():
                        output = model(
                            examples,
                            clover_supervisor=supervisor,
                            pseudo_experts=pseudo,
                            pseudo_expert_mask=pseudo_mask,
                        )
                        loss = output["loss"]
                        _assert_proposal_sanitization(output["metrics"])
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        if gradient_gate is not None:
                            gradient_gate.assert_complete(accelerator)
                            gradient_gate = None
                        next_step = completed_steps + 1
                        if (
                            next_step <= diagnostics_steps
                            and next_step % log_every_steps == 0
                        ):
                            latest_grad_norms = _optimizer_group_grad_norms(optimizer)
                        accelerator.clip_grad_norm_(
                            model.parameters(),
                            float(trainer.get("gradient_clip", 1.0)),
                        )
                    optimizer.step()
                    if accelerator.sync_gradients:
                        scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                samples += len(examples) * accelerator.num_processes
                profile_samples += len(examples) * accelerator.num_processes
                if accelerator.sync_gradients:
                    completed_steps += 1
                    batch_count = float(len(examples))
                    scalar_values = {
                        "loss": loss,
                        **output["losses"],
                        **output["metrics"],
                    }
                    running[: len(running_names)] += torch.stack(
                        [scalar_values[name].detach().float() for name in running_names]
                    ) * batch_count
                    running[-1] += batch_count
                    if completed_steps % log_every_steps == 0:
                        reduced = accelerator.reduce(running, reduction="sum")
                        count = reduced[-1].clamp_min(1.0)
                        means = {
                            name: float(reduced[index] / count)
                            for index, name in enumerate(running_names)
                        }
                        grad_names = tuple(latest_grad_norms)
                        if grad_names:
                            grad_values = accelerator.reduce(
                                torch.stack(
                                    [latest_grad_norms[name] for name in grad_names]
                                ),
                                reduction="mean",
                            )
                            grad_norms = {
                                name: float(grad_values[index])
                                for index, name in enumerate(grad_names)
                            }
                        else:
                            grad_norms = {}
                        if accelerator.is_main_process:
                            elapsed = time.perf_counter() - start
                            print(
                                "CLOVER Stage-1 progress "
                                f"epoch={epoch}/{epochs} step={completed_steps} "
                                f"loss={means['loss']:.5f} "
                                f"gt={means['trajectory_gt']:.5f} "
                                f"pseudo={means['pseudo_expert_coverage']:.5f} "
                                f"scorer={means['scorer']:.5f} "
                                f"direct={means['scorer_aggregate']:.5f} "
                                f"listwise={means['scorer_listwise']:.5f} "
                                f"pairwise={means['scorer_pairwise']:.5f} "
                                f"selected_pdms={means['selected_true_pdms']:.5f} "
                                f"oracle64={means['oracle_pdms_64']:.5f} "
                                f"regret={means['scorer_regret']:.5f} "
                                f"clamp={means['proposal_xy_clamped_rate']:.7f} "
                                f"grad_norms={json.dumps(grad_norms, sort_keys=True)} "
                                f"samples_per_second={samples / max(elapsed, 1e-9):.3f}",
                                flush=True,
                            )
                            with (output_dir / "metrics.jsonl").open(
                                "a", encoding="utf-8"
                            ) as stream:
                                stream.write(
                                    json.dumps(
                                        {
                                            "type": "optimizer_step",
                                            "epoch": epoch,
                                            "optimizer_step": completed_steps,
                                            **means,
                                            "gradient_norms": grad_norms,
                                        },
                                        sort_keys=True,
                                    )
                                    + "\n"
                                )
                        running.zero_()
                        latest_grad_norms = {}
                    if args.profile_steps and completed_steps >= args.profile_steps:
                        profile_reached = True
                        break
            if args.profile_steps:
                if profile_reached:
                    break
                continue
            validation = _evaluate(model, val_loader, supervisor, accelerator)
            epoch_seconds = time.perf_counter() - start
            selected = float(validation["selected_true_pdms"])
            if not math.isfinite(selected):
                raise RuntimeError("validation selected PDMS is not finite")
            improved, exhausted = early.update(selected)
            should_stop = bool(early_config.get("enabled", True)) and exhausted
            progress.epoch = epoch
            progress.completed_steps = completed_steps
            progress.early_best = early.best
            progress.early_bad_epochs = early.bad_epochs
            save_epoch = epoch in save_epochs or epoch == epochs or should_stop
            accelerator.wait_for_everyone()
            if save_epoch or improved:
                accelerator.save_state(
                    str(output_dir / "checkpoints" / f"epoch_{epoch:02d}")
                )
                full_state = accelerator.get_state_dict(model)
            else:
                full_state = None
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                unwrapped = accelerator.unwrap_model(model)
                if save_epoch:
                    _save_components(
                        output_dir=output_dir,
                        stem=f"epoch_{epoch:02d}",
                        model=unwrapped,
                        config=config,
                        full_state=full_state,
                        validation=validation,
                        epoch=epoch,
                        completed_steps=completed_steps,
                        pseudo_store=pseudo_store,
                    )
                if improved:
                    _save_components(
                        output_dir=output_dir,
                        stem="best_pdms",
                        model=unwrapped,
                        config=config,
                        full_state=full_state,
                        validation=validation,
                        epoch=epoch,
                        completed_steps=completed_steps,
                        pseudo_store=pseudo_store,
                    )
                with (output_dir / "metrics.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(
                        json.dumps(
                            {
                                "epoch": epoch,
                                "optimizer_step": completed_steps,
                                "samples": samples,
                                "epoch_seconds": epoch_seconds,
                                "sec_per_step": epoch_seconds / max(steps_per_epoch, 1),
                                "samples_per_second": samples / max(epoch_seconds, 1e-9),
                                **validation,
                            },
                            sort_keys=True,
                        )
                        + "\n"
                    )
                print(f"CLOVER Stage-1 epoch {epoch}: {validation}")
            stop = accelerator.reduce(
                torch.tensor(int(should_stop), device=accelerator.device),
                reduction="max",
            )
            if bool(stop.item()):
                break
    finally:
        supervisor.close()
    accelerator.wait_for_everyone()
    if args.profile_steps:
        if not profile_reached:
            raise RuntimeError(
                "profile step target exceeds the configured Stage-1 epoch budget"
            )
        wall_time = time.perf_counter() - profile_start
        peak_memory = (
            torch.cuda.max_memory_allocated(accelerator.device)
            if torch.cuda.is_available()
            else 0
        )
        profile_values = accelerator.reduce(
            torch.tensor(
                [wall_time, float(peak_memory)],
                device=accelerator.device,
                dtype=torch.float32,
            ),
            reduction="max",
        )
        if accelerator.is_main_process:
            report = {
                "schema_version": 1,
                "status": "profile_complete",
                "optimizer_steps": int(completed_steps),
                "global_samples": int(profile_samples),
                "wall_time_seconds": float(profile_values[0]),
                "seconds_per_step": float(profile_values[0])
                / max(completed_steps, 1),
                "samples_per_second": float(profile_samples)
                / max(float(profile_values[0]), 1.0e-9),
                "peak_memory_bytes_per_rank": int(profile_values[1]),
                "formal_training_complete": False,
            }
            atomic_json(output_dir / "profile_complete.json", report)
            print(f"CLOVER Stage-1 production profile complete: {report}")
        return
    if accelerator.is_main_process:
        generator = output_dir / "best_pdms_generator.pt"
        scorer = output_dir / "best_pdms_scorer.pt"
        if not generator.is_file() or not scorer.is_file():
            raise RuntimeError("CLOVER Stage-1 did not produce paired best checkpoints")
        atomic_json(
            output_dir / "training_complete.json",
            {
                "schema_version": 1,
                "status": "complete",
                "stage": "clover_stage1",
                "completed_epochs": int(last_epoch),
                "completed_steps": int(completed_steps),
                "selection_metric": "val_selected_true_pdms",
                "selection_metric_value": float(early.best),
                "generator_checkpoint": generator.name,
                "generator_sha256": sha256_file(generator),
                "scorer_checkpoint": scorer.name,
                "scorer_sha256": sha256_file(scorer),
                "pseudo_expert_sha256": pseudo_store.sha256,
            },
        )


if __name__ == "__main__":
    main()
