#!/usr/bin/env python3
"""Stage G: train Qwen + global Q-Former + deterministic Register generator."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from starVLA.dataloader import build_dataloader
from starVLA.model.framework import build_framework
from starVLA.model.modules.register_planner.checkpoint import (
    save_register_generator_checkpoint,
    sha256_file,
    stable_config_hash,
)
from starVLA.training.config_loader import load_training_config
from starVLA.training.register_stage_utils import (
    EarlyStopping,
    TrainingProgress,
    atomic_json,
    cosine_schedule,
    optimizer_steps_per_epoch,
)


def _repository_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        cwd=Path(__file__).resolve().parents[2],
    ).stdout.strip()


def build_generator_optimizer(model, config) -> torch.optim.AdamW:
    """Create explicit, non-overlapping Stage-G learning-rate groups."""

    learning_rates = config.optimizer.learning_rates
    modules = []
    if "qwen_visual" in learning_rates:
        modules.append(("qwen_visual", model.qwen_visual))
    modules.extend(
        (
            ("qwen_vl_interface", model.qwen_vl_interface),
            ("scene_encoder", model.scene_encoder),
            ("register_generator", model.register_generator),
            ("action_input_model", model.action_input_model),
        )
    )
    groups = []
    seen: set[int] = set()
    for name, module in modules:
        parameters = []
        for parameter in module.parameters():
            if parameter.requires_grad and id(parameter) not in seen:
                parameters.append(parameter)
                seen.add(id(parameter))
        if parameters:
            groups.append(
                {
                    "name": name,
                    "params": parameters,
                    "lr": float(learning_rates[name]),
                }
            )
    expected = {
        id(parameter)
        for parameter in model.parameters()
        if parameter.requires_grad
    }
    if seen != expected:
        raise RuntimeError("Stage-G optimizer does not cover every trainable parameter once")
    return torch.optim.AdamW(
        groups,
        betas=tuple(float(value) for value in config.optimizer.betas),
        weight_decay=float(config.optimizer.weight_decay),
        eps=float(config.optimizer.get("eps", 1.0e-8)),
        fused=bool(config.optimizer.get("fused", True)) and torch.cuda.is_available(),
    )


def trainable_parameters_without_grad(model) -> list[str]:
    """Return trainable parameters disconnected from the current loss graph."""

    return [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.grad is None
    ]


def assert_all_trainable_parameters_have_grad(model) -> None:
    """Fail before the first optimizer step if Stage G has unused parameters."""

    unused = trainable_parameters_without_grad(model)
    if unused:
        preview = ", ".join(unused[:20])
        suffix = "" if len(unused) <= 20 else f" ... (+{len(unused) - 20})"
        raise RuntimeError(
            "Stage-G gradient gate found trainable parameters with grad=None: "
            f"{preview}{suffix}"
        )


class FirstBackwardGradientGate:
    """Track autograd use before DDP/ZeRO can partition or clear gradients."""

    def __init__(self, model) -> None:
        self._parameter_names: list[str] = []
        self._seen: set[str] = set()
        self._handles = []
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            self._parameter_names.append(name)
            self._handles.append(
                parameter.register_hook(
                    lambda gradient, parameter_name=name: self._record(
                        parameter_name, gradient
                    )
                )
            )

    def _record(self, name: str, gradient: torch.Tensor) -> torch.Tensor:
        self._seen.add(name)
        return gradient

    def missing_local(self) -> list[str]:
        return [name for name in self._parameter_names if name not in self._seen]

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def assert_complete(self, accelerator: Accelerator) -> None:
        local_seen = torch.tensor(
            [int(name in self._seen) for name in self._parameter_names],
            device=accelerator.device,
            dtype=torch.int32,
        )
        seen_per_parameter = accelerator.reduce(local_seen, reduction="sum")
        missing = [
            (name, int(seen_count))
            for name, seen_count in zip(
                self._parameter_names, seen_per_parameter.tolist()
            )
            if int(seen_count) != accelerator.num_processes
        ]
        self.close()
        if missing:
            preview = ", ".join(
                f"{name} ({seen}/{accelerator.num_processes} ranks)"
                for name, seen in missing[:20]
            )
            suffix = "" if len(missing) <= 20 else f" ... (+{len(missing) - 20})"
            raise RuntimeError(
                "Stage-G gradient gate found trainable parameters absent from "
                f"the first backward: {preview}{suffix}"
            )


def generator_component_checkpoint_names(
    *,
    epoch: int,
    final_epoch: int,
    save_epochs: set[int],
    should_stop: bool,
    improved_minade: bool,
) -> list[str]:
    """Return independent periodic and geometry-selected artifacts."""

    names: list[str] = []
    if epoch in save_epochs or epoch == final_epoch or should_stop:
        names.append(f"generator_epoch_{epoch:02d}.pt")
    if improved_minade:
        names.append("best_minade_generator.pt")
    return names


def _fixed_validation_loader(config, batch_size: int) -> DataLoader:
    validation_config = OmegaConf.create(
        OmegaConf.to_container(config, resolve=False)
    )
    validation = validation_config.validation
    validation_config.datasets.vla_data.split = str(
        validation.get("dataset_split", validation.get("split", "val"))
    )
    if validation.get("datalist_path"):
        validation_config.datasets.vla_data.datalist_path = validation.datalist_path
    validation_config.datasets.vla_data.per_device_batch_size = batch_size
    validation_config.datasets.vla_data.shuffle = False
    loader = build_dataloader(
        cfg=validation_config,
        dataset_py=validation_config.datasets.vla_data.dataset_py,
    )
    count = min(int(validation.get("num_scenes", 1024)), len(loader.dataset))
    return DataLoader(
        Subset(loader.dataset, range(count)),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=loader.collate_fn,
        num_workers=int(validation.get("num_workers", 4)),
        pin_memory=True,
        persistent_workers=int(validation.get("num_workers", 4)) > 0,
    )


def summarize_register_usage(usage: torch.Tensor) -> dict[str, Any]:
    """Turn the globally reduced winner counts into collapse diagnostics."""

    if usage.ndim != 1 or usage.numel() == 0:
        raise ValueError("register usage must be a non-empty vector")
    probabilities = usage / usage.sum().clamp_min(1)
    nonzero = probabilities > 0
    entropy = -(probabilities[nonzero] * probabilities[nonzero].log()).sum()
    if usage.numel() > 1:
        entropy = entropy / math.log(usage.numel())
    return {
        "register_usage_entropy": float(entropy),
        "active_register_ratio": float(nonzero.float().mean()),
        "top1_register_fraction": float(probabilities.max()),
        "register_usage_histogram": [int(value) for value in usage.tolist()],
    }


@torch.no_grad()
def evaluate_generator(model, dataloader, accelerator: Accelerator) -> dict[str, Any]:
    model.eval()
    scalar_names = (
        "min_ade_1",
        "min_ade_64",
        "min_fde_1",
        "min_fde_64",
        "pairwise_ade",
        "pairwise_fde",
        "endpoint_std",
        "endpoint_covariance",
    )
    totals = torch.zeros(len(scalar_names) + 1, device=accelerator.device)
    usage = torch.zeros(
        accelerator.unwrap_model(model).register_generator.proposal_num,
        device=accelerator.device,
    )
    for examples in dataloader:
        with accelerator.autocast():
            output = model(examples)
        count = float(len(examples))
        for index, name in enumerate(scalar_names):
            totals[index] += output["metrics"][name].float() * count
        totals[-1] += count
        usage += output["metrics"]["register_usage_histogram"].to(usage)
    totals = accelerator.reduce(totals, reduction="sum")
    usage = accelerator.reduce(usage, reduction="sum")
    count = totals[-1].clamp_min(1)
    metrics = {
        name: float(totals[index] / count)
        for index, name in enumerate(scalar_names)
    }
    metrics.update(summarize_register_usage(usage))
    model.train()
    return metrics


def _component_metadata(model, config) -> dict[str, Any]:
    unwrapped = model
    return {
        "schema_version": 1,
        "stage": "register_generator",
        "qwen_base_model": str(config.framework.qwenvl.base_vlm),
        "proposal_num": int(unwrapped.register_generator.proposal_num),
        "num_poses": int(unwrapped.register_generator.num_poses),
        "state_dim": int(unwrapped.register_generator.state_dim),
        "scene_queries": int(unwrapped.scene_encoder.num_queries),
        "scene_dim": int(unwrapped.scene_encoder.output_dim),
        "decoder_layers": int(unwrapped.register_generator.num_layers),
        "decoder_heads": int(unwrapped.register_generator.num_heads),
        "proposal_head_style": str(
            unwrapped.register_generator.proposal_head_style
        ),
        "stage_loss_mode": str(unwrapped.register_generator.stage_loss_mode),
        "proposal_head_count": int(
            unwrapped.register_generator.proposal_head_count
        ),
        "commit": _repository_commit(),
        "config_hash": stable_config_hash(config),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = load_training_config(args.config)
    trainer = config.trainer
    epochs = int(trainer.get("max_epochs", 25))
    if not 1 <= epochs <= 25:
        raise ValueError("Stage G max_epochs must be in [1,25]")
    accumulation = int(trainer.get("gradient_accumulation_steps", 1))
    accelerator = Accelerator(
        gradient_accumulation_steps=accumulation,
        mixed_precision=str(config.get("precision", "bf16")),
    )
    set_seed(int(config.get("seed", 42)), device_specific=True)
    global_batch = int(trainer.get("global_batch_size", 64))
    denominator = accelerator.num_processes * accumulation
    if global_batch % denominator:
        raise ValueError(
            "global_batch_size must divide world_size * gradient_accumulation_steps"
        )
    per_device_batch = global_batch // denominator
    config.datasets.vla_data.per_device_batch_size = per_device_batch
    output_dir = Path(str(config.run_root_dir)) / str(config.run_id)
    config.output_dir = str(output_dir)
    train_loader = build_dataloader(
        cfg=config, dataset_py=config.datasets.vla_data.dataset_py
    )
    val_loader = _fixed_validation_loader(config, per_device_batch)
    model = build_framework(config)
    if model.__class__.__name__ != "QwenRegisterGenerator":
        raise RuntimeError("Stage-G entry accepts only QwenRegisterGenerator")
    model.assert_qwen_trainability()
    optimizer = build_generator_optimizer(model, config)
    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )
    gradient_gate_config = trainer.get("gradient_gate", {})
    gradient_gate = (
        FirstBackwardGradientGate(model)
        if bool(gradient_gate_config.get("enabled", True))
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
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "checkpoints").mkdir(exist_ok=True)
        OmegaConf.save(config, output_dir / "config.yaml")
        accelerator.unwrap_model(model).log_architecture_summary(
            type("PrintLogger", (), {"info": staticmethod(lambda msg, *a: print(msg % a))})
        )
        print(
            f"Stage G: global_batch={global_batch} steps/epoch={steps_per_epoch} "
            f"total_steps={total_steps}"
        )
    early_cfg = trainer.get("early_stopping", {})
    early = EarlyStopping(
        patience=int(early_cfg.get("patience", 5)),
        mode=str(early_cfg.get("mode", "min")),
    )
    early_enabled = bool(early_cfg.get("enabled", True))
    resume_from = trainer.get("resume_from")
    if resume_from:
        accelerator.load_state(str(resume_from))
        early.best = progress.early_best
        early.bad_epochs = progress.early_bad_epochs
    elif accelerator.is_main_process:
        (output_dir / "metrics.jsonl").write_text("", encoding="utf-8")
    accelerator.wait_for_everyone()
    start_epoch = progress.epoch + 1
    save_epochs = {int(value) for value in trainer.get("save_epochs", [5, 10, 15, 20, 25])}
    completed_steps = progress.completed_steps
    last_epoch = progress.epoch
    should_stop = False
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(start_epoch, epochs + 1):
        last_epoch = epoch
        model.train()
        epoch_start = time.perf_counter()
        samples = 0
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(accelerator.device)
        for examples in train_loader:
            with accelerator.accumulate(model):
                with accelerator.autocast():
                    output = model(examples)
                    loss = output["loss"]
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    if gradient_gate is not None:
                        gradient_gate.assert_complete(accelerator)
                        gradient_gate = None
                        if accelerator.is_main_process:
                            print(
                                "Stage G gradient gate: every trainable parameter "
                                "received a gradient"
                            )
                    accelerator.clip_grad_norm_(
                        model.parameters(),
                        float(trainer.get("gradient_clip", 1.0)),
                    )
                optimizer.step()
                if accelerator.sync_gradients:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            samples += len(examples) * accelerator.num_processes
            if accelerator.sync_gradients:
                completed_steps += 1
        validation = evaluate_generator(model, val_loader, accelerator)
        epoch_seconds = time.perf_counter() - epoch_start
        peak_memory = (
            int(torch.cuda.max_memory_allocated(accelerator.device))
            if torch.cuda.is_available()
            else 0
        )
        minade_value = float(validation["min_ade_64"])
        if not math.isfinite(minade_value):
            raise RuntimeError(f"Stage-G min_ade_64 is not finite: {minade_value}")
        improved_minade, patience_exhausted = early.update(minade_value)
        should_stop = early_enabled and patience_exhausted
        progress.epoch = epoch
        progress.completed_steps = completed_steps
        progress.early_best = early.best
        progress.early_bad_epochs = early.bad_epochs
        save_this_epoch = (
            epoch in save_epochs
            or improved_minade
            or epoch == epochs
            or should_stop
        )
        accelerator.wait_for_everyone()
        if save_this_epoch:
            state_dir = output_dir / "checkpoints" / f"epoch_{epoch:02d}"
            accelerator.save_state(str(state_dir))
            gathered_state = accelerator.get_state_dict(model)
        else:
            gathered_state = None
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            unwrapped = accelerator.unwrap_model(model)
            if save_this_epoch:
                metadata = _component_metadata(unwrapped, config)
                metadata.update(
                    epoch=epoch,
                    completed_steps=completed_steps,
                    validation=validation,
                )
                component_names = generator_component_checkpoint_names(
                    epoch=epoch,
                    final_epoch=epochs,
                    save_epochs=save_epochs,
                    should_stop=should_stop,
                    improved_minade=improved_minade,
                )
                for component_name in component_names:
                    save_register_generator_checkpoint(
                        output_dir / component_name,
                        qwen_vl_interface=unwrapped.qwen_vl_interface,
                        action_input_model=unwrapped.action_input_model,
                        scene_encoder=unwrapped.scene_encoder,
                        register_generator=unwrapped.register_generator,
                        metadata=metadata,
                        full_model_state_dict=gathered_state,
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
                            "peak_memory_bytes": peak_memory,
                            **validation,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
            print(f"Stage G epoch {epoch}: {validation}")
        stop_tensor = torch.tensor(
            int(should_stop), device=accelerator.device, dtype=torch.int32
        )
        stop_tensor = accelerator.reduce(stop_tensor, reduction="max")
        if bool(stop_tensor.item()):
            break
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        selected_checkpoint = output_dir / "best_minade_generator.pt"
        if not selected_checkpoint.is_file():
            raise RuntimeError(
                "Stage G completed without best_minade_generator.pt; "
                "geometry validation must run before handoff"
            )
        atomic_json(
            output_dir / "training_complete.json",
            {
                "schema_version": 1,
                "status": "complete",
                "stage": "register_generator",
                "completed_epochs": int(last_epoch),
                "completed_steps": int(completed_steps),
                "early_stopped": bool(should_stop),
                "selection_metric": "min_ade_64",
                "selection_metric_value": float(early.best),
                "selected_checkpoint": selected_checkpoint.name,
                "selected_checkpoint_sha256": sha256_file(selected_checkpoint),
            },
        )


if __name__ == "__main__":
    main()
