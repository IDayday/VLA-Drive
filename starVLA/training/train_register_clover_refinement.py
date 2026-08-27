#!/usr/bin/env python3
"""One CLOVER Stage-2 generator phase from a freshly scored teacher bank."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from omegaconf import OmegaConf
from torch.utils.data import Subset

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from starVLA.candidate_bank import CandidateBankDataset, CandidateBankReader
from starVLA.candidate_bank.schema import CANDIDATE_METRICS
from starVLA.candidate_bank.dataset import build_candidate_bank_dataloader
from starVLA.dataloader import build_dataloader
from starVLA.model.framework import build_framework
from starVLA.model.modules.planning.trajectory_codec import TrajectoryCodec
from starVLA.model.modules.register_planner import (
    CloverStage2GeneratorLoss,
    build_teacher_target_sets,
    selected_set_enrichment,
    selected_set_enrichment_per_scene,
)
from starVLA.model.modules.register_planner.checkpoint import (
    load_register_generator_checkpoint,
    load_stage_component_checkpoint,
    save_register_generator_checkpoint,
    sha256_file,
)
from starVLA.training.config_loader import load_training_config
from starVLA.training.register_stage_utils import (
    MeanAccumulator,
    atomic_json,
    cosine_schedule,
    move_bank_batch,
    optimizer_steps_per_epoch,
)
from starVLA.training.train_register_drivor import build_drivor_scorer
from starVLA.training.train_register_generator import _component_metadata


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def _generator_expected_metadata(model, config) -> dict:
    return {
        "qwen_base_model": str(config.framework.qwenvl.base_vlm),
        "proposal_num": 64,
        "num_poses": 8,
        "state_dim": 3,
        "scene_queries": 16,
        "scene_dim": 256,
        "decoder_layers": 4,
        "decoder_heads": 1,
        "proposal_head_style": "donor_mlp_v1",
        "stage_loss_mode": str(model.register_generator.stage_loss_mode),
        "proposal_head_count": int(model.register_generator.proposal_head_count),
    }


def _load_scorer(config, bank, device: torch.device):
    scorer = build_drivor_scorer(config.scorer)
    model_config = config.scorer.model
    checkpoint = str(config.scorer.checkpoint)
    load_stage_component_checkpoint(
        checkpoint,
        stage="drivor_scorer",
        module=scorer,
        expected_metadata={
            "generator_checkpoint_sha256": bank.manifest.generator_checkpoint_sha256,
            "proposal_num": 64,
            "scene_dim": int(model_config.scene_dim),
            "model_dim": int(model_config.model_dim),
            "decoder_layers": int(model_config.num_layers),
            "decoder_heads": int(model_config.num_heads),
            "aggregate_weights": {
                name: float(model_config.get(name))
                for name in ("noc", "dac", "ddc", "ttc", "ep", "comfort")
            },
            "label_protocol": "navsim_v1_1_pdms_two_way",
            "metric_schema": list(CANDIDATE_METRICS),
            "training_profile": "clover_pdms_value_bank_v1",
            "selection_mode": "calibrated_hybrid",
            "aggregate_head": True,
            "aggregate_temperature": float(
                model_config.get("aggregate_temperature", 1.0)
            ),
            "loss_type": "pdms_value",
        },
    )
    scorer.to(device)
    scorer.eval()
    for parameter in scorer.parameters():
        parameter.requires_grad_(False)
    return scorer


@torch.no_grad()
def _enrichment_gate(config, scorer, accelerator: Accelerator) -> dict[str, float]:
    validation = CandidateBankDataset(
        str(config.candidate_bank.val_root),
        expected_generator_checkpoint_sha256=sha256_file(
            str(config.generator_checkpoint)
        ),
        strict=True,
    )
    if validation.manifest.label_protocol != "navsim_v1_1_pdms_two_way":
        raise RuntimeError("CLOVER refinement gate requires a v1 PDMS val bank")
    indices = range(accelerator.process_index, len(validation), accelerator.num_processes)
    loader = build_candidate_bank_dataloader(
        Subset(validation, list(indices)),
        batch_size=int(config.enrichment_gate.get("batch_size", 64)),
        shuffle=False,
        num_workers=int(config.enrichment_gate.get("num_workers", 4)),
        distributed=False,
    )
    names = (
        "pool_true_mean",
        "topk_true_mean",
        "pareto_true_mean",
        "topk_enrichment",
        "pareto_enrichment",
        "topk_enriched_scene_ratio",
        "pareto_enriched_scene_ratio",
        "pool_high_score_rate",
        "topk_high_score_rate",
        "pareto_high_score_rate",
        "topk_high_score_enrichment",
        "pareto_high_score_enrichment",
    )
    totals = torch.zeros(2 * len(names) + 1, device=accelerator.device)
    target_config = config.teacher_targets
    high_score_threshold = float(
        config.enrichment_gate.get("high_score_threshold", 0.9)
    )
    for raw_batch in loader:
        batch = move_bank_batch(raw_batch, accelerator.device)
        with accelerator.autocast():
            output = scorer(
                batch["proposals"],
                batch["scene_global_tokens"],
                batch["ego_state"],
                topm=64,
            )
        targets = build_teacher_target_sets(
            batch["proposals"],
            output.metric_logits,
            output.aggregate_score,
            topk=int(target_config.get("topk", 8)),
            pareto_max_size=int(target_config.get("pareto_max_size", 8)),
            pareto_min_size=int(target_config.get("pareto_min_size", 2)),
            reward_threshold=float(target_config.get("reward_threshold", 0.4)),
        )
        values = selected_set_enrichment_per_scene(
            targets,
            batch["metrics"]["aggregate_score"],
            high_score_threshold=high_score_threshold,
        )
        count = float(batch["proposals"].shape[0])
        stacked = torch.stack([values[name].float() for name in names])
        totals[: len(names)] += stacked.sum(dim=1)
        totals[len(names) : 2 * len(names)] += stacked.square().sum(dim=1)
        totals[-1] += count
    validation.close()
    totals = accelerator.reduce(totals, reduction="sum")
    denominator = totals[-1].clamp_min(1.0)
    report = {
        name: float(totals[index] / denominator)
        for index, name in enumerate(names)
    }
    confidence_z = float(config.enrichment_gate.get("confidence_z", 1.96))
    if confidence_z < 0:
        raise ValueError("enrichment confidence_z must be non-negative")
    for index, name in enumerate(names):
        mean = totals[index] / denominator
        if float(denominator) > 1:
            variance = (
                totals[len(names) + index] - denominator * mean.square()
            ).clamp_min(0.0) / (denominator - 1.0)
            stderr = (variance / denominator).sqrt()
        else:
            stderr = mean.new_tensor(float("inf"))
        report[f"{name}_stderr"] = float(stderr)
        report[f"{name}_lcb95"] = float(mean - confidence_z * stderr)
    report["high_score_threshold"] = high_score_threshold
    report["confidence_z"] = confidence_z
    report["num_scenes"] = float(denominator)
    thresholds = {
        "topk_enrichment_lcb95": float(
            config.enrichment_gate.get("min_topk_enrichment_lcb", 0.0)
        ),
        "pareto_enrichment_lcb95": float(
            config.enrichment_gate.get("min_pareto_enrichment_lcb", 0.0)
        ),
        "topk_enriched_scene_ratio": float(
            config.enrichment_gate.get("min_topk_scene_ratio", 0.5)
        ),
        "pareto_enriched_scene_ratio": float(
            config.enrichment_gate.get("min_pareto_scene_ratio", 0.5)
        ),
    }
    failures = {
        name: (report.get(name, float("-inf")), threshold)
        for name, threshold in thresholds.items()
        if report.get(name, float("-inf")) < threshold
    }
    if failures:
        detail = ", ".join(
            f"{name}={actual:.6f}<{threshold:.6f}"
            for name, (actual, threshold) in failures.items()
        )
        raise RuntimeError(
            "CLOVER selected-set enrichment gate failed; refusing a harmful "
            f"generator update: {detail}"
        )
    return report


def _teacher_batch(reader: CandidateBankReader, examples, device: torch.device):
    records = [reader.get(str(example["token"])) for example in examples]
    return {
        "proposals": torch.stack([record["proposals"] for record in records]).to(
            device=device, dtype=torch.float32
        ),
        "scene_global_tokens": torch.stack(
            [record["scene_global_tokens"] for record in records]
        ).to(device=device, dtype=torch.float32),
        "ego_state": torch.stack([record["ego_state"] for record in records]).to(
            device=device, dtype=torch.float32
        ),
        "aggregate_score": torch.stack(
            [record["metrics"]["aggregate_score"] for record in records]
        ).to(device=device, dtype=torch.float32),
    }


def main() -> None:
    args = _parse_args()
    config = load_training_config(args.config)
    if str(config.framework.name) != "QwenRegisterGenerator":
        raise ValueError("CLOVER refinement requires QwenRegisterGenerator")
    if int(config.trainer.get("epochs", 1)) != 1:
        raise ValueError("one invocation is exactly one CLOVER generator phase")
    accelerator = Accelerator(
        gradient_accumulation_steps=int(
            config.trainer.get("gradient_accumulation_steps", 1)
        ),
        mixed_precision=str(config.get("precision", "bf16")),
    )
    set_seed(int(config.get("seed", 2)), device_specific=True)
    input_checkpoint = Path(str(config.generator_checkpoint))
    if not input_checkpoint.is_file():
        raise FileNotFoundError(f"teacher generator is missing: {input_checkpoint}")
    train_bank = CandidateBankReader(
        str(config.candidate_bank.train_root),
        expected_generator_checkpoint_sha256=sha256_file(input_checkpoint),
        strict=True,
    )
    if train_bank.manifest.label_protocol != "navsim_v1_1_pdms_two_way":
        raise RuntimeError("CLOVER refinement requires NAVSIM-v1 PDMS labels")
    model = build_framework(config)
    qwen_state_names = set(model.baseline_qwen_trainable_names)
    load_register_generator_checkpoint(
        input_checkpoint,
        qwen_vl_interface=model.qwen_vl_interface,
        action_input_model=model.action_input_model,
        scene_encoder=model.scene_encoder,
        register_generator=model.register_generator,
        expected_metadata=_generator_expected_metadata(model, config),
    )
    # CLOVER freezes its visual/perception backbone in Stage 2. Qwen and the
    # history-token adapter are the corresponding backbone in this VLA route;
    # only Q-Former and Register64 are refined.
    for module in (model.qwen_vl_interface, model.action_input_model):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    trainable_modules = (model.scene_encoder, model.register_generator)
    parameters = [
        parameter
        for module in trainable_modules
        for parameter in module.parameters()
        if parameter.requires_grad
    ]
    if not parameters:
        raise RuntimeError("CLOVER generator phase has no trainable parameters")
    scorer = _load_scorer(config, train_bank, accelerator.device)
    enrichment = _enrichment_gate(config, scorer, accelerator)

    accumulation = int(config.trainer.get("gradient_accumulation_steps", 1))
    global_batch = int(config.trainer.get("global_batch_size", 32))
    denominator = accelerator.num_processes * accumulation
    if global_batch % denominator:
        raise ValueError("global batch must divide world size * accumulation")
    per_device_batch = global_batch // denominator
    config.datasets.vla_data.per_device_batch_size = per_device_batch
    loader = build_dataloader(
        cfg=config, dataset_py=config.datasets.vla_data.dataset_py
    )
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(config.optimizer.get("lr", 3.0e-5)),
        betas=tuple(float(value) for value in config.optimizer.get("betas", [0.9, 0.95])),
        weight_decay=float(config.optimizer.get("weight_decay", 1.0e-3)),
        eps=float(config.optimizer.get("eps", 1.0e-8)),
        fused=bool(config.optimizer.get("fused", True)) and torch.cuda.is_available(),
    )
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
    steps = optimizer_steps_per_epoch(len(loader), accumulation)
    scheduler = cosine_schedule(
        optimizer,
        total_steps=steps,
        warmup_ratio=float(config.scheduler.get("warmup_ratio", 0.05)),
    )
    loss_config = config.loss
    loss_module = CloverStage2GeneratorLoss(
        trajectory_weight=float(loss_config.get("trajectory_weight", 0.1)),
        diversity_weight=float(loss_config.get("diversity_weight", 0.02)),
        topk_weight=float(loss_config.get("topk_weight", 1.0)),
        pareto_weight=float(loss_config.get("pareto_weight", 1.0)),
        stability_weight=float(loss_config.get("stability_weight", 0.05)),
    )
    output_dir = Path(str(config.run_root_dir)) / str(config.run_id)
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        if (output_dir / "training_complete.json").is_file():
            raise FileExistsError(
                f"completed CLOVER generator phase already exists: {output_dir}"
            )
        OmegaConf.save(config, output_dir / "config.yaml")
        atomic_json(output_dir / "enrichment_gate.json", enrichment)
        print(f"CLOVER generator phase: steps={steps} enrichment={enrichment}")
    accelerator.wait_for_everyone()
    target_config = config.teacher_targets
    accumulator = MeanAccumulator()
    completed_steps = 0
    start = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    for examples in loader:
        teacher = _teacher_batch(train_bank, examples, accelerator.device)
        with torch.no_grad(), accelerator.autocast():
            scorer_output = scorer(
                teacher["proposals"],
                teacher["scene_global_tokens"],
                teacher["ego_state"],
                topm=64,
            )
            target_sets = build_teacher_target_sets(
                teacher["proposals"],
                scorer_output.metric_logits,
                scorer_output.aggregate_score,
                topk=int(target_config.get("topk", 8)),
                pareto_max_size=int(target_config.get("pareto_max_size", 8)),
                pareto_min_size=int(target_config.get("pareto_min_size", 2)),
                reward_threshold=float(target_config.get("reward_threshold", 0.4)),
            )
        ground_truth = TrajectoryCodec().flow_to_navsim(
            torch.as_tensor(
                np.asarray([example["action"] for example in examples]),
                device=accelerator.device,
                dtype=torch.float32,
            )
        )
        with accelerator.accumulate(model):
            with accelerator.autocast():
                generated = model(examples, generate_only=True)["generator_output"]
                loss_output = loss_module(
                    generated.proposal_list,
                    ground_truth,
                    teacher["proposals"],
                    target_sets,
                )
            accelerator.backward(loss_output.loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(
                    parameters, float(config.trainer.get("gradient_clip", 1.0))
                )
            optimizer.step()
            if accelerator.sync_gradients:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)
        accumulator.update(
            {
                "loss": loss_output.loss,
                "trajectory_loss": loss_output.trajectory_loss,
                "gt_loss": loss_output.gt_loss,
                "diversity_loss": loss_output.diversity_loss,
                "topk_loss": loss_output.topk_loss,
                "pareto_loss": loss_output.pareto_loss,
                "stability_loss": loss_output.stability_loss,
                **selected_set_enrichment(
                    target_sets, teacher["aggregate_score"]
                ),
            },
            weight=float(len(examples)),
        )
        if accelerator.sync_gradients:
            completed_steps += 1
    accelerator.wait_for_everyone()
    full_state = accelerator.get_state_dict(model)
    wall_time = time.perf_counter() - start
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)
        metadata = _component_metadata(unwrapped, config)
        metadata.update(
            training_stage="clover_stage2_generator",
            teacher_generator_sha256=sha256_file(input_checkpoint),
            teacher_bank_manifest_hash=train_bank.manifest.build_identity_hash,
            scorer_checkpoint_sha256=sha256_file(str(config.scorer.checkpoint)),
            completed_steps=completed_steps,
            enrichment_gate=enrichment,
        )
        destination = output_dir / "refined_generator.pt"
        save_register_generator_checkpoint(
            destination,
            qwen_vl_interface=unwrapped.qwen_vl_interface,
            action_input_model=unwrapped.action_input_model,
            scene_encoder=unwrapped.scene_encoder,
            register_generator=unwrapped.register_generator,
            metadata=metadata,
            full_model_state_dict=full_state,
            qwen_state_names=qwen_state_names,
        )
        metrics = accumulator.means()
        atomic_json(
            output_dir / "training_complete.json",
            {
                "schema_version": 1,
                "status": "complete",
                "stage": "clover_stage2_generator",
                "completed_steps": completed_steps,
                "wall_time_seconds": wall_time,
                "sec_per_step": wall_time / max(completed_steps, 1),
                "teacher_generator_sha256": sha256_file(input_checkpoint),
                "selected_checkpoint": destination.name,
                "selected_checkpoint_sha256": sha256_file(destination),
                "metrics": metrics,
                "enrichment_gate": enrichment,
            },
        )
        print(f"CLOVER generator phase complete: {metrics}")
    train_bank.close()


if __name__ == "__main__":
    main()
