#!/usr/bin/env python3
"""Stage S: train only DrivoR from an immutable Register64 candidate bank."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from omegaconf import OmegaConf

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from starVLA.candidate_bank.dataset import (
    CandidateBankDataset,
    build_candidate_bank_dataloader,
)
from starVLA.candidate_bank.schema import CANDIDATE_METRICS, manifest_hash
from starVLA.model.modules.register_planner.checkpoint import (
    load_stage_component_checkpoint,
    save_stage_component_checkpoint,
    sha256_file,
)
from starVLA.model.modules.trajectory_scorer.drivor_dynamic_scorer import (
    DrivoRDynamicScorer,
)
from starVLA.model.modules.trajectory_scorer.losses import (
    DrivoRMetricLoss,
    PDMSValueLoss,
)
from starVLA.training.config_loader import load_training_config
from starVLA.training.register_stage_utils import (
    EarlyStopping,
    atomic_json,
    cosine_schedule,
    move_bank_batch,
    optimizer_steps_per_epoch,
    selector_statistics,
    validate_bank_only_training_profile,
)


def build_drivor_scorer(config: Mapping[str, Any]) -> DrivoRDynamicScorer:
    model = config["model"] if "model" in config else config
    return DrivoRDynamicScorer(
        scene_dim=int(model.get("scene_dim", 256)),
        ego_state_dim=int(model.get("ego_state_dim", 4)),
        model_dim=int(model.get("model_dim", 256)),
        ffn_dim=int(model.get("ffn_dim", 1024)),
        num_layers=int(model.get("num_layers", 4)),
        num_heads=int(model.get("num_heads", 1)),
        dropout=float(model.get("dropout", 0.0)),
        decoder_style=str(model.get("decoder_style", "donor_register")),
        proj_drop=float(model.get("proj_drop", 0.1)),
        drop_path=float(model.get("drop_path", 0.2)),
        layer_scale_init=float(model.get("layer_scale_init", 0.0)),
        noc=float(model.get("noc", 1.0)),
        dac=float(model.get("dac", 1.0)),
        ddc=float(model.get("ddc", 0.0)),
        ttc=float(model.get("ttc", 5.0)),
        ep=float(model.get("ep", 5.0)),
        comfort=float(model.get("comfort", 2.0)),
        aggregate_head=bool(model.get("aggregate_head", False)),
        selection_mode=str(model.get("selection_mode", "formula")),
        aggregate_temperature=float(model.get("aggregate_temperature", 1.0)),
        selection_alpha=float(model.get("selection_alpha", 0.0)),
        debug_validate_finite=bool(model.get("debug_validate_finite", False)),
    )


def build_drivor_loss(config: Mapping[str, Any]):
    loss = config.get("loss", {})
    loss_type = str(loss.get("type", "drivor_submetrics"))
    if loss_type == "drivor_submetrics":
        return DrivoRMetricLoss()
    if loss_type == "pdms_value":
        return PDMSValueLoss(
            submetric_weight=float(loss.get("submetric_weight", 1.0)),
            aggregate_weight=float(loss.get("aggregate_weight", 1.0)),
            listwise_weight=float(loss.get("listwise_weight", 1.0)),
            pairwise_weight=float(loss.get("pairwise_weight", 0.5)),
            listwise_temperature=float(loss.get("listwise_temperature", 0.1)),
            pairwise_temperature=float(loss.get("pairwise_temperature", 0.1)),
            pairwise_margin=float(loss.get("pairwise_margin", 0.05)),
        )
    raise ValueError(f"unsupported DrivoR loss type {loss_type!r}")


def drivor_training_step(
    scorer: DrivoRDynamicScorer,
    batch: Mapping[str, Any],
    *,
    topm: int = 32,
    loss_module: DrivoRMetricLoss | PDMSValueLoss | None = None,
    compute_statistics: bool = True,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """One bank-only step; this function has no raw-data or evaluator path."""

    loss_module = loss_module or DrivoRMetricLoss()
    output = scorer(
        batch["proposals"],
        batch["scene_global_tokens"],
        batch["ego_state"],
        topm=min(int(topm), int(batch["proposals"].shape[1])),
    )
    if isinstance(loss_module, PDMSValueLoss):
        loss, components = loss_module(
            output.metric_logits, output.aggregate_logit, batch["metrics"]
        )
    else:
        loss, components = loss_module(output.metric_logits, batch["metrics"])
    statistics = (
        selector_statistics(
            output.aggregate_score, batch["metrics"]["aggregate_score"]
        )
        if compute_statistics
        else {}
    )
    if compute_statistics:
        statistics.update(
            {f"loss_{name}": value for name, value in components.items()}
        )
    statistics["loss"] = loss.detach()
    return loss, statistics


@torch.no_grad()
def evaluate_drivor(
    scorer: DrivoRDynamicScorer,
    dataloader,
    accelerator: Accelerator,
    *,
    topm: int,
) -> dict[str, float]:
    scorer.eval()
    names = (
        "selected_true_score",
        "oracle_true_score",
        "regret",
        "pairwise_ranking_accuracy",
        "recall_at_1",
        "recall_at_5",
        "recall_at_10",
        "recall_at_32",
    )
    totals = torch.zeros(len(names) + 1, device=accelerator.device)
    for raw_batch in dataloader:
        batch = move_bank_batch(raw_batch, accelerator.device)
        with accelerator.autocast():
            output = scorer(
                batch["proposals"],
                batch["scene_global_tokens"],
                batch["ego_state"],
                topm=min(topm, batch["proposals"].shape[1]),
            )
        values = selector_statistics(
            output.aggregate_score.float(),
            batch["metrics"]["aggregate_score"].float(),
        )
        count = float(batch["proposals"].shape[0])
        totals[:-1] += torch.stack([values[name] for name in names]) * count
        totals[-1] += count
    totals = accelerator.reduce(totals, reduction="sum")
    count = totals[-1].clamp_min(1.0)
    scorer.train()
    return {name: float(totals[index] / count) for index, name in enumerate(names)}


@torch.no_grad()
def calibrate_hybrid_selector(
    scorer: DrivoRDynamicScorer,
    dataloader,
    accelerator: Accelerator,
    *,
    grid_size: int = 21,
) -> dict[str, float]:
    """Choose direct/structured fusion on true validation PDMS.

    The grid contains both endpoints, so validation selection cannot be worse
    than using the DriveVLA-M0 direct head alone (alpha=0) or the structured
    DrivoR formula alone (alpha=1).  Only one scorer forward is needed per
    validation batch.
    """

    module = accelerator.unwrap_model(scorer)
    if module.selection_mode != "calibrated_hybrid":
        return {}
    if grid_size < 2:
        raise ValueError("hybrid selector calibration grid_size must be >= 2")
    scorer.eval()
    alphas = torch.linspace(0.0, 1.0, grid_size, device=accelerator.device)
    totals = torch.zeros(grid_size + 1, device=accelerator.device)
    for raw_batch in dataloader:
        batch = move_bank_batch(raw_batch, accelerator.device)
        with accelerator.autocast():
            output = scorer(
                batch["proposals"],
                batch["scene_global_tokens"],
                batch["ego_state"],
                topm=batch["proposals"].shape[1],
            )
        if output.aggregate_logit is None or output.formula_score is None:
            raise RuntimeError("calibrated selector requires both scorer branches")
        direct = module._scene_standardize(
            output.aggregate_logit.float() / module.aggregate_temperature
        )
        structured = module._scene_standardize(output.formula_score.float())
        blended = (
            (1.0 - alphas[:, None, None]) * direct[None]
            + alphas[:, None, None] * structured[None]
        )
        selected = blended.argmax(dim=-1)
        true_score = batch["metrics"]["aggregate_score"].float()
        selected_true = torch.gather(
            true_score[None].expand(grid_size, -1, -1),
            2,
            selected[..., None],
        ).squeeze(-1)
        totals[:-1] += selected_true.sum(dim=1)
        totals[-1] += true_score.shape[0]
    totals = accelerator.reduce(totals, reduction="sum")
    count = totals[-1].clamp_min(1.0)
    means = totals[:-1] / count
    best_index = int(means.argmax().item())
    best_alpha = float(alphas[best_index].item())
    module.set_selection_alpha(best_alpha)
    scorer.train()
    return {
        "selector_alpha": best_alpha,
        "selector_calibrated_pdms": float(means[best_index]),
        "selector_direct_pdms": float(means[0]),
        "selector_structured_pdms": float(means[-1]),
        "selector_calibration_scenes": float(count),
    }


def _checkpoint_metadata(config, dataset: CandidateBankDataset) -> dict[str, Any]:
    model = config.model
    return {
        "candidate_bank_manifest_hash": manifest_hash(dataset.manifest),
        "generator_checkpoint_sha256": dataset.manifest.generator_checkpoint_sha256,
        "proposal_num": dataset.manifest.proposal_num,
        "scene_dim": int(model.scene_dim),
        "model_dim": int(model.model_dim),
        "decoder_layers": int(model.num_layers),
        "decoder_heads": int(model.num_heads),
        "aggregate_weights": {
            name: float(value)
            for name, value in config.model.items()
            if name in {"noc", "dac", "ddc", "ttc", "ep", "comfort"}
        },
        "label_protocol": str(dataset.manifest.label_protocol),
        "metric_schema": list(CANDIDATE_METRICS),
        "training_profile": str(config.training_profile.name),
        "selection_mode": str(model.get("selection_mode", "formula")),
        "aggregate_head": bool(model.get("aggregate_head", False)),
        "aggregate_temperature": float(model.get("aggregate_temperature", 1.0)),
        "loss_type": str(config.get("loss", {}).get("type", "drivor_submetrics")),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = load_training_config(args.config)
    expected_profile = str(config.training_profile.name)
    if expected_profile not in {
        "drivor_offline_bank_v1",
        "clover_pdms_value_bank_v1",
    }:
        raise ValueError(f"unsupported DrivoR training profile {expected_profile!r}")
    validate_bank_only_training_profile(config, expected_name=expected_profile)
    trainer = config.trainer
    epochs = int(trainer.get("epochs", 5))
    max_epochs = int(trainer.get("max_epochs", 10))
    if not 1 <= epochs <= max_epochs <= 10:
        raise ValueError("Stage S requires 1 <= epochs <= max_epochs <= 10")
    accumulation = int(trainer.get("gradient_accumulation_steps", 1))
    accelerator = Accelerator(
        gradient_accumulation_steps=accumulation,
        mixed_precision=str(config.get("precision", "bf16")),
    )
    set_seed(int(config.get("seed", 42)), device_specific=True)
    train_dataset = CandidateBankDataset(
        str(config.candidate_bank.train_root), strict=True
    )
    val_dataset = CandidateBankDataset(
        str(config.candidate_bank.val_root),
        expected_generator_checkpoint_sha256=(
            train_dataset.manifest.generator_checkpoint_sha256
        ),
        expected_generator_config_hash=(
            train_dataset.manifest.generator_config_hash
        ),
        strict=True,
    )
    if (
        train_dataset.manifest.proposal_num != 64
        or val_dataset.manifest.proposal_num != 64
    ):
        raise RuntimeError("production DrivoR stage requires Register64 banks")
    if (
        train_dataset.manifest.scene_dim != int(config.model.scene_dim)
        or val_dataset.manifest.scene_dim != int(config.model.scene_dim)
    ):
        raise RuntimeError("candidate-bank scene dimension differs from DrivoR config")
    expected_protocol = str(
        config.candidate_bank.get("label_protocol", "navsim_v2_epdms")
    )
    for name, dataset in (("train", train_dataset), ("val", val_dataset)):
        if str(dataset.manifest.label_protocol) != expected_protocol:
            raise RuntimeError(
                f"{name} candidate bank label protocol "
                f"{dataset.manifest.label_protocol!r} != {expected_protocol!r}"
            )
    global_batch = int(trainer.get("global_batch_size", 256))
    denominator = accelerator.num_processes * accumulation
    if global_batch % denominator:
        raise ValueError(
            "global_batch_size must divide world_size * gradient_accumulation_steps"
        )
    per_device_batch = global_batch // denominator
    workers = int(trainer.get("num_workers", 8))
    train_loader = build_candidate_bank_dataloader(
        train_dataset,
        batch_size=per_device_batch,
        shuffle=True,
        num_workers=workers,
    )
    val_loader = build_candidate_bank_dataloader(
        val_dataset,
        batch_size=per_device_batch,
        shuffle=False,
        num_workers=workers,
    )
    scorer = build_drivor_scorer(config)
    metadata = _checkpoint_metadata(config, train_dataset)
    initialize_from = config.get("initialize_from")
    if initialize_from:
        initialize_mode = str(config.get("initialize_mode", "strict"))
        initialize_stage = str(config.get("initialize_stage", "drivor_scorer"))
        if initialize_mode == "strict":
            expected_initialize_metadata = metadata
        elif initialize_mode == "weights_only":
            expected_initialize_metadata = {
                key: metadata[key]
                for key in (
                    "proposal_num",
                    "scene_dim",
                    "model_dim",
                    "decoder_layers",
                    "decoder_heads",
                    "aggregate_weights",
                    "label_protocol",
                    "selection_mode",
                    "aggregate_head",
                    "aggregate_temperature",
                    "loss_type",
                )
            }
        else:
            raise ValueError("initialize_mode must be strict or weights_only")
        load_stage_component_checkpoint(
            str(initialize_from),
            stage=initialize_stage,
            module=scorer,
            expected_metadata=expected_initialize_metadata,
        )
    optimizer_cfg = config.optimizer
    optimizer = torch.optim.AdamW(
        scorer.parameters(),
        lr=float(optimizer_cfg.get("lr", 2.0e-4)),
        betas=tuple(float(value) for value in optimizer_cfg.get("betas", [0.9, 0.95])),
        weight_decay=float(optimizer_cfg.get("weight_decay", 1.0e-3)),
        eps=float(optimizer_cfg.get("eps", 1.0e-8)),
        fused=bool(optimizer_cfg.get("fused", True)) and torch.cuda.is_available(),
    )
    scorer, optimizer, train_loader, val_loader = accelerator.prepare(
        scorer, optimizer, train_loader, val_loader
    )
    steps_per_epoch = optimizer_steps_per_epoch(len(train_loader), accumulation)
    total_steps = steps_per_epoch * epochs
    scheduler = cosine_schedule(
        optimizer,
        total_steps=total_steps,
        warmup_ratio=float(config.scheduler.get("warmup_ratio", 0.05)),
    )
    output_dir = Path(str(config.run_root_dir)) / str(config.run_id)
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(config, output_dir / "config.yaml")
        (output_dir / "metrics.jsonl").write_text("", encoding="utf-8")
        print(
            f"Stage S: scenes={len(train_dataset)} global_batch={global_batch} "
            f"steps/epoch={steps_per_epoch} total_steps={total_steps}"
        )
    early_cfg = trainer.get("early_stopping", {})
    early = EarlyStopping(
        patience=int(early_cfg.get("patience", 2)),
        mode=str(early_cfg.get("mode", "min")),
    )
    early_enabled = bool(early_cfg.get("enabled", True))
    loss_module = build_drivor_loss(config)
    topm = int(config.model.get("dynamic_topm", 32))
    best_selected = -math.inf
    optimizer.zero_grad(set_to_none=True)
    completed_steps = 0
    last_epoch = 0
    should_stop = False
    for epoch in range(1, epochs + 1):
        last_epoch = epoch
        scorer.train()
        start = time.perf_counter()
        for raw_batch in train_loader:
            batch = move_bank_batch(raw_batch, accelerator.device)
            with accelerator.accumulate(scorer):
                with accelerator.autocast():
                    loss, _ = drivor_training_step(
                        scorer,
                        batch,
                        topm=topm,
                        loss_module=loss_module,
                        compute_statistics=False,
                    )
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        scorer.parameters(),
                        float(trainer.get("gradient_clip", 1.0)),
                    )
                optimizer.step()
                if accelerator.sync_gradients:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            if accelerator.sync_gradients:
                completed_steps += 1
        calibration = calibrate_hybrid_selector(
            scorer,
            val_loader,
            accelerator,
            grid_size=int(config.get("selector_calibration", {}).get("grid_size", 21)),
        )
        validation = evaluate_drivor(
            scorer, val_loader, accelerator, topm=topm
        )
        validation.update(calibration)
        epoch_seconds = time.perf_counter() - start
        improved_regret, patience_exhausted = early.update(validation["regret"])
        should_stop = early_enabled and patience_exhausted
        improved_selected = validation["selected_true_score"] > best_selected
        best_selected = max(best_selected, validation["selected_true_score"])
        accelerator.wait_for_everyone()
        gathered_state = accelerator.get_state_dict(scorer)
        if accelerator.is_main_process:
            unwrapped = accelerator.unwrap_model(scorer)
            epoch_metadata = {
                **metadata,
                "epoch": epoch,
                "completed_steps": completed_steps,
                "selection_alpha": float(unwrapped.selection_alpha.item()),
                "validation": validation,
            }
            save_stage_component_checkpoint(
                output_dir / "last.pt",
                stage="drivor_scorer",
                module=unwrapped,
                metadata=epoch_metadata,
                state_dict=gathered_state,
            )
            if improved_regret:
                save_stage_component_checkpoint(
                    output_dir / "best_regret.pt",
                    stage="drivor_scorer",
                    module=unwrapped,
                    metadata=epoch_metadata,
                    state_dict=gathered_state,
                )
            if improved_selected:
                save_stage_component_checkpoint(
                    output_dir / "best_selected_pdms.pt",
                    stage="drivor_scorer",
                    module=unwrapped,
                    metadata=epoch_metadata,
                    state_dict=gathered_state,
                )
            with (output_dir / "metrics.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(
                    json.dumps(
                        {
                            "epoch": epoch,
                            "completed_steps": completed_steps,
                            "epoch_seconds": epoch_seconds,
                            "sec_per_step": epoch_seconds / max(steps_per_epoch, 1),
                            **validation,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
            print(f"Stage S epoch {epoch}: {validation}")
        stop_tensor = torch.tensor(
            int(should_stop), device=accelerator.device, dtype=torch.int32
        )
        stop_tensor = accelerator.reduce(stop_tensor, reduction="max")
        if bool(stop_tensor.item()):
            break
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        selected_checkpoint = output_dir / "best_regret.pt"
        if not selected_checkpoint.is_file():
            raise RuntimeError("Stage S completed without best_regret.pt")
        atomic_json(
            output_dir / "training_complete.json",
            {
                "schema_version": 1,
                "status": "complete",
                "stage": "drivor_scorer",
                "completed_epochs": int(last_epoch),
                "completed_steps": int(completed_steps),
                "early_stopped": bool(should_stop),
                "selected_checkpoint": selected_checkpoint.name,
                "selected_checkpoint_sha256": sha256_file(selected_checkpoint),
            },
        )
    train_dataset.close()
    val_dataset.close()


if __name__ == "__main__":
    main()
