#!/usr/bin/env python3
"""Stage SD/SH: train optional DriveSuprim selectors from candidate banks."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Optional

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from omegaconf import OmegaConf
from torch import Tensor, nn

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from starVLA.candidate_bank.dataset import (
    CandidateBankDataset,
    build_candidate_bank_dataloader,
)
from starVLA.candidate_bank.schema import manifest_hash
from starVLA.model.modules.register_planner.checkpoint import (
    load_stage_component_checkpoint,
    save_stage_component_checkpoint,
    sha256_file,
    stable_config_hash,
)
from starVLA.model.modules.register_planner.selectors import (
    DynamicDriveSuprimSelector,
    HybridDriveSuprimSelector,
    dynamic_coarse_output,
)
from starVLA.model.modules.trajectory_scorer.drivesuprim_joint_scorer import (
    DriveSuprimCoarseScorer,
    DriveSuprimFineRefiner,
)
from starVLA.model.modules.trajectory_scorer.losses import (
    SUPRIM_METRICS,
    DriveSuprimMetricLoss,
    gather_metric_targets,
)
from starVLA.model.modules.trajectory_scorer.static_score_store import (
    StaticVocabScoreStore,
)
from starVLA.training.config_loader import load_training_config
from starVLA.training.register_stage_utils import (
    EarlyStopping,
    atomic_json,
    cosine_schedule,
    freeze_module,
    move_bank_batch,
    optimizer_steps_per_epoch,
    selector_statistics,
    validate_bank_only_training_profile,
)
from starVLA.training.train_register_drivor import build_drivor_scorer


def _sha256_path(path: str | Path) -> str:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    return sha256_file(path)


def validate_hybrid_parity_gate(config) -> dict[str, Any]:
    """Fail before training unless static labels passed the parity audit."""

    path = Path(str(config.parity_gate.report_path))
    if not path.is_file():
        raise FileNotFoundError(
            f"hybrid metric-parity report is missing: {path}; run the parity tool first"
        )
    with path.open("r", encoding="utf-8") as stream:
        report = json.load(stream)
    if not bool(report.get("passed", False)):
        raise RuntimeError("hybrid metric-parity gate did not pass")
    vocabulary_sha = _sha256_path(config.model.static_vocab_path)
    if report.get("static_vocabulary_sha256") != vocabulary_sha:
        raise RuntimeError("parity report was produced for a different static vocabulary")
    expected_cache_sha = str(config.parity_gate.static_score_cache_vocabulary_sha256)
    if report.get("static_score_cache_vocabulary_sha256") != expected_cache_sha:
        raise RuntimeError("static score cache vocabulary hash differs from parity report")
    if vocabulary_sha != expected_cache_sha:
        raise RuntimeError("static vocabulary and static score cache vocabulary hashes differ")
    return report


class RegisterSuprimStage(nn.Module):
    """Bank-only frozen-DrivoR plus trainable dynamic or hybrid DriveSuprim."""

    def __init__(
        self,
        *,
        mode: str,
        drivor: nn.Module,
        selector: nn.Module,
        dynamic_topm: int = 32,
        memory_source: str = "global_scene_tokens",
        use_imitation: bool = True,
        coarse_loss_weight: float = 1.0,
        fine_loss_weight: float = 1.0,
    ) -> None:
        super().__init__()
        if mode not in {"dynamic", "hybrid"}:
            raise ValueError("DriveSuprim stage mode must be dynamic or hybrid")
        if memory_source not in {"global_scene_tokens", "dense_scene_memory"}:
            raise ValueError("unsupported DriveSuprim fine memory source")
        self.mode = mode
        self.drivor = freeze_module(drivor)
        self.selector = selector
        self.dynamic_topm = int(dynamic_topm)
        self.memory_source = memory_source
        self.use_imitation = bool(use_imitation)
        self.coarse_loss_weight = float(coarse_loss_weight)
        self.fine_loss_weight = float(fine_loss_weight)
        self.loss_module = DriveSuprimMetricLoss()

    def train(self, mode: bool = True):
        super().train(mode)
        self.drivor.eval()
        return self

    def _fine_memory(self, batch: Mapping[str, Any]) -> tuple[Tensor, Optional[Tensor]]:
        if self.memory_source == "global_scene_tokens":
            return batch["scene_global_tokens"], None
        if "scene_dense_memory" not in batch or "attention_mask" not in batch:
            raise RuntimeError(
                "dense_scene_memory requires a candidate bank built with include_dense_memory=true"
            )
        # Candidate-bank attention_mask uses True=valid; MHA uses True=padding.
        return batch["scene_dense_memory"], ~batch["attention_mask"].bool()

    def forward(
        self,
        batch: Mapping[str, Any],
        static_targets: Optional[Mapping[str, Tensor]] = None,
        *,
        compute_statistics: bool = True,
    ) -> dict[str, Any]:
        with torch.no_grad():
            drivor = self.drivor(
                batch["proposals"],
                batch["scene_global_tokens"],
                batch["ego_state"],
                topm=self.dynamic_topm,
            )
        fine_memory, fine_mask = self._fine_memory(batch)
        dynamic_targets = {
            name: batch["metrics"][name] for name in SUPRIM_METRICS
        }
        topm_targets = gather_metric_targets(
            dynamic_targets, drivor.topm_indices
        )
        topm_true_score = torch.gather(
            batch["metrics"]["aggregate_score"], 1, drivor.topm_indices
        )
        rows = torch.arange(
            batch["proposals"].shape[0], device=batch["proposals"].device
        )
        drivor_top1_true = topm_true_score[:, 0]

        if self.mode == "dynamic":
            coarse = dynamic_coarse_output(drivor)
            fine = self.selector.fine_refiner(coarse, fine_memory, fine_mask)
            fine_loss, details = self.loss_module.refinement(
                fine.layer_metric_logits,
                topm_targets,
                coarse.topk_trajectories_40,
                batch["gt_trajectory"],
                use_imitation=self.use_imitation,
            )
            selected_true = topm_true_score[rows, fine.selected_topk_index]
            statistics = (
                selector_statistics(fine.aggregate_score, topm_true_score)
                if compute_statistics
                else {}
            )
            total_loss = self.fine_loss_weight * fine_loss
            selected_register = fine.selected_source_index
            coarse_output = coarse
        else:
            if static_targets is None:
                raise RuntimeError("hybrid DriveSuprim requires static score targets")
            missing = set(SUPRIM_METRICS).difference(static_targets)
            if missing:
                raise KeyError(f"static targets are missing {sorted(missing)}")
            hybrid = self.selector(
                drivor,
                batch["scene_global_tokens"],
                batch["ego_state"],
                fine_memory,
                fine_mask,
            )
            coarse_output, fine = hybrid.coarse, hybrid.fine
            joint_targets = {
                name: torch.cat((static_targets[name], topm_targets[name]), dim=1)
                for name in SUPRIM_METRICS
            }
            coarse_loss, coarse_details = self.loss_module.one_layer(
                coarse_output.metric_logits,
                joint_targets,
                coarse_output.joint_candidates_40,
                batch["gt_trajectory"],
                use_imitation=self.use_imitation,
            )
            fine_targets = gather_metric_targets(
                joint_targets, coarse_output.topk_indices
            )
            fine_loss, details = self.loss_module.refinement(
                fine.layer_metric_logits,
                fine_targets,
                coarse_output.topk_trajectories_40,
                batch["gt_trajectory"],
                use_imitation=self.use_imitation,
            )
            if "aggregate_score" not in static_targets:
                raise RuntimeError(
                    "hybrid validation requires static cache pdm_score/aggregate_score"
                )
            joint_true_score = torch.cat(
                (static_targets["aggregate_score"], topm_true_score), dim=1
            )
            fine_true_score = torch.gather(
                joint_true_score, 1, coarse_output.topk_indices
            )
            selected_true = fine_true_score[rows, fine.selected_topk_index]
            statistics = (
                selector_statistics(fine.aggregate_score, fine_true_score)
                if compute_statistics
                else {}
            )
            total_loss = (
                self.coarse_loss_weight * coarse_loss
                + self.fine_loss_weight * fine_loss
            )
            selected_register = fine.selected_source_index
            details = {
                **{f"coarse_{name}": value for name, value in coarse_details.items()},
                **details,
            }
        statistics["loss"] = total_loss.detach()
        if compute_statistics:
            statistics.update(
                {
                    "drivor_top1_true_score": drivor_top1_true.mean(),
                    "suprim_selected_true_score": selected_true.mean(),
                    "refinement_gain": (selected_true - drivor_top1_true).mean(),
                }
            )
        return {
            "loss": total_loss,
            "statistics": statistics,
            "loss_details": details,
            "drivor": drivor,
            "coarse": coarse_output,
            "fine": fine,
            "selected_register": selected_register,
        }


def build_suprim_stage(config, dataset: CandidateBankDataset) -> RegisterSuprimStage:
    mode = str(config.stage_mode)
    drivor = build_drivor_scorer({"model": config.drivor_model})
    load_stage_component_checkpoint(
        str(config.drivor_checkpoint),
        stage="drivor_scorer",
        module=drivor,
        expected_metadata={
            "candidate_bank_manifest_hash": manifest_hash(dataset.manifest),
            "generator_checkpoint_sha256": dataset.manifest.generator_checkpoint_sha256,
            "proposal_num": dataset.manifest.proposal_num,
            "scene_dim": dataset.manifest.scene_dim,
            "model_dim": int(config.drivor_model.get("model_dim", 256)),
            "decoder_layers": int(config.drivor_model.get("num_layers", 4)),
            "decoder_heads": int(config.drivor_model.get("num_heads", 1)),
            "aggregate_weights": {
                name: float(value)
                for name, value in drivor.aggregate_weights.items()
            },
            "label_protocol": "navsim_v2_epdms",
            "training_profile": "drivor_offline_bank_v1",
        },
    )
    fine_cfg = config.model.fine
    fine = DriveSuprimFineRefiner(
        scene_dim=int(fine_cfg.get("scene_dim", 256)),
        model_dim=int(fine_cfg.get("model_dim", 256)),
        ffn_dim=int(fine_cfg.get("ffn_dim", 1024)),
        num_heads=int(fine_cfg.get("num_heads", 8)),
        num_layers=int(fine_cfg.get("refinement_layers", 3)),
        dropout=float(fine_cfg.get("dropout", 0.0)),
        use_mid_output=bool(fine_cfg.get("use_mid_output", True)),
        use_imitation=bool(fine_cfg.get("use_imitation", True)),
    )
    if mode == "dynamic":
        selector: nn.Module = DynamicDriveSuprimSelector(fine)
    elif mode == "hybrid":
        coarse_cfg = config.model.coarse
        coarse = DriveSuprimCoarseScorer(
            vocab_path=str(config.model.static_vocab_path),
            vocab_size=int(coarse_cfg.get("static_vocab_size", 8192)),
            num_poses=40,
            scene_dim=int(coarse_cfg.get("scene_dim", 256)),
            model_dim=int(coarse_cfg.get("model_dim", 256)),
            ffn_dim=int(coarse_cfg.get("ffn_dim", 1024)),
            num_heads=int(coarse_cfg.get("num_heads", 8)),
            num_layers=int(coarse_cfg.get("coarse_layers", 3)),
            coarse_topk=int(coarse_cfg.get("coarse_topk", 256)),
            dropout=float(coarse_cfg.get("dropout", 0.0)),
            normalize_vocab_pos=False,
        )
        expected = int(coarse_cfg.get("static_vocab_size", 8192)) + int(
            config.model.dynamic_topm
        )
        if expected != int(coarse_cfg.get("joint_candidate_count", 8224)):
            raise ValueError("hybrid joint candidate count must be 8192 + 32 = 8224")
        selector = HybridDriveSuprimSelector(coarse, fine)
    else:
        raise ValueError("stage_mode must be dynamic or hybrid")
    return RegisterSuprimStage(
        mode=mode,
        drivor=drivor,
        selector=selector,
        dynamic_topm=int(config.model.get("dynamic_topm", 32)),
        memory_source=str(config.model.get("fine_memory_source", "global_scene_tokens")),
        use_imitation=bool(fine_cfg.get("use_imitation", True)),
        coarse_loss_weight=float(config.loss.get("coarse_weight", 1.0)),
        fine_loss_weight=float(config.loss.get("fine_weight", 1.0)),
    )


def _static_store(config, *, split: str) -> Optional[StaticVocabScoreStore]:
    if str(config.stage_mode) != "hybrid":
        return None
    store = config.static_score_store
    return StaticVocabScoreStore(
        str(store.cache_root),
        split=split,
        vocab_size=int(config.model.coarse.static_vocab_size),
        cache_size=int(store.get("cache_size", 64)),
        mmap=bool(store.get("mmap", True)),
        include_aggregate_score=True,
    )


def _get_static_targets(store, tokens, device) -> Optional[dict[str, Tensor]]:
    if store is None:
        return None
    values = store.get(tokens, device=device, dtype=torch.float32)
    if "aggregate_score" not in values:
        raise RuntimeError(
            "hybrid static score cache must expose pdm_score or aggregate_score"
        )
    return values


@torch.no_grad()
def evaluate_suprim(stage, dataloader, accelerator, static_store) -> dict[str, float]:
    stage.eval()
    names = (
        "drivor_top1_true_score",
        "suprim_selected_true_score",
        "regret",
        "refinement_gain",
        "recall_at_1",
        "recall_at_5",
        "recall_at_10",
        "recall_at_32",
    )
    totals = torch.zeros(len(names) + 1, device=accelerator.device)
    for raw_batch in dataloader:
        batch = move_bank_batch(raw_batch, accelerator.device)
        static_targets = _get_static_targets(
            static_store, batch["token"], accelerator.device
        )
        with accelerator.autocast():
            output = stage(batch, static_targets)
        count = float(batch["proposals"].shape[0])
        values = output["statistics"]
        totals[:-1] += torch.stack([values[name].float() for name in names]) * count
        totals[-1] += count
    totals = accelerator.reduce(totals, reduction="sum")
    count = totals[-1].clamp_min(1)
    stage.train()
    return {name: float(totals[i] / count) for i, name in enumerate(names)}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = load_training_config(args.config)
    mode = str(config.stage_mode)
    expected_profile = {
        "dynamic": "drivesuprim_dynamic_bank_v1",
        "hybrid": "drivesuprim_hybrid_bank_v1",
    }.get(mode)
    if expected_profile is None:
        raise ValueError("stage_mode must be dynamic or hybrid")
    validate_bank_only_training_profile(
        config, expected_name=expected_profile
    )
    if mode == "hybrid":
        parity_report = validate_hybrid_parity_gate(config)
    else:
        parity_report = None
    trainer = config.trainer
    epochs = int(trainer.epochs)
    max_epochs = int(trainer.max_epochs)
    limit = 5 if mode == "dynamic" else 10
    if not 1 <= epochs <= max_epochs <= limit:
        raise ValueError(f"{mode} Stage requires 1 <= epochs <= max_epochs <= {limit}")
    accumulation = int(trainer.get("gradient_accumulation_steps", 1))
    accelerator = Accelerator(
        gradient_accumulation_steps=accumulation,
        mixed_precision=str(config.get("precision", "bf16")),
    )
    set_seed(int(config.get("seed", 42)), device_specific=True)
    train_dataset = CandidateBankDataset(str(config.candidate_bank.train_root))
    val_dataset = CandidateBankDataset(
        str(config.candidate_bank.val_root),
        expected_generator_checkpoint_sha256=train_dataset.manifest.generator_checkpoint_sha256,
        expected_generator_config_hash=train_dataset.manifest.generator_config_hash,
    )
    if (
        train_dataset.manifest.proposal_num != 64
        or val_dataset.manifest.proposal_num != 64
    ):
        raise RuntimeError("production DriveSuprim stages require Register64 banks")
    expected_scene_dim = int(config.drivor_model.get("scene_dim", 256))
    if (
        train_dataset.manifest.scene_dim != expected_scene_dim
        or val_dataset.manifest.scene_dim != expected_scene_dim
    ):
        raise RuntimeError(
            "candidate-bank scene dimension differs from DriveSuprim config"
        )
    memory_source = str(config.model.get("fine_memory_source", "global_scene_tokens"))
    if memory_source == "dense_scene_memory" and not (
        train_dataset.manifest.include_dense_memory
        and val_dataset.manifest.include_dense_memory
    ):
        raise RuntimeError(
            "fine_memory_source=dense_scene_memory requires include_dense_memory=true "
            "in both train and validation banks"
        )
    global_batch = int(trainer.global_batch_size)
    denominator = accelerator.num_processes * accumulation
    if global_batch % denominator:
        raise ValueError("global batch is not divisible by distributed accumulation")
    per_device = global_batch // denominator
    workers = int(trainer.get("num_workers", 8))
    train_loader = build_candidate_bank_dataloader(
        train_dataset,
        batch_size=per_device,
        shuffle=True,
        num_workers=workers,
    )
    val_loader = build_candidate_bank_dataloader(
        val_dataset,
        batch_size=per_device,
        shuffle=False,
        num_workers=workers,
    )
    stage = build_suprim_stage(config, train_dataset)
    parameters = [parameter for parameter in stage.selector.parameters() if parameter.requires_grad]
    if any(parameter.requires_grad for parameter in stage.drivor.parameters()):
        raise AssertionError("DrivoR must remain frozen during DriveSuprim stages")
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(config.optimizer.lr),
        betas=tuple(float(v) for v in config.optimizer.get("betas", [0.9, 0.95])),
        weight_decay=float(config.optimizer.weight_decay),
        fused=bool(config.optimizer.get("fused", True)) and torch.cuda.is_available(),
    )
    stage, optimizer, train_loader, val_loader = accelerator.prepare(
        stage, optimizer, train_loader, val_loader
    )
    steps_per_epoch = optimizer_steps_per_epoch(len(train_loader), accumulation)
    scheduler = cosine_schedule(
        optimizer,
        total_steps=steps_per_epoch * epochs,
        warmup_ratio=float(config.scheduler.get("warmup_ratio", 0.05)),
    )
    train_store = _static_store(config, split="train")
    val_store = _static_store(config, split="val")
    output_dir = Path(str(config.run_root_dir)) / str(config.run_id)
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(config, output_dir / "config.yaml")
        (output_dir / "metrics.jsonl").write_text("", encoding="utf-8")
        print(
            f"Stage S{mode[0].upper()}: global_batch={global_batch} "
            f"steps/epoch={steps_per_epoch} total_steps={steps_per_epoch * epochs}"
        )
    early_cfg = trainer.early_stopping
    early = EarlyStopping(
        patience=int(early_cfg.patience), mode=str(early_cfg.mode)
    )
    early_enabled = bool(early_cfg.get("enabled", True))
    completed_steps = 0
    last_epoch = 0
    should_stop = False
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(1, epochs + 1):
        last_epoch = epoch
        stage.train()
        start = time.perf_counter()
        for raw_batch in train_loader:
            batch = move_bank_batch(raw_batch, accelerator.device)
            static_targets = _get_static_targets(
                train_store, batch["token"], accelerator.device
            )
            with accelerator.accumulate(stage):
                with accelerator.autocast():
                    output = stage(
                        batch, static_targets, compute_statistics=False
                    )
                accelerator.backward(output["loss"])
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        parameters, float(trainer.get("gradient_clip", 1.0))
                    )
                optimizer.step()
                if accelerator.sync_gradients:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            if accelerator.sync_gradients:
                completed_steps += 1
        validation = evaluate_suprim(stage, val_loader, accelerator, val_store)
        improved, patience_exhausted = early.update(validation["regret"])
        should_stop = early_enabled and patience_exhausted
        accelerator.wait_for_everyone()
        gathered_stage_state = accelerator.get_state_dict(stage)
        selector_state = {
            key[len("selector.") :]: value
            for key, value in gathered_stage_state.items()
            if key.startswith("selector.")
        }
        if accelerator.is_main_process:
            unwrapped = accelerator.unwrap_model(stage)
            metadata = {
                "candidate_bank_manifest_hash": manifest_hash(train_dataset.manifest),
                "generator_checkpoint_sha256": train_dataset.manifest.generator_checkpoint_sha256,
                "drivor_checkpoint_sha256": sha256_file(config.drivor_checkpoint),
                "selector_type": mode,
                "dynamic_topm": int(config.model.dynamic_topm),
                "scene_dim": train_dataset.manifest.scene_dim,
                "memory_source": memory_source,
                "epoch": epoch,
                "completed_steps": completed_steps,
                "validation": validation,
                "training_profile": str(config.training_profile.name),
                "training_config_hash": stable_config_hash(config),
            }
            if parity_report is not None:
                metadata["static_vocabulary_sha256"] = parity_report[
                    "static_vocabulary_sha256"
                ]
            stage_name = f"suprim_{mode}"
            save_stage_component_checkpoint(
                output_dir / "last.pt",
                stage=stage_name,
                module=unwrapped.selector,
                metadata=metadata,
                state_dict=selector_state,
            )
            if improved:
                save_stage_component_checkpoint(
                    output_dir / "best_regret.pt",
                    stage=stage_name,
                    module=unwrapped.selector,
                    metadata=metadata,
                    state_dict=selector_state,
                )
            with (output_dir / "metrics.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(
                    json.dumps(
                        {
                            "epoch": epoch,
                            "completed_steps": completed_steps,
                            "epoch_seconds": time.perf_counter() - start,
                            **validation,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
            print(f"Stage S{mode[0].upper()} epoch {epoch}: {validation}")
        stop = accelerator.reduce(
            torch.tensor(int(should_stop), device=accelerator.device),
            reduction="max",
        )
        if bool(stop.item()):
            break
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        selected_checkpoint = output_dir / "best_regret.pt"
        if not selected_checkpoint.is_file():
            raise RuntimeError(
                f"Stage S{mode[0].upper()} completed without best_regret.pt"
            )
        atomic_json(
            output_dir / "training_complete.json",
            {
                "schema_version": 1,
                "status": "complete",
                "stage": f"suprim_{mode}",
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
