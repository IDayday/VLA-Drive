"""Small shared utilities for bank-only Register64 training stages."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

import torch
from torch import Tensor, nn


def move_bank_batch(
    batch: Mapping[str, Any], device: torch.device
) -> dict[str, Any]:
    """Move one lazy bank batch without changing token strings."""

    result: dict[str, Any] = {}
    for name, value in batch.items():
        if torch.is_tensor(value):
            value = value.to(device=device, non_blocking=True)
            if value.is_floating_point():
                value = value.float()
            result[name] = value
        elif isinstance(value, Mapping):
            result[name] = {
                key: tensor.to(
                    device=device, dtype=torch.float32, non_blocking=True
                )
                for key, tensor in value.items()
            }
        else:
            result[name] = value
    return result


def cosine_schedule(
    optimizer: torch.optim.Optimizer,
    *,
    total_steps: int,
    warmup_ratio: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if not 0.0 <= warmup_ratio < 1.0:
        raise ValueError("warmup_ratio must be in [0,1)")
    warmup_steps = int(round(total_steps * warmup_ratio))

    def scale(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = float(step - warmup_steps) / float(
            max(1, total_steps - warmup_steps)
        )
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


def pairwise_ranking_accuracy(predicted: Tensor, target: Tensor) -> Tensor:
    """Exact non-tied pair ordering accuracy for small candidate pools."""

    if predicted.shape != target.shape or predicted.ndim != 2:
        raise ValueError("ranking tensors must share shape [B,K]")
    candidate_count = predicted.shape[1]
    if candidate_count < 2:
        return predicted.new_ones(())
    rows, cols = torch.triu_indices(
        candidate_count, candidate_count, offset=1, device=predicted.device
    )
    pred_delta = predicted[:, rows] - predicted[:, cols]
    target_delta = target[:, rows] - target[:, cols]
    valid = target_delta != 0
    correct = (pred_delta * target_delta) > 0
    return (correct & valid).sum().float() / valid.sum().clamp_min(1)


def selector_statistics(
    predicted_score: Tensor,
    true_score: Tensor,
    *,
    recall_ks: Sequence[int] = (1, 5, 10, 32),
) -> dict[str, Tensor]:
    """Return selection quality, regret, recall, and rank agreement."""

    if predicted_score.shape != true_score.shape or predicted_score.ndim != 2:
        raise ValueError("selector scores must share shape [B,K]")
    rows = torch.arange(predicted_score.shape[0], device=predicted_score.device)
    selected_index = predicted_score.argmax(dim=1)
    oracle_index = true_score.argmax(dim=1)
    selected_true = true_score[rows, selected_index]
    oracle_true = true_score[rows, oracle_index]
    metrics = {
        "selected_true_score": selected_true.mean(),
        "oracle_true_score": oracle_true.mean(),
        "regret": (oracle_true - selected_true).mean(),
        "pairwise_ranking_accuracy": pairwise_ranking_accuracy(
            predicted_score, true_score
        ),
    }
    order = predicted_score.argsort(dim=1, descending=True)
    for k in recall_ks:
        effective = min(int(k), predicted_score.shape[1])
        metrics[f"recall_at_{k}"] = (
            order[:, :effective] == oracle_index[:, None]
        ).any(dim=1).float().mean()
    return metrics


@dataclass
class MeanAccumulator:
    totals: MutableMapping[str, float] = field(default_factory=dict)
    weights: MutableMapping[str, float] = field(default_factory=dict)

    def update(
        self, values: Mapping[str, Tensor | float], *, weight: float = 1.0
    ) -> None:
        for name, value in values.items():
            if torch.is_tensor(value):
                if value.numel() != 1:
                    continue
                value = float(value.detach().float().cpu())
            self.totals[name] = self.totals.get(name, 0.0) + float(value) * weight
            self.weights[name] = self.weights.get(name, 0.0) + weight

    def means(self) -> dict[str, float]:
        return {
            name: total / max(self.weights[name], 1e-12)
            for name, total in self.totals.items()
        }


@dataclass
class EarlyStopping:
    patience: int
    mode: str = "min"
    best: float | None = None
    bad_epochs: int = 0

    def update(self, value: float) -> tuple[bool, bool]:
        if self.mode not in {"min", "max"}:
            raise ValueError("early-stopping mode must be min or max")
        improved = self.best is None or (
            value < self.best if self.mode == "min" else value > self.best
        )
        if improved:
            self.best = float(value)
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        return improved, self.bad_epochs >= self.patience


@dataclass
class TrainingProgress:
    """Small Accelerate-checkpointable state for exact epoch-level resume."""

    epoch: int = 0
    completed_steps: int = 0
    early_best: float | None = None
    early_bad_epochs: int = 0
    best_oracle_pdms: float | None = None

    def state_dict(self) -> dict[str, Any]:
        return {
            "epoch": int(self.epoch),
            "completed_steps": int(self.completed_steps),
            "early_best": self.early_best,
            "early_bad_epochs": int(self.early_bad_epochs),
            "best_oracle_pdms": self.best_oracle_pdms,
        }

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        self.epoch = int(state_dict.get("epoch", 0))
        self.completed_steps = int(state_dict.get("completed_steps", 0))
        value = state_dict.get("early_best")
        self.early_best = None if value is None else float(value)
        self.early_bad_epochs = int(state_dict.get("early_bad_epochs", 0))
        oracle_value = state_dict.get("best_oracle_pdms")
        self.best_oracle_pdms = (
            None if oracle_value is None else float(oracle_value)
        )


def optimizer_steps_per_epoch(
    dataloader_length: int, gradient_accumulation_steps: int
) -> int:
    return math.ceil(dataloader_length / max(1, gradient_accumulation_steps))


def parameter_count(module: nn.Module, *, trainable_only: bool = False) -> int:
    return sum(
        parameter.numel()
        for parameter in module.parameters()
        if not trainable_only or parameter.requires_grad
    )


def atomic_json(path: os.PathLike[str] | str, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(dict(payload), stream, indent=2, sort_keys=True)
    os.replace(temporary, path)


def freeze_module(module: nn.Module) -> nn.Module:
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    return module


_BANK_ONLY_DONOR_IDENTITIES = {
    "drivor_offline_bank_v1": {
        "paper": "DrivoR: Driving on Registers",
        "repository": "https://github.com/valeoai/DrivoR",
        "revision": "f02665403df799c1b4ddd8b0d34e073f0555c13a",
    },
    "clover_pdms_value_bank_v1": {
        "paper": "CLOVER: Closed-Loop Value Estimation and Ranking",
        "repository": "https://github.com/WilliamXuanYu/CLOVER",
        "revision": "6aba8b7b08a6b2cdba1ecee9325e51a544dd64c3",
    },
    "drivesuprim_dynamic_bank_v1": {
        "paper": (
            "DriveSuprim: Towards Precise Trajectory Selection for End-to-End "
            "Planning"
        ),
        "repository": "https://github.com/William-Yao-2000/DriveSuprim",
        "revision": "80fe792d7654a596d92e20d030d1650f6f605c02",
    },
    "drivesuprim_hybrid_bank_v1": {
        "paper": (
            "DriveSuprim: Towards Precise Trajectory Selection for End-to-End "
            "Planning"
        ),
        "repository": "https://github.com/William-Yao-2000/DriveSuprim",
        "revision": "80fe792d7654a596d92e20d030d1650f6f605c02",
    },
}

_BANK_ONLY_DONOR_RECIPES = {
    "drivor_offline_bank_v1": {
        "optimizer": "AdamW",
        "learning_rate": 2.0e-4,
        "global_batch_size": 64,
        "warmup_ratio": 0.10,
        "navsim_v2_epochs": 10,
        "navsim_v2_aggregate_weights": {
            "noc": 10.0,
            "dac": 13.0,
            "ddc": 6.0,
            "ttc": 14.0,
            "ep": 15.0,
            "comfort": 2.0,
        },
    },
    "clover_pdms_value_bank_v1": {
        "optimizer": "AdamW",
        "learning_rate": 3.0e-5,
        "global_batch_size": 32,
        "cycles": 30,
        "critic_epochs_per_cycle": 1,
        "generator_epochs_per_cycle": 1,
        "starting_phase": "scorer_fitting",
    },
    "drivesuprim_dynamic_bank_v1": {
        "optimizer": "Adam",
        "learning_rate": 7.5e-5,
        "global_batch_size": 64,
        "epochs_vit": 6,
        "epochs_cnn": 10,
        "refinement_layers": 3,
        "coarse_topk": 256,
    },
    "drivesuprim_hybrid_bank_v1": {
        "optimizer": "Adam",
        "learning_rate": 7.5e-5,
        "global_batch_size": 64,
        "epochs_vit": 6,
        "epochs_cnn": 10,
        "refinement_layers": 3,
        "coarse_topk": 256,
    },
}


def validate_bank_only_training_profile(
    config: Mapping[str, Any], *, expected_name: str
) -> Mapping[str, Any]:
    """Reject accidental reuse of the Stage-G/main-model training recipe."""

    if config.get("framework") is not None:
        raise ValueError(
            "bank-only scorer/refiner configs must not contain a Qwen framework"
        )
    profile = config.get("training_profile")
    if not isinstance(profile, Mapping):
        raise ValueError("bank-only training requires a training_profile mapping")
    if str(profile.get("name", "")) != expected_name:
        raise ValueError(
            f"expected training_profile.name={expected_name!r}, found "
            f"{profile.get('name')!r}"
        )
    if not bool(profile.get("independent_from_stage_g", False)):
        raise ValueError("bank-only training profile must be independent from Stage G")
    donor = profile.get("donor")
    if not isinstance(donor, Mapping):
        raise ValueError("training profile must record its donor reference")
    expected_donor = _BANK_ONLY_DONOR_IDENTITIES.get(expected_name)
    if expected_donor is None:
        raise ValueError(f"unknown bank-only training profile {expected_name!r}")
    for field_name, expected_value in expected_donor.items():
        actual_value = str(donor.get(field_name, ""))
        if actual_value != expected_value:
            raise ValueError(
                f"training profile donor {field_name} mismatch: expected "
                f"{expected_value!r}, found {actual_value!r}"
            )
    reference_recipe = donor.get("reference_recipe")
    if not isinstance(reference_recipe, Mapping):
        raise ValueError("training profile donor requires a reference_recipe")
    for field_name, expected_value in _BANK_ONLY_DONOR_RECIPES[
        expected_name
    ].items():
        actual_value = reference_recipe.get(field_name)
        if actual_value != expected_value:
            raise ValueError(
                f"training profile donor recipe {field_name} mismatch: expected "
                f"{expected_value!r}, found {actual_value!r}"
            )
    optimizer = config.get("optimizer")
    scheduler = config.get("scheduler")
    if not isinstance(optimizer, Mapping) or str(optimizer.get("name")) != "AdamW":
        raise ValueError("Register bank-only stages require their own AdamW optimizer")
    if not isinstance(scheduler, Mapping) or str(scheduler.get("type")) != "cosine":
        raise ValueError("Register bank-only stages require their own cosine scheduler")
    return profile
