"""Small scene-aware MLP for the predicted-consequence feasibility gate."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn


def seed_torch(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ConsequenceMLP(nn.Module):
    def __init__(
        self, input_dim: int, output_dim: int, hidden_dims: tuple[int, ...]
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        previous = input_dim
        for hidden in hidden_dims:
            layers.extend(
                [
                    nn.Linear(previous, hidden),
                    nn.LayerNorm(hidden),
                    nn.GELU(),
                ]
            )
            previous = hidden
        layers.append(nn.Linear(previous, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


def scene_inverse(tokens: np.ndarray, device: torch.device) -> torch.Tensor:
    _, inverse = np.unique(np.asarray(tokens), return_inverse=True)
    return torch.as_tensor(inverse, dtype=torch.long, device=device)


def center_by_scene(
    values: torch.Tensor, inverse: torch.Tensor
) -> torch.Tensor:
    scene_count = int(inverse.max().item()) + 1
    sums = torch.zeros(
        (scene_count, values.shape[1]),
        dtype=values.dtype,
        device=values.device,
    )
    sums.index_add_(0, inverse, values)
    counts = torch.bincount(inverse, minlength=scene_count).to(values.dtype)
    means = sums / counts[:, None].clamp_min(1.0)
    return values - means[inverse]


@dataclass
class Normalization:
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    target_mean: np.ndarray
    target_scale: np.ndarray
    target_min: np.ndarray
    target_max: np.ndarray
    delta_scale_normalized: np.ndarray
    delta_active: np.ndarray


def normalization_for(
    features: np.ndarray,
    targets: np.ndarray,
    scene_tokens: np.ndarray,
) -> Normalization:
    feature_mean = np.mean(features, axis=0, dtype=np.float64)
    feature_scale = np.std(features, axis=0, dtype=np.float64)
    feature_scale[feature_scale < 1e-5] = 1.0
    target_mean = np.mean(targets, axis=0, dtype=np.float64)
    target_scale = np.std(targets, axis=0, dtype=np.float64)
    target_scale[target_scale < 1e-6] = 1.0
    normalized_target = (targets - target_mean) / target_scale
    centered = np.empty_like(normalized_target, dtype=np.float64)
    for token in np.unique(scene_tokens):
        mask = scene_tokens == token
        centered[mask] = normalized_target[mask] - np.mean(
            normalized_target[mask], axis=0
        )
    delta_scale = np.std(centered, axis=0, dtype=np.float64)
    delta_active = delta_scale >= 0.01
    delta_scale = np.maximum(delta_scale, 0.05)
    return Normalization(
        feature_mean=feature_mean.astype(np.float32),
        feature_scale=feature_scale.astype(np.float32),
        target_mean=target_mean.astype(np.float32),
        target_scale=target_scale.astype(np.float32),
        target_min=np.min(targets, axis=0).astype(np.float32),
        target_max=np.max(targets, axis=0).astype(np.float32),
        delta_scale_normalized=delta_scale.astype(np.float32),
        delta_active=delta_active,
    )


def normalized_tensors(
    features: np.ndarray,
    targets: np.ndarray,
    scene_tokens: np.ndarray,
    normalization: Normalization,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = np.clip(
        (features - normalization.feature_mean)
        / normalization.feature_scale,
        -12.0,
        12.0,
    )
    y = (
        (targets - normalization.target_mean)
        / normalization.target_scale
    )
    return (
        torch.as_tensor(x, dtype=torch.float32, device=device),
        torch.as_tensor(y, dtype=torch.float32, device=device),
        scene_inverse(scene_tokens, device),
    )


def effect_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    inverse: torch.Tensor,
    delta_scale: torch.Tensor,
    delta_active: torch.Tensor,
    delta_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    raw = torch.mean((prediction - target) ** 2)
    if delta_weight <= 0 or not bool(torch.any(delta_active)):
        delta = torch.zeros((), dtype=raw.dtype, device=raw.device)
    else:
        prediction_delta = center_by_scene(prediction, inverse)
        target_delta = center_by_scene(target, inverse)
        normalized_error = (
            prediction_delta[:, delta_active]
            - target_delta[:, delta_active]
        ) / delta_scale[delta_active]
        delta = torch.mean(normalized_error**2)
    return raw + float(delta_weight) * delta, raw, delta


@dataclass
class TorchEffectPredictor:
    model: ConsequenceMLP
    normalization: Normalization
    device: torch.device

    @torch.inference_mode()
    def predict(self, features: np.ndarray) -> np.ndarray:
        self.model.eval()
        normalized = np.clip(
            (features - self.normalization.feature_mean)
            / self.normalization.feature_scale,
            -12.0,
            12.0,
        )
        output = []
        for start in range(0, len(normalized), 2048):
            tensor = torch.as_tensor(
                normalized[start : start + 2048],
                dtype=torch.float32,
                device=self.device,
            )
            output.append(self.model(tensor).cpu().numpy())
        prediction = np.concatenate(output, axis=0)
        prediction = (
            prediction * self.normalization.target_scale
            + self.normalization.target_mean
        )
        prediction = np.clip(
            prediction,
            self.normalization.target_min,
            self.normalization.target_max,
        )
        return np.round(prediction, decimals=6).astype(np.float32)


def _loss_values(
    model: ConsequenceMLP,
    x: torch.Tensor,
    y: torch.Tensor,
    inverse: torch.Tensor,
    delta_scale: torch.Tensor,
    delta_active: torch.Tensor,
    delta_weight: float,
) -> tuple[float, float, float]:
    with torch.inference_mode():
        total, raw, delta = effect_loss(
            model(x),
            y,
            inverse,
            delta_scale,
            delta_active,
            delta_weight,
        )
    return float(total.cpu()), float(raw.cpu()), float(delta.cpu())


def tune_epochs(
    features: np.ndarray,
    targets: np.ndarray,
    scene_tokens: np.ndarray,
    train_indices: np.ndarray,
    monitor_indices: np.ndarray,
    *,
    hidden_dims: tuple[int, ...],
    delta_weight: float,
    learning_rate: float,
    weight_decay: float,
    max_epochs: int,
    patience: int,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    """Select an epoch on inner logs; held-out OOF logs are not touched."""

    seed_torch(seed)
    normalization = normalization_for(
        features[train_indices],
        targets[train_indices],
        scene_tokens[train_indices],
    )
    x_train, y_train, inverse_train = normalized_tensors(
        features[train_indices],
        targets[train_indices],
        scene_tokens[train_indices],
        normalization,
        device,
    )
    x_monitor, y_monitor, inverse_monitor = normalized_tensors(
        features[monitor_indices],
        targets[monitor_indices],
        scene_tokens[monitor_indices],
        normalization,
        device,
    )
    model = ConsequenceMLP(
        features.shape[1], targets.shape[1], hidden_dims
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    delta_scale = torch.as_tensor(
        normalization.delta_scale_normalized,
        dtype=torch.float32,
        device=device,
    )
    delta_active = torch.as_tensor(
        normalization.delta_active, dtype=torch.bool, device=device
    )
    best_loss = float("inf")
    best_epoch = -1
    stale = 0
    history: list[dict[str, float | int]] = []
    for epoch in range(int(max_epochs)):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total, raw, delta = effect_loss(
            model(x_train),
            y_train,
            inverse_train,
            delta_scale,
            delta_active,
            delta_weight,
        )
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        model.eval()
        monitor_total, monitor_raw, monitor_delta = _loss_values(
            model,
            x_monitor,
            y_monitor,
            inverse_monitor,
            delta_scale,
            delta_active,
            delta_weight,
        )
        record = {
            "epoch": epoch,
            "train_total": float(total.detach().cpu()),
            "train_raw": float(raw.detach().cpu()),
            "train_delta": float(delta.detach().cpu()),
            "monitor_total": monitor_total,
            "monitor_raw": monitor_raw,
            "monitor_delta": monitor_delta,
        }
        history.append(record)
        if monitor_total < best_loss - 1e-5:
            best_loss = monitor_total
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
            if stale >= int(patience):
                break
    if best_epoch < 0:
        raise RuntimeError("MLP monitor loss never became finite")
    initial = float(history[0]["monitor_total"])
    best_record = history[best_epoch]
    return {
        "best_epoch": int(best_epoch),
        "selected_epochs": int(best_epoch + 1),
        "epochs_observed": int(len(history)),
        "stopped_early": bool(len(history) < int(max_epochs)),
        "initial_monitor_loss": initial,
        "best_monitor_loss": float(best_loss),
        "relative_monitor_improvement": (
            (initial - best_loss) / initial if initial > 0 else 0.0
        ),
        "train_loss_at_best": float(best_record["train_total"]),
        "monitor_train_gap_at_best": float(
            best_loss - float(best_record["train_total"])
        ),
        "delta_active_feature_count": int(
            np.sum(normalization.delta_active)
        ),
        "history": history,
    }


def fit_fixed_epochs(
    features: np.ndarray,
    targets: np.ndarray,
    scene_tokens: np.ndarray,
    train_indices: np.ndarray,
    *,
    hidden_dims: tuple[int, ...],
    delta_weight: float,
    learning_rate: float,
    weight_decay: float,
    epochs: int,
    seed: int,
    device: torch.device,
) -> tuple[TorchEffectPredictor, list[dict[str, float | int]]]:
    """Refit on every allowed training log for a preselected epoch count."""

    seed_torch(seed)
    normalization = normalization_for(
        features[train_indices],
        targets[train_indices],
        scene_tokens[train_indices],
    )
    x, y, inverse = normalized_tensors(
        features[train_indices],
        targets[train_indices],
        scene_tokens[train_indices],
        normalization,
        device,
    )
    model = ConsequenceMLP(
        features.shape[1], targets.shape[1], hidden_dims
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    delta_scale = torch.as_tensor(
        normalization.delta_scale_normalized,
        dtype=torch.float32,
        device=device,
    )
    delta_active = torch.as_tensor(
        normalization.delta_active, dtype=torch.bool, device=device
    )
    history: list[dict[str, float | int]] = []
    for epoch in range(int(epochs)):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total, raw, delta = effect_loss(
            model(x),
            y,
            inverse,
            delta_scale,
            delta_active,
            delta_weight,
        )
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        history.append(
            {
                "epoch": epoch,
                "train_total": float(total.detach().cpu()),
                "train_raw": float(raw.detach().cpu()),
                "train_delta": float(delta.detach().cpu()),
            }
        )
    model.eval()
    return TorchEffectPredictor(model, normalization, device), history
