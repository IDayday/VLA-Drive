"""Cross-entropy losses and multiclass metrics for Retrieve Model V1."""

from typing import Dict, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def multiclass_metrics(
    logits: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
) -> Dict[str, torch.Tensor]:
    """Compute foreground class and macro metrics over a complete batch."""
    if logits.ndim != 4 or logits.shape[1] != num_classes:
        raise ValueError(
            f"Expected logits [B,{num_classes},H,W], got {tuple(logits.shape)}"
        )
    if target.shape != (logits.shape[0], logits.shape[2], logits.shape[3]):
        raise ValueError(
            f"Target must be [B,H,W], got {tuple(target.shape)} for "
            f"logits {tuple(logits.shape)}"
        )

    prediction = logits.argmax(dim=1)
    epsilon = torch.tensor(1e-6, device=logits.device)
    result: Dict[str, torch.Tensor] = {
        "accuracy": (prediction == target).float().mean(),
    }
    foreground_iou = []
    foreground_precision = []
    foreground_recall = []
    foreground_f1 = []

    for class_index in range(1, num_classes):
        predicted_class = prediction == class_index
        target_class = target == class_index
        true_positive = (predicted_class & target_class).sum().float()
        false_positive = (predicted_class & ~target_class).sum().float()
        false_negative = (~predicted_class & target_class).sum().float()
        union = (predicted_class | target_class).sum().float()

        precision = true_positive / (true_positive + false_positive + epsilon)
        recall = true_positive / (true_positive + false_negative + epsilon)
        iou = true_positive / (union + epsilon)
        f1 = 2.0 * precision * recall / (precision + recall + epsilon)
        prefix = f"class_{class_index}"
        result[f"{prefix}_iou"] = iou
        result[f"{prefix}_precision"] = precision
        result[f"{prefix}_recall"] = recall
        result[f"{prefix}_f1"] = f1
        result[f"{prefix}_target_ratio"] = target_class.float().mean()
        result[f"{prefix}_prediction_ratio"] = predicted_class.float().mean()
        foreground_iou.append(iou)
        foreground_precision.append(precision)
        foreground_recall.append(recall)
        foreground_f1.append(f1)

    result["iou"] = torch.stack(foreground_iou).mean()
    result["precision"] = torch.stack(foreground_precision).mean()
    result["recall"] = torch.stack(foreground_recall).mean()
    result["f1"] = torch.stack(foreground_f1).mean()
    return result


class RetrieveLoss(nn.Module):
    """Weighted cross entropy for independent Map and Agent semantic targets."""

    def __init__(
        self,
        map_class_weights: Sequence[float],
        agent_class_weights: Sequence[float],
        alpha: float = 1.0,
        class_weight_cap: float = 50.0,
        map_num_classes: int = 4,
        agent_num_classes: int = 3,
        distance_weighting_enabled: bool = False,
        map_distance_strength: float = 1.0,
        agent_distance_strength: float = 1.0,
        distance_scale_m: float = 8.0,
        distance_weight_cap: float = 3.0,
        near_radius_m: float = 12.0,
        bev_pixel_height: int = 128,
        bev_pixel_width: int = 256,
        bev_pixel_size: float = 0.25,
    ) -> None:
        super().__init__()
        self.map_num_classes = int(map_num_classes)
        self.agent_num_classes = int(agent_num_classes)
        map_weights = self._validate_weights(
            map_class_weights,
            expected_classes=self.map_num_classes,
            cap=class_weight_cap,
            branch="map",
        )
        agent_weights = self._validate_weights(
            agent_class_weights,
            expected_classes=self.agent_num_classes,
            cap=class_weight_cap,
            branch="agent",
        )
        self.alpha = float(alpha)
        self.class_weight_cap = float(class_weight_cap)
        self.distance_weighting_enabled = bool(distance_weighting_enabled)
        self.near_radius_m = float(near_radius_m)
        self.bev_pixel_height = int(bev_pixel_height)
        self.bev_pixel_width = int(bev_pixel_width)
        self.bev_pixel_size = float(bev_pixel_size)
        if distance_scale_m <= 0 or distance_weight_cap < 1.0:
            raise ValueError("Distance scale must be positive and cap must be >= 1")
        if map_distance_strength < 0 or agent_distance_strength < 0:
            raise ValueError("Distance strengths must be non-negative")
        if self.near_radius_m <= 0 or self.bev_pixel_size <= 0:
            raise ValueError("Near radius and BEV pixel size must be positive")
        self.register_buffer("map_class_weights", map_weights)
        self.register_buffer("agent_class_weights", agent_weights)
        distances = self._build_distance_grid(
            self.bev_pixel_height,
            self.bev_pixel_width,
            self.bev_pixel_size,
        )
        self.register_buffer("distance_meters", distances)
        self.register_buffer(
            "map_distance_weights",
            self._build_inverse_distance_weights(
                distances,
                strength=float(map_distance_strength),
                scale_m=float(distance_scale_m),
                cap=float(distance_weight_cap),
            ),
        )
        self.register_buffer(
            "agent_distance_weights",
            self._build_inverse_distance_weights(
                distances,
                strength=float(agent_distance_strength),
                scale_m=float(distance_scale_m),
                cap=float(distance_weight_cap),
            ),
        )

    @staticmethod
    def _build_distance_grid(
        height: int,
        width: int,
        pixel_size: float,
    ) -> torch.Tensor:
        if height <= 0 or width <= 0:
            raise ValueError("BEV dimensions must be positive")
        forward = (torch.arange(height, dtype=torch.float32) + 0.5) * pixel_size
        lateral = (
            torch.arange(width, dtype=torch.float32) + 0.5 - width / 2.0
        ) * pixel_size
        return torch.sqrt(forward[:, None].square() + lateral[None, :].square())

    @staticmethod
    def _build_inverse_distance_weights(
        distances: torch.Tensor,
        strength: float,
        scale_m: float,
        cap: float,
    ) -> torch.Tensor:
        # Bounded reciprocal decay: nearby cells receive more relative weight,
        # while the value remains finite at the ego origin.
        weights = 1.0 + strength / (1.0 + distances / scale_m)
        return weights.clamp(max=cap)

    @staticmethod
    def _weighted_cross_entropy(
        logits: torch.Tensor,
        target: torch.Tensor,
        class_weights: torch.Tensor,
        spatial_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pixel_loss = F.cross_entropy(
            logits,
            target,
            weight=class_weights,
            reduction="none",
        )
        spatial_weights = spatial_weights.to(
            device=logits.device,
            dtype=pixel_loss.dtype,
        )
        target_weights = class_weights[target].to(dtype=pixel_loss.dtype)
        denominator = (target_weights * spatial_weights).sum().clamp_min(1e-6)
        return (pixel_loss * spatial_weights).sum() / denominator, pixel_loss

    @staticmethod
    def _region_cross_entropy(
        pixel_loss: torch.Tensor,
        target: torch.Tensor,
        class_weights: torch.Tensor,
        region_mask: torch.Tensor,
    ) -> torch.Tensor:
        mask = region_mask.to(device=pixel_loss.device, dtype=pixel_loss.dtype)
        mask = mask.unsqueeze(0).expand_as(pixel_loss)
        target_weights = class_weights[target].to(dtype=pixel_loss.dtype)
        denominator = (target_weights * mask).sum().clamp_min(1e-6)
        return (pixel_loss * mask).sum() / denominator

    @staticmethod
    def _validate_weights(
        weights: Sequence[float],
        expected_classes: int,
        cap: float,
        branch: str,
    ) -> torch.Tensor:
        values = torch.as_tensor(weights, dtype=torch.float32)
        if values.shape != (expected_classes,):
            raise ValueError(
                f"{branch} class weights must contain {expected_classes} values, "
                f"got {values.tolist()}"
            )
        if not torch.isfinite(values).all() or (values <= 0).any():
            raise ValueError(f"{branch} class weights must be finite and positive")
        return values.clamp(max=float(cap))

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        map_logits = predictions["map_bev_logits"]
        agent_logits = predictions["agent_bev_logits"]
        map_target = targets["map_target"].long()
        agent_target = targets["agent_target"].long()
        expected_shape = (self.bev_pixel_height, self.bev_pixel_width)
        if map_logits.shape[-2:] != expected_shape or agent_logits.shape[-2:] != expected_shape:
            raise ValueError(
                f"Distance-aware loss expects BEV {expected_shape}, got "
                f"map={tuple(map_logits.shape[-2:])}, "
                f"agent={tuple(agent_logits.shape[-2:])}"
            )

        map_class_weights = self.map_class_weights.to(
            device=map_logits.device,
            dtype=map_logits.dtype,
        )
        agent_class_weights = self.agent_class_weights.to(
            device=agent_logits.device,
            dtype=agent_logits.dtype,
        )
        if self.distance_weighting_enabled:
            map_spatial_weights = self.map_distance_weights
            agent_spatial_weights = self.agent_distance_weights
        else:
            map_spatial_weights = torch.ones_like(self.map_distance_weights)
            agent_spatial_weights = torch.ones_like(self.agent_distance_weights)

        map_ce, map_pixel_loss = self._weighted_cross_entropy(
            map_logits,
            map_target,
            map_class_weights,
            map_spatial_weights,
        )
        agent_ce, agent_pixel_loss = self._weighted_cross_entropy(
            agent_logits,
            agent_target,
            agent_class_weights,
            agent_spatial_weights,
        )
        near_mask = self.distance_meters <= self.near_radius_m
        far_mask = ~near_mask
        result = {
            "loss": map_ce + self.alpha * agent_ce,
            "map_ce": map_ce,
            "agent_ce": agent_ce,
            "map_near_ce": self._region_cross_entropy(
                map_pixel_loss, map_target, map_class_weights, near_mask
            ),
            "map_far_ce": self._region_cross_entropy(
                map_pixel_loss, map_target, map_class_weights, far_mask
            ),
            "agent_near_ce": self._region_cross_entropy(
                agent_pixel_loss, agent_target, agent_class_weights, near_mask
            ),
            "agent_far_ce": self._region_cross_entropy(
                agent_pixel_loss, agent_target, agent_class_weights, far_mask
            ),
        }
        for name, value in multiclass_metrics(
            map_logits.detach(),
            map_target,
            num_classes=self.map_num_classes,
        ).items():
            result[f"map_{name}"] = value
        for name, value in multiclass_metrics(
            agent_logits.detach(),
            agent_target,
            num_classes=self.agent_num_classes,
        ).items():
            result[f"agent_{name}"] = value
        return result
