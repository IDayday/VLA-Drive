"""Exactly matched-capacity factor probes and deterministic input adapters."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F


COMMON_INPUT_DIM = 1024
BASE_RAW_DIM = 64
CURRENT_RAW_DIM = 256
AUXILIARY_RAW_DIM = 768
BASE_OUTPUT_DIM = 256
CURRENT_OUTPUT_DIM = 256
AUXILIARY_OUTPUT_DIM = 512
FACTOR_NAMES = ("NC", "DAC", "EP", "TTC", "Comfort")
SAFETY_FACTOR_INDICES = (0, 1, 3)
CONTINUOUS_FACTOR_INDICES = (2, 4)


def _stable_seed(name: str, seed: int) -> int:
    digest = hashlib.sha256(f"{name}:{seed}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**31)


def _projection(input_dim: int, output_dim: int, name: str, seed: int) -> Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(_stable_seed(name, seed))
    matrix = torch.randn(input_dim, output_dim, generator=generator, dtype=torch.float32)
    return matrix / math.sqrt(float(input_dim))


def pad_last_dimension(value: Tensor, target: int, name: str) -> Tensor:
    """Right-pad a tensor to a fixed schema, failing instead of truncating."""

    if value.ndim < 2:
        raise ValueError(f"{name} must have a feature dimension, got {tuple(value.shape)}")
    actual = value.shape[-1]
    if actual > target:
        raise ValueError(f"{name} has {actual} features, fixed limit is {target}")
    if actual == target:
        return value
    return F.pad(value, (0, target - actual))


class MatchedInputComposer(nn.Module):
    """Frozen random adapters with identical schemas for every scorer.

    Every method receives base, current, and auxiliary slots of the same raw
    widths. Missing information is represented by zeros. The matrices are
    buffers rather than parameters, so all scorer variants have exactly the
    same trainable capacity.
    """

    def __init__(self, seed: int = 20260827):
        super().__init__()
        self.register_buffer(
            "base_projection",
            _projection(BASE_RAW_DIM, BASE_OUTPUT_DIM, "base", seed),
            persistent=True,
        )
        self.register_buffer(
            "current_projection",
            _projection(CURRENT_RAW_DIM, CURRENT_OUTPUT_DIM, "current", seed),
            persistent=True,
        )
        self.register_buffer(
            "auxiliary_projection",
            _projection(AUXILIARY_RAW_DIM, AUXILIARY_OUTPUT_DIM, "auxiliary", seed),
            persistent=True,
        )

    @staticmethod
    def _stabilize(value: Tensor) -> Tensor:
        if not torch.isfinite(value).all():
            raise ValueError("probe input contains NaN/Inf")
        return torch.asinh(value.float().clamp(min=-1.0e4, max=1.0e4))

    def forward(self, base: Tensor, current: Tensor, auxiliary: Tensor) -> Tensor:
        base = pad_last_dimension(base, BASE_RAW_DIM, "base input")
        current = pad_last_dimension(current, CURRENT_RAW_DIM, "current input")
        auxiliary = pad_last_dimension(auxiliary, AUXILIARY_RAW_DIM, "auxiliary input")
        if base.shape[:-1] != current.shape[:-1] or base.shape[:-1] != auxiliary.shape[:-1]:
            raise ValueError(
                "base/current/auxiliary inputs must share scene and candidate axes"
            )
        composed = torch.cat(
            [
                self._stabilize(base) @ self.base_projection,
                self._stabilize(current) @ self.current_projection,
                self._stabilize(auxiliary) @ self.auxiliary_projection,
            ],
            dim=-1,
        )
        if composed.shape[-1] != COMMON_INPUT_DIM:
            raise AssertionError(f"composed feature shape is {tuple(composed.shape)}")
        return composed


class MatchedCapacityFactorProbe(nn.Module):
    """One shared factor head used for every G2/G3 frozen representation."""

    def __init__(self, hidden_dim: int = 256, input_dim: int = COMMON_INPUT_DIM):
        super().__init__()
        if hidden_dim <= 0 or input_dim != COMMON_INPUT_DIM:
            raise ValueError("probe requires positive hidden_dim and fixed 1024D input")
        self.input_norm = nn.LayerNorm(input_dim)
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.factor_head = nn.Linear(hidden_dim, len(FACTOR_NAMES))

    def forward(self, value: Tensor) -> Mapping[str, Tensor]:
        if value.ndim != 3 or value.shape[-1] != COMMON_INPUT_DIM:
            raise ValueError(f"probe input must be [scene,candidate,1024], got {tuple(value.shape)}")
        logits = self.factor_head(self.trunk(self.input_norm(value)))
        factors = torch.sigmoid(logits)
        nc, dac, ep, ttc, comfort = factors.unbind(dim=-1)
        score = nc * dac * (5.0 * ep + 5.0 * ttc + 2.0 * comfort) / 12.0
        return {"logits": logits, "factors": factors, "score": score}


def factorized_probe_loss(logits: Tensor, targets: Tensor) -> Tensor:
    if logits.shape != targets.shape or logits.shape[-1] != 5:
        raise ValueError(
            f"factor loss expects matching [...,5], got {tuple(logits.shape)} and {tuple(targets.shape)}"
        )
    if not torch.isfinite(targets).all():
        raise ValueError("factor targets contain NaN/Inf")
    safety = F.binary_cross_entropy_with_logits(
        logits[..., list(SAFETY_FACTOR_INDICES)],
        targets[..., list(SAFETY_FACTOR_INDICES)],
    )
    continuous = F.smooth_l1_loss(
        torch.sigmoid(logits[..., list(CONTINUOUS_FACTOR_INDICES)]),
        targets[..., list(CONTINUOUS_FACTOR_INDICES)],
    )
    return safety + continuous


def pairwise_ranking_loss(
    predictions: Tensor,
    targets: Tensor,
    max_pairs_per_scene: int = 4096,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Logistic pairwise loss, sampling pairs independently inside each scene."""

    if predictions.shape != targets.shape or predictions.ndim != 2:
        raise ValueError("pairwise loss requires matching [scene,candidate] tensors")
    losses: list[Tensor] = []
    candidates = predictions.shape[1]
    for scene_index in range(predictions.shape[0]):
        if candidates < 2:
            continue
        pair_count = min(max_pairs_per_scene, candidates * (candidates - 1) // 2)
        first = torch.randint(
            candidates,
            (pair_count,),
            generator=generator,
            device=predictions.device,
        )
        second = torch.randint(
            candidates,
            (pair_count,),
            generator=generator,
            device=predictions.device,
        )
        valid = first != second
        target_delta = targets[scene_index, first] - targets[scene_index, second]
        valid &= target_delta != 0
        if valid.any():
            predicted_delta = (
                predictions[scene_index, first[valid]]
                - predictions[scene_index, second[valid]]
            )
            sign = torch.sign(target_delta[valid])
            losses.append(F.softplus(-sign * predicted_delta).mean())
    if not losses:
        return predictions.sum() * 0.0
    return torch.stack(losses).mean()


@dataclass(frozen=True)
class ParameterAudit:
    trainable_parameters: int
    frozen_parameters: int
    approximate_flops_per_candidate: int


def audit_parameters(
    composer: MatchedInputComposer, probe: MatchedCapacityFactorProbe
) -> ParameterAudit:
    modules = (composer, probe)
    trainable = sum(
        parameter.numel()
        for module in modules
        for parameter in module.parameters()
        if parameter.requires_grad
    )
    frozen = sum(
        parameter.numel()
        for module in modules
        for parameter in module.parameters()
        if not parameter.requires_grad
    ) + sum(buffer.numel() for buffer in composer.buffers())
    linear_weights = sum(
        parameter.numel()
        for name, parameter in probe.named_parameters()
        if name.endswith("weight") and parameter.ndim == 2
    )
    projection_weights = sum(buffer.numel() for buffer in composer.buffers())
    return ParameterAudit(
        trainable_parameters=trainable,
        frozen_parameters=frozen,
        approximate_flops_per_candidate=2 * (linear_weights + projection_weights),
    )
