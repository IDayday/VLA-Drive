"""Deterministic training-only trajectory perturbations for consequence probes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch


@dataclass(frozen=True)
class PerturbedTrajectories:
    """Physical candidates ``[B,K,H,3]`` and stable source names."""

    physical: torch.Tensor
    source_names: Tuple[str, ...]


def build_consequence_perturbations(ground_truth: torch.Tensor) -> PerturbedTrajectories:
    """Build fixed perturbations around GT without assigning counterfactual truth."""

    if ground_truth.ndim == 2:
        ground_truth = ground_truth[None]
    if ground_truth.ndim != 3 or ground_truth.shape[-1] != 3:
        raise ValueError("ground_truth must have shape [B,H,3]")
    if ground_truth.shape[1] < 2:
        raise ValueError("trajectory horizon must be at least two")
    base = ground_truth.float()
    heading = base[..., 2]
    normal = torch.stack((-torch.sin(heading), torch.cos(heading)), dim=-1)
    progress = torch.linspace(
        0.0, 1.0, base.shape[1], device=base.device, dtype=base.dtype
    ).reshape(1, -1, 1)

    candidates = [base]
    names = ["gt_reference"]
    for offset in (-1.0, -0.5, 0.5, 1.0):
        shifted = base.clone()
        shifted[..., :2] = shifted[..., :2] + normal * float(offset)
        candidates.append(shifted)
        names.append(f"lateral_{offset:+.1f}m")
    for scale in (0.7, 0.85, 1.15):
        scaled = base.clone()
        scaled[..., :2] = base[..., :2] * float(scale)
        candidates.append(scaled)
        names.append(f"progress_x{scale:.2f}")
    for angle in (-0.15, 0.15):
        curved = base.clone()
        delta_heading = progress[..., 0] * float(angle)
        curved[..., 2] = torch.atan2(
            torch.sin(heading + delta_heading),
            torch.cos(heading + delta_heading),
        )
        curved[..., :2] = curved[..., :2] + normal * (
            progress * float(angle) * 4.0
        )
        candidates.append(curved)
        names.append(f"curvature_{angle:+.2f}rad")
    braking = base.clone()
    braking[..., :2] = base[..., :2] * (1.0 - 0.55 * progress)
    candidates.append(braking)
    names.append("extra_braking")
    return PerturbedTrajectories(torch.stack(candidates, dim=1), tuple(names))
