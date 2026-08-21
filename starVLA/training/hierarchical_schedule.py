"""Optimizer-step curriculum for the joint hierarchical planner."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HierarchicalTrainingSchedule:
    """Loss weights and candidate counts for one completed optimizer step."""

    progress: float
    dynamic_enabled: bool
    num_dynamic_candidates: int
    dynamic_topm: int
    lambda_flow: float
    lambda_drivor: float
    lambda_suprim_coarse: float
    lambda_suprim_fine: float


def build_hierarchical_schedule(
    completed_steps: int,
    max_train_steps: int,
    *,
    static_only_end: float = 0.10,
    dynamic_ramp_end: float = 0.20,
    num_dynamic_candidates: int = 64,
    dynamic_topm_start: int = 64,
    dynamic_topm_end: int = 32,
    lambda_flow: float = 1.0,
    lambda_drivor: float = 1.0,
    lambda_suprim_coarse: float = 1.0,
    lambda_suprim_fine: float = 1.0,
) -> HierarchicalTrainingSchedule:
    """Build curriculum state from completed optimizer steps, not micro-batches."""

    if max_train_steps <= 0 or completed_steps < 0:
        raise ValueError("training step counts must be non-negative with max > 0")
    if not (0.0 <= static_only_end < dynamic_ramp_end <= 1.0):
        raise ValueError("curriculum boundaries must satisfy 0 <= static < ramp <= 1")
    if not (0 < dynamic_topm_end <= dynamic_topm_start <= num_dynamic_candidates):
        raise ValueError("dynamic Top-M bounds must lie inside candidate count")
    progress = min(float(completed_steps) / float(max_train_steps), 1.0)
    if progress < static_only_end:
        return HierarchicalTrainingSchedule(
            progress=progress,
            dynamic_enabled=False,
            num_dynamic_candidates=0,
            dynamic_topm=0,
            lambda_flow=float(lambda_flow),
            lambda_drivor=0.0,
            lambda_suprim_coarse=float(lambda_suprim_coarse),
            lambda_suprim_fine=float(lambda_suprim_fine),
        )
    if progress < dynamic_ramp_end:
        alpha = (progress - static_only_end) / (dynamic_ramp_end - static_only_end)
        topm = round(dynamic_topm_start + alpha * (dynamic_topm_end - dynamic_topm_start))
        topm = min(dynamic_topm_start, max(dynamic_topm_end, topm))
        return HierarchicalTrainingSchedule(
            progress=progress,
            dynamic_enabled=True,
            num_dynamic_candidates=int(num_dynamic_candidates),
            dynamic_topm=int(topm),
            lambda_flow=float(lambda_flow),
            lambda_drivor=float(lambda_drivor) * alpha,
            lambda_suprim_coarse=float(lambda_suprim_coarse),
            lambda_suprim_fine=float(lambda_suprim_fine),
        )
    return HierarchicalTrainingSchedule(
        progress=progress,
        dynamic_enabled=True,
        num_dynamic_candidates=int(num_dynamic_candidates),
        dynamic_topm=int(dynamic_topm_end),
        lambda_flow=float(lambda_flow),
        lambda_drivor=float(lambda_drivor),
        lambda_suprim_coarse=float(lambda_suprim_coarse),
        lambda_suprim_fine=float(lambda_suprim_fine),
    )
