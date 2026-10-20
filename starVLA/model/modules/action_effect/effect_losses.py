"""Unit-aware losses and additive sufficient statistics for real effect labels.

Each pointwise task averages valid channels within a scene, then valid scenes.
Ranking preserves gap at distribution level, never normalizes within a scene.
Combine numerators/counts BEFORE reduction across microbatches/ranks.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import distributed as dist
from torch.nn import functional as F

from .effect_model import FeedbackLabelSpec


@dataclass
class LossContribution:
    numerator: torch.Tensor
    denominator: torch.Tensor

    def mean(self, *, distributed=False):
        count = self.denominator.detach().float().clone()
        world = 1
        if distributed and dist.is_available() and dist.is_initialized():
            dist.all_reduce(count)
            world = dist.get_world_size()
        # DDP averages gradients; scale local sums by world/global_count.
        return self.numerator.float() * (world / count.clamp_min(1.))


@dataclass
class EffectLossResult:
    total: torch.Tensor
    terms: dict[str, torch.Tensor]
    contributions: dict[str, LossContribution]
    statistics: dict[str, float]


def scene_contribution(values, valid):
    if values.shape != valid.shape or valid.dtype != torch.bool or values.ndim < 1:
        raise ValueError("Loss values/masks require matching scene-first shapes")
    if not torch.isfinite(values[valid]).all():
        raise ValueError("Nonfinite VALID prediction/label loss cannot be masked out")
    per_scene_count = valid.reshape(len(values), -1).sum(-1)
    clean = torch.where(valid, values.float(), torch.zeros_like(values, dtype=torch.float32))
    per_scene = clean.reshape(len(values), -1).sum(-1) / per_scene_count.clamp_min(1)
    return LossContribution(per_scene.sum(), (per_scene_count > 0).sum().float())


def combine_contributions(batches: list[Mapping[str, LossContribution]]):
    if not batches or any(set(b) != set(batches[0]) for b in batches):
        raise ValueError("Microbatches must carry the same named loss terms")
    return {name: LossContribution(sum(b[name].numerator for b in batches),
                                   sum(b[name].denominator for b in batches)) for name in batches[0]}


def reduce_contributions(contributions, weights, *, distributed=False, statistics=None):
    if set(contributions) != set(weights):
        raise ValueError("Explicit loss weights must exactly match the realized tasks")
    if any(not isinstance(w, (float, int)) or not torch.isfinite(torch.tensor(w)) or w < 0 for w in weights.values()):
        raise ValueError("Loss weights require finite nonnegative values")
    terms = {k: v.mean(distributed=distributed) for k, v in contributions.items()}
    total = sum(terms[k] * float(weights[k]) for k in terms)
    counts = {f"{k}_denominator": float(v.denominator.detach()) for k, v in contributions.items()}
    counts.update(statistics or {})
    return EffectLossResult(total, terms, dict(contributions), counts)


def ranking_contribution(predicted_delta, true_gap, valid, *, temperature=.1,
                         tie_epsilon=.01, sampling="uniform"):
    """K=2 whole-distribution objective, or its gap-sampled comparison stream.

    uniform: sum(|gap| * softplus)/sum(|gap|).
    gap_sampled: mean(softplus), because q already contains |gap|.
    Returned sums can be combined across accumulation before division.
    """
    if (predicted_delta.shape != true_gap.shape or valid.shape != true_gap.shape
            or true_gap.ndim != 1 or valid.dtype != torch.bool
            or temperature <= 0 or tie_epsilon < 0 or sampling not in ("uniform", "gap_sampled")):
        raise ValueError("Invalid ranking pair contract")
    if not torch.isfinite(true_gap[valid]).all() or not torch.isfinite(predicted_delta[valid]).all():
        raise ValueError("Nonfinite valid ranking pair")
    good = valid & (true_gap.abs() > tie_epsilon)
    gap = torch.where(good, true_gap.float(), torch.zeros_like(true_gap, dtype=torch.float32))
    prediction = torch.where(good, predicted_delta.float(), torch.zeros_like(predicted_delta, dtype=torch.float32))
    weight = good.float() if sampling == "gap_sampled" else gap.abs()
    loss = F.softplus(-gap.sign() * prediction / temperature)
    return LossContribution((loss * weight).sum(), weight.sum())


def effect_loss_contributions(prediction, *, before, after, before_valid, after_valid,
                              label_spec: FeedbackLabelSpec, mode: str,
                              event_targets=None, event_valid=None,
                              temporal_targets=None, temporal_valid=None,
                              repair_targets=None, repair_valid=None,
                              ranking_sampling="uniform", huber_beta=.1):
    """Supervision-side API; NONE of these targets are model forward inputs."""
    label_spec.validate()
    if mode not in ("absolute_score", "terminal_pdms", "absolute_feedback", "delta_feedback",
                    "trajectory_only", "shuffled_context"):
        raise ValueError("Unknown effect training mode")
    if (before.shape != after.shape or before.ndim != 2 or before.shape[1] != len(label_spec.names)
            or before_valid.shape != before.shape or after_valid.shape != before.shape
            or before_valid.dtype != torch.bool or after_valid.dtype != torch.bool or huber_beta <= 0):
        raise ValueError("Labels/masks do not match bound FeedbackLabelSpec")
    for values, mask in ((before, before_valid), (after, after_valid)):
        if not torch.isfinite(values[mask]).all():
            raise ValueError("Nonfinite valid physical label")
    scale = before.new_tensor(label_spec.scales).float()
    target = torch.where(before_valid & after_valid, after.float() - before.float(), torch.zeros_like(before, dtype=torch.float32))
    valid = before_valid & after_valid
    mean = prediction["mean_delta"].float()
    if mean.shape != before.shape or not torch.isfinite(mean).all():
        raise ValueError("Invalid finite effect prediction")
    choose = torch.ones(len(label_spec.names), dtype=torch.bool, device=before.device)
    full_feedback_supervision = mode not in ("absolute_score", "terminal_pdms")
    if mode.startswith("absolute") or mode == "terminal_pdms":
        if mode == "terminal_pdms":
            choose[:] = False; choose[label_spec.names.index("score")] = True
        elif mode == "absolute_score":
            choose = torch.tensor([u == "0..1" for u in label_spec.units], device=before.device)
        if prediction["absolute_before"] is None or prediction["absolute_after"] is None:
            raise ValueError("Absolute controls require actual absolute heads")
        targets = torch.stack((before, after), 1).float()
        masks = torch.stack((before_valid, after_valid), 1) & choose
        predicted = torch.stack((prediction["absolute_before"], prediction["absolute_after"]), 1).float()
        if not torch.isfinite(predicted).all():
            raise ValueError("Nonfinite absolute prediction")
        value = scene_contribution(F.smooth_l1_loss(predicted / scale, targets.nan_to_num() / scale,
                                                   reduction="none", beta=huber_beta), masks)
    else:
        value = scene_contribution(F.smooth_l1_loss(mean / scale, target / scale,
                                                   reduction="none", beta=huber_beta), valid)
    logvar = prediction["logvar_delta"].float()
    if logvar.shape != mean.shape or not torch.isfinite(logvar).all():
        raise ValueError("Invalid normalized-unit uncertainty head")
    error = (mean - target) / scale
    uncertainty = scene_contribution(.5 * (error.square() * (-logvar).exp() + logvar), valid & choose)
    score = label_spec.names.index("score")
    rank = ranking_contribution(mean[:, score], target[:, score], valid[:, score], sampling=ranking_sampling)
    # Explicit optional tasks retain a graph-connected zero when absent.
    event_logits = prediction["event_logits"].float()
    if event_logits.ndim != 3 or event_logits.shape[1:] != (3, 3) or not torch.isfinite(event_logits).all():
        raise ValueError("Expected finite three-event/three-transition logits")
    if event_targets is None or not full_feedback_supervision:
        event = LossContribution(event_logits.sum() * 0., event_logits.new_tensor(0.))
    else:
        if (event_targets.shape != event_logits.shape[:2] or event_valid is None
                or event_valid.shape != event_targets.shape or event_valid.dtype != torch.bool
                or event_targets.dtype != torch.long or not ((event_targets[event_valid] >= 0)
                    & (event_targets[event_valid] <= 2)).all()):
            raise ValueError("Event transitions must be -1/0/+1 encoded as class 0/1/2")
        safe_target = torch.where(event_valid, event_targets, torch.ones_like(event_targets))
        event = scene_contribution(F.cross_entropy(event_logits.transpose(1, 2), safe_target, reduction="none"), event_valid)
    temporal_pred = prediction["temporal_delta"].float()
    if not torch.isfinite(temporal_pred).all():
        raise ValueError("Nonfinite temporal effect prediction")
    if temporal_targets is None or not full_feedback_supervision:
        temporal = LossContribution(temporal_pred.sum() * 0., temporal_pred.new_tensor(0.))
    else:
        if temporal_targets.shape != temporal_pred.shape or temporal_valid is None or temporal_valid.shape != temporal_pred.shape:
            raise ValueError("Temporal events require actual matching grid and mask")
        if not torch.isfinite(temporal_targets[temporal_valid]).all():
            raise ValueError("Nonfinite valid temporal target")
        temporal = scene_contribution(F.smooth_l1_loss(temporal_pred, temporal_targets.nan_to_num().float(),
                                                      reduction="none", beta=huber_beta), temporal_valid)
    repair_logit = prediction["repair_logit"].float()
    if not torch.isfinite(repair_logit).all():
        raise ValueError("Nonfinite repair prediction")
    if repair_targets is None or not full_feedback_supervision:
        repair = LossContribution(repair_logit.sum() * 0., repair_logit.new_tensor(0.))
    else:
        if repair_targets.shape != repair_logit.shape or repair_valid is None or repair_valid.shape != repair_logit.shape:
            raise ValueError("Repair supervision requires explicit validity")
        valid_repair = repair_targets[repair_valid]
        if not torch.isfinite(valid_repair).all() or not ((valid_repair >= 0) & (valid_repair <= 1)).all():
            raise ValueError("Repair targets require finite valid probabilities")
        repair = scene_contribution(F.binary_cross_entropy_with_logits(repair_logit,
                                    repair_targets.nan_to_num().float(), reduction="none"), repair_valid)
    return {"value": value, "ranking": rank, "events": event, "temporal": temporal,
            "uncertainty": uncertainty, "repair": repair}


def effect_model_loss(prediction, *, weights=None, distributed=False, **targets):
    contributions = effect_loss_contributions(prediction, **targets)
    if weights is None:
        # Fixed startup recipe, not a claim that task scales are calibrated.
        weights = {"value": 1., "ranking": .2, "events": .2, "temporal": .2,
                   "uncertainty": .05, "repair": .2}
    return reduce_contributions(contributions, weights, distributed=distributed)


def refinement_loss(prediction, *, base, target_trajectory, target_coefficients,
                    coefficient_scales, high_quality, teacher_prediction=None,
                    target_effect=None, effect_valid=None, label_spec=None,
                    use_event_guidance=False):
    """A0/A1 share structure/data. A1 alone uses a frozen G1 teacher's graph.

    teacher_prediction must be evaluated on THIS proposed trajectory, not on GT
    or an offline teacher projection. Its parameters are frozen by the framework.
    This is predicted guidance, never a deployment safety guarantee.
    """
    trajectory, coefficients = prediction["trajectory"], prediction["coefficients"]
    if (trajectory.shape != target_trajectory.shape or trajectory.shape != base.shape
            or coefficients.shape != target_coefficients.shape or high_quality.shape != (len(base),)
            or high_quality.dtype != torch.bool):
        raise ValueError("Refiner target/keep contract mismatch")
    if not all(torch.isfinite(x).all() for x in (trajectory, coefficients, target_trajectory, target_coefficients)):
        raise ValueError("Nonfinite refiner prediction or verified supervision")
    scales = coefficients.new_tensor(coefficient_scales)
    if scales.shape != (coefficients.shape[1],) or not (scales > 0).all():
        raise ValueError("Explicit positive physical coefficient scales required")
    yaw_error = trajectory[..., 2] - target_trajectory[..., 2]
    fit = ((trajectory[..., :2] - target_trajectory[..., :2]) / 20.).square().mean((1, 2))
    fit = fit + (2. - 2. * yaw_error.cos()).mean(1)
    fit = fit + ((coefficients - target_coefficients) / scales).square().mean(1)
    magnitude = (coefficients / scales).square().mean(1)
    keep = ((trajectory[..., :2] - base[..., :2]) / 20.).square().mean((1, 2))
    keep = keep + (2. - 2. * (trajectory[..., 2] - base[..., 2]).cos()).mean(1)
    valid = torch.ones(len(base), dtype=torch.bool, device=base.device)
    contributions = {"supervision": scene_contribution(fit, valid),
                     "magnitude": scene_contribution(magnitude, valid),
                     "preserve": scene_contribution(keep, high_quality)}
    weights = {"supervision": 1., "magnitude": .01, "preserve": 1.}
    if teacher_prediction is not None:
        if label_spec is None or target_effect is None or effect_valid is None:
            raise ValueError("A1 requires identity-bound actual verified target effect and masks")
        scales = target_effect.new_tensor(label_spec.scales)
        effect = teacher_prediction["mean_delta"]
        if effect.shape != target_effect.shape or effect_valid.shape != effect.shape:
            raise ValueError("Teacher/verified effect schema mismatch")
        if not torch.isfinite(effect).all() or not torch.isfinite(target_effect[effect_valid]).all():
            raise ValueError("Nonfinite valid predicted/verified effect")
        contributions["effect_consistency"] = scene_contribution(
            F.smooth_l1_loss(effect / scales, target_effect.nan_to_num() / scales, reduction="none", beta=.1), effect_valid)
        protected = [label_spec.names.index(n) for n in ("no_at_fault_collisions", "drivable_area_compliance",
                     "time_to_collision_within_bound") if n in label_spec.names]
        if not protected:
            raise ValueError("A1 protected official metric heads are absent")
        predicted_regression = F.relu(-effect[:, protected]).square()
        contributions["predicted_safety"] = scene_contribution(predicted_regression, effect_valid[:, protected])
        weights.update(effect_consistency=1., predicted_safety=1.)
        if use_event_guidance:
            introduced = teacher_prediction["event_logits"].softmax(-1)[..., 2]
            contributions["predicted_new_event"] = scene_contribution(introduced, valid[:, None].expand_as(introduced))
            weights["predicted_new_event"] = .1
    return reduce_contributions(contributions, weights)
