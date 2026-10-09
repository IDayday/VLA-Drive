"""FP32 masked losses and numerator/denominator-correct DDP reduction."""
from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.distributed as dist
import torch.nn.functional as F
from .contracts import require


@dataclass
class LossTerm:
    numerator: torch.Tensor
    denominator: torch.Tensor


def distributed_term(term: LossTerm):
    denominator = term.denominator.detach().float().clone()
    value = term.numerator.detach().float().clone()
    world = dist.get_world_size() if dist.is_initialized() else 1
    if world > 1:
        dist.all_reduce(denominator)
        dist.all_reduce(value)
    # DDP averages gradients; compensate to recover the true global objective.
    loss = term.numerator * world / denominator.clamp_min(1)
    return loss, {"numerator": value.item(), "denominator": denominator.item(),
                  "mean": (value / denominator.clamp_min(1)).item()}


def reduce_terms(terms, weights=None):
    weights = weights or {k: 1.0 for k in terms}
    losses, logs = [], {}
    for key, term in terms.items():
        loss, log = distributed_term(term)
        losses.append(loss * weights.get(key, 1.0))
        logs[key] = log
    return sum(losses), logs


def expert_il_terms(prediction, target, *, trajectory_weight=1.0, prev_weight=0.0, masks=None, target_long=None):
    """Exact EpisodeDrive singleton trajectory IL recurrence, with inherited weights.

    Source: EpisodeDriveLoss.forward's trajectory block. No scorer/simulation/diversity
    is called; no cross-expert WTA. Unmasked singleton formula is L1 channel sum,
    time mean, batch mean, recursively accumulated across S0 intermediate heads.
    """
    target = target.detach().float()
    require(prediction.raw.shape == target.shape, "S0 target shape mismatch")
    valid = torch.ones(target.shape[:2], device=target.device, dtype=torch.bool) if masks is None else masks.bool()
    require(valid.shape == target.shape[:2], "target timestep mask")
    require(bool(torch.isfinite(target[valid]).all()), "nonfinite valid IL target")
    safe = torch.where(valid[..., None], target, torch.zeros_like(target))
    scene_valid = valid.any(-1)
    scene_loss = target.new_zeros(len(target))
    for output in prediction.intermediates:
        error = (output.float() - safe).abs().sum(-1)
        per_scene = (error * valid).sum(-1) / valid.sum(-1).clamp_min(1)
        if target_long is not None:
            require(target_long.shape == target.shape and bool(torch.isfinite(target_long).all()), "long target invalid")
            per_scene = per_scene + ((output.float() - target_long.detach().float()).abs().sum(-1) * valid).sum(-1) / valid.sum(-1).clamp_min(1)
        scene_loss = prev_weight * scene_loss + per_scene
    return {"il": LossTerm((scene_loss * scene_valid).sum() * trajectory_weight, scene_valid.sum().float())}


def scorer_terms(prediction, scores, score_valid, components, component_valid, *, huber_delta=0.1,
                 temperature=0.1, tie_epsilon=0.01, importance=None):
    require(temperature > 0 and huber_delta > 0, "loss temperatures/delta")
    pred = prediction.values.float()
    truth = scores.detach().float()
    require(pred.shape == truth.shape == score_valid.shape, "score/mask dimensions")
    valid = score_valid.bool() & prediction.valid & torch.isfinite(truth)
    require(bool(torch.isfinite(pred[score_valid]).all()), "nonfinite Scorer output for a valid label")
    require(bool(((truth[valid] >= 0) & (truth[valid] <= 1)).all()), "scorer labels must be [0,1]")
    safe_truth = torch.where(valid, truth, torch.zeros_like(truth))
    w = torch.ones_like(truth) if importance is None else importance.detach().float()[:, None].expand_as(truth)
    require(bool(torch.isfinite(w).all() and (w >= 0).all()), "importance weights")
    wm = w * valid
    value = F.huber_loss(pred, safe_truth, reduction="none", delta=huber_delta)
    terms = {"value": LossTerm((value * wm).sum(), wm.sum())}
    # Include every component head in the graph even on an empty rank.
    numerator = pred.sum() * 0
    denominator = pred.new_zeros(())
    require(not set(components) - set(prediction.component_logits), "labels name nonexistent metric head")
    require(set(components) == set(component_valid), "component labels/masks mismatch")
    for c, logits in prediction.component_logits.items():
        if c not in components:
            numerator = numerator + logits.sum() * 0
            continue
        label = components[c].detach().float()
        mask = valid & component_valid[c].bool() & torch.isfinite(label)
        require(bool(((label[mask] >= 0) & (label[mask] <= 1)).all()), f"component {c} not [0,1]")
        label = torch.where(mask, label, torch.zeros_like(label))
        weight = w * mask
        numerator = numerator + (F.binary_cross_entropy_with_logits(logits.float(), label, reduction="none") * weight).sum()
        denominator = denominator + weight.sum()
    terms["components"] = LossTerm(numerator, denominator)
    k = pred.shape[1]
    pairs = torch.triu_indices(k, k, 1, device=pred.device)
    gap = safe_truth[:, pairs[0]] - safe_truth[:, pairs[1]]
    pair_valid = valid[:, pairs[0]] & valid[:, pairs[1]] & (gap.abs() > tie_epsilon)
    gap_weight = gap.abs() * pair_valid
    rank = F.softplus(-gap.sign() * (pred[:, pairs[0]] - pred[:, pairs[1]]) / temperature)
    scene_denominator = gap_weight.sum(-1)
    scene_rank = (rank * gap_weight).sum(-1) / scene_denominator.clamp_min(torch.finfo(torch.float32).tiny)
    rank_valid = scene_denominator > 0
    terms["ranking"] = LossTerm((scene_rank * rank_valid).sum() + pred.sum() * 0, rank_valid.sum().float())
    return terms


def scorer_denominators(data, tie_epsilon=.01):
    valid = data["valid"].bool() & torch.isfinite(data["scores"]) & torch.isfinite(data["trajectories"]).all((-1, -2))
    component_count = data["scores"].new_zeros(())
    for c, label in data["components"].items():
        component_count += (valid & data["component_valid"][c] & torch.isfinite(label)).sum()
    pairs = torch.triu_indices(valid.shape[1], valid.shape[1], 1, device=valid.device)
    gap = data["scores"][:, pairs[0]] - data["scores"][:, pairs[1]]
    has_pair = (valid[:, pairs[0]] & valid[:, pairs[1]] & (gap.abs() > tie_epsilon)).any(-1)
    return {"value": valid.sum().float(), "components": component_count.float(), "ranking": has_pair.sum().float()}


def router_terms(logits, scores, valid, temperature=0.1):
    from .router import soft_targets
    labels, any_valid = soft_targets(scores, valid, temperature)
    # Invalid candidates do not participate in either target or model normalization.
    masked = logits.float().masked_fill(~valid, -1e9)
    loss = -(labels * F.log_softmax(masked, -1)).sum(-1)
    return {"router": LossTerm((loss * any_valid).sum(), any_valid.sum().float())}


def collective_error(local_error: str | None, device=None):
    """All ranks leave together for cache/label failures before forward/backward."""
    if dist.is_initialized():
        errors = [None] * dist.get_world_size()
        dist.all_gather_object(errors, local_error)
    else:
        errors = [local_error]
    require(not any(errors), f"distributed data failure: {errors}")
