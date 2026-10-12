"""Dataset gap weighting, never per-scene gap cancellation.

The comparison stream samples q=|gap|/sum|gap|. Its ordinary mean has the
gradient of the globally gap-weighted objective; multiplying by gap again
would incorrectly square the weights. Pointwise supervision stays natural.
"""
import torch
import torch.nn.functional as F
from iqe.losses import LossTerm
from iqe.contracts import require


def comparison_weights(scores, valid, tie_epsilon=.01):
    require(scores.ndim == 2 and scores.shape[1] == 2 and valid.shape == scores.shape, "two-candidate comparison bank required")
    gap = scores[:, 1]-scores[:, 0]
    use = valid.all(-1) & torch.isfinite(scores).all(-1) & (gap.abs() > tie_epsilon)
    return torch.where(use, gap.abs(), torch.zeros_like(gap))


def weighted_population_ranking(predicted, truth, valid, temperature=.1):
    """Reference objective for tests/evaluation, with one dataset-level denominator."""
    w = comparison_weights(truth, valid)
    gap = torch.nan_to_num(truth[:, 1]-truth[:, 0])
    loss = F.softplus(-gap.sign()*(predicted[:, 1]-predicted[:, 0])/temperature)
    return LossTerm((w*loss).sum(), w.sum())


def masks(batch):
    valid = batch["valid"].bool() & torch.isfinite(batch["scores"]) & torch.isfinite(batch["trajectories"]).all((-1, -2))
    natural = valid & batch["natural"].bool()[:, None]
    component = natural[..., None] & batch["component_valid"].bool() & torch.isfinite(batch["components"])
    ranking = (comparison_weights(batch["scores"], valid) > 0) & batch["comparison"].bool()
    return natural, component, ranking


def denominators(batch):
    natural, component, ranking = masks(batch)
    return {"value": natural.sum().float(), "components": component.sum().float(), "ranking": ranking.sum().float()}


def terms(prediction, batch):
    natural, component, ranking = masks(batch)
    p = prediction["values"].float()
    truth = torch.nan_to_num(batch["scores"].detach().float())
    labels = torch.nan_to_num(batch["components"].detach().float())
    require(((truth[natural] >= 0) & (truth[natural] <= 1)).all().item(), "score scale must be [0,1]")
    require(((labels[component] >= 0) & (labels[component] <= 1)).all().item(), "component scale must be [0,1]")
    value = F.smooth_l1_loss(p, truth, beta=.1, reduction="none")
    comp = F.binary_cross_entropy_with_logits(prediction["component_logits"].float(), labels, reduction="none")
    pair = F.softplus(-(truth[:, 1]-truth[:, 0]).sign()*(p[:, 1]-p[:, 0])/.1)
    return {"value": LossTerm((value*natural).sum(), natural.sum().float()),
            "components": LossTerm((comp*component).sum(), component.sum().float()),
            "ranking": LossTerm((pair*ranking).sum(), ranking.sum().float())}
