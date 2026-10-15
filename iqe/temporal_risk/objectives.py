"""An auxiliary cumulative-event BCE; never compose marginal probabilities."""
import torch
import torch.nn.functional as F
from iqe.contracts import require
from iqe.losses import LossTerm
from iqe.specialist_study.objectives import terms as base_terms, denominators as base_denominators


def event_mask(batch):
    return (batch['event_valid'].bool() & batch['valid'][..., None, None].bool()
            & batch['natural'][:, None, None, None].bool())


def denominators(batch):
    return base_denominators(batch) | {'events': event_mask(batch).flatten(1).any(-1).sum().float()}


def terms(prediction, batch):
    result = base_terms(prediction, batch)
    valid = event_mask(batch)
    logits = prediction['event_logits'].float()
    require(logits.shape == batch['events'].shape == valid.shape, 'temporal labels shape')
    require(torch.isfinite(logits[valid]).all().item(), 'nonfinite valid risk prediction')
    targets = batch['events'].detach().float()
    require(torch.isfinite(targets[valid]).all().item() and
            ((targets[valid] == 0) | (targets[valid] == 1)).all().item(), 'invalid event targets')
    loss = F.binary_cross_entropy_with_logits(torch.where(valid, logits, 0.),
        torch.where(valid, targets, 0.), reduction='none')
    # Natural scene mass stays uniform; more valid candidates/times do not
    # silently weight that scene more. No event labels on the comparison stream.
    values = (loss*valid).flatten(1).sum(-1)/valid.flatten(1).sum(-1).clamp_min(1)
    result['events'] = LossTerm(values.sum(), valid.flatten(1).any(-1).sum().float())
    return result
