"""FP32 raw feature losses with complete optimizer-batch normalization.

Counts are fixed before splitting into microbatches. DDP averages gradients,
so each rank contributes ``world_size * local_numerator / global_count``.
Backpropagate microbatch contributions by summation, without another division
by the accumulation length. No teacher feature or temporal delta is unit-normed.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import torch
import torch.distributed as dist


@dataclass(frozen=True)
class LossTerm:
    loss: torch.Tensor
    numerator: torch.Tensor  # detached local FP32 sum, for explicit aggregation
    local_count: torch.Tensor
    global_count: torch.Tensor


def _world_size(requested):
    actual = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
    if requested is None:
        return actual
    if type(requested) is not int or requested < 1 or (actual > 1 and requested != actual):
        raise ValueError('Loss world size differs from the active distributed group')
    return requested


def masked_mse(prediction, target, valid, global_count, *, world_size=None):
    """One local contribution to globally normalized, channel-wise raw MSE."""
    if (prediction.shape != target.shape or prediction.ndim < 2 or
            not prediction.is_floating_point() or not target.is_floating_point() or
            valid.shape != prediction.shape[:-1] or valid.dtype != torch.bool or
            prediction.shape[-1] < 1):
        raise ValueError('Prediction/target need matching floating shapes and a Boolean non-channel mask')
    if prediction.device != target.device or prediction.device != valid.device:
        raise ValueError('Prediction, labels and validity must share a device')
    if not torch.isfinite(prediction[valid]).all() or not torch.isfinite(target[valid]).all():
        raise ValueError('Nonfinite prediction/label at a valid supervised position')
    count = torch.as_tensor(global_count, device=prediction.device, dtype=torch.float64).detach()
    local_count = valid.sum(dtype=torch.float64) * prediction.shape[-1]
    if (count.ndim != 0 or not torch.isfinite(count) or count < local_count or
            count < 0 or count != count.round()):
        raise ValueError('Global denominator must be an integral count covering this microbatch')
    # Sanitize padding before arithmetic. NaN padding must neither poison a
    # differentiable zero nor propagate a NaN gradient through an invalid slot.
    p = torch.where(valid[..., None], prediction.float(), 0.)
    t = torch.where(valid[..., None], target.detach().float(), 0.)
    numerator = (p - t).square().sum(dtype=torch.float32)
    loss = numerator * _world_size(world_size) / count.clamp_min(1).float()
    return LossTerm(loss, numerator.detach(), local_count.detach(), count)


def _local_counts(channels, current_valid, future_valid, delta):
    if type(channels) is not int or channels < 1 or type(delta) is not bool:
        raise ValueError('Explicit positive teacher channels and Boolean delta required')
    if current_valid is None and future_valid is None:
        raise ValueError('No auxiliary objective was requested')
    masks = {}
    if current_valid is not None:
        if current_valid.ndim != 3 or current_valid.dtype != torch.bool:
            raise ValueError('Sampled current validity must be B,V,K Boolean')
        masks['current'] = current_valid
    if future_valid is not None:
        if future_valid.ndim != 4 or future_valid.shape[1] != 2 or future_valid.dtype != torch.bool:
            raise ValueError('Sampled future validity must be B,2,V,K Boolean')
        if current_valid is not None and current_valid.shape != future_valid[:, 0].shape:
            raise ValueError('Current/future requests must use the same original scene/view/patch batch')
        masks['future_a'], masks['future_b'] = future_valid[:, 0], future_valid[:, 1]
        if delta:
            masks['delta'] = future_valid[:, 0] & future_valid[:, 1]
    elif delta:
        raise ValueError('Delta requires both future endpoints')
    if any(min(m.shape) < 1 for m in masks.values()):
        raise ValueError('A rank must retain nonempty requests even when all labels are invalid')
    if len({m.device for m in masks.values()}) != 1:
        raise ValueError('All label masks must share a device')
    return masks


def optimizer_batch_counts(channels, *, current_valid=None, future_valid=None,
                           delta=False, distributed=True):
    """All-reduce counts once for the whole original optimizer batch.

Validation is collective as well, so malformed masks or a different active
objective on one rank fail every rank before any rank enters the SUM collective.
The caller must broadcast/coordinate failures in later model/data operations.
"""
    if type(distributed) is not bool:
        raise ValueError('distributed must be Boolean')
    active = distributed and dist.is_available() and dist.is_initialized()
    error, masks = None, {}
    try:
        masks = _local_counts(channels, current_valid, future_valid, delta)
        if active and dist.get_backend() == 'nccl' and any(m.device.type != 'cuda' for m in masks.values()):
            raise ValueError('NCCL count reduction requires masks on the current CUDA device')
    except (ValueError, TypeError, AttributeError) as exc:
        error = str(exc)
    inventory = {'error': error, 'names': sorted(masks), 'channels': channels}
    if active:
        inventories = [None] * dist.get_world_size()
        dist.all_gather_object(inventories, inventory)
        if any(item['error'] for item in inventories):
            raise ValueError('Distributed count validation failed: ' + str(inventories))
        if any(item != inventories[0] for item in inventories):
            raise ValueError('Distributed objective/count inventory differs across ranks')
    elif error is not None:
        raise ValueError(error)
    names = sorted(masks)
    counts = torch.stack([masks[name].sum(dtype=torch.float64) * channels for name in names])
    if active:
        dist.all_reduce(counts, op=dist.ReduceOp.SUM)
    return {name: counts[i].detach() for i, name in enumerate(names)}


def temporal_losses(global_counts, *, current_prediction=None, current_target=None,
                    current_valid=None, future_prediction=None, future_target=None,
                    future_valid=None, delta=False, world_size=None):
    """Return unweighted ``current/future/delta`` losses and per-term receipts.

Current tensors: B,V,K,C. Future tensors: B,2,V,K,C, from the same
pre-sampled requests. Future = .5*(MSE_a + MSE_b), with separate counts;
delta uses only positions valid at both endpoints and preserves raw magnitude.
"""
    if not isinstance(global_counts, Mapping) or type(delta) is not bool:
        raise ValueError('Explicit global count mapping and Boolean delta required')
    current = (current_prediction, current_target, current_valid)
    future = (future_prediction, future_target, future_valid)
    for name, values in (('current', current), ('future', future)):
        if any(v is not None for v in values) and any(v is None for v in values):
            raise ValueError(name + ' prediction, target and mask must be supplied together')
    has_current, has_future = current_prediction is not None, future_prediction is not None
    if not (has_current or has_future) or (delta and not has_future):
        raise ValueError('Requested losses have no matching predictions')
    expected = ({'current'} if has_current else set()) | ({'future_a', 'future_b'} if has_future else set())
    if delta:
        expected.add('delta')
    if set(global_counts) != expected:
        raise ValueError('Global count inventory does not match requested losses')
    terms, losses = {}, {}
    if has_current:
        if current_prediction.ndim != 4:
            raise ValueError('Current prediction must be B,V,K,C')
        terms['current'] = masked_mse(*current, global_counts['current'], world_size=world_size)
        losses['current'] = terms['current'].loss
    if has_future:
        if (future_prediction.ndim != 5 or future_prediction.shape[1] != 2 or
                future_target.shape != future_prediction.shape or
                future_valid.shape != future_prediction.shape[:-1]):
            raise ValueError('Future prediction/target must be B,2,V,K,C with matching mask')
        if has_current and current_prediction.shape != future_prediction[:, 0].shape:
            raise ValueError('Auxiliary objectives must share the original scene/view/patch batch')
        for endpoint, name in enumerate(('future_a', 'future_b')):
            terms[name] = masked_mse(future_prediction[:, endpoint], future_target[:, endpoint],
                                     future_valid[:, endpoint], global_counts[name], world_size=world_size)
        losses['future'] = .5 * (terms['future_a'].loss + terms['future_b'].loss)
        if delta:
            valid = future_valid[:, 0] & future_valid[:, 1]
            p = torch.where(valid[:, None, ..., None], future_prediction.float(), 0.)
            t = torch.where(valid[:, None, ..., None], future_target.detach().float(), 0.)
            terms['delta'] = masked_mse(p[:, 1] - p[:, 0], t[:, 1] - t[:, 0], valid,
                                        global_counts['delta'], world_size=world_size)
            losses['delta'] = terms['delta'].loss
    return losses, terms


def weighted_temporal_loss(losses, config):
    """Apply registered current/future/delta weights exactly once."""
    from .config import SpaceTimeConfig
    if not isinstance(config, SpaceTimeConfig):
        raise ValueError('A resolved SpaceTimeConfig is required')
    config.validate()
    expected = {name for name in ('current', 'future', 'delta') if getattr(config, name)}
    if not isinstance(losses, Mapping) or set(losses) != expected:
        raise ValueError('Returned loss terms differ from the registered active objectives')
    return sum(losses[name] * getattr(config, name + '_weight') for name in sorted(expected))
