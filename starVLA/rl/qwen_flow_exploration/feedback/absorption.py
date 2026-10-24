"""Positive scene-normalized candidate FM using the ORIGINAL native scheduler."""
import torch

from starVLA.model.modules.qwen_sourceflow.action_codec import ActionCodec
from starVLA.model.modules.qwen_sourceflow.vendor.scheduler_continuous import WanContinuousFlowMatchScheduler


def teacher_probabilities(scores, eligible, *, temperature=.1):
    if scores.ndim != 2 or scores.shape != eligible.shape or not torch.isfinite(scores).all() or temperature != .1:
        raise ValueError('Locked per-scene finite official score tensor required')
    if eligible.dtype != torch.bool or not eligible.any(1).all():
        raise ValueError('No synthetic success labels for scenes with no eligible target')
    logits = (scores.detach()/temperature).masked_fill(~eligible, -torch.inf)
    return torch.softmax(logits, dim=1).detach()


def draw_teacher(requests, probabilities, *, generator):
    if requests.ndim != 4 or requests.shape[2:] != (8, 3) or probabilities.shape != requests.shape[:2]:
        raise ValueError('Native scene/candidate/8/3 teacher contract')
    if not torch.isfinite(requests).all() or not torch.isfinite(probabilities).all() or (probabilities < 0).any():
        raise ValueError('No negative/nonfinite MSE weights')
    torch.testing.assert_close(probabilities.sum(1), torch.ones_like(probabilities[:, 0]), atol=1e-6, rtol=0)
    indices = torch.multinomial(probabilities.detach(), 1, generator=generator).squeeze(1)
    selected = requests[torch.arange(len(requests), device=requests.device), indices].detach()
    return ActionCodec.norm_odo(selected), indices


def native_source_fm_loss(velocity, condition, normalized_target, *, noise, timestep, valid=None):
    """Direct source sigma convention; noise/time must be independent of search.

    `velocity(x, t, condition)` may be the current native action runner/actor.
    This is supervised FM, neither NLL nor an old-policy likelihood ratio.
    """
    if normalized_target.ndim != 3 or normalized_target.shape[1:] != (8, 3) or noise.shape != normalized_target.shape or timestep.shape != (len(noise),):
        raise ValueError('Independent source FM draws must match native action shape')
    if not torch.isfinite(normalized_target).all() or not torch.isfinite(noise).all() or not torch.isfinite(timestep).all() or (timestep < 0).any() or (timestep > 1000).any():
        raise ValueError('Invalid source FM samples')
    scheduler = WanContinuousFlowMatchScheduler(1000, shift=5.)
    target = normalized_target.detach().float()
    noise = noise.detach().float()
    timestep = timestep.detach()
    x = scheduler.add_noise(target, noise, timestep)
    desired = scheduler.training_target(target, noise, timestep)
    prediction = velocity(x, timestep, condition)
    error = (prediction.float()-desired).square().mean(-1)
    if valid is not None:
        if valid.shape != error.shape or not valid.any(1).all():
            raise ValueError('Invalid time validity mask')
        mse = (error*valid).sum(1)/valid.sum(1)
    else:
        mse = error.mean(1)
    per_scene = mse*scheduler.training_weight(timestep)
    return per_scene.mean(), dict(per_scene=per_scene.detach(), mse=mse.detach(), timestep=timestep,
                                  label='positive_supervised_source_FM_not_policy_likelihood')
