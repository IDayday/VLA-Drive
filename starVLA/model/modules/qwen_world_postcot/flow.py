"""Clean-x supervision and mathematically matched positive-sigma transitions."""
import torch
from torch import nn


def shift_sigma(uniform, shift=5.0):
    if shift <= 0:
        raise ValueError('Flow time shift must be positive')
    value = uniform.float()
    return shift * value / (1 + (shift - 1) * value)


def _draw(shape, device, generator=None, *, normal=False):
    draw_device = device if generator is None else generator.device
    operation = torch.randn if normal else torch.rand
    return operation(shape, device=draw_device, dtype=torch.float32, generator=generator).to(device)


def sample_sigma(batch_size, device, *, generator=None, shift=5.0):
    return shift_sigma(_draw((batch_size,), device, generator), shift)


def _broadcast_sigma(sigma, state):
    sigma = torch.as_tensor(sigma, device=state.device, dtype=torch.float32)
    if sigma.ndim == 0:
        sigma = sigma.expand(len(state))
    if sigma.shape != (len(state),):
        raise ValueError('Sigma must be scalar or [B]')
    return sigma[:, None, None]


def clean_x_to_velocity(state, predicted_x0, sigma):
    value = _broadcast_sigma(sigma, state)
    if bool((value <= 0).any()):
        raise ValueError('Velocity conversion is legal only at strictly positive sigma')
    return (state.float() - predicted_x0.float()) / value


def clean_x_step(state, predicted_x0, sigma, sigma_next, *, validate=True):
    value = _broadcast_sigma(sigma, state)
    next_value = _broadcast_sigma(sigma_next, state)
    if validate and (bool((value <= 0).any()) or bool((next_value < 0).any()) or bool((next_value > value).any())):
        raise ValueError('Clean-x ODE needs positive current sigma and decreasing nonnegative next sigma')
    ratio = next_value / value
    # FP32 states and update arithmetic, independently of actor autocast.
    return ratio * state.float() + (1 - ratio) * predicted_x0.float()


def clean_x_l1(prediction, target, valid=None, *, return_statistics=False):
    if prediction.shape != target.shape or prediction.ndim != 3 or prediction.shape[1:] != (8, 3):
        raise ValueError('Clean-x L1 expects aligned native [B,8,3] trajectories')
    if valid is None:
        valid = torch.ones_like(target, dtype=torch.bool)
    elif valid.shape == target.shape[:2]:
        valid = valid[:, :, None].expand_as(target)
    elif valid.shape != target.shape:
        raise ValueError('Action validity must be [B,8] or [B,8,3]')
    valid = valid.detach().bool()
    difference = torch.where(valid, prediction.float() - target.float(), torch.zeros_like(prediction, dtype=torch.float32))
    error = difference.abs()
    numerator = error.sum()
    denominator = valid.sum(dtype=torch.float32)
    loss = numerator / denominator.clamp_min(1)
    if return_statistics:
        return loss, dict(numerator=numerator, denominator=denominator,
            per_scene_numerator=error.sum((1, 2)),
            per_scene_denominator=valid.sum((1, 2), dtype=torch.float32))
    return loss


class CleanXFlowPolicy(nn.Module):
    def __init__(self, actor, *, shift=5.0, inference_steps=10):
        super().__init__()
        if shift <= 0 or inference_steps <= 0:
            raise ValueError('Positive shift and inference step budget required')
        self.actor, self.shift, self.inference_steps = actor, float(shift), int(inference_steps)

    @property
    def prediction_type(self):
        return 'x0'

    def loss(self, prepared_context, normalized_action, *, valid=None, noise=None, sigma=None, generator=None):
        if normalized_action.ndim != 3 or normalized_action.shape[1:] != (8, 3):
            raise ValueError('Native normalized 8x3 trajectory required')
        target = normalized_action.detach().float()
        if valid is not None:
            coordinate_valid = valid.detach().bool()
            if coordinate_valid.shape == target.shape[:2]:
                coordinate_valid = coordinate_valid[:, :, None].expand_as(target)
            if coordinate_valid.shape != target.shape:
                raise ValueError('Action validity must be [B,8] or [B,8,3]')
            target = torch.where(coordinate_valid, target, torch.zeros_like(target))
        noise = _draw(target.shape, target.device, generator, normal=True) if noise is None else noise.detach().float()
        sigma = sample_sigma(len(target), target.device, generator=generator, shift=self.shift) if sigma is None else sigma.detach().float()
        if noise.shape != target.shape or sigma.shape != (len(target),):
            raise ValueError('Fixed noise/sigma are misaligned with this training microbatch')
        noisy = (1 - sigma[:, None, None]) * target + sigma[:, None, None] * noise
        prediction = self.actor(noisy, sigma, prepared_context)
        objective = getattr(self, 'coordinate_objective', None)
        if objective is None:
            loss, statistics = clean_x_l1(prediction, target, valid, return_statistics=True)
        else:
            loss, statistics = objective(prediction, target, valid)
        return loss, dict(statistics, sigma=sigma.detach(), prediction_x0=prediction, noisy_action=noisy)

    def forward(self, prepared_context, normalized_action, **randomness):
        return self.loss(prepared_context, normalized_action, **randomness)

    @torch.no_grad()
    def sample_ode(self, prepared_context, initial_noise, *, inference_steps=None):
        if initial_noise.ndim != 3 or initial_noise.shape[1:] != (8, 3):
            raise ValueError('One native [B,8,3] noise sample per deployment scene')
        steps = self.inference_steps if inference_steps is None else int(inference_steps)
        if steps <= 0:
            raise ValueError('Inference step count must be positive')
        # Small CPU schedule once: avoid a GPU scalar synchronization per step.
        schedule = shift_sigma(torch.linspace(1, 0, steps + 1, dtype=torch.float32), self.shift).tolist()
        state = initial_noise.float()
        for current, next_value in zip(schedule[:-1], schedule[1:]):
            sigma = torch.full((len(state),), current, device=state.device, dtype=torch.float32)
            prediction = self.actor(state, sigma, prepared_context)
            if next_value == 0:
                state = prediction.float()
            else:
                state = clean_x_step(state, prediction, sigma, next_value, validate=False)
        if not bool(torch.isfinite(state).all()):
            raise FloatingPointError('Nonfinite clean-x deployment trajectory')
        return state
