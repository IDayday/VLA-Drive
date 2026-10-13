"""Positive, scene-normalized FM extraction using independent interpolation draws."""

import torch


def sample_targets(actions, probabilities, *, generator):
    """Draw one candidate per scene, before the original FM repeat dimension."""
    if actions.ndim != 4 or probabilities.shape != actions.shape[:2]:
        raise ValueError("actions [scene,candidate,horizon,dim] and scene weights required")
    weights = probabilities.detach()
    if not torch.isfinite(actions).all() or not torch.isfinite(weights).all():
        raise ValueError("nonfinite candidate data")
    if (weights < 0).any() or not torch.allclose(weights.sum(1), torch.ones_like(weights[:, 0]), atol=1e-7, rtol=0):
        raise ValueError("negative or unnormalized FM candidate weights")
    indices = torch.multinomial(weights, 1, generator=generator).squeeze(1)
    selected = actions[torch.arange(len(actions), device=actions.device), indices].detach()
    return selected, indices


def interpolation(targets, noise, time, *, noise_s=0.999):
    if targets.shape != noise.shape or time.shape != (len(targets),):
        raise ValueError("FM target/noise/time shapes differ")
    # Preserve the original head's (noise_s - Beta(1.5,1))/noise_s law,
    # including its small negative-time tail; do not silently clamp/resample.
    if not 0 < noise_s <= 1:
        raise ValueError('invalid original FM noise_s')
    minimum = (noise_s - 1) / noise_s
    if not torch.isfinite(targets).all() or not torch.isfinite(noise).all() or not torch.isfinite(time).all() or ((time < minimum - 1e-7) | (time > 1)).any():
        raise ValueError("invalid independent FM draws")
    t = time.reshape(-1, *([1] * (targets.ndim - 1)))
    targets, noise = targets.detach(), noise.detach()
    return (1 - t) * noise + t * targets, targets - noise


def positive_flow_loss(prediction, target_velocity, *, scenes, repeats):
    if prediction.shape != target_velocity.shape or len(prediction) != scenes * repeats:
        raise ValueError("FM repeats must not change scene weighting")
    per_draw = (prediction.float() - target_velocity.detach().float()).square().flatten(1).mean(1)
    per_scene = per_draw.reshape(repeats, scenes).mean(0)
    return per_scene.mean(), per_scene


def reference_velocity_loss(current, reference, *, scenes, repeats):
    """Velocity retention, not a KL of the deployed ODE distribution."""
    return positive_flow_loss(current, reference.detach(), scenes=scenes, repeats=repeats)
