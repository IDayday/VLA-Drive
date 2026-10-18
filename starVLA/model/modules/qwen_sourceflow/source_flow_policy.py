"""Native source sigma/noise-data velocity convention, in three action dimensions."""
import torch
from torch import nn
from .cached_action_runner import CachedActionRunner,condition_fp32
from .vendor.scheduler_continuous import WanContinuousFlowMatchScheduler
from .action_codec import ActionCodec


class SourceFlowPolicy(nn.Module):
    def __init__(self, action_expert):
        super().__init__()
        self.runner = CachedActionRunner(action_expert)
        self.scheduler = WanContinuousFlowMatchScheduler(1000, shift=5.)

    @property
    def action_expert(self):
        return self.runner.action_expert

    def velocity(self, x, timestep, condition):
        return self.runner(x, timestep, condition)

    def forward(self, condition, normalized_action, *, noise=None, timestep=None, valid=None):
        if normalized_action.ndim != 3 or normalized_action.shape[1:] != (8, 3):
            raise ValueError('Native absolute source 8x3 actions required')
        target = normalized_action.float()
        noise = torch.randn_like(target) if noise is None else noise
        timestep = (self.scheduler.sample_training_t(len(target), target.device, torch.float32)
                    if timestep is None else timestep)
        x = self.scheduler.add_noise(target, noise, timestep)
        velocity = self.scheduler.training_target(target, noise, timestep)
        pred = self.velocity(x, timestep, condition)
        error = (pred.float() - velocity.float()).square().mean(-1)
        if valid is not None:
            if not valid.any(-1).all():
                raise ValueError('No valid GT future for a scene')
            mse = (error * valid).sum(-1) / valid.sum(-1)
        else:
            mse = error.mean(-1)
        weight = self.scheduler.training_weight(timestep)
        return (mse * weight).mean(), dict(mse=mse.detach(), weight=weight.detach(), timestep=timestep.detach())

    @torch.no_grad()
    def sample_ode(self, condition, initial_noise):
        if initial_noise.shape[1:] != (8, 3):
            raise ValueError('Native action prior must be 8x3')
        condition=condition_fp32(condition)
        times, deltas = self.scheduler.build_inference_schedule(10, initial_noise.device, torch.float32)
        x = initial_noise.float()
        for t, d in zip(times, deltas):
            v = self.velocity(x, t.expand(len(x)), condition).float()
            x = self.scheduler.step(v, d, x)
        if not torch.isfinite(x).all():
            raise FloatingPointError('Nonfinite source ODE trajectory')
        return x
