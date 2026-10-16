"""Source FM supervision on the original DDP or standalone source ActionDiT.

Both actors consume only original normalized Qwen action-query states. GT is
used solely by the supervised objective, never by velocity or deployment.
"""
import torch
from torch import nn
from starVLA.rl.simwam_recipe.vendor.scheduler_continuous import WanContinuousFlowMatchScheduler
from .vendor.action_dit import ActionDiT


SOURCE_ACTION_CONFIG = dict(hidden_dim=1024, action_dim=4, ffn_dim=4096,
    text_dim=2048, freq_dim=256, eps=1e-6, num_heads=24,
    attn_head_dim=128, num_layers=30, use_gradient_checkpointing=False)


class TransferPolicy(nn.Module):
    def __init__(self, head, architecture):
        super().__init__()
        if architecture not in ('ddp', 'source_action_dit'):
            raise ValueError(architecture)
        self.head = head
        self.architecture = architecture
        self.scheduler = WanContinuousFlowMatchScheduler(1000, shift=5.)

    def project(self, raw):
        if raw.ndim != 3 or raw.shape[1:] != (8, 2048):
            raise ValueError('Expected original pre-qwen_proj condition [B,8,2048]')
        return self.head._project_condition(raw) if self.architecture == 'ddp' else raw

    def velocity_sigma(self, action, sigma, condition):
        """Source convention: dx/dsigma = epsilon - data, sigma decreasing."""
        if action.shape != (condition.shape[0], 8, 4) or sigma.shape != (action.shape[0],):
            raise ValueError('Scene/action/time shape mismatch')
        if self.architecture == 'ddp':
            buckets = ((1 - sigma.float()) * self.head.num_timestep_buckets).long()
            return -self.head.predict_velocity(action, buckets, condition)
        return self.head(action, sigma.float()*1000., condition)

    def forward(self, raw, target, *, epsilon=None, sigma=None, repeat=8):
        if target.shape != (raw.shape[0], 8, 4) or repeat < 1:
            raise ValueError('Expected one original action label per scene')
        # Preserve scene-normalized supervision; repeats are independent FM
        # noise/time draws, not additional independent scene exposures.
        target = target.repeat_interleave(repeat, 0)
        condition = self.project(raw).repeat_interleave(repeat, 0)
        if epsilon is None:
            epsilon = torch.randn_like(target)
        if sigma is None:
            sigma = self.scheduler.sample_training_t(target.shape[0], target.device, torch.float32)/1000.
        if epsilon.shape != target.shape or sigma.shape != (target.shape[0],):
            raise ValueError('Explicit FM randomness shape mismatch')
        noisy = self.scheduler.add_noise(target, epsilon, sigma*1000.)
        velocity = self.velocity_sigma(noisy, sigma, condition)
        desired = self.scheduler.training_target(target, epsilon, sigma*1000.)
        mse = (velocity.float()-desired.float()).square().mean(dim=(1,2))
        weight = self.scheduler.training_weight(sigma*1000.).reshape(-1)
        loss = (mse*weight).reshape(raw.shape[0],repeat).mean()
        return loss, dict(unweighted_mse=mse.detach().mean(), time_weight=weight.detach().mean())

    @torch.no_grad()
    def sample_ode(self, raw, root, *, grid='native', steps=10):
        if self.training or steps != 10:
            raise ValueError('Deployment requires eval and the registered ten steps')
        if grid not in ('native','shift5'):
            raise ValueError(grid)
        condition = self.project(raw.float())
        action = root.float().clone()
        if grid == 'native' and self.architecture == 'ddp':
            # Keep the original DDP arithmetic/buckets exactly, including dt.
            return self.head._euler_sample(action, condition)
        times, deltas = self.scheduler.build_inference_schedule(steps, root.device,
            torch.float32, shift_override=1. if grid=='native' else 5.)
        for time, delta in zip(times, deltas):
            sigma = (time/1000.).expand(root.shape[0])
            action = self.scheduler.step(self.velocity_sigma(action,sigma,condition), delta, action)
        if not torch.isfinite(action).all():
            raise ValueError('Nonfinite deployed action')
        return action


def make_policy(architecture, *, device='cpu', verify_sft=True, tiny=None):
    if architecture == 'ddp':
        if tiny is not None:
            raise ValueError('DDP uses its actual immutable SFT topology')
        from starVLA.rl.constrained_flow_noise.loading import load_action_closure
        closure, _, _ = load_action_closure(dtype=torch.float32, device=device, verify_hash=verify_sft)
        head = closure.action_model
    elif architecture == 'source_action_dit':
        # No unavailable Wan-derived initialization is silently claimed. This
        # head is supervised from its source initialization with seed42.
        head = ActionDiT(**(tiny or SOURCE_ACTION_CONFIG)).to(device=device,dtype=torch.float32)
    else:
        raise ValueError(architecture)
    return TransferPolicy(head, architecture)
