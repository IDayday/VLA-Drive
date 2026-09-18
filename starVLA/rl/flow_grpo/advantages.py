"""Behavior-batch reward scaling; never recomputed between inner updates.

Reference: yifan123/flow_grpo, 879042cf, PerPromptStatTracker(global_std=True).
The baseline is still each scene's mean. Only the denominator may use all
candidate rewards in the global, accumulated behavior batch, including constant
groups. This is neither rank-local whitening nor a running historical baseline.
"""
import torch
import torch.distributed as dist

from .distributed import synchronized_call
from .math import centered_group_rewards, group_advantages, probability_tensor


MODES = {"group", "global_batch", "none"}


@torch.no_grad()
def behavior_advantages(rewards, *, normalization="group", epsilon=1e-6, clip=5.0):
    device = rewards.device

    def validate():
        if normalization not in MODES:
            raise ValueError("unknown advantage normalization: " + normalization)
        if rewards.ndim != 2 or not rewards.numel():
            raise ValueError("nonempty [scene,candidate] rewards required")
        # Also checks finite rewards, epsilon and clip, before any moment collective.
        return group_advantages(rewards, epsilon=epsilon, clip=clip)

    grouped = synchronized_call(validate, device)
    delta, std, _ = centered_group_rewards(rewards)
    stats = {"normalization": normalization, "moment_dtype": "torch.float64",
             "epsilon": epsilon, "clip": clip}
    if normalization == "group":
        return grouped, stats
    if normalization == "none":
        return delta.clamp(-clip, clip).to(probability_tensor(rewards).dtype), stats
    values = rewards.double()
    # Two-pass variance avoids E[x^2]-E[x]^2 cancellation for almost equal rewards.
    totals = torch.stack((values.sum(), values.new_tensor(values.numel())))
    distributed = dist.is_available() and dist.is_initialized()
    if distributed:
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    mean = totals[0] / totals[1]
    squared = (values - mean).square().sum()
    if distributed:
        dist.all_reduce(squared, op=dist.ReduceOp.SUM)
    global_std = (squared / totals[1]).sqrt()
    # A constant scene contributes exactly zero, regardless of other scenes.
    adv = torch.where(std > 0, delta / (global_std + epsilon), 0)
    stats.update(global_reward_std=float(global_std),
                 global_candidate_count=int(totals[1]), global_reward_mean=float(mean))
    return adv.clamp(-clip, clip).to(probability_tensor(rewards).dtype), stats


def assign_behavior_advantages(buffers, algorithm, device):
    """Called once after all local microbatches, only for freshly sampled buffers.

    Resume restores the saved advantages/statistics and bypasses this function.
    Refuse overwriting them to make accidental inner-epoch renormalization fail.
    """
    def collect():
        if not buffers or any(r.advantages is not None for r in buffers):
            raise ValueError("fresh unassigned behavior buffers required")
        return torch.cat([r.rewards for r in buffers], dim=0)

    rewards = synchronized_call(collect, device)
    advantages, statistics = behavior_advantages(
        rewards, normalization=algorithm.get("advantage_normalization", "group"),
        epsilon=algorithm["advantage_epsilon"], clip=algorithm["advantage_clip"])
    offset = 0
    for rollout in buffers:
        count = len(rollout.rewards)
        rollout.advantages = advantages[offset:offset+count].clone()
        rollout.advantage_statistics = dict(statistics)
        offset += count
